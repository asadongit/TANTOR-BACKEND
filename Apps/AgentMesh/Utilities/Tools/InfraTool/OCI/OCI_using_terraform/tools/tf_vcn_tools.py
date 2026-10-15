import os
import json
import logging
import asyncio
import tempfile
import shutil
import subprocess
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from sqlalchemy import select

from Database.Database import AsyncSessionLocal
from Database.Models import ToolStateModel
from Apps.AgentMesh.Utilities.State.ToolStateManager import ToolStateManager, process_state_data
from Apps.AgentMesh.Schema.ToolSchema import ToolMeta
from Apps.AgentMesh.Core.ToolRegistry import registry
from Apps.AgentMesh.Utilities.Tools.InfraTool.OCI.OCI_using_terraform.tools.tf_auth_tools import (
    run_terraform_in_sandbox,
    extract_data_from_state,
    resolve_oci_creds,
    run_tf_query,
    resolve_compartment_id,
    normalize_name
)

logger = logging.getLogger(__name__)

# Absolute path to the root Terraform OCI directory (parent of tools/)
TF_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))



@tool
async def tf_provision_vcn(
    vcn_name: str,
    cidr_block: str,
    dns_label: Optional[str] = None,
    compartment_id: Optional[str] = None,
    config: RunnableConfig = None
) -> str:
    """
    Provisions a new VCN in the configured OCI compartment using Terraform.
    CRITICAL: Before running this tool, you must ask the user which OCI compartment they want to build the VCN in, and ensure it is configured.
    """
    try:
        creds = await resolve_oci_creds(config)
        # thread_id still needed for state save/load of VCN terraform state
        _cfg = config.get("configurable", {}) if isinstance(config, dict) else (getattr(config, "configurable", {}) if config else {})
        thread_id = _cfg.get("thread_id") or "default_thread"

        config_data = {
            "user": creds.get("user_id"),
            "key_content": creds.get("private_key"),
            "fingerprint": creds.get("fingerprint"),
            "tenancy": creds.get("tenancy_id"),
            "region": creds.get("region"),
            "compartment_id": creds.get("compartment_id")
        }
        target_compartment_input = compartment_id or config_data.get("compartment_id") or config_data.get("tenancy") or ""

        env = os.environ.copy()
        key_content = config_data.get("key_content", "")
        
        if not key_content:
            return "Error: Private key is empty or missing. Please configure credentials first using tf_configure_auth."
            
        # Ensure key formatting is correct
        key_content = key_content.strip()
        if "-----BEGIN RSA PRIVATE KEY-----" in key_content:
            raw_b64 = key_content.replace("-----BEGIN RSA PRIVATE KEY-----", "").replace("-----END RSA PRIVATE KEY-----", "")
            raw_b64 = "".join(raw_b64.split())
            chunks = [raw_b64[i:i+64] for i in range(0, len(raw_b64), 64)]
            key_content = "-----BEGIN RSA PRIVATE KEY-----\n" + "\n".join(chunks) + "\n-----END RSA PRIVATE KEY-----\n"
        elif "-----BEGIN PRIVATE KEY-----" in key_content:
            raw_b64 = key_content.replace("-----BEGIN PRIVATE KEY-----", "").replace("-----END PRIVATE KEY-----", "")
            raw_b64 = "".join(raw_b64.split())
            chunks = [raw_b64[i:i+64] for i in range(0, len(raw_b64), 64)]
            key_content = "-----BEGIN PRIVATE KEY-----\n" + "\n".join(chunks) + "\n-----END PRIVATE KEY-----\n"

        temp_pem = tempfile.NamedTemporaryFile(mode="w", suffix=".pem", delete=False, newline="\n")
        try:
            temp_pem.write(key_content)
            temp_pem.close()
            
            # Resolve compartment name to OCID
            resolved_comp_id = await resolve_compartment_id(
                target_compartment_input,
                TF_DIR, env, {
                    "tenancy_ocid": config_data.get("tenancy"),
                    "user_ocid": config_data.get("user"),
                    "fingerprint": config_data.get("fingerprint"),
                    "private_key_path": temp_pem.name,
                    "region": config_data.get("region"),
                    "compartment_id": config_data.get("tenancy")
                }
            )
            if not resolved_comp_id:
                return f"Error: Could not resolve compartment input '{target_compartment_input}' to a valid OCI compartment OCID."

            # Build variables map
            vcn_info = {
                "cidr_block": cidr_block
            }
            if dns_label:
                vcn_info["dns_label"] = dns_label
                
            tfvars = {
                "tenancy_ocid": config_data.get("tenancy"),
                "user_ocid": config_data.get("user"),
                "fingerprint": config_data.get("fingerprint"),
                "private_key_path": temp_pem.name,
                "region": config_data.get("region"),
                "compartment_id": resolved_comp_id,
                "vcns": {
                    vcn_name: vcn_info
                }
            }
            
            # Load existing state if any (for updates/idempotency)
            state_key = f"tf_state_vcn_{vcn_name}"
            existing_state = await ToolStateManager.get_state(thread_id, state_key)
            
            if not existing_state:
                # Fallback: search for this state globally across threads
                async with AsyncSessionLocal() as session:
                    query = select(ToolStateModel).where(ToolStateModel.tool_key == state_key)
                    result = await session.execute(query)
                    tool_state = result.scalars().first()
                    if tool_state:
                        existing_state = process_state_data(tool_state.state, mode="decrypt")
                        logger.info(f"Loaded existing VCN state for {vcn_name} from global DB fallback")
            
            # Execute apply sandbox
            apply_res, output_res, final_state = await run_terraform_in_sandbox(
                tf_dir=TF_DIR,
                env=env,
                tfvars=tfvars,
                target="module.vcn",
                state_content=existing_state
            )
            
            # Save updated state
            if final_state:
                await ToolStateManager.save_state(thread_id, state_key, final_state)
                
        finally:
            if os.path.exists(temp_pem.name):
                os.remove(temp_pem.name)

        if apply_res.returncode != 0:
            logger.error(f"Terraform provision VCN failed: {apply_res.stderr}")
            return f"Error executing Terraform VCN deployment: {apply_res.stderr}"
            
        if not output_res or output_res.returncode != 0:
            return "Error: Could not retrieve VCN output details from Terraform."

        outputs = json.loads(output_res.stdout)
        vcn_ids = outputs.get("vcn_ids", {}).get("value", {})
        vcn_ocid = vcn_ids.get(vcn_name, "Unknown")
        
        return f"Successfully provisioned VCN '{vcn_name}' (OCID: {vcn_ocid}) in compartment: {compartment_id}"
    except Exception as e:
        logger.exception("Failed to provision VCN")
        return f"Error provisioning VCN: {str(e)}"

@tool
async def tf_destroy_vcn(
    vcn_name: str,
    compartment_id: Optional[str] = None,
    config: RunnableConfig = None
) -> str:
    """
    Destroys the specified VCN using Terraform.
    If the VCN was created outside Terraform, it will automatically look up its OCID, import it, and destroy it.
    """
    try:
        creds = await resolve_oci_creds(config)
        # thread_id still needed for state save/load of VCN terraform state
        _cfg = config.get("configurable", {}) if isinstance(config, dict) else (getattr(config, "configurable", {}) if config else {})
        thread_id = _cfg.get("thread_id") or "default_thread"

        config_data = {
            "user": creds.get("user_id"),
            "key_content": creds.get("private_key"),
            "fingerprint": creds.get("fingerprint"),
            "tenancy": creds.get("tenancy_id"),
            "region": creds.get("region"),
            "compartment_id": creds.get("compartment_id")
        }
        target_compartment = compartment_id or config_data.get("compartment_id") or config_data.get("tenancy") or ""

        env = os.environ.copy()
        key_content = config_data.get("key_content", "")
        
        if not key_content:
            return "Error: Private key is empty or missing. Please configure credentials first."
            
        key_content = key_content.strip()
        if "-----BEGIN RSA PRIVATE KEY-----" in key_content:
            raw_b64 = key_content.replace("-----BEGIN RSA PRIVATE KEY-----", "").replace("-----END RSA PRIVATE KEY-----", "")
            raw_b64 = "".join(raw_b64.split())
            chunks = [raw_b64[i:i+64] for i in range(0, len(raw_b64), 64)]
            key_content = "-----BEGIN RSA PRIVATE KEY-----\n" + "\n".join(chunks) + "\n-----END RSA PRIVATE KEY-----\n"
        elif "-----BEGIN PRIVATE KEY-----" in key_content:
            raw_b64 = key_content.replace("-----BEGIN PRIVATE KEY-----", "").replace("-----END PRIVATE KEY-----", "")
            raw_b64 = "".join(raw_b64.split())
            chunks = [raw_b64[i:i+64] for i in range(0, len(raw_b64), 64)]
            key_content = "-----BEGIN PRIVATE KEY-----\n" + "\n".join(chunks) + "\n-----END PRIVATE KEY-----\n"

        temp_pem = tempfile.NamedTemporaryFile(mode="w", suffix=".pem", delete=False, newline="\n")
        try:
            temp_pem.write(key_content)
            temp_pem.close()
            
            tfvars = {
                "tenancy_ocid": config_data.get("tenancy"),
                "user_ocid": config_data.get("user"),
                "fingerprint": config_data.get("fingerprint"),
                "private_key_path": temp_pem.name,
                "region": config_data.get("region"),
                "compartment_id": target_compartment,
                "vcns": {
                    vcn_name: {
                        "cidr_block": "10.0.0.0/16"
                    }
                }
            }
            
            # Load existing state
            state_key = f"tf_state_vcn_{vcn_name}"
            existing_state = await ToolStateManager.get_state(thread_id, state_key)
            
            if not existing_state:
                # Fallback 1: search for this state globally across threads
                async with AsyncSessionLocal() as session:
                    query = select(ToolStateModel).where(ToolStateModel.tool_key == state_key)
                    result = await session.execute(query)
                    tool_state = result.scalars().first()
                    if tool_state:
                        existing_state = process_state_data(tool_state.state, mode="decrypt")
                        logger.info(f"Loaded existing VCN state for {vcn_name} from global DB fallback")
            
            # Fallback 2: If still no state, query OCI to find the untracked VCN's OCID
            vcn_ocid = None
            if not existing_state:
                logger.info(f"VCN '{vcn_name}' state not found in database. Looking up OCID in OCI...")
                
                # Run the listing query to find the VCN
                apply_res, output_res, _ = await run_terraform_in_sandbox(
                    tf_dir=TF_DIR,
                    env=env,
                    tfvars=tfvars,
                    target="data.oci_core_vcns.all_vcns"
                )
                
                if apply_res.returncode == 0 and output_res and output_res.returncode == 0:
                    outputs = json.loads(output_res.stdout)
                    all_vcns = outputs.get("all_vcns", {}).get("value", [])
                    for v in all_vcns:
                        if v.get("display_name") == vcn_name:
                            vcn_ocid = v.get("id")
                            break
                            
            # Case A: We have a tracked state file. Use standard destroy.
            if existing_state:
                destroy_res, _, final_state = await run_terraform_in_sandbox(
                    tf_dir=TF_DIR,
                    env=env,
                    tfvars=tfvars,
                    target="module.vcn",
                    destroy=True,
                    state_content=existing_state
                )
                
                # Delete state key on successful destruction (globally)
                if destroy_res.returncode == 0:
                    async with AsyncSessionLocal() as session:
                        try:
                            from sqlalchemy import delete as sql_delete
                            stmt = sql_delete(ToolStateModel).where(ToolStateModel.tool_key == state_key)
                            await session.execute(stmt)
                            await session.commit()
                        except Exception as e:
                            await session.rollback()
                elif final_state:
                    await ToolStateManager.save_state(thread_id, state_key, final_state)
                    
                if destroy_res.returncode != 0:
                    return f"Error executing Terraform VCN destruction: {destroy_res.stderr}"
                    
                return f"Successfully destroyed VCN '{vcn_name}' using Terraform."
                
            # Case B: Untracked VCN. We found its OCID, so we run a temporary import + destroy.
            elif vcn_ocid:
                logger.info(f"Found VCN '{vcn_name}' in OCI (OCID: {vcn_ocid}). Running auto-import and destroy...")
                
                temp_dir = tempfile.mkdtemp()
                try:
                    for file_name in ["provider.tf", "variables.tf", "main.tf", "outputs.tf"]:
                        shutil.copy(os.path.join(TF_DIR, file_name), temp_dir)
                    shutil.copytree(os.path.join(TF_DIR, "modules"), os.path.join(temp_dir, "modules"))
                    
                    terraform_plugins_src = os.path.join(TF_DIR, ".terraform")
                    if os.path.exists(terraform_plugins_src):
                        try:
                            os.symlink(terraform_plugins_src, os.path.join(temp_dir, ".terraform"), target_is_directory=True)
                        except Exception:
                            shutil.copytree(terraform_plugins_src, os.path.join(temp_dir, ".terraform"))
                    
                    with open(os.path.join(temp_dir, "terraform.tfvars.json"), "w") as f:
                        json.dump(tfvars, f)

                    import sys
                    python_dir = os.path.dirname(sys.executable)
                    tf_cmd = os.path.join(python_dir, "terraform.exe")
                    if not os.path.exists(tf_cmd):
                        tf_cmd = "terraform"

                    # 1. Init
                    await asyncio.to_thread(subprocess.run, [tf_cmd, "init"], cwd=temp_dir, env=env, capture_output=True)
                    
                    # 2. Import
                    import_process = await asyncio.to_thread(
                        subprocess.run,
                        [tf_cmd, "import", f'module.vcn.oci_core_vcn.this["{vcn_name}"]', vcn_ocid],
                        cwd=temp_dir, env=env, capture_output=True, text=True
                    )
                    if import_process.returncode != 0:
                        return f"Error importing VCN for destruction: {import_process.stderr}"

                    # 3. Destroy
                    destroy_process = await asyncio.to_thread(
                        subprocess.run,
                        [tf_cmd, "destroy", "-auto-approve", "-target", f'module.vcn.oci_core_vcn.this["{vcn_name}"]'],
                        cwd=temp_dir, env=env, capture_output=True, text=True
                    )
                    if destroy_process.returncode != 0:
                        return f"Error executing Terraform VCN destruction: {destroy_process.stderr}"
                        
                    return f"Successfully imported and destroyed untracked VCN '{vcn_name}' (OCID: {vcn_ocid}) using Terraform."
                finally:
                    try:
                        shutil.rmtree(temp_dir)
                    except Exception:
                        pass
            else:
                return f"Error: VCN '{vcn_name}' was not found in the OCI compartment (untracked) and has no local state."
                
        finally:
            if os.path.exists(temp_pem.name):
                os.remove(temp_pem.name)
    except Exception as e:
        logger.exception("Failed to destroy VCN")
        return f"Error destroying VCN: {str(e)}"

# Registering VCN Tools
registry.register(
    tf_provision_vcn,
    ToolMeta(
        name="tf_provision_vcn",
        display_name="OCI Terraform: Create VCN",
        description="Creates a new virtual cloud network (VCN) with a specified CIDR block and DNS label using Terraform. CRITICAL: You must ask the user which compartment they want to build the VCN in before calling this.",
        category="Infrastructure",
        package="oci_terraform"
    )
)

registry.register(
    tf_destroy_vcn,
    ToolMeta(
        name="tf_destroy_vcn",
        display_name="OCI Terraform: Delete VCN",
        description="Deletes/destroys an existing virtual cloud network (VCN) using Terraform. Automatically imports and destroys untracked VCNs by name.",
        category="Infrastructure",
        package="oci_terraform"
    )
)

class ListVCNsInput(BaseModel):
    compartment_id: str = Field(description="The name or OCID of the compartment to list VCNs from.")
    detail: bool = Field(default=False, description="Set to True to retrieve detailed information (DNS labels, domain names) for each VCN.")
    vcn_name: Optional[str] = Field(default=None, description="Optional display name or OCID of a specific VCN to filter.")


@tool(args_schema=ListVCNsInput)
async def tf_list_vcns(
    compartment_id: str,
    detail: bool = False,
    vcn_name: Optional[str] = None,
    config: RunnableConfig = None
) -> str:
    """
    Lists all VCNs (both Terraform-managed and untracked ones) inside the specified OCI compartment.
    CRITICAL: You must ask the user which OCI compartment they want to list VCNs in before running this tool.
    """
    try:
        resolved_comp_id = await resolve_compartment_id(compartment_id, config)
        if not resolved_comp_id:
            return f"Error: Could not resolve compartment input '{compartment_id}' to a valid OCI compartment OCID."

        vcns = await run_tf_query(config, "data.oci_core_vcns.all_vcns", {"compartment_id": resolved_comp_id})
        if not vcns:
            return f"No VCNs found or could not retrieve VCN data from compartment: {compartment_id}"

        if vcn_name:
            # Fuzzy match VCN name
            search_norm = normalize_name(vcn_name)
            vcns = [v for v in vcns if search_norm in normalize_name(v.get("display_name", "")) or vcn_name == v.get("id")]
            if not vcns:
                return f"No VCN matching '{vcn_name}' was found in compartment: {compartment_id}"

        result_lines = [f"Found {len(vcns)} VCN(s) in compartment (via Terraform):\n"]
        for vcn in vcns:
            name = vcn.get("display_name", "N/A")
            ocid = vcn.get("id", "N/A")
            cidr = vcn.get("cidr_block", "N/A")
            state = vcn.get("lifecycle_state", vcn.get("state", "N/A"))
            
            if detail or vcn_name:
                dns = vcn.get("dns_label", "N/A")
                domain = vcn.get("vcn_domain_name", "N/A")
                result_lines.append(
                    f"- **{name}** (State: `{state}`)\n"
                    f"  CIDR Block: `{cidr}`\n"
                    f"  DNS Label: `{dns}` | Domain: `{domain}`\n"
                    f"  OCID: `{ocid}`"
                )
            else:
                result_lines.append(f"- **{name}** (CIDR: `{cidr}`)\n  State: `{state}`\n  OCID: `{ocid}`")
            
        return "\n".join(result_lines)
    except Exception as e:
        logger.exception("Failed to list VCNs")
        return f"Error listing VCNs: {str(e)}"

registry.register(
    tf_list_vcns,
    ToolMeta(
        name="tf_list_vcns",
        display_name="OCI Terraform: List VCNs",
        description="Lists all virtual cloud networks (VCNs) in a compartment, including ones not created by Terraform.",
        category="Infrastructure",
        package="oci_terraform"
    )
)