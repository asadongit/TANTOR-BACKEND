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
from Apps.AgentMesh.Utilities.Tools.InfraTool.OCI.OCI_using_terraform.tools.tf_auth_tools import run_terraform_in_sandbox, extract_data_from_state, resolve_oci_creds
from Apps.AgentMesh.Utilities.Tools.InfraTool.OCI.OCI_using_terraform.tools.tf_vcn_tools import resolve_compartment_id

logger = logging.getLogger(__name__)

# Absolute path to the root Terraform OCI directory
TF_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

@tool
async def tf_provision_igw(
    igw_name: str,
    vcn_id: str,
    enabled: Optional[bool] = True,
    compartment_id: Optional[str] = None,
    config: RunnableConfig = None
) -> str:
    """
    Provisions a new Internet Gateway (IGW) inside a VCN in the OCI compartment using Terraform.
    CRITICAL: You must ask the user which OCI compartment they want to list/manage resources in, and ensure it is configured.
    """
    try:
        creds = await resolve_oci_creds(config)
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

            igw_info = {
                "vcn_id": vcn_id,
                "enabled": enabled
            }
                
            tfvars = {
                "tenancy_ocid": config_data.get("tenancy"),
                "user_ocid": config_data.get("user"),
                "fingerprint": config_data.get("fingerprint"),
                "private_key_path": temp_pem.name,
                "region": config_data.get("region"),
                "compartment_id": resolved_comp_id,
                "igws": {
                    igw_name: igw_info
                }
            }
            
            # Load existing state if any (for updates/idempotency)
            state_key = f"tf_state_igw_{igw_name}"
            existing_state = await ToolStateManager.get_state(thread_id, state_key)
            
            if not existing_state:
                # Fallback: search for this state globally across threads
                async with AsyncSessionLocal() as session:
                    query = select(ToolStateModel).where(ToolStateModel.tool_key == state_key)
                    result = await session.execute(query)
                    tool_state = result.scalars().first()
                    if tool_state:
                        existing_state = process_state_data(tool_state.state, mode="decrypt")
                        logger.info(f"Loaded existing IGW state for {igw_name} from global DB fallback")
            
            # Execute apply sandbox
            apply_res, output_res, final_state = await run_terraform_in_sandbox(
                tf_dir=TF_DIR,
                env=env,
                tfvars=tfvars,
                target="module.igw",
                state_content=existing_state
            )
            
            # Save updated state
            if final_state:
                await ToolStateManager.save_state(thread_id, state_key, final_state)
                
        finally:
            if os.path.exists(temp_pem.name):
                os.remove(temp_pem.name)

        if apply_res.returncode != 0:
            logger.error(f"Terraform provision IGW failed: {apply_res.stderr}")
            return f"Error executing Terraform IGW deployment: {apply_res.stderr}"
            
        if not output_res or output_res.returncode != 0:
            return "Error: Could not retrieve IGW output details from Terraform."

        outputs = json.loads(output_res.stdout)
        igw_ids = outputs.get("igw_ids", {}).get("value", {})
        igw_ocid = igw_ids.get(igw_name, "Unknown")
        
        return f"Successfully provisioned Internet Gateway '{igw_name}' (OCID: {igw_ocid}) inside VCN: {vcn_id}"
    except Exception as e:
        logger.exception("Failed to provision IGW")
        return f"Error provisioning Internet Gateway: {str(e)}"

@tool
async def tf_destroy_igw(
    igw_name: str,
    compartment_id: Optional[str] = None,
    config: RunnableConfig = None
) -> str:
    """
    Destroys the specified Internet Gateway using Terraform.
    If the IGW was created outside Terraform, it will automatically look up its OCID, import it, and destroy it.
    """
    try:
        creds = await resolve_oci_creds(config)
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
            
            # Resolve compartment name to OCID
            resolved_comp_id = await resolve_compartment_id(
                target_compartment,
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
                return f"Error: Could not resolve compartment input '{target_compartment}' to a valid OCI compartment OCID."

            # Construct tfvars to import/destroy
            tfvars = {
                "tenancy_ocid": config_data.get("tenancy"),
                "user_ocid": config_data.get("user"),
                "fingerprint": config_data.get("fingerprint"),
                "private_key_path": temp_pem.name,
                "region": config_data.get("region"),
                "compartment_id": resolved_comp_id,
                "igws": {
                    igw_name: {
                        "vcn_id": "dummy_vcn_id"
                    }
                }
            }
            
            # Load existing state
            state_key = f"tf_state_igw_{igw_name}"
            existing_state = await ToolStateManager.get_state(thread_id, state_key)
            
            if not existing_state:
                # Fallback 1: search for this state globally across threads
                async with AsyncSessionLocal() as session:
                    query = select(ToolStateModel).where(ToolStateModel.tool_key == state_key)
                    result = await session.execute(query)
                    tool_state = result.scalars().first()
                    if tool_state:
                        existing_state = process_state_data(tool_state.state, mode="decrypt")
                        logger.info(f"Loaded existing IGW state for {igw_name} from global DB fallback")
            
            # Fallback 2: If still no state, query OCI to find the untracked IGW's OCID
            igw_ocid = None
            if not existing_state:
                logger.info(f"Internet Gateway '{igw_name}' state not found in database. Looking up OCID in OCI...")
                
                apply_res, output_res, _ = await run_terraform_in_sandbox(
                    tf_dir=TF_DIR,
                    env=env,
                    tfvars=tfvars,
                    target="data.oci_core_internet_gateways.all_igws"
                )
                
                if apply_res.returncode == 0 and output_res and output_res.returncode == 0:
                    outputs = json.loads(output_res.stdout)
                    all_igws = outputs.get("all_igws", {}).get("value", [])
                    for i in all_igws:
                        if i.get("display_name") == igw_name:
                            igw_ocid = i.get("id")
                            tfvars["igws"][igw_name]["vcn_id"] = i.get("vcn_id", "dummy_vcn_id")
                            break
                            
            # Case A: We have a tracked state file. Use standard destroy.
            if existing_state:
                destroy_res, _, final_state = await run_terraform_in_sandbox(
                    tf_dir=TF_DIR,
                    env=env,
                    tfvars=tfvars,
                    target="module.igw",
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
                    return f"Error executing Terraform IGW destruction: {destroy_res.stderr}"
                    
                return f"Successfully destroyed Internet Gateway '{igw_name}' using Terraform."
                
            # Case B: Untracked IGW. Run temporary import + destroy.
            elif igw_ocid:
                logger.info(f"Found Internet Gateway '{igw_name}' in OCI (OCID: {igw_ocid}). Running auto-import and destroy...")
                
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
 
                    await asyncio.to_thread(subprocess.run, [tf_cmd, "init"], cwd=temp_dir, env=env, capture_output=True)
                    
                    # Import
                    import_process = await asyncio.to_thread(
                        subprocess.run,
                        [tf_cmd, "import", f'module.igw.oci_core_internet_gateway.this["{igw_name}"]', igw_ocid],
                        cwd=temp_dir, env=env, capture_output=True, text=True
                    )
                    if import_process.returncode != 0:
                        return f"Error importing Internet Gateway for destruction: {import_process.stderr}"
 
                    # Destroy
                    destroy_process = await asyncio.to_thread(
                        subprocess.run,
                        [tf_cmd, "destroy", "-auto-approve", "-target", f'module.igw.oci_core_internet_gateway.this["{igw_name}"]'],
                        cwd=temp_dir, env=env, capture_output=True, text=True
                    )
                    if destroy_process.returncode != 0:
                        return f"Error executing Terraform Internet Gateway destruction: {destroy_process.stderr}"
                        
                    return f"Successfully imported and destroyed untracked Internet Gateway '{igw_name}' (OCID: {igw_ocid}) using Terraform."
                finally:
                    try:
                        shutil.rmtree(temp_dir)
                    except Exception:
                        pass
            else:
                return f"Error: Internet Gateway '{igw_name}' was not found in the OCI compartment (untracked) and has no local state."
                
        finally:
            if os.path.exists(temp_pem.name):
                os.remove(temp_pem.name)
    except Exception as e:
        logger.exception("Failed to destroy Internet Gateway")
        return f"Error destroying Internet Gateway: {str(e)}"

@tool
async def tf_list_igws(
    compartment_id: str,
    vcn_id: Optional[str] = None,
    igw_name: Optional[str] = None,
    detail: bool = False,
    config: RunnableConfig = None
) -> str:
    """
    Lists all Internet Gateways inside the specified OCI compartment, optionally filtered by a VCN ID.
    CRITICAL: You must ask the user which OCI compartment they want to list IGWs in before running this tool.
    """
    try:
        creds = await resolve_oci_creds(config)
 
        config_data = {
            "user": creds.get("user_id"),
            "key_content": creds.get("private_key"),
            "fingerprint": creds.get("fingerprint"),
            "tenancy": creds.get("tenancy_id"),
            "region": creds.get("region"),
            "compartment_id": creds.get("compartment_id")
        }
        target_compartment = compartment_id or config_data.get("compartment_id") or config_data.get("tenancy") or ""
 
        if not target_compartment:
            return "Error: No compartment specified and no default OCI configuration found."
 
        env = os.environ.copy()
        key_content = config_data.get("key_content", "")
        
        if not key_content:
            return "Error: OCI Private key is empty. Please run tf_configure_auth first."
            
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
                target_compartment,
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
                return f"Error: Could not resolve compartment input '{target_compartment}' to a valid OCI compartment OCID."
 
            tfvars = {
                "tenancy_ocid": config_data.get("tenancy"),
                "user_ocid": config_data.get("user"),
                "fingerprint": config_data.get("fingerprint"),
                "private_key_path": temp_pem.name,
                "region": config_data.get("region"),
                "compartment_id": resolved_comp_id,
                "vcn_id": vcn_id or ""
            }
            
            apply_res, output_res, final_state = await run_terraform_in_sandbox(
                tf_dir=TF_DIR,
                env=env,
                tfvars=tfvars,
                target="data.oci_core_internet_gateways.all_igws"
            )
        finally:
            if os.path.exists(temp_pem.name):
                os.remove(temp_pem.name)
 
        if apply_res.returncode != 0:
            return f"Error executing Terraform: {apply_res.stderr}"
            
        igws = None
        if output_res and output_res.returncode == 0:
            try:
                outputs = json.loads(output_res.stdout)
                igws = outputs.get("all_igws", {}).get("value", [])
            except Exception:
                pass
 
        if not igws and final_state:
            # Fallback: Parse directly from state content
            igws = extract_data_from_state(
                final_state,
                "oci_core_internet_gateways",
                "all_igws",
                "gateways"
            )
 
        if not igws:
            return f"No Internet Gateways found or could not retrieve IGW data from compartment: {target_compartment}"
 
        if igw_name:
            igws = [i for i in igws if igw_name.lower() in i.get("display_name", "").lower() or igw_name == i.get("id")]
            if not igws:
                return f"No IGW matching '{igw_name}' was found in compartment: {target_compartment}"

        result_lines = [f"Found {len(igws)} Internet Gateway(s) in compartment (via Terraform):\n"]
        for i in igws:
            name = i.get("display_name", "N/A")
            ocid = i.get("id", "N/A")
            state = i.get("lifecycle_state", i.get("state", "N/A"))
            enabled_status = i.get("is_enabled", i.get("enabled", True))
            v_id = i.get("vcn_id", "N/A")
            
            if detail or igw_name:
                result_lines.append(
                    f"- **{name}** (State: `{state}`)\n"
                    f"  Enabled: `{enabled_status}`\n"
                    f"  VCN OCID: `{v_id}`\n"
                    f"  Time Created: `{i.get('time_created', 'N/A')}`\n"
                    f"  OCID: `{ocid}`"
                )
            else:
                result_lines.append(f"- **{name}** (Enabled: `{enabled_status}`)\n  State: `{state}`\n  VCN OCID: `{v_id}`\n  OCID: `{ocid}`")
            
        return "\n".join(result_lines)
    except Exception as e:
        logger.exception("Failed to list Internet Gateways")
        return f"Error listing Internet Gateways: {str(e)}"

# Registering IGW Tools
registry.register(
    tf_provision_igw,
    ToolMeta(
        name="tf_provision_igw",
        display_name="OCI Terraform: Create Internet Gateway",
        description="Creates a new Internet Gateway inside a VCN using Terraform.",
        category="Infrastructure",
        package="oci_terraform"
    )
)

registry.register(
    tf_destroy_igw,
    ToolMeta(
        name="tf_destroy_igw",
        display_name="OCI Terraform: Delete Internet Gateway",
        description="Deletes/destroys an existing Internet Gateway using Terraform. Automatically imports and destroys untracked IGWs by name.",
        category="Infrastructure",
        package="oci_terraform"
    )
)

registry.register(
    tf_list_igws,
    ToolMeta(
        name="tf_list_igws",
        display_name="OCI Terraform: List Internet Gateways",
        description="Lists all Internet Gateways in a compartment, optionally filtered by a VCN ID.",
        category="Infrastructure",
        package="oci_terraform"
    )
)