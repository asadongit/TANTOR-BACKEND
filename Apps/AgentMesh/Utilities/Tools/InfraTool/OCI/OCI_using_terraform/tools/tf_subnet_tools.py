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
async def tf_provision_subnet(
    subnet_name: str,
    cidr_block: str,
    vcn_id: str,
    dns_label: Optional[str] = None,
    compartment_id: Optional[str] = None,
    config: RunnableConfig = None
) -> str:
    """
    Provisions a new Subnet inside a VCN in the OCI compartment using Terraform.
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

            subnet_info = {
                "vcn_id": vcn_id,
                "cidr_block": cidr_block
            }
            if dns_label:
                subnet_info["dns_label"] = dns_label
                
            tfvars = {
                "tenancy_ocid": config_data.get("tenancy"),
                "user_ocid": config_data.get("user"),
                "fingerprint": config_data.get("fingerprint"),
                "private_key_path": temp_pem.name,
                "region": config_data.get("region"),
                "compartment_id": resolved_comp_id,
                "subnets": {
                    subnet_name: subnet_info
                }
            }
            
            # Load existing state if any (for updates/idempotency)
            state_key = f"tf_state_subnet_{subnet_name}"
            existing_state = await ToolStateManager.get_state(thread_id, state_key)
            
            if not existing_state:
                # Fallback: search for this state globally across threads
                async with AsyncSessionLocal() as session:
                    query = select(ToolStateModel).where(ToolStateModel.tool_key == state_key)
                    result = await session.execute(query)
                    tool_state = result.scalars().first()
                    if tool_state:
                        existing_state = process_state_data(tool_state.state, mode="decrypt")
                        logger.info(f"Loaded existing Subnet state for {subnet_name} from global DB fallback")
            
            # Execute apply sandbox
            apply_res, output_res, final_state = await run_terraform_in_sandbox(
                tf_dir=TF_DIR,
                env=env,
                tfvars=tfvars,
                target="module.subnet",
                state_content=existing_state
            )
            
            # Save updated state
            if final_state:
                await ToolStateManager.save_state(thread_id, state_key, final_state)
                
        finally:
            if os.path.exists(temp_pem.name):
                os.remove(temp_pem.name)

        if apply_res.returncode != 0:
            logger.error(f"Terraform provision Subnet failed: {apply_res.stderr}")
            return f"Error executing Terraform Subnet deployment: {apply_res.stderr}"
            
        if not output_res or output_res.returncode != 0:
            return "Error: Could not retrieve Subnet output details from Terraform."

        outputs = json.loads(output_res.stdout)
        subnet_ids = outputs.get("subnet_ids", {}).get("value", {})
        subnet_ocid = subnet_ids.get(subnet_name, "Unknown")
        
        return f"Successfully provisioned Subnet '{subnet_name}' (OCID: {subnet_ocid}) inside VCN: {vcn_id}"
    except Exception as e:
        logger.exception("Failed to provision Subnet")
        return f"Error provisioning Subnet: {str(e)}"

@tool
async def tf_destroy_subnet(
    subnet_name: str,
    compartment_id: Optional[str] = None,
    config: RunnableConfig = None
) -> str:
    """
    Destroys the specified Subnet using Terraform.
    If the Subnet was created outside Terraform, it will automatically look up its OCID, import it, and destroy it.
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
                "subnets": {
                    subnet_name: {
                        "vcn_id": "dummy_vcn_id",
                        "cidr_block": "10.0.1.0/24"
                    }
                }
            }
            
            # Load existing state
            state_key = f"tf_state_subnet_{subnet_name}"
            existing_state = await ToolStateManager.get_state(thread_id, state_key)
            
            if not existing_state:
                # Fallback 1: search for this state globally across threads
                async with AsyncSessionLocal() as session:
                    query = select(ToolStateModel).where(ToolStateModel.tool_key == state_key)
                    result = await session.execute(query)
                    tool_state = result.scalars().first()
                    if tool_state:
                        existing_state = process_state_data(tool_state.state, mode="decrypt")
                        logger.info(f"Loaded existing Subnet state for {subnet_name} from global DB fallback")
            
            # Fallback 2: If still no state, query OCI to find the untracked subnet's OCID
            subnet_ocid = None
            if not existing_state:
                logger.info(f"Subnet '{subnet_name}' state not found in database. Looking up OCID in OCI...")
                
                apply_res, output_res, _ = await run_terraform_in_sandbox(
                    tf_dir=TF_DIR,
                    env=env,
                    tfvars=tfvars,
                    target="data.oci_core_subnets.all_subnets"
                )
                
                if apply_res.returncode == 0 and output_res and output_res.returncode == 0:
                    outputs = json.loads(output_res.stdout)
                    all_subnets = outputs.get("all_subnets", {}).get("value", [])
                    for s in all_subnets:
                        if s.get("display_name") == subnet_name:
                            subnet_ocid = s.get("id")
                            tfvars["subnets"][subnet_name]["vcn_id"] = s.get("vcn_id", "dummy_vcn_id")
                            break
                            
            # Case A: We have a tracked state file. Use standard destroy.
            if existing_state:
                destroy_res, _, final_state = await run_terraform_in_sandbox(
                    tf_dir=TF_DIR,
                    env=env,
                    tfvars=tfvars,
                    target="module.subnet",
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
                    return f"Error executing Terraform Subnet destruction: {destroy_res.stderr}"
                    
                return f"Successfully destroyed Subnet '{subnet_name}' using Terraform."
                
            # Case B: Untracked subnet. Run temporary import + destroy.
            elif subnet_ocid:
                logger.info(f"Found Subnet '{subnet_name}' in OCI (OCID: {subnet_ocid}). Running auto-import and destroy...")
                
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
                        [tf_cmd, "import", f'module.subnet.oci_core_subnet.this["{subnet_name}"]', subnet_ocid],
                        cwd=temp_dir, env=env, capture_output=True, text=True
                    )
                    if import_process.returncode != 0:
                        return f"Error importing Subnet for destruction: {import_process.stderr}"
 
                    # Destroy
                    destroy_process = await asyncio.to_thread(
                        subprocess.run,
                        [tf_cmd, "destroy", "-auto-approve", "-target", f'module.subnet.oci_core_subnet.this["{subnet_name}"]'],
                        cwd=temp_dir, env=env, capture_output=True, text=True
                    )
                    if destroy_process.returncode != 0:
                        return f"Error executing Terraform Subnet destruction: {destroy_process.stderr}"
                        
                    return f"Successfully imported and destroyed untracked Subnet '{subnet_name}' (OCID: {subnet_ocid}) using Terraform."
                finally:
                    try:
                        shutil.rmtree(temp_dir)
                    except Exception:
                        pass
            else:
                return f"Error: Subnet '{subnet_name}' was not found in the OCI compartment (untracked) and has no local state."
                
        finally:
            if os.path.exists(temp_pem.name):
                os.remove(temp_pem.name)
    except Exception as e:
        logger.exception("Failed to destroy Subnet")
        return f"Error destroying Subnet: {str(e)}"

@tool
async def tf_list_subnets(
    compartment_id: str,
    vcn_id: Optional[str] = None,
    config: RunnableConfig = None
) -> str:
    """
    Lists all Subnets inside the specified OCI compartment, optionally filtered by a specific VCN ID.
    CRITICAL: You must ask the user which OCI compartment they want to list subnets in before running this tool.
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
                target="data.oci_core_subnets.all_subnets"
            )
        finally:
            if os.path.exists(temp_pem.name):
                os.remove(temp_pem.name)
 
        if apply_res.returncode != 0:
            return f"Error executing Terraform: {apply_res.stderr}"
            
        subnets = None
        if output_res and output_res.returncode == 0:
            try:
                outputs = json.loads(output_res.stdout)
                subnets = outputs.get("all_subnets", {}).get("value", [])
            except Exception:
                pass
 
        if not subnets and final_state:
            # Fallback: Parse directly from state content
            subnets = extract_data_from_state(
                final_state,
                "oci_core_subnets",
                "all_subnets",
                "subnets"
            )
 
        if not subnets:
            return f"No subnets found or could not retrieve subnet data from compartment: {target_compartment}"
 
        result_lines = [f"Found {len(subnets)} subnet(s) in compartment (via Terraform):\n"]
        for sub in subnets:
            name = sub.get("display_name", "N/A")
            ocid = sub.get("id", "N/A")
            cidr = sub.get("cidr_block", "N/A")
            state = sub.get("state", sub.get("lifecycle_state", "N/A"))
            v_id = sub.get("vcn_id", "N/A")
            result_lines.append(f"- **{name}** (CIDR: `{cidr}`)\n  State: `{state}`\n  VCN OCID: `{v_id}`\n  OCID: `{ocid}`")
            
        return "\n".join(result_lines)
    except Exception as e:
        logger.exception("Failed to list subnets")
        return f"Error listing subnets: {str(e)}"

@tool
async def tf_update_subnet_security_lists(
    subnet_name: str,
    security_list_ids: list,
    compartment_id: Optional[str] = None,
    config: RunnableConfig = None
) -> str:
    """
    Updates/attaches the specified security list OCIDs to an existing Subnet using Terraform.
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

            state_key = f"tf_state_subnet_{subnet_name}"
            existing_state = await ToolStateManager.get_state(thread_id, state_key)
            
            if not existing_state:
                # Fallback: search for this state globally across threads
                async with AsyncSessionLocal() as session:
                    query = select(ToolStateModel).where(ToolStateModel.tool_key == state_key)
                    result = await session.execute(query)
                    tool_state = result.scalars().first()
                    if tool_state:
                        existing_state = process_state_data(tool_state.state, mode="decrypt")
                        logger.info(f"Loaded existing Subnet state for {subnet_name} from global DB fallback")

            vcn_id = None
            cidr_block = None
            dns_label = None
            subnet_ocid = None

            existing_route_table_id = None
            if existing_state:
                try:
                    state_json = json.loads(existing_state)
                    for resource in state_json.get("resources", []):
                        if resource.get("type") == "oci_core_subnet" and resource.get("name") == "this":
                            instances = resource.get("instances", [])
                            if instances:
                                attributes = instances[0].get("attributes", {})
                                vcn_id = attributes.get("vcn_id")
                                cidr_block = attributes.get("cidr_block")
                                dns_label = attributes.get("dns_label")
                                existing_route_table_id = attributes.get("route_table_id")
                                break
                except Exception as e:
                    logger.warning(f"Failed to parse subnet details from existing state: {e}")

            # If not found in state, look it up in OCI
            if not vcn_id or not cidr_block:
                logger.info(f"Subnet '{subnet_name}' not found in local state. Looking up in OCI...")
                tfvars_lookup = {
                    "tenancy_ocid": config_data.get("tenancy"),
                    "user_ocid": config_data.get("user"),
                    "fingerprint": config_data.get("fingerprint"),
                    "private_key_path": temp_pem.name,
                    "region": config_data.get("region"),
                    "compartment_id": resolved_comp_id
                }
                apply_res, output_res, _ = await run_terraform_in_sandbox(
                    tf_dir=TF_DIR,
                    env=env,
                    tfvars=tfvars_lookup,
                    target="data.oci_core_subnets.all_subnets"
                )
                if apply_res.returncode == 0 and output_res and output_res.returncode == 0:
                    outputs = json.loads(output_res.stdout)
                    all_subnets = outputs.get("all_subnets", {}).get("value", [])
                    for s in all_subnets:
                        if s.get("display_name") == subnet_name:
                            subnet_ocid = s.get("id")
                            vcn_id = s.get("vcn_id")
                            cidr_block = s.get("cidr_block")
                            dns_label = s.get("dns_label")
                            existing_route_table_id = s.get("route_table_id")
                            break

            if not vcn_id or not cidr_block:
                return f"Error: Subnet '{subnet_name}' was not found in OCI or in local state."

            # Construct tfvars for the apply/update
            subnet_payload = {
                "vcn_id": vcn_id,
                "cidr_block": cidr_block,
                "dns_label": dns_label,
                "security_list_ids": security_list_ids
            }
            if existing_route_table_id is not None:
                subnet_payload["route_table_id"] = existing_route_table_id

            tfvars = {
                "tenancy_ocid": config_data.get("tenancy"),
                "user_ocid": config_data.get("user"),
                "fingerprint": config_data.get("fingerprint"),
                "private_key_path": temp_pem.name,
                "region": config_data.get("region"),
                "compartment_id": resolved_comp_id,
                "subnets": {
                    subnet_name: subnet_payload
                }
            }

            # If untracked, run import first to initialize the state
            if not existing_state and subnet_ocid:
                logger.info(f"Subnet '{subnet_name}' is untracked. Running temporary import...")
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
                    
                    import_process = await asyncio.to_thread(
                        subprocess.run,
                        [tf_cmd, "import", f'module.subnet.oci_core_subnet.this["{subnet_name}"]', subnet_ocid],
                        cwd=temp_dir, env=env, capture_output=True, text=True
                    )
                    if import_process.returncode == 0:
                        state_path = os.path.join(temp_dir, "terraform.tfstate")
                        if os.path.exists(state_path):
                            with open(state_path, "r") as f:
                                existing_state = f.read()
                finally:
                    try:
                        shutil.rmtree(temp_dir)
                    except Exception:
                        pass

            # Execute apply sandbox to update security lists
            apply_res, output_res, final_state = await run_terraform_in_sandbox(
                tf_dir=TF_DIR,
                env=env,
                tfvars=tfvars,
                target="module.subnet",
                state_content=existing_state
            )
            
            # Save updated state
            if final_state:
                await ToolStateManager.save_state(thread_id, state_key, final_state)
                
        finally:
            if os.path.exists(temp_pem.name):
                os.remove(temp_pem.name)

        if apply_res.returncode != 0:
            logger.error(f"Terraform update Subnet failed: {apply_res.stderr}")
            return f"Error executing Terraform Subnet update: {apply_res.stderr}"
            
        return f"Successfully updated Subnet '{subnet_name}' with security lists: {security_list_ids}"
    except Exception as e:
        logger.exception("Failed to update Subnet security lists")
        return f"Error updating Subnet security lists: {str(e)}"

@tool
async def tf_update_subnet_route_table(
    subnet_name: str,
    route_table_id: str,
    compartment_id: Optional[str] = None,
    config: RunnableConfig = None
) -> str:
    """
    Updates/attaches the specified Route Table OCID to an existing Subnet using Terraform.
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

            state_key = f"tf_state_subnet_{subnet_name}"
            existing_state = await ToolStateManager.get_state(thread_id, state_key)
            
            if not existing_state:
                # Fallback: search for this state globally across threads
                async with AsyncSessionLocal() as session:
                    query = select(ToolStateModel).where(ToolStateModel.tool_key == state_key)
                    result = await session.execute(query)
                    tool_state = result.scalars().first()
                    if tool_state:
                        existing_state = process_state_data(tool_state.state, mode="decrypt")
                        logger.info(f"Loaded existing Subnet state for {subnet_name} from global DB fallback")

            vcn_id = None
            cidr_block = None
            dns_label = None
            subnet_ocid = None
            existing_security_list_ids = None

            if existing_state:
                try:
                    state_json = json.loads(existing_state)
                    for resource in state_json.get("resources", []):
                        if resource.get("type") == "oci_core_subnet" and resource.get("name") == "this":
                            instances = resource.get("instances", [])
                            if instances:
                                attributes = instances[0].get("attributes", {})
                                vcn_id = attributes.get("vcn_id")
                                cidr_block = attributes.get("cidr_block")
                                dns_label = attributes.get("dns_label")
                                existing_security_list_ids = attributes.get("security_list_ids")
                                break
                except Exception as e:
                    logger.warning(f"Failed to parse subnet details from existing state: {e}")

            # If not found in state, look it up in OCI
            if not vcn_id or not cidr_block:
                logger.info(f"Subnet '{subnet_name}' not found in local state. Looking up in OCI...")
                tfvars_lookup = {
                    "tenancy_ocid": config_data.get("tenancy"),
                    "user_ocid": config_data.get("user"),
                    "fingerprint": config_data.get("fingerprint"),
                    "private_key_path": temp_pem.name,
                    "region": config_data.get("region"),
                    "compartment_id": resolved_comp_id
                }
                apply_res, output_res, _ = await run_terraform_in_sandbox(
                    tf_dir=TF_DIR,
                    env=env,
                    tfvars=tfvars_lookup,
                    target="data.oci_core_subnets.all_subnets"
                )
                if apply_res.returncode == 0 and output_res and output_res.returncode == 0:
                    outputs = json.loads(output_res.stdout)
                    all_subnets = outputs.get("all_subnets", {}).get("value", [])
                    for s in all_subnets:
                        if s.get("display_name") == subnet_name:
                            subnet_ocid = s.get("id")
                            vcn_id = s.get("vcn_id")
                            cidr_block = s.get("cidr_block")
                            dns_label = s.get("dns_label")
                            existing_security_list_ids = s.get("security_list_ids")
                            break

            if not vcn_id or not cidr_block:
                return f"Error: Subnet '{subnet_name}' was not found in OCI or in local state."

            # Construct tfvars for the apply/update
            subnet_payload = {
                "vcn_id": vcn_id,
                "cidr_block": cidr_block,
                "dns_label": dns_label,
                "route_table_id": route_table_id
            }
            if existing_security_list_ids is not None:
                subnet_payload["security_list_ids"] = existing_security_list_ids

            tfvars = {
                "tenancy_ocid": config_data.get("tenancy"),
                "user_ocid": config_data.get("user"),
                "fingerprint": config_data.get("fingerprint"),
                "private_key_path": temp_pem.name,
                "region": config_data.get("region"),
                "compartment_id": resolved_comp_id,
                "subnets": {
                    subnet_name: subnet_payload
                }
            }

            # If untracked, run import first to initialize the state
            if not existing_state and subnet_ocid:
                logger.info(f"Subnet '{subnet_name}' is untracked. Running temporary import...")
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
                    
                    import_process = await asyncio.to_thread(
                        subprocess.run,
                        [tf_cmd, "import", f'module.subnet.oci_core_subnet.this["{subnet_name}"]', subnet_ocid],
                        cwd=temp_dir, env=env, capture_output=True, text=True
                    )
                    if import_process.returncode == 0:
                        state_path = os.path.join(temp_dir, "terraform.tfstate")
                        if os.path.exists(state_path):
                            with open(state_path, "r") as f:
                                existing_state = f.read()
                finally:
                    try:
                        shutil.rmtree(temp_dir)
                    except Exception:
                        pass

            # Execute apply sandbox to update route table
            apply_res, output_res, final_state = await run_terraform_in_sandbox(
                tf_dir=TF_DIR,
                env=env,
                tfvars=tfvars,
                target="module.subnet",
                state_content=existing_state
            )
            
            # Save updated state
            if final_state:
                await ToolStateManager.save_state(thread_id, state_key, final_state)
                
        finally:
            if os.path.exists(temp_pem.name):
                os.remove(temp_pem.name)

        if apply_res.returncode != 0:
            logger.error(f"Terraform update Subnet Route Table failed: {apply_res.stderr}")
            return f"Error executing Terraform Subnet Route Table update: {apply_res.stderr}"
            
        return f"Successfully updated Subnet '{subnet_name}' with route table: {route_table_id}"
    except Exception as e:
        logger.exception("Failed to update Subnet route table")
        return f"Error updating Subnet route table: {str(e)}"

# Registering Subnet Tools
registry.register(
    tf_update_subnet_security_lists,
    ToolMeta(
        name="tf_update_subnet_security_lists",
        display_name="OCI Terraform: Update Subnet Security Lists",
        description="Updates the security lists associated with a subnet by attaching/detaching security list OCIDs.",
        category="Infrastructure",
        package="oci_terraform"
    )
)
registry.register(
    tf_update_subnet_route_table,
    ToolMeta(
        name="tf_update_subnet_route_table",
        display_name="OCI Terraform: Update Subnet Route Table",
        description="Updates the Route Table associated with a Subnet.",
        category="Infrastructure",
        package="oci_terraform"
    )
)
registry.register(
    tf_provision_subnet,
    ToolMeta(
        name="tf_provision_subnet",
        display_name="OCI Terraform: Create Subnet",
        description="Creates a new subnet inside an existing VCN in the specified compartment using Terraform.",
        category="Infrastructure",
        package="oci_terraform"
    )
)

registry.register(
    tf_destroy_subnet,
    ToolMeta(
        name="tf_destroy_subnet",
        display_name="OCI Terraform: Delete Subnet",
        description="Deletes/destroys an existing subnet using Terraform. Automatically imports and destroys untracked subnets by name.",
        category="Infrastructure",
        package="oci_terraform"
    )
)

registry.register(
    tf_list_subnets,
    ToolMeta(
        name="tf_list_subnets",
        display_name="OCI Terraform: List Subnets",
        description="Lists all subnets in a compartment, optionally filtered by a VCN ID.",
        category="Infrastructure",
        package="oci_terraform"
    )
)