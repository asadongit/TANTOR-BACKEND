import os
import json
import logging
import asyncio
import tempfile
import shutil
import subprocess
from typing import Optional, List, Dict, Any
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
async def tf_provision_drg(
    drg_name: str,
    compartment_id: Optional[str] = None,
    config: RunnableConfig = None
) -> str:
    """
    Provisions a new OCI Dynamic Routing Gateway (DRG) using Terraform.
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
            # write temporary key
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

            tfvars = {
                "tenancy_ocid": config_data.get("tenancy"),
                "user_ocid": config_data.get("user"),
                "fingerprint": config_data.get("fingerprint"),
                "private_key_path": temp_pem.name,
                "region": config_data.get("region"),
                "compartment_id": resolved_comp_id,
                "drgs": {
                    drg_name: {}
                }
            }
            
            # Load existing state if any (for updates/idempotency)
            state_key = f"tf_state_drg_{drg_name}"
            existing_state = await ToolStateManager.get_state(thread_id, state_key)
            
            if not existing_state:
                # Fallback: search for this state globally across threads
                async with AsyncSessionLocal() as session:
                    query = select(ToolStateModel).where(ToolStateModel.tool_key == state_key)
                    result = await session.execute(query)
                    tool_state = result.scalars().first()
                    if tool_state:
                        existing_state = process_state_data(tool_state.state, mode="decrypt")
                        logger.info(f"Loaded existing DRG state for {drg_name} from global DB fallback")
            
            # Execute apply sandbox
            apply_res, output_res, final_state = await run_terraform_in_sandbox(
                tf_dir=TF_DIR,
                env=env,
                tfvars=tfvars,
                target="module.drg",
                state_content=existing_state
            )
            
            # Save updated state
            if final_state:
                await ToolStateManager.save_state(thread_id, state_key, final_state)
                
        finally:
            if os.path.exists(temp_pem.name):
                os.remove(temp_pem.name)

        if apply_res.returncode != 0:
            logger.error(f"Terraform provision DRG failed: {apply_res.stderr}")
            return f"Error executing Terraform DRG deployment: {apply_res.stderr}"
            
        if not output_res or output_res.returncode != 0:
            return "Error: Could not retrieve DRG output details from Terraform."

        outputs = json.loads(output_res.stdout)
        drg_ids = outputs.get("drg_ids", {}).get("value", {})
        drg_ocid = drg_ids.get(drg_name, "Unknown")
        
        return f"Successfully provisioned Dynamic Routing Gateway (DRG) '{drg_name}' (OCID: {drg_ocid})"
    except Exception as e:
        logger.exception("Failed to provision DRG")
        return f"Error provisioning DRG: {str(e)}"

@tool
async def tf_attach_vcn_to_drg(
    attachment_name: str,
    drg_id: str,
    vcn_id: str,
    compartment_id: Optional[str] = None,
    config: RunnableConfig = None
) -> str:
    """
    Attaches a VCN network to a Dynamic Routing Gateway (DRG) using Terraform.
    Inputs: attachment_name, drg_id, vcn_id, compartment_id (optional).
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
            # write temporary key
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

            tfvars = {
                "tenancy_ocid": config_data.get("tenancy"),
                "user_ocid": config_data.get("user"),
                "fingerprint": config_data.get("fingerprint"),
                "private_key_path": temp_pem.name,
                "region": config_data.get("region"),
                "compartment_id": resolved_comp_id,
                "drg_attachments": {
                    attachment_name: {
                        "drg_id": drg_id,
                        "vcn_id": vcn_id
                    }
                }
            }
            
            # Load existing state if any (for updates/idempotency)
            state_key = f"tf_state_drg_attachment_{attachment_name}"
            existing_state = await ToolStateManager.get_state(thread_id, state_key)
            
            if not existing_state:
                # Fallback: search for this state globally across threads
                async with AsyncSessionLocal() as session:
                    query = select(ToolStateModel).where(ToolStateModel.tool_key == state_key)
                    result = await session.execute(query)
                    tool_state = result.scalars().first()
                    if tool_state:
                        existing_state = process_state_data(tool_state.state, mode="decrypt")
                        logger.info(f"Loaded existing DRG Attachment state for {attachment_name} from global DB fallback")
            
            # Execute apply sandbox
            apply_res, output_res, final_state = await run_terraform_in_sandbox(
                tf_dir=TF_DIR,
                env=env,
                tfvars=tfvars,
                target="module.drg",
                state_content=existing_state
            )
            
            # Save updated state
            if final_state:
                await ToolStateManager.save_state(thread_id, state_key, final_state)
                
        finally:
            if os.path.exists(temp_pem.name):
                os.remove(temp_pem.name)

        if apply_res.returncode != 0:
            logger.error(f"Terraform VCN DRG attachment failed: {apply_res.stderr}")
            return f"Error executing Terraform VCN DRG attachment: {apply_res.stderr}"
            
        if not output_res or output_res.returncode != 0:
            return "Error: Could not retrieve DRG attachment details from Terraform."

        outputs = json.loads(output_res.stdout)
        attachment_ids = outputs.get("drg_attachment_ids", {}).get("value", {})
        attachment_ocid = attachment_ids.get(attachment_name, "Unknown")
        
        return f"Successfully attached VCN (ID: {vcn_id}) to DRG (ID: {drg_id}) with Attachment OCID: {attachment_ocid}"
    except Exception as e:
        logger.exception("Failed to attach VCN to DRG")
        return f"Error attaching VCN to DRG: {str(e)}"

@tool
async def tf_destroy_drg(
    drg_name: str,
    compartment_id: Optional[str] = None,
    config: RunnableConfig = None
) -> str:
    """
    Destroys the specified Dynamic Routing Gateway (DRG) using Terraform.
    If the DRG was created outside Terraform, it will automatically look up its OCID, import it, and destroy it.
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
            # write temporary key
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
                "drgs": {
                    drg_name: {}
                }
            }
            
            # Load existing state
            state_key = f"tf_state_drg_{drg_name}"
            existing_state = await ToolStateManager.get_state(thread_id, state_key)
            
            if not existing_state:
                # Fallback 1: search for this state globally across threads
                async with AsyncSessionLocal() as session:
                    query = select(ToolStateModel).where(ToolStateModel.tool_key == state_key)
                    result = await session.execute(query)
                    tool_state = result.scalars().first()
                    if tool_state:
                        existing_state = process_state_data(tool_state.state, mode="decrypt")
                        logger.info(f"Loaded existing DRG state for {drg_name} from global DB fallback")
            
            # Fallback 2: If still no state, query OCI to find the untracked DRG's OCID
            drg_ocid = None
            if not existing_state:
                logger.info(f"DRG '{drg_name}' state not found in database. Looking up OCID in OCI...")
                
                apply_res, output_res, _ = await run_terraform_in_sandbox(
                    tf_dir=TF_DIR,
                    env=env,
                    tfvars=tfvars,
                    target="data.oci_core_drgs.all_drgs"
                )
                
                if apply_res.returncode == 0 and output_res and output_res.returncode == 0:
                    outputs = json.loads(output_res.stdout)
                    all_drgs = outputs.get("all_drgs", {}).get("value", [])
                    for d in all_drgs:
                        if d.get("display_name") == drg_name:
                            drg_ocid = d.get("id")
                            break
                            
            # Case A: We have a tracked state file. Use standard destroy.
            if existing_state:
                destroy_res, _, final_state = await run_terraform_in_sandbox(
                    tf_dir=TF_DIR,
                    env=env,
                    tfvars=tfvars,
                    target="module.drg",
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
                    return f"Error executing Terraform DRG destruction: {destroy_res.stderr}"
                    
                return f"Successfully destroyed DRG '{drg_name}' using Terraform."
                
            # Case B: Untracked DRG. Run temporary import + destroy.
            elif drg_ocid:
                logger.info(f"Found DRG '{drg_name}' in OCI (OCID: {drg_ocid}). Running auto-import and destroy...")
                
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
                        [tf_cmd, "import", f'module.drg.oci_core_drg.this["{drg_name}"]', drg_ocid],
                        cwd=temp_dir, env=env, capture_output=True, text=True
                    )
                    if import_process.returncode != 0:
                        return f"Error importing DRG for destruction: {import_process.stderr}"
 
                    # Destroy
                    destroy_process = await asyncio.to_thread(
                        subprocess.run,
                        [tf_cmd, "destroy", "-auto-approve", "-target", f'module.drg.oci_core_drg.this["{drg_name}"]'],
                        cwd=temp_dir, env=env, capture_output=True, text=True
                    )
                    if destroy_process.returncode != 0:
                        return f"Error executing Terraform DRG destruction: {destroy_process.stderr}"
                        
                    return f"Successfully imported and destroyed untracked DRG '{drg_name}' (OCID: {drg_ocid}) using Terraform."
                finally:
                    try:
                        shutil.rmtree(temp_dir)
                    except Exception:
                        pass
            else:
                return f"Error: DRG '{drg_name}' was not found in the OCI compartment (untracked) and has no local state."
                
        finally:
            if os.path.exists(temp_pem.name):
                os.remove(temp_pem.name)
    except Exception as e:
        logger.exception("Failed to destroy DRG")
        return f"Error destroying DRG: {str(e)}"

@tool
async def tf_detach_vcn_from_drg(
    attachment_name: str,
    compartment_id: Optional[str] = None,
    config: RunnableConfig = None
) -> str:
    """
    Detaches a VCN network from a Dynamic Routing Gateway (DRG) by destroying the attachment resource.
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
            # write temporary key
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
                "drg_attachments": {
                    attachment_name: {
                        "drg_id": "dummy_drg_id",
                        "vcn_id": "dummy_vcn_id"
                    }
                }
            }
            
            # Load existing state
            state_key = f"tf_state_drg_attachment_{attachment_name}"
            existing_state = await ToolStateManager.get_state(thread_id, state_key)
            
            if not existing_state:
                # Fallback 1: search globally
                async with AsyncSessionLocal() as session:
                    query = select(ToolStateModel).where(ToolStateModel.tool_key == state_key)
                    result = await session.execute(query)
                    tool_state = result.scalars().first()
                    if tool_state:
                        existing_state = process_state_data(tool_state.state, mode="decrypt")
            
            # Fallback 2: Look up OCID in OCI
            attachment_ocid = None
            if not existing_state:
                logger.info(f"DRG attachment '{attachment_name}' state not found in database. Looking up OCID in OCI...")
                
                apply_res, output_res, _ = await run_terraform_in_sandbox(
                    tf_dir=TF_DIR,
                    env=env,
                    tfvars=tfvars,
                    target="data.oci_core_drg_attachments.all_drg_attachments"
                )
                
                if apply_res.returncode == 0 and output_res and output_res.returncode == 0:
                    outputs = json.loads(output_res.stdout)
                    all_attachments = outputs.get("all_drg_attachments", {}).get("value", [])
                    for a in all_attachments:
                        if a.get("display_name") == attachment_name:
                            attachment_ocid = a.get("id")
                            tfvars["drg_attachments"][attachment_name]["drg_id"] = a.get("drg_id")
                            # Extract VCN ID from network details
                            net_details = a.get("network_details", [{}])[0]
                            tfvars["drg_attachments"][attachment_name]["vcn_id"] = net_details.get("id", "dummy_vcn_id")
                            break
                            
            # Case A: Tracked state
            if existing_state:
                destroy_res, _, final_state = await run_terraform_in_sandbox(
                    tf_dir=TF_DIR,
                    env=env,
                    tfvars=tfvars,
                    target="module.drg",
                    destroy=True,
                    state_content=existing_state
                )
                if destroy_res.returncode == 0:
                    async with AsyncSessionLocal() as session:
                        try:
                            from sqlalchemy import delete as sql_delete
                            stmt = sql_delete(ToolStateModel).where(ToolStateModel.tool_key == state_key)
                            await session.execute(stmt)
                            await session.commit()
                        except Exception:
                            await session.rollback()
                elif final_state:
                    await ToolStateManager.save_state(thread_id, state_key, final_state)
                    
                if destroy_res.returncode != 0:
                    return f"Error executing Terraform DRG detachment: {destroy_res.stderr}"
                return f"Successfully detached DRG Attachment '{attachment_name}'."
                
            # Case B: Untracked
            elif attachment_ocid:
                logger.info(f"Found DRG Attachment '{attachment_name}' in OCI (OCID: {attachment_ocid}). Running auto-import and destroy...")
                
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
                        [tf_cmd, "import", f'module.drg.oci_core_drg_attachment.this["{attachment_name}"]', attachment_ocid],
                        cwd=temp_dir, env=env, capture_output=True, text=True
                    )
                    if import_process.returncode != 0:
                        return f"Error importing DRG attachment: {import_process.stderr}"
 
                    # Destroy
                    destroy_process = await asyncio.to_thread(
                        subprocess.run,
                        [tf_cmd, "destroy", "-auto-approve", "-target", f'module.drg.oci_core_drg_attachment.this["{attachment_name}"]'],
                        cwd=temp_dir, env=env, capture_output=True, text=True
                    )
                    if destroy_process.returncode != 0:
                        return f"Error executing Terraform DRG attachment destruction: {destroy_process.stderr}"
                        
                    return f"Successfully imported and destroyed untracked DRG attachment '{attachment_name}' (OCID: {attachment_ocid})."
                finally:
                    try:
                        shutil.rmtree(temp_dir)
                    except Exception:
                        pass
            else:
                return f"Error: DRG Attachment '{attachment_name}' was not found in OCI (untracked) and has no local state."
        finally:
            if os.path.exists(temp_pem.name):
                os.remove(temp_pem.name)
    except Exception as e:
        logger.exception("Failed to detach VCN from DRG")
        return f"Error detaching VCN from DRG: {str(e)}"

@tool
async def tf_list_drgs(
    compartment_id: str,
    config: RunnableConfig = None
) -> str:
    """
    Lists all Dynamic Routing Gateways (DRGs) inside the specified OCI compartment.
    CRITICAL: You must ask the user which OCI compartment they want to list DRGs in before running this tool.
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
            # write temporary key
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
                "compartment_id": resolved_comp_id
            }
 
            apply_res, output_res, final_state = await run_terraform_in_sandbox(
                tf_dir=TF_DIR,
                env=env,
                tfvars=tfvars,
                target="data.oci_core_drgs.all_drgs"
            )
        finally:
            if os.path.exists(temp_pem.name):
                os.remove(temp_pem.name)
 
        if apply_res.returncode != 0:
            return f"Error executing Terraform: {apply_res.stderr}"
            
        drgs = None
        if output_res and output_res.returncode == 0:
            try:
                outputs = json.loads(output_res.stdout)
                drgs = outputs.get("all_drgs", {}).get("value", [])
            except Exception:
                pass
 
        if not drgs and final_state:
            drgs = extract_data_from_state(
                final_state,
                "oci_core_drgs",
                "all_drgs",
                "drgs"
            )
 
        if not drgs:
            return f"No Dynamic Routing Gateways (DRGs) found in compartment: {target_compartment}"
 
        if drg_name:
            drgs = [d for d in drgs if drg_name.lower() in d.get("display_name", "").lower() or drg_name == d.get("id")]
            if not drgs:
                return f"No DRG matching '{drg_name}' was found in compartment: {target_compartment}"

        result_lines = [f"Found {len(drgs)} Dynamic Routing Gateway(s) in compartment (via Terraform):\n"]
        for d in drgs:
            name = d.get("display_name", "N/A")
            ocid = d.get("id", "N/A")
            state = d.get("lifecycle_state", d.get("state", "N/A"))
            
            if detail or drg_name:
                result_lines.append(
                    f"- **{name}** (State: `{state}`)\n"
                    f"  Time Created: `{d.get('time_created', 'N/A')}`\n"
                    f"  OCID: `{ocid}`"
                )
            else:
                result_lines.append(
                    f"- **{name}** (State: `{state}`)\n"
                    f"  OCID: `{ocid}`"
                )
            
        return "\n".join(result_lines)
    except Exception as e:
        logger.exception("Failed to list DRGs")
        return f"Error listing DRGs: {str(e)}"

@tool
async def tf_list_drg_attachments(
    compartment_id: str,
    drg_id: Optional[str] = None,
    config: RunnableConfig = None
) -> str:
    """
    Lists all Dynamic Routing Gateway (DRG) VCN attachments inside the specified OCI compartment.
    Optionally filters by drg_id.
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
            # write temporary key
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
                "drg_filter_id": drg_id or ""
            }
 
            apply_res, output_res, final_state = await run_terraform_in_sandbox(
                tf_dir=TF_DIR,
                env=env,
                tfvars=tfvars,
                target="data.oci_core_drg_attachments.all_drg_attachments"
            )
        finally:
            if os.path.exists(temp_pem.name):
                os.remove(temp_pem.name)
 
        if apply_res.returncode != 0:
            return f"Error executing Terraform: {apply_res.stderr}"
            
        attachments = None
        if output_res and output_res.returncode == 0:
            try:
                outputs = json.loads(output_res.stdout)
                attachments = outputs.get("all_drg_attachments", {}).get("value", [])
            except Exception:
                pass
 
        if not attachments and final_state:
            attachments = extract_data_from_state(
                final_state,
                "oci_core_drg_attachments",
                "all_drg_attachments",
                "drg_attachments"
            )
 
        if not attachments:
            return f"No DRG attachments found in compartment: {target_compartment}"
 
        if attachment_name:
            attachments = [a for a in attachments if attachment_name.lower() in a.get("display_name", "").lower() or attachment_name == a.get("id")]
            if not attachments:
                return f"No DRG Attachment matching '{attachment_name}' was found in compartment: {target_compartment}"

        result_lines = [f"Found {len(attachments)} DRG Attachment(s) in compartment (via Terraform):\n"]
        for a in attachments:
            name = a.get("display_name", "N/A")
            ocid = a.get("id", "N/A")
            d_id = a.get("drg_id", "N/A")
            state = a.get("lifecycle_state", a.get("state", "N/A"))
            net_details = a.get("network_details", [{}])[0]
            vcn_ocid = net_details.get("id", "N/A")
            
            if detail or attachment_name:
                result_lines.append(
                    f"- **{name}** (State: `{state}`)\n"
                    f"  DRG OCID: `{d_id}`\n"
                    f"  Attached VCN OCID: `{vcn_ocid}`\n"
                    f"  Network Details: `{net_details}`\n"
                    f"  Time Created: `{a.get('time_created', 'N/A')}`\n"
                    f"  OCID: `{ocid}`"
                )
            else:
                result_lines.append(
                    f"- **{name}** (State: `{state}`)\n"
                    f"  DRG OCID: `{d_id}`\n"
                    f"  Attached VCN OCID: `{vcn_ocid}`\n"
                    f"  OCID: `{ocid}`"
                )
            
        return "\n".join(result_lines)
    except Exception as e:
        logger.exception("Failed to list DRG attachments")
        return f"Error listing DRG attachments: {str(e)}"

# Registering DRG Tools
registry.register(
    tf_provision_drg,
    ToolMeta(
        name="tf_provision_drg",
        display_name="OCI Terraform: Create DRG",
        description="Creates a new Dynamic Routing Gateway (DRG) in a compartment using Terraform.",
        category="Infrastructure",
        package="oci_terraform"
    )
)

registry.register(
    tf_attach_vcn_to_drg,
    ToolMeta(
        name="tf_attach_vcn_to_drg",
        display_name="OCI Terraform: Attach VCN to DRG",
        description="Attaches a VCN to a Dynamic Routing Gateway (DRG) using Terraform.",
        category="Infrastructure",
        package="oci_terraform"
    )
)

registry.register(
    tf_destroy_drg,
    ToolMeta(
        name="tf_destroy_drg",
        display_name="OCI Terraform: Delete DRG",
        description="Deletes/destroys an existing Dynamic Routing Gateway using Terraform. Automatically imports and destroys untracked DRGs by name.",
        category="Infrastructure",
        package="oci_terraform"
    )
)

registry.register(
    tf_detach_vcn_from_drg,
    ToolMeta(
        name="tf_detach_vcn_from_drg",
        display_name="OCI Terraform: Detach VCN from DRG",
        description="Detaches a VCN network from a DRG by deleting the attachment.",
        category="Infrastructure",
        package="oci_terraform"
    )
)

registry.register(
    tf_list_drgs,
    ToolMeta(
        name="tf_list_drgs",
        display_name="OCI Terraform: List DRGs",
        description="Lists all Dynamic Routing Gateways (DRGs) in a compartment.",
        category="Infrastructure",
        package="oci_terraform"
    )
)

registry.register(
    tf_list_drg_attachments,
    ToolMeta(
        name="tf_list_drg_attachments",
        display_name="OCI Terraform: List DRG Attachments",
        description="Lists all DRG VCN attachments in a compartment.",
        category="Infrastructure",
        package="oci_terraform"
    )
)