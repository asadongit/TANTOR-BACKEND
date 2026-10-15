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
from Apps.AgentMesh.Utilities.Tools.InfraTool.OCI.OCI_using_terraform.tools.tf_auth_tools import (
    run_terraform_in_sandbox,
    extract_data_from_state,
    resolve_oci_creds,
    run_tf_query,
    resolve_compartment_id,
    resolve_vm_ocid,
    normalize_name,
    resolve_volume_ocid
)
from Apps.AgentMesh.Utilities.Tools.InfraTool.OCI.OCI_using_terraform.tools.tf_compute_tools import resolve_availability_domain, tf_ssh_executor, get_workflow_tool_config

logger = logging.getLogger(__name__)

# Absolute path to the root Terraform OCI directory
TF_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

class ProvisionVolumeInput(BaseModel):
    volume_name: str = Field(description="The display name of the volume to create.")
    availability_domain: Optional[str] = Field(default=None, description="Optional availability domain. Defaults to the first AD.")
    size_in_gbs: int = Field(description="The size of the volume in GBs (minimum 50).")
    compartment_id: Optional[str] = Field(default=None, description="The name or OCID of the target compartment.")

@tool(args_schema=ProvisionVolumeInput)
async def tf_provision_volume(
    volume_name: str,
    availability_domain: Optional[str] = None,
    size_in_gbs: int = 50,
    compartment_id: Optional[str] = None,
    config: RunnableConfig = None
) -> str:
    """
    Provisions a new block volume in OCI using Terraform (default 50 GB).
    Inputs: volume_name, availability_domain (optional, default first AD), size_in_gbs (optional, default 50).
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
            resolved_comp_id = await resolve_compartment_id(target_compartment_input, config)
            if not resolved_comp_id:
                return f"Error: Could not resolve compartment input '{target_compartment_input}' to a valid OCI compartment OCID."

            # Resolve Availability Domain
            resolved_ad = await resolve_availability_domain(availability_domain, TF_DIR, env, {
                "tenancy_ocid": config_data.get("tenancy"),
                "user_ocid": config_data.get("user"),
                "fingerprint": config_data.get("fingerprint"),
                "private_key_path": temp_pem.name,
                "region": config_data.get("region"),
                "compartment_id": resolved_comp_id
            })
            if not resolved_ad:
                return f"Error: Could not resolve availability domain '{availability_domain}' in OCI region."

            tfvars = {
                "tenancy_ocid": config_data.get("tenancy"),
                "user_ocid": config_data.get("user"),
                "fingerprint": config_data.get("fingerprint"),
                "private_key_path": temp_pem.name,
                "region": config_data.get("region"),
                "compartment_id": resolved_comp_id,
                "volumes": {
                    volume_name: {
                        "availability_domain": resolved_ad,
                        "size_in_gbs": size_in_gbs
                    }
                }
            }
            
            # Load existing state if any (for updates/idempotency)
            state_key = f"tf_state_volume_{volume_name}"
            existing_state = await ToolStateManager.get_state(thread_id, state_key)
            
            if not existing_state:
                # Fallback: search globally
                async with AsyncSessionLocal() as session:
                    query = select(ToolStateModel).where(ToolStateModel.tool_key == state_key)
                    result = await session.execute(query)
                    tool_state = result.scalars().first()
                    if tool_state:
                        existing_state = process_state_data(tool_state.state, mode="decrypt")
            
            # Execute apply sandbox
            apply_res, output_res, final_state = await run_terraform_in_sandbox(
                tf_dir=TF_DIR,
                env=env,
                tfvars=tfvars,
                target="module.block_volume",
                state_content=existing_state
            )
            
            # Save updated state
            if final_state:
                await ToolStateManager.save_state(thread_id, state_key, final_state)
                
        finally:
            if os.path.exists(temp_pem.name):
                os.remove(temp_pem.name)

        if apply_res.returncode != 0:
            logger.error(f"Terraform provision volume failed: {apply_res.stderr}")
            return f"Error executing Terraform Block Volume deployment: {apply_res.stderr}"
            
        if not output_res or output_res.returncode != 0:
            return "Error: Could not retrieve volume details from Terraform."

        outputs = json.loads(output_res.stdout)
        vol_ids = outputs.get("volume_ids", {}).get("value", {})
        vol_ocid = vol_ids.get(volume_name, "Unknown")
        
        return f"Successfully provisioned Block Volume '{volume_name}' (OCID: {vol_ocid}, Size: {size_in_gbs} GB)"
    except Exception as e:
        logger.exception("Failed to provision volume")
        return f"Error provisioning volume: {str(e)}"



class ResizeVolumeInput(BaseModel):
    volume_id: str = Field(description="The display name or OCID of the block volume to resize/extend.")
    size_in_gbs: int = Field(description="The new target size of the block volume in GBs (must be larger than the current size).")
    compartment_id: Optional[str] = Field(default=None, description="Optional name or OCID of the compartment where the volume resides.")


@tool(args_schema=ResizeVolumeInput)
async def tf_resize_block_volume(
    volume_id: str,
    size_in_gbs: int,
    compartment_id: Optional[str] = None,
    config: RunnableConfig = None
) -> str:
    """
    Resizes/extends an existing block volume in-place to a larger size using Terraform.
    Inputs: volume_id (name or OCID), size_in_gbs (new size in GBs).
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
        key_content = config_data.get("key_content", "").strip()
        
        if not key_content:
            return "Error: Private key is empty or missing. Please configure credentials first."
            
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

            resolved_comp_id = await resolve_compartment_id(target_compartment_input, config)
            if not resolved_comp_id:
                return f"Error: Could not resolve compartment input '{target_compartment_input}' to a valid OCI compartment OCID."

            resolved_vol_id = await resolve_volume_ocid(volume_id, resolved_comp_id, config)
            if not resolved_vol_id:
                return f"Error: Could not resolve block volume input '{volume_id}' to a valid OCI volume OCID."

            # Find display name and AD of the volume from live list
            volume_name = volume_id
            resolved_ad = None
            vols = await run_tf_query(config, "data.oci_core_volumes.all_volumes", {"compartment_id": resolved_comp_id})
            if vols:
                for v in vols:
                    if v.get("id") == resolved_vol_id or v.get("display_name") == volume_id:
                        volume_name = v.get("display_name")
                        resolved_ad = v.get("availability_domain")
                        break

            if not resolved_ad:
                resolved_ad = await resolve_availability_domain(None, TF_DIR, env, {
                    "tenancy_ocid": config_data.get("tenancy"),
                    "user_ocid": config_data.get("user"),
                    "fingerprint": config_data.get("fingerprint"),
                    "private_key_path": temp_pem.name,
                    "region": config_data.get("region"),
                    "compartment_id": resolved_comp_id
                })

            tfvars = {
                "tenancy_ocid": config_data.get("tenancy"),
                "user_ocid": config_data.get("user"),
                "fingerprint": config_data.get("fingerprint"),
                "private_key_path": temp_pem.name,
                "region": config_data.get("region"),
                "compartment_id": resolved_comp_id,
                "volumes": {
                    volume_name: {
                        "availability_domain": resolved_ad,
                        "size_in_gbs": size_in_gbs
                    }
                }
            }

            state_key = f"tf_state_volume_{volume_name}"
            existing_state = await ToolStateManager.get_state(thread_id, state_key)
            if not existing_state:
                # Fallback: search globally
                async with AsyncSessionLocal() as session:
                    query = select(ToolStateModel).where(ToolStateModel.tool_key == state_key)
                    result = await session.execute(query)
                    tool_state = result.scalars().first()
                    if tool_state:
                        existing_state = process_state_data(tool_state.state, mode="decrypt")

            # If untracked volume, run dynamic import to bring it into existing_state
            if not existing_state and resolved_vol_id:
                temp_dir = tempfile.mkdtemp()
                try:
                    for file_name in ["provider.tf", "variables.tf", "main.tf", "outputs.tf"]:
                        src_file = os.path.join(TF_DIR, file_name)
                        if os.path.exists(src_file):
                            shutil.copy(src_file, temp_dir)
                    terraform_plugins_src = os.path.join(TF_DIR, ".terraform")
                    if os.path.exists(terraform_plugins_src):
                        try:
                            os.symlink(terraform_plugins_src, os.path.join(temp_dir, ".terraform"), target_is_directory=True)
                        except Exception:
                            shutil.copytree(terraform_plugins_src, os.path.join(temp_dir, ".terraform"))
                            
                    tfvars_path = os.path.join(temp_dir, "terraform.tfvars.json")
                    with open(tfvars_path, "w") as f:
                        json.dump(tfvars, f)
                        
                    import sys
                    python_dir = os.path.dirname(sys.executable)
                    tf_cmd = os.path.join(python_dir, "terraform.exe")
                    if not os.path.exists(tf_cmd):
                        tf_cmd = "terraform"
                    await asyncio.to_thread(subprocess.run, [tf_cmd, "init"], cwd=temp_dir, env=env, capture_output=True)
                    
                    import_process = await asyncio.to_thread(
                        subprocess.run,
                        [tf_cmd, "import", f'module.block_volume.oci_core_volume.this["{volume_name}"]', resolved_vol_id],
                        cwd=temp_dir, env=env, capture_output=True, text=True
                    )
                    if import_process.returncode == 0:
                        state_path = os.path.join(temp_dir, "terraform.tfstate")
                        if os.path.exists(state_path):
                            with open(state_path, "r") as sf:
                                existing_state = sf.read()
                except Exception:
                    pass
                finally:
                    try:
                        shutil.rmtree(temp_dir)
                    except Exception:
                        pass

            # Execute resize apply in sandbox
            apply_res, output_res, final_state = await run_terraform_in_sandbox(
                tf_dir=TF_DIR,
                env=env,
                tfvars=tfvars,
                target="module.block_volume",
                state_content=existing_state
            )
            
            if final_state:
                await ToolStateManager.save_state(thread_id, state_key, final_state)

        finally:
            if os.path.exists(temp_pem.name):
                try:
                    os.remove(temp_pem.name)
                except Exception:
                    pass

        if apply_res.returncode != 0:
            return f"Error executing block volume resize: {apply_res.stderr}"
            
        return f"Successfully resized Block Volume '{volume_name}' to {size_in_gbs} GB in-place."
    except Exception as e:
        logger.exception("Failed to resize block volume")
        return f"Error resizing block volume: {str(e)}"


class AttachVolumeInput(BaseModel):
    attachment_name: str = Field(description="The display name for the block volume attachment resource.")
    volume_id: str = Field(description="The name or OCID of the target block volume.")
    instance_id: str = Field(description="The display name or OCID of the target VM instance.")
    attachment_type: str = Field(default="paravirtualized", description="The attachment type: 'iscsi' or 'paravirtualized'.")
    compartment_id: Optional[str] = Field(default=None, description="The name or OCID of the target compartment.")

@tool(args_schema=AttachVolumeInput)
async def tf_attach_volume(
    attachment_name: str,
    volume_id: str,
    instance_id: str,
    attachment_type: str = "paravirtualized",
    compartment_id: Optional[str] = None,
    config: RunnableConfig = None
) -> str:
    """
    Attaches an existing Block Volume to a VM Compute instance using Terraform.
    Inputs: attachment_name, volume_id (name or OCID), instance_id (name or OCID), attachment_type (default: 'paravirtualized').
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
            resolved_comp_id = await resolve_compartment_id(target_compartment_input, config)
            if not resolved_comp_id:
                return f"Error: Could not resolve compartment input '{target_compartment_input}' to a valid OCI compartment OCID."

            # Resolve Volume Name to OCID
            resolved_vol_id = await resolve_volume_ocid(volume_id, resolved_comp_id, config)
            if not resolved_vol_id:
                return f"Error: Could not resolve block volume input '{volume_id}' to a valid OCI volume OCID."

            # Resolve Instance Name to OCID
            resolved_inst_id = await resolve_vm_ocid(instance_id, resolved_comp_id, config)
            if not resolved_inst_id:
                return f"Error: Could not resolve compute instance input '{instance_id}' to a valid OCI instance OCID."

            tfvars = {
                "tenancy_ocid": config_data.get("tenancy"),
                "user_ocid": config_data.get("user"),
                "fingerprint": config_data.get("fingerprint"),
                "private_key_path": temp_pem.name,
                "region": config_data.get("region"),
                "compartment_id": resolved_comp_id,
                "volume_attachments": {
                    attachment_name: {
                        "attachment_type": attachment_type,
                        "instance_id": resolved_inst_id,
                        "volume_id": resolved_vol_id
                    }
                }
            }
            
            # Load existing state if any (for updates/idempotency)
            state_key = f"tf_state_volume_attachment_{attachment_name}"
            existing_state = await ToolStateManager.get_state(thread_id, state_key)
            
            if not existing_state:
                # Fallback: search globally
                async with AsyncSessionLocal() as session:
                    query = select(ToolStateModel).where(ToolStateModel.tool_key == state_key)
                    result = await session.execute(query)
                    tool_state = result.scalars().first()
                    if tool_state:
                        existing_state = process_state_data(tool_state.state, mode="decrypt")
            
            # Execute apply sandbox
            apply_res, output_res, final_state = await run_terraform_in_sandbox(
                tf_dir=TF_DIR,
                env=env,
                tfvars=tfvars,
                target="module.block_volume",
                state_content=existing_state
            )
            
            # Save updated state
            if final_state:
                await ToolStateManager.save_state(thread_id, state_key, final_state)
                
        finally:
            if os.path.exists(temp_pem.name):
                os.remove(temp_pem.name)

        if apply_res.returncode != 0:
            logger.error(f"Terraform volume attachment failed: {apply_res.stderr}")
            return f"Error executing Terraform Block Volume attachment: {apply_res.stderr}"
            
        if not output_res or output_res.returncode != 0:
            return "Error: Could not retrieve attachment details from Terraform."

        outputs = json.loads(output_res.stdout)
        attachment_ids = outputs.get("volume_attachment_ids", {}).get("value", {})
        attachment_ocid = attachment_ids.get(attachment_name, "Unknown")
        
        return f"Successfully attached Block Volume (ID: {volume_id}) to VM (ID: {instance_id}) with Attachment OCID: {attachment_ocid}"
    except Exception as e:
        logger.exception("Failed to attach volume")
        return f"Error attaching volume: {str(e)}"

class DestroyVolumeInput(BaseModel):
    volume_name: str = Field(description="The display name of the block volume to delete.")
    compartment_id: Optional[str] = Field(default=None, description="The name or OCID of the target compartment.")

@tool(args_schema=DestroyVolumeInput)
async def tf_destroy_volume(
    volume_name: str,
    compartment_id: Optional[str] = None,
    config: RunnableConfig = None
) -> str:
    """
    Destroys the specified Block Volume using Terraform.
    If the volume was created outside Terraform, it will automatically look up its OCID, import it, and destroy it.
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
                "volumes": {
                    volume_name: {
                        "availability_domain": "dummy_ad"
                    }
                }
            }
            
            # Load existing state
            state_key = f"tf_state_volume_{volume_name}"
            existing_state = await ToolStateManager.get_state(thread_id, state_key)
            
            if not existing_state:
                # Fallback: search globally
                async with AsyncSessionLocal() as session:
                    query = select(ToolStateModel).where(ToolStateModel.tool_key == state_key)
                    result = await session.execute(query)
                    tool_state = result.scalars().first()
                    if tool_state:
                        existing_state = process_state_data(tool_state.state, mode="decrypt")
            
            # Fallback 2: Look up OCID in OCI
            vol_ocid = None
            if not existing_state:
                logger.info(f"Volume '{volume_name}' state not found in DB. Looking up OCID in OCI...")
                
                apply_res, output_res, _ = await run_terraform_in_sandbox(
                    tf_dir=TF_DIR,
                    env=env,
                    tfvars=tfvars,
                    target="data.oci_core_volumes.all_volumes"
                )
                
                if apply_res.returncode == 0 and output_res and output_res.returncode == 0:
                    outputs = json.loads(output_res.stdout)
                    all_vols = outputs.get("all_volumes", {}).get("value", [])
                    for v in all_vols:
                        if v.get("display_name") == volume_name:
                            vol_ocid = v.get("id")
                            tfvars["volumes"][volume_name]["availability_domain"] = v.get("availability_domain", "dummy_ad")
                            break
                            
            # Case A: Tracked state
            if existing_state:
                destroy_res, _, final_state = await run_terraform_in_sandbox(
                    tf_dir=TF_DIR,
                    env=env,
                    tfvars=tfvars,
                    target="module.block_volume",
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
                    return f"Error executing Terraform volume destruction: {destroy_res.stderr}"
                return f"Successfully destroyed Block Volume '{volume_name}' using Terraform."
                
            # Case B: Untracked
            elif vol_ocid:
                logger.info(f"Found Block Volume '{volume_name}' in OCI (OCID: {vol_ocid}). Running auto-import and destroy...")
                
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
                        [tf_cmd, "import", f'module.block_volume.oci_core_volume.this["{volume_name}"]', vol_ocid],
                        cwd=temp_dir, env=env, capture_output=True, text=True
                    )
                    if import_process.returncode != 0:
                        return f"Error importing Block Volume: {import_process.stderr}"
 
                    # Destroy
                    destroy_process = await asyncio.to_thread(
                        subprocess.run,
                        [tf_cmd, "destroy", "-auto-approve", "-target", f'module.block_volume.oci_core_volume.this["{volume_name}"]'],
                        cwd=temp_dir, env=env, capture_output=True, text=True
                    )
                    if destroy_process.returncode != 0:
                        return f"Error executing Terraform volume destruction: {destroy_process.stderr}"
                        
                    return f"Successfully imported and destroyed untracked Block Volume '{volume_name}' (OCID: {vol_ocid})."
                finally:
                    try:
                        shutil.rmtree(temp_dir)
                    except Exception:
                        pass
            else:
                return f"Error: Block Volume '{volume_name}' was not found in OCI (untracked) and has no local state."
        finally:
            if os.path.exists(temp_pem.name):
                os.remove(temp_pem.name)
    except Exception as e:
        logger.exception("Failed to destroy Block Volume")
        return f"Error destroying Block Volume: {str(e)}"

class DetachVolumeInput(BaseModel):
    attachment_name: Optional[str] = Field(default=None, description="The display name of the block volume attachment to detach. Optional if volume_id or device is specified.")
    volume_id: Optional[str] = Field(default=None, description="The name or OCID of the block volume to detach. Optional if attachment_name or device is specified.")
    device: Optional[str] = Field(default=None, description="The mount device name (e.g. '/dev/sdb') to detach. Optional if attachment_name or volume_id is specified.")
    compartment_id: Optional[str] = Field(default=None, description="The name or OCID of the target compartment.")

@tool(args_schema=DetachVolumeInput)
async def tf_detach_volume(
    attachment_name: Optional[str] = None,
    volume_id: Optional[str] = None,
    device: Optional[str] = None,
    compartment_id: Optional[str] = None,
    config: RunnableConfig = None
) -> str:
    """
    Detaches a Block Volume from a VM Compute instance by destroying the volume attachment resource.
    Supports detaching by attachment_name, volume_id (display name or OCID), or device (e.g. '/dev/sdb').
    
    *** IMPORTANT INSTRUCTION FOR THE AGENT ***
    Do NOT ask the user for the attachment_name! The backend code automatically resolves the attachment name using the volume_id or device. You should call this tool immediately with just the volume_id (e.g. 'final_vol') and compartment_id!
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

            # Fetch all attachments to perform lookup mapping if needed
            matched_att = None
            logger.info("Querying volume attachments for resolution...")
            
            # Simple dummy tfvars for query
            query_tfvars = {
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
                tfvars=query_tfvars,
                target="data.oci_core_volume_attachments.all_volume_attachments"
            )
            
            all_attachments = []
            if apply_res.returncode == 0 and output_res and output_res.returncode == 0:
                outputs = json.loads(output_res.stdout)
                all_attachments = outputs.get("all_volume_attachments", {}).get("value", [])

            # Match criteria
            if device:
                search_device = device.strip().lower()
                for a in all_attachments:
                    att_dev = a.get("device", "").strip().lower()
                    if att_dev == search_device or att_dev.replace("p", "") == search_device.replace("p", ""):
                        matched_att = a
                        break
            elif volume_id:
                # Query volumes to resolve display name/OCID to exact volume OCID
                vol_ocid = volume_id
                if not volume_id.startswith("ocid1.volume.oc1."):
                    vols = await run_tf_query(config, "data.oci_core_volumes.all_volumes", {"compartment_id": resolved_comp_id})
                    if vols:
                        search_norm = normalize_name(volume_id)
                        for v in vols:
                            if normalize_name(v.get("display_name", "")) == search_norm or volume_id == v.get("id"):
                                vol_ocid = v.get("id")
                                break
                for a in all_attachments:
                    if a.get("volume_id") == vol_ocid:
                        matched_att = a
                        break
            elif attachment_name:
                for a in all_attachments:
                    if a.get("display_name") == attachment_name:
                        matched_att = a
                        break

            # Resolve finalized names
            final_attachment_name = attachment_name
            attachment_ocid = None
            instance_id = "dummy_instance_id"
            volume_id_val = "dummy_volume_id"
            attachment_type = "paravirtualized"

            if matched_att:
                final_attachment_name = matched_att.get("display_name") or f"attachment_{matched_att.get('id')[-8:]}"
                attachment_ocid = matched_att.get("id")
                instance_id = matched_att.get("instance_id")
                volume_id_val = matched_att.get("volume_id")
                attachment_type = matched_att.get("attachment_type", "paravirtualized")
            elif not final_attachment_name:
                return "Error: Could not identify volume attachment. Please provide 'attachment_name', 'volume_id', or 'device'."

            # Construct tfvars to import/destroy
            tfvars = {
                "tenancy_ocid": config_data.get("tenancy"),
                "user_ocid": config_data.get("user"),
                "fingerprint": config_data.get("fingerprint"),
                "private_key_path": temp_pem.name,
                "region": config_data.get("region"),
                "compartment_id": resolved_comp_id,
                "volume_attachments": {
                    final_attachment_name: {
                        "attachment_type": attachment_type,
                        "instance_id": instance_id,
                        "volume_id": volume_id_val
                    }
                }
            }
            
            # Load existing state
            state_key = f"tf_state_volume_attachment_{final_attachment_name}"
            existing_state = await ToolStateManager.get_state(thread_id, state_key)
            
            # Case A: Tracked state
            if existing_state:
                destroy_res, _, final_state = await run_terraform_in_sandbox(
                    tf_dir=TF_DIR,
                    env=env,
                    tfvars=tfvars,
                    target="module.block_volume",
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
                    return f"Error executing Terraform volume detachment: {destroy_res.stderr}"
                return f"Successfully detached Block Volume Attachment '{final_attachment_name}'."
                
            # Case B: Untracked
            elif attachment_ocid:
                logger.info(f"Found Volume Attachment '{final_attachment_name}' in OCI (OCID: {attachment_ocid}). Running auto-import and destroy...")
                
                # Use project space-free path for temporary directory
                temp_dir = tempfile.mkdtemp(dir=os.path.dirname(TF_DIR))
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
                    
                    # Copy private key locally to avoid path spaces/drive letters on Windows
                    local_key_path = os.path.join(temp_dir, "private_key.pem")
                    shutil.copy(temp_pem.name, local_key_path)
                    
                    tfvars_local = tfvars.copy()
                    tfvars_local["private_key_path"] = os.path.abspath(local_key_path).replace("\\", "/")
                    
                    with open(os.path.join(temp_dir, "terraform.tfvars.json"), "w") as f:
                        json.dump(tfvars_local, f)
 
                    import sys
                    python_dir = os.path.dirname(sys.executable)
                    tf_cmd = os.path.join(python_dir, "terraform.exe")
                    if not os.path.exists(tf_cmd):
                        tf_cmd = "terraform"
 
                    await asyncio.to_thread(subprocess.run, [tf_cmd, "init"], cwd=temp_dir, env=env, capture_output=True)
                    
                    # Import
                    import_process = await asyncio.to_thread(
                        subprocess.run,
                        [tf_cmd, "import", f'module.block_volume.oci_core_volume_attachment.this["{final_attachment_name}"]', attachment_ocid],
                        cwd=temp_dir, env=env, capture_output=True, text=True
                    )
                    if import_process.returncode != 0:
                        return f"Error importing Block Volume attachment: {import_process.stderr}"
 
                    # Destroy
                    destroy_process = await asyncio.to_thread(
                        subprocess.run,
                        [tf_cmd, "destroy", "-auto-approve", "-target", f'module.block_volume.oci_core_volume_attachment.this["{final_attachment_name}"]'],
                        cwd=temp_dir, env=env, capture_output=True, text=True
                    )
                    if destroy_process.returncode != 0:
                        return f"Error executing Terraform Block Volume attachment destruction: {destroy_process.stderr}"
                        
                    return f"Successfully imported and destroyed untracked Block Volume Attachment '{final_attachment_name}' (OCID: {attachment_ocid})."
                finally:
                    try:
                        shutil.rmtree(temp_dir)
                    except Exception:
                        pass
            else:
                return f"Error: Block Volume Attachment '{final_attachment_name}' was not found in OCI (untracked) and has no local state."
        finally:
            if os.path.exists(temp_pem.name):
                os.remove(temp_pem.name)
    except Exception as e:
        logger.exception("Failed to detach volume")
        return f"Error detaching Block Volume: {str(e)}"

class ListVolumesInput(BaseModel):
    compartment_id: str = Field(description="The name or OCID of the compartment to list volumes from.")
    detail: bool = Field(default=False, description="Set to True to retrieve detailed information including attachment mapping to VM Compute instances.")
    volume_name: Optional[str] = Field(default=None, description="Optional display name or OCID of a specific block volume to filter.")


@tool(args_schema=ListVolumesInput)
async def tf_list_volumes(
    compartment_id: str,
    detail: bool = False,
    volume_name: Optional[str] = None,
    config: RunnableConfig = None
) -> str:
    """
    Lists all Block Volumes inside the specified OCI compartment.
    Optionally maps block volumes to the VM Compute instances they are attached to, showing sizes and states.
    CRITICAL: You must ask the user which OCI compartment they want to list volumes in before running this tool.
    """
    try:
        resolved_comp_id = await resolve_compartment_id(compartment_id, config)
        if not resolved_comp_id:
            return f"Error: Could not resolve compartment input '{compartment_id}' to a valid OCI compartment OCID."

        vols = await run_tf_query(config, "data.oci_core_volumes.all_volumes", {"compartment_id": resolved_comp_id})
        if not vols:
            return f"No Block Volumes found in compartment: {compartment_id}"

        if volume_name:
            search_norm = normalize_name(volume_name)
            vols = [v for v in vols if search_norm in normalize_name(v.get("display_name", "")) or volume_name == v.get("id")]
            if not vols:
                return f"No Block Volume matching '{volume_name}' was found in compartment: {compartment_id}"

        attachments_map = {}
        vms_map = {}
        if detail:
            # Query all volume attachments in the compartment
            attachments = await run_tf_query(config, "data.oci_core_volume_attachments.all_volume_attachments", {"compartment_id": resolved_comp_id})
            if attachments:
                for att in attachments:
                    vol_id = att.get("volume_id")
                    if vol_id:
                        attachments_map[vol_id] = att

            # Query all compute instances in the compartment
            vms = await run_tf_query(config, "data.oci_core_instances.all_instances", {"compartment_id": resolved_comp_id})
            if vms:
                for vm in vms:
                    vm_id = vm.get("id")
                    if vm_id:
                        vms_map[vm_id] = vm.get("display_name", "N/A")

        result_lines = [f"Found {len(vols)} Block Volume(s) in compartment (via Terraform):\n"]
        for v in vols:
            name = v.get("display_name", "N/A")
            ocid = v.get("id", "N/A")
            size = v.get("size_in_gbs", "N/A")
            state = v.get("lifecycle_state", v.get("state", "N/A"))
            ad = v.get("availability_domain", "N/A")
            
            if detail:
                attachment_info = "Not Attached"
                device_path = ""
                att = attachments_map.get(ocid)
                if att:
                    vm_id = att.get("instance_id")
                    vm_name = vms_map.get(vm_id, vm_id)
                    att_state = att.get("lifecycle_state", att.get("state", "ATTACHED"))
                    attachment_info = f"Attached to VM: **{vm_name}** (State: `{att_state}`)"
                    if att.get("device"):
                        device_path = f" | Device: `{att.get('device')}`"
                
                result_lines.append(
                    f"- **{name}** (State: `{state}`)\n"
                    f"  Size: `{size} GB` | AD: `{ad}`\n"
                    f"  Attachment Status: {attachment_info}{device_path}\n"
                    f"  OCID: `{ocid}`"
                )
            else:
                result_lines.append(
                    f"- **{name}** (State: `{state}`)\n"
                    f"  Size: `{size} GB` | AD: `{ad}`\n"
                    f"  OCID: `{ocid}`"
                )
            
        return "\n".join(result_lines)
    except Exception as e:
        logger.exception("Failed to list Block Volumes")
        return f"Error listing Block Volumes: {str(e)}"


class ListVolumeAttachmentsInput(BaseModel):
    compartment_id: str = Field(description="The name or OCID of the compartment to list volume attachments from.")
    instance_id: Optional[str] = Field(default=None, description="Optional display name or OCID of a VM instance to filter attachments.")


@tool(args_schema=ListVolumeAttachmentsInput)
async def tf_list_volume_attachments(
    compartment_id: str,
    instance_id: Optional[str] = None,
    config: RunnableConfig = None
) -> str:
    """
    Lists all Block Volume attachments inside the specified OCI compartment, optionally filtered by a VM Compute Instance ID.
    """
    try:
        resolved_comp_id = await resolve_compartment_id(compartment_id, config)
        if not resolved_comp_id:
            return f"Error: Could not resolve compartment input '{compartment_id}' to a valid OCI compartment OCID."

        resolved_inst_id = None
        if instance_id:
            resolved_inst_id = await resolve_vm_ocid(instance_id, resolved_comp_id, config)
            if not resolved_inst_id:
                return f"Error: Could not resolve VM Instance name/ID '{instance_id}'."

        filter_vars = {"compartment_id": resolved_comp_id}
        if resolved_inst_id:
            filter_vars["volume_filter_instance_id"] = resolved_inst_id

        attachments = await run_tf_query(config, "data.oci_core_volume_attachments.all_volume_attachments", filter_vars)
        if not attachments:
            return f"No Block Volume attachments found in compartment: {compartment_id}"

        result_lines = [f"Found {len(attachments)} Volume Attachment(s) in compartment (via Terraform):\n"]
        for a in attachments:
            name = a.get("display_name", "N/A")
            ocid = a.get("id", "N/A")
            inst_id = a.get("instance_id", "N/A")
            vol_id = a.get("volume_id", "N/A")
            state = a.get("lifecycle_state", a.get("state", "N/A"))
            conn_type = a.get("attachment_type", "N/A")
            
            result_lines.append(
                f"- **{name}** (State: `{state}`)\n"
                f"  Connection Type: `{conn_type}`\n"
                f"  Compute Instance OCID: `{inst_id}`\n"
                f"  Block Volume OCID: `{vol_id}`\n"
                f"  OCID: `{ocid}`"
            )
            
        return "\n".join(result_lines)
    except Exception as e:
        logger.exception("Failed to list Block Volume attachments")
        return f"Error listing Block Volume attachments: {str(e)}"

class BackupBlockVolumeInput(BaseModel):
    backup_name: str = Field(description="The display name for the block volume backup.")
    volume_id: str = Field(description="The display name of the block volume or the block volume OCID.")
    type: str = Field(default="INCREMENTAL", description="Backup type: 'FULL' or 'INCREMENTAL'.")
    compartment_id: Optional[str] = Field(default=None, description="The name or OCID of the target compartment.")

@tool(args_schema=BackupBlockVolumeInput)
async def tf_backup_block_volume(
    backup_name: str,
    volume_id: str,
    type: str = "INCREMENTAL",
    compartment_id: Optional[str] = None,
    config: RunnableConfig = None
) -> str:
    """
    Creates a backup of an existing Block Volume in a compartment using Terraform.
    Inputs: backup_name, volume_id (name or OCID), type ('FULL' or 'INCREMENTAL', default: 'INCREMENTAL').
    """
    try:
        if type.upper() not in ["FULL", "INCREMENTAL"]:
            return f"Error: Invalid backup type '{type}'. Supported types are 'FULL' and 'INCREMENTAL'."

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
            resolved_comp_id = await resolve_compartment_id(target_compartment_input, config)
            if not resolved_comp_id:
                return f"Error: Could not resolve compartment input '{target_compartment_input}' to a valid OCI compartment OCID."

            # Resolve Volume Name to OCID
            resolved_vol_id = await resolve_volume_ocid(volume_id, resolved_comp_id, config)
            if not resolved_vol_id:
                return f"Error: Could not resolve block volume input '{volume_id}' to a valid OCI volume OCID."

            tfvars = {
                "tenancy_ocid": config_data.get("tenancy"),
                "user_ocid": config_data.get("user"),
                "fingerprint": config_data.get("fingerprint"),
                "private_key_path": temp_pem.name,
                "region": config_data.get("region"),
                "compartment_id": resolved_comp_id,
                "volume_backups": {
                    backup_name: {
                        "volume_id": resolved_vol_id,
                        "type": type.upper()
                    }
                }
            }

            state_key = f"tf_state_volume_backup_{backup_name}"
            existing_state = await ToolStateManager.get_state(thread_id, state_key)
            if not existing_state:
                async with AsyncSessionLocal() as session:
                    query = select(ToolStateModel).where(ToolStateModel.tool_key == state_key)
                    result = await session.execute(query)
                    tool_state = result.scalars().first()
                    if tool_state:
                        existing_state = process_state_data(tool_state.state, mode="decrypt")

            apply_res, output_res, final_state = await run_terraform_in_sandbox(
                tf_dir=TF_DIR,
                env=env,
                tfvars=tfvars,
                target="module.block_volume",
                state_content=existing_state
            )

            if final_state:
                await ToolStateManager.save_state(thread_id, state_key, final_state)

        finally:
            if os.path.exists(temp_pem.name):
                os.remove(temp_pem.name)

        if apply_res.returncode != 0:
            logger.error(f"Terraform volume backup failed: {apply_res.stderr}")
            return f"Error executing Terraform Block Volume backup: {apply_res.stderr}"

        if not output_res or output_res.returncode != 0:
            return "Error: Could not retrieve backup details from Terraform."

        outputs = json.loads(output_res.stdout)
        backup_ids = outputs.get("volume_backup_ids", {}).get("value", {})
        backup_ocid = backup_ids.get(backup_name, "Unknown")

        return f"Successfully created Block Volume Backup '{backup_name}' (OCID: {backup_ocid}, Type: {type.upper()}) for Volume: {resolved_vol_id}"
    except Exception as e:
        logger.exception("Failed to create volume backup")
        return f"Error creating volume backup: {str(e)}"

# Registering Block Volume Tools
registry.register(
    tf_backup_block_volume,
    ToolMeta(
        name="tf_backup_block_volume",
        display_name="OCI Terraform: Create Block Volume Backup",
        description="Creates a backup of an existing Block Volume using Terraform.",
        category="Infrastructure",
        package="oci_terraform"
    )
)

registry.register(
    tf_provision_volume,
    ToolMeta(
        name="tf_provision_volume",
        display_name="OCI Terraform: Create Block Volume",
        description="Creates a new Block Volume in a compartment using Terraform.",
        category="Infrastructure",
        package="oci_terraform"
    )
)

registry.register(
    tf_resize_block_volume,
    ToolMeta(
        name="tf_resize_block_volume",
        display_name="OCI Terraform: Resize Block Volume In-Place",
        description="Resizes/extends an existing block volume in-place to a larger size using Terraform.",
        category="Infrastructure",
        package="oci_terraform"
    )
)

registry.register(
    tf_attach_volume,
    ToolMeta(
        name="tf_attach_volume",
        display_name="OCI Terraform: Attach Block Volume",
        description="Attaches a Block Volume to a VM Compute instance using Terraform.",
        category="Infrastructure",
        package="oci_terraform"
    )
)

registry.register(
    tf_destroy_volume,
    ToolMeta(
        name="tf_destroy_volume",
        display_name="OCI Terraform: Delete Block Volume",
        description="Deletes/destroys an existing Block Volume using Terraform. Automatically imports and destroys untracked volumes by name.",
        category="Infrastructure",
        package="oci_terraform"
    )
)

registry.register(
    tf_detach_volume,
    ToolMeta(
        name="tf_detach_volume",
        display_name="OCI Terraform: Detach Block Volume",
        description="Detaches a Block Volume from a VM instance by deleting its volume attachment.",
        category="Infrastructure",
        package="oci_terraform"
    )
)

registry.register(
    tf_list_volumes,
    ToolMeta(
        name="tf_list_volumes",
        display_name="OCI Terraform: List Block Volumes",
        description="Lists all Block Volumes in a compartment.",
        category="Infrastructure",
        package="oci_terraform"
    )
)

registry.register(
    tf_list_volume_attachments,
    ToolMeta(
        name="tf_list_volume_attachments",
        display_name="OCI Terraform: List Volume Attachments",
        description="Lists all Block Volume attachments in a compartment, optionally filtered by a VM Instance ID.",
        category="Infrastructure",
        package="oci_terraform"
    )
)


# --- OS-Level Block Volume LVM tools ---

@tool
async def tf_setup_block_volume_os(
    device_name: str = "/dev/sdb",
    vg_name: str = "vg_data",
    lv_name: str = "lv_data",
    mount_path: str = "/data",
    target_vm_ip: Optional[str] = None,
    target_os_user: Optional[str] = None,
    ssh_private_key: Optional[str] = None,
    vm_name: Optional[str] = None,
    instance_id: Optional[str] = None,
    config: RunnableConfig = None,
    **kwargs
) -> str:
    """
    OS-Level: Formats a newly attached block volume, configures LVM (PV, VG, LV), mounts it, and persists in /etc/fstab.
    
    *** IMPORTANT: Always ask the user to confirm the target device (e.g. '/dev/sdb'), volume group (VG) name, logical volume (LV) name, and mount path BEFORE executing this tool. ***
    ALWAYS ask the user for the VM public IP address (target_vm_ip) and SSH username (target_os_user, usually 'opc') before calling this tool.
    Do NOT guess or auto-resolve the IP or username — they MUST come from the user in the conversation.
    The SSH private key is automatically resolved from the backend settings. Do NOT ask the user for the private key.
    """
    try:
        logger.info(f"tf_setup_block_volume_os entry: device_name={device_name}, vg_name={vg_name}, lv_name={lv_name}, mount_path={mount_path}, target_vm_ip={target_vm_ip}, target_os_user={target_os_user}")
        logger.info(f"Triggering OS LVM setup for {device_name} (VG: {vg_name}, LV: {lv_name}) mounted to {mount_path}")
        
        # Remove any forwarded connection parameters from kwargs to prevent overriding in tf_ssh_executor
        if isinstance(kwargs, dict):
            kwargs.pop("ip_address", None)
            kwargs.pop("username", None)
            kwargs.pop("vm_public_ip", None)
            kwargs.pop("os_username", None)
            kwargs.pop("ssh_private_key", None)
            kwargs.pop("private_key", None)
            
        current_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.abspath(os.path.join(current_dir, "..", "..", "scripts", "configure_lvm.sh"))
        
        if not os.path.exists(script_path):
            return f"Error: Helper shell script configure_lvm.sh was not found at {script_path}."
            
        with open(script_path, "r") as f:
            script_content = f.read()
            
        escaped_script = script_content.replace('\\', '\\\\').replace('"', '\\"').replace('$', '\\$')
        cmd_args = f"'{device_name}' '{vg_name}' '{lv_name}' '{mount_path}'"
        full_command = f"bash -c \"{escaped_script}\" -- {cmd_args}"
        
        ssh_func = getattr(tf_ssh_executor, "coroutine", None) or getattr(tf_ssh_executor, "func", None) or tf_ssh_executor
        ssh_result = await ssh_func(
            command=full_command,
            vm_public_ip=target_vm_ip,
            os_username=target_os_user,
            ssh_private_key=ssh_private_key,
            vm_name=vm_name,
            instance_id=instance_id,
            config=config,
            **kwargs
        )
        
        return f"--- Block Volume OS Setup Hook ---\nDevice: {device_name}\nMount Path: {mount_path}\n\nExecution Log:\n{ssh_result}"
    except Exception as e:
        logger.exception("Failed to run OS setup for block volume")
        return f"Error during OS setup: {str(e)}"


@tool
async def tf_resize_fs_lvm(
    device_name: str = "/dev/sdb",
    vg_name: str = "vg_data",
    lv_name: str = "lv_data",
    target_vm_ip: Optional[str] = None,
    target_os_user: Optional[str] = None,
    ssh_private_key: Optional[str] = None,
    vm_name: Optional[str] = None,
    instance_id: Optional[str] = None,
    config: RunnableConfig = None,
    **kwargs
) -> str:
    """
    OS-Level: Online resizes Partition, Physical Volume, Logical Volume, and filesystem after block volume expansion.
    Supports both raw disks (e.g. '/dev/sdb') and partitioned disks (e.g. '/dev/sdb1').
    If a partitioned disk is passed (e.g. '/dev/sdb1'), it automatically resizes the partition using growpart.
    
    *** IMPORTANT: Always ask the user to confirm the target device/partition (e.g. '/dev/sdb1'), volume group (VG) name, logical volume (LV) name, and mount path BEFORE executing this tool. ***
    ALWAYS ask the user for the VM public IP address (target_vm_ip) and SSH username (target_os_user, usually 'opc') before calling this tool.
    Do NOT guess or auto-resolve the IP or username — they MUST come from the user in the conversation.
    The SSH private key is automatically resolved from the backend settings. Do NOT ask the user for the private key.
    """
    try:
        logger.info(f"tf_resize_fs_lvm entry: device_name={device_name}, vg_name={vg_name}, lv_name={lv_name}, target_vm_ip={target_vm_ip}, target_os_user={target_os_user}")
        logger.info(f"Triggering OS LVM extension for {device_name} (VG: {vg_name}, LV: {lv_name})")
        
        # Remove any forwarded connection parameters from kwargs to prevent overriding in tf_ssh_executor
        if isinstance(kwargs, dict):
            kwargs.pop("ip_address", None)
            kwargs.pop("username", None)
            kwargs.pop("vm_public_ip", None)
            kwargs.pop("os_username", None)
            kwargs.pop("ssh_private_key", None)
            kwargs.pop("private_key", None)
            
        current_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.abspath(os.path.join(current_dir, "..", "..", "scripts", "extend_lvm.sh"))
        
        if not os.path.exists(script_path):
            return f"Error: Helper shell script extend_lvm.sh was not found at {script_path}."
            
        with open(script_path, "r") as f:
            script_content = f.read()
            
        escaped_script = script_content.replace('\\', '\\\\').replace('"', '\\"').replace('$', '\\$')
        cmd_args = f"'{device_name}' '{vg_name}' '{lv_name}'"
        full_command = f"bash -c \"{escaped_script}\" -- {cmd_args}"
        
        ssh_func = getattr(tf_ssh_executor, "coroutine", None) or getattr(tf_ssh_executor, "func", None) or tf_ssh_executor
        ssh_result = await ssh_func(
            command=full_command,
            vm_public_ip=target_vm_ip,
            os_username=target_os_user,
            ssh_private_key=ssh_private_key,
            vm_name=vm_name,
            instance_id=instance_id,
            config=config,
            **kwargs
        )
        
        return f"--- Storage Resize Hook ---\nDevice: {device_name}\n\nExecution Log:\n{ssh_result}"
    except Exception as e:
        logger.exception("Failed to run OS storage resize")
        return f"Error during OS resize: {str(e)}"


@tool
async def tf_delete_lvm_os(
    device_name: str = "/dev/sdb",
    vg_name: str = "vg_data",
    lv_name: str = "lv_data",
    mount_path: str = "/data",
    target_vm_ip: Optional[str] = None,
    target_os_user: Optional[str] = None,
    ssh_private_key: Optional[str] = None,
    vm_name: Optional[str] = None,
    instance_id: Optional[str] = None,
    config: RunnableConfig = None,
    **kwargs
) -> str:
    """
    OS-Level: Safely unmounts filesystems, cleans /etc/fstab, removes LV, VG, PV, and wipes the partitions.
    
    ALWAYS ask the user for the VM public IP address (target_vm_ip) and SSH username (target_os_user, usually 'opc') before calling this tool.
    Do NOT guess or auto-resolve the IP or username — they MUST come from the user in the conversation.
    The SSH private key is automatically resolved from the backend settings. Do NOT ask the user for the private key.
    """
    try:
        logger.info(f"tf_delete_lvm_os entry: device_name={device_name}, vg_name={vg_name}, lv_name={lv_name}, mount_path={mount_path}, target_vm_ip={target_vm_ip}, target_os_user={target_os_user}")
        logger.info(f"Triggering OS LVM deletion for {device_name} (VG: {vg_name}, LV: {lv_name})")
        
        # Remove any forwarded connection parameters from kwargs to prevent overriding in tf_ssh_executor
        if isinstance(kwargs, dict):
            kwargs.pop("ip_address", None)
            kwargs.pop("username", None)
            kwargs.pop("vm_public_ip", None)
            kwargs.pop("os_username", None)
            kwargs.pop("ssh_private_key", None)
            kwargs.pop("private_key", None)
            
        current_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.abspath(os.path.join(current_dir, "..", "..", "scripts", "delete_lvm.sh"))
        
        if not os.path.exists(script_path):
            return f"Error: Helper shell script delete_lvm.sh was not found at {script_path}."
            
        with open(script_path, "r") as f:
            script_content = f.read()
            
        escaped_script = script_content.replace('\\', '\\\\').replace('"', '\\"').replace('$', '\\$')
        cmd_args = f"'{device_name}' '{vg_name}' '{lv_name}' '{mount_path}'"
        full_command = f"bash -c \"{escaped_script}\" -- {cmd_args}"
        
        ssh_func = getattr(tf_ssh_executor, "coroutine", None) or getattr(tf_ssh_executor, "func", None) or tf_ssh_executor
        ssh_result = await ssh_func(
            command=full_command,
            vm_public_ip=target_vm_ip,
            os_username=target_os_user,
            ssh_private_key=ssh_private_key,
            vm_name=vm_name,
            instance_id=instance_id,
            config=config,
            **kwargs
        )
        return f"--- LVM OS Delete Hook ---\nDevice: {device_name}\nMount Path: {mount_path}\n\nExecution Log:\n{ssh_result}"
    except Exception as e:
        logger.exception("Failed to delete LVM on VM")
        return f"Error during LVM OS delete: {str(e)}"


# --- Disk Usage Tool ---

@tool
async def tf_check_disk_usage(
    target_vm_ip: Optional[str] = None,
    target_os_user: Optional[str] = None,
    ssh_private_key: Optional[str] = None,
    vm_name: Optional[str] = None,
    instance_id: Optional[str] = None,
    config: RunnableConfig = None,
    **kwargs
) -> str:
    """
    OS-Level: Checks the disk usage and block device layout on the target VM.
    Executes 'df -h' and 'lsblk' to show mounted filesystems and available block devices.
    
    ALWAYS ask the user for the VM public IP address (target_vm_ip) and SSH username (target_os_user, usually 'opc') before calling this tool.
    Do NOT guess or auto-resolve the IP or username — they MUST come from the user in the conversation.
    The SSH private key is automatically resolved from the backend settings. Do NOT ask the user for the private key.
    """
    try:
        logger.info(f"tf_check_disk_usage entry: target_vm_ip={target_vm_ip}, target_os_user={target_os_user}")
        
        # Remove any forwarded connection parameters from kwargs to prevent overriding in tf_ssh_executor
        if isinstance(kwargs, dict):
            kwargs.pop("ip_address", None)
            kwargs.pop("username", None)
            kwargs.pop("vm_public_ip", None)
            kwargs.pop("os_username", None)
            kwargs.pop("ssh_private_key", None)
            kwargs.pop("private_key", None)
            
        full_command = "echo '=== Block Devices (lsblk) ==='; lsblk; echo ''; echo '=== Disk Usage (df -h) ==='; df -h"
        
        ssh_func = getattr(tf_ssh_executor, "coroutine", None) or getattr(tf_ssh_executor, "func", None) or tf_ssh_executor
        ssh_result = await ssh_func(
            command=full_command,
            vm_public_ip=target_vm_ip,
            os_username=target_os_user,
            ssh_private_key=ssh_private_key,
            vm_name=vm_name,
            instance_id=instance_id,
            config=config,
            **kwargs
        )
        return f"--- Disk Usage Report ---\n\n{ssh_result}"
    except Exception as e:
        logger.exception("Failed to check disk usage on VM")
        return f"Error checking disk usage: {str(e)}"

# --- Tool Registrations ---

def register_volume_os_tools():
    # 1. tf_setup_block_volume_os
    registry.register(
        tf_setup_block_volume_os,
        ToolMeta(
            name="tf_setup_block_volume_os",
            display_name="OCI Terraform: OS Block Volume LVM Setup",
            description="OS-Level LVM partitioning, formatting, mounting, and fstab config for newly attached block volumes.",
            category="Infrastructure",
            package="oci_terraform",
            version="1.0.0"
        )
    )

    # 2. tf_resize_fs_lvm
    registry.register(
        tf_resize_fs_lvm,
        ToolMeta(
            name="tf_resize_fs_lvm",
            display_name="OCI Terraform: OS Expand Volume Filesystem",
            description="OS-Level online resizing of LVM partitions and filesystems. Automatically handles partition growth via growpart for partition paths (e.g. /dev/sdb1).",
            category="Infrastructure",
            package="oci_terraform",
            version="1.0.0"
        )
    )

    # 3. tf_delete_lvm_os
    registry.register(
        tf_delete_lvm_os,
        ToolMeta(
            name="tf_delete_lvm_os",
            display_name="OCI Terraform: Delete LVM configuration from VM",
            description="Safely unmounts filesystems, cleans /etc/fstab, removes LV, VG, PV, and wipes the partitions.",
            category="Infrastructure",
            package="oci_terraform",
            version="1.0.0"
        )
    )

    # 4. tf_check_disk_usage
    registry.register(
        tf_check_disk_usage,
        ToolMeta(
            name="tf_check_disk_usage",
            display_name="OCI Terraform: Check VM Disk Usage",
            description="OS-Level check of block devices and filesystem disk usage using lsblk and df -h.",
            category="Infrastructure",
            package="oci_terraform",
            version="1.0.0"
        )
    )

register_volume_os_tools()
