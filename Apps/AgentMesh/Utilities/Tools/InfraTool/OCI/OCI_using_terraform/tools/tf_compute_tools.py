import os
import json
import logging
import asyncio
import tempfile
import shutil
import subprocess
import paramiko
import io
import secrets
import string
from typing import Optional, List, Dict, Any, Union
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
    resolve_subnet_id,
    resolve_vm_ocid,
    normalize_name
)

logger = logging.getLogger(__name__)

# Absolute path to the root Terraform OCI directory
TF_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

class ListImagesInput(BaseModel):
    compartment_id: str = Field(description="The name or OCID of the compartment to list images from.")
    operating_system: Optional[str] = Field(default=None, description="Optional OS filter (e.g. 'Oracle Linux', 'Ubuntu').")

@tool(args_schema=ListImagesInput)
async def tf_list_images(
    compartment_id: str,
    operating_system: Optional[str] = None,
    config: RunnableConfig = None
) -> str:
    """
    Lists available OCI images (OS images) in the specified compartment.
    Optionally filters by operating_system (e.g. 'Oracle Linux', 'Ubuntu').
    CRITICAL: You must ask the user which OCI compartment they want to list/manage resources in, and ensure it is configured.
    """
    try:
        resolved_comp_id = await resolve_compartment_id(compartment_id, config)
        if not resolved_comp_id:
            return f"Error: Could not resolve compartment input '{compartment_id}' to a valid OCI compartment OCID."

        filter_vars = {"compartment_id": resolved_comp_id}
        if operating_system:
            filter_vars["image_filter_operating_system"] = operating_system

        images = await run_tf_query(config, "data.oci_core_images.all_images", filter_vars)
        if not images:
            return f"No OCI images found in compartment: {compartment_id}"

        result_lines = [f"Found {len(images)} OS Image(s) in compartment (via Terraform):\n"]
        for img in images:
            name = img.get("display_name", "N/A")
            ocid = img.get("id", "N/A")
            os_name = img.get("operating_system", "N/A")
            os_ver = img.get("operating_system_version", "N/A")
            result_lines.append(
                f"- **{name}**\n"
                f"  OS: `{os_name}` (Version: `{os_ver}`)\n"
                f"  OCID: `{ocid}`"
            )
        return "\n".join(result_lines)
    except Exception as e:
        logger.exception("Failed to list OCI images")
        return f"Error listing OCI images: {str(e)}"


class ListShapesInput(BaseModel):
    compartment_id: str = Field(description="The name or OCID of the compartment to list shapes from.")
    image_id: Optional[str] = Field(default=None, description="Optional image OCID to filter compatible shapes.")


@tool(args_schema=ListShapesInput)
async def tf_list_shapes(
    compartment_id: str,
    image_id: Optional[str] = None,
    config: RunnableConfig = None
) -> str:
    """
    Lists available compute shapes in the specified compartment, optionally filtered by image OCID.
    CRITICAL: You must ask the user which OCI compartment they want to list/manage resources in, and ensure it is configured.
    """
    try:
        resolved_comp_id = await resolve_compartment_id(compartment_id, config)
        if not resolved_comp_id:
            return f"Error: Could not resolve compartment input '{compartment_id}' to a valid OCI compartment OCID."

        filter_vars = {"compartment_id": resolved_comp_id}
        if image_id:
            filter_vars["shape_filter_image_id"] = image_id

        shapes = await run_tf_query(config, "data.oci_core_shapes.all_shapes", filter_vars)
        if not shapes:
            return f"No compute shapes found in compartment: {compartment_id}"

        result_lines = [f"Found {len(shapes)} Shape(s) in compartment (via Terraform):\n"]
        for s in shapes:
            name = s.get("name", s.get("shape", "N/A"))
            ocpus = s.get("ocpus", "N/A")
            mem = s.get("memory_in_gbs", "N/A")
            proc = s.get("processor_description", "N/A")
            
            result_lines.append(
                f"- **{name}**\n"
                f"  Specs: `{ocpus}` OCPUs, `{mem}` GB Memory\n"
                f"  Processor: {proc}"
            )

        return "\n".join(result_lines)
    except Exception as e:
        logger.exception("Failed to list OCI shapes")
        return f"Error listing OCI shapes: {str(e)}"



async def resolve_availability_domain(ad_input: Optional[str], tf_dir: str, env: dict, tfvars: dict) -> Optional[str]:
    if ad_input and ad_input.strip():
        val = ad_input.strip()
        if ":" in val:
            return val
            
    # Query ADs
    apply_res, output_res, _ = await run_terraform_in_sandbox(
        tf_dir=tf_dir,
        env=env,
        tfvars=tfvars,
        target="data.oci_identity_availability_domains.all_ads"
    )
    if apply_res.returncode == 0 and output_res and output_res.returncode == 0:
        try:
            outputs = json.loads(output_res.stdout)
            ads = outputs.get("all_ads", {}).get("value", [])
            if ads:
                if ad_input and ad_input.strip():
                    ad_search = ad_input.strip().lower()
                    for ad in ads:
                        ad_name = ad.get("name", "")
                        if ad_search in ad_name.lower():
                            return ad_name
                return ads[0].get("name")
        except Exception:
            pass
    return ad_input


class ProvisionVMInput(BaseModel):
    vm_name: str = Field(
        ...,
        description="The display name of the compute instance VM to create."
    )
    shape: str = Field(
        ...,
        description="The compute shape/size to use (e.g., VM.Standard.E3.Flex, VM.Standard2.1)."
    )
    subnet_id: str = Field(
        ...,
        description="The name or OCID of the target subnet."
    )
    image_id: str = Field(
        ...,
        description="The OCID of the OS image to use."
    )
    availability_domain: Optional[str] = Field(
        default=None,
        description="Optional availability domain (e.g., AD-1, ap-tokyo-1-ad-1). If omitted, defaults to the first AD."
    )
    assign_public_ip: bool = Field(
        default=True,
        description="Whether to assign a public IP address to the VM."
    )
    nsg_ids: Optional[List[str]] = Field(
        default=None,
        description="Optional list of Network Security Group OCIDs."
    )
    ssh_authorized_key: Optional[str] = Field(
        default=None,
        description="Optional SSH public key content to authorize on the VM."
    )
    ocpus: Optional[float] = Field(
        default=None,
        description="Number of OCPUs for flexible shapes."
    )
    memory_in_gbs: Optional[float] = Field(
        default=None,
        description="Memory size in GB for flexible shapes."
    )
    compartment_id: Optional[str] = Field(
        default=None,
        description="The name or OCID of the compartment where the VM should reside."
    )
    ssh_public_key: Optional[str] = Field(
        default=None,
        description="Optional SSH public key configured in the UI."
    )


@tool(args_schema=ProvisionVMInput)
async def tf_provision_vm(
    vm_name: str,
    shape: str,
    subnet_id: str,
    image_id: str,
    availability_domain: Optional[str] = None,
    assign_public_ip: bool = True,
    nsg_ids: Optional[List[str]] = None,
    ssh_authorized_key: Optional[str] = None,
    ocpus: Optional[float] = None,
    memory_in_gbs: Optional[float] = None,
    compartment_id: Optional[str] = None,
    config: RunnableConfig = None,
    ssh_public_key: Optional[str] = None,
    **kwargs
) -> str:
    """
    Provisions a new Virtual Machine (VM Compute Instance) in OCI using Terraform.
    Inputs: vm_name, shape, subnet_id (name or OCID), image_id, availability_domain (optional, default first AD), assign_public_ip (optional), nsg_ids (optional), ssh_authorized_key (optional), ocpus (optional), memory_in_gbs (optional).
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

            # Resolve Subnet Name to OCID
            resolved_subnet_id = await resolve_subnet_id(subnet_id, resolved_comp_id, config)
            if not resolved_subnet_id:
                return f"Error: Could not resolve subnet input '{subnet_id}' to a valid OCI subnet OCID."

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

            # Resolve SSH Public Key fallback
            resolved_ssh_key = ssh_authorized_key or ssh_public_key
            if not resolved_ssh_key:
                wf_config = await get_workflow_tool_config(config, "tf_provision_vm")
                resolved_ssh_key = wf_config.get("ssh_public_key")
            if not resolved_ssh_key:
                resolved_ssh_key = creds.get("ssh_public_key")

            tfvars = {
                "tenancy_ocid": config_data.get("tenancy"),
                "user_ocid": config_data.get("user"),
                "fingerprint": config_data.get("fingerprint"),
                "private_key_path": temp_pem.name,
                "region": config_data.get("region"),
                "compartment_id": resolved_comp_id,
                "instances": {
                    vm_name: {
                        "availability_domain": resolved_ad,
                        "shape": shape,
                        "subnet_id": resolved_subnet_id,
                        "image_id": image_id,
                        "assign_public_ip": assign_public_ip,
                        "nsg_ids": nsg_ids,
                        "ssh_authorized_key": resolved_ssh_key,
                        "ocpus": ocpus,
                        "memory_in_gbs": memory_in_gbs
                    }
                }
            }
            
            # Load existing state if any (for updates/idempotency)
            state_key = f"tf_state_vm_{vm_name}"
            existing_state = await ToolStateManager.get_state(thread_id, state_key)
            
            if not existing_state:
                # Fallback: search for this state globally across threads
                async with AsyncSessionLocal() as session:
                    query = select(ToolStateModel).where(ToolStateModel.tool_key == state_key)
                    result = await session.execute(query)
                    tool_state = result.scalars().first()
                    if tool_state:
                        existing_state = process_state_data(tool_state.state, mode="decrypt")
                        logger.info(f"Loaded existing VM state for {vm_name} from global DB fallback")
            
            # Execute apply sandbox
            apply_res, output_res, final_state = await run_terraform_in_sandbox(
                tf_dir=TF_DIR,
                env=env,
                tfvars=tfvars,
                target="module.compute",
                state_content=existing_state
            )
            
            # Save updated state
            if final_state:
                await ToolStateManager.save_state(thread_id, state_key, final_state)
                
        finally:
            if os.path.exists(temp_pem.name):
                os.remove(temp_pem.name)

        if apply_res.returncode != 0:
            logger.error(f"Terraform VM provisioning failed: {apply_res.stderr}")
            return f"Error executing Terraform Compute VM deployment: {apply_res.stderr}"
            
        if not output_res or output_res.returncode != 0:
            return "Error: Could not retrieve VM output details from Terraform."

        outputs = json.loads(output_res.stdout)
        inst_details = outputs.get("instance_details", {}).get("value", {})
        details = inst_details.get(vm_name, {})
        ocid = details.get("id", "Unknown")
        pub_ip = details.get("public_ip", "N/A")
        priv_ip = details.get("private_ip", "N/A")
        state = details.get("state", "PROVISIONING")
        
        return (
            f"Successfully provisioned VM Compute Instance '{vm_name}' (State: `{state}`):\n"
            f"- OCID: `{ocid}`\n"
            f"- Public IP: `{pub_ip}`\n"
            f"- Private IP: `{priv_ip}`"
        )
    except Exception as e:
        logger.exception("Failed to provision VM")
        return f"Error provisioning VM: {str(e)}"

class DestroyVMInput(BaseModel):
    vm_name: str = Field(description="The display name of the VM compute instance to delete.")
    compartment_id: Optional[str] = Field(default=None, description="Optional name or OCID of the compartment.")

@tool(args_schema=DestroyVMInput)
async def tf_destroy_vm(
    vm_name: str,
    compartment_id: Optional[str] = None,
    config: RunnableConfig = None
) -> str:
    """
    Destroys the specified VM Compute Instance using Terraform.
    If the VM was created outside Terraform, it will automatically look up its OCID, import it, and destroy it.
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
            resolved_comp_id = await resolve_compartment_id(target_compartment, config)
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
                "instances": {
                    vm_name: {
                        "availability_domain": "dummy_ad",
                        "shape": "dummy_shape",
                        "subnet_id": "dummy_subnet",
                        "image_id": "dummy_image"
                    }
                }
            }
            
            # Load existing state
            state_key = f"tf_state_vm_{vm_name}"
            existing_state = await ToolStateManager.get_state(thread_id, state_key)
            
            if not existing_state:
                # Fallback 1: search for this state globally across threads
                async with AsyncSessionLocal() as session:
                    query = select(ToolStateModel).where(ToolStateModel.tool_key == state_key)
                    result = await session.execute(query)
                    tool_state = result.scalars().first()
                    if tool_state:
                        existing_state = process_state_data(tool_state.state, mode="decrypt")
                        logger.info(f"Loaded existing VM state for {vm_name} from global DB fallback")
            
            # Fallback 2: If still no state, query OCI to find the untracked VM's OCID using fuzzy match
            vm_ocid = None
            if not existing_state:
                logger.info(f"VM '{vm_name}' state not found in database. Looking up OCID in OCI...")
                vm_ocid = await resolve_vm_ocid(vm_name, resolved_comp_id, config)
                if vm_ocid:
                    # Find VM details from live OCI list to construct tfvars
                    vms_list = await run_tf_query(config, "data.oci_core_instances.all_instances", {"compartment_id": resolved_comp_id})
                    if vms_list:
                        for v in vms_list:
                            if v.get("id") == vm_ocid:
                                tfvars["instances"][vm_name]["availability_domain"] = v.get("availability_domain", "dummy_ad")
                                tfvars["instances"][vm_name]["shape"] = v.get("shape", "dummy_shape")
                                break
                            
            # Case A: We have a tracked state file. Use standard destroy.
            if existing_state:
                destroy_res, _, final_state = await run_terraform_in_sandbox(
                    tf_dir=TF_DIR,
                    env=env,
                    tfvars=tfvars,
                    target="module.compute",
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
                    return f"Error executing Terraform VM destruction: {destroy_res.stderr}"
                    
                return f"Successfully destroyed VM Instance '{vm_name}' using Terraform."
                
            # Case B: Untracked VM. Run temporary import + destroy.
            elif vm_ocid:
                logger.info(f"Found VM Compute Instance '{vm_name}' in OCI (OCID: {vm_ocid}). Running auto-import and destroy...")
                
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
                        [tf_cmd, "import", f'module.compute.oci_core_instance.this["{vm_name}"]', vm_ocid],
                        cwd=temp_dir, env=env, capture_output=True, text=True
                    )
                    if import_process.returncode != 0:
                        return f"Error importing Compute Instance for destruction: {import_process.stderr}"
 
                    # Destroy
                    destroy_process = await asyncio.to_thread(
                        subprocess.run,
                        [tf_cmd, "destroy", "-auto-approve", "-target", f'module.compute.oci_core_instance.this["{vm_name}"]'],
                        cwd=temp_dir, env=env, capture_output=True, text=True
                    )
                    if destroy_process.returncode != 0:
                        return f"Error executing Terraform Compute Instance destruction: {destroy_process.stderr}"
                        
                    return f"Successfully imported and destroyed untracked VM Instance '{vm_name}' (OCID: {vm_ocid}) using Terraform."
                finally:
                    try:
                        shutil.rmtree(temp_dir)
                    except Exception:
                        pass
            else:
                return f"Error: VM Instance '{vm_name}' was not found in the OCI compartment (untracked) and has no local state."
                
        finally:
            if os.path.exists(temp_pem.name):
                os.remove(temp_pem.name)
    except Exception as e:
        logger.exception("Failed to destroy VM Compute Instance")
        return f"Error destroying VM: {str(e)}"

class ListVMsInput(BaseModel):
    compartment_id: str = Field(description="The name or OCID of the compartment to list VMs from.")
    detail: bool = Field(default=False, description="Set to True to retrieve detailed information (IP addresses, shapes, availability domain) for each VM.")
    vm_name: Optional[str] = Field(default=None, description="Optional display name or OCID of a specific VM instance to filter and show details for.")

@tool(args_schema=ListVMsInput)
async def tf_list_vms(
    compartment_id: str,
    detail: bool = False,
    vm_name: Optional[str] = None,
    config: RunnableConfig = None
) -> str:
    """
    Lists all VM Compute Instances inside the specified OCI compartment.
    CRITICAL: You must ask the user which OCI compartment they want to list VMs in before running this tool.
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
        key_content = config_data.get("key_content", "").strip()
        if not key_content:
            return "Error: OCI Private key is empty. Please run tf_configure_auth first."
            
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

        resolved_comp_id = await resolve_compartment_id(target_compartment, config)
        if not resolved_comp_id:
            return f"Error: Could not resolve compartment input '{target_compartment}' to a valid OCI compartment OCID."

        vms = await run_tf_query(config, "data.oci_core_instances.all_instances", {"compartment_id": resolved_comp_id})
        if not vms:
            return f"No VM Instances found in compartment: {target_compartment}"

        if vm_name:
            # Fuzzy match VM name
            search_norm = normalize_name(vm_name)
            vms = [v for v in vms if search_norm in normalize_name(v.get("display_name", "")) or vm_name == v.get("id")]
            if not vms:
                return f"No VM Instance matching '{vm_name}' was found in compartment: {target_compartment}"

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
                "compartment_id": resolved_comp_id
            }

            result_lines = [f"Found {len(vms)} VM Instance(s) in compartment (via Terraform):\n"]
            for v in vms:
                name = v.get("display_name", "N/A")
                ocid = v.get("id", "N/A")
                state = v.get("state", v.get("lifecycle_state", "N/A"))
                shape = v.get("shape", "N/A")
                
                if detail or vm_name:
                    public_ip = "N/A"
                    private_ip = "N/A"
                    ad = v.get("availability_domain", "N/A")
                    
                    # 1. Try to fetch live IPs using OCI CLI (fast & live!)
                    try:
                        temp_dir_vnic = tempfile.mkdtemp()
                        config_path_vnic = os.path.join(temp_dir_vnic, "config")
                        key_path_vnic = os.path.join(temp_dir_vnic, "key.pem")
                        with open(key_path_vnic, "w") as f:
                            f.write(key_content)
                        config_content_vnic = f"""[DEFAULT]
user={config_data.get('user')}
fingerprint={config_data.get('fingerprint')}
key_file={key_path_vnic.replace('\\', '/')}
tenancy={config_data.get('tenancy')}
region={config_data.get('region')}
"""
                        with open(config_path_vnic, "w") as f:
                            f.write(config_content_vnic)
                            
                        from Apps.AgentMesh.Utilities.Tools.InfraTool.OCI.OCI_using_terraform.tools.tf_nsg_tools import run_oci_cli as run_oci_cli_nsg
                        vnic_attach_res = await run_oci_cli_nsg([
                            "compute", "vnic-attachment", "list",
                            "--compartment-id", resolved_comp_id,
                            "--instance-id", ocid
                        ], config_path_vnic)
                        attachments = vnic_attach_res.get("data", [])
                        if attachments:
                            vnic_id = attachments[0].get("vnic-id")
                            if vnic_id:
                                vnic_res = await run_oci_cli_nsg([
                                    "network", "vnic", "get",
                                    "--vnic-id", vnic_id
                                ], config_path_vnic)
                                vnic_data = vnic_res.get("data", {})
                                if vnic_data.get("public-ip"):
                                    public_ip = vnic_data.get("public-ip")
                                if vnic_data.get("private-ip"):
                                    private_ip = vnic_data.get("private-ip")
                        shutil.rmtree(temp_dir_vnic)
                    except Exception as e:
                        logger.warning(f"Failed to fetch live IP addresses via OCI CLI: {e}")

                    # 2. Fallback to database saved state if IPs are N/A
                    if (public_ip == "N/A" or private_ip == "N/A") and name != "N/A":
                        _cfg = config.get("configurable", {}) if isinstance(config, dict) else (getattr(config, "configurable", {}) if config else {})
                        thread_id = _cfg.get("thread_id") or "default_thread"
                        state_key = f"tf_state_vm_{name}"
                        db_state = await ToolStateManager.get_state(thread_id, state_key)
                        if not db_state:
                            from Database.Database import AsyncSessionLocal
                            from Database.Models import ToolStateModel
                            from sqlalchemy import select
                            async with AsyncSessionLocal() as session:
                                query = select(ToolStateModel).where(ToolStateModel.tool_key == state_key)
                                result = await session.execute(query)
                                tool_state = result.scalars().first()
                                if tool_state:
                                    db_state = process_state_data(tool_state.state, mode="decrypt")
                        if db_state:
                            try:
                                state_json = json.loads(db_state)
                                for res in state_json.get("resources", []):
                                    if res.get("type") == "oci_core_instance":
                                        for inst in res.get("instances", []):
                                            if inst.get("index_key") == name:
                                                attrs = inst.get("attributes", {})
                                                if attrs.get("public_ip"):
                                                    public_ip = attrs.get("public_ip")
                                                if attrs.get("private_ip"):
                                                    private_ip = attrs.get("private_ip")
                            except Exception:
                                pass

                    # 2. Fallback to dynamic terraform import if still N/A (for untracked VMs!)
                    if (public_ip == "N/A" or private_ip == "N/A") and name != "N/A" and ocid != "N/A":
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
                                [tf_cmd, "import", f'module.compute.oci_core_instance.this["{name}"]', ocid],
                                cwd=temp_dir, env=env, capture_output=True, text=True
                            )
                            if import_process.returncode == 0:
                                state_path = os.path.join(temp_dir, "terraform.tfstate")
                                if os.path.exists(state_path):
                                    with open(state_path, "r") as sf:
                                        state_json = json.load(sf)
                                    for res in state_json.get("resources", []):
                                        if res.get("type") == "oci_core_instance":
                                            for inst in res.get("instances", []):
                                                if inst.get("index_key") == name:
                                                    attrs = inst.get("attributes", {})
                                                    if attrs.get("public_ip"):
                                                        public_ip = attrs.get("public_ip")
                                                    if attrs.get("private_ip"):
                                                        private_ip = attrs.get("private_ip")
                        except Exception:
                            pass
                        finally:
                            try:
                                shutil.rmtree(temp_dir)
                            except Exception:
                                pass

                    result_lines.append(
                        f"- **{name}** (State: `{state}`)\n"
                        f"  Shape: `{shape}`\n"
                        f"  Public IP: `{public_ip}`\n"
                        f"  Private IP: `{private_ip}`\n"
                        f"  Availability Domain: `{ad}`\n"
                        f"  OCID: `{ocid}`"
                    )
                else:
                    result_lines.append(
                        f"- **{name}** (State: `{state}`)\n"
                        f"  OCID: `{ocid}`"
                    )
                
            return "\n".join(result_lines)
        finally:
            if os.path.exists(temp_pem.name):
                try:
                    os.remove(temp_pem.name)
                except Exception:
                    pass
    except Exception as e:
        logger.exception("Failed to list VM Compute Instances")
        return f"Error listing VM Compute Instances: {str(e)}"

class UpdateVMStateInput(BaseModel):
    vm_name: str = Field(description="The display name or OCID of the VM compute instance.")
    state: str = Field(description="The target state: 'RUNNING' or 'STOPPED'.")
    compartment_id: Optional[str] = Field(default=None, description="Optional name or OCID of the compartment.")

@tool(args_schema=UpdateVMStateInput)
async def tf_update_vm_state(
    vm_name: str,
    state: str,
    compartment_id: Optional[str] = None,
    config: RunnableConfig = None
) -> str:
    """
    Updates the power state of an existing Virtual Machine (VM Compute Instance) to RUNNING or STOPPED using Terraform.
    Inputs: vm_name, state ('RUNNING' or 'STOPPED'), compartment_id (optional).
    """
    try:
        if state.upper() not in ["RUNNING", "STOPPED"]:
            return f"Error: Invalid state '{state}'. Supported states are 'RUNNING' and 'STOPPED'."

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

            # Load existing state
            state_key = f"tf_state_vm_{vm_name}"
            existing_state = await ToolStateManager.get_state(thread_id, state_key)
            if not existing_state:
                async with AsyncSessionLocal() as session:
                    query = select(ToolStateModel).where(ToolStateModel.tool_key == state_key)
                    result = await session.execute(query)
                    tool_state = result.scalars().first()
                    if tool_state:
                        existing_state = process_state_data(tool_state.state, mode="decrypt")

            if not existing_state:
                return f"Error: No existing Terraform state found for VM '{vm_name}'. You must first create or import the VM using 'tf_provision_vm' before managing its power state."

            # Reconstruct variables from existing state
            ad = None
            shape = None
            subnet = None
            image = None
            assign_public = True
            nsg = None
            ssh_key = None
            ocpus_val = None
            mem_val = None

            try:
                state_data = json.loads(existing_state)
                resources = state_data.get("resources", [])
                for res in resources:
                    if res.get("type") == "oci_core_instance" and res.get("name") == "this":
                        for inst in res.get("instances", []):
                            if inst.get("index_key") == vm_name:
                                attrs = inst.get("attributes", {})
                                ad = attrs.get("availability_domain")
                                shape = attrs.get("shape")
                                # Subnet details
                                vnic_list = attrs.get("create_vnic_details", [])
                                if vnic_list:
                                    vnic = vnic_list[0]
                                    subnet = vnic.get("subnet_id")
                                    assign_public = vnic.get("assign_public_ip", True)
                                    nsg = vnic.get("nsg_ids")
                                # Source details
                                src_list = attrs.get("source_details", [])
                                if src_list:
                                    src = src_list[0]
                                    image = src.get("source_id")
                                # SSH Key
                                meta = attrs.get("metadata", {})
                                ssh_key = meta.get("ssh_authorized_keys")
                                # Shape config
                                s_cfg_list = attrs.get("shape_config", [])
                                if s_cfg_list:
                                    s_cfg = s_cfg_list[0]
                                    ocpus_val = s_cfg.get("ocpus")
                                    mem_val = s_cfg.get("memory_in_gbs")
                                break
            except Exception as e:
                logger.error(f"Error parsing existing state attributes: {e}")

            if not ad or not shape or not subnet or not image:
                return f"Error: Could not retrieve all required VM attributes (AD, Shape, Subnet, Image) from existing Terraform state."

            tfvars = {
                "tenancy_ocid": config_data.get("tenancy"),
                "user_ocid": config_data.get("user"),
                "fingerprint": config_data.get("fingerprint"),
                "private_key_path": temp_pem.name,
                "region": config_data.get("region"),
                "compartment_id": resolved_comp_id,
                "instances": {
                    vm_name: {
                        "availability_domain": ad,
                        "shape": shape,
                        "subnet_id": subnet,
                        "image_id": image,
                        "assign_public_ip": assign_public,
                        "nsg_ids": nsg,
                        "ssh_authorized_key": ssh_key,
                        "ocpus": ocpus_val,
                        "memory_in_gbs": mem_val,
                        "state": state.upper()
                    }
                }
            }

            # Execute apply sandbox
            apply_res, output_res, final_state = await run_terraform_in_sandbox(
                tf_dir=TF_DIR,
                env=env,
                tfvars=tfvars,
                target="module.compute",
                state_content=existing_state
            )

            # Save updated state
            if final_state:
                await ToolStateManager.save_state(thread_id, state_key, final_state)

        finally:
            if os.path.exists(temp_pem.name):
                os.remove(temp_pem.name)

        if apply_res.returncode != 0:
            logger.error(f"Terraform VM power state update failed: {apply_res.stderr}")
            return f"Error executing Terraform VM power state update: {apply_res.stderr}"

        return f"Successfully updated VM Instance '{vm_name}' power state to {state.upper()} using Terraform."
    except Exception as e:
        logger.exception("Failed to update VM state")
        return f"Error updating VM state: {str(e)}"


class BackupBootVolumeInput(BaseModel):
    backup_name: str = Field(description="The display name for the boot volume backup.")
    vm_name: str = Field(description="The display name of the VM compute instance or the boot volume OCID.")
    type: str = Field(default="INCREMENTAL", description="Backup type: 'FULL' or 'INCREMENTAL'.")
    compartment_id: Optional[str] = Field(default=None, description="Optional name or OCID of the compartment.")

@tool(args_schema=BackupBootVolumeInput)
async def tf_backup_boot_volume(
    backup_name: str,
    vm_name: str,
    type: str = "INCREMENTAL",
    compartment_id: Optional[str] = None,
    config: RunnableConfig = None
) -> str:
    """
    Creates a backup of a VM Compute Instance's Boot Volume using Terraform.
    Inputs:
      backup_name: The name for the backup.
      vm_name: The friendly VM name (e.g., 'prak_vm') or the boot volume OCID. Pass the VM name directly; the tool automatically resolves it to the correct boot volume.
      type: 'FULL' or 'INCREMENTAL' (default: 'INCREMENTAL').
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

            backup_detail = {
                "type": type.upper()
            }
            if vm_name.startswith("ocid1.bootvolume.oc1..") or vm_name.startswith("ocid1.instance.oc1.."):
                if vm_name.startswith("ocid1.bootvolume.oc1.."):
                    backup_detail["boot_volume_id"] = vm_name
                else:
                    try:
                        from Apps.AgentMesh.Utilities.Tools.InfraTool.OCI.OCI_using_terraform.tools.tf_nsg_tools import run_oci_cli as run_oci_cli_nsg
                        inst_res = await run_oci_cli_nsg([
                            "compute", "instance", "get",
                            "--instance-id", vm_name
                        ], temp_pem.name)
                        inst_data = inst_res.get("data", {})
                        ad = inst_data.get("availability-domain")
                        true_comp_id = inst_data.get("compartment-id") or resolved_comp_id
                        if ad:
                            res = await run_oci_cli_nsg([
                                "compute", "boot-volume-attachment", "list",
                                "--compartment-id", true_comp_id,
                                "--instance-id", vm_name,
                                "--availability-domain", ad
                            ], temp_pem.name)
                            attachments = res.get("data", [])
                            if attachments:
                                boot_vol_id = attachments[0].get("boot-volume-id")
                                if boot_vol_id:
                                    backup_detail["boot_volume_id"] = boot_vol_id
                                    resolved_comp_id = true_comp_id
                    except Exception as e:
                        logger.warning(f"Failed to resolve boot volume ID for VM OCID '{vm_name}' via OCI CLI: {e}")
            else:
                try:
                    from Apps.AgentMesh.Utilities.Tools.InfraTool.OCI.OCI_using_terraform.tools.tf_auth_tools import resolve_vm_ocid
                    from Apps.AgentMesh.Utilities.Tools.InfraTool.OCI.OCI_using_terraform.tools.tf_nsg_tools import run_oci_cli as run_oci_cli_nsg
                    
                    vm_ocid = await resolve_vm_ocid(vm_name, resolved_comp_id, config)
                    if vm_ocid:
                        inst_res = await run_oci_cli_nsg([
                            "compute", "instance", "get",
                            "--instance-id", vm_ocid
                        ], temp_pem.name)
                        inst_data = inst_res.get("data", {})
                        ad = inst_data.get("availability-domain")
                        true_comp_id = inst_data.get("compartment-id") or resolved_comp_id
                        if ad:
                            res = await run_oci_cli_nsg([
                                "compute", "boot-volume-attachment", "list",
                                "--compartment-id", true_comp_id,
                                "--instance-id", vm_ocid,
                                "--availability-domain", ad
                            ], temp_pem.name)
                            attachments = res.get("data", [])
                            if attachments:
                                boot_vol_id = attachments[0].get("boot-volume-id")
                                if boot_vol_id:
                                    backup_detail["boot_volume_id"] = boot_vol_id
                                    resolved_comp_id = true_comp_id
                except Exception as e:
                    logger.warning(f"Failed to dynamically resolve boot volume ID for VM '{vm_name}' via OCI CLI: {e}")
                
            if "boot_volume_id" not in backup_detail:
                backup_detail["vm_name"] = vm_name

            tfvars = {
                "tenancy_ocid": config_data.get("tenancy"),
                "user_ocid": config_data.get("user"),
                "fingerprint": config_data.get("fingerprint"),
                "private_key_path": temp_pem.name,
                "region": config_data.get("region"),
                "compartment_id": resolved_comp_id,
                "boot_volume_backups": {
                    backup_name: backup_detail
                }
            }

            state_key = f"tf_state_boot_volume_backup_{backup_name}"
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
                target="module.compute",
                state_content=existing_state
            )

            if final_state:
                await ToolStateManager.save_state(thread_id, state_key, final_state)

        finally:
            if os.path.exists(temp_pem.name):
                os.remove(temp_pem.name)

        if apply_res.returncode != 0:
            logger.error(f"Terraform boot volume backup failed: {apply_res.stderr}")
            return f"Error executing Terraform Boot Volume backup: {apply_res.stderr}"

        if not output_res or output_res.returncode != 0:
            return "Error: Could not retrieve backup details from Terraform."

        outputs = json.loads(output_res.stdout)
        backup_ids = outputs.get("boot_volume_backup_ids", {}).get("value", {})
        backup_ocid = backup_ids.get(backup_name, "Unknown")

        return f"Successfully created Boot Volume Backup '{backup_name}' (OCID: {backup_ocid}, Type: {type.upper()}) for VM: {vm_name}"
    except Exception as e:
        logger.exception("Failed to create boot volume backup")
        return f"Error creating boot volume backup: {str(e)}"

# Registering Compute Tools
registry.register(
    tf_backup_boot_volume,
    ToolMeta(
        name="tf_backup_boot_volume",
        display_name="OCI Terraform: Create Boot Volume Backup",
        description="Creates a backup of a VM Compute Instance's Boot Volume using Terraform. Pass the friendly VM name (e.g. 'prak_vm') directly to 'vm_name' to automatically resolve its boot volume.",
        category="Infrastructure",
        package="oci_terraform"
    )
)

# Registering Compute Tools
registry.register(
    tf_update_vm_state,
    ToolMeta(
        name="tf_update_vm_state",
        display_name="OCI Terraform: Update Compute VM Power State",
        description="Starts (RUNNING) or Stops (STOPPED) an existing Compute VM instance using Terraform.",
        category="Infrastructure",
        package="oci_terraform"
    )
)

registry.register(
    tf_list_images,
    ToolMeta(
        name="tf_list_images",
        display_name="OCI Terraform: List OS Images",
        description="Lists all available OS images in a compartment.",
        category="Infrastructure",
        package="oci_terraform"
    )
)

registry.register(
    tf_list_shapes,
    ToolMeta(
        name="tf_list_shapes",
        display_name="OCI Terraform: List Compute Shapes",
        description="Lists all available compute shapes/sizes in a compartment.",
        category="Infrastructure",
        package="oci_terraform"
    )
)

registry.register(
    tf_provision_vm,
    ToolMeta(
        name="tf_provision_vm",
        display_name="OCI Terraform: Create Compute VM",
        description="Creates a new Virtual Machine Compute instance with an image, shape, subnet, ssh key, and optional NSGs.",
        category="Infrastructure",
        package="oci_terraform",
        version="1.0.0",
        configs=[
            {"name": "ssh_public_key", "type": "string", "description": "SSH public key content for VM access", "required": False}
        ]
    )
)

registry.register(
    tf_destroy_vm,
    ToolMeta(
        name="tf_destroy_vm",
        display_name="OCI Terraform: Delete Compute VM",
        description="Deletes/destroys an existing Compute VM instance.",
        category="Infrastructure",
        package="oci_terraform"
    )
)

registry.register(
    tf_list_vms,
    ToolMeta(
        name="tf_list_vms",
        display_name="OCI Terraform: List Compute VMs",
        description="Lists all VM instances in a compartment.",
        category="Infrastructure",
        package="oci_terraform"
    )
)


# --- Helper State / Config Methods (Decoupled from OCI SDK) ---

async def get_ssh_config(config: RunnableConfig) -> dict:
    """Retrieves persistent SSH configuration from thread state."""
    try:
        if isinstance(config, dict):
            configurable = config.get("configurable", {})
        else:
            configurable = getattr(config, "configurable", {}) if config else {}
        thread_id = configurable.get("thread_id") or "default_thread"
        return await ToolStateManager.get_state(thread_id, "ssh_config") or {}
    except Exception:
        return {}

async def save_ssh_config(config: RunnableConfig, ssh_data: dict):
    """Saves SSH configuration persistently for the current thread."""
    try:
        if isinstance(config, dict):
            configurable = config.get("configurable", {})
        else:
            configurable = getattr(config, "configurable", {}) if config else {}
        thread_id = configurable.get("thread_id") or "default_thread"
        await ToolStateManager.save_state(thread_id, "ssh_config", ssh_data)
    except Exception as e:
        logger.error(f"Failed to save SSH configuration: {str(e)}")

async def get_workflow_tool_config(config: RunnableConfig, tool_name: str) -> dict:
    """Retrieves the configuration for a specific tool from the active workflow definition."""
    try:
        from Apps.AgentMesh.Core.WorkflowRegistry import workflow_registry
        if isinstance(config, dict):
            configurable = config.get("configurable", {})
        else:
            configurable = getattr(config, "configurable", {}) if config else {}
        context = configurable.get("context", {})
        workflow_id = context.get("workflow_id")
        
        if not workflow_id:
            return {}
            
        config_obj = workflow_registry.get_config(workflow_id)
        if not config_obj or not config_obj.nodes:
            return {}
            
        for node in config_obj.nodes:
            tool_configs = getattr(node, "tool_configs", {}) or {}
            if tool_configs and tool_name in tool_configs:
                return tool_configs[tool_name]
                
        return {}
    except Exception as e:
        logger.error(f"Failed to retrieve workflow tool config: {e}")
        return {}


# --- Pydantic Schema for SSH Execution ---
class SSHExecutorInput(BaseModel):
    command: str = Field(
        ...,
        description="The shell command to execute on the remote VM."
    )
    ip_address: Optional[str] = Field(
        default=None,
        description="Optional public IP address of the target VM. If omitted, it will be automatically resolved."
    )
    username: Optional[str] = Field(
        default=None,
        description="Optional SSH username. Defaults to 'opc'."
    )
    vm_public_ip: Optional[str] = Field(
        default=None,
        description="Optional public IP address of the target VM."
    )
    os_username: Optional[str] = Field(
        default=None,
        description="Optional SSH connection username (defaults to 'opc')."
    )
    ssh_private_key: Optional[str] = Field(
        default=None,
        description="Optional SSH private key content to authenticate with."
    )
    private_key: Optional[str] = Field(
        default=None,
        description="Optional SSH private key content to authenticate with."
    )
    vm_name: Optional[str] = Field(
        default=None,
        description="Optional name of the target VM instance (e.g. 'agentic_vm', 'prak_vm')."
    )
    instance_id: Optional[str] = Field(
        default=None,
        description="Optional OCID of the target VM instance."
    )

from pydantic import BaseModel, Field

async def resolve_vm_ip(vm_input: str, compartment_id: Optional[str] = None, config: RunnableConfig = None) -> Optional[str]:
    if not vm_input:
        return None
    import re
    if re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", vm_input.strip()):
        return vm_input.strip()
        
    try:
        from Apps.AgentMesh.Utilities.Tools.InfraTool.OCI.OCI_using_terraform.tools.tf_auth_tools import resolve_oci_creds, resolve_compartment_id, resolve_vm_ocid
        from Apps.AgentMesh.Utilities.Tools.InfraTool.OCI.OCI_using_terraform.tools.tf_nsg_tools import run_oci_cli as run_oci_cli_nsg
        import tempfile
        import shutil
        
        creds = await resolve_oci_creds(config)
        target_comp = compartment_id or creds.get("compartment_id") or creds.get("tenancy") or ""
        resolved_comp_id = await resolve_compartment_id(target_comp, config)
        if not resolved_comp_id:
            return None
            
        vm_ocid = await resolve_vm_ocid(vm_input, resolved_comp_id, config)
        if not vm_ocid:
            return None
            
        temp_dir = tempfile.mkdtemp()
        config_path = os.path.join(temp_dir, "config")
        key_path = os.path.join(temp_dir, "key.pem")
        with open(key_path, "w") as f:
            f.write(creds.get("private_key"))
        config_content = f"""[DEFAULT]
user={creds.get('user_id')}
fingerprint={creds.get('fingerprint')}
key_file={key_path.replace('\\', '/')}
tenancy={creds.get('tenancy_id')}
region={creds.get('region')}
"""
        with open(config_path, "w") as f:
            f.write(config_content)
            
        vnic_attach_res = await run_oci_cli_nsg([
            "compute", "vnic-attachment", "list",
            "--compartment-id", resolved_comp_id,
            "--instance-id", vm_ocid
        ], config_path)
        attachments = vnic_attach_res.get("data", [])
        public_ip = None
        if attachments:
            vnic_id = attachments[0].get("vnic-id")
            if vnic_id:
                vnic_res = await run_oci_cli_nsg([
                    "network", "vnic", "get",
                    "--vnic-id", vnic_id
                ], config_path)
                vnic_data = vnic_res.get("data", {})
                public_ip = vnic_data.get("public-ip")
        shutil.rmtree(temp_dir)
        return public_ip
    except Exception as e:
        logger.warning(f"resolve_vm_ip failed for {vm_input}: {e}")
        return None

async def tf_ssh_executor(
    command: str,
    ip_address: Optional[str] = None,
    username: Optional[str] = None,
    vm_public_ip: Optional[str] = None,
    os_username: Optional[str] = None,
    ssh_private_key: Optional[str] = None,
    private_key: Optional[str] = None,
    vm_name: Optional[str] = None,
    instance_id: Optional[str] = None,
    config: RunnableConfig = None,
    **kwargs
) -> str:
    """
    Runs shell commands on a remote VM via SSH.
    
    CRITICAL IP RULE: You MUST provide the correct, fresh `vm_public_ip` and `os_username` (e.g. 'opc' or 'ubuntu').
    - If you recently listed the VMs, extract the 'Public IP' from that fresh output and pass it here.
    - If you do NOT have a fresh IP from a recent listing or the user's prompt, you MUST STOP and explicitly ASK the user for the VM's current Public IP and SSH username before running this tool.
    - DO NOT guess, DO NOT leave blank, and DO NOT reuse stale IPs from older turns in the chat history.
    
    The SSH private key is automatically resolved from the backend settings. Do NOT ask the user for the private key.
    """
    try:
        if isinstance(config, dict):
            configurable = config.get("configurable", {})
        else:
            configurable = getattr(config, "configurable", {}) if config else {}
        
        context = configurable.get("context", {}) or {}
        wf_vars = context.get("workflow_variables", {}) if isinstance(context, dict) else {}
        creds = await resolve_oci_creds(config) or {}

        logger.info(f"tf_ssh_executor entry: command={command[:50]}..., ip_address={ip_address}, vm_public_ip={vm_public_ip}, username={username}, os_username={os_username}")

        # IP Address & Username: Prioritize explicit arguments (from chat), then fall back to saved UI configuration
        saved_config = await get_ssh_config(config)
        
        target_ip = ip_address or vm_public_ip
        if not target_ip:
            target_ip = saved_config.get("vm_public_ip") or saved_config.get("ip_address")
            
        target_user = username or os_username
        if not target_user:
            target_user = saved_config.get("os_username") or saved_config.get("username")
        
        logger.info(f"tf_ssh_executor resolved targets: target_ip={target_ip}, target_user={target_user}")
            
        # Private Key Resolution
        target_key = configurable.get("ssh_private_key")
        if not target_key:
            target_key = wf_vars.get("ssh_private_key") or wf_vars.get("oci_ssh_private_key")
        if not target_key:
            target_key = creds.get("ssh_private_key")
        if not target_key:
            for lvm_tool in ["tf_setup_block_volume_os", "tf_resize_fs_lvm", "tf_delete_lvm_os"]:
                wf_config = await get_workflow_tool_config(config, lvm_tool)
                target_key = wf_config.get("ssh_private_key")
                if target_key:
                    break
        if not target_key:
            wf_config = await get_workflow_tool_config(config, "tf_ssh_executor")
            target_key = wf_config.get("ssh_private_key")
        if not target_key:
            target_key = ssh_private_key
        if not target_key:
            saved_config = await get_ssh_config(config)
            target_key = saved_config.get("ssh_private_key")
 
        # Sanitize IP address string
        if target_ip:
            target_ip = str(target_ip).strip().replace("'", "").replace('"', "")
            if target_ip.lower() in ["null", "none", ""]:
                target_ip = None
 
        # Validation
        if not target_ip:
            return "Error: VM public IP address was not provided. Please ask the user for the VM's public IP address and username, then call this tool again with those values."
        if not target_key:
            return "Error: SSH private key is not configured. Please save 'ssh_private_key' in UI settings."
 
        # Clean private key format
        target_key = target_key.strip()
        if "\\n" in target_key:
            target_key = target_key.replace("\\n", "\n")
        if "-----BEGIN" in target_key and "\n" not in target_key:
            target_key = target_key.replace("-----BEGIN RSA PRIVATE KEY-----", "-----BEGIN RSA PRIVATE KEY-----\n")
            target_key = target_key.replace("-----END RSA PRIVATE KEY-----", "\n-----END RSA PRIVATE KEY-----")
            target_key = target_key.replace("-----BEGIN OPENSSH PRIVATE KEY-----", "-----BEGIN OPENSSH PRIVATE KEY-----\n")
            target_key = target_key.replace("-----END OPENSSH PRIVATE KEY-----", "\n-----END OPENSSH PRIVATE KEY-----")

        # Save only the SSH key to thread state (not IP/username to avoid stale cache)
        try:
            await save_ssh_config(config, {
                "ssh_private_key": target_key
            })
        except Exception as save_err:
            logger.warning(f"Failed to persist SSH key to thread state: {save_err}")
 
        # Connection and Execution
        logger.info(f"Connecting to VM at {target_ip} as user {target_user}...")
        
        def run_ssh():
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            try:
                pkey = None
                for key_class in [paramiko.RSAKey, paramiko.Ed25519Key, paramiko.ECDSAKey]:
                    try:
                        pkey = key_class.from_private_key(io.StringIO(target_key))
                        break
                    except Exception:
                        continue
                
                if pkey is None:
                    raise ValueError("Invalid private key format. Could not parse as RSA, Ed25519, or ECDSA key.")
                
                ssh.connect(hostname=target_ip, username=target_user, pkey=pkey, timeout=15)
                
                logger.info(f"Executing command: {command}")
                stdin, stdout, stderr = ssh.exec_command(command, timeout=300)
                
                out = stdout.read().decode('utf-8', errors='ignore')
                err = stderr.read().decode('utf-8', errors='ignore')
                exit_status = stdout.channel.recv_exit_status()
                
                return exit_status, out, err
            except Exception as inner_e:
                logger.exception("Error executing SSH command block")
                raise inner_e
            finally:
                ssh.close()
 
        exit_code, out_str, err_str = await asyncio.to_thread(run_ssh)
        
        if exit_code == 0:
            return out_str
        else:
            return f"Error executing command (Exit Code: {exit_code}):\nSTDOUT:\n{out_str}\nSTDERR:\n{err_str}"
            
    except Exception as e:
        logger.exception("SSH execution tool failed")
        return f"Error connecting/executing over SSH: {str(e)}"


# --- OS Management Tool ---

class OSManagementInput(BaseModel):
    action: Optional[str] = Field(
        default=None,
        description="The OS action to perform. Allowed values: 'CREATE', 'DELETE', 'UNLOCK'."
    )
    target_username: Optional[str] = Field(
        default=None,
        description="The target OS username to CREATE, DELETE, or UNLOCK (e.g. 'prakriti'). DO NOT pass the SSH connection username (e.g. 'opc') here."
    )
    username: Optional[str] = Field(
        default=None,
        description="[DEPRECATED, use target_username instead] The target OS username to CREATE, DELETE, or UNLOCK."
    )
    password: Optional[str] = Field(
        default=None,
        description="Password for the user (only applicable for 'CREATE' action). If not provided, a strong random password will be auto-generated.",
        repr=False,
    )
    ip_address: Optional[str] = Field(
        default=None,
        description="The target VM public IP address. CRITICAL: You MUST provide the fresh IP here from recent tool outputs. Do NOT leave blank."
    )
    vm_public_ip: Optional[str] = Field(
        default=None,
        description="The target VM public IP address. CRITICAL: You MUST provide the fresh IP here from recent tool outputs. Do NOT leave blank."
    )
    os_username: Optional[str] = Field(
        default=None,
        description="The SSH login username (e.g. 'opc'). CRITICAL: You MUST provide this explicitly. Do NOT leave blank."
    )
    ssh_private_key: Optional[str] = Field(
        default=None,
        description="SSH private key. Do NOT ask the user for this — it is automatically resolved from the backend."
    )
    private_key: Optional[str] = Field(
        default=None,
        description="SSH private key. Do NOT ask the user for this — it is automatically resolved from the backend."
    )
    vm_name: Optional[str] = Field(
        default=None,
        description="Optional name of the target VM instance (e.g. 'agentic_vm', 'prak_vm')."
    )
    instance_id: Optional[str] = Field(
        default=None,
        description="Optional OCID of the target VM instance."
    )
    task_name: Optional[str] = Field(
        default=None,
        description="Legacy parameter for backward compatibility. E.g. 'create_user', 'delete_user'."
    )
    params: Optional[Union[List[str], str]] = Field(
        default=None,
        description="Legacy parameter for backward compatibility."
    )

def generate_strong_password(length=12):
    alphabet = string.ascii_letters + string.digits + "!@#$%"
    while True:
        password = ''.join(secrets.choice(alphabet) for i in range(length))
        if (any(c.islower() for c in password)
                and any(c.isupper() for c in password)
                and any(c.isdigit() for c in password)):
            return password

@tool(args_schema=OSManagementInput)
async def tf_run_os_task(
    action: Optional[str] = None,
    username: Optional[str] = None,
    target_username: Optional[str] = None,
    password: Optional[str] = None,
    task_name: Optional[str] = None,
    params: Optional[Union[List[str], str]] = None,
    ip_address: Optional[str] = None,
    vm_public_ip: Optional[str] = None,
    os_username: Optional[str] = None,
    ssh_private_key: Optional[str] = None,
    private_key: Optional[str] = None,
    vm_name: Optional[str] = None,
    instance_id: Optional[str] = None,
    config: RunnableConfig = None,
    **kwargs
) -> str:
    """
    Executes OS-level user management operations (CREATE, DELETE, UNLOCK) on a VM.
    
    CRITICAL IP RULE: You MUST provide the correct, fresh `vm_public_ip` and `os_username` (e.g. 'opc' or 'ubuntu').
    - If you recently listed the VMs using `tf_list_vms` (or similar), extract the 'Public IP' from that fresh output and pass it here.
    - If you do NOT have a fresh IP from a recent listing or the user's current prompt, you MUST STOP and explicitly ASK the user for the VM's current Public IP and SSH username before running this tool.
    - DO NOT guess, DO NOT leave blank, and DO NOT reuse stale IPs from older turns in the chat history.
    
    The SSH private key is automatically resolved from the UI settings. Do NOT ask the user for the private key.
    """
    try:
        resolved_action = action
        resolved_username = target_username or username
        resolved_password = password
 
        if task_name:
            resolved_action = task_name.replace("_user", "").upper()
 
        if params:
            if isinstance(params, str):
                import re
                parsed_params = re.split(r'[\s,]+', params.strip())
            else:
                parsed_params = params
           
            if parsed_params and len(parsed_params) > 0:
                if not resolved_username:
                    resolved_username = parsed_params[0]
                if len(parsed_params) > 1 and not resolved_password:
                    resolved_password = parsed_params[1]
 
        if not resolved_action:
            return "Error: OS Task action or task_name is missing. Please provide action (CREATE, DELETE, UNLOCK)."
       
        resolved_action = resolved_action.upper()
        if resolved_action not in ["CREATE", "DELETE", "UNLOCK"]:
            return f"Error: Invalid action '{resolved_action}'. Supported actions are: CREATE, DELETE, UNLOCK."
 
        if not resolved_username:
            return "Error: Target username is missing. Please specify a username."
 
        # Load the shell script contents from local OCI/scripts/ file relative to this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        script_file_name = f"{resolved_action.lower()}_user.sh"
        # Relative path back out of OCI_using_terraform/tools/ to OCI/scripts
        script_path = os.path.abspath(os.path.join(current_dir, "..", "..", "scripts", script_file_name))
 
        if not os.path.exists(script_path):
            return f"Error: Helper shell script {script_file_name} was not found at {script_path}."
 
        with open(script_path, "r") as f:
            script_content = f.read()
 
        escaped_script = script_content.replace('\\', '\\\\').replace('"', '\\"').replace('$', '\\$')
 
        cmd_args = resolved_username
        actual_password = None
        if resolved_action == "CREATE":
            actual_password = resolved_password or generate_strong_password()
            cmd_args += f" '{actual_password}'"
 
        full_command = f"bash -c \"{escaped_script}\" -- {cmd_args}"
 
        logger.info(f"Triggering OS Task over SSH: Action={resolved_action}, User={resolved_username}")
 
        ssh_func = getattr(tf_ssh_executor, "coroutine", None) or getattr(tf_ssh_executor, "func", None) or tf_ssh_executor
        ssh_result = await ssh_func(
            command=full_command,
            ip_address=ip_address,
            vm_public_ip=vm_public_ip,
            os_username=os_username,
            ssh_private_key=ssh_private_key,
            private_key=private_key,
            vm_name=vm_name,
            instance_id=instance_id,
            config=config,
            **kwargs
        )
 
        response = f"--- OS Action: {resolved_action} user '{resolved_username}' ---\n"
        response += ssh_result
       
        if resolved_action == "CREATE" and "successfully" in ssh_result.lower():
            response += f"\n[CRITICAL INFO] Account created successfully.\nUsername: {resolved_username}\nPassword: {actual_password}\n(Please keep these credentials safe!)"
 
        return response
 
    except Exception as e:
        logger.exception("Failed to run OS management task")
        return f"Error running OS task: {str(e)}"


# --- Tool Registrations ---

def register_ssh_tools():
    # 1. tf_run_os_task
    registry.register(
        tf_run_os_task,
        ToolMeta(
            name="tf_run_os_task",
            display_name="OCI Terraform: Run VM OS Task",
            description="Manage system users (CREATE, DELETE, UNLOCK) on a remote VM. Authentication is handled automatically.",
            category="Infrastructure",
            package="oci_terraform",
            version="1.0.0"
        )
    )

register_ssh_tools()
