import os
import json
import logging
import asyncio
import subprocess
import tempfile
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

from langchain_core.tools import tool
from Apps.AgentMesh.Core.ToolRegistry import registry, ToolMeta
from Apps.AgentMesh.Schema.ToolSchema import ToolConfigMeta
from langchain_core.runnables import RunnableConfig

# Auth & Sandbox Helpers
from Apps.AgentMesh.Utilities.Tools.InfraTool.OCI.OCI_using_terraform.tools.tf_auth_tools import (
    resolve_oci_creds,
    run_terraform_in_sandbox
)
from Apps.AgentMesh.Utilities.Tools.InfraTool.OCI.OCI_using_terraform.tools.tf_compute_tools import get_workflow_tool_config, resolve_availability_domain
from Apps.AgentMesh.Utilities.State.ToolStateManager import ToolStateManager

logger = logging.getLogger(__name__)

class DeployRacVmInput(BaseModel):
    # Network
    vcn_name: str = Field(default="rac-vcn", description="Name of the Virtual Cloud Network.")
    vcn_cidr: str = Field(default="10.0.0.0/16", description="CIDR block for the VCN.")
    public_subnet_cidr: str = Field(default="10.0.1.0/24", description="CIDR block for the public subnet (Client Network).")
    private_subnet_cidr: str = Field(default="10.0.2.0/24", description="CIDR block for the private subnet (Interconnect).")
    
    # Compute
    shape: str = Field(default="VM.Standard.E5.Flex", description="Compute shape for the RAC nodes.")
    ocpus: int = Field(default=2, description="Number of OCPUs per node.")
    memory_in_gbs: int = Field(default=8, description="Memory in GBs per node.")
    boot_volume_size_in_gbs: int = Field(default=50, description="Boot volume size in GBs per node.")
    image_ocid: str = Field(
        default="ocid1.image.oc1.ap-tokyo-1.aaaaaaaaqqgcviukagw5mxguu2tdxrgg2thsevsglkdixod22ea7su2efh7q", 
        description="OS Image OCID for the VMs."
    )
    ssh_public_key_path: Optional[str] = Field(default=None, description="Optional. Path or raw string for SSH public key. If blank, auto-resolves from UI settings.")
    availability_domain: str = Field(default="", description="Target Availability Domain.")

    # Storage
    local_block_volume_size: int = Field(default=50, description="Size of the local block volume (u01) per node in GBs.")
    shared_block_volume_size: int = Field(default=100, description="Size of the shared ASM block volume in GBs.")
    
    # Naming
    cluster_prefix: str = Field(default="racvm", description="Prefix for all RAC resources (nodes, vips, scans, volumes).")
    
    compartment_id: Optional[str] = Field(default=None, description="Optional compartment OCID. If left blank, falls back to saved credentials.")
    
    phase: str = Field(
        default="network", 
        description="The phase of the deployment. MUST run sequentially: 'network', 'compute', then 'storage_and_ips'."
    )

@tool(args_schema=DeployRacVmInput)
async def tf_deploy_rac_vm(
    vcn_name: str = "rac-vcn",
    vcn_cidr: str = "10.0.0.0/16",
    public_subnet_cidr: str = "10.0.1.0/24",
    private_subnet_cidr: str = "10.0.2.0/24",
    shape: str = "VM.Standard.E5.Flex",
    ocpus: int = 2,
    memory_in_gbs: int = 8,
    boot_volume_size_in_gbs: int = 50,
    image_ocid: str = "ocid1.image.oc1.ap-tokyo-1.aaaaaaaaqqgcviukagw5mxguu2tdxrgg2thsevsglkdixod22ea7su2efh7q",
    ssh_public_key_path: Optional[str] = None,
    availability_domain: str = "",
    local_block_volume_size: int = 50,
    shared_block_volume_size: int = 100,
    cluster_prefix: str = "racvm",
    compartment_id: Optional[str] = None,
    phase: str = "network",
    config: RunnableConfig = None,
    **kwargs
) -> str:
    """
    Automates the deployment of a 2-Node RAC Virtual Machine Architecture on OCI using Terraform in 3 separate phases.
    
    *** EXTREMELY IMPORTANT CONVERSATIONAL FLOW INSTRUCTIONS ***
    The deployment must be run in 3 phases sequentially to avoid hanging the chat.
    1. First call with phase="network". When it returns, notify the user that Network is done, and ask for confirmation to proceed to Compute.
    2. Second call with phase="compute". When it returns, notify the user that Compute is done, and ask for confirmation to proceed to Storage & IPs.
    3. Third call with phase="storage_and_ips". When it returns, the entire RAC is deployed.
    """
    try:
        # Resolve backend credentials
        creds = await resolve_oci_creds(config)
        env = os.environ.copy()
        
        target_compartment = compartment_id or creds.get("compartment_id") or creds.get("tenancy_id")
        
        node1 = f"{cluster_prefix}1"
        node2 = f"{cluster_prefix}2"
        
        resolved_ssh_key = ssh_public_key_path
        if not resolved_ssh_key:
            wf_config = await get_workflow_tool_config(config, "tf_deploy_rac_vm")
            resolved_ssh_key = wf_config.get("ssh_public_key")
        if not resolved_ssh_key:
            resolved_ssh_key = creds.get("ssh_public_key")
            
        if not resolved_ssh_key:
            return "Error: SSH Public Key is required but was not provided and could not be found in UI credentials or tool properties."
            
        temp_pub_path = None
        if "ssh-rsa" in resolved_ssh_key or "ssh-ed25519" in resolved_ssh_key or "ecdsa-sha2-nistp256" in resolved_ssh_key or "BEGIN" in resolved_ssh_key:
            temp_pub = tempfile.NamedTemporaryFile(mode="w", suffix=".pub", delete=False)
            temp_pub.write(resolved_ssh_key)
            temp_pub.close()
            temp_pub_path = temp_pub.name
            final_ssh_key_path = temp_pub_path
        else:
            final_ssh_key_path = resolved_ssh_key
        
        key_content = creds.get("private_key", "")
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
        temp_pem.write(key_content)
        temp_pem.close()

        tf_dir = os.path.join(os.path.dirname(__file__), "..", "2-TERRF_OCI_RAC")
        tf_root_dir = os.path.join(os.path.dirname(__file__), "..")

        resolved_ad = await resolve_availability_domain(availability_domain, tf_root_dir, env, {
            "tenancy_ocid": creds.get("tenancy_id"),
            "user_ocid": creds.get("user_id"),
            "fingerprint": creds.get("fingerprint"),
            "private_key_path": temp_pem.name,
            "region": creds.get("region"),
            "compartment_id": target_compartment
        })
        if not resolved_ad:
            return f"Error: Could not resolve availability domain '{availability_domain}' in OCI region."

        # Build the dynamic Terraform variables dictionary
        tfvars = {
            "tenancy_ocid": creds.get("tenancy_id"),
            "user_ocid": creds.get("user_id"),
            "fingerprint": creds.get("fingerprint"),
            "private_key_path": temp_pem.name,
            "region": creds.get("region"),
            "compartment_id": target_compartment,
            
            "vcn_name": vcn_name,
            "vcn_cidr": vcn_cidr,
            "public_subnet_name": f"{vcn_name}-public",
            "public_subnet_cidr": public_subnet_cidr,
            "private_subnet_name": f"{vcn_name}-private",
            "private_subnet_cidr": private_subnet_cidr,
            
            "availability_domain": resolved_ad,
            
            "instances": {
                node1: {
                    "availability_domain": resolved_ad,
                    "shape": shape,
                    "ocpus": ocpus,
                    "memory_in_gbs": memory_in_gbs,
                    "assign_public_ip": True,
                    "ssh_public_key_path": final_ssh_key_path,
                    "image_ocid": image_ocid,
                    "boot_volume_size_in_gbs": boot_volume_size_in_gbs,
                    "hostname_label": node1
                },
                node2: {
                    "availability_domain": resolved_ad,
                    "shape": shape,
                    "ocpus": ocpus,
                    "memory_in_gbs": memory_in_gbs,
                    "assign_public_ip": True,
                    "ssh_public_key_path": final_ssh_key_path,
                    "image_ocid": image_ocid,
                    "boot_volume_size_in_gbs": boot_volume_size_in_gbs,
                    "hostname_label": node2
                }
            },
            
            "vip_ips": {
                f"{node1}-vip": {"instance_name": node1},
                f"{node2}-vip": {"instance_name": node2}
            },
            
            "scan_ips": {
                f"{cluster_prefix}-scan1": {"instance_name": node1},
                f"{cluster_prefix}-scan2": {"instance_name": node2},
                f"{cluster_prefix}-scan3": {"instance_name": node1}
            },
            
            "local_block_volumes": {
                f"{node1}-u01": {"instance_name": node1, "size_in_gbs": local_block_volume_size},
                f"{node2}-u01": {"instance_name": node2, "size_in_gbs": local_block_volume_size}
            },
            
            "shared_block_volume": {
                "display_name": "rac-asm-shared",
                "size_in_gbs": shared_block_volume_size
            }
        }
        
        # We also need to map network_config since the tf module requires it. We can read it from the network.auto.tfvars.json if present, or provide a default hardcoded one.
        # To avoid duplicating a massive JSON object here, we will load it dynamically from the 2-TERRF_OCI_RAC folder.
        tf_dir = os.path.join(os.path.dirname(__file__), "..", "2-TERRF_OCI_RAC")
        
        network_auto_file = os.path.join(tf_dir, "network.auto.tfvars.json")
        if os.path.exists(network_auto_file):
            with open(network_auto_file, "r") as f:
                network_data = json.load(f)
                if "network_config" in network_data:
                    tfvars["network_config"] = network_data["network_config"]
        # Determine the target and load state based on the phase
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        tf_target = None
        state_content = None
        
        if phase == "network":
            tf_target = ["oci_core_subnet.public_subnet", "oci_core_subnet.private_subnet"]
            # No state to load for the first phase
        elif phase == "compute":
            tf_target = ["oci_core_instance.this"]
            state_content = await ToolStateManager.get_state(thread_id, "tf_deploy_rac_vm")
            if not state_content:
                return "Error: Could not find network state. Please run phase='network' first."
        elif phase == "storage_and_ips":
            tf_target = None # Full apply for remaining resources
            state_content = await ToolStateManager.get_state(thread_id, "tf_deploy_rac_vm")
            if not state_content:
                return "Error: Could not find compute state. Please run phase='compute' first."
        else:
            return f"Error: Unknown phase '{phase}'. Must be 'network', 'compute', or 'storage_and_ips'."
                    
        try:
            # Execute terraform apply in sandbox
            init_res, apply_res, final_state = await run_terraform_in_sandbox(
                tf_dir=tf_dir,
                env=env,
                tfvars=tfvars,
                target=tf_target,
                state_content=state_content
            )
        finally:
            if temp_pub_path and os.path.exists(temp_pub_path):
                try:
                    os.remove(temp_pub_path)
                except Exception:
                    pass
            if temp_pem and os.path.exists(temp_pem.name):
                try:
                    os.remove(temp_pem.name)
                except Exception:
                    pass
        
        if apply_res and apply_res.returncode == 0:
            if final_state:
                await ToolStateManager.save_state(
                    thread_id,
                    "tf_deploy_rac_vm",
                    final_state
                )
            if phase == "network":
                return "SUCCESS: Phase 1 (Network) complete! The VCN and Subnets have been deployed. Please confirm with the user before proceeding to Phase 2 (Compute)."
            elif phase == "compute":
                return "SUCCESS: Phase 2 (Compute) complete! The RAC Virtual Machines have been deployed. Please confirm with the user before proceeding to Phase 3 (Storage and IPs)."
            else:
                return "SUCCESS: Phase 3 (Storage and IPs) complete! The 2-Node RAC Virtual Machine Architecture has been successfully provisioned!"
        else:
            err = apply_res.stderr if apply_res else init_res.stderr if init_res else "Unknown error."
            try:
                with open(r"d:\InfraAgentic(Updated)\tf_rac_error.log", "w") as f:
                    f.write(err)
            except Exception:
                pass
            return f"Failed to provision RAC VM at phase '{phase}'. Error: {err}"
            
    except Exception as e:
        logger.exception("Failed to deploy RAC VM.")
        return f"Error: {str(e)}"

# --- Tool Registrations ---
def register_rac_tools():
    registry.register(
        tf_deploy_rac_vm,
        ToolMeta(
            name="tf_deploy_rac_vm",
            display_name="OCI Terraform: Deploy RAC VM",
            description="Interactively deploys a 2-Node RAC VM architecture.",
            category="Infrastructure",
            package="oci_terraform",
            version="1.0.0",
            configs=[
                ToolConfigMeta(
                    name="ssh_public_key",
                    type="string",
                    description="SSH Public Key (Path or Raw String). Used if global UI setting is empty.",
                    required=False
                )
            ]
        )
    )

register_rac_tools()
