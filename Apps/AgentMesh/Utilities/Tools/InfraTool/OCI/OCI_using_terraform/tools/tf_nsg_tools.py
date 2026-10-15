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
import base64
from Apps.AgentMesh.Utilities.Tools.InfraTool.OCI.OCI_using_terraform.tools.tf_auth_tools import run_terraform_in_sandbox, extract_data_from_state, resolve_oci_creds, run_tf_query, resolve_vm_ocid
from Apps.AgentMesh.Utilities.Tools.InfraTool.OCI.OCI_using_terraform.tools.tf_vcn_tools import resolve_compartment_id
from Apps.AgentMesh.Utilities.Tools.InfraTool.OCI.OCI_using_terraform.tools.tf_compute_tools import tf_ssh_executor


logger = logging.getLogger(__name__)

# Absolute path to the root Terraform OCI directory
TF_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

@tool
async def tf_provision_nsg(
    nsg_name: str,
    vcn_id: str,
    rules: Optional[List[Dict[str, Any]]] = None,
    compartment_id: Optional[str] = None,
    config: RunnableConfig = None
) -> str:
    """
    Provisions a new Network Security Group (NSG) with optional rules inside a VCN using Terraform.
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

            # Structure rules
            nsg_rules_map = {}
            if rules:
                for idx, r in enumerate(rules):
                    rule_key = f"{nsg_name}_rule_{idx}"
                    nsg_rules_map[rule_key] = {
                        "nsg_name": nsg_name,
                        "direction": r.get("direction", "INGRESS"),
                        "protocol": str(r.get("protocol", "all")),
                        "source": r.get("source"),
                        "source_type": r.get("source_type", "CIDR_BLOCK") if r.get("source") else None,
                        "destination": r.get("destination"),
                        "destination_type": r.get("destination_type", "CIDR_BLOCK") if r.get("destination") else None,
                        "stateless": r.get("stateless", False),
                        "tcp_options": r.get("tcp_options"),
                        "udp_options": r.get("udp_options")
                    }

            tfvars = {
                "tenancy_ocid": config_data.get("tenancy"),
                "user_ocid": config_data.get("user"),
                "fingerprint": config_data.get("fingerprint"),
                "private_key_path": temp_pem.name,
                "region": config_data.get("region"),
                "compartment_id": resolved_comp_id,
                "nsgs": {
                    nsg_name: {
                        "vcn_id": vcn_id
                    }
                },
                "nsg_rules": nsg_rules_map
            }
            
            # Load existing state if any (for updates/idempotency)
            state_key = f"tf_state_nsg_{nsg_name}"
            existing_state = await ToolStateManager.get_state(thread_id, state_key)
            
            if not existing_state:
                # Fallback: search for this state globally across threads
                async with AsyncSessionLocal() as session:
                    query = select(ToolStateModel).where(ToolStateModel.tool_key == state_key)
                    result = await session.execute(query)
                    tool_state = result.scalars().first()
                    if tool_state:
                        existing_state = process_state_data(tool_state.state, mode="decrypt")
                        logger.info(f"Loaded existing NSG state for {nsg_name} from global DB fallback")
            
            # Execute apply sandbox
            apply_res, output_res, final_state = await run_terraform_in_sandbox(
                tf_dir=TF_DIR,
                env=env,
                tfvars=tfvars,
                target="module.nsg",
                state_content=existing_state
            )
            
            # Save updated state
            if final_state:
                await ToolStateManager.save_state(thread_id, state_key, final_state)
                
        finally:
            if os.path.exists(temp_pem.name):
                os.remove(temp_pem.name)

        if apply_res.returncode != 0:
            logger.error(f"Terraform provision NSG failed: {apply_res.stderr}")
            return f"Error executing Terraform NSG deployment: {apply_res.stderr}"
            
        if not output_res or output_res.returncode != 0:
            return "Error: Could not retrieve NSG output details from Terraform."

        outputs = json.loads(output_res.stdout)
        nsg_ids = outputs.get("nsg_ids", {}).get("value", {})
        nsg_ocid = nsg_ids.get(nsg_name, "Unknown")
        
        return f"Successfully provisioned Network Security Group '{nsg_name}' (OCID: {nsg_ocid}) inside VCN: {vcn_id}"
    except Exception as e:
        logger.exception("Failed to provision NSG")
        return f"Error provisioning NSG: {str(e)}"



@tool
async def tf_destroy_nsg(
    nsg_name: str,
    compartment_id: Optional[str] = None,
    config: RunnableConfig = None
) -> str:
    """
    Destroys the specified Network Security Group (NSG) and its rules using Terraform.
    If the NSG was created outside Terraform, it will automatically look up its OCID, import it, and destroy it.
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
                "nsgs": {
                    nsg_name: {
                        "vcn_id": "dummy_vcn_id"
                    }
                }
            }
            
            # Load existing state
            state_key = f"tf_state_nsg_{nsg_name}"
            existing_state = await ToolStateManager.get_state(thread_id, state_key)
            
            if not existing_state:
                # Fallback 1: search for this state globally across threads
                async with AsyncSessionLocal() as session:
                    query = select(ToolStateModel).where(ToolStateModel.tool_key == state_key)
                    result = await session.execute(query)
                    tool_state = result.scalars().first()
                    if tool_state:
                        existing_state = process_state_data(tool_state.state, mode="decrypt")
                        logger.info(f"Loaded existing NSG state for {nsg_name} from global DB fallback")
            
            # Fallback 2: If still no state, query OCI to find the untracked NSG's OCID
            nsg_ocid = None
            if not existing_state:
                logger.info(f"NSG '{nsg_name}' state not found in database. Looking up OCID in OCI...")
                
                apply_res, output_res, _ = await run_terraform_in_sandbox(
                    tf_dir=TF_DIR,
                    env=env,
                    tfvars=tfvars,
                    target="data.oci_core_network_security_groups.all_nsgs"
                )
                
                if apply_res.returncode == 0 and output_res and output_res.returncode == 0:
                    outputs = json.loads(output_res.stdout)
                    all_nsgs = outputs.get("all_nsgs", {}).get("value", [])
                    for n in all_nsgs:
                        if n.get("display_name") == nsg_name:
                            nsg_ocid = n.get("id")
                            tfvars["nsgs"][nsg_name]["vcn_id"] = n.get("vcn_id", "dummy_vcn_id")
                            break
                            
            # Case A: We have a tracked state file. Use standard destroy.
            if existing_state:
                destroy_res, _, final_state = await run_terraform_in_sandbox(
                    tf_dir=TF_DIR,
                    env=env,
                    tfvars=tfvars,
                    target="module.nsg",
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
                    return f"Error executing Terraform NSG destruction: {destroy_res.stderr}"
                    
                return f"Successfully destroyed Network Security Group '{nsg_name}' using Terraform."
                
            # Case B: Untracked NSG. Run temporary import + destroy.
            elif nsg_ocid:
                logger.info(f"Found Network Security Group '{nsg_name}' in OCI (OCID: {nsg_ocid}). Running auto-import and destroy...")
                
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
                        [tf_cmd, "import", f'module.nsg.oci_core_network_security_group.this["{nsg_name}"]', nsg_ocid],
                        cwd=temp_dir, env=env, capture_output=True, text=True
                    )
                    if import_process.returncode != 0:
                        return f"Error importing Network Security Group for destruction: {import_process.stderr}"
 
                    # Destroy
                    destroy_process = await asyncio.to_thread(
                        subprocess.run,
                        [tf_cmd, "destroy", "-auto-approve", "-target", f'module.nsg.oci_core_network_security_group.this["{nsg_name}"]'],
                        cwd=temp_dir, env=env, capture_output=True, text=True
                    )
                    if destroy_process.returncode != 0:
                        return f"Error executing Terraform NSG destruction: {destroy_process.stderr}"
                        
                    return f"Successfully imported and destroyed untracked Network Security Group '{nsg_name}' (OCID: {nsg_ocid}) using Terraform."
                finally:
                    try:
                        shutil.rmtree(temp_dir)
                    except Exception:
                        pass
            else:
                return f"Error: Network Security Group '{nsg_name}' was not found in the OCI compartment (untracked) and has no local state."
                
        finally:
            if os.path.exists(temp_pem.name):
                os.remove(temp_pem.name)
    except Exception as e:
        logger.exception("Failed to destroy Network Security Group")
        return f"Error destroying Network Security Group: {str(e)}"

@tool
async def tf_list_nsgs(
    compartment_id: str,
    vcn_id: Optional[str] = None,
    nsg_name: Optional[str] = None,
    detail: bool = False,
    config: RunnableConfig = None
) -> str:
    """
    Lists all Network Security Groups (NSGs) inside the specified OCI compartment, optionally filtered by a VCN ID.
    CRITICAL: You must ask the user which OCI compartment they want to list NSGs in before running this tool.
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
                "vcn_id": vcn_id or ""
            }
 
            apply_res, output_res, final_state = await run_terraform_in_sandbox(
                tf_dir=TF_DIR,
                env=env,
                tfvars=tfvars,
                target="data.oci_core_network_security_groups.all_nsgs"
            )
        finally:
            if os.path.exists(temp_pem.name):
                os.remove(temp_pem.name)
 
        if apply_res.returncode != 0:
            return f"Error executing Terraform: {apply_res.stderr}"
            
        nsgs = None
        if output_res and output_res.returncode == 0:
            try:
                outputs = json.loads(output_res.stdout)
                nsgs = outputs.get("all_nsgs", {}).get("value", [])
            except Exception:
                pass
 
        if not nsgs and final_state:
            # Fallback: Parse directly from state content
            nsgs = extract_data_from_state(
                final_state,
                "oci_core_network_security_groups",
                "all_nsgs",
                "network_security_groups"
            )
 
        if not nsgs:
            return f"No Network Security Groups found or could not retrieve NSG data from compartment: {target_compartment}"
 
        if nsg_name:
            nsgs = [n for n in nsgs if nsg_name.lower() in n.get("display_name", "").lower() or nsg_name == n.get("id")]
            if not nsgs:
                return f"No NSG matching '{nsg_name}' was found in compartment: {target_compartment}"

        result_lines = [f"Found {len(nsgs)} Network Security Group(s) in compartment (via Terraform):\n"]
        for n in nsgs:
            name = n.get("display_name", "N/A")
            ocid = n.get("id", "N/A")
            state = n.get("lifecycle_state", n.get("state", "N/A"))
            v_id = n.get("vcn_id", "N/A")
            
            if detail or nsg_name:
                result_lines.append(
                    f"- **{name}** (State: `{state}`)\n"
                    f"  VCN OCID: `{v_id}`\n"
                    f"  Time Created: `{n.get('time_created', 'N/A')}`\n"
                    f"  OCID: `{ocid}`"
                )
            else:
                result_lines.append(
                    f"- **{name}** (State: `{state}`)\n"
                    f"  VCN OCID: `{v_id}`\n"
                    f"  OCID: `{ocid}`"
                )
            
        return "\n".join(result_lines)
    except Exception as e:
        logger.exception("Failed to list Network Security Groups")
        return f"Error listing Network Security Groups: {str(e)}"

# Registering NSG Tools
registry.register(
    tf_provision_nsg,
    ToolMeta(
        name="tf_provision_nsg",
        display_name="OCI Terraform: Create Network Security Group",
        description="Creates a new Network Security Group inside a VCN using Terraform.",
        category="Infrastructure",
        package="oci_terraform"
    )
)


registry.register(
    tf_destroy_nsg,
    ToolMeta(
        name="tf_destroy_nsg",
        display_name="OCI Terraform: Delete Network Security Group",
        description="Deletes/destroys an existing Network Security Group using Terraform. Automatically imports and destroys untracked NSGs by name.",
        category="Infrastructure",
        package="oci_terraform"
    )
)

registry.register(
    tf_list_nsgs,
    ToolMeta(
        name="tf_list_nsgs",
        display_name="OCI Terraform: List Network Security Groups",
        description="Lists all Network Security Groups in a compartment, optionally filtered by a VCN ID.",
        category="Infrastructure",
        package="oci_terraform"
    )
)

async def resolve_nsg_ocid(nsg_input: str, compartment_id: str, config: RunnableConfig = None) -> Optional[str]:
    if not nsg_input:
        return None
    nsg_input = nsg_input.strip()
    if nsg_input.startswith("ocid1.networksecuritygroup.oc1."):
        return nsg_input

    try:
        creds = await resolve_oci_creds(config)
        key_content = creds.get("private_key", "")
        if not key_content:
            logger.error("OCI credentials or private key missing in resolve_nsg_ocid.")
            return None
            
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

        temp_dir = tempfile.mkdtemp()
        config_path = os.path.join(temp_dir, "config")
        key_path = os.path.join(temp_dir, "key.pem")
        
        with open(key_path, "w") as f:
            f.write(key_content)
            
        config_content = f"""[DEFAULT]
user={creds.get('user_id')}
fingerprint={creds.get('fingerprint')}
key_file={key_path.replace('\\', '/')}
tenancy={creds.get('tenancy_id')}
region={creds.get('region')}
"""
        with open(config_path, "w") as f:
            f.write(config_content)

        try:
            comps_to_search = [compartment_id]
            if compartment_id.startswith("ocid1.tenancy."):
                try:
                    comp_res = await run_oci_cli([
                        "identity", "compartment", "list",
                        "--compartment-id", compartment_id,
                        "--compartment-id-in-subtree", "true",
                        "--all"
                    ], config_path)
                    for c in comp_res.get("data", []):
                        if c.get("lifecycle-state") == "ACTIVE" or c.get("lifecycle_state") == "ACTIVE":
                            comps_to_search.append(c["id"])
                except Exception as e:
                    logger.error(f"Failed to list subtree compartments in resolve_nsg_ocid: {e}")

            nsgs = []
            async def list_nsgs_in_comp(cid):
                try:
                    res = await run_oci_cli([
                        "network", "nsg", "list",
                        "--compartment-id", cid
                    ], config_path)
                    return res.get("data", [])
                except Exception:
                    return []

            results = await asyncio.gather(*(list_nsgs_in_comp(cid) for cid in comps_to_search))
            for r in results:
                nsgs.extend(r)
        finally:
            shutil.rmtree(temp_dir)

        if nsgs:
            search_norm = nsg_input.strip().lower().replace("-", "").replace("_", "").replace(" ", "")
            for n in nsgs:
                nsg_norm = n.get("display-name", n.get("display_name", "")).strip().lower().replace("-", "").replace("_", "").replace(" ", "")
                if nsg_norm == search_norm or nsg_input == n.get("id"):
                    return n.get("id")
    except Exception as e:
        logger.error(f"Error in resolve_nsg_ocid via OCI CLI: {e}")
    return None

async def run_oci_cli(args: List[str], config_path: str) -> dict:
    import sys
    python_dir = os.path.dirname(sys.executable)
    oci_cmd = os.path.join(python_dir, "oci.exe")
    if not os.path.exists(oci_cmd):
        oci_cmd = "oci"

    env = os.environ.copy()
    env["OCI_CLI_SUPPRESS_FILE_PERMISSIONS_WARNING"] = "True"
    env["SUPPRESS_LABEL_WARNING"] = "True"

    cmd = [oci_cmd] + args + ["--config-file", config_path]
    res = await asyncio.to_thread(
        subprocess.run,
        cmd,
        capture_output=True,
        text=True,
        shell=False,
        env=env
    )
    if res.returncode != 0:
        raise ValueError(f"OCI CLI command failed: {res.stderr}")
    return json.loads(res.stdout) if res.stdout.strip() else {}

class AttachNsgToVmInput(BaseModel):
    vm_name: str = Field(description="The display name or OCID of the VM compute instance.")
    nsg_name: str = Field(description="The display name or OCID of the Network Security Group (NSG) to attach.")
    compartment_id: Optional[str] = Field(default=None, description="The name or OCID of the target compartment.")
    vm_public_ip: Optional[str] = Field(default=None, description="DO NOT ask the user for this. Leave it None/blank as it is automatically resolved from backend auth settings.")
    os_username: Optional[str] = Field(default=None, description="DO NOT ask the user for this. Leave it None/blank as it is automatically resolved from backend auth settings.")
    ssh_private_key: Optional[str] = Field(default=None, description="DO NOT ask the user for this. Leave it None/blank as it is automatically resolved from backend auth settings.", repr=False)

@tool(args_schema=AttachNsgToVmInput)
async def tf_attach_nsg_to_vm(
    vm_name: str,
    nsg_name: str,
    compartment_id: Optional[str] = None,
    vm_public_ip: Optional[str] = None,
    os_username: Optional[str] = None,
    ssh_private_key: Optional[str] = None,
    config: RunnableConfig = None,
    **kwargs
) -> str:
    """
    Attaches a Network Security Group (NSG) to a VM compute instance's primary VNIC using an SSH-based bash script.
    
    *** DO NOT ask the user for SSH keys or IP address details in the chat. ***
    All credentials and connection configurations are resolved securely in the backend.
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

            # Resolve NSG OCID
            nsg_ocid = await resolve_nsg_ocid(nsg_name, resolved_comp_id, config)
            if not nsg_ocid:
                return f"Error: Network Security Group with name '{nsg_name}' not found in compartment."

            # Resolve VM OCID
            vm_ocid = vm_name
            if not vm_name.startswith("ocid1.instance."):
                vm_ocid = await resolve_vm_ocid(vm_name, resolved_comp_id, config)
                if not vm_ocid:
                    return f"Error: Could not resolve VM name '{vm_name}' to a valid OCI instance OCID."

            # Execute locally via OCI CLI config
            temp_dir = tempfile.mkdtemp()
            config_path = os.path.join(temp_dir, "config")
            key_path = os.path.join(temp_dir, "key.pem")
            
            with open(key_path, "w") as f:
                f.write(key_content)
                
            config_content = f"""[DEFAULT]
user={config_data['user']}
fingerprint={config_data['fingerprint']}
key_file={key_path.replace('\\', '/')}
tenancy={config_data['tenancy']}
region={config_data['region']}
"""
            with open(config_path, "w") as f:
                f.write(config_content)
                
            try:
                # Get VNIC attachments
                vnics_res = await run_oci_cli([
                    "compute", "vnic-attachment", "list",
                    "--compartment-id", resolved_comp_id,
                    "--instance-id", vm_ocid
                ], config_path)
                
                vnic_attachments = vnics_res.get("data", [])
                if not vnic_attachments:
                    return f"Error: No VNIC attachments found for VM instance '{vm_name}'."
                vnic_id = vnic_attachments[0]["vnic-id"]
                
                # Get current VNIC details
                vnic_res = await run_oci_cli([
                    "network", "vnic", "get",
                    "--vnic-id", vnic_id
                ], config_path)
                
                current_nsgs = vnic_res.get("data", {}).get("nsg-ids") or []
                
                # Attach: Add to list if not present
                updated_nsgs = list(set(current_nsgs + [nsg_ocid]))
                
                # Update VNIC NSGs
                nsgs_str = json.dumps(updated_nsgs)
                await run_oci_cli([
                    "network", "vnic", "update",
                    "--vnic-id", vnic_id,
                    "--nsg-ids", nsgs_str,
                    "--force"
                ], config_path)
                
                return f"Successfully attached Network Security Group '{nsg_name}' to VM '{vm_name}' using OCI CLI."
                
            finally:
                shutil.rmtree(temp_dir)

        finally:
            if os.path.exists(temp_pem.name):
                os.remove(temp_pem.name)
    except Exception as e:
        logger.exception("Failed to attach NSG to VM using Terraform")
        return f"Error attaching NSG to VM: {str(e)}"

class DetachNsgFromVmInput(BaseModel):
    vm_name: str = Field(description="The display name or OCID of the VM compute instance.")
    nsg_name: str = Field(description="The display name or OCID of the Network Security Group (NSG) to detach.")
    compartment_id: Optional[str] = Field(default=None, description="The name or OCID of the target compartment.")
    vm_public_ip: Optional[str] = Field(default=None, description="DO NOT ask the user for this. Leave it None/blank as it is automatically resolved from backend auth settings.")
    os_username: Optional[str] = Field(default=None, description="DO NOT ask the user for this. Leave it None/blank as it is automatically resolved from backend auth settings.")
    ssh_private_key: Optional[str] = Field(default=None, description="DO NOT ask the user for this. Leave it None/blank as it is automatically resolved from backend auth settings.", repr=False)

@tool(args_schema=DetachNsgFromVmInput)
async def tf_detach_nsg_from_vm(
    vm_name: str,
    nsg_name: str,
    compartment_id: Optional[str] = None,
    vm_public_ip: Optional[str] = None,
    os_username: Optional[str] = None,
    ssh_private_key: Optional[str] = None,
    config: RunnableConfig = None,
    **kwargs
) -> str:
    """
    Detaches a Network Security Group (NSG) from a VM compute instance's VNIC using an SSH-based bash script.
    
    *** DO NOT ask the user for SSH keys or IP address details in the chat. ***
    All credentials and connection configurations are resolved securely in the backend.
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

            # Resolve NSG OCID
            nsg_ocid = await resolve_nsg_ocid(nsg_name, resolved_comp_id, config)
            if not nsg_ocid:
                return f"Error: Network Security Group with name '{nsg_name}' not found in compartment."

            # Resolve VM OCID
            vm_ocid = vm_name
            if not vm_name.startswith("ocid1.instance."):
                vm_ocid = await resolve_vm_ocid(vm_name, resolved_comp_id, config)
                if not vm_ocid:
                    return f"Error: Could not resolve VM name '{vm_name}' to a valid OCI instance OCID."

            # Execute locally via OCI CLI config
            temp_dir = tempfile.mkdtemp()
            config_path = os.path.join(temp_dir, "config")
            key_path = os.path.join(temp_dir, "key.pem")
            
            with open(key_path, "w") as f:
                f.write(key_content)
                
            config_content = f"""[DEFAULT]
user={config_data['user']}
fingerprint={config_data['fingerprint']}
key_file={key_path.replace('\\', '/')}
tenancy={config_data['tenancy']}
region={config_data['region']}
"""
            with open(config_path, "w") as f:
                f.write(config_content)
                
            try:
                # Get VNIC attachments
                vnics_res = await run_oci_cli([
                    "compute", "vnic-attachment", "list",
                    "--compartment-id", resolved_comp_id,
                    "--instance-id", vm_ocid
                ], config_path)
                
                vnic_attachments = vnics_res.get("data", [])
                if not vnic_attachments:
                    return f"Error: No VNIC attachments found for VM instance '{vm_name}'."
                vnic_id = vnic_attachments[0]["vnic-id"]
                
                # Get current VNIC details
                vnic_res = await run_oci_cli([
                    "network", "vnic", "get",
                    "--vnic-id", vnic_id
                ], config_path)
                
                current_nsgs = vnic_res.get("data", {}).get("nsg-ids") or []
                
                # Detach: Remove from list if present
                updated_nsgs = [n for n in current_nsgs if n != nsg_ocid]
                
                # Update VNIC NSGs
                nsgs_str = json.dumps(updated_nsgs)
                await run_oci_cli([
                    "network", "vnic", "update",
                    "--vnic-id", vnic_id,
                    "--nsg-ids", nsgs_str,
                    "--force"
                ], config_path)
                
                return f"Successfully detached Network Security Group '{nsg_name}' from VM '{vm_name}' using OCI CLI."
                
            finally:
                shutil.rmtree(temp_dir)

        finally:
            if os.path.exists(temp_pem.name):
                os.remove(temp_pem.name)
    except Exception as e:
        logger.exception("Failed to detach NSG from VM using Terraform")
        return f"Error detaching NSG from VM: {str(e)}"

# Registering NSG Tools
registry.register(
    tf_provision_nsg,
    ToolMeta(
        name="tf_provision_nsg",
        display_name="OCI Terraform: Create Network Security Group",
        description="Creates a new Network Security Group inside a VCN using Terraform.",
        category="Infrastructure",
        package="oci_terraform"
    )
)


registry.register(
    tf_destroy_nsg,
    ToolMeta(
        name="tf_destroy_nsg",
        display_name="OCI Terraform: Delete Network Security Group",
        description="Deletes/destroys an existing Network Security Group using Terraform. Automatically imports and destroys untracked NSGs by name.",
        category="Infrastructure",
        package="oci_terraform"
    )
)

registry.register(
    tf_list_nsgs,
    ToolMeta(
        name="tf_list_nsgs",
        display_name="OCI Terraform: List Network Security Groups",
        description="Lists all Network Security Groups in a compartment, optionally filtered by a VCN ID.",
        category="Infrastructure",
        package="oci_terraform"
    )
)

registry.register(
    tf_attach_nsg_to_vm,
    ToolMeta(
        name="tf_attach_nsg_to_vm",
        display_name="OCI Terraform: Attach NSG to VM (SSH)",
        description="Attaches a Network Security Group (NSG) to a VM compute instance's primary VNIC using an SSH-based OCI CLI runner.",
        category="Infrastructure",
        package="oci_terraform"
    )
)

registry.register(
    tf_detach_nsg_from_vm,
    ToolMeta(
        name="tf_detach_nsg_from_vm",
        display_name="OCI Terraform: Detach NSG from VM (SSH)",
        description="Detaches a Network Security Group (NSG) from a VM compute instance's VNIC using an SSH-based OCI CLI runner.",
        category="Infrastructure",
        package="oci_terraform"
    )
)
