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
async def tf_provision_security_list(
    security_list_name: str,
    vcn_id: str,
    ingress_rules: Optional[List[Dict[str, Any]]] = None,
    egress_rules: Optional[List[Dict[str, Any]]] = None,
    compartment_id: Optional[str] = None,
    config: RunnableConfig = None
) -> str:
    """
    Provisions a new Security List inside a VCN in the OCI compartment using Terraform.
    CRITICAL: You must ask the user which OCI compartment they want to list/manage resources in, and ensure it is configured.
    CRITICAL SCHEMA RULE for ingress_rules/egress_rules: 
    When providing tcp_options or udp_options, provide 'min' and 'max' directly inside the options object. 
    DO NOT nest them inside a 'destination_port_range' key.
    Correct Example: "tcp_options": {"min": 22, "max": 22}
    Incorrect Example: "tcp_options": {"destination_port_range": {"min": 22, "max": 22}}
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

            tfvars = {
                "tenancy_ocid": config_data.get("tenancy"),
                "user_ocid": config_data.get("user"),
                "fingerprint": config_data.get("fingerprint"),
                "private_key_path": temp_pem.name,
                "region": config_data.get("region"),
                "compartment_id": resolved_comp_id,
                "security_lists": {
                    security_list_name: {
                        "vcn_id": vcn_id,
                        "ingress_rules": ingress_rules or [],
                        "egress_rules": egress_rules or []
                    }
                }
            }
            
            # Load existing state if any (for updates/idempotency)
            state_key = f"tf_state_sec_list_{security_list_name}"
            existing_state = await ToolStateManager.get_state(thread_id, state_key)
            
            if not existing_state:
                # Fallback: search for this state globally across threads
                async with AsyncSessionLocal() as session:
                    query = select(ToolStateModel).where(ToolStateModel.tool_key == state_key)
                    result = await session.execute(query)
                    tool_state = result.scalars().first()
                    if tool_state:
                        existing_state = process_state_data(tool_state.state, mode="decrypt")
                        logger.info(f"Loaded existing Security List state for {security_list_name} from global DB fallback")
            
            # Execute apply sandbox
            apply_res, output_res, final_state = await run_terraform_in_sandbox(
                tf_dir=TF_DIR,
                env=env,
                tfvars=tfvars,
                target="module.security_list",
                state_content=existing_state
            )
            
            # Save updated state
            if final_state:
                await ToolStateManager.save_state(thread_id, state_key, final_state)
                
        finally:
            if os.path.exists(temp_pem.name):
                os.remove(temp_pem.name)

        if apply_res.returncode != 0:
            logger.error(f"Terraform provision Security List failed: {apply_res.stderr}")
            return f"Error executing Terraform Security List deployment: {apply_res.stderr}"
            
        if not output_res or output_res.returncode != 0:
            return "Error: Could not retrieve Security List output details from Terraform."

        outputs = json.loads(output_res.stdout)
        sec_list_ids = outputs.get("security_list_ids", {}).get("value", {})
        sec_list_ocid = sec_list_ids.get(security_list_name, "Unknown")
        
        return f"Successfully provisioned Security List '{security_list_name}' (OCID: {sec_list_ocid}) inside VCN: {vcn_id}"
    except Exception as e:
        logger.exception("Failed to provision Security List")
        return f"Error provisioning Security List: {str(e)}"

@tool
async def tf_update_security_list(
    security_list_name: str,
    vcn_id: str,
    ingress_rules: Optional[List[Dict[str, Any]]] = None,
    egress_rules: Optional[List[Dict[str, Any]]] = None,
    compartment_id: Optional[str] = None,
    config: RunnableConfig = None
) -> str:
    """
    Updates an existing Security List's ingress and egress rules inside OCI using Terraform.
    Pass the complete lists of ingress and egress rules as you want them to be set.
    CRITICAL SCHEMA RULE for ingress_rules/egress_rules: 
    When providing tcp_options or udp_options, provide 'min' and 'max' directly inside the options object. 
    DO NOT nest them inside a 'destination_port_range' key.
    Correct Example: "tcp_options": {"min": 22, "max": 22}
    Incorrect Example: "tcp_options": {"destination_port_range": {"min": 22, "max": 22}}
    """
    # Updating in Terraform is identical to provisioning with updated values.
    # The sandbox run automatically diffs and applies the updates in place.
    return await tf_provision_security_list.coroutine(
        security_list_name=security_list_name,
        vcn_id=vcn_id,
        ingress_rules=ingress_rules,
        egress_rules=egress_rules,
        compartment_id=compartment_id,
        config=config
    )

@tool
async def tf_destroy_security_list(
    security_list_name: str,
    compartment_id: Optional[str] = None,
    config: RunnableConfig = None
) -> str:
    """
    Destroys the specified Security List using Terraform.
    If the Security List was created outside Terraform, it will automatically look up its OCID, import it, and destroy it.
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
                "security_lists": {
                    security_list_name: {
                        "vcn_id": "dummy_vcn_id",
                        "ingress_rules": [],
                        "egress_rules": []
                    }
                }
            }
            
            # Load existing state
            state_key = f"tf_state_sec_list_{security_list_name}"
            existing_state = await ToolStateManager.get_state(thread_id, state_key)
            
            if not existing_state:
                # Fallback 1: search for this state globally across threads
                async with AsyncSessionLocal() as session:
                    query = select(ToolStateModel).where(ToolStateModel.tool_key == state_key)
                    result = await session.execute(query)
                    tool_state = result.scalars().first()
                    if tool_state:
                        existing_state = process_state_data(tool_state.state, mode="decrypt")
                        logger.info(f"Loaded existing Security List state for {security_list_name} from global DB fallback")
            
            # Fallback 2: If still no state, query OCI to find the untracked Security List's OCID
            sec_list_ocid = None
            if not existing_state:
                logger.info(f"Security List '{security_list_name}' state not found in database. Looking up OCID in OCI...")
                
                apply_res, output_res, _ = await run_terraform_in_sandbox(
                    tf_dir=TF_DIR,
                    env=env,
                    tfvars=tfvars,
                    target="data.oci_core_security_lists.all_security_lists"
                )
                
                if apply_res.returncode == 0 and output_res and output_res.returncode == 0:
                    outputs = json.loads(output_res.stdout)
                    all_sec_lists = outputs.get("all_security_lists", {}).get("value", [])
                    for s in all_sec_lists:
                        if s.get("display_name") == security_list_name:
                            sec_list_ocid = s.get("id")
                            tfvars["security_lists"][security_list_name]["vcn_id"] = s.get("vcn_id", "dummy_vcn_id")
                            break
                            
            # Case A: We have a tracked state file. Use standard destroy.
            if existing_state:
                destroy_res, _, final_state = await run_terraform_in_sandbox(
                    tf_dir=TF_DIR,
                    env=env,
                    tfvars=tfvars,
                    target="module.security_list",
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
                    return f"Error executing Terraform Security List destruction: {destroy_res.stderr}"
                    
                return f"Successfully destroyed Security List '{security_list_name}' using Terraform."
                
            # Case B: Untracked Security List. Run temporary import + destroy.
            elif sec_list_ocid:
                logger.info(f"Found Security List '{security_list_name}' in OCI (OCID: {sec_list_ocid}). Running auto-import and destroy...")
                
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
                        [tf_cmd, "import", f'module.security_list.oci_core_security_list.this["{security_list_name}"]', sec_list_ocid],
                        cwd=temp_dir, env=env, capture_output=True, text=True
                    )
                    if import_process.returncode != 0:
                        return f"Error importing Security List for destruction: {import_process.stderr}"
 
                    # Destroy
                    destroy_process = await asyncio.to_thread(
                        subprocess.run,
                        [tf_cmd, "destroy", "-auto-approve", "-target", f'module.security_list.oci_core_security_list.this["{security_list_name}"]'],
                        cwd=temp_dir, env=env, capture_output=True, text=True
                    )
                    if destroy_process.returncode != 0:
                        return f"Error executing Terraform Security List destruction: {destroy_process.stderr}"
                        
                    return f"Successfully imported and destroyed untracked Security List '{security_list_name}' (OCID: {sec_list_ocid}) using Terraform."
                finally:
                    try:
                        shutil.rmtree(temp_dir)
                    except Exception:
                        pass
            else:
                return f"Error: Security List '{security_list_name}' was not found in the OCI compartment (untracked) and has no local state."
                
        finally:
            if os.path.exists(temp_pem.name):
                os.remove(temp_pem.name)
    except Exception as e:
        logger.exception("Failed to destroy Security List")
        return f"Error destroying Security List: {str(e)}"

@tool
async def tf_list_security_lists(
    compartment_id: str,
    vcn_id: Optional[str] = None,
    security_list_name: Optional[str] = None,
    detail: bool = False,
    config: RunnableConfig = None
) -> str:
    """
    Lists all Security Lists inside the specified OCI compartment, optionally filtered by a VCN ID.
    CRITICAL: You must ask the user which OCI compartment they want to list Security Lists in before running this tool.
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
                target="data.oci_core_security_lists.all_security_lists"
            )
        finally:
            if os.path.exists(temp_pem.name):
                os.remove(temp_pem.name)
 
        if apply_res.returncode != 0:
            return f"Error executing Terraform: {apply_res.stderr}"
            
        sec_lists = None
        if output_res and output_res.returncode == 0:
            try:
                outputs = json.loads(output_res.stdout)
                sec_lists = outputs.get("all_security_lists", {}).get("value", [])
            except Exception:
                pass
 
        if not sec_lists and final_state:
            # Fallback: Parse directly from state content
            sec_lists = extract_data_from_state(
                final_state,
                "oci_core_security_lists",
                "all_security_lists",
                "security_lists"
            )
 
        if not sec_lists:
            return f"No Security Lists found or could not retrieve Security List data from compartment: {target_compartment}"
 
        if security_list_name:
            sec_lists = [s for s in sec_lists if security_list_name.lower() in s.get("display_name", "").lower() or security_list_name == s.get("id")]
            if not sec_lists:
                return f"No Security List matching '{security_list_name}' was found in compartment: {target_compartment}"

        result_lines = [f"Found {len(sec_lists)} Security List(s) in compartment (via Terraform):\n"]
        for s in sec_lists:
            name = s.get("display_name", "N/A")
            ocid = s.get("id", "N/A")
            state = s.get("lifecycle_state", s.get("state", "N/A"))
            v_id = s.get("vcn_id", "N/A")
            
            in_rules = s.get("ingress_security_rules", [])
            eg_rules = s.get("egress_security_rules", [])
            
            if detail or security_list_name:
                rules_details = []
                rules_details.append(f"  VCN OCID: `{v_id}`")
                rules_details.append(f"  OCID: `{ocid}`")
                rules_details.append(f"  Ingress Rules ({len(in_rules)}):")
                for r in in_rules:
                    proto = r.get("protocol", "all")
                    src = r.get("source", "N/A")
                    udp_tcp = r.get("tcp_options") or r.get("udp_options") or {}
                    ports = f"{udp_tcp.get('destination_port_range', {}).get('min', 'all')}-{udp_tcp.get('destination_port_range', {}).get('max', 'all')}" if udp_tcp else "all"
                    rules_details.append(f"    - Protocol: `{proto}` | Source: `{src}` | Destination Ports: `{ports}`")
                
                rules_details.append(f"  Egress Rules ({len(eg_rules)}):")
                for r in eg_rules:
                    proto = r.get("protocol", "all")
                    dest = r.get("destination", "N/A")
                    udp_tcp = r.get("tcp_options") or r.get("udp_options") or {}
                    ports = f"{udp_tcp.get('destination_port_range', {}).get('min', 'all')}-{udp_tcp.get('destination_port_range', {}).get('max', 'all')}" if udp_tcp else "all"
                    rules_details.append(f"    - Protocol: `{proto}` | Destination: `{dest}` | Destination Ports: `{ports}`")
                
                result_lines.append(f"- **{name}** (State: `{state}`)\n" + "\n".join(rules_details))
            else:
                result_lines.append(
                    f"- **{name}** (State: `{state}`)\n"
                    f"  VCN OCID: `{v_id}`\n"
                    f"  Rules: {len(in_rules)} Ingress, {len(eg_rules)} Egress\n"
                    f"  OCID: `{ocid}`"
                )
            
        return "\n".join(result_lines)
    except Exception as e:
        logger.exception("Failed to list Security Lists")
        return f"Error listing Security Lists: {str(e)}"

# Registering Security List Tools
registry.register(
    tf_provision_security_list,
    ToolMeta(
        name="tf_provision_security_list",
        display_name="OCI Terraform: Create Security List",
        description="Creates a new Security List inside a VCN using Terraform.",
        category="Infrastructure",
        package="oci_terraform"
    )
)

registry.register(
    tf_update_security_list,
    ToolMeta(
        name="tf_update_security_list",
        display_name="OCI Terraform: Update Security List",
        description="Updates an existing Security List's rules inside a VCN using Terraform.",
        category="Infrastructure",
        package="oci_terraform"
    )
)

registry.register(
    tf_destroy_security_list,
    ToolMeta(
        name="tf_destroy_security_list",
        display_name="OCI Terraform: Delete Security List",
        description="Deletes/destroys an existing Security List using Terraform. Automatically imports and destroys untracked Security Lists by name.",
        category="Infrastructure",
        package="oci_terraform"
    )
)

registry.register(
    tf_list_security_lists,
    ToolMeta(
        name="tf_list_security_lists",
        display_name="OCI Terraform: List Security Lists",
        description="Lists all Security Lists in a compartment, optionally filtered by a VCN ID.",
        category="Infrastructure",
        package="oci_terraform"
    )
)