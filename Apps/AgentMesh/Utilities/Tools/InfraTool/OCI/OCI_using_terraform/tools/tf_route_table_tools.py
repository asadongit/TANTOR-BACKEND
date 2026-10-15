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
async def tf_provision_route_table(
    route_table_name: str,
    vcn_id: str,
    route_rules: Optional[List[Dict[str, Any]]] = None,
    compartment_id: Optional[str] = None,
    config: RunnableConfig = None
) -> str:
    """
    Provisions a new Route Table with custom routing rules inside a VCN using Terraform.
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
            rules_list = []
            if route_rules:
                for r in route_rules:
                    rules_list.append({
                        "network_entity_id": r.get("network_entity_id"),
                        "destination": r.get("destination", "0.0.0.0/0"),
                        "destination_type": r.get("destination_type", "CIDR_BLOCK")
                    })

            tfvars = {
                "tenancy_ocid": config_data.get("tenancy"),
                "user_ocid": config_data.get("user"),
                "fingerprint": config_data.get("fingerprint"),
                "private_key_path": temp_pem.name,
                "region": config_data.get("region"),
                "compartment_id": resolved_comp_id,
                "route_tables": {
                    route_table_name: {
                        "vcn_id": vcn_id,
                        "route_rules": rules_list
                    }
                }
            }
            
            # Load existing state if any (for updates/idempotency)
            state_key = f"tf_state_route_table_{route_table_name}"
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
                target="module.route_table",
                state_content=existing_state
            )
            
            # Save updated state
            if final_state:
                await ToolStateManager.save_state(thread_id, state_key, final_state)
                
        finally:
            if os.path.exists(temp_pem.name):
                os.remove(temp_pem.name)

        if apply_res.returncode != 0:
            logger.error(f"Terraform Route Table failed: {apply_res.stderr}")
            return f"Error executing Terraform Route Table deployment: {apply_res.stderr}"
            
        if not output_res or output_res.returncode != 0:
            return "Error: Could not retrieve Route Table details from Terraform."

        outputs = json.loads(output_res.stdout)
        rt_ids = outputs.get("route_table_ids", {}).get("value", {})
        rt_ocid = rt_ids.get(route_table_name, "Unknown")
        
        return f"Successfully provisioned Route Table '{route_table_name}' (OCID: {rt_ocid}) inside VCN: {vcn_id}"
    except Exception as e:
        logger.exception("Failed to provision Route Table")
        return f"Error provisioning Route Table: {str(e)}"

@tool
async def tf_update_route_table(
    route_table_name: str,
    vcn_id: str,
    route_rules: Optional[List[Dict[str, Any]]] = None,
    compartment_id: Optional[str] = None,
    config: RunnableConfig = None
) -> str:
    """
    Updates an existing Route Table's routing rules inside OCI using Terraform.
    Pass the complete list of route rules as you want them to be set.
    """
    return await tf_provision_route_table.coroutine(
        route_table_name=route_table_name,
        vcn_id=vcn_id,
        route_rules=route_rules,
        compartment_id=compartment_id,
        config=config
    )

@tool
async def tf_destroy_route_table(
    route_table_name: str,
    compartment_id: Optional[str] = None,
    config: RunnableConfig = None
) -> str:
    """
    Destroys the specified Route Table using Terraform.
    If the Route Table was created outside Terraform, it will automatically look up its OCID, import it, and destroy it.
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
                "route_tables": {
                    route_table_name: {
                        "vcn_id": "dummy_vcn_id"
                    }
                }
            }
            
            # Load existing state
            state_key = f"tf_state_route_table_{route_table_name}"
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
            rt_ocid = None
            if not existing_state:
                logger.info(f"Route Table '{route_table_name}' state not found in database. Looking up OCID in OCI...")
                
                apply_res, output_res, _ = await run_terraform_in_sandbox(
                    tf_dir=TF_DIR,
                    env=env,
                    tfvars=tfvars,
                    target="data.oci_core_route_tables.all_route_tables"
                )
                
                if apply_res.returncode == 0 and output_res and output_res.returncode == 0:
                    outputs = json.loads(output_res.stdout)
                    all_rts = outputs.get("all_route_tables", {}).get("value", [])
                    for r in all_rts:
                        if r.get("display_name") == route_table_name:
                            rt_ocid = r.get("id")
                            tfvars["route_tables"][route_table_name]["vcn_id"] = r.get("vcn_id", "dummy_vcn_id")
                            break
                            
            # Case A: Tracked state
            if existing_state:
                destroy_res, _, final_state = await run_terraform_in_sandbox(
                    tf_dir=TF_DIR,
                    env=env,
                    tfvars=tfvars,
                    target="module.route_table",
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
                    return f"Error executing Terraform Route Table destruction: {destroy_res.stderr}"
                return f"Successfully destroyed Route Table '{route_table_name}' using Terraform."
                
            # Case B: Untracked
            elif rt_ocid:
                logger.info(f"Found Route Table '{route_table_name}' in OCI (OCID: {rt_ocid}). Running auto-import and destroy...")
                
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
                        [tf_cmd, "import", f'module.route_table.oci_core_route_table.this["{route_table_name}"]', rt_ocid],
                        cwd=temp_dir, env=env, capture_output=True, text=True
                    )
                    if import_process.returncode != 0:
                        return f"Error importing Route Table for destruction: {import_process.stderr}"
 
                    # Destroy
                    destroy_process = await asyncio.to_thread(
                        subprocess.run,
                        [tf_cmd, "destroy", "-auto-approve", "-target", f'module.route_table.oci_core_route_table.this["{route_table_name}"]'],
                        cwd=temp_dir, env=env, capture_output=True, text=True
                    )
                    if destroy_process.returncode != 0:
                        return f"Error executing Terraform Route Table destruction: {destroy_process.stderr}"
                        
                    return f"Successfully imported and destroyed untracked Route Table '{route_table_name}' (OCID: {rt_ocid})."
                finally:
                    try:
                        shutil.rmtree(temp_dir)
                    except Exception:
                        pass
            else:
                return f"Error: Route Table '{route_table_name}' was not found in OCI (untracked) and has no local state."
        finally:
            if os.path.exists(temp_pem.name):
                os.remove(temp_pem.name)
    except Exception as e:
        logger.exception("Failed to destroy Route Table")
        return f"Error destroying Route Table: {str(e)}"

@tool
async def tf_list_route_tables(
    compartment_id: str,
    vcn_id: Optional[str] = None,
    route_table_name: Optional[str] = None,
    detail: bool = False,
    config: RunnableConfig = None
) -> str:
    """
    Lists all Route Tables inside the specified OCI compartment, optionally filtered by a VCN ID.
    CRITICAL: You must ask the user which OCI compartment they want to list Route Tables in before running this tool.
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
                target="data.oci_core_route_tables.all_route_tables"
            )
        finally:
            if os.path.exists(temp_pem.name):
                os.remove(temp_pem.name)
 
        if apply_res.returncode != 0:
            return f"Error executing Terraform: {apply_res.stderr}"
            
        rts = None
        if output_res and output_res.returncode == 0:
            try:
                outputs = json.loads(output_res.stdout)
                rts = outputs.get("all_route_tables", {}).get("value", [])
            except Exception:
                pass
 
        if not rts and final_state:
            rts = extract_data_from_state(
                final_state,
                "oci_core_route_tables",
                "all_route_tables",
                "route_tables"
            )
 
        if not rts:
            return f"No Route Tables found in compartment: {target_compartment}"
 
        if route_table_name:
            rts = [r for r in rts if route_table_name.lower() in r.get("display_name", "").lower() or route_table_name == r.get("id")]
            if not rts:
                return f"No Route Table matching '{route_table_name}' was found in compartment: {target_compartment}"

        result_lines = [f"Found {len(rts)} Route Table(s) in compartment (via Terraform):\n"]
        for r in rts:
            name = r.get("display_name", "N/A")
            ocid = r.get("id", "N/A")
            state = r.get("lifecycle_state", r.get("state", "N/A"))
            v_id = r.get("vcn_id", "N/A")
            rules = r.get("route_rules", [])
            
            if detail or route_table_name:
                rules_details = []
                rules_details.append(f"  VCN OCID: `{v_id}`")
                rules_details.append(f"  OCID: `{ocid}`")
                rules_details.append(f"  Routing Rules ({len(rules)}):")
                for rl in rules:
                    dest = rl.get("destination", "N/A")
                    dest_type = rl.get("destination_type", "CIDR_BLOCK")
                    target = rl.get("network_entity_id", "N/A")
                    rules_details.append(f"    - Target: `{target}` | Destination: `{dest} ({dest_type})`")
                result_lines.append(f"- **{name}** (State: `{state}`)\n" + "\n".join(rules_details))
            else:
                result_lines.append(
                    f"- **{name}** (State: `{state}`)\n"
                    f"  VCN OCID: `{v_id}`\n"
                    f"  Rules: {len(rules)} rule(s) configured\n"
                    f"  OCID: `{ocid}`"
                )
            
        return "\n".join(result_lines)
    except Exception as e:
        logger.exception("Failed to list Route Tables")
        return f"Error listing Route Tables: {str(e)}"

# Registering Route Table Tools
registry.register(
    tf_provision_route_table,
    ToolMeta(
        name="tf_provision_route_table",
        display_name="OCI Terraform: Create Route Table",
        description="Creates a new Route Table with custom routing rules inside a VCN using Terraform.",
        category="Infrastructure",
        package="oci_terraform"
    )
)

registry.register(
    tf_update_route_table,
    ToolMeta(
        name="tf_update_route_table",
        display_name="OCI Terraform: Update Route Table",
        description="Updates an existing Route Table's routing rules inside a VCN using Terraform.",
        category="Infrastructure",
        package="oci_terraform"
    )
)

registry.register(
    tf_destroy_route_table,
    ToolMeta(
        name="tf_destroy_route_table",
        display_name="OCI Terraform: Delete Route Table",
        description="Deletes/destroys an existing Route Table using Terraform. Automatically imports and destroys untracked Route Tables by name.",
        category="Infrastructure",
        package="oci_terraform"
    )
)

registry.register(
    tf_list_route_tables,
    ToolMeta(
        name="tf_list_route_tables",
        display_name="OCI Terraform: List Route Tables",
        description="Lists all Route Tables in a compartment, optionally filtered by a VCN ID.",
        category="Infrastructure",
        package="oci_terraform"
    )
)