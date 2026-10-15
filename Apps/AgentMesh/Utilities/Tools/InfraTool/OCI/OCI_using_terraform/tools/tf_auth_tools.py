import os
import json
import logging
import asyncio
import tempfile
import shutil
import subprocess
from typing import Optional, Tuple, Any, List
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

from Apps.AgentMesh.Utilities.State.ToolStateManager import ToolStateManager
from Apps.AgentMesh.Schema.ToolSchema import ToolMeta
from Apps.AgentMesh.Core.ToolRegistry import registry

logger = logging.getLogger(__name__)

# Absolute path to the root Terraform OCI directory (parent of tools/)
TF_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


async def resolve_oci_creds(config) -> dict:
    """
    Centralized OCI credential resolver.  Checks all sources in order:
      1. Thread-scoped state  (tf_oci_creds / oci_creds)
      2. default_thread state (fallback — auth agent always saves here)
      3. workflow_variables   (UI-configured credentials in workflow context)
    Returns a flat dict with keys: user_id, fingerprint, tenancy_id, region,
    private_key, compartment_id, ssh_public_key, ssh_private_key.
    """
    # --- extract configurable safely (config may be dict OR object) ---
    if isinstance(config, dict):
        configurable = config.get("configurable", {})
    else:
        configurable = getattr(config, "configurable", {}) if config else {}

    thread_id = configurable.get("thread_id") or "default_thread"
    context = configurable.get("context", {})

    # --- 1. thread-scoped state ---
    creds = (
        await ToolStateManager.get_state(thread_id, "tf_oci_creds")
        or await ToolStateManager.get_state(thread_id, "oci_creds")
    )

    # --- 2. default_thread fallback ---
    if not creds and thread_id != "default_thread":
        creds = (
            await ToolStateManager.get_state("default_thread", "tf_oci_creds")
            or await ToolStateManager.get_state("default_thread", "oci_creds")
        )

    creds = creds or {}

    # --- 3. fill blanks from workflow_variables (UI creds) ---
    wf_vars = context.get("workflow_variables", {}) if isinstance(context, dict) else {}
    KEY_MAP = {
        "user_id":        ["user_id", "user_ocid", "oci_user_ocid"],
        "fingerprint":    ["fingerprint", "oci_fingerprint"],
        "tenancy_id":     ["tenancy_id", "tenancy_ocid", "oci_tenancy_ocid"],
        "region":         ["region", "oci_region"],
        "private_key":    ["private_key", "oci_private_key", "api_key"],
        "compartment_id": ["compartment_id", "oci_compartment_id"],
        "ssh_public_key": ["ssh_public_key", "oci_ssh_public_key"],
        "ssh_private_key":["ssh_private_key", "oci_ssh_private_key"],
    }
    for canon, aliases in KEY_MAP.items():
        if not creds.get(canon):
            for alias in aliases:
                val = wf_vars.get(alias)
                if val:
                    creds[canon] = val
                    break

    logger.debug(
        f"resolve_oci_creds: thread={thread_id}, "
        f"has_private_key={bool(creds.get('private_key'))}, "
        f"has_user_id={bool(creds.get('user_id'))}, "
        f"wf_vars_keys={list(wf_vars.keys()) if wf_vars else '[]'}"
    )
    return creds


async def run_terraform_in_sandbox(
    tf_dir: str,
    env: dict,
    tfvars: dict,
    target: Any = None,
    destroy: bool = False,
    state_content: Optional[str] = None,
    import_target: Optional[str] = None,
    import_id: Optional[str] = None
) -> Tuple[subprocess.CompletedProcess, Optional[subprocess.CompletedProcess], Optional[str]]:
    import sys
    # Find terraform.exe in the same folder as Python interpreter, fallback to "terraform"
    python_dir = os.path.dirname(sys.executable)
    tf_cmd = os.path.join(python_dir, "terraform.exe")
    if not os.path.exists(tf_cmd):
        tf_cmd = "terraform"

    temp_dir = tempfile.mkdtemp(dir=os.path.dirname(tf_dir))
    try:
        # Copy the 4 root HCL files
        for file_name in ["provider.tf", "variables.tf", "main.tf", "outputs.tf"]:
            src_file = os.path.join(tf_dir, file_name)
            if os.path.exists(src_file):
                shutil.copy(src_file, temp_dir)
                
        # Copy modules directory if present
        modules_src = os.path.join(tf_dir, "modules")
        if os.path.exists(modules_src):
            shutil.copytree(modules_src, os.path.join(temp_dir, "modules"))
            
        # Copy/Symlink .terraform directory to avoid redownloading providers
        terraform_plugins_src = os.path.join(tf_dir, ".terraform")
        if os.path.exists(terraform_plugins_src):
            try:
                # Windows support: try creating a directory symlink/junction
                if os.name == 'nt':
                    import subprocess
                    subprocess.run(
                        ["cmd", "/c", "mklink", "/J", os.path.join(temp_dir, ".terraform").replace("/", "\\"), terraform_plugins_src.replace("/", "\\")],
                        capture_output=True,
                        check=True
                    )
                else:
                    os.symlink(terraform_plugins_src, os.path.join(temp_dir, ".terraform"), target_is_directory=True)
            except Exception:
                shutil.copytree(terraform_plugins_src, os.path.join(temp_dir, ".terraform"))
                
        lock_file = os.path.join(tf_dir, ".terraform.lock.hcl")
        if os.path.exists(lock_file):
            shutil.copy(lock_file, temp_dir)
            
        # Write state content if provided
        if state_content:
            state_path = os.path.join(temp_dir, "terraform.tfstate")
            with open(state_path, "w") as f:
                f.write(state_content)
            
        # Copy the private key locally into the sandbox to avoid Windows path/space/formatting issues
        if tfvars.get("private_key_path"):
            original_key_path = tfvars["private_key_path"]
            if os.path.exists(original_key_path):
                local_key_path = os.path.join(temp_dir, "private_key.pem")
                shutil.copy(original_key_path, local_key_path)
                tfvars = tfvars.copy()
                tfvars["private_key_path"] = os.path.abspath(local_key_path).replace("\\", "/")

        # Write variables payload as JSON tfvars (Terraform natively parses this)
        tfvars_path = os.path.join(temp_dir, "terraform.tfvars.json")
        with open(tfvars_path, "w") as f:
            json.dump(tfvars, f)
            
        # 1. Run terraform init
        init_process = await asyncio.to_thread(
            subprocess.run,
            [tf_cmd, "init"],
            cwd=temp_dir,
            env=env,
            capture_output=True,
            text=True
        )
        if init_process.returncode != 0:
            return init_process, None, None

        # 1b. Run terraform import if requested and no existing state was written
        if not state_content and import_target and import_id:
            import_cmd = [tf_cmd, "import", f"-var-file={tfvars_path}", import_target, import_id]
            import_process = await asyncio.to_thread(
                subprocess.run,
                import_cmd,
                cwd=temp_dir,
                env=env,
                capture_output=True,
                text=True
            )
            # Log import output for debugging but don't fail immediately if already imported or similar
            logger.info(f"Terraform import returned code {import_process.returncode}. Output: {import_process.stdout} Error: {import_process.stderr}")

        # 2. Run terraform apply or destroy
        cmd = [tf_cmd, "apply", "-auto-approve", "-input=false"]
        if destroy:
            cmd = [tf_cmd, "destroy", "-auto-approve", "-input=false"]
        if target:
            if isinstance(target, list):
                for t in target:
                    cmd.append(f"-target={t}")
            else:
                cmd.append(f"-target={target}")
            
        run_process = await asyncio.to_thread(
            subprocess.run,
            cmd,
            cwd=temp_dir,
            env=env,
            capture_output=True,
            text=True
        )
        
        # Read final state content
        final_state = None
        final_state_path = os.path.join(temp_dir, "terraform.tfstate")
        if os.path.exists(final_state_path):
            with open(final_state_path, "r") as f:
                final_state = f.read()
        
        # 3. Read outputs if successful
        output_process = None
        if not destroy and run_process.returncode == 0:
            output_cmd = [tf_cmd, "output", "-json"]
            output_process = await asyncio.to_thread(
                subprocess.run,
                output_cmd,
                cwd=temp_dir,
                env=env,
                capture_output=True,
                text=True
            )
            
        return run_process, output_process, final_state
    finally:
        # Clean up temp folder completely
        try:
            if os.name == 'nt':
                junction_path = os.path.join(temp_dir, ".terraform")
                if os.path.exists(junction_path):
                    import subprocess
                    subprocess.run(["cmd", "/c", "rmdir", junction_path.replace("/", "\\")], capture_output=True)
            shutil.rmtree(temp_dir)
        except Exception:
            pass


async def run_tf_query(
    config: RunnableConfig,
    target: str,
    filter_vars: Optional[dict] = None
) -> Optional[Any]:
    """
    Centralized, clean OCI query execution using Terraform data sources.
    Resolves OCI credentials, sets up PEM file, runs init + plan/apply, and returns parsed JSON.
    """
    try:
        creds = await resolve_oci_creds(config)
        key_content = creds.get("private_key", "")
        if not key_content:
            logger.error("OCI credentials or private key missing.")
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

        temp_pem = tempfile.NamedTemporaryFile(mode="w", suffix=".pem", delete=False, newline="\n")
        try:
            temp_pem.write(key_content)
            temp_pem.close()

            tfvars = {
                "tenancy_ocid": creds.get("tenancy_id"),
                "user_ocid": creds.get("user_id"),
                "fingerprint": creds.get("fingerprint"),
                "private_key_path": temp_pem.name,
                "region": creds.get("region"),
                "compartment_id": filter_vars.get("compartment_id") if filter_vars and "compartment_id" in filter_vars else (creds.get("compartment_id") or creds.get("tenancy_id"))
            }
            if filter_vars:
                for k, v in filter_vars.items():
                    if k != "compartment_id":
                        tfvars[k] = v

            apply_res, output_res, final_state = await run_terraform_in_sandbox(
                tf_dir=TF_DIR,
                env=os.environ.copy(),
                tfvars=tfvars,
                target=target
            )
        finally:
            if os.path.exists(temp_pem.name):
                try:
                    os.remove(temp_pem.name)
                except Exception:
                    pass

        if apply_res.returncode == 0 and output_res and output_res.returncode == 0:
            try:
                outputs = json.loads(output_res.stdout)
                target_to_output = {
                    "data.oci_identity_compartments.all_compartments": "compartments_json"
                }
                output_name = target_to_output.get(target, target.split(".")[-1])
                val = outputs.get(output_name, {}).get("value")
                if val is not None:
                    return val
            except Exception as e:
                logger.error(f"Failed to parse tf output: {e}")
                
        # Fallback to state extraction
        if final_state:
            try:
                # Try-except import to prevent circular import loop
                from Apps.AgentMesh.Utilities.Tools.InfraTool.OCI.OCI_using_terraform.tools.tf_compute_tools import extract_data_from_state
                target_to_output = {
                    "data.oci_identity_compartments.all_compartments": "compartments_json"
                }
                output_name = target_to_output.get(target, target.split(".")[-1])
                # In state content, oci_identity_compartments usually maps to compartments array
                state_key_map = {
                    "compartments_json": "compartments"
                }
                state_key = state_key_map.get(output_name, output_name)
                val = extract_data_from_state(final_state, target.split(".")[1], state_key, None)
                if val:
                    return val
            except Exception:
                pass
                
        return None
    except Exception as e:
        logger.exception(f"Error executing run_tf_query: {e}")
        return None


def normalize_name(name: str) -> str:
    if not name:
        return ""
    return name.lower().replace(" ", "").replace("-", "").replace("_", "")


async def resolve_compartment_id(
    compartment_input: str,
    *args,
    config: RunnableConfig = None,
    **kwargs
) -> Optional[str]:
    if not compartment_input:
        return None
    compartment_input = compartment_input.strip()
    if compartment_input.startswith("ocid1.compartment.oc1..") or compartment_input.startswith("ocid1.tenancy.oc1.."):
        return compartment_input

    # Check if config was passed positionally or if we got old arguments
    for arg in args:
        if hasattr(arg, "configurable") or (isinstance(arg, dict) and "configurable" in arg):
            config = arg
            break

    try:
        creds = await resolve_oci_creds(config)
        key_content = creds.get("private_key", "")
        if not key_content:
            logger.error("OCI credentials or private key missing in resolve_compartment_id.")
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
            res = await run_oci_cli([
                "identity", "compartment", "list",
                "--compartment-id", creds.get('tenancy_id'),
                "--compartment-id-in-subtree", "true",
                "--all"
            ], config_path)
            compartments = res.get("data", [])
        finally:
            shutil.rmtree(temp_dir)

        if compartments:
            search_norm = normalize_name(compartment_input)
            # 1. Exact match
            for comp in compartments:
                comp_name = comp.get("name", "")
                if normalize_name(comp_name) == search_norm:
                    return comp.get("id")
            # 2. Fuzzy match
            for comp in compartments:
                comp_name = comp.get("name", "")
                if search_norm in normalize_name(comp_name) or normalize_name(comp_name) in search_norm:
                    return comp.get("id")
    except Exception as e:
        logger.error(f"Error in resolve_compartment_id via OCI CLI: {e}")
    return None


async def resolve_subnet_id(subnet_input: str, compartment_id: str, config: RunnableConfig = None) -> Optional[str]:
    if not subnet_input:
        return None
    subnet_input = subnet_input.strip()
    if subnet_input.startswith("ocid1.subnet.oc1.") or subnet_input.startswith("ocid1.subnet.oc1.."):
        return subnet_input

    subnets = await run_tf_query(config, "data.oci_core_subnets.all_subnets", {"compartment_id": compartment_id})
    if subnets:
        search_norm = normalize_name(subnet_input)
        for sub in subnets:
            if normalize_name(sub.get("display_name", "")) == search_norm:
                return sub.get("id")
        for sub in subnets:
            sub_name = sub.get("display_name", "")
            if search_norm in normalize_name(sub_name) or normalize_name(sub_name) in search_norm:
                return sub.get("id")
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


async def resolve_vm_ocid(vm_input: str, compartment_id: str, config: RunnableConfig = None) -> Optional[str]:
    if not vm_input:
        return None
    vm_input = vm_input.strip()
    if vm_input.startswith("ocid1.instance.oc1.") or vm_input.startswith("ocid1.instance.oc1.."):
        return vm_input

    try:
        creds = await resolve_oci_creds(config)
        key_content = creds.get("private_key", "")
        if not key_content:
            logger.error("OCI credentials or private key missing in resolve_vm_ocid.")
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
                    logger.error(f"Failed to list subtree compartments in resolve_vm_ocid: {e}")

            vms = []
            async def list_vms_in_comp(cid):
                try:
                    res = await run_oci_cli([
                        "compute", "instance", "list",
                        "--compartment-id", cid
                    ], config_path)
                    return res.get("data", [])
                except Exception:
                    return []

            results = await asyncio.gather(*(list_vms_in_comp(cid) for cid in comps_to_search))
            for r in results:
                vms.extend(r)
        finally:
            shutil.rmtree(temp_dir)

        if vms:
            search_norm = normalize_name(vm_input)
            for vm in vms:
                display_name = vm.get("display-name", vm.get("display_name", ""))
                if normalize_name(display_name) == search_norm:
                    return vm.get("id")
            for vm in vms:
                display_name = vm.get("display-name", vm.get("display_name", ""))
                if search_norm in normalize_name(display_name) or normalize_name(display_name) in search_norm:
                    return vm.get("id")
    except Exception as e:
        logger.error(f"Error in resolve_vm_ocid via OCI CLI: {e}")
    return None


async def resolve_volume_ocid(volume_input: str, compartment_id: str, config: RunnableConfig = None) -> Optional[str]:
    if not volume_input:
        return None
    volume_input = volume_input.strip()
    if volume_input.startswith("ocid1.volume.oc1.") or volume_input.startswith("ocid1.volume.oc1.."):
        return volume_input

    try:
        creds = await resolve_oci_creds(config)
        key_content = creds.get("private_key", "")
        if not key_content:
            logger.error("OCI credentials or private key missing in resolve_volume_ocid.")
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
                    logger.error(f"Failed to list subtree compartments in resolve_volume_ocid: {e}")

            vols = []
            async def list_vols_in_comp(cid):
                try:
                    res = await run_oci_cli([
                        "bv", "volume", "list",
                        "--compartment-id", cid
                    ], config_path)
                    return res.get("data", [])
                except Exception:
                    return []

            results = await asyncio.gather(*(list_vols_in_comp(cid) for cid in comps_to_search))
            for r in results:
                vols.extend(r)
        finally:
            shutil.rmtree(temp_dir)

        if vols:
            search_norm = normalize_name(volume_input)
            for vol in vols:
                display_name = vol.get("display-name", vol.get("display_name", ""))
                if normalize_name(display_name) == search_norm:
                    return vol.get("id")
            for vol in vols:
                display_name = vol.get("display-name", vol.get("display_name", ""))
                if search_norm in normalize_name(display_name) or normalize_name(display_name) in search_norm:
                    return vol.get("id")
    except Exception as e:
        logger.error(f"Error in resolve_volume_ocid via OCI CLI: {e}")
    return None


@tool
async def tf_configure_auth(
    user_id: str,
    fingerprint: str,
    tenancy_id: str,
    region: str,
    private_key: str,
    compartment_id: Optional[str] = None,
    ssh_public_key: Optional[str] = None,
    ssh_private_key: Optional[str] = None,
    config: RunnableConfig = None
) -> str:
    """
    Manual override for OCI credentials used by Terraform.
    This saves the credentials into the agent's memory so Terraform can use them.
    """
    try:
        _cfg = config.get("configurable", {}) if isinstance(config, dict) else (getattr(config, "configurable", {}) if config else {})
        thread_id = _cfg.get("thread_id") or "default_thread"
        
        creds = {
            "user_id": user_id,
            "fingerprint": fingerprint,
            "tenancy_id": tenancy_id,
            "region": region,
            "private_key": private_key,
            "compartment_id": compartment_id,
            "ssh_public_key": ssh_public_key,
            "ssh_private_key": ssh_private_key
        }
        
        await ToolStateManager.save_state(thread_id, "tf_oci_creds", creds)
        return "Terraform OCI credentials configured successfully."
    except Exception as e:
        logger.exception("Failed to save Terraform OCI credentials")
        return f"Error saving Terraform OCI credentials: {str(e)}"

def extract_data_from_state(state_str: str, resource_type: str, resource_name: str, attribute_name: str) -> Optional[Any]:
    if not state_str:
        return None
    try:
        state = json.loads(state_str)
        for res in state.get("resources", []):
            if res.get("type") == resource_type and res.get("name") == resource_name:
                for inst in res.get("instances", []):
                    attr = inst.get("attributes", {})
                    if attribute_name in attr:
                        return attr[attribute_name]
    except Exception:
        pass
    return None

@tool
async def tf_list_compartments(config: RunnableConfig = None) -> str:
    """
    You MUST call this tool whenever the user asks to see or list compartments. 
    It fetches the real OCI compartments using Terraform.
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

        env = os.environ.copy()
        key_content = config_data.get("key_content", "")
        
        if not key_content:
            return "Error: Private key is empty or missing. Please configure your OCI credentials using the Terraform Configure Authentication tool in the UI first."
            
        # Ensure the private key has correct PEM formatting (64 chars per line) for Terraform
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

        # Create temporary .pem key file for local execution
        temp_pem = tempfile.NamedTemporaryFile(mode="w", suffix=".pem", delete=False, newline="\n")
        try:
            temp_pem.write(key_content)
            temp_pem.close()
            
            # Prepare tfvars map
            tfvars = {
                "tenancy_ocid": config_data.get("tenancy"),
                "user_ocid": config_data.get("user"),
                "fingerprint": config_data.get("fingerprint"),
                "private_key_path": temp_pem.name,
                "region": config_data.get("region"),
                "compartment_id": config_data.get("compartment_id") or config_data.get("tenancy")
            }
            
            # Run in sandbox targeting only the compartments query
            apply_res, output_res, final_state = await run_terraform_in_sandbox(
                tf_dir=TF_DIR,
                env=env,
                tfvars=tfvars,
                target="data.oci_identity_compartments.all_compartments"
            )
        finally:
            if os.path.exists(temp_pem.name):
                os.remove(temp_pem.name)

        if apply_res.returncode != 0:
            logger.error(f"Terraform run failed: {apply_res.stderr}")
            return f"Error executing Terraform: {apply_res.stderr}"
            
        compartments = None
        if output_res and output_res.returncode == 0:
            try:
                outputs = json.loads(output_res.stdout)
                compartments = outputs.get("compartments_json", {}).get("value", [])
            except Exception:
                pass

        if not compartments and final_state:
            # Fallback: Parse directly from state content
            compartments = extract_data_from_state(
                final_state,
                "oci_identity_compartments",
                "all_compartments",
                "compartments"
            )

        if not compartments:
            return "Error: Could not retrieve compartment data from Terraform output or state."

        result_lines = ["Available OCI Compartments (via Terraform):\n"]
        for comp in compartments:
            name = comp.get("name", "Unknown")
            ocid = comp.get("id", "Unknown")
            state = comp.get("lifecycle_state", comp.get("state", ""))
            
            if state == "ACTIVE":
                result_lines.append(f"- {name} (OCID: {ocid})")
                
        return "\n".join(result_lines)
    except Exception as e:
        logger.exception("Failed to run terraform list compartments")
        return f"Error executing Terraform tool: {str(e)}"

@tool
async def tf_provision_compartment(
    compartment_name: str,
    parent_compartment: Optional[str] = None,
    description: Optional[str] = None,
    config: RunnableConfig = None
) -> str:
    """
    Provisions a new child compartment in OCI using Terraform.
    Inputs: compartment_name, parent_compartment (name or OCID, optional, default: tenancy root), description (optional).
    """
    try:
        # Local import to prevent circular import loops
        from Apps.AgentMesh.Utilities.Tools.InfraTool.OCI.OCI_using_terraform.tools.tf_vcn_tools import resolve_compartment_id

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
        
        # Default parent to tenancy/root if not provided
        parent_compartment_input = parent_compartment or config_data.get("compartment_id") or config_data.get("tenancy") or ""

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

            # Resolve parent compartment name to OCID
            resolved_parent_id = await resolve_compartment_id(
                parent_compartment_input,
                TF_DIR, env, {
                    "tenancy_ocid": config_data.get("tenancy"),
                    "user_ocid": config_data.get("user"),
                    "fingerprint": config_data.get("fingerprint"),
                    "private_key_path": temp_pem.name,
                    "region": config_data.get("region"),
                    "compartment_id": config_data.get("tenancy")
                }
            )
            if not resolved_parent_id:
                return f"Error: Could not resolve parent compartment '{parent_compartment_input}' to a valid OCI OCID."

            tfvars = {
                "tenancy_ocid": config_data.get("tenancy"),
                "user_ocid": config_data.get("user"),
                "fingerprint": config_data.get("fingerprint"),
                "private_key_path": temp_pem.name,
                "region": config_data.get("region"),
                "compartment_id": resolved_parent_id,
                "compartments": {
                    compartment_name: {
                        "parent_compartment_id": resolved_parent_id,
                        "description": description or f"Child compartment '{compartment_name}' created by AgentMesh"
                    }
                }
            }

            state_key = f"tf_state_compartment_{compartment_name}"
            existing_state = await ToolStateManager.get_state(thread_id, state_key)
            if not existing_state:
                # Direct SQL check fallback
                from Database.Database import AsyncSessionLocal
                from Database.Models import ToolStateModel
                from sqlalchemy import select
                from Apps.AgentMesh.Utilities.State.ToolStateManager import process_state_data

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
                target="module.compartment",
                state_content=existing_state
            )

            # Save state
            if final_state:
                await ToolStateManager.save_state(thread_id, state_key, final_state)

        finally:
            if os.path.exists(temp_pem.name):
                os.remove(temp_pem.name)

        if apply_res.returncode != 0:
            logger.error(f"Terraform compartment creation failed: {apply_res.stderr}")
            return f"Error executing Terraform Compartment creation: {apply_res.stderr}"

        if not output_res or output_res.returncode != 0:
            return "Error: Could not retrieve compartment details from Terraform output."

        outputs = json.loads(output_res.stdout)
        comp_ids = outputs.get("compartment_ids", {}).get("value", {})
        comp_ocid = comp_ids.get(compartment_name, "Unknown")

        return f"Successfully provisioned child Compartment '{compartment_name}' (OCID: {comp_ocid}) under Parent: {resolved_parent_id}"
    except Exception as e:
        logger.exception("Failed to provision compartment")
        return f"Error provisioning compartment: {str(e)}"

# Registering Compartment Tools
registry.register(
    tf_provision_compartment,
    ToolMeta(
        name="tf_provision_compartment",
        display_name="OCI Terraform: Create Compartment",
        description="Creates a new child compartment inside OCI tenancy using Terraform.",
        category="Infrastructure",
        package="oci_terraform"
    )
)

registry.register(
    tf_configure_auth,
    ToolMeta(
        name="tf_configure_auth",
        display_name="OCI Terraform: Configure Authentication",
        description="Manual override for OCI credentials used by Terraform tools.",
        category="Infrastructure",
        package="oci_terraform",
        version="1.0.0",
        allow_duplicate=False,
        configs=[
            {"name": "user_id", "type": "string", "description": "OCI User OCID", "required": True},
            {"name": "fingerprint", "type": "string", "description": "OCI API Key Fingerprint", "required": True},
            {"name": "tenancy_id", "type": "string", "description": "OCI Tenancy OCID", "required": True},
            {"name": "region", "type": "string", "description": "OCI Region", "required": True},
            {"name": "private_key", "type": "string", "description": "OCI Private Key Content", "required": True},
            {"name": "compartment_id", "type": "string", "description": "Default Compartment OCID", "required": False},
            {"name": "ssh_public_key", "type": "string", "description": "SSH public key content", "required": False},
            {"name": "ssh_private_key", "type": "string", "description": "SSH private key content", "required": False}
        ]
    )
)

registry.register(
    tf_list_compartments,
    ToolMeta(
        name="tf_list_compartments",
        display_name="OCI Terraform: List Compartments",
        description="You MUST call this tool whenever the user asks to see or list compartments. It fetches the real OCI compartments using Terraform.",
        category="Infrastructure",
        package="oci_terraform"
    )
)
