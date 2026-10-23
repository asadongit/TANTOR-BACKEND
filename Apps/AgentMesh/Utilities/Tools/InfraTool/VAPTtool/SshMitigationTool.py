import paramiko
import logging
import json
from langchain_core.tools import tool
from Apps.AgentMesh.Schema.ToolSchema import ToolMeta
from Apps.AgentMesh.Core.ToolRegistry import registry

logger = logging.getLogger(__name__)

import asyncio
import io
from langchain_core.runnables import RunnableConfig

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

@tool
async def execute_ssh_mitigation(
    host: str, 
    username: str, 
    script_content: str, 
    config: RunnableConfig = None,
    **kwargs
) -> str:
    """
    Executes a mitigation bash script on a remote host via SSH.
    'host': The target IP address or hostname.
    'username': The SSH username.
    'script_content': The raw bash script commands to execute.
    Returns the stdout and stderr output from the server.
    """
    try:
        wf_config = await get_workflow_tool_config(config, "execute_ssh_mitigation")
        ssh_private_key = """-----BEGIN OPENSSH PRIVATE KEY-----
b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAAAMwAAAAtzc2gtZW
QyNTUxOQAAACDe9pfhYQYjIwmdmUam5/8w1KofOq44qfE1vRUeCeKeawAAAJBAna5qQJ2u
agAAAAtzc2gtZWQyNTUxOQAAACDe9pfhYQYjIwmdmUam5/8w1KofOq44qfE1vRUeCeKeaw
AAAEAuxDjX9AUZ2of0PEx2TxAfXPscSsv3v9OTpmiiwFyoI972l+FhBiMjCZ2ZRqbn/zDU
qh86rjip8TW9FR4J4p5rAAAADHJhaHVsQGxhcHRvcAE=
-----END OPENSSH PRIVATE KEY-----"""
        ssh_password = "094123"
        #ssh_private_key = kwargs.get("ssh_private_key", "") or wf_config.get("ssh_private_key", "")
        ssh_password = kwargs.get("ssh_password", "") or wf_config.get("ssh_password", "")
        
        if ssh_private_key:
            auth_type = 'key'
            resolved_secret = ssh_private_key
        elif ssh_password:
            auth_type = 'password'
            resolved_secret = ssh_password
        else:
            return "Error: No SSH credentials (key or password) provided in the UI tool configuration."
        
        sudo_pwd = ssh_password

        def run_ssh():
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            if auth_type.lower() == 'password':
                client.connect(hostname=host, username=username, password=resolved_secret, timeout=10)
            elif auth_type.lower() == 'key':
                # Fix common UI formatting issues where newlines get escaped as literal '\n'
                clean_secret = resolved_secret.replace('\\n', '\n').strip()
                for marker in ['OPENSSH PRIVATE KEY', 'RSA PRIVATE KEY', 'EC PRIVATE KEY', 'PRIVATE KEY']:
                    begin_marker = f'-----BEGIN {marker}-----'
                    end_marker = f'-----END {marker}-----'
                    if begin_marker in clean_secret and f'{begin_marker}\n' not in clean_secret:
                        clean_secret = clean_secret.replace(begin_marker, f'{begin_marker}\n')
                    if end_marker in clean_secret and f'\n{end_marker}' not in clean_secret:
                        clean_secret = clean_secret.replace(end_marker, f'\n{end_marker}')
                    
                key_classes = [
                    paramiko.RSAKey,
                    paramiko.Ed25519Key,
                    paramiko.ECDSAKey
                ]
                key = None
                for key_cls in key_classes:
                    try:
                        key = key_cls.from_private_key(io.StringIO(clean_secret))
                        break
                    except Exception:
                        pass
                
                if not key:
                    raise ValueError("Invalid private key format or passphrase required. Supported formats: RSA, ED25519, ECDSA.")
                    
                client.connect(hostname=host, username=username, pkey=key, timeout=10)
            else:
                raise ValueError("auth_type must be 'password' or 'key'.")

            logger.info(f"Executing mitigation script on {host} as {username}...")
            
            nonlocal script_content
            if sudo_pwd and "sudo " in script_content and "sudo -S" not in script_content:
                escaped_pwd = sudo_pwd.replace("'", "'\\''")
                script_content = script_content.replace("sudo ", f"echo '{escaped_pwd}' | sudo -S ")
            
            command = "bash -s"
            # Strip output suppression so we can capture errors and diagnose failures
            script_content = script_content.replace(" > /dev/null 2>&1", "").replace(" > /dev/null", "")
            stdin, stdout, stderr = client.exec_command(command)
            stdin.write(script_content)
            stdin.flush()
            stdin.channel.shutdown_write()  # Signal EOF to the remote bash process
            
            exit_status = stdout.channel.recv_exit_status()
            out = stdout.read().decode('utf-8', errors='ignore')
            err = stderr.read().decode('utf-8', errors='ignore')
            
            if exit_status == 0:
                verify_script = """
echo "Patch Verification Results:"
echo "| Package | Old Version | New Version |"
echo "|---------|-------------|-------------|"
for LOG in /var/log/dpkg.log /var/log/dpkg.log.*.gz
do
    [ -e "$LOG" ] || continue
    if [[ "$LOG" == *.gz ]]; then
        zcat "$LOG" 2>/dev/null
    else
        cat "$LOG" 2>/dev/null
    fi
done | grep " upgrade " | awk '{printf "| %s | %s | %s |\\n", $4, $5, $6}'
"""
                v_stdin, v_stdout, v_stderr = client.exec_command("bash -s")
                v_stdin.write(verify_script)
                v_stdin.flush()
                v_stdin.channel.shutdown_write()
                v_out = v_stdout.read().decode('utf-8', errors='ignore')
                out = out + "\n\n" + v_out

            client.close()
            return exit_status, out, err

        exit_status, out, err = await asyncio.to_thread(run_ssh)
        
        result = {
            "status": "success" if exit_status == 0 else "failed",
            "exit_code": exit_status,
            "stdout": out,
            "stderr": err
        }
        
        return json.dumps(result, indent=2)

    except Exception as e:
        logger.error(f"SSH Execution failed on {host}: {e}")
        return json.dumps({"status": "error", "message": str(e)})

# Register the tool
registry.register(
    execute_ssh_mitigation,
    ToolMeta(
        name="execute_ssh_mitigation",
        display_name="SSH Mitigation Execution",
        description="Connects to a target server via SSH to execute a bash script for patching vulnerabilities.",
        category="SecurityTools",
        configs=[
            {"name": "ssh_private_key", "type": "string", "description": "SSH private key content for authentication (if auth_type is key)", "required": False},
            {"name": "ssh_password", "type": "string", "description": "SSH password for authentication (if auth_type is password)", "required": False}
        ]
    )
)
