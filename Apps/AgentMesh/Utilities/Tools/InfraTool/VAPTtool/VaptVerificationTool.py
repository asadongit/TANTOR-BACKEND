import paramiko
import logging
import asyncio
import io
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from Apps.AgentMesh.Schema.ToolSchema import ToolMeta
from Apps.AgentMesh.Core.ToolRegistry import registry

logger = logging.getLogger(__name__)

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
async def verify_vapt_patching(
    host: str, 
    username: str, 
    config: RunnableConfig = None,
    **kwargs
) -> str:
    """
    Connects to a remote Linux host over SSH and parses the dpkg.log to return a table of recently upgraded packages.
    Useful for verifying that patching or remediation was successful.
    
    Args:
        host: IP address or hostname of the target VM.
        username: SSH username.
    """
    
    tool_config = await get_workflow_tool_config(config, "verify_vapt_patching")
    
    ssh_private_key = kwargs.get("ssh_private_key", "") or tool_config.get("ssh_private_key", "")
    secret = kwargs.get("ssh_password", "") or tool_config.get("ssh_password", "")
    
    if ssh_private_key:
        auth_type = "key"
        ssh_private_key = ssh_private_key.replace('\\n', '\n')
        ssh_private_key = ssh_private_key.replace('-----BEGIN OPENSSH PRIVATE KEY-----', '-----BEGIN OPENSSH PRIVATE KEY-----\n')
        ssh_private_key = ssh_private_key.replace('-----END OPENSSH PRIVATE KEY-----', '\n-----END OPENSSH PRIVATE KEY-----')
        ssh_private_key = ssh_private_key.replace('-----BEGIN RSA PRIVATE KEY-----', '-----BEGIN RSA PRIVATE KEY-----\n')
        ssh_private_key = ssh_private_key.replace('-----END RSA PRIVATE KEY-----', '\n-----END RSA PRIVATE KEY-----')
    elif secret:
        auth_type = "password"
    else:
        return "Error: No SSH credentials (key or password) provided in the UI tool configuration."

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        if auth_type == "password":
            ssh.connect(hostname=host, username=username, password=secret, timeout=10)
        elif auth_type == "key" and ssh_private_key:
            pkey = None
            key_file = io.StringIO(ssh_private_key)
            # Try parsing the key in multiple formats
            for pkey_class in (paramiko.RSAKey, paramiko.Ed25519Key, paramiko.ECDSAKey):
                try:
                    key_file.seek(0)
                    pkey = pkey_class.from_private_key(key_file)
                    break
                except paramiko.SSHException:
                    continue
            
            if not pkey:
                return "Error: Could not parse the provided private SSH key. Ensure it's a valid RSA/ED25519/ECDSA key."
                
            ssh.connect(hostname=host, username=username, pkey=pkey, timeout=10)
        else:
            return "Error: Invalid authentication method or missing credentials."

        # The bash script logic to run directly on the host
        remote_script = """
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
        
        stdin, stdout, stderr = ssh.exec_command(remote_script)
        
        output = stdout.read().decode('utf-8')
        error = stderr.read().decode('utf-8')
        
        if error and not output:
            return f"Error executing verification script:\n{error}"
            
        return f"Patch Verification Results:\n\n{output}"

    except Exception as e:
        logger.error(f"SSH Verification Error: {str(e)}")
        return f"Error connecting to SSH or executing script: {str(e)}"
    finally:
        ssh.close()

# Register the tool
registry.register(
    verify_vapt_patching,
    ToolMeta(
        name="verify_vapt_patching",
        display_name="Verify VAPT Patching",
        description="Connects via SSH and parses dpkg logs to verify package upgrades",
        category="SecurityTools",
        configs=[
            {"name": "ssh_private_key", "type": "string", "description": "The private SSH key block", "required": False},
            {"name": "ssh_password", "type": "string", "description": "SSH password for authentication", "required": False}
        ]
    )
)
