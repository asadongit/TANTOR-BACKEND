from typing import Optional, List, Dict
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from ..Models import MCPServerModel
from Apps.AgentMesh.Utilities.Security import (
    SECRET_MASK,
    contains_sensitive_text,
    encrypt_string,
    is_encrypted,
    is_secret_mask,
    is_sensitive_field,
    merge_masked_sensitive_data,
    protect_sensitive_data,
    redact_sensitive_data,
    reveal_sensitive_data,
)


def process_mcp_command(command: str, mode: str = "encrypt") -> str:
    """Protect the uncommon case where a credential is embedded in a command."""
    command = str(command or "")
    if mode == "encrypt":
        if is_secret_mask(command):
            return command
        return encrypt_string(command) if contains_sensitive_text(command) else command
    if mode == "decrypt":
        return reveal_sensitive_data(command)
    if mode == "redact":
        return SECRET_MASK if is_encrypted(command) or contains_sensitive_text(command) else command
    raise ValueError("mode must be one of: encrypt, decrypt, redact")


def process_mcp_args(args: Optional[List[str]], mode: str = "encrypt") -> List[str]:
    """Protect credential values passed through MCP command arguments."""
    if not args:
        return []
    output: List[str] = []
    previous_was_secret_flag = False
    for raw_arg in args:
        arg = str(raw_arg)
        flag, separator, value = arg.partition("=")
        normalized_flag = flag.lstrip("-")
        inline_secret = bool(separator and is_sensitive_field(normalized_flag))
        if mode == "encrypt":
            if (
                (previous_was_secret_flag or inline_secret or contains_sensitive_text(arg))
                and arg
                and not is_secret_mask(arg)
            ):
                output.append(encrypt_string(arg))
            else:
                output.append(arg)
        elif mode == "decrypt":
            output.append(reveal_sensitive_data(arg))
        elif mode == "redact":
            output.append(SECRET_MASK if (is_encrypted(arg) or previous_was_secret_flag or inline_secret) else arg)
        else:
            raise ValueError("mode must be one of: encrypt, decrypt, redact")
        previous_was_secret_flag = not separator and is_sensitive_field(normalized_flag)
    return output

def process_mcp_env(env: Optional[Dict[str, str]], mode: str = "encrypt") -> Dict[str, str]:
    """Protect, reveal or redact every MCP environment value."""
    if not env:
        return {}
    if mode == "encrypt":
        return protect_sensitive_data(env, force=True)
    if mode == "decrypt":
        return reveal_sensitive_data(env)
    if mode == "redact":
        return redact_sensitive_data(env, force=True)
    raise ValueError("mode must be one of: encrypt, decrypt, redact")

async def save_mcp_server(
    db: AsyncSession, 
    name: str, 
    command: str, 
    args: Optional[List[str]] = None, 
    env: Optional[Dict[str, str]] = None, 
    transport: str = "stdio"
):
    """Saves or updates an MCP server configuration."""
    stmt = select(MCPServerModel).where(MCPServerModel.name == name)
    result = await db.execute(stmt)
    db_server = result.scalar_one_or_none()
    
    if db_server:
        merged_command = merge_masked_sensitive_data(command, db_server.command)
        merged_env = merge_masked_sensitive_data(env or {}, db_server.env or {})
        merged_args = merge_masked_sensitive_data(args or [], db_server.args or [])
        safe_command = process_mcp_command(merged_command, mode="encrypt")
        safe_env = process_mcp_env(merged_env, mode="encrypt")
        safe_args = process_mcp_args(merged_args, mode="encrypt")
        db_server.command = safe_command
        db_server.args = safe_args
        db_server.env = safe_env
        db_server.transport = transport
    else:
        if is_secret_mask(command):
            raise ValueError("A masked MCP command cannot be used for a new server")
        safe_command = process_mcp_command(command, mode="encrypt")
        safe_env = process_mcp_env(env, mode="encrypt")
        safe_args = process_mcp_args(args, mode="encrypt")
        db_server = MCPServerModel(
            name=name,
            command=safe_command,
            args=safe_args,
            env=safe_env,
            transport=transport
        )
        db.add(db_server)
    
    await db.commit()
    await db.refresh(db_server)
    return db_server

async def list_mcp_servers(
    db: AsyncSession,
    *,
    reveal_secrets: bool = False,
    sealed: bool = False,
):
    """Return masked API, sealed registry, or explicit point-of-use copies."""
    stmt = select(MCPServerModel)
    result = await db.execute(stmt)
    servers = result.scalars().all()
    
    for s in servers:
        db.expunge(s)
        if not sealed:
            mode = "decrypt" if reveal_secrets else "redact"
            s.command = process_mcp_command(s.command, mode=mode)
            if s.env:
                s.env = process_mcp_env(s.env, mode=mode)
            s.args = process_mcp_args(s.args, mode=mode)
            
    return servers

async def delete_mcp_server(db: AsyncSession, name: str):
    """Removes an MCP server by name."""
    stmt = select(MCPServerModel).where(MCPServerModel.name == name)
    result = await db.execute(stmt)
    db_server = result.scalar_one_or_none()
    if db_server:
        await db.delete(db_server)
        await db.commit()
        return True
    return False
