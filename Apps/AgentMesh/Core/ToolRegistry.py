from typing import List, Dict, Callable, Any, Optional
import os
import asyncio
import logging
from langchain_core.tools import BaseTool
from Apps.AgentMesh.Schema.ToolSchema import ToolMeta
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)

class MCPToolWrapper(BaseTool):
    """
    A LangChain BaseTool that wraps an official MCP Tool from a ClientSession.
    Delegates execution to the session.call_tool method.
    """
    session: Any  # mcp.ClientSession
    tool_name: str
    
    def __init__(self, name: str, description: str, args_schema: Any, session: Any):
        tool_name = name.split("--")[-1] if "--" in name else name
        super().__init__(
            name=name,
            description=description,
            args_schema=args_schema,
            session=session,
            tool_name=tool_name
        )

    async def _arun(self, **kwargs) -> str:
        """Execute the tool via the MCP session."""
        try:
            result = await self.session.call_tool(self.tool_name, arguments=kwargs)
            
            # Handle text content
            content_list = []
            for content in result.content:
                if hasattr(content, 'text'):
                    content_list.append(content.text)
                else:
                    content_list.append(str(content))
            
            return "\n".join(content_list)
        except Exception as e:
            return f"Error executing MCP tool '{self.name}': {str(e)}"

    def _run(self, **kwargs) -> str:
        """Synchronous run is not supported for MCP tools as they are natively async."""
        raise NotImplementedError("MCP tools are async-only. Use arun.")

class ToolRegistry:
    """
    Central registry for all tools available to the AgentMesh Engine.
    Provides methods to list UI metadata and resolve string names to LangChain Tool objects.
    Now supports Official MCP SDK for stateful, persistent connections.
    """
    
    def __init__(self):
        self._static_tools: Dict[str, BaseTool] = {}
        self._static_meta: Dict[str, ToolMeta] = {}
        
        # Stores dynamic tool loaders (e.g., for custom API-based tools)
        self._dynamic_loaders: Dict[str, Callable[[str], BaseTool]] = {}
        
        # MCP state
        self._mcp_configs: Dict[str, Dict[str, Any]] = {}
        self._mcp_sessions: Dict[str, ClientSession] = {}
        self._mcp_tools: Dict[str, BaseTool] = {}
        self._exit_stacks: Dict[str, Any] = {} # To manage session cleanup

    def register(self, tool_obj: BaseTool, meta: ToolMeta):
        """Register a hardcoded LangChain tool (typically built-in)."""
        if not isinstance(tool_obj, BaseTool):
            raise ValueError(f"Tool {meta.name} must be a LangChain BaseTool instance.")
        # Auto-extract arguments schema if present
        if not meta.args:
            if hasattr(tool_obj, "args") and isinstance(tool_obj.args, dict):
                meta.args = tool_obj.args
            elif hasattr(tool_obj, "args_schema") and tool_obj.args_schema:
                if hasattr(tool_obj.args_schema, "schema"):
                    meta.args = tool_obj.args_schema.schema()
                elif hasattr(tool_obj.args_schema, "model_json_schema"):
                    meta.args = tool_obj.args_schema.model_json_schema()
        self._static_tools[meta.name] = tool_obj
        self._static_meta[meta.name] = meta

    def add_mcp_server(self, server_name: str, config: Dict[str, Any]):
        """
        Store an MCP server configuration.
        Example config: {"command": "npx", "args": ["@modelcontextprotocol/server-everything"]}
        """
        self._mcp_configs[server_name] = config

    def remove_mcp_server(self, server_name: str):
        """Removes an MCP server and its tools from the registry."""
        self._mcp_configs.pop(server_name, None)
        
        # Cleanup session if exists
        asyncio.create_task(self._cleanup_session(server_name))
        
        # Remove all tools associated with this server
        keys_to_remove = [k for k in self._mcp_tools.keys() if k.startswith(f"{server_name}--")]
        for k in keys_to_remove:
            self._mcp_tools.pop(k, None)

    async def _cleanup_session(self, server_name: str):
        """Gracefully closes an MCP session and its process."""
        if server_name in self._mcp_sessions:
            logger.info(f"Closing MCP session for {server_name}...")
            # Cleanup logic here if using ExitStack, or manual close
            self._mcp_sessions.pop(server_name, None)

    async def sync_mcp_tools(self):
        """
        Connects to all registered MCP servers using official SDK and fetches their tools.
        Maintains persistent sessions for stateful servers (like Colab).
        """
        if not self._mcp_configs:
            return

        for server_name, config in self._mcp_configs.items():
            try:
                # If session already exists, we just refresh the tools
                if server_name in self._mcp_sessions:
                    session = self._mcp_sessions[server_name]
                    await self._refresh_tools_for_session(server_name, session)
                    continue

                # Otherwise, establish new connection
                logger.info(f"Establishing official MCP connection to {server_name}...")
                
                # Start in a separate task or use a persistent connection manager
                # For simplicity in this registry, we start it and keep it in memory
                asyncio.create_task(self._start_mcp_session(server_name, config))
                
            except Exception as e:
                logger.error(f"Error syncing MCP server {server_name}: {e}")

    async def _start_mcp_session(self, server_name: str, config: Dict[str, Any]):
        """Background worker to start and maintain an MCP session."""
        from contextlib import AsyncExitStack
        from Apps.AgentMesh.Utilities.Security import reveal_sensitive_data
        
        stack = AsyncExitStack()
        self._exit_stacks[server_name] = stack
        
        try:
            runtime_config = reveal_sensitive_data(config)
            params = StdioServerParameters(
                command=runtime_config["command"],
                args=runtime_config.get("args", []),
                env=runtime_config.get("env")
            )
            
            read, write = await stack.enter_async_context(stdio_client(params))
            session = await stack.enter_async_context(ClientSession(read, write))
            
            await session.initialize()
            self._mcp_sessions[server_name] = session
            logger.info(f"MCP Session initialized for {server_name}")
            
            # Initial tool fetch
            await self._refresh_tools_for_session(server_name, session)
            
        except Exception as e:
            logger.error(f"Failed to start MCP session {server_name}: {e}")
            await stack.aclose()

    async def _refresh_tools_for_session(self, server_name: str, session: ClientSession):
        """Calls list_tools on an active session and updates the tool registry."""
        try:
            mcp_tools_list = await session.list_tools()
            
            # Update our internal registry
            for meta_tool in mcp_tools_list.tools:
                full_name = f"{server_name}--{meta_tool.name}"
                
                # Convert the JSON Schema from MCP into a format LangChain can understand for metadata purposes.
                # BaseTool expects args_schema to be a Pydantic model usually, 
                # but many LangChain implementations accept a dict or handle the schema manually.
                # For UI display, we store the raw dict schema.
                
                wrapper = MCPToolWrapper(
                    name=full_name,
                    description=meta_tool.description,
                    args_schema=meta_tool.inputSchema, 
                    session=session
                )
                
                self._mcp_tools[full_name] = wrapper
                
            logger.info(f"Refreshed {len(mcp_tools_list.tools)} tools for {server_name}")
            
        except Exception as e:
            logger.error(f"Failed to refresh tools for {server_name}: {e}")

    def get_tool(self, name: str) -> BaseTool:
        """
        Resolve a string name to a LangChain Tool.
        Priority: Static -> MCP -> Dynamic Loaders
        """
        # Resolve base tool name if suffix "::" is present (e.g. rag_tool::instance_1)
        lookup_name = name.split("::")[0] if "::" in name else name

        # 1. Check Static Registry
        if lookup_name in self._static_tools:
            return self._static_tools[lookup_name]

        # 2. Check Fetched MCP Tools
        if lookup_name in self._mcp_tools:
            return self._mcp_tools[lookup_name]

        # 3. Fallback: Parse name for source prefix
        if "/" in lookup_name:
            src, t_name = lookup_name.split("/", 1)
            if src in self._dynamic_loaders:
                return self._dynamic_loaders[src](t_name)

        raise KeyError(f"Tool '{name}' not found in any registry.")

    def get_tools(self, names: List[str]) -> List[BaseTool]:
        """Resolve a list of string names to LangChain Tools."""
        tools = []
        for name in names:
            try:
                base_tool = self.get_tool(name)
                # Clone/copy the tool to prevent sharing mutated state
                try:
                    tool_clone = base_tool.copy()
                except Exception:
                    # Fallback to copy.copy or custom StructuredTool creation if copy() fails
                    import copy
                    try:
                        tool_clone = copy.copy(base_tool)
                    except Exception:
                        tool_clone = base_tool
                
                # Assign the requested custom name (including suffix) to the cloned instance
                tool_clone.name = name
                tools.append(tool_clone)
            except KeyError:
                logger.warning(f"Tool '{name}' requested but not available.")
        return tools

    def get_meta(self, name: str) -> Optional[ToolMeta]:
        """Retrieve metadata for a specific tool by name."""
        if name in self._static_meta:
            meta = self._static_meta[name]
            tool_obj = self._static_tools.get(name)
            if tool_obj and not meta.args:
                if hasattr(tool_obj, "args") and isinstance(tool_obj.args, dict):
                    meta.args = tool_obj.args
                elif hasattr(tool_obj, "args_schema") and tool_obj.args_schema:
                    if hasattr(tool_obj.args_schema, "schema"):
                        meta.args = tool_obj.args_schema.schema()
                    elif hasattr(tool_obj.args_schema, "model_json_schema"):
                        meta.args = tool_obj.args_schema.model_json_schema()
            return meta
            
        if name in self._mcp_tools:
            return self._construct_mcp_meta(name, self._mcp_tools[name])
            
        return None

    def get_all_meta(self) -> List[ToolMeta]:
        """Constructs a comprehensive list of metadata for all registered tools."""
        all_meta = []
        for name, meta in self._static_meta.items():
            tool_obj = self._static_tools.get(name)
            if tool_obj and not meta.args:
                if hasattr(tool_obj, "args") and isinstance(tool_obj.args, dict):
                    meta.args = tool_obj.args
                elif hasattr(tool_obj, "args_schema") and tool_obj.args_schema:
                    if hasattr(tool_obj.args_schema, "schema"):
                        meta.args = tool_obj.args_schema.schema()
                    elif hasattr(tool_obj.args_schema, "model_json_schema"):
                        meta.args = tool_obj.args_schema.model_json_schema()
            all_meta.append(meta)
            
        for name, tool_obj in self._mcp_tools.items():
            all_meta.append(self._construct_mcp_meta(name, tool_obj))
        return all_meta

    def _construct_mcp_meta(self, name: str, tool_obj: Any) -> ToolMeta:
        server_name = name.split("--")[0] if "--" in name else "mcp"
        display_name = name.split("--")[1].replace("_", " ").title() if "--" in name else name.replace("_", " ").title()
        
        args = None
        if hasattr(tool_obj, "args_schema") and tool_obj.args_schema:
            args = tool_obj.args_schema
        elif hasattr(tool_obj, "args") and isinstance(tool_obj.args, dict):
            args = tool_obj.args

        return ToolMeta(
            name=name,
            display_name=display_name,
            description=tool_obj.description or "No description provided.",
            category="mcp",
            package=f"MCP: {server_name}",
            version="v1.0",
            args=args,
            allow_duplicate=True
        )

    def discover_local_tools(self, directory_path: str):
        if not os.path.exists(directory_path):
            logger.warning(f"Tools directory not found: {directory_path}")
            return

        for root, _, files in os.walk(directory_path):
            for filename in files:
                if filename.endswith(".py") and not filename.startswith("__"):
                    module_name = filename[:-3]
                    file_path = os.path.join(root, filename)
                    self._load_tool_module(module_name, file_path)

    def _load_tool_module(self, module_name: str, file_path: str):
        import importlib.util
        import sys
        if module_name in sys.modules: return
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)
            except Exception as e:
                logger.error(f"Error loading tool module {file_path}: {e}")

    async def shutdown(self):
        """Cleanup all MCP sessions on application stop."""
        for server_name in list(self._exit_stacks.keys()):
            stack = self._exit_stacks.pop(server_name)
            logger.info(f"Shutting down MCP session: {server_name}")
            await stack.aclose()

registry = ToolRegistry()

