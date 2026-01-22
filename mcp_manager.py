"""
MCP (Model Context Protocol) Server Handler
Supports multiple transport types: stdio, HTTP/HTTPS
"""
import os
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import httpx

# Configure logging to match uvicorn's format and level
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:     %(message)s',
    force=True  # Override any existing config
)
logger = logging.getLogger(__name__)


class MCPTransport(ABC):
    """Abstract base class for MCP transports"""
    
    @abstractmethod
    async def initialize(self) -> ClientSession:
        """Initialize and return MCP client session"""
        pass
    
    @abstractmethod
    async def cleanup(self):
        """Cleanup transport resources"""
        pass


class StdioTransport(MCPTransport):
    """MCP transport using stdio (subprocess)"""
    
    def __init__(self, command: str = "npx", args: Optional[List[str]] = None):
        """
        Initialize stdio transport
        
        Args:
            command: Command to start MCP server (e.g., "npx", "python")
            args: Arguments for the MCP server
        """
        self.command = command
        self.args = args or ["-y", "@modelcontextprotocol/server-everything"]
        self.stdio = None
        self.write = None
        self.session = None
        self.stdio_context = None  # Keep context manager alive
    
    async def initialize(self) -> ClientSession:
        """Start MCP server as subprocess and connect via stdio"""
        server_params = StdioServerParameters(
            command=self.command,
            args=self.args,
            env=None
        )
        
        # Connect to MCP server via context manager
        self.stdio_context = stdio_client(server_params)
        stdio_transport = await self.stdio_context.__aenter__()
        self.stdio, self.write = stdio_transport
        self.session = ClientSession(self.stdio, self.write)
        
        await self.session.__aenter__()
        await self.session.initialize()
        
        return self.session
    
    async def cleanup(self):
        """Cleanup stdio transport"""
        if self.session:
            await self.session.__aexit__(None, None, None)
        if self.stdio_context:
            await self.stdio_context.__aexit__(None, None, None)


class HTTPTransport(MCPTransport):
    """MCP transport using HTTP with optional SSE support"""
    
    def __init__(self, base_url: str, api_key: Optional[str] = None, use_sse: bool = False):
        """
        Initialize HTTP transport
        
        Args:
            base_url: Base URL of MCP server (e.g., "http://localhost:3000")
            api_key: Optional API key for authentication
            use_sse: Whether to use SSE (Server-Sent Events) for streaming (default: False)
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.use_sse = use_sse
        self.client = None
        self.session = None
    
    async def initialize(self):
        """Connect to MCP server via HTTP"""
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=None if self.use_sse else 30.0,
            headers=headers
        )
        
        # Create single session class with optional SSE
        self.session = HTTPMCPSession(self.client, use_sse=self.use_sse)
        await self.session.initialize()
        
        return self.session
    
    async def cleanup(self):
        """Cleanup HTTP transport"""
        if self.session:
            await self.session.close()
        if self.client:
            await self.client.aclose()


class HTTPMCPSession:
    """
    HTTP request-response MCP session
    Optionally supports SSE for server-to-client events
    """
    
    def __init__(self, client: httpx.AsyncClient, use_sse: bool = False):
        self.client = client
        self.use_sse = use_sse
        self.sse_stream = None
        self.message_id = 0
    
    async def initialize(self):
        """Initialize MCP protocol"""
        response = await self.client.post(
            "/messages",
            json={
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "mcp-python-client",
                        "version": "1.0.0"
                    }
                }
            }
        )
        response.raise_for_status()
        
        # Optionally open SSE connection for server events
        if self.use_sse:
            self.sse_stream = await self.client.stream("GET", "/sse")
            await self.sse_stream.__aenter__()
    
    def _next_id(self):
        """Generate next message ID"""
        self.message_id += 1
        return self.message_id
    
    async def list_tools(self):
        """List available tools via HTTP"""
        response = await self.client.post(
            "/messages",
            json={
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "tools/list",
                "params": {}
            }
        )
        response.raise_for_status()
        data = response.json()
        
        # Convert to format compatible with stdio transport
        class ToolsList:
            def __init__(self, tools):
                self.tools = tools
        
        class Tool:
            def __init__(self, name, title, description, input_schema=None, output_schema=None, annotations=None, execution=None):
                self.name = name
                self.title = title
                self.description = description
                self.input_schema = input_schema
                self.output_schema = output_schema
                self.annotations = annotations
                self.execution = execution
        
        result = data.get("result", {})
        tools = [
            Tool(
                name=t["name"], 
                title=t["title"],
                description=t["description"],
                input_schema=t["inputSchema"],
                output_schema=t["outputSchema"],
                annotations=t["annotations"],
                execution=t["execution"]
            )
            for t in result.get("tools", [])
        ]
        return ToolsList(tools)
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]):
        """Call a tool via HTTP"""
        response = await self.client.post(
            "/messages",
            json={
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments
                }
            }
        )
        response.raise_for_status()
        data = response.json()
        
        # Extract result from MCP response
        result = data.get("result", {})
        content = result.get("content", [])
        
        # Combine all text content
        if isinstance(content, list):
            return "\n".join(
                item.get("text", "") 
                for item in content 
                if item.get("type") == "text"
            )
        return str(result)
    
    async def close(self):
        """Close the HTTP session and optional SSE connection"""
        if self.sse_stream:
            await self.sse_stream.__aexit__(None, None, None)


class MCPServerManager:
    """
    High-level MCP server manager
    Handles transport selection, tool management, and session lifecycle
    """
    
    def __init__(self, transport: MCPTransport):
        """
        Initialize MCP manager
        
        Args:
            transport: Transport implementation (Stdio or HTTP)
        """
        self.transport = transport
        self.session = None
        self._tools_cache = None
    
    @classmethod
    def create_stdio(cls, command: str = "npx", args: Optional[List[str]] = None):
        """Create manager with stdio transport"""
        return cls(StdioTransport(command, args))
    
    @classmethod
    def create_http(cls, base_url: str, api_key: Optional[str] = None, use_sse: bool = False):
        """
        Create manager with HTTP transport
        
        Args:
            base_url: Base URL of MCP server
            api_key: Optional API key for authentication
            use_sse: Whether to use SSE streaming (default: False for standard HTTP)
        """
        return cls(HTTPTransport(base_url, api_key, use_sse))
    
    @classmethod
    def create_from_config(cls, config: Dict[str, Any]):
        """
        Create manager from configuration
        
        Config format:
        {
            "transport_type": "stdio" | "http",
            "server_command": "npx",  # for stdio
            "server_args": [...],     # for stdio
            "base_url": "http://localhost:3000",  # for http
            "api_key": "...",  # for http (optional)
            "use_sse": false   # for http (optional, default: false)
        }
        """
        transport_type = config.get("transport_type", "stdio")
        
        if transport_type == "stdio":
            return cls.create_stdio(
                command=config.get("server_command", "npx"),
                args=config.get("server_args")
            )
        elif transport_type == "http":
            return cls.create_http(
                base_url=config["base_url"],
                api_key=config.get("api_key"),
                use_sse=config.get("use_sse", False)
            )
        else:
            raise ValueError(f"Unsupported transport type: {transport_type}")
    
    async def initialize(self):
        """Initialize MCP connection"""
        self.session = await self.transport.initialize()
        logger.info(f"MCP server manager initialized via {self.transport.__class__.__name__}")
    
    async def list_tools(self) -> List[Dict[str, str]]:
        """
        List available tools
        
        Returns:
            List of tools with name and description
        """
        if not self.session:
            await self.initialize()
        
        tools_list = await self.session.list_tools()
        
        self._tools_cache = [{
            "name": tool.name,
            "title": getattr(tool, 'title', tool.name),
            "description": tool.description or f"MCP tool: {tool.name}",
            "inputSchema": getattr(tool, 'inputSchema', getattr(tool, 'input_schema', {})),
            "outputSchema": getattr(tool, 'outputSchema', getattr(tool, 'output_schema', {})),
            "annotations": getattr(tool, 'annotations', {}),
            "execution": getattr(tool, 'execution', {})
        }
            for tool in tools_list.tools
        ]
        
        return self._tools_cache
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Call a tool
        
        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments as dictionary
            
        Returns:
            Tool execution result
        """
        if not self.session:
            await self.initialize()
        
        result = await self.session.call_tool(tool_name, arguments=arguments)
        return result
    
    async def cleanup(self):
        """Cleanup MCP connection"""
        await self.transport.cleanup()
        self.session = None
        self._tools_cache = None


# Convenience functions for common use cases

async def create_mcp_manager(
    transport_type: str = "stdio",
    **kwargs
) -> MCPServerManager:
    """
    Create and initialize MCP manager
    
    Args:
        transport_type: "stdio" or "http"
        **kwargs: Transport-specific arguments
            For HTTP:
                - base_url: Base URL of MCP server (required)
                - api_key: API key (optional)
                - use_sse: Whether to use SSE streaming (optional, default: False)
        
    Returns:
        Initialized MCPServerManager
    """
    if transport_type == "stdio":
        manager = MCPServerManager.create_stdio(
            command=kwargs.get("command", "npx"),
            args=kwargs.get("args")
        )
    elif transport_type == "http":
        manager = MCPServerManager.create_http(
            base_url=kwargs["base_url"],
            api_key=kwargs.get("api_key"),
            use_sse=kwargs.get("use_sse", False)
        )
    else:
        raise ValueError(f"Unsupported transport type: {transport_type}")
    
    await manager.initialize()
    return manager


async def create_mcp_from_env() -> MCPServerManager:
    """
    Create MCP manager from environment variables
    
    Environment variables:
        MCP_TRANSPORT: "stdio" or "http" (default: "stdio")
        MCP_COMMAND: Command for stdio transport (default: "npx")
        MCP_ARGS: Comma-separated args for stdio (default: "-y,@modelcontextprotocol/server-everything")
        MCP_BASE_URL: Base URL for HTTP transport
        MCP_API_KEY: API key for HTTP transport (optional)
        MCP_USE_SSE: Whether to use SSE for HTTP transport (default: "false")
    """
    transport_type = os.getenv("MCP_TRANSPORT", "stdio")
    
    if transport_type == "stdio":
        command = os.getenv("MCP_COMMAND", "npx")
        args_str = os.getenv("MCP_ARGS", "-y,@modelcontextprotocol/server-everything")
        args = args_str.split(",") if args_str else None
        return await create_mcp_manager("stdio", command=command, args=args)
    
    elif transport_type == "http":
        base_url = os.getenv("MCP_BASE_URL")
        if not base_url:
            raise ValueError("MCP_BASE_URL required for HTTP transport")
        api_key = os.getenv("MCP_API_KEY")
        use_sse = os.getenv("MCP_USE_SSE", "false").lower() == "true"
        return await create_mcp_manager("http", base_url=base_url, api_key=api_key, use_sse=use_sse)
    
    else:
        raise ValueError(f"Unsupported MCP_TRANSPORT: {transport_type}")
