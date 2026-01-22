"""
MCP Server HTTP API Wrapper
Exposes MCP server via FastAPI HTTP endpoints
Translates HTTP requests to MCP stdio protocol
"""
import asyncio
import json
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Configure logging to match uvicorn's format and level
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:     %(message)s',
    force=True  # Override any existing config
)
logger = logging.getLogger(__name__)

# Path to mcp_server.py (now in same folder as this file)
current_dir = Path(__file__).parent
mcp_server_path = current_dir / "mcp_server.py"

# Global MCP session and transport
mcp_session: ClientSession = None
stdio_context = None


async def initialize_mcp():
    """Initialize MCP server connection via stdio"""
    global mcp_session, stdio_context
    
    try:
        server_params = StdioServerParameters(
            command="python",
            args=[str(mcp_server_path)],
            env=None
        )
        
        logger.info("Starting MCP server...")
        # Create context manager but keep it open
        stdio_context = stdio_client(server_params)
        stdio_transport = await stdio_context.__aenter__()
        stdio, write = stdio_transport
        mcp_session = ClientSession(stdio, write)
        
        await mcp_session.__aenter__()
        await mcp_session.initialize()
        logger.info("MCP server initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize MCP server: {e}", exc_info=True)
        raise


async def cleanup_mcp():
    """Cleanup MCP server connection"""
    global mcp_session, stdio_context
    
    try:
        if mcp_session:
            await mcp_session.__aexit__(None, None, None)
            logger.info("MCP session cleaned up")
    except Exception as e:
        logger.error(f"Error cleaning up MCP session: {e}")
    
    try:
        if stdio_context:
            await stdio_context.__aexit__(None, None, None)
            logger.info("Stdio context cleaned up")
    except Exception as e:
        logger.error(f"Error cleaning up stdio context: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager"""
    # Startup
    await initialize_mcp()
    yield
    # Shutdown
    await cleanup_mcp()


# Create FastAPI app with lifespan
app = FastAPI(
    title="MCP Server API",
    description="HTTP wrapper for MCP (Model Context Protocol) server",
    lifespan=lifespan
)


@app.post("/messages")
async def handle_message(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle JSON-RPC 2.0 MCP requests
    
    Supported methods:
    - initialize: Initialize MCP protocol
    - tools/list: List available tools
    - tools/call: Call a specific tool
    """
    global mcp_session
    
    if not mcp_session:
        raise HTTPException(status_code=503, detail="MCP server not initialized")
    
    try:
        method = request_data.get("method")
        params = request_data.get("params", {})
        message_id = request_data.get("id")
        
        logger.debug(f"Handling method: {method}, params: {params}")
        
        if method == "initialize":
            # Already initialized in startup, return success
            return {
                "jsonrpc": "2.0",
                "id": message_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "serverInfo": {
                        "name": "mcp-http-wrapper",
                        "version": "1.0.0"
                    }
                }
            }
        
        elif method == "tools/list":
            # List available tools
            tools = await mcp_session.list_tools()
            #print(tools)
            # for tool in tools.tools:
                # print(f"Available tool: {tool}")

            return {
                "jsonrpc": "2.0",
                "id": message_id,
                "result": {
                    "tools": [
                        {
                            "name": tool.name,
                            "title": getattr(tool, 'title', tool.name),
                            "description": tool.description or f"MCP tool: {tool.name}",
                            "inputSchema": getattr(tool, 'inputSchema', getattr(tool, 'input_schema', {})),
                            "outputSchema": getattr(tool, 'outputSchema', getattr(tool, 'output_schema', {})),
                            "annotations": getattr(tool, 'annotations', {}),
                            "execution": getattr(tool, 'execution', {})
                        }
                        for tool in tools.tools
                    ]
                }
            }
        
        elif method == "tools/call":
            # Call a specific tool
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            
            if not tool_name:
                return {
                    "jsonrpc": "2.0",
                    "id": message_id,
                    "error": {
                        "code": -32602,
                        "message": "Missing tool name"
                    }
                }
            
            logger.info(f"Calling tool: {tool_name} with args: {arguments}")
            result = await mcp_session.call_tool(tool_name, arguments)
            
            return {
                "jsonrpc": "2.0",
                "id": message_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": str(result)
                        }
                    ]
                }
            }
        
        else:
            return {
                "jsonrpc": "2.0",
                "id": message_id,
                "error": {
                    "code": -32601,
                    "message": f"Unknown method: {method}"
                }
            }
    
    except Exception as e:
        logger.error(f"Error handling message: {e}", exc_info=True)
        return {
            "jsonrpc": "2.0",
            "id": request_data.get("id"),
            "error": {
                "code": -32603,
                "message": str(e)
            }
        }


@app.get("/sse")
async def sse_stream():
    """
    Server-Sent Events stream for MCP server events
    Currently a placeholder - streams keep-alive pings
    """
    
    async def event_generator():
        """Generate SSE events"""
        try:
            # Send initial connection event
            yield b"data: {\"type\": \"connection\", \"status\": \"connected\"}\n\n"
            
            # Keep connection alive with periodic pings
            while True:
                await asyncio.sleep(30)
                yield b"data: {\"type\": \"ping\"}\n\n"
        
        except asyncio.CancelledError:
            logger.info("SSE stream closed")
        except Exception as e:
            logger.error(f"SSE error: {e}")
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint"""
    return {
        "status": "healthy",
        "mcp_connected": mcp_session is not None
    }


@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "name": "MCP Server HTTP API",
        "version": "1.0.0",
        "description": "HTTP wrapper for MCP (Model Context Protocol) server",
        "endpoints": {
            "messages": "POST /messages - JSON-RPC 2.0 endpoint",
            "sse": "GET /sse - Server-Sent Events stream",
            "health": "GET /health - Health check"
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "mcp_api:app",
        host="0.0.0.0",
        port=3000,
        reload=True,
        log_level="info"
    )
