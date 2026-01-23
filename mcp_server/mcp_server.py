"""
Custom MCP Server with Dynamic Tool Discovery
Discovers and registers tools from mcp_server/tools/ directory structure
"""
import asyncio
import json
import logging
import os
import sys
import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server import Server
from mcp.types import Tool, CallToolResult, TextContent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s',
    stream=sys.stderr  # Avoid writing logs to stdout; stdout is reserved for MCP protocol
)
logger = logging.getLogger(__name__)

# Create MCP server
app = Server("custom-tools-server")

# Store loaded tools and their handlers
loaded_tools: Dict[str, Dict[str, Any]] = {}
tools_dir = Path(__file__).parent / "tools"


def load_tools_from_directory(tools_directory: Path) -> Dict[str, Dict[str, Any]]:
    """
    Dynamically discover and load tools from directory structure (recursive)
    
    Expected structure:
    /tools/
      /category/
        /tool_name/
          schema.json      (tool metadata and input schema)
          handler.py       (handler function)
      /tool_name/
        schema.json        (flat structure also supported)
        handler.py
    """
    tools_map = {}
    
    if not tools_directory.exists():
        logger.warning(f"Tools directory not found: {tools_directory}")
        return tools_map
    
    logger.info(f"Scanning tools directory: {tools_directory}")
    
    # Recursively find all directories containing schema.json and handler.py
    def find_tool_folders(directory: Path) -> List[Path]:
        """Recursively find all valid tool folders"""
        tool_folders = []
        
        for item in sorted(directory.iterdir()):
            if not item.is_dir() or item.name.startswith('_'):
                continue
            
            schema_file = item / "schema.json"
            handler_file = item / "handler.py"
            
            # If this directory has both files, it's a tool
            if schema_file.exists() and handler_file.exists():
                tool_folders.append(item)
            else:
                # Otherwise, recurse into it
                tool_folders.extend(find_tool_folders(item))
        
        return tool_folders
    
    # Find all tool folders
    tool_folders = find_tool_folders(tools_directory)
    
    # Load each tool
    for tool_folder in tool_folders:
        try:
            schema_file = tool_folder / "schema.json"
            handler_file = tool_folder / "handler.py"
            
            # Load schema
            with open(schema_file, 'r') as f:
                schema = json.load(f)
            
            tool_name = schema.get("name", tool_folder.name)
            
            # Load handler function
            spec = importlib.util.spec_from_file_location(f"handler_{tool_name}", handler_file)
            handler_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(handler_module)
            
            # Get the handler function (should be named 'handler')
            if not hasattr(handler_module, 'handler'):
                logger.warning(f"No 'handler' function found in {tool_folder.name}/handler.py")
                continue
            
            handler_func = handler_module.handler
            
            # Store tool info
            tools_map[tool_name] = {
                "schema": schema,
                "handler": handler_func,
                "tool_folder": tool_folder
            }
            
            logger.info(f"✓ Loaded tool: {tool_name}")
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {tool_folder.name}/schema.json: {e}")
        except Exception as e:
            logger.error(f"Failed to load tool from {tool_folder.name}: {e}", exc_info=True)
    
    return tools_map


def validate_schema(schema: Dict[str, Any]) -> bool:
    """Validate that schema has required fields"""
    required_fields = ["name", "description", "inputSchema"]
    for field in required_fields:
        if field not in schema:
            logger.error(f"Schema missing required field: {field}")
            return False
    return True


@app.list_tools()
async def list_tools() -> List[Tool]:
    """List all available tools"""
    tools_list = []
    
    for tool_name, tool_info in loaded_tools.items():
        schema = tool_info["schema"]
        
        tool = Tool(
            name=schema.get("name", tool_name),
            title=schema.get("title", schema.get("name", tool_name)),
            description=schema.get("description", f"Tool: {tool_name}"),
            inputSchema=schema.get("inputSchema", {}),
            outputSchema=schema.get("outputSchema") or None,
            annotations=schema.get("annotations", {}),
            execution=schema.get("execution", {})
        )
        tools_list.append(tool)
    
    logger.info(f"Returning {len(tools_list)} available tools")
    return tools_list


@app.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> CallToolResult:
    """Execute a tool by name"""
    
    if name not in loaded_tools:
        raise ValueError(f"Unknown tool: {name}")
    
    try:
        tool_info = loaded_tools[name]
        handler = tool_info["handler"]
        
        logger.info(f"Calling tool: {name} with arguments: {arguments}")
        
        # Call the handler function
        # Support both sync and async handlers
        result = handler(**arguments)
        
        if asyncio.iscoroutine(result):
            result = await result
        
        logger.info(f"Tool {name} returned: {result}")
        return CallToolResult(content=[TextContent(type="text", text=str(result))])
    
    except TypeError as e:
        logger.error(f"Tool {name} argument error: {e}", exc_info=True)
        raise ValueError(f"Invalid arguments for tool {name}: {e}")
    except Exception as e:
        logger.error(f"Tool {name} execution error: {e}", exc_info=True)
        raise


async def main():
    """Initialize and run the MCP server"""
    global loaded_tools
    
    # Load all tools from directory
    loaded_tools = load_tools_from_directory(tools_dir)
    
    if not loaded_tools:
        logger.warning("No tools loaded! Make sure tools are in /tools/ subdirectories")
    else:
        logger.info(f"Successfully loaded {len(loaded_tools)} tools")
        logger.info(f"Available tools: {', '.join(loaded_tools.keys())}")
    
    # Run the server with stdio transport
    from mcp.server.stdio import stdio_server
    from mcp.server import InitializationOptions
    
    async with stdio_server() as (read_stream, write_stream):
        from mcp.types import ServerCapabilities
        init_options = InitializationOptions(
            server_name="custom-tools-server",
            server_version="1.0.0",
            capabilities=ServerCapabilities()
        )

        await app.run(read_stream, write_stream, init_options)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        sys.exit(1)