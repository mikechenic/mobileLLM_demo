import os
import yaml
import asyncio
import threading
import queue
import signal
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, create_model
from pydantic.fields import Field
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
import uvicorn
import json
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from mcp_manager import MCPServerManager


class SyncMCPProxy:
    """Synchronous proxy for MCP operations using queue-based communication with worker thread"""
    
    def __init__(self, tool_call_queue: queue.Queue, tool_response_queue: queue.Queue, agent_api: 'MCPAgentAPI' = None):
        self.tool_call_queue = tool_call_queue
        self.tool_response_queue = tool_response_queue
        self.agent_api = agent_api
    
    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Call a tool synchronously via queue communication"""
        # Check if tool call limit exceeded
        if self.agent_api and self.agent_api._tool_call_count >= self.agent_api.max_tool_calls:
            error_msg = (
                f"Tool call limit ({self.agent_api.max_tool_calls}) reached. "
                f"Cannot execute tool '{tool_name}'. The assistant will respond based on previous tool calls."
            )
            logger.warning(error_msg)
            return error_msg
        
        # Increment tool call counter
        if self.agent_api:
            self.agent_api._tool_call_count += 1
        
        try:
            self.tool_call_queue.put({"tool": tool_name, "args": arguments})
            response = self.tool_response_queue.get(timeout=30)
            if "error" in response:
                return f"Error: {response['error']}"
            return response["result"]
        except queue.Empty:
            return "Error: Tool call timeout"
        except Exception as e:
            logger.error(f"Error in SyncMCPProxy.call_tool: {e}", exc_info=True)
            return f"Error: {str(e)}"

# Load configuration from config.yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Load environment variables
load_dotenv()

# Configure logging to match uvicorn's format and level
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:     %(message)s',
    force=True  # Override any existing config
)
logger = logging.getLogger(__name__)

# FastAPI app for LLM service
app = FastAPI(title="LLM Agent API", description="ReAct agent with MCP tools")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    text: str
    sender: str
    tool_calls: List[Dict[str, Any]]
    conversation_id: Optional[str] = None

class MCPAgentAPI:
    """Agent API that uses Langchain and MCP to interact with tools"""
    
    def __init__(self):
        """
        Initialize the MCP Agent API
        """
        # MCP config from config.yaml
        mcp_cfg = config.get('mcp', {})
        self.mcp_config = {
            "type": mcp_cfg.get("transport_type", "stdio"),
            "command": mcp_cfg.get("server_command", "npx"),
            "args": mcp_cfg.get("server_args", ["-y", "@modelcontextprotocol/server-everything"]),
            "base_url": mcp_cfg.get("base_url"),  # for HTTP
            "api_key": mcp_cfg.get("api_key"),     # for HTTP
            "use_sse": mcp_cfg.get("use_sse", False), # whether to use SSE (Server-Sent Events) for streaming
        }
        
        # Agent config
        agent_cfg = config.get('agent', {})
        self.max_tool_calls = agent_cfg.get("max_tool_calls", 5)  # Default limit of 5 tool calls per message
        logger.info(f"Tool call limit set to {self.max_tool_calls}")
        
        # Initialize OpenAI with custom endpoint
        self.llm = init_chat_model(
            model="gpt-4o-mini",
            temperature=0.5,
            timeout=30,
            max_retries=3,
            max_tokens=4096,
            base_url=os.getenv("OPENAI_API_ENDPOINT"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
        
        self.agent = None
        self.mcp_manager: Optional[MCPServerManager] = None
        self._current_tool_calls: Optional[List[Dict[str, Any]]] = None
        self._tool_call_count = 0  # Counter for consecutive tool calls in current message
        
        # Worker thread setup
        self.tool_executor = ThreadPoolExecutor(max_workers=1)
        self.mcp_ready = threading.Event()
        self.worker_loop: Optional[asyncio.AbstractEventLoop] = None
        self.worker_loop_ready = threading.Event()
        self.shutdown_event = threading.Event()  # Signal to shutdown worker thread
        
        # Queue for tool call requests/responses
        self.tool_call_queue: queue.Queue = queue.Queue()
        self.tool_response_queue: queue.Queue = queue.Queue()
        
    async def initialize(self):
        """Initialize MCP connection and create agent"""
        # Start the worker thread with persistent event loop and MCP manager
        self.tool_executor.submit(self._worker_thread_main)
        
        # Wait for worker thread's event loop to be ready
        if not self.worker_loop_ready.wait(timeout=10):
            raise RuntimeError("Worker thread event loop failed to initialize")
        
        # Wait for MCP manager to be ready
        if not self.mcp_ready.wait(timeout=15):
            raise RuntimeError("Worker thread MCP manager failed to initialize")
        
        # Get available tools from MCP
        # Use run_coroutine_threadsafe to call async method in worker thread
        future = asyncio.run_coroutine_threadsafe(
            self.mcp_manager.list_tools(),
            self.worker_loop
        )
        tools_list = future.result(timeout=10)
        logger.info(f"Found {len(tools_list)} tools from MCP")
        
        # Create synchronous proxy for tool calls (pass self to track call limits)
        mcp_proxy = SyncMCPProxy(self.tool_call_queue, self.tool_response_queue, agent_api=self)
        
        # Convert MCP tools to Langchain tools
        langchain_tools = []
        # print(tools_list)
        for tool_info in tools_list:
            tool_name = tool_info.get("name", "unknown_tool")
            tool_desc = tool_info.get("description", f"MCP tool: {tool_name}")
            tool_input_schema = tool_info.get("inputSchema", {})
            
            # Create a proper closure to capture tool_name correctly
            def make_tool_func(tn: str):
                return lambda **kwargs: mcp_proxy.call_tool(tn, kwargs)
            
            # Convert JSON schema to Pydantic model for args_schema
            args_schema_model = self._json_schema_to_pydantic(tool_input_schema)
            
            # Create tool with properly captured tool name
            langchain_tool = StructuredTool(
                name=tool_name,
                func=make_tool_func(tool_name),
                description=tool_desc,
                args_schema=args_schema_model,
            )
            langchain_tools.append(langchain_tool)

        logger.info(f"Loaded {len(langchain_tools)} tools from MCP")
        logger.info(langchain_tools)

        # Create ReAct agent with tools
        self.agent = create_agent(
            self.llm,
            langchain_tools,
            system_prompt=SystemMessage(
                content="You are a helpful AI assistant with access to various tools. "
                        "Use the tools when needed to answer questions accurately."
            )
        )
        
        logger.info(f"Agent initialized with {len(langchain_tools)} tools")
    
    def _json_type_to_python(self, json_type: str):
        """Convert JSON schema type to Python type"""
        type_mapping = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        return type_mapping.get(json_type, str)
    
    def _json_schema_to_pydantic(self, schema: Dict[str, Any]) -> Optional[type]:
        """Convert JSON schema dict to Pydantic BaseModel class"""
        if not schema or not schema.get("properties"):
            return None
        
        try:
            field_definitions = {}
            properties = schema.get("properties", {})
            required_fields = schema.get("required", [])
            
            for field_name, field_info in properties.items():
                field_type = self._json_type_to_python(field_info.get("type", "string"))
                field_description = field_info.get("description", "")
                is_required = field_name in required_fields
                
                if is_required:
                    field_definitions[field_name] = (field_type, Field(..., description=field_description))
                else:
                    field_definitions[field_name] = (field_type, Field(default=None, description=field_description))
            
            # Create dynamic Pydantic model with sanitized name
            model_name = "ToolInput"
            if "title" in schema:
                model_name = schema["title"].replace("-", "_").replace(" ", "_")
            
            pydantic_model = create_model(model_name, **field_definitions)
            return pydantic_model
        except Exception as e:
            logger.error(f"Failed to create Pydantic schema from {schema}: {e}", exc_info=True)
            return None
    
    def _worker_thread_main(self):
        """
        Main function for the worker thread.
        Creates and runs its own event loop, initializes MCP manager, and processes tool calls.
        """
        try:
            # Create and set a new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self.worker_loop = loop
            
            # Signal that the event loop is ready
            self.worker_loop_ready.set()
            logger.info("Worker thread event loop ready")
            
            # Initialize MCP manager asynchronously within the worker thread's loop
            loop.run_until_complete(self._init_mcp_in_worker())
            
            # Create and schedule the queue listener as a background task
            # This keeps the loop alive and responsive to scheduled coroutines
            queue_task = asyncio.ensure_future(self._queue_listener_task())
            
            # Run the event loop - it will stay alive and process scheduled coroutines
            # The queue_listener_task will run concurrently
            loop.run_until_complete(queue_task)
            
        except Exception as e:
            logger.error(f"Worker thread error: {e}", exc_info=True)
            self.mcp_ready.set()  # Signal anyway to unblock
        finally:
            if self.worker_loop:
                self.worker_loop.close()
                self.worker_loop = None
            logger.info("Worker thread exited")
    
    async def _queue_listener_task(self):
        """
        Async task that listens for tool call requests on the queue.
        Runs indefinitely in the worker thread's event loop.
        """
        while not self.shutdown_event.is_set():
            try:
                # Use a small sleep to avoid busy-waiting and allow other coroutines to run
                await asyncio.sleep(0.01)
                
                # Non-blocking check for queue items
                try:
                    request = self.tool_call_queue.get_nowait()
                except queue.Empty:
                    continue
                
                # Process the tool call
                result = await self._process_tool_call(request["tool"], request["args"])
                
                # Send response back
                self.tool_response_queue.put(result)
                
            except Exception as e:
                logger.error(f"Error in queue listener: {e}", exc_info=True)
        
        logger.info("Queue listener task ending, cleaning up MCP...")
        # Cleanup MCP manager before exiting
        if self.mcp_manager:
            await self.mcp_manager.cleanup()
            logger.info("MCP manager cleaned up")
    
    async def _init_mcp_in_worker(self):
        """
        Initialize the MCP manager in the worker thread's event loop.
        This must be called from within the worker thread.
        """
        try:
            self.mcp_manager = MCPServerManager.create_from_config(self.mcp_config)
            await self.mcp_manager.initialize()
            self.mcp_ready.set()
            logger.info("MCP manager initialized in worker thread")
        except Exception as e:
            logger.error(f"Failed to initialize MCP: {e}", exc_info=True)
            self.mcp_ready.set()  # Signal anyway
    
    async def _process_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a tool call using the MCP manager.
        This runs in the worker thread's event loop.
        """
        try:
            result = await self.mcp_manager.call_tool(tool_name, arguments)
            
            # Format result for better readability
            formatted_result = self._format_tool_result(result)
            
            if self._current_tool_calls is not None:
                self._current_tool_calls.append({
                    "name": tool_name,
                    "arguments": arguments,
                    "result": formatted_result
                })
            
            return {"result": formatted_result}
        except Exception as e:
            logger.error(f"Error processing tool call {tool_name}: {e}", exc_info=True)
            error_msg = f"Failed to execute {tool_name}: {str(e)}"
            
            if self._current_tool_calls is not None:
                self._current_tool_calls.append({
                    "name": tool_name,
                    "arguments": arguments,
                    "result": error_msg,
                    "error": True
                })
            
            return {"error": error_msg}
    
    def _format_tool_result(self, result: Any) -> str:
        """Format tool result for display, extracting content from MCP response."""
        try:
            # Handle MCP CallToolResult object
            if hasattr(result, 'content'):
                content_list = result.content
                if content_list:
                    # Extract text from TextContent objects
                    text_parts = []
                    for item in content_list:
                        if hasattr(item, 'text'):
                            text_parts.append(item.text)
                        else:
                            text_parts.append(str(item))
                    
                    combined_text = '\n'.join(text_parts)
                    
                    # Check if this is an error message
                    if hasattr(result, 'isError') and result.isError:
                        # Try to extract meaningful error from MCP error format
                        if 'MCP error' in combined_text or 'Input validation error' in combined_text:
                            # Extract just the core error message
                            if 'Input validation error:' in combined_text:
                                parts = combined_text.split('Input validation error:', 1)
                                if len(parts) > 1:
                                    error_details = parts[1].strip()
                                    # Try to parse as JSON for better formatting
                                    try:
                                        import re
                                        # Extract the validation errors
                                        match = re.search(r'Invalid arguments for tool (\w+):', error_details)
                                        tool_name = match.group(1) if match else 'unknown'
                                        return f"Tool '{tool_name}' error: Missing or invalid required parameters. Please check the input format."
                                    except:
                                        return f"Error: {combined_text}"
                        return f"Error: {combined_text}"
                    
                    return combined_text
            
            # Fallback to string representation
            return str(result)
        except Exception as e:
            logger.error(f"Error formatting tool result: {e}", exc_info=True)
            return str(result)

    async def chat(self, message: str) -> Dict[str, Any]:
        """
        Process a chat message using the agent
        
        Args:
            message: User's input message
            
        Returns:
            Dict with response text and metadata
        """
        if not self.agent:
            await self.initialize()

        # Reset tool call counter for this message
        self._tool_call_count = 0
        
        # Collect tool call details for this interaction
        self._current_tool_calls = []
        # Run the agent with the user's message
        response = await self.agent.ainvoke({
            "messages": [HumanMessage(content=message)]
        })
        # Extract the final response
        final_message = response["messages"][-1]

        tool_calls = self._current_tool_calls or []
        # Reset after use to avoid leaking across requests
        self._current_tool_calls = None

        return {
            "text": final_message.content,
            "sender": "assistant",
            "tool_calls": tool_calls
        }

    async def cleanup(self):
        """Cleanup MCP session"""
        if self.mcp_manager:
            await self.mcp_manager.cleanup()


# Global agent instance
agent_api = None


async def get_agent():
    """Get or create the global agent instance"""
    global agent_api
    if agent_api is None:
        agent_api = MCPAgentAPI()
        await agent_api.initialize()
    return agent_api


async def chat_with_agent(message: str) -> Dict[str, Any]:
    agent = await get_agent()
    return await agent.chat(message)


# FastAPI endpoints
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Chat endpoint for the LLM agent
    
    Args:
        request: ChatRequest with message and optional conversation_id
        
    Returns:
        ChatResponse with agent's reply
    """
    async def stream_response():
        try:
            response = await chat_with_agent(request.message)
            text = response["text"]
            sender = response["sender"]
            tool_calls = response.get("tool_calls", [])

            # Stream text as chunks (simple chunking for demo)
            chunk_size = 256
            for i in range(0, len(text), chunk_size):
                chunk = text[i:i+chunk_size]
                yield f"data: {json.dumps({'sender': sender, 'text': chunk})}\n\n"

            # Send tool_calls as final event
            yield f"data: {json.dumps({'tool_calls': tool_calls, 'conversation_id': request.conversation_id})}\n\n"
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            yield f"data: {json.dumps({'error': str(e), 'trace': error_detail})}\n\n"

    return StreamingResponse(stream_response(), media_type="text/event-stream")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "service": "llm-agent-api"}


async def shutdown_agent():
    """Cleanup agent resources"""
    global agent_api
    if agent_api:
        logger.info("Shutting down agent API...")
        agent_api.shutdown_event.set()
        # Wait for worker thread to finish (with timeout)
        # Worker thread will cleanup MCP manager
        logger.info("Agent API shutdown complete")


def handle_shutdown_signal(signum, frame):
    """Handle shutdown signals gracefully"""
    logger.info(f"Received signal {signum}, initiating shutdown...")
    global agent_api
    if agent_api:
        agent_api.shutdown_event.set()
    # Give the worker thread a moment to cleanup
    import time
    time.sleep(2)
    exit(0)


@app.on_event("shutdown")
async def on_shutdown():
    """FastAPI shutdown event handler"""
    await shutdown_agent()


if __name__ == "__main__":
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, handle_shutdown_signal)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, handle_shutdown_signal)
    
    server_config = config['llm']
    uvicorn.run(
        "llm_api:app",  # Import string instead of app object for reload
        host=server_config['host'],
        port=server_config['port'],
        reload=True,  # Auto-reload on code changes
        log_level="info"  # Verbose logging
    )
