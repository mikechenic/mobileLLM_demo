from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import yaml
import os
import logging
import httpx
from typing import Optional, List, Dict, Any

# Load configuration from config.yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Configure logging to match uvicorn's format and level
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:     %(message)s',
    force=True  # Override any existing config
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Frontend API", description="API for Frontend Requests")

# Enable CORS so React frontend can communicate with FastAPI backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request/response models
class MessageRequest(BaseModel):
    text: str

class MessageResponse(BaseModel):
    text: str
    sender: str
    tool_calls: Optional[List[Dict[str, Any]]] = None

# Endpoint to serve configuration to frontend
@app.get("/api/config")
async def get_config():
    """
    Returns the API configuration for the frontend
    """
    return {
        "api": config['api']
    }

# Dummy endpoint to receive user input and return assistant response
@app.post(config['api']['endpoints']['chat'])
async def chat(message: MessageRequest):
    """
    Receives a user message and streams the LLM response
    """
    user_input = message.text

    # Forward request to LLM service
    host = config['llm']['host']
    if not host.startswith("http://") and not host.startswith("https://"):
        host = "http://" + host
    llm_service_url = host + ":" + str(config['llm']['port']) + config['llm']['endpoints']['chat']

    async def proxy_stream():
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST",
                    llm_service_url,
                    json={"message": user_input},
                    headers={"accept": "text/event-stream"}
                ) as resp:
                    resp.raise_for_status()
                    async for chunk in resp.aiter_bytes():
                        yield chunk
        except Exception as e:
            yield f"data: {{\"error\": \"{str(e)}\"}}\n\n".encode()

    return StreamingResponse(proxy_stream(), media_type="text/event-stream")

@app.get(config['api']['endpoints']['health'])
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "ok"}

if __name__ == "__main__":
    server_config = config['api']
    uvicorn.run(
        "api:app", 
        host='0.0.0.0',
        port=server_config['port'], 
        reload=True,
        log_level="info"
    )
