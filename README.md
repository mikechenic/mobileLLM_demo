# i3-demo

## Goal
- Full-stack chat demo with streaming ReAct agent calling MCP tools to interact with containers
- Components: React UI (SSE), FastAPI gateway, FastAPI LLM/agent, MCP server with dynamic tools under `mcp_server/tools`
- Containerized via Dockerfiles for api, llm, and mcp services

## What’s implemented
- React chat UI with SSE streaming and tool-call rendering
- FastAPI gateway (`api.py`) and LLM agent service (`llm_api.py`) using LangChain ReAct and MCP tools
- MCP server (`mcp_server/`) with dynamic tool discovery from `mcp_server/tools/*/schema.json` + `handler.py`
- Transport abstraction for MCP (stdio/HTTP) and tool call limiting
- Dockerfiles using `uv` + `pyproject.toml`/`uv.lock` (no requirements.txt)

## Local setup (Python)
Prereqs: Python 3.11+, uv installed (`pip install uv`).

```bash
cd d:\development\i3_demo
uv sync --frozen
uv run python api.py        # gateway on :8000
uv run python llm_api.py    # LLM/agent on :8001
uv run python mcp_server/mcp_api.py  # MCP HTTP wrapper on :3000
```

## Docker build & run
Prereqs: Docker/Compose.

```bash
cd d:\development\i3_demo
docker-compose build
docker-compose up -d
```

Services (default):
- gateway: 8000
- llm/agent: 8001
- mcp server api: 3000

## MCP dynamic tools
- Add tools under `mcp_server/tools/<tool-name>/` with `schema.json` (input schema) and `handler.py` (function `handler`).
- Restart the MCP server to pick them up.

## Frontend
- React app entry: `index.js`, component `ChatInterface.js`; styles in `ChatInterface.css`.
- Use your preferred dev server/bundler (currently React + parcel cache present).

## Notes
- Python deps are defined in `pyproject.toml`; lock in `uv.lock`.
- Docker images install deps via `uv sync` into `.venv` inside the container.
