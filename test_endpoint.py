import requests
import yaml
import asyncio

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
api = config['api']['host'] + ":" + str(config['api']['port'])
chat_endpoint = api + config['api']['endpoints']['chat']
health_endpoint = api + config['api']['endpoints']['health']
llm_endpoint = config['llm']['host'] + ":" + str(config['llm']['port']) + config['llm']['endpoints']['chat']

method = "POST"

def test_chat_endpoint():
    url = chat_endpoint
    payload = {"text": "Hello, bot!"}
    response = requests.post(url, json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "text" in data
    assert data["sender"] == "bot"
    print("Chat Endpoint Response:", data)

def test_health_endpoint():
    url = health_endpoint
    response = requests.get(url)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    print("Health Endpoint Response:", data)

async def test_mcp_endpoint():
    from mcp_manager import MCPServerManager
    mcp_config = config['mcp']
    mcp_manager = MCPServerManager.create_from_config(mcp_config)
    await mcp_manager.initialize()
    tools_list = await mcp_manager.list_tools()
    print("MCP Tools List:", tools_list)

    await mcp_manager.cleanup()

def test_llm_api_endpoint():
    url = "http://localhost:8001/chat"
    payload = {"message": "Hello, LLM!"}
    response = requests.post(url, json=payload)
    assert response.status_code == 200
    print("LLM API Endpoint Response:", response.text)

if __name__ == "__main__":
    # test_chat_endpoint()
    # test_health_endpoint()
    asyncio.run(test_mcp_endpoint())
    # test_llm_api_endpoint()