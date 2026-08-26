
import httpx2
from mcp.server import MCPServer
from typing_extensions import Any

mcp = MCPServer("connectome")

API_BASE_URL = "https://qa.daybid.dev/connectome"
USER_AGENT = "connectome/0.1.0"

@mcp.tool(
    name="get_memory",
    description="Retrieve a memory by its key",
)
async def get_memory():
    pass

@mcp.tool(
    name="remember",
    description="Store a memory",
)
async def remember():
    pass

async def fetch_memory(key: str) -> Any:
    path = f"/memory?key={key}"
    return await make_connectome_request(path)


async def make_connectome_request(path: str) -> Any:
    headers = {"User-Agent": USER_AGENT}
    async with httpx2.AsyncClient() as client:
        try:
            response = await client.get(f"{API_BASE_URL}{path}", headers=headers)
            _ = response.raise_for_status()
            return response.json()
        except Exception as e:
            raise e
