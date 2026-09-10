## daybidMCP

Python MCP server for the Daybid connectome memory service.

### Prerequisites

- `uv` installed
- The Go backend running locally on `http://localhost:8080`
- A valid API key matching `backend/.env`

### Environment

Create `daybidMCP/.env` with:

```dotenv
DAYBID_API_BASE_URL=http://localhost:8080/api/connectome
DAYBID_API_KEY=your-api-key
```

The MCP server loads this file automatically on startup. `DAYBID_API_BASE_URL` defaults to `http://localhost:8080/api/connectome` if omitted.

The backend mounts every memory route under the `/api/connectome` prefix (`/api` base group + `/connectome` group). See the "API" section of the root [`README.md`](../README.md) for the canonical route list.

### Run locally

From the MCP project directory:

```bash
cd /home/dfavela/src/DaybidDev/daybidMCP
uv sync
uv run daybidmcp
```

This starts the MCP server over `stdio`.

### Run with MCP dev tools

```bash
cd /home/dfavela/src/DaybidDev/daybidMCP
uv sync
uv run mcp dev connectome.py
```

### Start the backend

From the repo root:

```bash
cd /home/dfavela/src/DaybidDev
docker compose up --build
```

The MCP tools call these authenticated backend routes:

- `GET /api/connectome/memory/?key=...`
- `POST /api/connectome/memory/`
- `POST /api/connectome/memory/batch`
- `POST /api/connectome/memory/batch/read`
- `GET /api/connectome/memory/list`
- `DELETE /api/connectome/memory/`
