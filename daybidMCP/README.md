## daybidMCP

Python MCP server for the Daybid connectome memory service.

### Prerequisites

- `uv` installed
- Node.js / `npx` (only for `uv run mcp dev`, which downloads the MCP Inspector)
- The Go backend running locally on `http://localhost:8080`
- A valid API key matching `backend/.env`

### Environment

Copy the example env file and set the API key:

```bash
cp .env.example .env
```

```dotenv
DAYBID_API_BASE_URL=http://localhost:8080/api/connectome
DAYBID_API_KEY=your-api-key
```

Set `DAYBID_API_KEY` to the same value as `apikey` in `backend/.env`.

The MCP server loads this file automatically on startup. `DAYBID_API_BASE_URL` defaults to `http://localhost:8080/api/connectome` if omitted.

The backend mounts every memory route under the `/api/connectome` prefix (`/api` base group + `/connectome` group). See the "API" section of the root [`README.md`](../README.md) for the canonical route list.

### Run locally

From the `daybidMCP/` directory:

```bash
uv sync
uv run daybidmcp
```

This starts the MCP server over `stdio`.

### Run with MCP dev tools

Verified on `mcp` 2.1.1 — the invocation is unchanged from 1.x. Requires
Node.js/`npx`, which `uv run mcp dev` uses to download and launch the MCP
Inspector on `http://127.0.0.1:6274`.

```bash
uv sync
uv run mcp dev connectome.py
```

### Start the backend

From the repo root:

```bash
docker compose up --build
```

The MCP tools call these authenticated backend routes:

- `GET /api/connectome/memory/?key=...`
- `POST /api/connectome/memory/`
- `POST /api/connectome/memory/batch`
- `POST /api/connectome/memory/batch/read`
- `GET /api/connectome/memory/list`
- `DELETE /api/connectome/memory/`
