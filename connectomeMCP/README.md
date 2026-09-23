## connectomeMCP

Python MCP server for the Connectome connectome memory service.

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
CONNECTOME_API_BASE_URL=http://localhost:8080/api/connectome
CONNECTOME_API_KEY=your-api-key
```

Set `CONNECTOME_API_KEY` to the same value as `apikey` in `backend/.env`.

If the backend sits behind Cloudflare Access, also set a service token (both are required; the headers are skipped otherwise):

```dotenv
CF_ACCESS_CLIENT_ID=your-service-token-id.access
CF_ACCESS_CLIENT_SECRET=your-service-token-secret
```

The MCP server loads this file automatically on startup. `CONNECTOME_API_BASE_URL` defaults to `http://localhost:8080/api/connectome` if omitted.

The backend mounts every memory route under the `/api/connectome` prefix (`/api` base group + `/connectome` group). See the "API" section of the root [`README.md`](../README.md) for the canonical route list.

### Run locally

From the `connectomeMCP/` directory:

```bash
uv sync
uv run connectomemcp
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
- `PATCH /api/connectome/memory/relationship` — used by `supersede_relationship` to set (or clear) `superseded_by` on one relationship entry in place, without re-embedding the memory.
