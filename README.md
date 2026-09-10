# Connectome

Connectome is a memory service with a Go HTTP backend and an optional Python MCP server. Memories can be stored in Amazon S3 or on the local filesystem, and the backend can use Ollama for embeddings.

## Requirements

- Docker and Docker Compose for the containerized workflow
- Go 1.26+ for backend development
- Python 3.13+ and [`uv`](https://docs.astral.sh/uv/) for the MCP server
- AWS credentials when using the S3 memory manager

## Quick start

1. Create the environment files:

   ```bash
   cp .env.example .env
   cp backend/.env.example backend/.env
   ```

2. Set `apikey` in `backend/.env`. Keep the same value available to MCP clients as `DAYBID_API_KEY`.

3. Start the backend and Ollama:

   ```bash
   docker compose up --build
   ```

The API is available at `http://localhost:8080`. The repository’s `.connectome` directory is mounted into the backend container at `/root/.connectome`.

## Storage

Set `MEMORY_MANAGER` in `backend/.env` to one of:

- `s3` (default): stores objects in the `daybid-dev` S3 bucket.
- `local`: stores objects under `$HOME/.connectome`.

For S3, configure `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `AWS_REGION` in the root `.env` or your shell environment. The Compose file passes these values to the backend.

## API

All backend routes are served under the `/api` base group. Memory routes live under
the `/api/connectome` prefix and LLM routes under `/api/llm`. This is the canonical
statement of the API prefix; other docs and the MCP server's `DEFAULT_API_BASE_URL`
should match it.

Memory endpoints are authenticated with `Authorization: Bearer <apikey>`:

```text
GET    /api/connectome/memory/?key=<key>
POST   /api/connectome/memory/
POST   /api/connectome/memory/batch
POST   /api/connectome/memory/batch/read
GET    /api/connectome/memory/list
DELETE /api/connectome/memory/
POST   /api/llm/embed
```

For example:

```bash
curl -H "Authorization: Bearer test" \
  http://localhost:8080/api/connectome/memory/list
```

## Backend development

```bash
cd backend
go run .
```

Run the backend tests with:

```bash
cd backend
go test ./...
```

The backend loads `.env` from the current working directory when available.

## MCP server

The MCP server exposes Daybid memory operations over stdio. Configure `daybidMCP/.env`:

```dotenv
DAYBID_API_BASE_URL=http://localhost:8080/api/connectome
DAYBID_API_KEY=test
```

Then run it from the MCP directory:

```bash
cd daybidMCP
uv sync
uv run daybidmcp
```

To run it with the MCP development inspector:

```bash
uv run mcp dev connectome.py
```

## Project layout

- `backend/` — Go API, storage managers, authentication, and Ollama integration
- `daybidMCP/` — Python MCP server and client-side memory formatting
- `docker-compose.yml` — backend and Ollama services
- `.connectome/` — local memory volume used by the local storage manager
