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

   Compose brings up three services: `ollama`, a one-shot `ollama-pull` that
   downloads the `nomic-embed-text` embedding model into the `./.ollama` volume,
   and `backend`. The `backend` service waits for `ollama-pull` to finish, so the
   first run blocks for a minute or two while the model downloads; later runs are
   fast because the model is already cached in `./.ollama`.

The API is available at `http://localhost:8080`. The repository’s `.connectome` directory is mounted into the backend container at `/root/.connectome`.

### Smoke check

Once `docker compose up --build` reports the `backend` container as healthy, verify the
memory loop end to end from a fresh clone:

```bash
# 1. Backend is reachable (no auth required).
curl -fsS http://localhost:8080/api/
# => {"message":"Hello, World!"}

# 2. Authenticated memory list works (uses the apikey from backend/.env).
curl -fsS -H "Authorization: Bearer test" \
  http://localhost:8080/api/connectome/memory/list
# => {"Contents":[...]}

# 3. Embeddings work (confirms ollama-pull fetched nomic-embed-text).
curl -fsS -H "Authorization: Bearer test" \
  -H "Content-Type: application/json" \
  -d '{"input":"hello world"}' \
  http://localhost:8080/api/llm/embed
# => {"embeddings":[0.01, -0.02, ...]}
```

`docker compose ps` should show `backend` as `healthy` and `ollama-pull` as `exited (0)`.

## Storage

Set `MEMORY_MANAGER` in `backend/.env` to one of:

- `local` (default in `backend/.env.example`): stores objects under `$HOME/.connectome`. Needs no AWS credentials, so `docker compose up --build` works out of the box.
- `s3`: stores objects in the `daybid-dev` S3 bucket. **Requires** `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `AWS_REGION` in the root `.env` or your shell environment; the Compose file passes these through to the backend. The backend will fail to start if `MEMORY_MANAGER=s3` and no credentials are available.

> Note: when `MEMORY_MANAGER` is unset, the backend falls back to `s3`. The provided `backend/.env.example` sets it to `local` explicitly.

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

```bash
cp daybidMCP/.env.example daybidMCP/.env
```

```dotenv
DAYBID_API_BASE_URL=http://localhost:8080/api/connectome
DAYBID_API_KEY=test
```

Set `DAYBID_API_KEY` to the same value as `apikey` in `backend/.env`.

Then run it from the MCP directory:

```bash
cd daybidMCP
uv sync
uv run daybidmcp
```

To run it with the MCP development inspector (verified on `mcp` 2.1.1 — the
invocation is unchanged from 1.x; it requires Node.js/`npx`, which downloads and
launches the MCP Inspector):

```bash
uv run mcp dev connectome.py
```

Run the MCP server tests with:

```bash
cd daybidMCP
uv run pytest
```

The end-to-end test builds and runs the Go backend with the local storage
manager, so it needs the Go toolchain on `PATH` (it is skipped otherwise).

## Project layout

- `backend/` — Go API, storage managers, authentication, and Ollama integration
- `daybidMCP/` — Python MCP server and client-side memory formatting
- `docker-compose.yml` — `backend`, `ollama`, and the one-shot `ollama-pull` model fetcher
- `.connectome/` — local memory volume used by the local storage manager
