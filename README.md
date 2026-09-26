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

2. Set `apikey` in `backend/.env`. Keep the same value available to MCP clients as `CONNECTOME_API_KEY`.

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

### GPU acceleration (optional)

By default `ollama` runs on the CPU, so the stack starts on any machine. Every
memory write and every `cmd/reindex` run embeds through Ollama, so a GPU makes
indexing much faster. GPU support is opt-in through a second compose file
layered on top of `docker-compose.yml`; only `ollama` gets the GPU
(`ollama-pull` just downloads the model).

**NVIDIA.** Prerequisites on the host:

1. The NVIDIA driver (`nvidia-smi` on the host should list your GPU).
2. The [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
   registered with Docker:

   ```bash
   sudo nvidia-ctk runtime configure --runtime=docker
   ```

   ```bash
   sudo systemctl restart docker
   ```

Then start the stack with the GPU file:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

**AMD (ROCm).** With the `amdgpu` kernel driver loaded (`/dev/kfd` and
`/dev/dri` present on the host), use `docker-compose.rocm.yml` instead. It
switches `ollama` to the `ollama/ollama:rocm` image and passes those devices
through:

```bash
docker compose -f docker-compose.yml -f docker-compose.rocm.yml up --build
```

Passing `-f` turns off Compose's automatic loading of
`docker-compose.override.yml`. If you keep a local override, list it too
(`-f docker-compose.yml -f docker-compose.override.yml -f docker-compose.gpu.yml`),
or set it once in the repo-root `.env` so plain `docker compose up` picks it up
(use `;` instead of `:` as the separator on Windows):

```bash
COMPOSE_FILE=docker-compose.yml:docker-compose.override.yml:docker-compose.gpu.yml
```

**Checking that it works.** On NVIDIA, the GPU should be visible inside the
container:

```bash
docker compose exec ollama nvidia-smi
```

Ollama logs the compute device it found at startup. Look for an
`inference compute` line naming your GPU (it says `library=cpu` when no GPU was
detected):

```bash
docker compose logs ollama | grep -i "inference compute"
```

After an embed request (for example the smoke check above), `ollama ps` shows
where the model is loaded: `100% GPU` in the `PROCESSOR` column.

```bash
docker compose exec ollama ollama ps
```

## Storage

Set `MEMORY_MANAGER` in `backend/.env` to one of:

- `local` (default in `backend/.env.example`): stores objects under `$HOME/.connectome`. Needs no AWS credentials, so `docker compose up --build` works out of the box.
- `s3`: stores objects in the S3 bucket named by `S3_BUCKET` (set in `backend/.env`). **Requires** `S3_BUCKET`, plus `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `AWS_REGION` in the root `.env` or your shell environment; the Compose file passes the AWS credentials through to the backend. The backend will fail to start if `MEMORY_MANAGER=s3` and `S3_BUCKET` or credentials are unavailable.

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
PATCH  /api/connectome/memory/relationship
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

### Rebuilding the search index

The `embeddings` table is derived, disposable state: everything it holds can
be reconstructed from the memory blob store (`mem_*.md` files) alone. To
rebuild it from scratch — for example, after dropping the table, or to
recover from a bad embedding run — with `docker compose up` already running:

```bash
cd backend
POSTGRES_HOST=localhost go run ./cmd/reindex
```

`backend/.env` sets `POSTGRES_HOST=postgres`, the compose service name, which
only resolves on the container network; the `postgres` service publishes
5432 to the host precisely so a host-run tool like this can override it to
`localhost` instead. (Running the whole backend the same way, `go run .`,
needs the same override.)

This truncates `embeddings` and re-chunks and re-embeds every memory in the
blob store, so it's safe to run against a table that already has rows in it.
Pass `-dry-run` to see how many memories would be indexed without touching
the table or calling the embedder:

```bash
cd backend
POSTGRES_HOST=localhost go run ./cmd/reindex -dry-run
```

**Upgrading from a version without embedding task prefixes:** stored chunks
are now embedded as `search_document: <text>` and search queries as
`search_query: <text>`, the task prefixes `nomic-embed-text` was trained
with (issue #9). Vectors written before this change were embedded from raw
text and don't match the new query embeddings, so run `cmd/reindex` once
after upgrading to rebuild them. (`POST /api/llm/embed` still embeds its
input as-is, with no prefix.)

### Hybrid search ranking

`POST /api/connectome/memory/search` ranks results with a blend of vector
similarity and Postgres full-text search, combined via reciprocal-rank
fusion (RRF): each signal contributes `weight / (k + rank)` to a chunk's
score, where `rank` is that chunk's position in that signal's own ranked
candidate list. This is what lets an exact name or rare term rank correctly
even when its embedding similarity alone is mediocre.

Blend weights are configurable via `backend/.env` (see `backend/.env.example`):

- `SEARCH_VECTOR_WEIGHT` (default `0.6`)
- `SEARCH_TEXT_WEIGHT` (default `0.4`)
- `SEARCH_RRF_K` (default `60`) - the RRF rank constant; higher values flatten
  the influence of rank position, so weights matter more than exact rank.

### Indexing concurrency

Each written memory's chunks are embedded in a single Ollama request.
`INDEX_CONCURRENCY` in `backend/.env` (default `4`) caps how many of those
embed requests are in flight at once across all writes, and how many files
one `POST /api/connectome/memory/batch` processes at once. Cancelling a
request cancels its in-flight embed and skips memories still waiting.

## MCP server

The MCP server exposes Connectome memory operations over stdio. Configure `connectomeMCP/.env`:

```bash
cp connectomeMCP/.env.example connectomeMCP/.env
```

```dotenv
CONNECTOME_API_BASE_URL=http://localhost:8080/api/connectome
CONNECTOME_API_KEY=test
```

Set `CONNECTOME_API_KEY` to the same value as `apikey` in `backend/.env`.

Then run it from the MCP directory:

```bash
cd connectomeMCP
uv sync
uv run connectomemcp
```

To run it with the MCP development inspector (verified on `mcp` 2.1.1 — the
invocation is unchanged from 1.x; it requires Node.js/`npx`, which downloads and
launches the MCP Inspector):

```bash
uv run mcp dev connectome.py
```

Run the MCP server tests with:

```bash
cd connectomeMCP
uv run pytest
```

The end-to-end test builds and runs the Go backend with the local storage
manager, so it needs the Go toolchain on `PATH` (it is skipped otherwise).

## Python client

`connectomeClient/` is a minimal async HTTP client for the `/api/connectome`
routes, independent of the MCP server and protocol. Applications that talk to
Connectome directly over HTTP (rather than via MCP) - a Discord bot, a script,
another service - depend on this package instead of vendoring their own copy
of the request/formatting logic. See [`connectomeClient/README.md`](connectomeClient/README.md).

## Project layout

- `backend/` — Go API, storage managers, authentication, and Ollama integration
- `connectomeMCP/` — Python MCP server and client-side memory formatting
- `connectomeClient/` — Python HTTP client library for applications that talk to Connectome directly (not via MCP)
- `evals/locomo/` — LoCoMo retrieval-recall eval harness (evidence recall@k per question category); see [`evals/locomo/README.md`](evals/locomo/README.md)
- `docker-compose.yml` — `backend`, `ollama`, and the one-shot `ollama-pull` model fetcher
- `docker-compose.gpu.yml` / `docker-compose.rocm.yml` — opt-in NVIDIA / AMD GPU overrides for `ollama`
- `docker-compose.eval.yml` — opt-in override that publishes Ollama on `127.0.0.1:11434` for the LoCoMo eval's answer/judge calls
- `.connectome/` — local memory volume used by the local storage manager

Application-specific logic built on top of the client (e.g. a Discord bot)
lives in its own repo and is not part of this one.
