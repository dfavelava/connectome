## connectomeclient

Minimal async Python HTTP client for the Connectome memory service's
`/api/connectome` routes. Talks to the Go backend directly over HTTP - no
dependency on `connectomeMCP` or the MCP protocol - so any Python application
(a Discord bot, a script, another service) can read and write Connectome
memories without embedding its own copy of the request/formatting logic.

### Install

From another project, add it as a `uv`/pip git dependency pointed at this
subdirectory, e.g. in `pyproject.toml`:

```toml
dependencies = [
    "connectomeclient @ git+https://github.com/dfavelava/connectome#subdirectory=connectomeClient",
]
```

### Usage

```python
from connectomeclient import ConnectomeClient

client = ConnectomeClient(
    base_url="http://localhost:8080/api/connectome",  # or CONNECTOME_API_BASE_URL
    api_key="your-api-key",                            # or CONNECTOME_API_KEY
    source_type="my-app",                               # tags memories with where they came from
)

await client.remember("David likes tea.", entities=["david"])
results = await client.recall("what does david drink")
```

`base_url` defaults to `http://localhost:8080/api/connectome` and `api_key`
falls back to the `CONNECTOME_API_KEY` (or legacy `DAYBID_API_KEY` / `apikey`)
environment variable if not passed explicitly. This package does not load a
`.env` file itself - the calling application is responsible for loading its
own environment (e.g. via `python-dotenv`) before constructing a client.

### Tomes

A tome is an isolated namespace: memories written to one tome are invisible to
reads, searches, and listings of any other. Every method that touches memory
data takes an optional `tome` argument (`remember`, `recall`, `get_memory`,
`get_entity`, `browse_all`, `forget`, `supersede_relationship`,
`assert_relationship`); omit it to use the default tome. A memory must be read
and deleted with the same `tome` it was written with.

```python
await client.remember("Caroline attended a support group.", tome="temp-locomo-1")
await client.recall("support group", tome="temp-locomo-1")

# Irreversible: removes every memory, entity record, and embedding in the tome.
await client.destroy_tome("temp-locomo-1")
```

`destroy_tome` never touches the default tome, and the backend only destroys
tome ids starting with `temp-` or `test-` unless you pass `confirm=True`
(anything else raises `httpx.HTTPStatusError` with a 403). Use a `temp-`
prefix for scratch data such as eval runs.

See the root [`README.md`](../README.md#api) for the canonical list of
backend routes this client calls.

### Development

From the `connectomeClient/` directory:

```bash
uv sync
uv run pytest
```
