## LoCoMo retrieval eval

Measures how well Connectome's `recall` retrieves the evidence for
[LoCoMo](https://github.com/snap-research/locomo) questions. Each LoCoMo QA
item lists the dialog turns that support its answer (e.g. `D1:3` = session 1,
turn 3), so retrieval is scored directly against those ids - no LLM involved.

### What it does

For each of the dataset's conversations:

1. Ingests every dialog turn as one `event` memory into its own scratch tome,
   `temp-locomo-<run-id>-<sample-id>`. The memory text is
   `[<session date>] <speaker>: <text>` (image turns get their caption
   appended), and `occurred_at` is set to the session date (UTC assumed, as
   the dataset has no timezone). `created_at` stays ingestion time.
2. Sends each QA question to `recall` with `k = max(--ks)` and maps the
   returned memory keys back to dialog ids (memory keys are random UUIDs).
3. Destroys the tome - also when the run fails or is interrupted.

It then reports, overall and per category (single-hop, multi-hop, temporal,
open-domain, adversarial):

- **recall@k** - mean fraction of a question's evidence turns in the top k.
- **hit@k** - fraction of questions with at least one evidence turn in the top k.

Questions with no usable evidence (a handful in `locomo10.json`) are skipped
and counted under `skipped` in the output. A few evidence strings in the
dataset are malformed (`"D8:6; D9:17"`, `"D:11:26"`) and are normalized
rather than dropped. Category names follow LoCoMo's own evaluation code
(1 multi-hop, 2 temporal, 3 open-domain, 4 single-hop, 5 adversarial).

### Dataset

The dataset is not vendored (it's licensed CC BY-NC 4.0). Download it into
`data/`, which is gitignored:

```bash
mkdir -p evals/locomo/data
curl -fsSL -o evals/locomo/data/locomo10.json https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
```

### Running

Start the backend (`docker compose up --build` from the repo root), then from
`evals/locomo/`:

```bash
uv sync
uv run locomo-eval
```

The harness talks to the backend through `connectomeclient`, so it reads the
same `CONNECTOME_API_BASE_URL` / `CONNECTOME_API_KEY` environment variables
(a `.env` in `evals/locomo/` is loaded if present). It prints a table and
writes `results/<run-id>.json` containing the run config (backend URL, git
commit, dataset sha256, k values, search weights, embedding model, chunking),
the per-category summary, skip counts, and every question's evidence and
retrieved dialog ids.

Useful flags (`uv run locomo-eval --help` for all):

- `--samples conv-26,conv-30` - run a subset of conversations.
- `--ks 1,5,10,20` - k values to report (max 50, the backend's search cap).
- `--run-id NAME` - fixed tome/result name instead of a timestamp.
- `--no-occurred-at` - leave `occurred_at` unset; the date stays in the text.
- `--concurrency N` - in-flight requests during ingestion and querying.

Ingestion embeds each turn, so a full run (~5,900 turns, ~1,980 scored
questions) takes a while on CPU-only Ollama; `--samples` is the quick loop.

### Vector-only vs hybrid

The search blend is configured on the backend, not per request: set
`SEARCH_TEXT_WEIGHT=0` in `backend/.env` and restart the backend for a
vector-only run. The backend doesn't expose its weights over HTTP, so export
the same `SEARCH_VECTOR_WEIGHT` / `SEARCH_TEXT_WEIGHT` / `SEARCH_RRF_K`
values when running the harness for them to be recorded correctly in the
result config (unset values are recorded as the backend defaults):

```bash
SEARCH_TEXT_WEIGHT=0 uv run locomo-eval --run-id vector-only
```

### Cleanup and reproducibility

Every run uses fresh tomes and destroys them when each conversation is scored,
so reruns start from an empty index and leave nothing behind. If a run is
killed hard (or used `--keep-tomes`), destroy its tomes with:

```bash
uv run locomo-eval --cleanup <run-id>
```

### Tests

```bash
uv run pytest
```

The tests cover dataset parsing, scoring, and the run loop against an
in-memory fake client; they don't need a backend.
