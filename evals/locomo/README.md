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
2. Sends each QA question to `recall` with `k = max(--ks + [--answer-k])` and
   `hydrate: true`, and maps the returned memory keys back to dialog ids
   (memory keys are random UUIDs).
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
the per-category summary, skip counts, and a record for every question. Each
record has a stable `question_id` (`<sample_id>#<qa_index>`), the evidence and
retrieved dialog ids, the gold `answer` (categories 1-4) or
`adversarial_answer` (category 5, a plausible but wrong answer), and
`contexts`: each retrieved turn's `{dia_id, text}` in rank order. That's
everything an answering stage needs, so it can run offline from this file
without calling the backend.

Useful flags (`uv run locomo-eval --help` for all):

- `--samples conv-26,conv-30` - run a subset of conversations.
- `--ks 1,5,10,20` - k values to report (max 50, the backend's search cap).
- `--answer-k N` - retrieved turns a later answering stage will use (default
  10). Recall fetches at least this many, and `contexts` holds them all.
- `--run-id NAME` - fixed tome/result name instead of a timestamp.
- `--no-occurred-at` - leave `occurred_at` unset; the date stays in the text.
- `--concurrency N` - in-flight requests during ingestion and querying.
- `--keep-tomes` - leave the tomes in place and write
  `results/<run-id>.keys.json` (the memory key -> dialog id map).
- `--reuse-tomes RUN_ID` - skip ingestion and query the tomes a `--keep-tomes`
  run left behind (see below).

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

### Comparing search settings on one index

Search settings only affect querying, so ingest once and re-query the same
tomes after restarting the backend with each setting:

```bash
uv run locomo-eval --run-id base --keep-tomes
# restart the backend with SEARCH_TEXT_QUERY=or, then:
SEARCH_TEXT_QUERY=or uv run locomo-eval --run-id text-or --reuse-tomes base
uv run locomo-eval --cleanup base
```

As with the weights, `SEARCH_TEXT_QUERY` is recorded from the harness's
environment, so export the value the backend is running with.

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

### LLM calls (answering and judging)

Answering and judging run on the local Ollama that `docker compose` already
runs for embeddings, so no API key is needed. `locomo_eval.llm` is the
provider layer:

- Models are named `provider:exact-model-id`, e.g. `ollama:qwen3:8b`. Only
  `ollama` is implemented; hosted providers can be added behind the same
  interface without changing result files.
- Calls use Ollama's `/api/chat` at `OLLAMA_HOST` (default
  `http://localhost:11434`) with temperature 0, a fixed seed, a fixed
  `num_ctx`, bounded output and thinking off (any `<think>` block that still
  appears is stripped). Connection errors, timeouts and 5xx responses are
  retried with backoff; the per-call timeout is generous for CPU inference.
- The run config records each model's id, its digest (a tag like `qwen3:8b`
  can move to new weights; the digest pins what actually ran) and the
  sampling options.
- Usage is recorded per stage and model: calls, token counts, wall-clock
  seconds and output tokens/sec. `cost_usd` comes from
  [`pricing.toml`](pricing.toml), where `ollama:*` is 0; models it doesn't
  list record `null`.

The compose `ollama` service doesn't publish its port, and `ollama-pull`
only fetches the embedding model. From the repo root, layer on the eval
override and pull a chat model:

```bash
docker compose -f docker-compose.yml -f docker-compose.eval.yml up --build -d
docker compose exec ollama ollama pull qwen3:8b
```

Local Ollama serves one request at a time unless `OLLAMA_NUM_PARALLEL` is
set (the override passes it through), so keep LLM concurrency at 1-2.

### Answer and judge prompts

The prompts are versioned files in
[`src/locomo_eval/prompts/`](src/locomo_eval/prompts/), loaded with
`load_prompt("answer_v1")`, which returns `(text, version, sha256)`. The
answer stage records each prompt's version and sha256 in the run config
(`Prompt.config()`).

**Never edit a committed prompt version.** Changing the wording, even a typo,
means adding `answer_v2.txt` / `judge_v2.txt` and switching to it, so a
result's recorded version always names the exact text it ran with.

- `answer_v1` shows the retrieved turns sorted by date, each prefixed with its
  session date, and the question. It asks for a short answer, or exactly
  `Not mentioned in the conversation.` when the turns don't contain it.
- `judge_v1` follows the Mem0/LoCoMo judge: given the question, gold answer
  and generated answer it returns `{"reasoning": ..., "label": "CORRECT" |
  "WRONG"}`, lenient on phrasing and date format, strict on facts. For
  adversarial questions the gold answer is the abstention and the dataset's
  `adversarial_answer` is shown as a trap: only abstaining is CORRECT, and
  repeating the trap is WRONG.

The judge is a small local model, so its reply is constrained with Ollama
structured outputs (`format` set to the verdict's JSON schema) and then
parsed strictly - no fences, extra keys or free-text labels. A malformed
reply is retried once with the next seed at temperature 0.3 (at temperature 0
decoding is greedy, so a new seed alone would repeat the reply); if that fails
too the question gets `judge_label: null`.
Null verdicts are counted as `judge_null` next to the accuracy, which is over
judged questions only; they are never scored as CORRECT or WRONG.

### Scoring answers (`locomo-eval answer`)

Scoring is a second stage that runs offline over a finished retrieval run,
with no backend calls:

```bash
uv run locomo-eval --run-id base                  # 1. retrieval, stores contexts
uv run locomo-eval answer base \
  --answer-model ollama:qwen3:8b --judge-model ollama:qwen3:14b   # 2. answer + judge
```

For each question it answers from the top `answer_k` stored contexts
(default: the retrieval run's `--answer-k`; `--answer-k N` uses fewer), scores
the answer by token F1, then asks the judge for a verdict. A failed or killed
answer stage never re-ingests or re-queries, and one retrieval run can be
scored by several answer and judge configs over the same contexts.

**Setup.** Publish Ollama's port and pull the chat models as described under
[LLM calls](#llm-calls-answering-and-judging) (`docker-compose.eval.yml`, then
`docker compose exec ollama ollama pull <model>`). Model ids are
`ollama:<exact tag>`; `OLLAMA_HOST` points the harness at another Ollama
(default `http://localhost:11434`). The stage fails before any call if a model
isn't pulled.

**Results.** Scores are written into `results/<run-id>.json`, next to the
retrieval metrics, under `answers.<cfg-hash>`, where the hash covers the
models, prompt versions and sha256s, `answer_k` and sampling options:

- `config` - answer and judge model ids with their Ollama digests, both
  prompts' `{version, sha256}`, `answer_k`, temperature and options, the
  matched `pricing.toml` entries, and any `--samples`/`--limit` subset.
- `summary` - per category `n`, `f1`, `judge_acc` and `judge_null`, plus
  `overall_excl_adversarial` (comparable to most published LoCoMo numbers,
  which leave out category 5) and `overall`.
- `usage` - per stage: calls, tokens, seconds and `cost_usd`, over every call
  behind the scores, including ones reused from a checkpoint.
- `questions` - each question's `id`, `answer`, `f1`, `judge_label` and
  `judge_reasoning`.

The printed table puts the retrieval and answer columns side by side, with
`overall_excl_adversarial` before `overall`.

**Prompt versions.** `--answer-prompt` / `--judge-prompt` select a prompt
version (default `answer_v1` / `judge_v1`); see
[Answer and judge prompts](#answer-and-judge-prompts). A new version is a new
config, so its scores sit beside the old ones.

**Resuming.** Every finished stage call is appended to
`results/<run-id>.<cfg-hash>.jsonl` and fsynced, keyed by question, stage,
model, prompt version and sha256, a hash of the call's inputs, and the
sampling options. Rerunning the same command skips everything already there,
so a killed run loses only the calls in flight. The answer's input hash
covers the question and the contexts it was shown, so a changed context (say,
after re-running retrieval under the same id) forces a fresh answer, and a
changed answer forces a fresh verdict. Answers and verdicts are cached
separately and every checkpoint of the run is read, so changing only the
judge model or prompt reuses the answers already computed.

**Quick check.** `--limit N` scores the first N questions and `--samples
conv-26` one conversation; both reuse the same checkpoint, so a later full
run picks up where they stopped. Progress, with calls made, calls reused,
output tokens/sec and an ETA, is printed every few seconds.

**How long it takes.** A full run is ~1,980 questions x 2 calls (plus a rare
judge retry). Rough figures for an 8B answer model and a 14B judge, which vary
a lot with hardware:

| Setup | Per call | Full run |
| --- | --- | --- |
| CPU only (8-16 cores) | ~5-20 s | ~6-20 hours |
| GPU (`docker-compose.gpu.yml` / `docker-compose.rocm.yml`) | ~0.3-1 s | ~20-60 minutes |

Time `--samples conv-26` (~150 questions) first and scale by ~13x.

**Judge reliability.** A small local judge is noisier than the hosted judges
behind published LoCoMo numbers, so `judge_acc` compares our own runs with
each other, not with papers; token F1 is the number that compares with
published results. Use a judge at least as large as the answer model, ideally
from another model family, and spot-check its labels with `--judge-sample N`,
which prints N random judged items (question, gold, answer, label, reasoning).

### Extracting memories (`locomo-eval extract`)

The retrieval run stores every dialog turn as a memory, so it only tests
search. The extract stage tests what Connectome would hold if an agent chose
the memories: a local LLM reads each conversation and decides what to
remember. It **never sees the QA items**.

```bash
uv run locomo-eval extract --samples conv-26 --extractor-model ollama:qwen3:8b
```

Set up Ollama as under [LLM calls](#llm-calls-answering-and-judging); the GPU
override (`docker-compose.gpu.yml` / `docker-compose.rocm.yml`) is strongly
recommended, since each call generates a few thousand tokens.

**How it works.** Each conversation is extracted one session at a time, in
order. Each call sees the session transcript with its dialog ids
(`D3:7 Caroline: ...`) and date, the entities recorded in earlier sessions
(`id | name | kind`, so ids stay the same across sessions), and the
extraction prompt ([`extract_v1`](src/locomo_eval/prompts/extract_v1.txt)),
which is generic and says nothing about LoCoMo's question categories. It
returns entities and memories, each with:

- `content` - self-contained text, relative dates resolved from the session date;
- `memory_type` (`note`/`fact`/`preference`/`event`) and `occurred_at`;
- `entities` and `relationships` (`subjectEntityId`/`predicate`/`objectEntityId`/`kind`);
- `source_dia_ids` - provenance, used only for scoring.

The reply is constrained with Ollama structured outputs and validated. A reply
with the wrong shape is retried (`--attempts`, default 3), then recorded as a
failed session; the run carries on. The first attempt runs at temperature 0
with a 4,096-token output cap; retries take the next seed at temperature 0.3,
since at temperature 0 the seed is ignored and the reply would repeat. A reply
cut off at the cap (Ollama's `done_reason: "length"`, which otherwise shows up
as unterminated JSON) is retried with twice the cap, up to 8,192 tokens, and a
larger context if the prompt and cap no longer fit. Only retries change, so the
cache key - and every cached session - stays the same; each record also stores
the `attempt_options` its result came from and its `truncated_attempts`. Within a valid reply,
source ids not in the session, entity references to unknown entities,
relationships with unknown endpoints and unparseable `occurred_at` values are
dropped and counted under `dropped`. Entity ids are normalized to lowercase
with hyphens.

**Keeping the questions out.** The extractor's input is built from
`Sample.turns` only: `sessions_of` and `extract_sample` take a sample's turns,
not the sample, and the extractor interface has no parameter that could
carry QA items. A test checks that no question or answer text appears in any
prompt.

**Cache.** Results go to `results/extractions.jsonl` (`--cache`), one fsynced
line per session, keyed by dataset sha256, sample, session, extractor model,
prompt version and sha256, and sampling options. Each record also holds the
model's Ollama digest, the entity ids it was shown, token counts and wall
time. A rerun skips sessions already extracted - a fully cached run makes no
LLM calls and doesn't need Ollama - and retries failed ones. If an earlier
session is re-extracted, later cached sessions keep the entities they were
extracted with.

`--concurrency N` extracts N conversations at once; sessions within a
conversation always run in order, since each depends on the entities before it.
A new prompt version is a new cache key, so its results sit beside the old ones.

### Running the LLM stages on Kaggle

The `extract` and `answer` stages only need Ollama and the harness - not the
backend - so they can run on a free Kaggle GPU (a T4 x2 or P100, about 30 GPU
hours a week) while retrieval runs on your own machine.
[`kaggle.ipynb`](kaggle.ipynb) does the whole session: it installs Ollama and
uv, clones this repository, pulls the models, restores earlier results, runs
one stage under a time limit, and leaves `results/` as the version's output.

1. On Kaggle, verify your phone number, then create a **private** dataset
   (LoCoMo is CC BY-NC) with `locomo10.json` and, for the `answer` stage, the
   retrieval run's `results/<run-id>.json`.
2. Import the notebook, set **Accelerator: GPU** and **Internet: on**, and add
   the dataset as an input. For a private repository, add a GitHub token as
   the secret `GITHUB_TOKEN`.
3. Set `STAGE` and the models in the first cell, then **Save Version -> Save &
   Run All (Commit)**. The run continues with the browser closed.
4. Download `results/` from the version's Output tab.

To resume a stopped run, add the previous version's output as an input and
commit again: the notebook merges every attached `results/` folder back in,
and the stage skips work already cached. The stage stops itself after
`TIME_LIMIT_HOURS` (default 11) so the output is saved before Kaggle's
~12-hour limit. Time `SAMPLES = "conv-26"` first; a full run is about 13x that.
