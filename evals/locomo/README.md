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
3. Destroys the tomes - also when the run fails or is interrupted.

These are the `ingest`, `retrieve` and `cleanup` [stages](#stages), which the
default command runs back to back and which can also run one at a time.

It then reports, overall and per category (single-hop, multi-hop, temporal,
open-domain, adversarial):

- **recall@k** - mean fraction of a question's evidence turns in the top k.
- **hit@k** - fraction of questions with at least one evidence turn in the top k.
- **recall@Bt** - recall at an equal budget of B tokens of memory text, and
  **coverage** - see [Scoring extracted memories](#scoring-extracted-memories---ingest-extracted).

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
- `--budgets 64,128,256` - token budgets for equal-budget recall (`0` for none).
- `--ingest extracted` - ingest a cached extraction instead of the turns (see
  [below](#scoring-extracted-memories---ingest-extracted)).
- `--answer-k N` - retrieved turns a later answering stage will use (default
  10). Recall fetches at least this many, and `contexts` holds them all.
- `--run-id NAME` - fixed tome/result name instead of a timestamp.
- `--no-occurred-at` - leave `occurred_at` unset; the date stays in the text.
- `--concurrency N` - in-flight requests during ingestion and querying.
- `--keep-tomes` - skip cleanup, leaving the tomes for `locomo-eval retrieve`.
- `--reuse-tomes RUN_ID` - the same as `locomo-eval retrieve RUN_ID`, but
  writing `results/<run-id>.json` (see [Stages](#stages)).

Ingestion embeds each turn, so a full run (~5,900 turns, ~1,980 scored
questions) takes a while on CPU-only Ollama; `--samples` is the quick loop.

### Search settings

Ranking settings are sent per request, so no backend restart is needed.
`--ranking` takes a JSON object or repeatable `key=value` pairs, applied to
every query; anything left out keeps the backend's env or default value
(`SEARCH_*` in `backend/.env`). Keys: `vector_weight`, `text_weight`, `rrf_k`,
`text_query` (`plain`, `websearch`, `or`, `and_or`, `bm25`, `rare_or`),
`bm25_k1`, `bm25_b`, `text_max_df`. The backend rejects bad values with a 400.

The result config's `search` block is the ranking the backend echoed in its
search responses, so it records what actually ran, not this process's
environment. A backend from before per-request ranking echoes nothing, and
`search` is then `null`.

```bash
uv run locomo-eval --run-id vector-only --ranking text_weight=0
```

### Stages

The default command is three stages, each of which can also run on its own.
They pass data through files in `results/`, like `extract` and `answer`:

| Stage | Command | Reads | Writes |
|---|---|---|---|
| extract | `locomo-eval extract` | the dataset | `results/extractions.jsonl` ([below](#extracting-memories-locomo-eval-extract)) |
| ingest | `locomo-eval ingest --run-id base` | the dataset (and extractions) | the tomes, `results/base.ingest.json` |
| retrieve | `locomo-eval retrieve base [--tag NAME]` | `base.ingest.json`, the tomes | `results/base.json` or `results/base.<tag>.json` |
| answer + judge | `locomo-eval answer base [--tag NAME]` | the retrieval results | answers in the same file ([below](#scoring-answers-locomo-eval-answer)) |
| cleanup | `locomo-eval cleanup base` | `base.ingest.json` | destroys the tomes |

```bash
uv run locomo-eval ingest --run-id base [--ingest turns|extracted] [--samples ...]
uv run locomo-eval retrieve base [--ks 1,5,10] [--answer-k 10] [--ranking ...] [--tag bm25]
uv run locomo-eval answer base --answer-model ... --judge-model ...
uv run locomo-eval cleanup base
uv run locomo-eval          # ingest -> retrieve -> cleanup, as always
```

**`ingest`** fills `temp-locomo-<run-id>-<sample-id>` and leaves the tomes in
place. It takes everything about how they are filled: `--ingest`,
`--superseded`, the extraction cache options, `--no-occurred-at`,
`--embedding-model` and `--concurrency`. Its manifest,
`results/<run-id>.ingest.json`, holds the key map (memory key -> source dialog
ids, written after supersession, so memories `--superseded forget` removed
are gone from it), the dataset sha256, git commit, embedding model,
`ingestion` (the mode and, for extracted runs, the extraction config with its
`variant` and `recall` block), `chunking` (with `superseded`), per-sample
memory counts, and whether the tomes are still `kept`. It refuses a run id
whose tomes are still kept, since ingesting on top would mix the two.

**`retrieve <run-id>`** scores the tomes without ingesting anything. How they
were filled - mode, lifecycle variant, `superseded` - comes from the manifest,
so it can't be given differently on the command line. It stops with a clear
message if the manifest is missing, the tomes were cleaned up or are gone from
the backend, `--data` isn't the dataset that was ingested, or `--samples`
names a conversation the ingest didn't. `--samples` defaults to all the
ingested ones. `--tag NAME` writes `results/<run-id>.<tag>.json`, so several
retrieval configs can sit next to one ingest; pass `--tag` to `answer` to
score one. Results have the same format as the default command's, with
`reused_tomes` naming the ingest run.

**`cleanup <run-id>`** destroys the tomes the manifest lists (one per dataset
sample, or `--samples`, when there is no manifest) and marks the manifest
`destroyed`. `--cleanup RUN_ID` still works, as a deprecated alias.

**The default command** runs ingest, retrieve and cleanup in one process.
Its results are the same as before the split, plus the manifest.
`--keep-tomes` skips cleanup, and `--reuse-tomes X` is `retrieve X`, writing
`results/<run-id>.json`. A `--keep-tomes` run from before the manifest (a
`<run-id>.keys.json` next to its results) can still be reused and cleaned up.

Search settings only affect querying, so ingest once and sweep them:

```bash
uv run locomo-eval ingest --run-id base
uv run locomo-eval retrieve base --tag plain
uv run locomo-eval retrieve base --tag text-or --ranking text_query=or --compare base.plain
uv run locomo-eval retrieve base --tag bm25 --ranking '{"text_query": "bm25", "bm25_k1": 1.5}' --compare base.plain
uv run locomo-eval cleanup base
```

### Cleanup and reproducibility

Every run uses fresh tomes and destroys them once it has scored them, so
reruns start from an empty index and leave nothing behind. A failed ingest
destroys the tomes it wrote. If a run is killed hard, or you ran `ingest` or
`--keep-tomes`, destroy its tomes with:

```bash
uv run locomo-eval cleanup <run-id>
```

### Tests

```bash
uv run pytest
```

The tests cover dataset parsing, scoring, and the ingest, retrieve and cleanup
stages against an in-memory fake client; they don't need a backend.

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
- `answer_v2` (`--answer-prompt answer_v2`; the default stays `answer_v1`)
  fixes what `answer_v1` gets wrong with an 8B model answering. It describes
  both kinds of context: a turn (`[date] speaker: text`) and an extracted
  memory (a fact on its own, with its own date), so `--ingest extracted` runs
  are no longer scored with a prompt that doesn't describe their input. It
  spells out the date arithmetic with examples and one worked example: a
  message that says "yesterday", "last week" or "last Sunday" is answered with
  the date worked out from the session date (`11 March 2023`, `the week
  before 9 June 2023`), never the session date itself, which is what most
  wrong `answer_v1` temporal answers were. And it lets the model answer with
  a likely answer the excerpts support instead of abstaining, while checking
  the excerpts are about the person and thing asked about, and abstaining
  with the same exact text when they say nothing relevant.
- `judge_v1` follows the Mem0/LoCoMo judge: given the question, gold answer
  and generated answer it returns `{"reasoning": ..., "label": "CORRECT" |
  "WRONG"}`, lenient on phrasing and date format, strict on facts. For
  adversarial questions the gold answer is the abstention and the dataset's
  `adversarial_answer` is shown as a trap: only abstaining is CORRECT, and
  repeating the trap is WRONG. It is stricter than the Mem0/LoCoMo judge
  behind published numbers: a list with one extra or missing item, a date
  more specific than the gold one, or an answer without the gold's qualifier
  is WRONG, which hits multi-hop list questions hardest.
- `judge_v2` (`--judge-prompt judge_v2`; the default stays `judge_v1`)
  aligns the leniency with the Mem0/LoCoMo judge, which accepts an answer on
  the same topic as the gold. A date more specific than the gold period
  is CORRECT when it falls within it (gold `July 2023`, answer `3 July 2023`),
  and a date outside it is WRONG. A list is CORRECT when it contains at least
  one gold item and nothing that contradicts the gold answer, so extra items
  and missing items are both accepted, and a list with none of the gold items
  is WRONG. An answer that gets the core fact right but leaves out a qualifier
  is CORRECT unless the question asks for that qualifier. Wrong facts, the
  adversarial rule, the output format and the strict parsing are the same as
  in `judge_v1`. Its examples use a made-up person, so none of them is also
  an item being judged.

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
`--tag NAME` scores a tagged retrieval, `results/<run-id>.<tag>.json` from
`locomo-eval retrieve --tag`; its answer checkpoints are its own.

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

**Thinking.** `--answer-think` lets the answer model think before it answers
(the judge's options are unchanged). Thinking tokens count against the output
cap, so the answer call's `num_predict` is raised to 4096. It changes the
config hash (the config records the answer call's options as
`answer_options`), so compare its `usage.answer.seconds` per call against the
accuracy gain over the same prompt without it.

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

**Checking a judge against hand labels.** A judge prompt can't be validated
by its own scores, so `locomo-eval judge-agreement` compares every judge that
scored a run with a hand-labelled sample, offline:

```bash
# 1. Draw a labelling sheet from one answer config (weighted towards multi-hop and temporal).
uv run locomo-eval judge-agreement <run-id> --draw --config <cfg-hash> \
  --samples conv-26 --out labels/judge-<run-id>-conv-26.jsonl
# 2. Fill in each item's "label" (CORRECT or WRONG, under judge_v2's rules).
# 3. Rescore with the other judge prompt (reuses the cached answers), then compare.
uv run locomo-eval answer <run-id> ... --judge-prompt judge_v2
uv run locomo-eval judge-agreement <run-id> --labels labels/judge-<run-id>-conv-26.jsonl
```

The sheet has each item's question, gold answer and generated answer, but
not the judge's verdict, so labelling stays blind; `--weights` sets the
questions per category (default `multi-hop=20,temporal=20,single-hop=10,
open-domain=5,adversarial=5`) and `--seed` the draw. The report gives, per
answer config and category, the agreement with the hand labels, false
CORRECTs (the judge accepted an answer labelled WRONG), false WRONGs and null
verdicts. A label only counts against a config that judged the identical
answer, so configs that differ only in the judge are compared on the same
items. Sheets quote LoCoMo questions and answers, which are CC BY-NC, so
they are kept out of git: `labels/*.jsonl` is gitignored (see
[`labels/`](labels/)). Keep a sheet with the run's results, since its labels
only match the answers it was drawn from.

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

**`extract_v2`.** [`extract_v2`](src/locomo_eval/prompts/extract_v2.txt) is
the second add-only prompt (`--extract-prompt extract_v2`; the default stays
`extract_v1`). `extract_v1` memories retrieve better than raw turns but answer
worse, because the text often loses or changes the fact it cites. v2 makes
exactness the first rule, with a worked example: the speaker's own specifics,
no similar item swapped in, no list turned into a category. It also puts an
absolute date in the text of every memory tied to a time, sets `occurred_at`
to the event's own date (the start of a period, or null, never the session
date by default), keeps reasons, feelings, frequencies and replies as memories
of their own, discourages summary memories, and reuses an earlier entity id
only for the same thing. The wording stays generic, and the same tests check
it as v1.

### Scoring extracted memories (`--ingest extracted`)

Once a conversation's extraction is cached, the retrieval run can ingest those
memories instead of the raw turns and score them the same way:

```bash
uv run locomo-eval --run-id turns                       # the raw-turn baseline
uv run locomo-eval --run-id extracted --ingest extracted \
  --extractor-model ollama:qwen3:8b --compare turns
```

`--ingest extracted` reads the extraction cache (`--extraction-cache`,
`--extractor-model`, `--extract-prompt` pick the config, as in the extract
stage) and makes no LLM calls. Each memory is written with `remember` as it
was extracted: content, memory type, entities, relationships, `occurred_at`
(unless `--no-occurred-at`) and the run's tome. Sessions whose extraction
failed or never ran are left out with a warning; a conversation with no cached
extraction at all stops the run. The default stays `--ingest turns`, so
earlier results remain comparable.

Every recall hit maps back to its memory's `source_dia_ids`, so the key map is
memory key -> source dialog ids (one id per key for turns). The stages work in
both modes: `retrieve` takes the mode from the ingest manifest, and
`--reuse-tomes` refuses an `--ingest` or `--superseded` that disagrees with
it.

Metrics, per category (see [`metrics.py`](src/locomo_eval/metrics.py)):

- **coverage** - the fraction of a question's evidence turns cited by *any*
  memory in the conversation, retrieved or not. It's the most retrieval could
  find, so it splits what extraction lost from what retrieval lost. It is 1.0
  for turns.
- **recall@k / hit@k** - k counts memories; the top k are expanded to their
  source turns in rank order, duplicates removed. For turns this is the same
  number as before.
- **recall@Bt** - recall over the top-ranked memories whose text fits in B
  tokens (`--budgets`, default `64,128,256`; a memory that would overflow
  ends the list). A memory citing many turns inflates recall@k, so this is
  the fair comparison between modes. Tokens are approximated as words plus
  punctuation marks, alike for both modes. `underfilled@Bt` is the fraction
  of questions whose whole retrieved list fit in fewer than B tokens - raise
  `--ks` or `--answer-k` to fetch more before trusting that budget.

The results also hold `memories` (per sample and overall: memory count,
sources per memory, entities per conversation), and the config records
`ingestion.mode` and, for extracted runs, `ingestion.extraction`: the cache,
extractor model and digests, prompt version and sha256, sampling options,
and totals from the cache (sessions extracted, failed and never extracted,
memories, attempts, tokens, seconds, and dropped ids by reason).

`--compare RUN_ID` prints another finished run's metrics, recomputed with this
run's `--ks` and `--budgets`, under each category's row. A run from before
coverage was recorded prints `-` there.

Recall ranks on memory content only today: entities and relationships are
stored but don't affect ranking, so this measures extracted text against raw
turns, not the graph. Storing them now lets graph-expanded recall be measured
later on the same cached extraction.

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
3. Set `STAGE` and the models in the first cell (and, for `extract`,
   `EXTRACT_PROMPT`: `extract_v1`, `extract_v2`, `lifecycle_v1`, `lifecycle_v2`, `lifecycle_v3` or `lifecycle_v4`; for
   `answer`, `RUN_ID`, `ANSWER_PROMPT`, `ANSWER_K` and `ANSWER_THINK`), then **Save Version ->
   Save & Run All (Commit)**. The run continues with the browser closed.
4. Download `results/` from the version's Output tab.

To resume a stopped run, add the previous version's output as an input and
commit again: the notebook merges every attached `results/` folder back in,
and the stage skips work already cached. The stage stops itself after
`TIME_LIMIT_HOURS` (default 11) so the output is saved before Kaggle's
~12-hour limit. Time `SAMPLES = "conv-26"` first; a full run is about 13x that.

### Lifecycle extraction: recall before writing (`lifecycle_v1`)

The extractor above only adds memories; it never looks at what's already
stored. The lifecycle variant does what an agent that recalls before writing
would: it avoids duplicates and replaces claims that no longer hold. It is
selected by its prompt, [`lifecycle_v1`](src/locomo_eval/prompts/lifecycle_v1.txt),
so it is cached beside the add-only extraction under its own prompt version:

```bash
uv run locomo-eval extract --samples conv-26 --extractor-model ollama:qwen3:8b                                  # add-only
uv run locomo-eval extract --samples conv-26 --extractor-model ollama:qwen3:8b --extract-prompt lifecycle_v1    # lifecycle
```

On Kaggle, set `EXTRACT_PROMPT = "lifecycle_v1"` in the notebook's first cell.
Commit the add-only and lifecycle runs as separate versions with the same
`EXTRACTOR_MODEL` and `SAMPLES`, attaching the earlier version's output so both
end up in one `extractions.jsonl`, then score them locally as below.

**How it works.** Sessions still run in order. Before each one, the
conversation's own earlier memories - what its tome would hold after the
sessions so far, less anything already superseded - are searched with each
turn as a query (BM25 on the memory text; up to 3 hits per turn and 30 in all,
`RECALL_PER_TURN` / `RECALL_LIMIT`). The hits are shown in the prompt as
`M<session>.<n> | <occurred_at date> | <content>`. Besides entities and
memories, the reply has:

- `supersedes` on each new memory - ids of stored memories it replaces
  because they are no longer true (a plan carried out or cancelled, a move, a
  new job);
- `duplicates` - stored memories the session only repeats, with the turns
  that repeat them. Nothing is written for them.

Only recalled ids can be referenced; other ids, a memory superseded twice, or
one both superseded and repeated are dropped and counted under
`dropped.memory_refs`. Each record also stores `recalled_memory_ids`. The
recall is local and deterministic rather than a call to the backend's
`recall`, so extraction still needs no backend and a cached session stays
valid; changing the recall limits means a new prompt version. The extract
table gains `duplicates`, `verbatim`, `repeats` and `superseded` columns
(always 0 for add-only).

**Copies.** The extractor sometimes supersedes a memory with one of exactly
the same text, where it should have listed a duplicate, or writes again a
memory that is already stored. When the records are applied, copies are
folded into the memory they copy (normalized text: case and whitespace
ignored):

- a memory whose text equals one it supersedes is a *verbatim supersede*: it
  isn't written, and the old memory isn't superseded;
- a memory whose text equals a current memory - stored and not superseded,
  or earlier in the same reply - is a *repeat* and isn't written either.

Anything else a copy supersedes is superseded by the memory it copies, and
later references to a copy go to that memory. Like a duplicate, a copy's
turns are cited by nothing. The cache records keep the copies, and recall
during extraction sees them as extracted, so an existing cache is reused
as is. `superseded` counts what is superseded once copies are folded.

**Scoring.** `--ingest extracted --extract-prompt lifecycle_v1` ingests it
like any extraction. Every memory is written, then each superseded one is
handled per `--superseded`:

- `mark` (default) - each of its relationships gets `superseded_by` the new
  memory's key (`supersede_relationship`), as an agent would do today. Recall
  doesn't filter on `superseded_by` yet, so the old memory can still be
  retrieved.
- `forget` - it is forgotten, so recall can't return it and it no longer
  counts towards coverage.

Duplicates aren't written, so a turn that only repeats an earlier memory is
cited by nothing; if a question's evidence is that later turn, coverage drops.
That cost is part of what the comparison measures.

**Comparing the variants.** Run both on the same samples and extractor model,
then compare per category:

```bash
uv run locomo-eval --run-id add-only --ingest extracted --extractor-model ollama:qwen3:8b
uv run locomo-eval --run-id lifecycle --ingest extracted --extractor-model ollama:qwen3:8b \
  --extract-prompt lifecycle_v1 --compare add-only
uv run locomo-eval --run-id lifecycle-forget --ingest extracted --extractor-model ollama:qwen3:8b \
  --extract-prompt lifecycle_v1 --superseded forget --compare add-only
```

With `--compare`, each category shows this run, the baseline, and a `diff`
row (this run minus the baseline). Runs are labelled by what they ingested,
e.g. `extracted: add-only` or `extracted: lifecycle, forget`. The config
records `ingestion.extraction.variant`, its `recall` settings, the
`duplicates`, `verbatim_supersedes`, `repeats` and `superseded` totals, and
`chunking.superseded`. The answer
stage (`locomo-eval answer <run-id>`) scores each run the same way, so the
temporal category's judge accuracy can be compared as well.

**`lifecycle_v2`.** On all 10 conversations `lifecycle_v1` scored below
add-only `extract_v1` (judge accuracy excluding adversarial 0.265 against
0.322), mostly because its memories drop the detail a question needs: a date,
a list's items, the actual reply, or a fact folded into a merged summary.
[`lifecycle_v2`](src/locomo_eval/prompts/lifecycle_v2.txt)
(`--extract-prompt lifecycle_v2`) starts from `extract_v2`'s rules instead of
`extract_v1`'s, word for word, and adds to them:

- a memory that supersedes or adds to a stored one keeps every date, name,
  list item and quantity of both, and never generalizes a dated event;
- new details are written as a memory of their own, not merged with the
  stored memory into a combined summary;
- a stored memory is superseded only when it is no longer true; one the new
  memory would only restate is listed under `duplicates`;
- a memory cites the few turns it comes from, not a whole session.

Its fair baseline is an add-only `extract_v2` run with the same extractor
model.

**`lifecycle_v3`.** On all 10 conversations `lifecycle_v2` stopped verbatim
supersedes and many-turn memories, but stored the gold answer no more often
than `lifecycle_v1` (37.7% of non-adversarial questions against 38.0%; add-only
44.3%). It cited the evidence turns as often as add-only in the first session,
when nothing is stored yet, but only 70% of them from the sixth session on: it
skipped messages it judged already stored and listed them nowhere. It also began
every memory "As of <session date>", often as `2023-10-04`, dating events by
when they were mentioned. [`lifecycle_v3`](src/locomo_eval/prompts/lifecycle_v3.txt)
(`--extract-prompt lifecycle_v3`) keeps v2 and changes three things:

- every message worth remembering must be cited by a new memory or listed in
  a duplicate, and a stored memory on the same topic is no reason to skip one;
- a stored memory is a duplicate only when it already states every specific
  of the message;
- an event is dated with its own date, in words ("on 3 October 2023"), with a
  worked example; "As of" is kept for ongoing states.

**`lifecycle_v4`.** On conv-26 and conv-30 `lifecycle_v3` brought coverage
back (gold in store 42.9% of non-adversarial questions, add-only 34.3%) but
still answered below add-only (judge accuracy excluding adversarial 0.260
against 0.299), with temporal the furthest behind (0.111 against 0.175). 71% of
its memories still began "As of <session date>", and relative times were
resolved no more often than with add-only, so a memory such as "As of 8 May
2023, Caroline attended a LGBTQ support group on the previous day" led
`answer_v1` to give the session date instead of the event's.
[`lifecycle_v4`](src/locomo_eval/prompts/lifecycle_v4.txt)
(`--extract-prompt lifecycle_v4`) keeps v3 and changes only the date rule:

- no relative time ("yesterday", "the previous day", "next month", ...) may
  be left in a memory, and the session date is never written beside the date
  it was resolved to;
- no memory starts with "As of": an event takes the day it happened or is
  planned for, a vaguer time becomes a period ("the week before 4 October
  2023"), and an ongoing state says when it began ("since about April 2023")
  or ends with "(mentioned on <session date>)";
- worked conversions for one session date, and wrong-and-right examples for
  a past and a planned event; the examples in rules 1 and 5 lose their "As of".
