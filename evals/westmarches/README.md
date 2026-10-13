## West Marches belief eval

Measures what LoCoMo can't: whether Connectome's `recall` respects who may see
what, prefers current claims over superseded ones, keeps rumors and hypotheses
labelled as such, gives each viewer their own version of an event, and
answers multi-hop questions in a campaign's own terms. LoCoMo is two people
chatting, with no access lists, viewpoints, rumors or superseded claims (see
#67). This eval decides between Phase 2 (the graph) and Phase 3 (lifecycle),
and it is the bar for Phase 1.4 (beliefs in recall).

### The campaign

[`campaign/campaign.yaml`](campaign/campaign.yaml) is a small West Marches
campaign, The Marches of Vell: a GM, three parties of three players (the
Lanterns, the Greyhawks, the Mire Wardens), a 14-hex map and over a dozen NPCs
over eight sessions. Its 93 memories cover every kind of write the eval is
about:

- **Public facts and events**: the hex map with each hex's neighbours, the
  NPCs, prices, who explored which hex.
- **Rumors and hypotheses**: relationships of kind `rumor` or `hypothesis`.
  Several state the claim flatly ("The miller's ghost haunts the Hollow Mill."),
  so only the relationship's kind marks it as a rumor.
- **Party findings** (`acl: [party-…]`), **players' private backstories**
  (`acl: [<player>]`) and **GM secrets** (`acl: [gm]`). Every restricted acl
  also includes the GM.
- **Facets**: three events (the Ostry Ford bridge, the market fire, the bell)
  each have a public root and one diverging facet per party, written with
  `derived_from` and acl'd to that party.
- **Supersessions**: a new watch captain, new prices, a herbalist who moved,
  a new party leader, a missing knight who came back and hexes leaving the
  map's fog. Each one marks every relationship of the old memory
  `superseded_by` the new one.

Its 48 questions each have a viewer (`as`), a gold answer and gold memories,
and, depending on their kind, the superseded memories a current answer must
not rest on, the gold truth status, or the hidden memories a leakage question
fishes for. The file's header documents the format. `load_campaign` checks
the file before any run: every id resolves, gold memories are visible to the
question's viewer, probed memories are not, and stale memories really were
superseded.

**How it was generated.** The campaign was written once for this eval,
drafted by Claude Code in the session that added the harness, and is
reviewed like any other code change. It is fictional and checked in as is,
not generated on each run, so every run ingests the same memories. It is not taken from a real campaign's logs: a real
campaign holds players' private content, which can't be checked in, and
anonymizing one by hand would end up with a hand-written campaign anyway.
Writing it by hand also means every gold answer and acl is known exactly,
with no extraction step in between. The cost is that it is small and tidier
than a real campaign. If a real session's logs later show a pattern this
campaign misses, add memories and questions for it here, written the same
way.

### Scores

Leakage, staleness, perspective and gold recall are retrieval scores and need
no LLM. Each run scores them twice, with `hydrate` off and on; the ranking is
the same, so the two should match.

- **Leakage**: the share of all results, over every question, that the viewer
  may not see under the campaign's own acls and party memberships. The ten
  leakage questions aim straight at GM secrets, other parties' findings and
  private backstories, but every question's results count. The target is
  zero, and any leak is a bug, not a score: the report lists each one.
- **Staleness@k**: over the staleness questions, the share of relevant top-k
  results (gold or stale) that are superseded claims. `stale_first@k` is the
  share of those questions whose highest-ranked relevant result is stale.
- **Perspective@k**: the share of perspective questions whose viewer's own
  facet is in the top k and outranks the root and the other facets.
  `root_first@k` is the share where the root outranks it.
- **Graph recall@k**: the mean share of a graph question's gold memories in
  the top k; `all_gold@k` is the share with every one there, which is what
  answering it takes. `recall@k` is also reported for each question kind.

Truth status and graph answers need the answer and judge stage. It reuses
LoCoMo's LLM client, judge parsing and prompt versioning (`locomo_eval.llm`,
`locomo_eval.prompts`) rather than adding a second harness:

- **Truth status**: the answer model answers each truth question from its top
  `answer_k` results, and `truth_v1` judges whether the answer gives the gold
  claim with the gold truth status: a rumor called a rumor, a hypothesis
  called a suspicion, a fact stated plainly. The same answers are judged on
  content alone by LoCoMo's `judge_v3` (`truth content accuracy`), which tells
  a wrong status apart from a wrong claim.
- **Graph answers**: `judge_v3` on the answers to the graph questions, such as
  "which unexplored hexes border the Ashwood" and "who in the Greyhawks has
  met Kestrel".

The answer model sees each result as recall returned it. With `hydrate` off
that is the snippet only. With `hydrate` on it is the memory's body, prefixed
with what a client can read from its frontmatter: the date it happened, a
`(rumor)` or `(hypothesis)` tag from its relationships' kinds, and
`(superseded)` when every relationship has been superseded. The gap between
the two shows what surfacing beliefs in recall (Phase 1.4) is worth.

Prompts are versioned like LoCoMo's (`src/westmarches_eval/prompts/<name>_v<N>.txt`;
a committed version is never edited) and every result records their versions
and sha256.

### Running

Start the backend (`docker compose up --build` from the repo root), then from
`evals/westmarches/`:

```bash
uv sync
uv run westmarches-eval
```

This writes the campaign into a fresh `temp-westmarches-<run-id>` tome, asks
every question as its viewer with `hydrate` off and on, destroys the tome
(also when the run fails), prints every retrieval score and writes
`results/<run-id>.json`. To score answers too, publish Ollama to the host
(`docker-compose.eval.yml`, see the LoCoMo README) and name the models:

```bash
uv run westmarches-eval --answer-model ollama:qwen3:8b --judge-model ollama:qwen3:14b
```

That one command reports all five scores. The harness reads
`CONNECTOME_API_BASE_URL` / `CONNECTOME_API_KEY` like LoCoMo's (a `.env` here
is loaded if present) and `OLLAMA_HOST` for the LLM calls.

Useful flags (`uv run westmarches-eval --help` for all):

- `--ks 5,10` - k values to report (max 50, the backend's search cap).
- `--answer-k N` - results an answer is given (default 10).
- `--run-id NAME` - fixed tome/result name instead of a timestamp.
- `--keep-tome` - skip cleanup, leaving the tome for `westmarches-eval retrieve`.
- `--fail-on-leak` - exit with status 1 if any result leaked.

### Stages

As in LoCoMo, the default command runs stages that can also run apart:

```bash
uv run westmarches-eval ingest --run-id wm1       # fill temp-westmarches-wm1, write results/wm1.ingest.json
uv run westmarches-eval retrieve wm1             # score it, write results/wm1.json
uv run westmarches-eval retrieve wm1 --tag again # score it again, write results/wm1.again.json
uv run westmarches-eval answer wm1 --answer-model ollama:qwen3:8b --judge-model ollama:qwen3:14b
uv run westmarches-eval cleanup wm1              # destroy the tome
```

`answer` runs offline from `results/<name>.json`, which holds every
question's hits and contexts in both modes, and stores its scores there under
`answers`, keyed by a hash of the answer config (models, prompt versions and
sha256s, `answer_k`, sampling options). `retrieve` refuses a run whose tome
was destroyed or whose campaign file changed since it was ingested.

### CI

The `West Marches eval` job in `.github/workflows/ci.yml` runs the tests,
then builds and starts the backend against Postgres and Ollama
(`nomic-embed-text` only) and runs `westmarches-eval --fail-on-leak`. Leakage
needs no LLM, so every pull request checks it. The job's log also holds that
commit's staleness, perspective and graph recall.

### How results map to the roadmap

- Any leakage: fix it before anything else.
- High staleness, or truth-status errors: Phase 1.4, then Phase 3.
- Weak graph questions: Phase 2.

### Tests

```bash
uv sync
uv run pytest
uv run ruff check
```

The tests need no backend or LLM: they check the campaign's consistency, the
scores on a small campaign, the client calls ingestion makes, and the answer
stage against a mocked Ollama.
