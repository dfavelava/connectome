# Hand-labelled judge samples

Sheets drawn with `locomo-eval judge-agreement <run-id> --draw` go here, one
item per line with the `id`, `category`, `question`, `gold` answer, generated
`answer`, and a `label` of `CORRECT` or `WRONG` (plus an optional `note`).
Items are labelled under `judge_v2`'s rules, without looking at any judge's
verdict.

Labels stay binary, like the judges' verdicts. When an item is a close call
under the rules, still pick `CORRECT` or `WRONG`, set `"borderline": true`
and say why in the `note`. Decide the flag while labelling, never after
seeing a verdict. The report then splits each judge's agreement between
clear and borderline items, which tells a judge that misses clear cases from
rules that leave hard cases open. Sheets without the field count every item
as clear.

The sheets quote LoCoMo questions and answers, which are CC BY-NC, so
`*.jsonl` here is gitignored: keep them local, next to the run's results.

`locomo-eval judge-agreement <run-id> --labels labels/<file>.jsonl` compares
every judge that scored the run with these labels. A label only counts
against a judge that saw the identical generated answer, so a sheet belongs
to the answers it was drawn from. Never edit a label to match a judge; draw
a new sheet instead.
