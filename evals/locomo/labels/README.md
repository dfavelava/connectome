# Hand-labelled judge samples

Each `.jsonl` file here is a sheet drawn with `locomo-eval judge-agreement
<run-id> --draw` and labelled by hand: one item per line with the `id`,
`category`, `question`, `gold` answer, generated `answer`, and a `label` of
`CORRECT` or `WRONG` (plus an optional `note`). Items are labelled under
`judge_v2`'s rules, without looking at any judge's verdict.

`locomo-eval judge-agreement <run-id> --labels labels/<file>.jsonl` compares
every judge that scored the run with these labels. A label only counts
against a judge that saw the identical generated answer, so a sheet belongs
to the answers it was drawn from. Never edit a committed label to match a
judge; draw a new sheet instead.
