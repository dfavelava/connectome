# Hand-labelled judge samples

Sheets drawn with `locomo-eval judge-agreement <run-id> --draw` go here, one
item per line with the `id`, `category`, `question`, `gold` answer, generated
`answer`, and a `label` of `CORRECT` or `WRONG` (plus an optional `note`).
Items are labelled under `judge_v3`'s rules, without looking at any judge's
verdict.

Labels stay binary, like the judges' verdicts. When an item is a close call
under the rules, still pick `CORRECT` or `WRONG`, set `"borderline": true`
and say why in the `note`. Decide the flag while labelling, never after
seeing a verdict. The report then splits each judge's agreement between
clear and borderline items, which tells a judge that misses clear cases from
rules that leave hard cases open. Sheets without the field count every item
as clear.

[`labeller.html`](labeller.html) is a small page for labelling a sheet: open
it in a browser (straight from disk, no server), open the sheet, and label it
with `C` / `W`, flag close calls with `B` and write a note with `N`. It shows
the question, gold answer and generated answer, never a verdict, and keeps
every other field of a line as it is. Chromium-based browsers save the sheet
back in place as you go; elsewhere Save downloads the labelled copy to move
over the original. The sheet never leaves the browser.

The sheets quote LoCoMo questions and answers, which are CC BY-NC, so
`*.jsonl` here is gitignored: keep them local, next to the run's results.

`locomo-eval judge-agreement <run-id> --labels labels/<file>.jsonl` compares
every judge that scored the run with these labels. A label only counts
against a judge that saw the identical generated answer, so a sheet belongs
to the answers it was drawn from. Never edit a label to match a judge; draw
a new sheet instead.
