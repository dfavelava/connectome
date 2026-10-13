import asyncio
import json

import httpx

from westmarches_eval import answering
from westmarches_eval.answering import HYDRATE_OFF, HYDRATE_ON

HYDRATED = """---
version: connectome/memory/0.2
id: mem_1.md
type: note
occurred_at: "2026-03-07T19:00:00Z"
relationships:
  - subjectEntityId: ghost
    predicate: haunts
    objectEntityId: mill
    kind: rumor
    superseded_by: mem_2.md
acl: []
derived_from: null
---
The miller's ghost haunts the Hollow Mill.
"""


def test_context_text_tags_a_hydrated_result():
    assert answering.context_text({"content": HYDRATED}) == "[2026-03-07] (rumor) (superseded) The miller's ghost haunts the Hollow Mill."


def test_context_text_of_a_plain_fact_has_only_its_date():
    content = HYDRATED.replace("    kind: rumor\n", "    kind: fact\n").replace("    superseded_by: mem_2.md\n", "")
    assert answering.context_text({"content": content}) == "[2026-03-07] The miller's ghost haunts the Hollow Mill."


def test_context_text_of_a_snippet_is_the_text_alone():
    assert answering.context_text({"snippet": " The miller's ghost haunts the Hollow Mill. "}) == "The miller's ghost haunts the Hollow Mill."


def test_prompts_load_with_their_placeholders():
    config = answering.answer_config("ollama:a", "ollama:j", 10)
    assert config.answer_prompt.version == "answer_v1"
    assert config.judge_prompt.version == "judge_v3"
    assert config.truth_judge_prompt.render(question="q", gold_answer="g", truth="rumor", generated_answer="a").endswith("Generated answer: a\n")
    assert "$context" not in config.answer_prompt.render(context="c", question="q")
    assert len(config.hash()) == 12


def fake_ollama(answers: dict[str, str]):
    """A /api/chat that answers from `answers` by question and judges an
    answer CORRECT when it mentions "rumor"."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        prompt = body["messages"][-1]["content"]
        if "format" in body:
            generated = prompt.rsplit("Generated answer:", 1)[1]
            label = "CORRECT" if "rumor" in generated else "WRONG"
            text = json.dumps({"reasoning": "r", "label": label})
        else:
            question = prompt.rsplit("Question: ", 1)[1].split("\n")[0]
            text = answers[question]
        return httpx.Response(200, json={"message": {"content": text}, "prompt_eval_count": 1, "eval_count": 1, "done_reason": "stop"})

    return httpx.MockTransport(handler)


def test_answer_run_scores_truth_and_graph_per_mode():
    run = {
        "questions": [
            {"id": "t", "kind": "truth", "question": "What haunts the mill?", "answer": "A ghost, by rumor.", "truth": "rumor", "contexts": {HYDRATE_OFF: ["x"], HYDRATE_ON: ["y"]}},
            {"id": "g", "kind": "graph", "question": "Which hexes?", "answer": "Wolfden.", "truth": None, "contexts": {HYDRATE_OFF: [], HYDRATE_ON: []}},
            {"id": "l", "kind": "leakage", "question": "Secret?", "answer": "Not known.", "truth": None, "contexts": {HYDRATE_OFF: [], HYDRATE_ON: []}},
        ]
    }
    config = answering.answer_config("ollama:a", "ollama:j", 10)
    client = answering.LLMClient(transport=fake_ollama({"What haunts the mill?": "A ghost, says rumor.", "Which hexes?": "Saltcliff."}), pricing={})
    modes = asyncio.run(answering.answer_run(client, config, run))
    for mode in (HYDRATE_OFF, HYDRATE_ON):
        summary = modes[mode]["summary"]
        assert summary["truth_status"]["accuracy"] == 1.0
        assert summary["truth_content"]["accuracy"] == 1.0
        assert summary["graph"]["accuracy"] == 0.0
        # Leakage questions are retrieval-only.
        assert [r["question_id"] for r in modes[mode]["questions"]] == ["t", "g"]
