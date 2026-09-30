import asyncio
import hashlib
import json

import httpx
import pytest

from locomo_eval.dataset import CATEGORY_NAMES
from locomo_eval.llm import RETRY_TEMPERATURE, LLMClient
from locomo_eval.metrics import Context
from locomo_eval.prompts import (
    ANSWER_V2_VERSION,
    ANSWER_VERSION,
    JUDGE_SCHEMA,
    JUDGE_V2_VERSION,
    JUDGE_VERSION,
    NOT_MENTIONED,
    PROMPTS_DIR,
    JudgeParseError,
    JudgeResult,
    Verdict,
    answer_prompt,
    judge,
    judge_prompt,
    judge_summary,
    load_prompt,
    parse_judge_output,
)
from locomo_eval.scoring import ADVERSARIAL, SINGLE_HOP


@pytest.mark.parametrize("version", [ANSWER_VERSION, ANSWER_V2_VERSION, JUDGE_VERSION, JUDGE_V2_VERSION])
def test_load_committed_prompt(version):
    prompt = load_prompt(version)
    data = (PROMPTS_DIR / f"{version}.txt").read_bytes()
    assert prompt.version == version
    assert prompt.text == data.decode("utf-8")
    assert prompt.sha256 == hashlib.sha256(data).hexdigest()
    assert prompt.config() == {"version": version, "sha256": prompt.sha256}


def test_load_prompt_hash_follows_file(tmp_path):
    (tmp_path / "answer_v1.txt").write_text("one $question")
    (tmp_path / "answer_v2.txt").write_text("two $question")
    first, second = load_prompt("answer_v1", tmp_path), load_prompt("answer_v2", tmp_path)
    assert (first.text, first.version) == ("one $question", "answer_v1")
    assert first.sha256 != second.sha256


@pytest.mark.parametrize("version", ["answer", "answer_v", "../answer_v1", "answer_v1.txt"])
def test_load_prompt_rejects_malformed_version(version):
    with pytest.raises(ValueError, match="_v<N>"):
        load_prompt(version)


def test_load_prompt_missing_version():
    with pytest.raises(FileNotFoundError, match="answer_v999"):
        load_prompt("answer_v999")


@pytest.mark.parametrize("version", [ANSWER_VERSION, ANSWER_V2_VERSION])
def test_answer_prompt_sorts_contexts_by_date(version):
    contexts = [
        Context("D10:2", "[1:00 pm on 3 July, 2023] Caroline: third"),
        Context("D2:5", "[2:00 pm on 8 May, 2023] Melanie: second"),
        Context("D2:1", "[2:00 pm on 8 May, 2023] Caroline: first"),
    ]
    text = answer_prompt(load_prompt(version), "What happened?", contexts)
    assert "Question: What happened?" in text
    assert "[2:00 pm on 8 May, 2023] Caroline: first\n[2:00 pm on 8 May, 2023] Melanie: second\n[1:00 pm on 3 July, 2023] Caroline: third" in text
    assert NOT_MENTIONED in text
    assert "$" not in text


def test_answer_v2_shows_extracted_memories_and_keeps_the_abstention():
    contexts = [
        Context("D3:4", "Sam moved to a new city on 2 June 2023."),
        Context("D1:2", "Sam finished a first pottery class on 11 March 2023."),
    ]
    text = answer_prompt(load_prompt(ANSWER_V2_VERSION), "When did Sam move?", contexts)
    assert "Sam finished a first pottery class on 11 March 2023.\nSam moved to a new city on 2 June 2023.\n" in text
    # The adversarial scoring depends on the exact abstention.
    assert f"reply exactly: {NOT_MENTIONED}" in text
    assert text.endswith("Question: When did Sam move?\nAnswer:\n")


def test_answer_v2_is_generic():
    lowered = load_prompt(ANSWER_V2_VERSION).text.lower()
    # No dataset speakers or category names; the examples use a made-up person.
    for word in ("caroline", "melanie", "lgbtq", *CATEGORY_NAMES.values()):
        assert word not in lowered


def test_judge_prompt_regular_question():
    text = judge_prompt(load_prompt(JUDGE_VERSION), "Where?", SINGLE_HOP, "In Paris.", answer="Paris")
    assert "Question: Where?\nGold answer: Paris\nTrap answer: (none)\nGenerated answer: In Paris." in text
    # The examples' JSON braces survive substitution.
    assert '{"reasoning": "Same date in a different format.", "label": "CORRECT"}' in text


def test_judge_prompt_adversarial_question():
    text = judge_prompt(load_prompt(JUDGE_VERSION), "What did she paint?", ADVERSARIAL, "A horse.", adversarial_answer="a horse")
    assert f"Gold answer: {NOT_MENTIONED}\nTrap answer: a horse\nGenerated answer: A horse." in text


@pytest.mark.parametrize("version", [JUDGE_VERSION, JUDGE_V2_VERSION])
def test_judge_prompt_fills_every_placeholder(version):
    text = judge_prompt(load_prompt(version), "What did she paint?", ADVERSARIAL, "A horse.", adversarial_answer="a horse")
    assert text.endswith(f"Question: What did she paint?\nGold answer: {NOT_MENTIONED}\nTrap answer: a horse\nGenerated answer: A horse.\n")
    assert "$" not in text


def test_judge_v2_examples_are_valid_verdicts():
    text = load_prompt(JUDGE_V2_VERSION).text
    examples = [line for line in text.splitlines() if line.startswith("{")]
    verdicts = [parse_judge_output(line) for line in examples]
    # One example per leniency, the strict counterparts and the adversarial rule.
    assert len(verdicts) == 8
    assert {v.label for v in verdicts} == {"CORRECT", "WRONG"}


def test_judge_v2_keeps_the_strict_rules():
    text = load_prompt(JUDGE_V2_VERSION).text
    # The adversarial rule is judge_v1's, word for word.
    adversarial_rule = next(line for line in load_prompt(JUDGE_VERSION).text.splitlines() if line.startswith('- If the gold answer is "Not mentioned'))
    assert adversarial_rule in text
    assert "repeats the trap answer" in text
    assert "a wrong name, place, number, date or item in place of the gold one is WRONG" in text
    assert "An answer that contains none of the gold items is WRONG." in text
    assert "A date outside the gold period is WRONG." in text


def test_judge_v2_is_generic():
    lowered = load_prompt(JUDGE_V2_VERSION).text.lower()
    # Made-up examples, so none of them is also an item being judged.
    for word in ("caroline", "melanie", "lgbtq", "transgender", *CATEGORY_NAMES.values()):
        assert word not in lowered


def test_judge_prompt_requires_gold_answer():
    with pytest.raises(ValueError, match="no gold answer"):
        judge_prompt(load_prompt(JUDGE_VERSION), "Where?", SINGLE_HOP, "Paris")


def test_parse_judge_output():
    assert parse_judge_output('{"reasoning": "Same city.", "label": "CORRECT"}') == Verdict("Same city.", "CORRECT")
    assert parse_judge_output(' \n{"label": "WRONG", "reasoning": ""}\n') == Verdict("", "WRONG")


@pytest.mark.parametrize(
    "text",
    [
        "",
        "CORRECT",
        '```json\n{"reasoning": "x", "label": "CORRECT"}\n```',
        '{"reasoning": "x", "label": "CORRECT"',
        '["CORRECT"]',
        '{"label": "CORRECT"}',
        '{"reasoning": "x", "label": "CORRECT", "confidence": 1}',
        '{"reasoning": "x", "label": "correct"}',
        '{"reasoning": "x", "label": "PARTIAL"}',
        '{"reasoning": null, "label": "WRONG"}',
    ],
)
def test_parse_judge_output_is_strict(text):
    with pytest.raises(JudgeParseError):
        parse_judge_output(text)


def judge_client(replies):
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": replies[len(requests) - 1]}, "prompt_eval_count": 10, "eval_count": 5})

    return LLMClient(host="http://ollama.test", backoff=0, pricing={}, transport=httpx.MockTransport(handler)), requests


def run_judge(replies):
    client, requests = judge_client(replies)

    async def go():
        async with client:
            return await judge(client, "ollama:qwen3:8b", "grade this"), client.usage_summary()

    result, usage = asyncio.run(go())
    return result, requests, usage


def test_judge_requests_structured_output():
    result, requests, usage = run_judge(['{"reasoning": "Same city.", "label": "CORRECT"}'])
    assert result == JudgeResult(judge_label="CORRECT", judge_reasoning="Same city.", attempts=1)
    (request,) = requests
    assert request["format"] == JUDGE_SCHEMA
    assert request["messages"] == [{"role": "user", "content": "grade this"}]
    assert usage[0]["stage"] == "judge"


def test_judge_retries_malformed_reply_once_with_new_seed_and_temperature():
    result, requests, _ = run_judge(["CORRECT", '{"reasoning": "Wrong city.", "label": "WRONG"}'])
    assert result == JudgeResult(judge_label="WRONG", judge_reasoning="Wrong city.", attempts=2)
    assert [r["options"]["seed"] for r in requests] == [42, 43]
    # At temperature 0 the new seed would be ignored and the reply repeated.
    assert [r["options"]["temperature"] for r in requests] == [0.0, RETRY_TEMPERATURE]


def test_judge_records_null_after_second_malformed_reply():
    result, requests, usage = run_judge(["CORRECT", '{"label": "CORRECT"}', "unused"])
    assert result.judge_label is None
    assert result.judge_reasoning is None
    assert result.attempts == 2
    assert "keys" in result.judge_error
    assert len(requests) == 2
    assert usage[0]["calls"] == 2


def test_judge_summary_counts_nulls_without_scoring_them():
    assert judge_summary(["CORRECT", "WRONG", None, "CORRECT", None]) == {
        "judged": 3,
        "correct": 2,
        "wrong": 1,
        "judge_null": 2,
        "accuracy": pytest.approx(2 / 3),
    }
    assert judge_summary([None])["accuracy"] is None
