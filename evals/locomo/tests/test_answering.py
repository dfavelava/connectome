import asyncio
import json
import random
from dataclasses import replace

import pytest

from locomo_eval import answering, cli
from locomo_eval.answering import (
    ANSWER,
    JUDGE,
    THINK_NUM_PREDICT,
    AnswerConfig,
    Checkpoint,
    Question,
    answers_entry,
    checkpoint_paths,
    run_answers,
    select_questions,
)
from locomo_eval.llm import Completion, LLMError, SamplingOptions
from locomo_eval.metrics import Context
from locomo_eval.prompts import (
    ANSWER_V2_VERSION,
    ANSWER_VERSION,
    DEFAULT_JUDGE_VERSION,
    JUDGE_V2_VERSION,
    JUDGE_VERSION,
    NOT_MENTIONED,
    load_prompt,
)

PRICING = {
    "ollama:*": {"input_per_mtok": 0.0, "output_per_mtok": 0.0},
    "fake:*": {"input_per_mtok": 1.0, "output_per_mtok": 10.0},
}


class FakeLLM:
    """Stands in for LLMClient. Answers with the question's first context
    turn (or the abstention when there is none) and judges CORRECT; every
    call reports 100 input and 10 output tokens. With fail_on=N the Nth call
    raises LLMError, as a client out of retries would."""

    def __init__(self, *, fail_on: int | None = None, judge_reply: str | None = None, **kwargs):
        self.options = kwargs.get("options", SamplingOptions())
        self.fail_on = fail_on
        self.judge_reply = judge_reply or json.dumps({"reasoning": "same facts", "label": "CORRECT"})
        self.calls: list[tuple[str, str]] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    async def complete(self, model, system, user, *, stage="default", format=None, options=None) -> Completion:
        self.calls.append((stage, model))
        if len(self.calls) == self.fail_on:
            raise LLMError(f"call {self.fail_on} failed")
        await asyncio.sleep(0)
        if stage == JUDGE:
            return Completion(self.judge_reply, 100, 10)
        excerpts = user.split("Conversation excerpts:\n", 1)[1].split("\n\n", 1)[0].strip()
        return Completion(excerpts.split("\n")[0].split(": ", 1)[-1] if excerpts else NOT_MENTIONED, 100, 10)

    async def run_config(self, models):
        return {"host": "fake", "models": {role: {"id": model, "digest": "sha256:fake"} for role, model in models.items()}}

    def stage_calls(self, stage: str) -> int:
        return sum(1 for s, _ in self.calls if s == stage)


def question(n: int, category: str = "single-hop", answer: str | None = "Paris", contexts=None) -> Question:
    if contexts is None:
        contexts = (Context(f"D1:{n}", f"[8 May, 2023] Caroline: {answer or 'Rome'}"), Context("D2:1", "[9 May, 2023] Melanie: unrelated"))
    return Question(
        question_id=f"conv-1#{n}",
        sample_id="conv-1",
        question=f"Question {n}?",
        category=category,
        answer=answer,
        adversarial_answer="Rome" if category == "adversarial" else None,
        contexts=contexts,
    )


QUESTIONS = [question(0), question(1, "temporal", "7 May 2023"), question(2, "adversarial", None, ()), question(3, "multi-hop", "a, b")]


def config(**overrides) -> AnswerConfig:
    defaults = dict(
        answer_model="fake:answerer",
        judge_model="fake:judge",
        answer_prompt=load_prompt(ANSWER_VERSION),
        judge_prompt=load_prompt(JUDGE_VERSION),
        answer_k=1,
    )
    return AnswerConfig(**{**defaults, **overrides})


def run(client, cfg, questions, tmp_path, run_id="base"):
    checkpoint = Checkpoint(tmp_path, run_id, cfg.hash())
    return asyncio.run(run_answers(client, cfg, questions, checkpoint, concurrency=2))


def test_run_answers_scores_every_question(tmp_path):
    client = FakeLLM()
    result = run(client, config(), QUESTIONS, tmp_path)
    by_id = {q["id"]: q for q in result.questions}
    assert [q["id"] for q in result.questions] == [q.question_id for q in QUESTIONS]
    assert by_id["conv-1#0"]["answer"] == "Paris" and by_id["conv-1#0"]["f1"] == 1.0
    # The adversarial question has no contexts, so the fake abstains.
    assert by_id["conv-1#2"]["answer"] == NOT_MENTIONED and by_id["conv-1#2"]["f1"] == 1.0
    assert by_id["conv-1#2"]["gold"] == NOT_MENTIONED
    assert all(q["judge_label"] == "CORRECT" and q["judge_reasoning"] == "same facts" for q in result.questions)
    assert (client.stage_calls(ANSWER), client.stage_calls(JUDGE)) == (4, 4)
    lines = (tmp_path / f"base.{config().hash()}.jsonl").read_text().splitlines()
    assert len(lines) == 8


def test_answer_uses_only_top_answer_k_contexts(tmp_path):
    prompts = []

    class Recording(FakeLLM):
        async def complete(self, model, system, user, **kwargs):
            prompts.append(user)
            return await super().complete(model, system, user, **kwargs)

    run(Recording(), config(answer_k=1), [question(0)], tmp_path)
    assert "Caroline: Paris" in prompts[0] and "unrelated" not in prompts[0]


def test_killed_run_resumes_without_repeating_calls(tmp_path):
    with pytest.raises(ExceptionGroup) as info:
        run(FakeLLM(fail_on=4), config(), QUESTIONS, tmp_path)
    assert info.group_contains(LLMError)
    saved = (tmp_path / f"base.{config().hash()}.jsonl").read_text().splitlines()
    assert 1 <= len(saved) < 8

    resumed = FakeLLM()
    result = run(resumed, config(), QUESTIONS, tmp_path)
    assert len(resumed.calls) == 8 - len(saved)
    assert len(result.questions) == 4
    # A third run is served entirely from the checkpoint.
    again = FakeLLM()
    assert run(again, config(), QUESTIONS, tmp_path).questions == result.questions
    assert again.calls == []


def test_truncated_checkpoint_line_is_redone(tmp_path):
    run(FakeLLM(), config(), QUESTIONS, tmp_path)
    path = tmp_path / f"base.{config().hash()}.jsonl"
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n" + lines[-1][:20])
    client = FakeLLM()
    run(client, config(), QUESTIONS, tmp_path)
    assert len(client.calls) == 1


def test_changed_contexts_force_a_fresh_answer(tmp_path):
    run(FakeLLM(), config(), QUESTIONS, tmp_path)
    changed = [question(0, contexts=(Context("D3:3", "[1 June, 2023] Caroline: Lyon"),)), *QUESTIONS[1:]]
    client = FakeLLM()
    result = run(client, config(), changed, tmp_path)
    # Only question 0's answer is redone, and its new answer needs a new verdict.
    assert client.calls == [(ANSWER, "fake:answerer"), (JUDGE, "fake:judge")]
    assert result.questions[0]["answer"] == "Lyon"


def test_contexts_past_answer_k_do_not_invalidate(tmp_path):
    run(FakeLLM(), config(answer_k=1), QUESTIONS, tmp_path)
    changed = [question(0, contexts=(QUESTIONS[0].contexts[0], Context("D9:9", "[1 June, 2023] Melanie: new"))), *QUESTIONS[1:]]
    client = FakeLLM()
    run(client, config(answer_k=1), changed, tmp_path)
    assert client.calls == []


def test_judge_only_change_reuses_answers(tmp_path):
    first = config()
    run(FakeLLM(), first, QUESTIONS, tmp_path)
    second = config(judge_model="fake:bigger-judge")
    assert second.hash() != first.hash()
    client = FakeLLM()
    result = run(client, second, QUESTIONS, tmp_path)
    assert client.stage_calls(ANSWER) == 0
    assert client.calls == [(JUDGE, "fake:bigger-judge")] * 4
    assert len(result.questions) == 4
    assert [p.name for p in checkpoint_paths(tmp_path, "base")] == sorted([f"base.{first.hash()}.jsonl", f"base.{second.hash()}.jsonl"])


def test_changed_prompt_or_options_invalidate(tmp_path):
    run(FakeLLM(), config(), QUESTIONS, tmp_path)
    client = FakeLLM()
    run(client, config(options=SamplingOptions(seed=7)), QUESTIONS, tmp_path)
    assert (client.stage_calls(ANSWER), client.stage_calls(JUDGE)) == (4, 4)

    edited = load_prompt(ANSWER_VERSION)._replace(sha256="0" * 64)
    client = FakeLLM()
    run(client, config(answer_prompt=edited), QUESTIONS, tmp_path)
    assert client.stage_calls(ANSWER) == 4


def test_checkpoints_of_other_runs_are_ignored(tmp_path):
    run(FakeLLM(), config(), QUESTIONS, tmp_path, run_id="base")
    client = FakeLLM()
    run(client, config(), QUESTIONS, tmp_path, run_id="base.v2")
    assert len(client.calls) == 8
    assert [p.name for p in checkpoint_paths(tmp_path, "base")] == [f"base.{config().hash()}.jsonl"]


def test_unparseable_judge_reply_is_null(tmp_path):
    client = FakeLLM(judge_reply="CORRECT")
    result = run(client, config(), QUESTIONS[:1], tmp_path)
    assert result.questions[0]["judge_label"] is None
    # judge() retried the malformed reply once; both calls land in one record.
    assert client.stage_calls(JUDGE) == 2
    entry = answers_entry(result, config(), PRICING)
    assert entry["usage"][JUDGE]["calls"] == 2
    assert entry["summary"]["overall"]["judge_null"] == 1


def test_cost_totals_include_resumed_calls(tmp_path):
    with pytest.raises(ExceptionGroup):
        run(FakeLLM(fail_on=5), config(), QUESTIONS, tmp_path)
    result = run(FakeLLM(), config(), QUESTIONS, tmp_path)
    entry = answers_entry(result, config(), PRICING)
    for stage in (ANSWER, JUDGE):
        usage = entry["usage"][stage]
        assert (usage["calls"], usage["input_tokens"], usage["output_tokens"]) == (4, 400, 40)
        # fake:* costs $1 per million input and $10 per million output tokens.
        assert usage["cost_usd"] == pytest.approx((400 * 1 + 40 * 10) / 1e6)
    assert entry["config"]["pricing"] == {"fake:answerer": PRICING["fake:*"], "fake:judge": PRICING["fake:*"]}


def test_unpriced_model_has_null_cost(tmp_path):
    result = run(FakeLLM(), config(), QUESTIONS[:1], tmp_path)
    entry = answers_entry(result, config(), {})
    assert entry["usage"][ANSWER]["cost_usd"] is None
    assert entry["config"]["pricing"] == {"fake:answerer": None, "fake:judge": None}


def test_answers_entry_pins_models_and_prompts(tmp_path):
    cfg = config()
    entry = answers_entry(run(FakeLLM(), cfg, QUESTIONS, tmp_path), cfg, PRICING)
    assert entry["config"]["answer_model"] == "fake:answerer"
    assert entry["config"]["judge_model"] == "fake:judge"
    assert entry["config"]["answer_prompt"] == {"version": ANSWER_VERSION, "sha256": load_prompt(ANSWER_VERSION).sha256}
    assert entry["config"]["judge_prompt"]["version"] == JUDGE_VERSION
    assert entry["config"]["answer_k"] == 1 and entry["config"]["temperature"] == 0.0
    summary = entry["summary"]
    assert list(summary) == ["adversarial", "multi-hop", "single-hop", "temporal", "overall_excl_adversarial", "overall"]
    assert summary["overall"]["n"] == 4 and summary["overall_excl_adversarial"]["n"] == 3
    assert set(entry["questions"][0]) >= {"id", "answer", "f1", "judge_label", "judge_reasoning"}


def test_config_hash_is_stable_and_sensitive():
    assert config().hash() == config().hash()
    assert config(answer_k=2).hash() != config().hash()
    assert config(options=replace(SamplingOptions(), temperature=0.5)).hash() != config().hash()


def test_answer_think_is_a_separate_config_and_only_the_answer_thinks(tmp_path):
    # Off, the config is as it was before the option, so earlier hashes hold.
    assert "answer_options" not in config().to_dict()
    thinking = config(answer_think=True)
    assert thinking.hash() != config().hash()
    assert thinking.to_dict()["answer_options"]["think"] is True

    seen = []

    class Recording(FakeLLM):
        async def complete(self, model, system, user, *, stage="default", options=None, **kwargs):
            seen.append((stage, options or self.options))
            return await super().complete(model, system, user, stage=stage, options=options, **kwargs)

    run(FakeLLM(), config(), QUESTIONS, tmp_path)
    run(Recording(), thinking, QUESTIONS, tmp_path)
    # Checkpointed answers without thinking aren't reused; the fake's answers
    # are the same, so the verdicts are.
    assert [stage for stage, _ in seen] == [ANSWER] * 4
    run(Recording(), thinking, QUESTIONS, tmp_path / "fresh")
    answers = [options for stage, options in seen if stage == ANSWER]
    judges = [options for stage, options in seen if stage == JUDGE]
    assert len(answers) == 8 and all(o.think and o.num_predict == THINK_NUM_PREDICT for o in answers)
    assert judges and all(not o.think and o.num_predict == SamplingOptions().num_predict for o in judges)


def test_select_questions():
    other = replace(question(9), sample_id="conv-2", question_id="conv-2#9")
    questions = [*QUESTIONS, question(4, answer=None), other]
    selected, skipped = select_questions(questions, None, None)
    assert [q.question_id for q in selected] == ["conv-1#0", "conv-1#1", "conv-1#2", "conv-1#3", "conv-2#9"]
    assert skipped == {"no_gold_answer": 1}
    assert [q.question_id for q in select_questions(questions, ["conv-2"], None)[0]] == ["conv-2#9"]
    assert len(select_questions(questions, None, 2)[0]) == 2


def retrieval_run(tmp_path, run_id="base"):
    questions = [
        {
            "sample_id": q.sample_id,
            "question": q.question,
            "category": q.category,
            "evidence": [q.contexts[0].dia_id] if q.contexts else ["D1:1"],
            "retrieved": [c.dia_id for c in q.contexts],
            "question_id": q.question_id,
            "answer": q.answer,
            "adversarial_answer": q.adversarial_answer,
            "contexts": [{"dia_id": c.dia_id, "text": c.text} for c in q.contexts],
        }
        for q in QUESTIONS
    ]
    run_file = {
        "config": {"run_id": run_id, "samples": ["conv-1"], "ks": [1, 5], "recall_k": 10, "answer_k": 10},
        "summary": {},
        "skipped": {},
        "questions": questions,
    }
    path = tmp_path / f"{run_id}.json"
    path.write_text(json.dumps(run_file))
    return path


def test_cli_answer_scores_a_tagged_retrieval(tmp_path, monkeypatch):
    base = retrieval_run(tmp_path)
    tagged = retrieval_run(tmp_path, run_id="base.bm25")
    monkeypatch.setattr(answering, "LLMClient", lambda **kwargs: FakeLLM(**kwargs))
    cli.main(["answer", "base", "--tag", "bm25", "--answer-model", "ollama:a", "--judge-model", "ollama:j", "--results-dir", str(tmp_path)])

    assert "answers" in json.loads(tagged.read_text())
    assert "answers" not in json.loads(base.read_text())
    # Its checkpoint is its own, and not one of the untagged run's.
    assert [p.name for p in checkpoint_paths(tmp_path, "base")] == []
    assert len(checkpoint_paths(tmp_path, "base.bm25")) == 1


def test_cli_answer_subcommand_writes_results_next_to_retrieval(tmp_path, monkeypatch, capsys):
    path = retrieval_run(tmp_path)
    clients = []

    def factory(**kwargs):
        clients.append(FakeLLM(**kwargs))
        return clients[-1]

    monkeypatch.setattr(answering, "LLMClient", factory)
    argv = ["answer", "base", "--answer-model", "ollama:a", "--judge-model", "ollama:j", "--results-dir", str(tmp_path), "--judge-sample", "2"]
    cli.main(argv)

    data = json.loads(path.read_text())
    assert data["questions"] and data["config"]["run_id"] == "base"
    (entry,) = data["answers"].values()
    assert entry["config"]["answer_k"] == 10
    assert entry["config"]["llm"]["models"]["answer"]["digest"] == "sha256:fake"
    assert entry["config"]["subset"] == {"samples": None, "limit": None}
    assert entry["summary"]["overall"]["judge_acc"] == 1.0
    assert entry["usage"][ANSWER]["cost_usd"] == 0.0
    out = capsys.readouterr().out
    assert "overall_excl_adversarial" in out and "recall@1" in out and "judge_acc" in out
    assert "random judged items" in out

    # A rerun is served from the checkpoint.
    cli.main(argv)
    assert clients[-1].calls == []


def test_cli_answer_selects_prompt_and_thinking(tmp_path, monkeypatch):
    path = retrieval_run(tmp_path)
    monkeypatch.setattr(answering, "LLMClient", lambda **kwargs: FakeLLM(**kwargs))
    base = ["answer", "base", "--answer-model", "ollama:a", "--judge-model", "ollama:j", "--results-dir", str(tmp_path)]
    cli.main(base)
    cli.main([*base, "--answer-prompt", ANSWER_V2_VERSION, "--answer-think"])

    entries = json.loads(path.read_text())["answers"].values()
    assert sorted((e["config"]["answer_prompt"]["version"], "answer_options" in e["config"]) for e in entries) == [
        (ANSWER_VERSION, False),
        (ANSWER_V2_VERSION, True),
    ]


def test_cli_answer_rescores_cached_answers_with_another_judge_prompt(tmp_path, monkeypatch):
    path = retrieval_run(tmp_path)
    clients = []

    def factory(**kwargs):
        clients.append(FakeLLM(**kwargs))
        return clients[-1]

    monkeypatch.setattr(answering, "LLMClient", factory)
    base = ["answer", "base", "--answer-model", "ollama:a", "--judge-model", "ollama:j", "--results-dir", str(tmp_path)]
    cli.main(base)
    cli.main([*base, "--judge-prompt", JUDGE_V2_VERSION])

    # Only the verdicts are redone; the answers come from the default judge's checkpoint.
    assert clients[-1].stage_calls(ANSWER) == 0 and clients[-1].stage_calls(JUDGE) > 0
    entries = list(json.loads(path.read_text())["answers"].values())
    assert sorted(e["config"]["judge_prompt"]["version"] for e in entries) == [JUDGE_V2_VERSION, DEFAULT_JUDGE_VERSION]
    assert [q["answer"] for q in entries[0]["questions"]] == [q["answer"] for q in entries[1]["questions"]]


def test_cli_answer_reports_failure_and_resumes(tmp_path, monkeypatch):
    retrieval_run(tmp_path)
    failing = [True]

    def factory(**kwargs):
        return FakeLLM(fail_on=3 if failing[0] else None, **kwargs)

    monkeypatch.setattr(answering, "LLMClient", factory)
    argv = ["answer", "base", "--answer-model", "ollama:a", "--judge-model", "ollama:j", "--results-dir", str(tmp_path), "--limit", "2"]
    with pytest.raises(SystemExit, match="rerun the same command to resume"):
        cli.main(argv)
    failing[0] = False
    cli.main(argv)
    (entry,) = json.loads((tmp_path / "base.json").read_text())["answers"].values()
    assert entry["summary"]["overall"]["n"] == 2
    assert entry["usage"][ANSWER]["calls"] == 2


def test_cli_answer_rejects_missing_run_and_bad_answer_k(tmp_path):
    with pytest.raises(SystemExit, match="no retrieval run"):
        cli.main(["answer", "nope", "--answer-model", "ollama:a", "--judge-model", "ollama:j", "--results-dir", str(tmp_path)])
    retrieval_run(tmp_path)
    with pytest.raises(SystemExit, match="answer-k"):
        cli.main(["answer", "base", "--answer-model", "ollama:a", "--judge-model", "ollama:j", "--results-dir", str(tmp_path), "--answer-k", "11"])


def test_judge_sample_prints_judged_items(capsys):
    questions = [{"id": "q1", "category": "temporal", "question": "When?", "gold": "May", "answer": "May", "judge_label": "CORRECT", "judge_reasoning": "ok"}]
    answering.print_judge_sample([*questions, {**questions[0], "id": "q2", "judge_label": None}], 5, random.Random(0))
    out = capsys.readouterr().out
    assert "1 random judged items" in out and "q1" in out and "q2" not in out
