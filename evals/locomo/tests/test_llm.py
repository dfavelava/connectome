import asyncio
import json

import httpx
import pytest

from locomo_eval.llm import (
    DEFAULT_PRICING_PATH,
    Completion,
    LLMClient,
    LLMError,
    ModelId,
    SamplingOptions,
    chat_request,
    cost_usd,
    load_pricing,
    parse_model_id,
    price_for,
    strip_thinking,
)

CHAT_RESPONSE = {
    "message": {"role": "assistant", "content": "Paris"},
    "prompt_eval_count": 120,
    "eval_count": 4,
    "eval_duration": 2_000_000_000,
}


def test_parse_model_id():
    assert parse_model_id("ollama:qwen3:8b") == ModelId("ollama", "qwen3:8b")
    assert parse_model_id("ollama:llama3.2") == ModelId("ollama", "llama3.2")
    assert str(parse_model_id("ollama:qwen3:8b")) == "ollama:qwen3:8b"


@pytest.mark.parametrize("model", ["qwen3", "ollama:", ":qwen3", ""])
def test_parse_model_id_rejects_malformed(model):
    with pytest.raises(ValueError, match="provider:model"):
        parse_model_id(model)


def test_parse_model_id_rejects_unknown_provider():
    with pytest.raises(ValueError, match="unknown model provider"):
        parse_model_id("anthropic:claude-sonnet-5")


def test_chat_request_options():
    body = chat_request(ModelId("ollama", "qwen3:8b"), "sys", "question", SamplingOptions(seed=7, num_ctx=4096, num_predict=64))
    assert body == {
        "model": "qwen3:8b",
        "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "question"}],
        "stream": False,
        "think": False,
        "options": {"temperature": 0.0, "seed": 7, "num_ctx": 4096, "num_predict": 64},
    }


def test_strip_thinking():
    assert strip_thinking("<think>\nhmm, Paris?\n</think>\n\nParis") == "Paris"
    assert strip_thinking("  Paris \n") == "Paris"


PRICING = {
    "hosted:exact": {"input_per_mtok": 3.0, "output_per_mtok": 15.0},
    "hosted:*": {"input_per_mtok": 1.0, "output_per_mtok": 1.0},
    "ollama:*": {"input_per_mtok": 0.0, "output_per_mtok": 0.0},
}


def test_price_lookup():
    assert price_for(PRICING, "ollama:qwen3:8b") == PRICING["ollama:*"]
    assert price_for(PRICING, "hosted:exact") == PRICING["hosted:exact"]
    assert price_for(PRICING, "hosted:other") == PRICING["hosted:*"]
    assert price_for(PRICING, "unknown:model") is None


def test_cost_usd():
    assert cost_usd(PRICING, "ollama:qwen3:8b", 1000, 100) == 0.0
    assert cost_usd(PRICING, "hosted:exact", 1_000_000, 100_000) == pytest.approx(4.5)
    assert cost_usd(PRICING, "unknown:model", 1000, 100) is None


def test_committed_pricing_prices_ollama_at_zero():
    pricing = load_pricing(DEFAULT_PRICING_PATH)
    assert cost_usd(pricing, "ollama:qwen3:8b", 5000, 500) == 0.0


def make_client(handler, **kwargs) -> LLMClient:
    kwargs.setdefault("pricing", PRICING)
    return LLMClient(host="http://ollama.test", backoff=0, transport=httpx.MockTransport(handler), **kwargs)


def test_complete_records_usage():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=CHAT_RESPONSE)

    async def go():
        async with make_client(handler) as client:
            first = await client.complete("ollama:qwen3:8b", "sys", "q1", stage="answer")
            await client.complete("ollama:qwen3:8b", "sys", "q2", stage="answer")
            await client.complete("ollama:qwen3:8b", "sys", "q3", stage="judge")
            return first, client.usage_summary()

    first, summary = asyncio.run(go())
    assert first == Completion("Paris", 120, 4)
    assert requests[0]["model"] == "qwen3:8b"
    assert requests[0]["options"]["temperature"] == 0.0
    answer, judge = summary
    assert answer["stage"] == "answer" and answer["model"] == "ollama:qwen3:8b"
    assert answer["calls"] == 2
    assert (answer["input_tokens"], answer["output_tokens"]) == (240, 8)
    assert answer["output_tokens_per_second"] == 2.0
    assert answer["cost_usd"] == 0.0
    assert judge["calls"] == 1


def test_complete_retries_transient_errors():
    attempts = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        if len(attempts) == 1:
            raise httpx.ConnectError("refused", request=request)
        if len(attempts) == 2:
            raise httpx.ReadTimeout("slow", request=request)
        if len(attempts) == 3:
            return httpx.Response(503, text="loading model")
        return httpx.Response(200, json=CHAT_RESPONSE)

    async def go():
        async with make_client(handler, max_retries=3) as client:
            return await client.complete("ollama:qwen3:8b", "sys", "q")

    assert asyncio.run(go()).text == "Paris"
    assert len(attempts) == 4


def test_complete_gives_up_after_max_retries():
    attempts = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        raise httpx.ConnectError("refused", request=request)

    async def go():
        async with make_client(handler, max_retries=2) as client:
            await client.complete("ollama:qwen3:8b", "sys", "q")

    with pytest.raises(LLMError, match="after 3 attempts"):
        asyncio.run(go())
    assert len(attempts) == 3


def test_complete_does_not_retry_client_errors():
    attempts = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        return httpx.Response(404, json={"error": "model 'qwen3:8b' not found"})

    async def go():
        async with make_client(handler) as client:
            await client.complete("ollama:qwen3:8b", "sys", "q")

    with pytest.raises(LLMError, match="404"):
        asyncio.run(go())
    assert len(attempts) == 1


def test_concurrency_is_bounded():
    in_flight = 0
    peak = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return httpx.Response(200, json=CHAT_RESPONSE)

    async def go():
        async with make_client(handler, concurrency=2) as client:
            await asyncio.gather(*(client.complete("ollama:qwen3:8b", "sys", f"q{i}") for i in range(6)))

    asyncio.run(go())
    assert peak == 2


def test_run_config_records_digest_and_options():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/show":
            return httpx.Response(200, json={"details": {"family": "qwen3", "parameter_size": "8.2B", "quantization_level": "Q4_K_M"}})
        if request.url.path == "/api/tags":
            return httpx.Response(
                200,
                json={
                    "models": [
                        {"name": "nomic-embed-text:latest", "model": "nomic-embed-text:latest", "digest": "aaa"},
                        {"name": "qwen3:8b", "model": "qwen3:8b", "digest": "bbb"},
                        {"name": "llama3.2:latest", "model": "llama3.2:latest", "digest": "ccc"},
                    ]
                },
            )
        return httpx.Response(404)

    async def go():
        async with make_client(handler, options=SamplingOptions(seed=1)) as client:
            return await client.run_config({"answer": "ollama:qwen3:8b", "judge": "ollama:llama3.2"})

    config = asyncio.run(go())
    assert config["host"] == "http://ollama.test"
    assert config["options"] == {"temperature": 0.0, "seed": 1, "num_ctx": 8192, "num_predict": 256, "think": False}
    assert config["models"]["answer"] == {
        "id": "ollama:qwen3:8b",
        "digest": "bbb",
        "details": {"family": "qwen3", "parameter_size": "8.2B", "quantization_level": "Q4_K_M"},
    }
    # An untagged name resolves to :latest, as Ollama does.
    assert config["models"]["judge"]["digest"] == "ccc"


def test_describe_model_requires_pulled_model():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/show":
            return httpx.Response(200, json={"details": {}})
        return httpx.Response(200, json={"models": []})

    async def go():
        async with make_client(handler) as client:
            await client.describe_model("ollama:qwen3:8b")

    with pytest.raises(LLMError, match="ollama pull qwen3:8b"):
        asyncio.run(go())


def test_host_defaults(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    assert LLMClient(pricing={}).host == "http://localhost:11434"
    monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:11434")
    assert LLMClient(pricing={}).host == "http://127.0.0.1:11434"
