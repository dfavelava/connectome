"""A thin LLM layer for the answer and judge stages.

Models are named `provider:exact-model-id` (e.g. `ollama:qwen3:8b`). Only
`ollama` is implemented; the prefix leaves room for hosted providers without
changing result files.

Calls go to Ollama's native /api/chat at OLLAMA_HOST with deterministic
options (temperature 0, a fixed seed and context size, bounded output), and
thinking off. Connection errors, timeouts and 5xx responses are retried with
exponential backoff, and a semaphore bounds in-flight calls, since a local
Ollama serves requests roughly one at a time unless OLLAMA_NUM_PARALLEL is set.

Local calls cost nothing, so usage is accounted in tokens and wall-clock time
per stage. `cost_usd` comes from pricing.toml and is null for models it
doesn't list.
"""

import asyncio
import fnmatch
import os
import re
import time
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NamedTuple

import httpx

PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_PRICING_PATH = PROJECT_DIR / "pricing.toml"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
PROVIDERS = ("ollama",)

# Reasoning models may still emit their thinking inline; it is never part of the answer.
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


class LLMError(Exception):
    pass


@dataclass(frozen=True)
class ModelId:
    provider: str
    name: str

    def __str__(self) -> str:
        return f"{self.provider}:{self.name}"


def parse_model_id(model: str) -> ModelId:
    """Split `provider:exact-model-id` at the first colon; the name may contain more (`qwen3:8b`)."""
    provider, sep, name = model.partition(":")
    if not sep or not provider or not name:
        raise ValueError(f"model id {model!r} is not of the form provider:model (e.g. ollama:qwen3:8b)")
    if provider not in PROVIDERS:
        raise ValueError(f"unknown model provider {provider!r} in {model!r}; supported: {', '.join(PROVIDERS)}")
    return ModelId(provider, name)


@dataclass(frozen=True)
class SamplingOptions:
    temperature: float = 0.0
    seed: int = 42
    # Room for the system prompt, answer_k turns and the question.
    num_ctx: int = 8192
    # Upper bound on generated tokens; answers and verdicts are short.
    num_predict: int = 256
    think: bool = False

    def ollama_options(self) -> dict:
        return {
            "temperature": self.temperature,
            "seed": self.seed,
            "num_ctx": self.num_ctx,
            "num_predict": self.num_predict,
        }


def chat_request(model: ModelId, system: str, user: str, options: SamplingOptions) -> dict:
    """The /api/chat request body for one non-streaming call."""
    return {
        "model": model.name,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "think": options.think,
        "options": options.ollama_options(),
    }


def strip_thinking(text: str) -> str:
    return _THINK_BLOCK.sub("", text).strip()


def load_pricing(path: Path = DEFAULT_PRICING_PATH) -> dict[str, dict[str, float]]:
    """Model pattern -> {input_per_mtok, output_per_mtok}, in file order."""
    with path.open("rb") as f:
        return tomllib.load(f).get("models", {})


def price_for(pricing: dict[str, dict[str, float]], model: str) -> dict[str, float] | None:
    """The entry for `model`: an exact key wins, else the first glob pattern that matches.
    None if nothing matches - an unknown model has no cost, not a guessed one."""
    if model in pricing:
        return pricing[model]
    for pattern, price in pricing.items():
        if fnmatch.fnmatchcase(model, pattern):
            return price
    return None


def cost_usd(pricing: dict[str, dict[str, float]], model: str, input_tokens: int, output_tokens: int) -> float | None:
    price = price_for(pricing, model)
    if price is None:
        return None
    return (input_tokens * price["input_per_mtok"] + output_tokens * price["output_per_mtok"]) / 1_000_000


class Completion(NamedTuple):
    text: str
    input_tokens: int
    output_tokens: int


@dataclass
class StageUsage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    # Wall-clock time spent in calls, retries included.
    seconds: float = 0.0
    # Ollama's own generation time, for tokens/sec.
    eval_seconds: float = 0.0


class LLMClient:
    """Runs chat completions against the providers named in model ids."""

    def __init__(
        self,
        *,
        host: str | None = None,
        options: SamplingOptions = SamplingOptions(),
        concurrency: int = 1,
        timeout: float = 600.0,
        max_retries: int = 4,
        backoff: float = 2.0,
        pricing: dict[str, dict[str, float]] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.host = (host or os.environ.get("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST).rstrip("/")
        if "://" not in self.host:
            # Ollama itself accepts OLLAMA_HOST without a scheme.
            self.host = f"http://{self.host}"
        self.options = options
        self.max_retries = max_retries
        self.backoff = backoff
        self.pricing = load_pricing() if pricing is None else pricing
        self.usage: dict[tuple[str, str], StageUsage] = {}
        self._semaphore = asyncio.Semaphore(concurrency)
        self._http = httpx.AsyncClient(base_url=self.host, timeout=timeout, transport=transport)

    async def __aenter__(self) -> "LLMClient":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._http.aclose()

    async def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        """Send a request, retrying connection errors, timeouts and 5xx responses."""
        for attempt in range(self.max_retries + 1):
            try:
                response = await self._http.request(method, path, json=body)
                if response.status_code < 500:
                    if response.is_error:
                        raise LLMError(f"{method} {path} failed ({response.status_code}): {response.text}")
                    return response.json()
                error: Exception = LLMError(f"{method} {path} failed ({response.status_code}): {response.text}")
            except httpx.TransportError as exc:  # includes timeouts
                error = exc
            if attempt == self.max_retries:
                raise LLMError(f"{method} {path} failed after {attempt + 1} attempts: {error!r}") from error
            await asyncio.sleep(self.backoff * 2**attempt)
        raise AssertionError("unreachable")

    async def complete(self, model: str, system: str, user: str, *, stage: str = "default") -> Completion:
        """One chat completion; usage is added to `stage`."""
        model_id = parse_model_id(model)
        async with self._semaphore:
            started = time.monotonic()
            data = await self._request("POST", "/api/chat", chat_request(model_id, system, user, self.options))
            elapsed = time.monotonic() - started
        completion = Completion(
            text=strip_thinking(data.get("message", {}).get("content", "")),
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
        )
        usage = self.usage.setdefault((stage, str(model_id)), StageUsage())
        usage.calls += 1
        usage.input_tokens += completion.input_tokens
        usage.output_tokens += completion.output_tokens
        usage.seconds += elapsed
        usage.eval_seconds += data.get("eval_duration", 0) / 1e9
        return completion

    async def describe_model(self, model: str) -> dict:
        """The model's identity for the run config: the id, the digest of the
        weights its tag currently points to, and its details."""
        model_id = parse_model_id(model)
        show = await self._request("POST", "/api/show", {"model": model_id.name})
        # /api/show has no digest; /api/tags lists it per local model, always with a tag.
        tag = model_id.name if ":" in model_id.name else f"{model_id.name}:latest"
        tags = await self._request("GET", "/api/tags")
        digest = next((m.get("digest") for m in tags.get("models", []) if tag in (m.get("name"), m.get("model"))), None)
        if digest is None:
            raise LLMError(f"model {model_id.name!r} is not pulled on {self.host}; run `docker compose exec ollama ollama pull {model_id.name}`")
        return {
            "id": str(model_id),
            "digest": digest,
            "details": show.get("details", {}),
        }

    async def run_config(self, models: dict[str, str]) -> dict:
        """The LLM part of a run config, for `models` like {"answer": ..., "judge": ...}."""
        return {
            "host": self.host,
            "options": asdict(self.options),
            "models": {role: await self.describe_model(model) for role, model in models.items()},
        }

    def usage_summary(self) -> list[dict]:
        """Usage per stage and model: tokens, calls, wall time, tokens/sec and cost (null if unpriced)."""
        return [
            {
                "stage": stage,
                "model": model,
                "calls": usage.calls,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "seconds": round(usage.seconds, 3),
                "output_tokens_per_second": round(usage.output_tokens / usage.eval_seconds, 2) if usage.eval_seconds else None,
                "cost_usd": cost_usd(self.pricing, model, usage.input_tokens, usage.output_tokens),
            }
            for (stage, model), usage in self.usage.items()
        ]
