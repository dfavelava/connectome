"""Regression test for the recall tool's MCP dispatch path.

Unlike test_e2e_memory.py, which calls `recall(...)` directly as a plain
Python function (bypassing MCP argument validation), this exercises the same
validate_arguments -> call_fn path the MCP server uses for a real tool call.
That path builds call kwargs keyed by each field's alias, so a field whose
alias isn't a valid Python identifier (like `as`, a reserved keyword) will
silently break every call unless the alias is validation-only.
"""

import asyncio

from mcp.server.mcpserver.utilities.func_metadata import func_metadata

from daybidmcp import server


def test_recall_dispatch_accepts_as_argument_without_crashing(monkeypatch):
    captured = {}

    async def fake_request(method, path, *, params=None, files=None, json_body=None):
        captured["method"] = method
        captured["path"] = path
        captured["json_body"] = json_body

        class FakeResponse:
            text = '{"results": []}'

        return FakeResponse()

    monkeypatch.setattr(server, "request", fake_request)

    meta = func_metadata(server.recall, skip_names=[])
    validated = meta.validate_arguments({"query": "test", "k": 1, "as": "someone"})

    result = asyncio.run(meta.call_fn(server.recall, True, validated, {}))

    assert result == '{"results": []}'
    assert captured["json_body"]["as"] == "someone"
