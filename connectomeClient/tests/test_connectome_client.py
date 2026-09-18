import json

import httpx
import pytest
import yaml

from connectomeclient import (
    DEFAULT_MEMORY_TYPE,
    DEFAULT_SOURCE_TYPE,
    MEMORY_SCHEMA_VERSION,
    ConnectomeClient,
)


class RecordingTransport(httpx.MockTransport):
    """A MockTransport that also records every request it handled."""

    def __init__(self, handler):
        self.requests: list[httpx.Request] = []

        def recording_handler(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            return handler(request)

        super().__init__(recording_handler)


@pytest.fixture(autouse=True)
def patch_async_client(monkeypatch):
    """Route every httpx.AsyncClient the client creates through a fresh RecordingTransport."""
    transports: list[RecordingTransport] = []
    original_init = httpx.AsyncClient.__init__

    def patched_init(self, *args, **kwargs):
        handler = patch_async_client.handler
        transport = RecordingTransport(handler)
        transports.append(transport)
        kwargs["transport"] = transport
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", patched_init)
    patch_async_client.handler = lambda request: httpx.Response(200, json={})
    patch_async_client.transports = transports
    yield patch_async_client


def set_handler(fixture, handler):
    fixture.handler = handler


def last_request(fixture) -> httpx.Request:
    return fixture.transports[-1].requests[-1]


def make_client(**kwargs) -> ConnectomeClient:
    return ConnectomeClient(base_url="http://example.test/api/connectome", api_key="test-key", **kwargs)


async def test_remember_posts_a_valid_memory_document(patch_async_client):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["request"] = request
        return httpx.Response(200, json={"message": "success"})

    set_handler(patch_async_client, handler)

    client = make_client()
    result = await client.remember("hello world", entities=["david"])

    request = captured["request"]
    assert request.method == "POST"
    assert str(request.url) == "http://example.test/api/connectome/memory/"
    assert request.headers["Authorization"] == "Bearer test-key"

    body = request.content.decode("utf-8")
    assert f'filename="{result["key"]}"' in body

    document = body.split("\r\n\r\n", 1)[1].rsplit("\r\n--", 1)[0]
    assert document.startswith("---\n")
    frontmatter_yaml, content = document.split("---\n", 2)[1:]
    metadata = yaml.safe_load(frontmatter_yaml)

    assert metadata["version"] == MEMORY_SCHEMA_VERSION
    assert metadata["type"] == DEFAULT_MEMORY_TYPE
    assert metadata["entities"] == ["david"]
    assert metadata["source"]["type"] == DEFAULT_SOURCE_TYPE
    assert "acl" not in metadata
    assert content.strip("\n") == "hello world"


async def test_remember_uses_the_client_supplied_source_type(patch_async_client):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["request"] = request
        return httpx.Response(200, json={"message": "success"})

    set_handler(patch_async_client, handler)

    client = make_client(source_type="discord")
    _ = await client.remember("hello world")

    body = captured["request"].content.decode("utf-8")
    document = body.split("\r\n\r\n", 1)[1].rsplit("\r\n--", 1)[0]
    frontmatter_yaml = document.split("---\n", 2)[1]
    metadata = yaml.safe_load(frontmatter_yaml)

    assert metadata["source"]["type"] == "discord"


async def test_remember_includes_relationships_when_given(patch_async_client):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["request"] = request
        return httpx.Response(200, json={"message": "success"})

    set_handler(patch_async_client, handler)

    client = make_client()
    _ = await client.remember(
        "David likes tea.",
        relationships=[
            {"subjectEntityId": "david", "predicate": "likes", "objectEntityId": "tea", "kind": "fact"}
        ],
    )

    body = captured["request"].content.decode("utf-8")
    document = body.split("\r\n\r\n", 1)[1].rsplit("\r\n--", 1)[0]
    frontmatter_yaml = document.split("---\n", 2)[1]
    metadata = yaml.safe_load(frontmatter_yaml)

    assert metadata["relationships"] == [
        {"subjectEntityId": "david", "predicate": "likes", "objectEntityId": "tea", "kind": "fact"}
    ]


async def test_remember_defaults_relationships_to_empty_list(patch_async_client):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["request"] = request
        return httpx.Response(200, json={"message": "success"})

    set_handler(patch_async_client, handler)

    client = make_client()
    _ = await client.remember("hello world")

    body = captured["request"].content.decode("utf-8")
    document = body.split("\r\n\r\n", 1)[1].rsplit("\r\n--", 1)[0]
    frontmatter_yaml = document.split("---\n", 2)[1]
    metadata = yaml.safe_load(frontmatter_yaml)

    assert metadata["relationships"] == []


async def test_remember_includes_acl_when_given(patch_async_client):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["request"] = request
        return httpx.Response(200, json={"message": "success"})

    set_handler(patch_async_client, handler)

    client = make_client()
    _ = await client.remember("hello", acl=["GM"])

    body = captured["request"].content.decode("utf-8")
    document = body.split("\r\n\r\n", 1)[1].rsplit("\r\n--", 1)[0]
    frontmatter_yaml = document.split("---\n", 2)[1]
    metadata = yaml.safe_load(frontmatter_yaml)

    assert metadata["acl"] == ["GM"]


async def test_recall_sends_query_and_filters(patch_async_client):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": []})

    set_handler(patch_async_client, handler)

    client = make_client()
    result = await client.recall("what does david drink", k=3, entity="david", hydrate=True)

    request = last_request(patch_async_client)
    assert request.method == "POST"
    assert str(request.url) == "http://example.test/api/connectome/memory/search"
    body = json.loads(request.content)
    assert body == {
        "query": "what does david drink",
        "k": 3,
        "hydrate": True,
        "filters": {"entity": "david"},
    }
    assert result == {"results": []}


async def test_assert_relationship_posts_subject_predicate_object(patch_async_client):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"subject": {"id": "west-1", "member_of": ["Party A"]}, "object": {"id": "Party A"}},
        )

    set_handler(patch_async_client, handler)

    client = make_client()
    result = await client.assert_relationship("west-1", "member_of", "Party A")

    request = last_request(patch_async_client)
    assert request.method == "POST"
    assert str(request.url) == "http://example.test/api/connectome/entity/relationship"
    assert json.loads(request.content) == {
        "subjectEntityId": "west-1",
        "predicate": "member_of",
        "objectEntityId": "Party A",
    }
    assert result["subject"]["member_of"] == ["Party A"]


async def test_assert_relationship_omits_optional_fields_when_not_given(patch_async_client):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"subject": {"id": "west-1"}})

    set_handler(patch_async_client, handler)

    client = make_client()
    _ = await client.assert_relationship("west-1", "is_tired")

    request = last_request(patch_async_client)
    assert json.loads(request.content) == {"subjectEntityId": "west-1", "predicate": "is_tired"}


async def test_assert_relationship_includes_kind_when_given(patch_async_client):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"subject": {"id": "west-1"}})

    set_handler(patch_async_client, handler)

    client = make_client()
    _ = await client.assert_relationship("west-1", "likes", "tea", kind="rumor")

    request = last_request(patch_async_client)
    body = json.loads(request.content)
    assert body["kind"] == "rumor"


async def test_assert_relationship_includes_subject_kind_and_meta_when_given(patch_async_client):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"subject": {"id": "thorin"}})

    set_handler(patch_async_client, handler)

    client = make_client()
    _ = await client.assert_relationship(
        "thorin",
        "member_of",
        "west-1-characters",
        subject_kind="character",
        subject_meta={"owner": "west-1"},
    )

    request = last_request(patch_async_client)
    body = json.loads(request.content)
    assert body["subjectKind"] == "character"
    assert body["subjectMeta"] == {"owner": "west-1"}


async def test_get_entity_parses_the_entity_json_content(patch_async_client):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"content": json.dumps({"id": "thorin", "kind": "character", "meta": {"owner": "west-1"}})},
        )

    set_handler(patch_async_client, handler)

    client = make_client()
    entity = await client.get_entity("thorin")

    request = last_request(patch_async_client)
    assert request.url.params["key"] == "ent_thorin.json"
    assert entity == {"id": "thorin", "kind": "character", "meta": {"owner": "west-1"}}


async def test_get_entity_returns_none_when_not_found(patch_async_client):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "not found"})

    set_handler(patch_async_client, handler)

    client = make_client()
    entity = await client.get_entity("thorin")

    assert entity is None


async def test_supersede_relationship_sends_patch_with_expected_body(patch_async_client):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": "success", "key": "mem_old.md"})

    set_handler(patch_async_client, handler)

    client = make_client()
    result = await client.supersede_relationship(
        "mem_old.md", "west-1", "plays", "thorin", superseded_by="mem_new.md"
    )

    request = last_request(patch_async_client)
    assert request.method == "PATCH"
    assert str(request.url) == "http://example.test/api/connectome/memory/relationship"
    assert json.loads(request.content) == {
        "key": "mem_old.md",
        "subjectEntityId": "west-1",
        "predicate": "plays",
        "objectEntityId": "thorin",
        "superseded_by": "mem_new.md",
    }
    assert result == {"message": "success", "key": "mem_old.md"}


async def test_supersede_relationship_omits_optional_fields_when_not_given(patch_async_client):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": "success"})

    set_handler(patch_async_client, handler)

    client = make_client()
    _ = await client.supersede_relationship("mem_old.md", "west-1", "plays")

    request = last_request(patch_async_client)
    assert json.loads(request.content) == {
        "key": "mem_old.md",
        "subjectEntityId": "west-1",
        "predicate": "plays",
        "objectEntityId": None,
        "superseded_by": None,
    }


async def test_get_memory_sends_key_as_query_param(patch_async_client):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"content": "..."})

    set_handler(patch_async_client, handler)

    client = make_client()
    _ = await client.get_memory("mem_abc.md")

    request = last_request(patch_async_client)
    assert request.method == "GET"
    assert request.url.params["key"] == "mem_abc.md"


async def test_forget_sends_delete_with_key_body(patch_async_client):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    set_handler(patch_async_client, handler)

    client = make_client()
    result = await client.forget("mem_abc.md")

    request = last_request(patch_async_client)
    assert request.method == "DELETE"
    assert json.loads(request.content) == {"key": "mem_abc.md"}
    assert result == {"message": "deleted", "key": "mem_abc.md"}


async def test_raises_on_http_error(patch_async_client):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "unauthorized"})

    set_handler(patch_async_client, handler)

    client = make_client()
    with pytest.raises(httpx.HTTPStatusError):
        _ = await client.browse_all()


def test_api_key_falls_back_to_daybid_env_vars(monkeypatch):
    monkeypatch.delenv("CONNECTOME_API_KEY", raising=False)
    monkeypatch.setenv("DAYBID_API_KEY", "fallback-key")

    client = ConnectomeClient(base_url="http://example.test")

    assert client.api_key == "fallback-key"
