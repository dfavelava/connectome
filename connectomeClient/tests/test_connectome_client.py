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


async def test_remember_writes_null_occurred_at_when_not_given(patch_async_client):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["request"] = request
        return httpx.Response(200, json={"message": "success"})

    set_handler(patch_async_client, handler)

    client = make_client()
    _ = await client.remember("hello")

    body = captured["request"].content.decode("utf-8")
    document = body.split("\r\n\r\n", 1)[1].rsplit("\r\n--", 1)[0]
    frontmatter_yaml = document.split("---\n", 2)[1]
    metadata = yaml.safe_load(frontmatter_yaml)

    assert metadata["occurred_at"] is None


async def test_remember_writes_explicit_occurred_at(patch_async_client):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["request"] = request
        return httpx.Response(200, json={"message": "success"})

    set_handler(patch_async_client, handler)

    client = make_client()
    occurred_at = "2023-06-15T09:30:00+00:00"
    _ = await client.remember("hello", occurred_at=occurred_at)

    body = captured["request"].content.decode("utf-8")
    document = body.split("\r\n\r\n", 1)[1].rsplit("\r\n--", 1)[0]
    frontmatter_yaml, content = document.split("---\n", 2)[1:]
    metadata = yaml.safe_load(frontmatter_yaml)

    assert metadata["occurred_at"] == occurred_at
    assert metadata["created_at"] != occurred_at
    assert content.strip("\n") == "hello"


async def test_remember_rejects_malformed_occurred_at(patch_async_client):
    client = make_client()
    with pytest.raises(ValueError, match="occurred_at"):
        _ = await client.remember("hello", occurred_at="not-a-date")


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


async def test_recall_sends_occurred_since_and_occurred_until_distinct_from_since_until(patch_async_client):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": []})

    set_handler(patch_async_client, handler)

    client = make_client()
    _ = await client.recall(
        "what happened",
        since="2020-01-01T00:00:00Z",
        occurred_since="1999-01-01T00:00:00Z",
        occurred_until="1999-12-31T00:00:00Z",
    )

    request = last_request(patch_async_client)
    body = json.loads(request.content)
    assert body["filters"] == {
        "since": "2020-01-01T00:00:00Z",
        "occurred_since": "1999-01-01T00:00:00Z",
        "occurred_until": "1999-12-31T00:00:00Z",
    }


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


async def test_remember_sends_tome_as_a_form_field(patch_async_client):
    set_handler(patch_async_client, lambda request: httpx.Response(200, json={"message": "success"}))

    client = make_client()
    result = await client.remember("hello", tome="temp-abc")

    body = last_request(patch_async_client).content.decode("utf-8")
    assert 'name="tome"\r\n\r\ntemp-abc\r\n' in body
    assert f'filename="{result["key"]}"' in body


async def test_remember_omits_tome_field_when_not_given(patch_async_client):
    set_handler(patch_async_client, lambda request: httpx.Response(200, json={"message": "success"}))

    client = make_client()
    _ = await client.remember("hello")

    body = last_request(patch_async_client).content.decode("utf-8")
    assert 'name="tome"' not in body


async def test_recall_sends_tome_as_a_filter(patch_async_client):
    set_handler(patch_async_client, lambda request: httpx.Response(200, json={"results": []}))

    client = make_client()
    _ = await client.recall("what does david drink", tome="temp-abc")

    body = json.loads(last_request(patch_async_client).content)
    assert body["filters"] == {"tome": "temp-abc"}


async def test_recall_combines_tome_with_other_filters(patch_async_client):
    set_handler(patch_async_client, lambda request: httpx.Response(200, json={"results": []}))

    client = make_client()
    _ = await client.recall("tea", entity="david", tome="temp-abc")

    body = json.loads(last_request(patch_async_client).content)
    assert body["filters"] == {"entity": "david", "tome": "temp-abc"}


async def test_assert_relationship_includes_tome_when_given(patch_async_client):
    set_handler(patch_async_client, lambda request: httpx.Response(200, json={"subject": {"id": "west-1"}}))

    client = make_client()
    _ = await client.assert_relationship("west-1", "member_of", "Party A", tome="temp-abc")

    body = json.loads(last_request(patch_async_client).content)
    assert body["tome"] == "temp-abc"


async def test_supersede_relationship_includes_tome_when_given(patch_async_client):
    set_handler(patch_async_client, lambda request: httpx.Response(200, json={"message": "success"}))

    client = make_client()
    _ = await client.supersede_relationship("mem_old.md", "west-1", "plays", tome="temp-abc")

    body = json.loads(last_request(patch_async_client).content)
    assert body["tome"] == "temp-abc"


async def test_get_memory_sends_tome_as_query_param(patch_async_client):
    set_handler(patch_async_client, lambda request: httpx.Response(200, json={"content": "..."}))

    client = make_client()
    _ = await client.get_memory("mem_abc.md", tome="temp-abc")

    params = last_request(patch_async_client).url.params
    assert params["key"] == "mem_abc.md"
    assert params["tome"] == "temp-abc"


async def test_get_memory_omits_tome_param_when_not_given(patch_async_client):
    set_handler(patch_async_client, lambda request: httpx.Response(200, json={"content": "..."}))

    client = make_client()
    _ = await client.get_memory("mem_abc.md")

    assert "tome" not in last_request(patch_async_client).url.params


async def test_get_entity_forwards_tome(patch_async_client):
    set_handler(
        patch_async_client,
        lambda request: httpx.Response(200, json={"content": json.dumps({"id": "thorin"})}),
    )

    client = make_client()
    entity = await client.get_entity("thorin", tome="temp-abc")

    params = last_request(patch_async_client).url.params
    assert params["key"] == "ent_thorin.json"
    assert params["tome"] == "temp-abc"
    assert entity == {"id": "thorin"}


async def test_browse_all_sends_tome_as_query_param(patch_async_client):
    set_handler(patch_async_client, lambda request: httpx.Response(200, json={"Contents": []}))

    client = make_client()
    _ = await client.browse_all(tome="temp-abc")

    request = last_request(patch_async_client)
    assert request.url.path == "/api/connectome/memory/list"
    assert request.url.params["tome"] == "temp-abc"


async def test_browse_all_omits_tome_param_when_not_given(patch_async_client):
    set_handler(patch_async_client, lambda request: httpx.Response(200, json={"Contents": []}))

    client = make_client()
    _ = await client.browse_all()

    assert "tome" not in last_request(patch_async_client).url.params


async def test_forget_includes_tome_in_body_when_given(patch_async_client):
    set_handler(patch_async_client, lambda request: httpx.Response(200, json={}))

    client = make_client()
    result = await client.forget("mem_abc.md", tome="temp-abc")

    assert json.loads(last_request(patch_async_client).content) == {"key": "mem_abc.md", "tome": "temp-abc"}
    assert result == {"message": "deleted", "key": "mem_abc.md"}


async def test_destroy_tome_sends_delete_without_confirm_by_default(patch_async_client):
    set_handler(
        patch_async_client,
        lambda request: httpx.Response(200, json={"message": "destroyed", "tome": "temp-abc"}),
    )

    client = make_client()
    result = await client.destroy_tome("temp-abc")

    request = last_request(patch_async_client)
    assert request.method == "DELETE"
    assert str(request.url) == "http://example.test/api/connectome/tome/temp-abc"
    assert result == {"message": "destroyed", "tome": "temp-abc"}


async def test_destroy_tome_sends_confirm_when_requested(patch_async_client):
    set_handler(patch_async_client, lambda request: httpx.Response(200, json={"message": "destroyed"}))

    client = make_client()
    _ = await client.destroy_tome("west-marches", confirm=True)

    request = last_request(patch_async_client)
    assert request.url.path == "/api/connectome/tome/west-marches"
    assert request.url.params["confirm"] == "true"


async def test_destroy_tome_percent_encodes_the_tome_id(patch_async_client):
    set_handler(patch_async_client, lambda request: httpx.Response(200, json={"message": "destroyed"}))

    client = make_client()
    _ = await client.destroy_tome("temp-a b?c#d")

    request = last_request(patch_async_client)
    assert request.url.raw_path.decode() == "/api/connectome/tome/temp-a%20b%3Fc%23d"
    assert request.url.query == b""


async def test_destroy_tome_refuses_the_default_tome_without_a_request(patch_async_client):
    client = make_client()

    with pytest.raises(ValueError, match="default tome"):
        _ = await client.destroy_tome("")

    assert not patch_async_client.transports


async def test_destroy_tome_surfaces_the_backends_guard_as_an_http_error(patch_async_client):
    set_handler(
        patch_async_client,
        lambda request: httpx.Response(403, json={"error": "destroying this tome requires confirm=true"}),
    )

    client = make_client()
    with pytest.raises(httpx.HTTPStatusError) as excinfo:
        _ = await client.destroy_tome("west-marches")

    assert excinfo.value.response.status_code == 403


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


def test_cf_access_headers_sent_only_when_both_credentials_set(monkeypatch):
    monkeypatch.delenv("CF_ACCESS_CLIENT_ID", raising=False)
    monkeypatch.delenv("CF_ACCESS_CLIENT_SECRET", raising=False)

    headers = make_client(cf_access_client_id="id.access", cf_access_client_secret="secret")._headers()
    assert headers["CF-Access-Client-Id"] == "id.access"
    assert headers["CF-Access-Client-Secret"] == "secret"

    headers = make_client(cf_access_client_id="id.access")._headers()
    assert "CF-Access-Client-Id" not in headers
    assert "CF-Access-Client-Secret" not in headers


def test_cf_access_credentials_fall_back_to_env_vars(monkeypatch):
    monkeypatch.setenv("CF_ACCESS_CLIENT_ID", "env-id.access")
    monkeypatch.setenv("CF_ACCESS_CLIENT_SECRET", "env-secret")

    headers = make_client()._headers()

    assert headers["CF-Access-Client-Id"] == "env-id.access"
    assert headers["CF-Access-Client-Secret"] == "env-secret"
