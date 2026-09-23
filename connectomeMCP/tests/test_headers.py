from connectomemcp.server import get_headers


def test_get_headers_includes_cf_access_service_token(monkeypatch):
    monkeypatch.setenv("CF_ACCESS_CLIENT_ID", "id.access")
    monkeypatch.setenv("CF_ACCESS_CLIENT_SECRET", "secret")

    headers = get_headers()

    assert headers["CF-Access-Client-Id"] == "id.access"
    assert headers["CF-Access-Client-Secret"] == "secret"


def test_get_headers_skips_cf_access_when_incomplete(monkeypatch):
    monkeypatch.setenv("CF_ACCESS_CLIENT_ID", "id.access")
    monkeypatch.delenv("CF_ACCESS_CLIENT_SECRET", raising=False)

    headers = get_headers()

    assert "CF-Access-Client-Id" not in headers
    assert "CF-Access-Client-Secret" not in headers
