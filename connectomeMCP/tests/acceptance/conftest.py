"""Shared backend and temp-tome fixtures for the acceptance suite (phase 1.2F).

Unlike test_e2e_memory.py's per-test `backend` fixture, which builds and boots
a fresh Go backend for every test, this suite boots the backend once per test
session (`acceptance_backend`) and reuses it across every acceptance test -
"one already-up backend + Postgres rather than a fresh stack per test". Each
test isolates itself instead with its own fresh temp-<uuid> tome
(`temp_tome`), destroyed afterward even if the test fails.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import tempfile
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_DIR = REPO_ROOT / "backend"
API_KEY = "acceptance-test-token"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_until_ready(url: str, proc: subprocess.Popen[str], log_path: Path, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"backend exited early with code {proc.returncode}:\n{log_path.read_text()}")
        try:
            if httpx.get(url, timeout=1.0).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.2)
    raise RuntimeError("backend did not become ready in time")


@dataclass
class AcceptanceBackend:
    base_url: str
    api_key: str


@pytest.fixture(scope="session")
def acceptance_backend() -> Iterator[AcceptanceBackend]:
    go = shutil.which("go")
    if go is None:
        pytest.skip("go toolchain not available")

    work_dir = Path(tempfile.mkdtemp(prefix="connectome-acceptance-"))
    home = work_dir / "home"
    (home / ".connectome").mkdir(parents=True)

    binary = work_dir / "connectome-backend"
    build = subprocess.run(
        [go, "build", "-o", str(binary), "."],
        cwd=BACKEND_DIR,
        capture_output=True,
        text=True,
    )
    if build.returncode != 0:
        shutil.rmtree(work_dir, ignore_errors=True)
        pytest.fail(f"go build failed:\n{build.stderr}")

    port = _free_port()
    env = {
        **os.environ,
        "HOME": str(home),
        "PORT": str(port),
        "MEMORY_MANAGER": "local",
        "apikey": API_KEY,
        "GIN_MODE": "release",
    }

    # A file, not subprocess.PIPE: this process stays alive for the whole
    # session across many tests, and an unread PIPE would eventually fill and
    # deadlock the backend once enough log lines accumulate.
    log_path = work_dir / "backend.log"
    log_file = log_path.open("w")
    proc = subprocess.Popen(
        [str(binary)],
        cwd=work_dir,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        _wait_until_ready(f"http://127.0.0.1:{port}/api/", proc, log_path)
    except Exception:
        proc.kill()
        log_file.close()
        shutil.rmtree(work_dir, ignore_errors=True)
        raise

    try:
        yield AcceptanceBackend(base_url=f"http://127.0.0.1:{port}/api/connectome", api_key=API_KEY)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        log_file.close()
        shutil.rmtree(work_dir, ignore_errors=True)


@pytest.fixture
def temp_tome(acceptance_backend: AcceptanceBackend, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """Points connectomemcp.server at the shared backend and yields a fresh temp-<uuid> tome id.

    Destroys the tome (every blob and embeddings row under it) afterward, even
    if the test fails. The temp- prefix matches the test-convention DestroyTome
    (backend/resources/tome.go) treats as obviously disposable, so this never
    needs confirm=true and can never reach the default or a real tome like
    "west-marches".
    """
    monkeypatch.setenv("CONNECTOME_API_BASE_URL", acceptance_backend.base_url)
    monkeypatch.setenv("CONNECTOME_API_KEY", acceptance_backend.api_key)

    tome = f"temp-{uuid.uuid4()}"
    try:
        yield tome
    finally:
        response = httpx.delete(
            f"{acceptance_backend.base_url}/tome/{tome}",
            headers={"Authorization": f"Bearer {acceptance_backend.api_key}"},
            timeout=10.0,
        )
        response.raise_for_status()
