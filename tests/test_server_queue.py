import threading
import time

import pytest
from fastapi.testclient import TestClient

from jeff import server


class Uniform:
    """A stand-in model that gives every option the same probability."""
    backend = "mlx"
    base_model = "Qwen/Qwen3.5-0.8B"

    def decide(self, rows, max_length):
        return [([1 / len(row["question"]["criteria"])] * len(row["question"]["criteria"]), 10) for row in rows]


REQUEST = {"model": "jeff", "state": "Book me a flight.",
           "questions": {"q": {"type": "choice", "instructions": "Which destination?",
                               "criteria": {"Paris": None, "Rome": None}}}}


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(server.service, "model", Uniform())
    monkeypatch.setattr(server.service, "max_options", 26)
    monkeypatch.setattr(server.service, "lock", threading.Lock())
    return TestClient(server.app)


def test_a_busy_model_answers_529_at_once_by_default(client: TestClient) -> None:
    assert server.service.queue_seconds == 0
    server.service.lock.acquire()
    started = time.monotonic()
    response = client.post("/v1/systemone", json=REQUEST)
    assert response.status_code == 529
    assert response.headers["Retry-After"] == "1"
    assert time.monotonic() - started < 0.5
    server.service.lock.release()


def test_a_request_waits_for_the_model_within_the_queue_time(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server.service, "queue_seconds", 2.0)
    server.service.lock.acquire()
    threading.Timer(0.2, server.service.lock.release).start()
    response = client.post("/v1/systemone", json=REQUEST)
    assert response.status_code == 200
    assert server.service.lock.acquire(blocking=False), "the request must release the lock"


def test_a_request_gets_529_when_the_queue_time_runs_out(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server.service, "queue_seconds", 0.2)
    server.service.lock.acquire()
    started = time.monotonic()
    response = client.post("/v1/systemone", json=REQUEST)
    assert response.status_code == 529
    assert 0.2 <= time.monotonic() - started < 1.5
    server.service.lock.release()
    assert server.service.lock.acquire(blocking=False), "a timed-out request must not hold the lock"


class Failing(Uniform):
    def decide(self, rows, max_length):
        raise ValueError("the state is too long")


def test_a_failed_decision_releases_the_lock(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server.service, "model", Failing())
    response = client.post("/v1/systemone", json=REQUEST)
    assert response.status_code == 422
    assert server.service.lock.acquire(blocking=False)


def test_the_queue_time_comes_from_jeff_queue_ms() -> None:
    assert server.queue_seconds(None) == 0
    assert server.queue_seconds("0") == 0
    assert server.queue_seconds("1500") == 1.5
    for value in ("-1", "abc", "1.5", ""):
        with pytest.raises(ValueError, match="JEFF_QUEUE_MS"):
            server.queue_seconds(value)
