import pytest
from fastapi.testclient import TestClient

from jeff import server


class Recording:
    """A stand-in MLX model that refuses inputs over the limit it is given, as the real backends do."""
    backend = "mlx"
    base_model = "Qwen/Qwen3.5-0.8B"

    def decide(self, rows, max_length):
        for row in rows:
            if len(row["state"].split()) > max_length:
                raise ValueError(f"Question branch exceeds the {max_length}-token limit; no input was truncated.")
        return [([1 / len(row["question"]["criteria"])] * len(row["question"]["criteria"]), 10) for row in rows]


def request(words: int) -> dict:
    return {"model": "jeff", "state": " ".join(["word"] * words),
            "questions": {"q": {"type": "choice", "instructions": "Which?", "criteria": {"a": None, "b": None}}}}


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(server.service, "model", Recording())
    monkeypatch.setattr(server.service, "max_options", 26)
    return TestClient(server.app)


def test_the_limit_defaults_to_the_trained_length(client: TestClient) -> None:
    assert server.service.max_tokens == 8192
    assert client.post("/v1/systemone", json=request(8192)).status_code == 200
    response = client.post("/v1/systemone", json=request(8193))
    assert response.status_code == 422
    assert "8192-token limit" in response.text


def test_jeff_max_tokens_raises_the_limit(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server.service, "max_tokens", 16384)
    assert client.post("/v1/systemone", json=request(15000)).status_code == 200


def test_the_limit_comes_from_jeff_max_tokens() -> None:
    assert server.max_tokens(None) == 8192
    assert server.max_tokens("32768") == 32768
    for value in ("0", "-1", "abc", "8k", ""):
        with pytest.raises(ValueError, match="JEFF_MAX_TOKENS"):
            server.max_tokens(value)
