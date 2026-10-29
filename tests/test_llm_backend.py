import pytest

from pumpking.exceptions import LLMBackendError
from pumpking.models import ChunkPayload
from pumpking.strategies.providers import (
    DEFAULT_BASE_URL,
    LLMBackend,
    LLMSummaryProvider,
)

UNREACHABLE = "http://127.0.0.1:9/v1"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def test_base_url_defaults_to_openai():
    backend = LLMBackend(api_key="test")
    assert backend.base_url == DEFAULT_BASE_URL
    assert str(backend.client.base_url).rstrip("/") == DEFAULT_BASE_URL


def test_base_url_reads_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
    backend = LLMBackend(api_key="test")
    assert backend.base_url == "http://localhost:11434/v1"


def test_explicit_base_url_wins_over_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
    backend = LLMBackend(base_url="http://example.test/v1", api_key="test")
    assert backend.base_url == "http://example.test/v1"


def test_timeout_and_retries_reach_the_client():
    backend = LLMBackend(api_key="test", timeout=5.0, max_retries=0)
    assert backend.client.timeout == 5.0
    assert backend.client.max_retries == 0


def test_api_key_reads_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    assert LLMBackend().client is not None


def test_repr_does_not_show_the_key():
    backend = LLMBackend(api_key="secret-value")
    assert "secret-value" not in repr(backend)


def test_missing_key_raises_clear_error():
    backend = LLMBackend()
    with pytest.raises(LLMBackendError, match="OPENAI_API_KEY"):
        backend.create_completion(messages=[], response_format=None)


def test_unreachable_endpoint_names_the_url():
    backend = LLMBackend(base_url=UNREACHABLE, api_key="test", timeout=2.0, max_retries=0)
    with pytest.raises(LLMBackendError, match="127.0.0.1:9"):
        backend.create_chat_completion(messages=[{"role": "user", "content": "hi"}])


def test_providers_do_not_swallow_backend_errors():
    backend = LLMBackend(base_url=UNREACHABLE, api_key="test", timeout=2.0, max_retries=0)
    provider = LLMSummaryProvider(backend=backend)
    with pytest.raises(LLMBackendError):
        provider.summarize([ChunkPayload(content="Some text to summarize.")])
