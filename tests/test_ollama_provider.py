from app.services.llm.providers.ollama_provider import OllamaProvider, ollama_openai_base_url


def test_ollama_openai_base_url_accepts_native_host_urls():
    assert ollama_openai_base_url("http://localhost:11434") == "http://localhost:11434/v1"
    assert ollama_openai_base_url("localhost:11434/api") == "http://localhost:11434/v1"
    assert ollama_openai_base_url("http://localhost:11434/v1/") == "http://localhost:11434/v1"


def test_ollama_provider_uses_openai_compatible_chat_base_url():
    provider = OllamaProvider(base_url="http://localhost:11434")

    assert provider.base_url == "http://localhost:11434/v1"
