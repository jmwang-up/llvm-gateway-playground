from app.schemas.chat import ChatRequest
from app.services.cache import build_cache_key


def test_cache_key_is_stable_and_client_isolated():
    request_a = ChatRequest.model_validate(
        {
            "model": "auto",
            "messages": [{"content": "hi", "role": "user"}],
            "temperature": 0.7,
            "max_tokens": 20,
        }
    )
    request_b = ChatRequest.model_validate_json(request_a.model_dump_json())

    assert build_cache_key("frontend", request_a) == build_cache_key("frontend", request_b)
    assert build_cache_key("frontend", request_a) != build_cache_key("worker", request_a)


def test_auto_and_explicit_model_do_not_share_cache():
    auto = ChatRequest(messages=[{"role": "user", "content": "hi"}])
    explicit = auto.model_copy(update={"model": "deepseek/deepseek-chat"})

    assert build_cache_key("frontend", auto) != build_cache_key("frontend", explicit)

