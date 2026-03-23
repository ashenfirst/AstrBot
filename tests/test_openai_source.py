from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from openai.types.chat.chat_completion import ChatCompletion

from astrbot.core.provider.sources.groq_source import ProviderGroq
from astrbot.core.provider.sources.openai_source import ProviderOpenAIOfficial


class _ErrorWithBody(Exception):
    def __init__(self, message: str, body: dict):
        super().__init__(message)
        self.body = body


class _ErrorWithResponse(Exception):
    def __init__(self, message: str, response_text: str):
        super().__init__(message)
        self.response = SimpleNamespace(text=response_text)


def _make_provider(overrides: dict | None = None) -> ProviderOpenAIOfficial:
    provider_config = {
        "id": "test-openai",
        "type": "openai_chat_completion",
        "model": "gpt-4o-mini",
        "key": ["test-key"],
    }
    if overrides:
        provider_config.update(overrides)
    return ProviderOpenAIOfficial(
        provider_config=provider_config,
        provider_settings={},
    )


def _make_groq_provider(overrides: dict | None = None) -> ProviderGroq:
    provider_config = {
        "id": "test-groq",
        "type": "groq_chat_completion",
        "model": "qwen/qwen3-32b",
        "key": ["test-key"],
    }
    if overrides:
        provider_config.update(overrides)
    return ProviderGroq(
        provider_config=provider_config,
        provider_settings={},
    )


@pytest.mark.asyncio
async def test_handle_api_error_content_moderated_removes_images():
    provider = _make_provider(
        {"image_moderation_error_patterns": ["file:content-moderated"]}
    )
    try:
        payloads = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hello"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/jpeg;base64,abcd"},
                        },
                    ],
                }
            ]
        }
        context_query = payloads["messages"]

        success, *_rest = await provider._handle_api_error(
            Exception("Content is moderated [WKE=file:content-moderated]"),
            payloads=payloads,
            context_query=context_query,
            func_tool=None,
            chosen_key="test-key",
            available_api_keys=["test-key"],
            retry_cnt=0,
            max_retries=10,
        )

        assert success is False
        updated_context = payloads["messages"]
        assert isinstance(updated_context, list)
        assert updated_context[0]["content"] == [{"type": "text", "text": "hello"}]
    finally:
        await provider.terminate()


@pytest.mark.asyncio
async def test_handle_api_error_model_not_vlm_removes_images_and_retries_text_only():
    provider = _make_provider()
    try:
        payloads = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hello"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/jpeg;base64,abcd"},
                        },
                    ],
                }
            ]
        }
        context_query = payloads["messages"]

        success, *_rest = await provider._handle_api_error(
            Exception("The model is not a VLM and cannot process images"),
            payloads=payloads,
            context_query=context_query,
            func_tool=None,
            chosen_key="test-key",
            available_api_keys=["test-key"],
            retry_cnt=0,
            max_retries=10,
        )

        assert success is False
        updated_context = payloads["messages"]
        assert isinstance(updated_context, list)
        assert updated_context[0]["content"] == [{"type": "text", "text": "hello"}]
    finally:
        await provider.terminate()


@pytest.mark.asyncio
async def test_handle_api_error_model_not_vlm_after_fallback_raises():
    provider = _make_provider()
    try:
        payloads = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hello"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/jpeg;base64,abcd"},
                        },
                    ],
                }
            ]
        }
        context_query = payloads["messages"]

        with pytest.raises(Exception, match="not a VLM"):
            await provider._handle_api_error(
                Exception("The model is not a VLM and cannot process images"),
                payloads=payloads,
                context_query=context_query,
                func_tool=None,
                chosen_key="test-key",
                available_api_keys=["test-key"],
                retry_cnt=1,
                max_retries=10,
                image_fallback_used=True,
            )
    finally:
        await provider.terminate()


@pytest.mark.asyncio
async def test_handle_api_error_content_moderated_with_unserializable_body():
    provider = _make_provider({"image_moderation_error_patterns": ["blocked"]})
    try:
        payloads = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hello"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/jpeg;base64,abcd"},
                        },
                    ],
                }
            ]
        }
        context_query = payloads["messages"]
        err = _ErrorWithBody(
            "upstream error",
            {"error": {"message": "blocked"}, "raw": object()},
        )

        success, *_rest = await provider._handle_api_error(
            err,
            payloads=payloads,
            context_query=context_query,
            func_tool=None,
            chosen_key="test-key",
            available_api_keys=["test-key"],
            retry_cnt=0,
            max_retries=10,
        )
        assert success is False
        assert payloads["messages"][0]["content"] == [{"type": "text", "text": "hello"}]
    finally:
        await provider.terminate()


def test_extract_error_text_candidates_truncates_long_response_text():
    long_text = "x" * 20000
    err = _ErrorWithResponse("upstream error", long_text)
    candidates = ProviderOpenAIOfficial._extract_error_text_candidates(err)
    assert candidates
    assert max(len(candidate) for candidate in candidates) <= (
        ProviderOpenAIOfficial._ERROR_TEXT_CANDIDATE_MAX_CHARS
    )


@pytest.mark.asyncio
async def test_openai_payload_keeps_reasoning_content_in_assistant_history():
    provider = _make_provider()
    try:
        payloads = {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "think", "think": "step 1"},
                        {"type": "text", "text": "final answer"},
                    ],
                }
            ]
        }

        provider._finally_convert_payload(payloads)

        assistant_message = payloads["messages"][0]
        assert assistant_message["content"] == [
            {"type": "text", "text": "final answer"}
        ]
        assert assistant_message["reasoning_content"] == "step 1"
    finally:
        await provider.terminate()


@pytest.mark.asyncio
async def test_groq_payload_drops_reasoning_content_from_assistant_history():
    provider = _make_groq_provider()
    try:
        payloads = {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "think", "think": "step 1"},
                        {"type": "text", "text": "final answer"},
                    ],
                }
            ]
        }

        provider._finally_convert_payload(payloads)

        assistant_message = payloads["messages"][0]
        assert assistant_message["content"] == [
            {"type": "text", "text": "final answer"}
        ]
        assert "reasoning_content" not in assistant_message
        assert "reasoning" not in assistant_message
    finally:
        await provider.terminate()


@pytest.mark.asyncio
async def test_handle_api_error_content_moderated_without_images_raises():
    provider = _make_provider(
        {"image_moderation_error_patterns": ["file:content-moderated"]}
    )
    try:
        payloads = {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "hello"}],
                }
            ]
        }
        context_query = payloads["messages"]
        err = Exception("Content is moderated [WKE=file:content-moderated]")

        with pytest.raises(Exception, match="content-moderated"):
            await provider._handle_api_error(
                err,
                payloads=payloads,
                context_query=context_query,
                func_tool=None,
                chosen_key="test-key",
                available_api_keys=["test-key"],
                retry_cnt=0,
                max_retries=10,
            )
    finally:
        await provider.terminate()


@pytest.mark.asyncio
async def test_handle_api_error_content_moderated_detects_structured_body():
    provider = _make_provider(
        {"image_moderation_error_patterns": ["content_moderated"]}
    )
    try:
        payloads = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hello"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/jpeg;base64,abcd"},
                        },
                    ],
                }
            ]
        }
        context_query = payloads["messages"]
        err = _ErrorWithBody(
            "upstream error",
            {"error": {"code": "content_moderated", "message": "blocked"}},
        )

        success, *_rest = await provider._handle_api_error(
            err,
            payloads=payloads,
            context_query=context_query,
            func_tool=None,
            chosen_key="test-key",
            available_api_keys=["test-key"],
            retry_cnt=0,
            max_retries=10,
        )
        assert success is False
        assert payloads["messages"][0]["content"] == [{"type": "text", "text": "hello"}]
    finally:
        await provider.terminate()


@pytest.mark.asyncio
async def test_handle_api_error_content_moderated_supports_custom_patterns():
    provider = _make_provider(
        {"image_moderation_error_patterns": ["blocked_by_policy_code_123"]}
    )
    try:
        payloads = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hello"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/jpeg;base64,abcd"},
                        },
                    ],
                }
            ]
        }
        context_query = payloads["messages"]
        err = Exception("upstream: blocked_by_policy_code_123")

        success, *_rest = await provider._handle_api_error(
            err,
            payloads=payloads,
            context_query=context_query,
            func_tool=None,
            chosen_key="test-key",
            available_api_keys=["test-key"],
            retry_cnt=0,
            max_retries=10,
        )
        assert success is False
        assert payloads["messages"][0]["content"] == [{"type": "text", "text": "hello"}]
    finally:
        await provider.terminate()


@pytest.mark.asyncio
async def test_handle_api_error_content_moderated_without_patterns_raises():
    provider = _make_provider()
    try:
        payloads = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hello"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/jpeg;base64,abcd"},
                        },
                    ],
                }
            ]
        }
        context_query = payloads["messages"]
        err = Exception("Content is moderated [WKE=file:content-moderated]")

        with pytest.raises(Exception, match="content-moderated"):
            await provider._handle_api_error(
                err,
                payloads=payloads,
                context_query=context_query,
                func_tool=None,
                chosen_key="test-key",
                available_api_keys=["test-key"],
                retry_cnt=0,
                max_retries=10,
            )
    finally:
        await provider.terminate()


@pytest.mark.asyncio
async def test_handle_api_error_unknown_image_error_raises():
    provider = _make_provider()
    try:
        payloads = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hello"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/jpeg;base64,abcd"},
                        },
                    ],
                }
            ]
        }
        context_query = payloads["messages"]

        with pytest.raises(Exception, match="unknown provider image upload error"):
            await provider._handle_api_error(
                Exception("some unknown provider image upload error"),
                payloads=payloads,
                context_query=context_query,
                func_tool=None,
                chosen_key="test-key",
                available_api_keys=["test-key"],
                retry_cnt=0,
                max_retries=10,
            )
    finally:
        await provider.terminate()


@pytest.mark.asyncio
async def test_apply_provider_specific_extra_body_overrides_disables_ollama_thinking():
    provider = _make_provider(
        {
            "provider": "ollama",
            "ollama_disable_thinking": True,
        }
    )
    try:
        extra_body = {
            "reasoning": {"effort": "high"},
            "reasoning_effort": "low",
            "think": True,
            "temperature": 0.2,
        }

        provider._apply_provider_specific_extra_body_overrides(extra_body)

        assert extra_body["reasoning_effort"] == "none"
        assert "reasoning" not in extra_body
        assert "think" not in extra_body
        assert extra_body["temperature"] == 0.2
    finally:
        await provider.terminate()


@pytest.mark.asyncio
async def test_query_injects_reasoning_effort_none_for_ollama(monkeypatch):
    provider = _make_provider(
        {
            "provider": "ollama",
            "ollama_disable_thinking": True,
            "custom_extra_body": {
                "reasoning": {"effort": "high"},
                "temperature": 0.1,
            },
        }
    )
    try:
        captured_kwargs = {}

        async def fake_create(**kwargs):
            captured_kwargs.update(kwargs)
            return ChatCompletion.model_validate(
                {
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "qwen3.5:4b",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "ok",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                }
            )

        monkeypatch.setattr(provider.client.chat.completions, "create", fake_create)

        await provider._query(
            payloads={
                "model": "qwen3.5:4b",
                "messages": [{"role": "user", "content": "hello"}],
            },
            tools=None,
        )

        extra_body = captured_kwargs["extra_body"]
        assert extra_body["reasoning_effort"] == "none"
        assert "reasoning" not in extra_body
        assert extra_body["temperature"] == 0.1
    finally:
        await provider.terminate()


@pytest.mark.asyncio
async def test_text_chat_uses_streaming_when_force_stream_enabled():
    provider = _make_provider({"force_stream_for_chat_completions": True})
    try:
        expected_response = SimpleNamespace(completion_text="streamed response")
        provider._prepare_chat_payload = AsyncMock(
            return_value=({"messages": [{"role": "user", "content": "hello"}]}, [])
        )

        async def fake_collect_streaming_query(payloads, func_tool):
            assert payloads == {"messages": [{"role": "user", "content": "hello"}]}
            assert func_tool is None
            return expected_response

        provider._collect_streaming_query = AsyncMock(
            side_effect=fake_collect_streaming_query
        )
        provider._query = AsyncMock(
            side_effect=AssertionError("_query should not be called")
        )

        response = await provider.text_chat(prompt="hello")

        assert response is expected_response
        provider._collect_streaming_query.assert_awaited_once()
        provider._query.assert_not_awaited()
    finally:
        await provider.terminate()


@pytest.mark.asyncio
async def test_text_chat_falls_back_to_streaming_when_provider_requires_it():
    provider = _make_provider()
    try:
        expected_response = SimpleNamespace(completion_text="stream fallback response")
        provider._prepare_chat_payload = AsyncMock(
            return_value=({"messages": [{"role": "user", "content": "hello"}]}, [])
        )
        provider._query = AsyncMock(
            side_effect=[
                Exception("Error code: 400 - {'detail': 'Stream must be set to true'}")
            ]
        )
        provider._collect_streaming_query = AsyncMock(return_value=expected_response)

        response = await provider.text_chat(prompt="hello")

        assert response is expected_response
        provider._query.assert_awaited_once()
        provider._collect_streaming_query.assert_awaited_once()
    finally:
        await provider.terminate()


@pytest.mark.asyncio
async def test_requires_streaming_chat_completion_detects_provider_error():
    provider = _make_provider()
    try:
        assert provider._requires_streaming_chat_completion(
            Exception("Error code: 400 - {'detail': 'Stream must be set to true'}")
        )
        assert not provider._requires_streaming_chat_completion(
            Exception("some other provider error")
        )
    finally:
        await provider.terminate()


@pytest.mark.asyncio
async def test_query_stream_falls_back_when_final_completion_snapshot_missing():
    provider = _make_provider()
    try:
        delta = SimpleNamespace(content="hello ", tool_calls=None)
        chunk = SimpleNamespace(
            id="chunk-1",
            choices=[SimpleNamespace(delta=delta)],
            usage=None,
        )

        class _BrokenState:
            def handle_chunk(self, _chunk):
                return None

            def get_final_completion(self):
                raise AssertionError("snapshot missing")

        async def fake_stream():
            yield chunk

        provider.client.chat.completions.create = AsyncMock(return_value=fake_stream())
        provider._parse_openai_completion = AsyncMock(
            side_effect=AssertionError("should not parse final completion")
        )

        original_state = ProviderOpenAIOfficial._query_stream.__globals__[
            "ChatCompletionStreamState"
        ]
        ProviderOpenAIOfficial._query_stream.__globals__[
            "ChatCompletionStreamState"
        ] = _BrokenState
        try:
            responses = [
                response
                async for response in provider._query_stream(
                    {"messages": [{"role": "user", "content": "hello"}]},
                    None,
                )
            ]
        finally:
            ProviderOpenAIOfficial._query_stream.__globals__[
                "ChatCompletionStreamState"
            ] = original_state

        assert len(responses) == 2
        assert responses[0].is_chunk is True
        assert responses[0].completion_text == "hello "
        assert responses[1].is_chunk is False
        assert responses[1].completion_text == "hello "
        provider._parse_openai_completion.assert_not_awaited()
    finally:
        await provider.terminate()


def test_build_fallback_stream_response_preserves_tool_calls():
    response = ProviderOpenAIOfficial._build_fallback_stream_response(
        response_id="resp-1",
        completion_text="",
        reasoning_content="",
        tool_call_state={
            "call-1": {
                "id": "call-1",
                "name": "astrbot_execute_shell",
                "arguments": '{"command":"cmd /c echo astrbot_shell_ok"}',
                "extra_content": None,
                "order": 0,
            }
        },
        usage=None,
    )

    assert response.role == "tool"
    assert response.tools_call_name == ["astrbot_execute_shell"]
    assert response.tools_call_ids == ["call-1"]
    assert response.tools_call_args == [{"command": "cmd /c echo astrbot_shell_ok"}]
