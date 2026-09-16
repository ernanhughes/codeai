"""OpenCodeCognitionAdapter.prompt(): one prompt in, one CallResult out.

No network, no credentials. http_post is patched with fixtures.
"""

from __future__ import annotations

from codeai.providers import OPENCODE_ZEN_API_KEY_ENV, OpenCodeCognitionAdapter

PARAGRAPH = "The service processes every request within one second."
INSTRUCTION = "Identify factual claims that require evidence."


def chat_fixture(text="The one-second claim requires measurement."):
    return {
        "id": "chatcmpl-zen-1",
        "model": "mimo-v2.5",
        "choices": [{"message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 8},
    }


def test_prompt_sends_instruction_and_text_and_returns_output():
    seen = {}

    def fake_post(url, payload, headers, timeout):
        seen.update(url=url, payload=payload)
        return chat_fixture()

    adapter = OpenCodeCognitionAdapter(
        model="mimo-v2.5", api_key="k", protocol="chat_completions", http_post=fake_post
    )
    result = adapter.prompt(PARAGRAPH, instruction=INSTRUCTION)

    assert result.status == "succeeded"
    assert result.raw_output == "The one-second claim requires measurement."
    assert seen["url"].endswith("/v1/chat/completions")
    content = seen["payload"]["messages"][0]["content"]
    assert content == f"{INSTRUCTION}\n\n{PARAGRAPH}"
    assert result.provider == "opencode"
    assert result.model == "mimo-v2.5"
    assert result.call_id  # generated, not empty
    assert (result.input_tokens, result.output_tokens) == (12, 8)


def test_prompt_without_instruction_sends_text_only():
    seen = {}

    def fake_post(url, payload, headers, timeout):
        seen["payload"] = payload
        return chat_fixture()

    adapter = OpenCodeCognitionAdapter(
        model="mimo-v2.5", api_key="k", protocol="chat_completions", http_post=fake_post
    )
    adapter.prompt(PARAGRAPH)
    assert seen["payload"]["messages"][0]["content"] == PARAGRAPH


def test_prompt_generates_distinct_call_ids():
    adapter = OpenCodeCognitionAdapter(
        model="mimo-v2.5",
        api_key="k",
        protocol="chat_completions",
        http_post=lambda *a: chat_fixture(),
    )
    assert adapter.prompt(PARAGRAPH).call_id != adapter.prompt(PARAGRAPH).call_id


def test_prompt_failure_is_a_status_not_content(monkeypatch):
    monkeypatch.delenv(OPENCODE_ZEN_API_KEY_ENV, raising=False)
    adapter = OpenCodeCognitionAdapter(model="mimo-v2.5", protocol="chat_completions")
    result = adapter.prompt(PARAGRAPH, instruction=INSTRUCTION)

    assert result.status == "failed"
    assert result.raw_output == ""
    assert result.error_kind == "missing_credentials"
    assert OPENCODE_ZEN_API_KEY_ENV in (result.error or "")
