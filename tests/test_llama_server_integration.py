"""End-to-end coverage for the llama_server loader: chat, tool calling,
reasoning, response_format, GPU offload, embeddings, and concurrency, all
through a real `llama-server` subprocess proxied over its native
OpenAI-compatible HTTP API."""

import concurrent.futures
import json
import time

import httpx
import pytest

OPENAI_API_BASE = "http://localhost:8000/v1"

_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


def _collect_streaming_tool_call(stream) -> dict:
    """Drain an OpenAI streaming response and rebuild the assistant message.

    Returns a dict with: ``content`` (concatenated content deltas),
    ``tool_calls`` (per-index dict of ``{id, name, arguments}`` — arguments
    concatenated across all fragments), ``finish_reason``, ``name_deltas``
    and ``args_deltas`` (counts, used to assert that streaming was actually
    incremental rather than a single buffered emission).
    """
    content_parts: list[str] = []
    tool_calls: dict[int, dict] = {}
    finish_reason: str | None = None
    name_deltas = 0
    args_deltas = 0
    chunks_with_tool_calls = 0

    for chunk in stream:
        choice = chunk.choices[0]
        delta = choice.delta
        if delta.content:
            content_parts.append(delta.content)
        if delta.tool_calls:
            chunks_with_tool_calls += 1
            for tc in delta.tool_calls:
                slot = tool_calls.setdefault(tc.index, {"id": None, "name": None, "arguments": ""})
                if tc.id is not None:
                    slot["id"] = tc.id
                if tc.function and tc.function.name:
                    slot["name"] = tc.function.name
                    name_deltas += 1
                if tc.function and tc.function.arguments:
                    slot["arguments"] += tc.function.arguments
                    args_deltas += 1
        if choice.finish_reason is not None:
            finish_reason = choice.finish_reason

    return {
        "content": "".join(content_parts),
        "tool_calls": tool_calls,
        "finish_reason": finish_reason,
        "name_deltas": name_deltas,
        "args_deltas": args_deltas,
        "chunks_with_tool_calls": chunks_with_tool_calls,
    }


@pytest.mark.integration
@pytest.mark.llama_server
class TestChatLlamaServer:
    """End-to-end chat, tool calling, reasoning, and concurrency through the
    `llama_server` loader (a `llama-server` subprocess proxied over its
    native OpenAI-compatible HTTP API). Reasoning and tool-call parsing is
    llama-server's own (`--jinja --reasoning-format auto`).
    """

    @pytest.fixture(autouse=True, scope="class")
    def _deploy(self, model_deployer):
        model_deployer.deploy("chat-llama-server")

    def test_chat_completion(self, client):
        # This deployment is Qwen3-0.6B (reasoning-capable), unlike
        # `chat-llama-server-plain`'s plain Qwen2.5 — it always emits a
        # `<think>...` preamble before content, so the token budget needs
        # headroom for reasoning to finish, not just the answer itself.
        completion = client.chat.completions.create(
            model="chat-llama-server",
            messages=[{"role": "user", "content": "What is the capital of France?"}],
            max_tokens=256,
        )
        content = completion.choices[0].message.content
        assert content
        assert "Paris" in content

    def test_tool_calling_llama_server_loader(self, client):
        """Round-trip a tool call through llama-server's own hermes-style
        parser (`--jinja`, auto-detected from the GGUF's chat template)."""
        completion = client.chat.completions.create(
            model="chat-llama-server",
            messages=[{"role": "user", "content": "What is the weather in Paris?"}],
            tools=[_WEATHER_TOOL],
            tool_choice="auto",
            max_tokens=128,
        )
        tool_calls = completion.choices[0].message.tool_calls
        assert tool_calls, f"expected a tool call, got content={completion.choices[0].message.content!r}"
        assert tool_calls[0].function.name == "get_weather"
        assert "Paris" in tool_calls[0].function.arguments
        assert completion.choices[0].finish_reason == "tool_calls"

    def test_tool_calling_streaming_llama_server_loader(self, client):
        """Stream a tool call through llama-server and verify the delta
        sequence matches the OpenAI streaming contract, same shape as the
        vLLM loader streaming tests."""
        stream = client.chat.completions.create(
            model="chat-llama-server",
            messages=[{"role": "user", "content": "What is the weather in Paris?"}],
            tools=[_WEATHER_TOOL],
            tool_choice="auto",
            max_tokens=128,
            stream=True,
        )

        collected = _collect_streaming_tool_call(stream)

        assert collected["tool_calls"], (
            f"expected at least one streamed tool call; got content={collected['content']!r}"
        )
        call_0 = collected["tool_calls"][0]
        assert call_0["name"] == "get_weather"
        assert collected["args_deltas"] >= 1
        parsed_args = json.loads(call_0["arguments"])
        assert parsed_args.get("city")
        assert "Paris" in parsed_args["city"]
        assert collected["finish_reason"] == "tool_calls"

    def test_reasoning_completion_llama_server(self):
        """Non-streaming: llama-server's own `--reasoning-format auto` routes
        the `<think>...</think>` block to `message.reasoning`, with the final
        answer in `message.content` and no marker leakage into either."""
        response = httpx.post(
            f"{OPENAI_API_BASE}/chat/completions",
            json={
                "model": "chat-llama-server",
                "messages": [{"role": "user", "content": "Briefly: what is 7 times 8?"}],
                "max_tokens": 1024,
            },
            timeout=300,
        )
        assert response.status_code == 200, response.text
        message = response.json()["choices"][0]["message"]
        assert message.get("reasoning"), f"expected reasoning content, got {message!r}"
        assert "<think>" not in (message.get("content") or "")
        assert "</think>" not in (message.get("content") or "")
        assert "<think>" not in message["reasoning"]
        assert "</think>" not in message["reasoning"]

    def test_reasoning_streaming_llama_server(self):
        """Streaming: at least one delta carries `reasoning`; concatenated
        reasoning is non-empty; markers never leak into either field."""
        with httpx.stream(
            "POST",
            f"{OPENAI_API_BASE}/chat/completions",
            json={
                "model": "chat-llama-server",
                "messages": [{"role": "user", "content": "Briefly: what is 7 times 8?"}],
                "max_tokens": 1024,
                "stream": True,
            },
            timeout=300,
        ) as response:
            assert response.status_code == 200
            reasoning_parts: list[str] = []
            content_parts: list[str] = []
            reasoning_deltas = 0
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[len("data: ") :]
                if payload == "[DONE]":
                    break
                chunk = json.loads(payload)
                delta = chunk["choices"][0].get("delta") or {}
                if delta.get("reasoning"):
                    reasoning_parts.append(delta["reasoning"])
                    reasoning_deltas += 1
                if delta.get("content"):
                    content_parts.append(delta["content"])

        assert reasoning_deltas >= 1, "expected at least one reasoning delta"
        assert "".join(reasoning_parts).strip(), "expected non-empty reasoning content"
        assert "<think>" not in "".join(reasoning_parts)
        assert "</think>" not in "".join(reasoning_parts)
        assert "<think>" not in "".join(content_parts)
        assert "</think>" not in "".join(content_parts)

    def test_reasoning_with_tools_llama_server(self, client):
        """Reasoning + tool calling in one round-trip: llama-server populates
        both `message.reasoning` and `message.tool_calls`, with
        `finish_reason="tool_calls"`."""
        completion = client.chat.completions.create(
            model="chat-llama-server",
            messages=[{"role": "user", "content": "What is the weather in Paris?"}],
            tools=[_WEATHER_TOOL],
            tool_choice="auto",
            max_tokens=1024,
        )
        message = completion.choices[0].message
        # The OpenAI Python SDK exposes unknown fields via `model_extra`.
        reasoning = getattr(message, "reasoning", None) or message.model_extra.get("reasoning")
        assert reasoning, f"expected reasoning, got message={message!r}"
        assert "<think>" not in reasoning
        tool_calls = message.tool_calls
        assert tool_calls, f"expected a tool call, got content={message.content!r}, reasoning={reasoning!r}"
        assert tool_calls[0].function.name == "get_weather"
        assert "Paris" in tool_calls[0].function.arguments
        assert completion.choices[0].finish_reason == "tool_calls"

    def test_tool_markers_inside_reasoning_not_double_counted_llama_server(self, client):
        """Verifies llama-server's own parser doesn't double-count a
        `<tool_call>...</tool_call>` illustration quoted inside `<think>`
        reasoning as a second, real call — a bug pattern plausible for any
        single-pass parser.

        Coaxes the model into illustrating tool-call syntax inside its
        reasoning before making one actual call, and asserts exactly one real
        `tool_calls` entry comes out. Real models are non-deterministic; if
        the prompt fails to produce literal markers in reasoning, the
        marker-routing assertion is skipped rather than flaking — the
        single-tool-call assertion still has value either way.
        """
        completion = client.chat.completions.create(
            model="chat-llama-server",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an assistant with access to tools. When you think inside "
                        "<think>...</think>, FIRST quote one example of tool-call syntax "
                        "verbatim inside angle brackets — e.g. write the literal text "
                        '<tool_call>{"name":"example","arguments":{}}</tool_call> as part '
                        "of your reasoning to remind yourself of the format. THEN decide "
                        "which real tool to call."
                    ),
                },
                {"role": "user", "content": "What is the weather in Paris?"},
            ],
            tools=[_WEATHER_TOOL],
            tool_choice="auto",
            max_tokens=1024,
        )
        message = completion.choices[0].message
        reasoning = getattr(message, "reasoning", None) or message.model_extra.get("reasoning") or ""
        tool_calls = message.tool_calls or []

        assert tool_calls, (
            f"expected exactly one real tool call, got content={message.content!r}, reasoning={reasoning!r}"
        )
        assert len(tool_calls) == 1, (
            f"expected exactly one tool call (markers inside <think> must not be double-counted); "
            f"got {len(tool_calls)} calls={[tc.function.name for tc in tool_calls]}"
        )
        assert tool_calls[0].function.name == "get_weather"
        assert completion.choices[0].finish_reason == "tool_calls"

        if "<tool_call>" in reasoning:
            assert "</tool_call>" in reasoning, (
                f"reasoning has an unmatched <tool_call> marker (open without close): {reasoning!r}"
            )

    def test_response_format_with_reasoning_llama_server(self, client):
        """llama-server handles a JSON-schema `response_format` combined with
        reasoning natively, routing `<think>...</think>` to `message.reasoning`
        and the schema-conforming JSON to `message.content`."""
        completion = client.chat.completions.create(
            model="chat-llama-server",
            messages=[{"role": "user", "content": "What is 2+2?"}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "schema": {
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                    },
                    "strict": True,
                },
            },
            max_tokens=1024,
        )
        message = completion.choices[0].message
        reasoning = getattr(message, "reasoning", None) or message.model_extra.get("reasoning")
        assert reasoning, f"expected reasoning, got message={message!r}"
        assert message.content
        parsed = json.loads(message.content)
        assert "answer" in parsed

    def test_tool_choice_required_is_a_noop_on_hermes_family(self, client):
        """Documents a real gap (A1 spike finding): `tool_choice: required` is
        grammar-enforced on harmony-style chat templates (e.g. gpt-oss) but a
        silent no-op on hermes-style ones — Qwen3 (this deployment) included.
        Real grammar forcing makes the free-text branch structurally
        unreachable (`message.content` would be empty/None); on this
        hermes-style model it stays reachable even under `required`, proving
        no grammar constraint was applied. (Verified against a live run: this
        0.6B model is unstable enough to *also* emit a spurious tool call
        alongside genuine free text on an irrelevant prompt — which is a
        model-quality quirk, not evidence of forcing, so this asserts on
        content reachability rather than tool_calls absence.) If llama.cpp
        starts enforcing this for hermes models, `content` goes empty and
        this test fails — update the docs/CLAUDE.md gap notes.
        """
        completion = client.chat.completions.create(
            model="chat-llama-server",
            messages=[{"role": "user", "content": "Say hello in one word."}],
            tools=[_WEATHER_TOOL],
            tool_choice="required",
            max_tokens=512,  # headroom for the reasoning preamble to not crowd out the answer
        )
        message = completion.choices[0].message
        assert message.content, f"expected the free-text branch to stay reachable, got message={message!r}"

    def test_named_function_tool_choice_falls_back_to_auto(self, client):
        """Documents a real gap (A1 spike finding): object-form `tool_choice`
        (named-function forcing) is globally unsupported in llama.cpp b9859
        (confirmed unchanged on the current b10200 pin) — it silently falls
        back to `auto` rather than forcing the named function or erroring.
        Same content-reachability technique and
        irrelevant-tool prompt as the `required` gap test above — real
        forcing would make the free-text branch structurally unreachable."""
        completion = client.chat.completions.create(
            model="chat-llama-server",
            messages=[{"role": "user", "content": "Say hello in one word."}],
            tools=[_WEATHER_TOOL],
            tool_choice={"type": "function", "function": {"name": "get_weather"}},
            max_tokens=512,  # headroom for the reasoning preamble to not crowd out the answer
        )
        message = completion.choices[0].message
        assert message.content, f"expected the free-text branch to stay reachable, got message={message!r}"

    def test_concurrent_requests_are_not_serialized(self, client):
        """The loader's headline capability: llama-server's `--parallel` slots
        let several requests run concurrently instead of being serialized
        behind a single lock. Time one request, then several at once, and
        assert the concurrent batch finishes well under what full
        serialization would take.
        """
        prompt = {
            "model": "chat-llama-server",
            "messages": [{"role": "user", "content": "Count from 1 to 50, one number per line."}],
            "max_tokens": 200,
        }

        start = time.monotonic()
        client.chat.completions.create(**prompt)
        baseline = time.monotonic() - start

        concurrency = 3
        start = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(client.chat.completions.create, **prompt) for _ in range(concurrency)]
            for future in concurrent.futures.as_completed(futures):
                future.result()
        concurrent_elapsed = time.monotonic() - start

        # Full serialization (a single-lock loader's behavior) would take
        # roughly concurrency * baseline; llama-server's parallel slots should
        # keep this well under that.
        assert concurrent_elapsed < baseline * (concurrency - 0.5), (
            f"expected concurrent requests to overlap via llama-server's parallel slots "
            f"(baseline={baseline:.1f}s, {concurrency} concurrent took {concurrent_elapsed:.1f}s)"
        )


@pytest.mark.integration
@pytest.mark.llama_server
class TestChatLlamaServerResponseFormat:
    """response_format tests for the llama_server loader. Uses
    `chat-llama-server-plain` (non-reasoning Qwen2.5-0.5B) rather than
    `chat-llama-server` (Qwen3, always emits a `<think>...` preamble) —
    response_format + reasoning together is covered separately by
    `TestChatLlamaServer.test_response_format_with_reasoning_llama_server`.
    """

    @pytest.fixture(autouse=True, scope="class")
    def _deploy(self, model_deployer):
        model_deployer.deploy("chat-llama-server-plain")

    def test_response_format_json_object_without_schema_is_unconstrained(self, client):
        """llama-server's own docs claim bare `{"type": "json_object"}` (no `schema` key) produces
        "plain JSON output" like other OpenAI-inspired providers, but verified
        directly against the b9859 binary (`curl` straight to `/v1/chat/completions`,
        bypassing modelship; confirmed unchanged on the current b10200 pin)
        this isn't enforced — the model answers in free
        text with no error. Constraining does work once a `schema` key is
        attached to the `response_format` object (an llama-server extension,
        not in the OpenAI spec — `type: json_schema`, which modelship's
        protocol sends for schema-constrained requests, does carry a schema
        and IS honored — see `test_response_format_json_schema_constrains_unprompted_output`).
        If this test starts failing (content parses as JSON), llama-server
        started honoring plain `json_object` — update this note and
        CLAUDE.md/AGENTS.md's llama_server gap list.
        """
        completion = client.chat.completions.create(
            model="chat-llama-server-plain",
            messages=[{"role": "user", "content": "What is the capital of France?"}],
            response_format={"type": "json_object"},
            max_tokens=64,
        )
        content = completion.choices[0].message.content
        assert content
        with pytest.raises(json.JSONDecodeError):
            json.loads(content)

    def test_response_format_json_schema_constrains_unprompted_output(self, client):
        """A natural-language question + json_schema → schema-conformant output."""
        schema = {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "country": {"type": "string"},
            },
            "required": ["city", "country"],
            "additionalProperties": False,
        }
        completion = client.chat.completions.create(
            model="chat-llama-server-plain",
            messages=[{"role": "user", "content": "Where is the Eiffel Tower located?"}],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "location", "schema": schema, "strict": True},
            },
            max_tokens=64,
        )
        content = completion.choices[0].message.content
        assert content
        parsed = json.loads(content)
        assert set(parsed.keys()) == {"city", "country"}
        assert isinstance(parsed["city"], str) and parsed["city"]
        assert isinstance(parsed["country"], str) and parsed["country"]

    def test_response_format_json_schema_streaming_constrains_unprompted_output(self, client):
        """Same intent on the streaming path."""
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        }
        stream = client.chat.completions.create(
            model="chat-llama-server-plain",
            messages=[{"role": "user", "content": "What is the capital of France?"}],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "answer", "schema": schema, "strict": True},
            },
            max_tokens=64,
            stream=True,
        )
        chunks = []
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                chunks.append(chunk.choices[0].delta.content)
        content = "".join(chunks)
        assert content
        parsed = json.loads(content)
        assert set(parsed.keys()) == {"answer"}
        assert isinstance(parsed["answer"], str) and parsed["answer"]

    def test_response_format_coexists_with_tool_choice_none(self, client):
        """tool_choice='none' is the safe escape valve: tools listed but inert,
        schema enforced on content output.
        """
        schema = {
            "type": "object",
            "properties": {"city": {"type": "string"}, "country": {"type": "string"}},
            "required": ["city", "country"],
            "additionalProperties": False,
        }
        completion = client.chat.completions.create(
            model="chat-llama-server-plain",
            messages=[{"role": "user", "content": "Where is the Eiffel Tower located?"}],
            tools=[_WEATHER_TOOL],
            tool_choice="none",
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "location", "schema": schema, "strict": True},
            },
            max_tokens=64,
        )
        assert not completion.choices[0].message.tool_calls
        content = completion.choices[0].message.content
        assert content
        parsed = json.loads(content)
        assert set(parsed.keys()) == {"city", "country"}


@pytest.mark.integration
@pytest.mark.llama_server
class TestChatLlamaServerGpu:
    """End-to-end GPU offload through the llama_server loader.

    Same GGUF and tool-calling shape as `TestChatLlamaServerResponseFormat`
    (CPU), but deployed with `num_gpus=1` so the actor gets a whole GPU and
    the loader passes `-ngl` for real offload instead of the forced `-ngl 0`
    it uses when `num_gpus` is `0`.
    """

    @pytest.fixture(autouse=True, scope="class")
    def _deploy(self, model_deployer):
        model_deployer.deploy("chat-llama-server-gpu")

    def test_chat_completion(self, client):
        completion = client.chat.completions.create(
            model="chat-llama-server-gpu",
            messages=[{"role": "user", "content": "What is the capital of France?"}],
            max_tokens=32,
        )
        content = completion.choices[0].message.content
        assert content
        assert "Paris" in content

    def test_tool_calling_llama_server_gpu_loader(self, client):
        completion = client.chat.completions.create(
            model="chat-llama-server-gpu",
            messages=[{"role": "user", "content": "What is the weather in Paris?"}],
            tools=[_WEATHER_TOOL],
            tool_choice="auto",
            max_tokens=128,
        )
        tool_calls = completion.choices[0].message.tool_calls
        assert tool_calls, f"expected a tool call, got content={completion.choices[0].message.content!r}"
        assert tool_calls[0].function.name == "get_weather"
        assert "Paris" in tool_calls[0].function.arguments
        assert completion.choices[0].finish_reason == "tool_calls"


@pytest.mark.integration
@pytest.mark.llama_server
def test_embeddings_llama_server(client, model_deployer):
    """Real embeddings through a live llama-server subprocess (`--embedding`).
    `test_embeddings` only exercises the vllm loader; this is the
    first live-binary coverage of llama_server's B4 embeddings support
    (previously only unit-tested against a mocked httpx transport)."""
    model_deployer.deploy("embed-model-llama-server")
    response = client.embeddings.create(model="embed-model-llama-server", input=["Hello world", "Modelship is great"])
    assert len(response.data) == 2
    assert len(response.data[0].embedding) > 0
