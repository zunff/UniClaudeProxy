import asyncio
import json
import unittest

from app.converters.anthropic_to_gemini import to_gemini_request
from app.converters.anthropic_to_openai import to_openai_responses_request
from app.converters.gemini_to_anthropic import (
    decode_gemini_part_signature,
    from_gemini_response,
    stream_gemini_to_anthropic,
)
from app.models import AnthropicRequest


class GeminiRoundTripTests(unittest.TestCase):
    def test_preserves_part_signatures_function_ids_and_parallel_calls(self):
        raw_response = {
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [
                        {"text": "analysis", "thought": True, "thoughtSignature": "thought-sig"},
                        {"text": "visible", "thoughtSignature": "text-sig"},
                        {
                            "functionCall": {
                                "id": "gemini-call-1",
                                "name": "first_tool",
                                "args": {"value": 1},
                            },
                            "thoughtSignature": "call-sig",
                        },
                        {
                            "functionCall": {
                                "id": "gemini-call-2",
                                "name": "second_tool",
                                "args": {"value": 2},
                            },
                        },
                    ],
                },
                "finishReason": "STOP",
            }],
            "usageMetadata": {},
        }

        anthropic = from_gemini_response(raw_response, "claude-sonnet-5")
        thinking = anthropic["content"][0]
        text = anthropic["content"][1]

        self.assertEqual(decode_gemini_part_signature(thinking["signature"]), "thought-sig")
        self.assertEqual(text["text"], "visible")
        text_marker = anthropic["content"][2]
        self.assertEqual(decode_gemini_part_signature(text_marker["signature"]), "text-sig")

        # Account for the signature-only marker inserted after the visible text.
        first_call = anthropic["content"][3]
        second_call = anthropic["content"][4]
        request = AnthropicRequest(
            model="claude-sonnet-5",
            max_tokens=4096,
            messages=[
                {"role": "assistant", "content": anthropic["content"]},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": first_call["id"],
                            "content": "one",
                        },
                        {
                            "type": "tool_result",
                            "tool_use_id": second_call["id"],
                            "content": "two",
                        },
                    ],
                },
            ],
        )

        rebuilt = to_gemini_request(request, "gemini-3.5-flash")
        model_parts = rebuilt["contents"][0]["parts"]
        self.assertEqual(model_parts[0]["thoughtSignature"], "thought-sig")
        self.assertEqual(model_parts[1]["thoughtSignature"], "text-sig")
        self.assertEqual(model_parts[2]["functionCall"]["id"], "gemini-call-1")
        self.assertEqual(model_parts[2]["thoughtSignature"], "call-sig")
        self.assertEqual(model_parts[3]["functionCall"]["id"], "gemini-call-2")
        self.assertNotIn("thoughtSignature", model_parts[3])

        response_parts = rebuilt["contents"][1]["parts"]
        self.assertEqual(response_parts[0]["functionResponse"]["id"], "gemini-call-1")
        self.assertEqual(response_parts[1]["functionResponse"]["id"], "gemini-call-2")

    def test_maps_native_tool_config_top_k_and_json_schema(self):
        request = AnthropicRequest(
            model="claude-sonnet-5",
            max_tokens=100,
            top_k=32,
            tool_choice={"type": "tool", "name": "lookup"},
            messages=[{"role": "user", "content": "test"}],
            tools=[{
                "name": "lookup",
                "description": "Lookup a value",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "value": {
                            "oneOf": [
                                {"type": "string"},
                                {"type": "integer", "minimum": 1},
                            ]
                        }
                    },
                    "required": ["value"],
                    "additionalProperties": False,
                },
            }],
        )

        body = to_gemini_request(request, "gemini-3.5-flash")
        self.assertEqual(body["generationConfig"]["topK"], 32)
        self.assertEqual(
            body["toolConfig"]["functionCallingConfig"],
            {"mode": "ANY", "allowedFunctionNames": ["lookup"]},
        )
        schema = body["tools"][0]["functionDeclarations"][0]["parametersJsonSchema"]
        self.assertIn("anyOf", schema["properties"]["value"])
        self.assertEqual(schema["properties"]["value"]["anyOf"][1]["minimum"], 1)
        self.assertFalse(schema["additionalProperties"])

    def test_stream_emits_signature_delta_for_signed_text(self):
        payload = {
            "candidates": [{
                "content": {
                    "parts": [{"text": "done", "thoughtSignature": "stream-text-sig"}],
                },
                "finishReason": "STOP",
            }],
            "usageMetadata": {},
        }

        async def source():
            yield f"data: {json.dumps(payload)}\n\n".encode()

        async def collect():
            return [event async for event in stream_gemini_to_anthropic(
                source(),
                "claude-sonnet-5",
            )]

        events = asyncio.run(collect())
        signature_events = [event for event in events if "signature_delta" in event]
        self.assertEqual(len(signature_events), 1)
        data_line = next(
            line for line in signature_events[0].splitlines()
            if line.startswith("data: ")
        )
        event_data = json.loads(data_line[6:])
        encoded = event_data["delta"]["signature"]
        self.assertEqual(decode_gemini_part_signature(encoded), "stream-text-sig")

    def test_maps_adaptive_thinking_effort_to_gemini(self):
        request = AnthropicRequest(
            model="claude-sonnet-5",
            messages=[{"role": "user", "content": "test"}],
            thinking={"type": "adaptive"},
            output_config={"effort": "high"},
        )
        body = to_gemini_request(request, "gemini-3.5-flash")
        self.assertEqual(
            body["generationConfig"]["thinkingConfig"],
            {"thinkingLevel": "high", "includeThoughts": True},
        )

    def test_maps_disabled_thinking_by_gemini_capability(self):
        request = AnthropicRequest(
            model="claude-sonnet-5",
            messages=[{"role": "user", "content": "test"}],
            thinking={"type": "disabled"},
        )
        flash = to_gemini_request(request, "gemini-3.5-flash")
        pro = to_gemini_request(request, "gemini-3.1-pro-preview")
        self.assertEqual(
            flash["generationConfig"]["thinkingConfig"]["thinkingLevel"],
            "minimal",
        )
        self.assertEqual(
            pro["generationConfig"]["thinkingConfig"]["thinkingLevel"],
            "low",
        )

    def test_maps_anthropic_effort_to_openai_responses(self):
        request = AnthropicRequest(
            model="claude-opus-4-8",
            messages=[{"role": "user", "content": "test"}],
            thinking={"type": "adaptive"},
            output_config={"effort": "max"},
        )
        body = to_openai_responses_request(request, "gpt-5.5")
        self.assertEqual(body["reasoning"], {"effort": "xhigh"})

        configured = to_openai_responses_request(
            request,
            "gpt-5.5",
            reasoning={"effort": "medium", "summary": "auto"},
        )
        self.assertEqual(
            configured["reasoning"],
            {"effort": "medium", "summary": "auto"},
        )


if __name__ == "__main__":
    unittest.main()
