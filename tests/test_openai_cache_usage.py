import asyncio
import json
import unittest

from app.converters.openai_to_anthropic import (
    _extract_cache_read_tokens,
    from_openai_chat_response,
    stream_openai_chat_to_anthropic,
)


class CacheReadExtractionTests(unittest.TestCase):
    def test_deepseek_hit_field(self):
        self.assertEqual(
            _extract_cache_read_tokens({
                "prompt_tokens": 100,
                "prompt_cache_hit_tokens": 80,
                "prompt_cache_miss_tokens": 20,
            }),
            80,
        )

    def test_explicit_zero_does_not_hide_details(self):
        self.assertEqual(
            _extract_cache_read_tokens({
                "prompt_tokens": 100,
                "prompt_cache_hit_tokens": 0,
                "prompt_tokens_details": {"cached_tokens": 80},
            }),
            80,
        )

    def test_derive_from_miss_when_hit_missing(self):
        self.assertEqual(
            _extract_cache_read_tokens({
                "prompt_tokens": 100,
                "prompt_cache_miss_tokens": 20,
            }),
            80,
        )

    def test_zero_hit_and_zero_miss_not_all_cached(self):
        self.assertEqual(
            _extract_cache_read_tokens({
                "prompt_tokens": 100,
                "prompt_cache_hit_tokens": 0,
                "prompt_cache_miss_tokens": 0,
            }),
            0,
        )

    def test_non_stream_response_keeps_cache_read(self):
        resp = from_openai_chat_response(
            {
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 2,
                    "prompt_cache_hit_tokens": 80,
                    "prompt_cache_miss_tokens": 20,
                },
            },
            "claude-sonnet-5",
        )
        self.assertEqual(resp["usage"]["cache_read_input_tokens"], 80)
        self.assertEqual(resp["usage"]["input_tokens"], 100)


class StreamCacheUsageTests(unittest.TestCase):
    def _run(self, chunks: list[str]) -> list[str]:
        async def gen():
            for c in chunks:
                yield c.encode("utf-8")

        async def collect():
            events = []
            async for event in stream_openai_chat_to_anthropic(gen(), "claude-sonnet-5"):
                events.append(event)
            return events

        return asyncio.run(collect())

    def _delta_usage(self, events: list[str]) -> dict:
        for event in events:
            for line in event.split("\n"):
                if not line.startswith("data: "):
                    continue
                data = json.loads(line[6:])
                if data.get("type") == "message_delta":
                    return data.get("usage") or {}
        self.fail("no message_delta usage found")
        return {}

    def test_usage_chunk_with_empty_choices(self):
        usage = self._delta_usage(self._run([
            'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n',
            'data: {"choices":[],"usage":{"prompt_tokens":100,"completion_tokens":2,"prompt_cache_hit_tokens":80,"prompt_cache_miss_tokens":20}}\n\n',
            "data: [DONE]\n\n",
        ]))
        self.assertEqual(usage["input_tokens"], 100)
        self.assertEqual(usage["cache_read_input_tokens"], 80)
        self.assertEqual(usage["output_tokens"], 2)

    def test_details_cached_tokens_when_hit_field_is_zero(self):
        usage = self._delta_usage(self._run([
            'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n',
            'data: {"choices":[],"usage":{"prompt_tokens":100,"completion_tokens":1,"prompt_cache_hit_tokens":0,"prompt_tokens_details":{"cached_tokens":80}}}\n\n',
            "data: [DONE]\n\n",
        ]))
        self.assertEqual(usage["cache_read_input_tokens"], 80)

    def test_usage_after_done_in_same_buffer(self):
        chunk = (
            'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
            "data: [DONE]\n"
            'data: {"choices":[],"usage":{"prompt_tokens":50,"completion_tokens":1,"prompt_tokens_details":{"cached_tokens":40}}}\n\n'
        )
        usage = self._delta_usage(self._run([chunk]))
        self.assertEqual(usage["input_tokens"], 50)
        self.assertEqual(usage["cache_read_input_tokens"], 40)

    def test_usage_leftover_without_trailing_newline(self):
        usage = self._delta_usage(self._run([
            'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n',
            'data: {"choices":[],"usage":{"prompt_tokens":50,"completion_tokens":1,"prompt_cache_hit_tokens":40}}',
        ]))
        self.assertEqual(usage["cache_read_input_tokens"], 40)


if __name__ == "__main__":
    unittest.main()
