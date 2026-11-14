"""Tests for the generic web-provider layer: parsers, config-declared sites, registry."""

import asyncio
import json
import os
import time
import unittest
import unittest.mock

from app.providers.web import registry
from app.providers.web.base import (
    MAX_CONVERSATIONS,
    Conversation,
    CooldownTracker,
    JsonLineParser,
    SseJsonParser,
    ToolCallTruncator,
    WebProviderError,
    WebSite,
    dig,
    match_conversation,
    record_conversation,
)
from app.providers.web.http import HttpRuntime
from app.providers.web.sites.custom import ConfigSite, JsonLineSiteParser, PathStreamParser, WholeJsonParser


class _HttpSite(WebSite):
    """Stand-in direct-HTTP site, so ``http.py`` stays covered without a vendor adapter."""

    name = "test-http"
    label = "Test HTTP"
    transport = "http"
    env_prefix = "TEST_HTTP"
    default_models = {"test-model": "Test Model"}

    def new_parser(self, route):
        return SseJsonParser()

    def build_http_request(self, route, system, turns, conv, prompt):
        return "POST", "https://example.test/api/chat", {"Authorization": "Bearer tok"}, {"prompt": prompt}


def _frame(obj, event=None):
    head = f"event: {event}\n" if event else ""
    return f"{head}data: {json.dumps(obj)}\n\n"


def _texts(parsed):
    return "".join(d for k, d in parsed if k == "text")


def _thinking(parsed):
    return "".join(d for k, d in parsed if k == "thinking")


class DigTests(unittest.TestCase):
    def test_reads_dict_and_list_segments(self):
        obj = {"a": {"b": [{"c": "x"}]}}
        self.assertEqual(dig(obj, "a.b.0.c"), "x")
        self.assertEqual(dig(obj, ["a", "b", 0, "c"]), "x")
        self.assertEqual(dig(obj, "a/0/c"), None)
        self.assertEqual(dig(obj, "a.missing.c"), None)
        self.assertEqual(dig(obj, None), obj)


class BaseParserTests(unittest.TestCase):
    class _Echo(SseJsonParser):
        def handle(self, obj, out):
            self.emit_text(dig(obj, "text"), out)

    def test_done_markers(self):
        for raw in ("data: [DONE]\n\n", "event: done\ndata: {}\n\n", _frame({"finished": True}, "finish")):
            p = self._Echo()
            p.feed(raw)
            self.assertTrue(p.done, raw)

    def test_frames_split_across_chunks_and_comments(self):
        p = self._Echo()
        raw = ": keep-alive\n\n" + _frame({"text": "ab"}) + _frame({"text": "cd"})
        out = []
        for i in range(0, len(raw), 5):
            out += p.feed(raw[i:i + 5])
        self.assertEqual(_texts(out), "abcd")

    def test_flush_handles_trailing_line_without_newline(self):
        p = self._Echo()
        self.assertEqual(p.feed('data: {"text": "x"}'), [])
        self.assertEqual(_texts(p.flush()), "x")

    def test_json_line_parser_accepts_bare_objects(self):
        p = JsonLineParser()
        p.handle = lambda obj, out: p.emit_text(dig(obj, "message"), out)
        out = p.feed('{"message": "a"}\n{"message": "b"}\n')
        self.assertEqual(_texts(out), "ab")


class _Provider:
    def __init__(self, **kwargs):
        self.base_url = kwargs.get("base_url", "https://example.test")
        self.api_key = kwargs.get("api_key", "tok")
        self.api_keys = kwargs.get("api_keys", [self.api_key] if self.api_key else [])
        self.headers = kwargs.get("headers", {})
        self.active_account = kwargs.get("active_account", "")
        self.web = kwargs.get("web", {})

    def resolved_api_keys(self):
        return list(self.api_keys)


class _ModelConfig:
    upstream_model_id = None
    extra_body = {}


class _Route:
    def __init__(self, provider, model_id="test-model", provider_type="web-http", extra_body=None):
        self.provider = provider
        self.provider_name = "test"
        self.provider_type = provider_type
        self.model_id = model_id
        mc = type("MC", (_ModelConfig,), {})()
        mc.extra_body = extra_body or {}
        self.model_config = mc


class ConfigSiteTests(unittest.TestCase):
    WEB = {
        "url": "https://niche.example/api/chat",
        "method": "POST",
        "headers": {"X-Version": "{{model}}"},
        "body": {"model": "{{model}}", "stream": True, "messages": "{{messages}}", "chat": "{{chat_id}}"},
        "stream": {
            "frames": "sse",
            "text": "data.text",
            "thinking": "data.think",
            "done_on": "data.finished",
            "chat_id": "cid",
        },
    }

    def _route(self, **web):
        provider = _Provider(base_url="https://niche.example", api_key="secret", web=web or self.WEB)
        return _Route(provider, model_id="their-model", provider_type="web-http")

    def test_body_and_headers_are_templated(self):
        site = ConfigSite()
        route = self._route()
        conv = Conversation(url="c1", system="", user_turns=["u"], n_messages=1)
        method, url, headers, body = site.build_http_request(
            route, "", [("user", "hi"), ("assistant", "yo")], conv, "p"
        )
        self.assertEqual((method, url), ("POST", "https://niche.example/api/chat"))
        self.assertEqual(headers["X-Version"], "their-model")
        self.assertEqual(headers["Authorization"], "Bearer secret")
        self.assertEqual(body["messages"], [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}])
        self.assertEqual(body["chat"], "c1")

    def test_sse_parser_uses_declared_paths(self):
        parser = PathStreamParser(self.WEB["stream"])
        out = parser.feed(_frame({"cid": "abc", "data": {"text": "He", "think": "hm"}}))
        out += parser.feed(_frame({"data": {"text": "llo", "finished": True}}))
        self.assertEqual(_texts(out), "Hello")
        self.assertEqual(_thinking(out), "hm")
        self.assertEqual(parser.chat_ref, "abc")
        self.assertTrue(parser.done)

    def test_jsonl_frames(self):
        parser = JsonLineSiteParser(self.WEB["stream"])
        out = parser.feed('{"data": {"text": "a"}}\n{"data": {"text": "b"}}\n')
        self.assertEqual(_texts(out), "ab")

    def test_whole_json_body(self):
        spec = dict(self.WEB["stream"], frames="json", text="data.text")
        parser = WholeJsonParser(spec)
        self.assertEqual(parser.feed('{"data": {"text": "x"}}'), [])
        self.assertEqual(_texts(parser.flush()), "x")
        self.assertTrue(parser.done)

    def test_missing_web_block_explains_itself(self):
        site = ConfigSite()
        route = _Route(_Provider(base_url="https://x", web={}), provider_type="web-http")
        with self.assertRaises(Exception) as ctx:
            site.new_parser(route)
        self.assertIn("web", str(ctx.exception))


class HttpRuntimeTests(unittest.TestCase):
    def test_masks_tokens_for_display(self):
        rt = HttpRuntime(_HttpSite())
        self.assertEqual(rt.mask("abcdefghijklmnop"), "abcd…mnop")
        self.assertEqual(rt.mask("short"), "…")
        self.assertEqual(rt.mask(""), "")

    def test_active_account_wins_then_first_key(self):
        rt = HttpRuntime(_HttpSite())
        self.assertEqual(rt.account_for(_Provider(api_keys=["a", "b"], active_account="b")), "b")
        self.assertEqual(rt.account_for(_Provider(api_keys=["a", "b"], active_account="zz")), "a")
        self.assertEqual(rt.account_for(None), "")


class ConversationTrackingTests(unittest.TestCase):
    def test_records_and_caps_the_list(self):
        convs = []
        for i in range(MAX_CONVERSATIONS + 4):
            record_conversation(convs, None, "sys", [("user", f"q{i}")], f"ref{i}")
        self.assertEqual(len(convs), MAX_CONVERSATIONS)
        self.assertEqual(convs[0].url, f"ref{MAX_CONVERSATIONS + 3}")

    def test_updates_an_existing_conversation(self):
        conv = Conversation(url="u", system="sys", user_turns=["a"], n_messages=1)
        convs = [conv]
        record_conversation(convs, conv, "sys", [("user", "a"), ("assistant", "b"), ("user", "c")], "u")
        self.assertEqual(conv.user_turns, ["a", "c"])
        self.assertEqual(conv.n_messages, 3)

    def test_next_agent_step_still_matches_after_recording(self):
        convs = []
        record_conversation(convs, None, "sys", [("user", "a")], "u")
        turns = [("user", "a"), ("assistant", "b"), ("user", "c")]
        self.assertIs(match_conversation(convs, "sys", turns), convs[0])


class RegistryTests(unittest.TestCase):
    def test_known_and_unknown_sites(self):
        self.assertTrue(registry.is_web_provider("deepseek-web"))
        self.assertTrue(registry.is_web_provider("hailuo-web"))
        self.assertTrue(registry.is_web_provider("web-http"))
        self.assertFalse(registry.is_web_provider("openai"))
        self.assertIsNone(registry.site_for("nope"))
        self.assertEqual(registry.site_for("deepseek-web").transport, "browser")
        self.assertEqual(registry.site_for("hailuo-web").transport, "browser")
        self.assertEqual(registry.site_for("web-http").transport, "http")

    def test_every_site_has_a_runtime(self):
        for site in registry.SITE_LIST:
            self.assertIsNotNone(registry.runtime_for(site.name))

    def test_truncator_is_shared_by_all_sites(self):
        t = ToolCallTruncator()
        emit, done = t.feed("a<tool_call><name>X</name><parameters>{}</parameters></tool_call> tail")
        self.assertTrue(done)
        self.assertTrue(emit.endswith("</tool_call>"))


class _FakeRuntime:
    def __init__(self, deltas):
        self.deltas = deltas
        self.calls = []

    def account_for(self, provider):
        return "acct"

    async def turn(self, system, turns, route):
        self.calls.append((system, turns))
        for delta in self.deltas:
            yield delta


class ApiPipelineTests(unittest.TestCase):
    """The transport-agnostic pipeline: normaliser -> truncator -> Anthropic SSE."""

    def _run(self, deltas, tools_in_system=False):
        import asyncio
        import json
        from unittest import mock

        from app.models import AnthropicRequest
        from app.providers import web

        runtime = _FakeRuntime(deltas)
        system = "sys <tool_call>" if tools_in_system else "sys"
        request = AnthropicRequest(model="m", system=system, messages=[{"role": "user", "content": "hi"}])
        route = _Route(_Provider(), model_id="test-model")

        async def collect():
            return [e async for e in web.send_streaming(request, route)]

        with (
            mock.patch.object(web.registry, "site_for", return_value=web.registry.site_for("web-http")),
            mock.patch.object(web.registry, "runtime_for", return_value=runtime),
        ):
            events = asyncio.run(collect())

        parsed = []
        for event in events:
            for line in event.splitlines():
                if line.startswith("data: "):
                    parsed.append(json.loads(line[6:]))
        return parsed, runtime

    def test_plain_text_streams_as_one_text_block(self):
        parsed, _ = self._run(["Hello ", "there"])
        self.assertEqual(parsed[0]["type"], "message_start")
        self.assertEqual(parsed[1]["content_block"]["type"], "text")
        self.assertEqual("".join(p["delta"]["text"] for p in parsed if p["type"] == "content_block_delta"), "Hello there")
        self.assertEqual([p for p in parsed if p["type"] == "message_delta"][0]["delta"]["stop_reason"], "end_turn")

    def test_tool_call_stops_generation_after_the_closing_tag(self):
        deltas = [
            "ok\n<tool_call>\n<name>Bash</name>\n<parameters>{}</parameters>\n</tool_call>",
            "\nrambling",
        ]
        parsed, _ = self._run(deltas, tools_in_system=True)
        text = "".join(p["delta"]["text"] for p in parsed if p["type"] == "content_block_delta")
        self.assertTrue(text.endswith("</tool_call>"))
        self.assertNotIn("rambling", text)

    def test_react_request_is_flattened_for_the_site(self):
        _, runtime = self._run(["x"])
        system, turns = runtime.calls[0]
        self.assertEqual(system, "sys")
        self.assertEqual(turns, [("user", "hi")])


class CooldownTests(unittest.TestCase):
    """A captcha/429 streak must stop the account instead of piling on more traffic."""

    def test_second_hard_failure_starts_a_cooldown(self):
        guard = CooldownTracker(threshold=2, cooldown_seconds=600, label="site")
        guard.check("a")  # usable before any failure
        guard.record("a", "challenge")
        guard.check("a")  # one failure is not enough
        guard.record("a", "challenge")
        self.assertGreater(guard.remaining("a"), 590)
        with self.assertRaises(WebProviderError) as ctx:
            guard.check("a")
        self.assertIn("cooling down", str(ctx.exception))

    def test_success_resets_the_streak(self):
        guard = CooldownTracker(threshold=2, cooldown_seconds=600)
        guard.record("a", "challenge")
        guard.success("a")
        guard.record("a", "challenge")
        guard.check("a")  # streak restarted, still usable

    def test_accounts_cool_down_independently(self):
        guard = CooldownTracker(threshold=1, cooldown_seconds=600)
        guard.record("a", "HTTP 429")
        with self.assertRaises(WebProviderError):
            guard.check("a")
        guard.check("b")


class HttpPacingTests(unittest.TestCase):
    """The direct transport must throttle too, not just the browser one."""

    def test_pace_waits_out_the_remaining_interval(self):
        runtime = HttpRuntime(_HttpSite())
        runtime.last_turn_end = time.monotonic()
        sleeper = unittest.mock.AsyncMock()
        with (
            unittest.mock.patch.dict(os.environ, {"TEST_HTTP_MIN_INTERVAL": "10"}),
            unittest.mock.patch("app.providers.web.http.asyncio.sleep", new=sleeper),
        ):
            asyncio.run(runtime._pace())
        slept = sleeper.await_args.args[0]
        self.assertGreater(slept, 9.0)
        self.assertLessEqual(slept, 10.0)

    def test_first_turn_is_not_delayed(self):
        runtime = HttpRuntime(_HttpSite())
        sleeper = unittest.mock.AsyncMock()
        with unittest.mock.patch("app.providers.web.http.asyncio.sleep", new=sleeper):
            asyncio.run(runtime._pace())
        sleeper.assert_not_awaited()


class TokenCaptureTests(unittest.TestCase):
    """The 'log in once, let the proxy keep the token' path."""

    def test_clean_token_handles_the_shapes_pages_use(self):
        from app.providers.web.token_capture import clean_token

        self.assertEqual(clean_token("Bearer abcdefghijklmnop"), "abcdefghijklmnop")
        self.assertEqual(clean_token('"abcdefghijklmnop"'), "abcdefghijklmnop")
        self.assertEqual(clean_token('{"value": "abcdefghijklmnop"}'), "abcdefghijklmnop")
        self.assertEqual(clean_token('{"userToken": "abcdefghijklmnop"}'), "abcdefghijklmnop")
        self.assertIsNone(clean_token("short"))
        self.assertIsNone(clean_token(None))
        self.assertIsNone(clean_token("has spaces in it at all"))

    def test_storage_scan_prefers_token_like_values(self):
        from app.providers.web.token_capture import token_from_storage

        jwt = "eyJhbGciOiJIUzI1NiJ9." + "a" * 40
        storage = {"theme": "dark", "userToken": json.dumps({"value": jwt})}
        self.assertEqual(token_from_storage(storage), jwt)
        self.assertIsNone(token_from_storage({"theme": "dark"}))
        self.assertIsNone(token_from_storage({}))

    def test_storage_scan_ignores_non_token_keys_and_junk_values(self):
        from app.providers.web.token_capture import token_from_storage

        # A visitor's device id under an unrelated key must not become the API token.
        self.assertIsNone(token_from_storage({"visitorId": "g9IE_HF_" + "x" * 280}))
        # A token-ish key holding junk is refused too.
        self.assertIsNone(token_from_storage({"accessToken": "hello world"}))

    def test_profile_dirs_never_take_a_token_as_a_name(self):
        from app.providers.web.engine import BrowserRuntime

        token = "g9IE_HF_" + "x" * 300
        safe = BrowserRuntime.safe_dir_name(token)
        self.assertLess(len(safe), 64)
        self.assertNotIn(" ", safe)
        self.assertEqual(BrowserRuntime.safe_dir_name("alice"), "alice")

    def test_browser_sites_are_told_to_use_the_login_window(self):
        from app.providers.web import registry

        with self.assertRaises(Exception) as ctx:
            asyncio.run(registry.capture_token("deepseek-web"))
        self.assertIn("login window", str(ctx.exception))


class HailuoParserTests(unittest.TestCase):
    """MiniMax Agent frames, as captured from the live answer stream."""

    def _parser(self):
        from app.providers.web.sites.hailuo import HailuoStreamParser

        return HailuoStreamParser()

    @staticmethod
    def _chunk(msg_id, text=None, index=0, finish=False, role="assistant"):
        frame = {
            "type": 6,
            "agent_message_chunk": {"msg_id": msg_id, "chunk_index": index, "role": role},
        }
        if text is not None:
            frame["agent_message_chunk"]["msg_content"] = text
        if finish:
            frame["agent_message_chunk"]["finish"] = True
            frame["agent_message_chunk"]["finish_reason"] = "stop"
        return frame

    @staticmethod
    def _message(msg_id, text, role="assistant"):
        return {"type": 2, "agent_message": {"msg_id": msg_id, "role": role, "msg_content": text}}

    def test_chunk_deltas_are_emitted_in_order(self):
        p = self._parser()
        out = p.feed(_frame(self._chunk("msg-1", "杭州", 0)))
        out += p.feed(_frame(self._chunk("msg-1", "是浙江省省会", 1)))
        out += p.feed(_frame(self._chunk("msg-1", index=2, finish=True)))
        self.assertEqual(_texts(out), "杭州是浙江省省会")
        self.assertTrue(p.done)

    def test_full_message_frame_does_not_duplicate_the_chunks(self):
        p = self._parser()
        out = p.feed(_frame(self._chunk("msg-1", "你好", 0)))
        out += p.feed(_frame(self._message("msg-1", "你好")))
        self.assertEqual(_texts(out), "你好")

    def test_full_message_frame_only_fills_a_missed_prefix(self):
        # The frame carrying the first chunk was lost: the closing whole-message
        # frame still has to deliver the missing opening words, and only those.
        p = self._parser()
        out = p.feed(_frame(self._chunk("msg-1", "是浙江省", 1)))
        out += p.feed(_frame(self._message("msg-1", "杭州是浙江省")))
        self.assertEqual(_texts(out), "是浙江省杭州")

    def test_user_echo_is_ignored(self):
        p = self._parser()
        out = p.feed(_frame(self._message("1", "用一句话介绍杭州", role="user")))
        out += p.feed(_frame(self._chunk("msg-1", "杭州很美")))
        self.assertEqual(_texts(out), "杭州很美")

    def test_frame_split_across_chunks_is_reassembled(self):
        p = self._parser()
        raw = _frame(self._chunk("msg-1", "完整的一句话"))
        out = p.feed(raw[:30])
        out += p.feed(raw[30:])
        self.assertEqual(_texts(out), "完整的一句话")

    def test_session_finish_ends_the_turn(self):
        p = self._parser()
        p.feed(_frame({"type": "session.finish", "payload": {"sessionId": "1"}}))
        self.assertTrue(p.done)

    def test_upstream_error_surfaces(self):
        p = self._parser()
        with self.assertRaises(RuntimeError):
            p.feed(_frame({"base_resp": {"status_code": 1004, "status_msg": "bad token"}}))


class ProbeTests(unittest.TestCase):
    """'Copy as cURL' -> (method, url, headers, body), so any site can be probed."""

    CURL = (
        "curl 'https://chat.example.com/api/chat' \
"
        "  -H 'accept: text/event-stream' \
"
        "  -H 'authorization: Bearer abc123' \
"
        "  -H 'content-length: 42' \
"
        "  -H 'accept-encoding: gzip, deflate, br' \
"
        "  --data-raw '{\"model\":\"x\",\"stream\":true}' \
"
        "  --compressed"
    )

    def test_parses_a_bash_curl(self):
        from app.providers.web.probe import parse_curl

        method, url, headers, body = parse_curl(self.CURL)
        self.assertEqual(method, "POST")
        self.assertEqual(url, "https://chat.example.com/api/chat")
        self.assertEqual(headers["authorization"], "Bearer abc123")
        self.assertNotIn("content-length", headers)
        self.assertNotIn("accept-encoding", headers)
        self.assertEqual(body, b'{"model":"x","stream":true}')

    def test_parses_cmd_line_continuations(self):
        from app.providers.web.probe import parse_curl

        curl = 'curl.exe "https://x/api" ^' + "\r\n" + '  -H "authorization: Bearer z"'
        method, url, headers, body = parse_curl(curl)
        self.assertEqual(url, "https://x/api")
        self.assertEqual(headers["authorization"], "Bearer z")
        self.assertIsNone(body)
        self.assertEqual(method, "GET")

    def test_rejects_non_curl_input(self):
        from app.providers.web.base import WebProviderError
        from app.providers.web.probe import parse_curl

        with self.assertRaises(WebProviderError):
            parse_curl("wget https://x")


if __name__ == "__main__":
    unittest.main()
