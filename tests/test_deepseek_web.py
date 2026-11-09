import asyncio
import json
import unittest

from app.models import AnthropicRequest
from app.providers.deepseek_web_provider import (
    LEGACY_PROFILE_DIR,
    Conversation,
    DeepSeekStreamParser,
    DsmlNormalizer,
    ToolCallTruncator,
    _account_id,
    _convs_for,
    _on_emit,
    _profile_dir,
    _profile_root,
    _state_for,
    _states,
    anthropic_events_from_text,
    build_incremental_prompt,
    dsml_to_tool_call,
    flatten_prompt,
    match_conversation,
)
from app.react.parser import parse_tool_calls
from app.react import transform_request, transform_stream


def _frame(obj, event=None):
    head = f"event: {event}\n" if event else ""
    return f"{head}data: {json.dumps(obj)}\n\n"


def _texts(parsed):
    return "".join(d for k, d in parsed if k == "text")


class StreamParserTests(unittest.TestCase):
    def test_fragment_append_then_inherited_path(self):
        p = DeepSeekStreamParser()
        out = []
        out += p.feed(_frame({"v": {"response": {"fragments": [{"type": "RESPONSE", "content": ""}]}}}))
        out += p.feed(_frame({"p": "response/fragments/-1/content", "o": "APPEND", "v": "Hel"}))
        out += p.feed(_frame({"v": "lo"}))  # p/o inherited
        out += p.feed(_frame({"v": " world"}))
        self.assertEqual(_texts(out), "Hello world")
        self.assertFalse(p.done)

    def test_think_fragment_is_separate_kind(self):
        p = DeepSeekStreamParser()
        out = []
        out += p.feed(_frame({"v": {"response": {"fragments": [{"type": "THINK", "content": "hm"}]}}}))
        out += p.feed(_frame({"p": "response/fragments/-1/content", "o": "APPEND", "v": "m"}))
        out += p.feed(_frame({"p": "response/fragments", "o": "APPEND", "v": [{"type": "RESPONSE", "content": "Hi"}]}))
        out += p.feed(_frame({"p": "response/fragments/-1/content", "o": "APPEND", "v": "!"}))
        self.assertEqual([d for k, d in out if k == "thinking"], ["hm", "m"])
        self.assertEqual(_texts(out), "Hi!")

    def test_set_op_emits_only_new_suffix(self):
        p = DeepSeekStreamParser()
        out = p.feed(_frame({"v": {"response": {"fragments": [{"type": "RESPONSE", "content": "ab"}]}}}))
        out += p.feed(_frame({"p": "response/fragments/0/content", "o": "SET", "v": "abcd"}))
        self.assertEqual(_texts(out), "abcd")

    def test_batch_finish_marks_done(self):
        p = DeepSeekStreamParser()
        p.feed(_frame({"v": {"response": {"fragments": []}}}))
        p.feed(_frame({"p": "response", "o": "BATCH", "v": [
            {"p": "accumulated_token_usage", "v": 12},
            {"p": "status", "v": "FINISHED"},
        ]}))
        self.assertTrue(p.done)

    def test_status_patch_and_event_markers_finish(self):
        p = DeepSeekStreamParser()
        p.feed(_frame({"p": "response/status", "v": "FINISHED"}))
        self.assertTrue(p.done)
        q = DeepSeekStreamParser()
        q.feed("event: finish\ndata: {}\n\n")
        self.assertTrue(q.done)
        r = DeepSeekStreamParser()
        r.feed("data: [DONE]\n\n")
        self.assertTrue(r.done)

    def test_frames_split_across_chunks(self):
        p = DeepSeekStreamParser()
        raw = _frame({"v": {"response": {"fragments": [{"type": "RESPONSE", "content": "x"}]}}})
        raw += _frame({"p": "response/fragments/-1/content", "o": "APPEND", "v": "yz"})
        out = []
        for i in range(0, len(raw), 7):
            out += p.feed(raw[i:i + 7])
        self.assertEqual(_texts(out), "xyz")

    def test_unknown_fragment_types_ignored(self):
        p = DeepSeekStreamParser()
        out = p.feed(_frame({"p": "response/fragments", "o": "APPEND", "v": [{"type": "SEARCH", "content": "q"}]}))
        self.assertEqual(out, [])

    def test_legacy_scalar_content(self):
        p = DeepSeekStreamParser()
        out = p.feed(_frame({"p": "response/content", "o": "APPEND", "v": "hi"}))
        out += p.feed(_frame({"v": "!"}))
        self.assertEqual(_texts(out), "hi!")


class ToolCallTruncatorTests(unittest.TestCase):
    def test_cuts_after_closing_tag_across_deltas(self):
        t = ToolCallTruncator()
        pieces = ["ok\n<tool_call>\n<name>Bash</name>\n<parameters>{}</parameters>\n</tool_", "call>\n\nextra rambling"]
        emitted = ""
        finished = False
        for d in pieces:
            e, finished = t.feed(d)
            emitted += e
            if finished:
                break
        self.assertTrue(finished)
        self.assertTrue(emitted.endswith("</tool_call>"))
        self.assertNotIn("rambling", emitted)

    def test_stray_close_tag_without_open_is_ignored(self):
        t = ToolCallTruncator()
        e, finished = t.feed("explaining </tool_call> usage")
        self.assertFalse(finished)
        self.assertEqual(e, "explaining </tool_call> usage")


class FlattenPromptTests(unittest.TestCase):
    def test_single_user_message_sent_verbatim(self):
        req = AnthropicRequest(model="m", messages=[{"role": "user", "content": "hello"}])
        self.assertEqual(flatten_prompt(req), "hello")

    def test_react_history_is_flattened(self):
        body = {
            "model": "m",
            "system": "You are helpful.",
            "tools": [{"name": "Bash", "description": "run", "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}}}],
            "messages": [
                {"role": "user", "content": "list files"},
                {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls"}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "a.txt"}]},
            ],
        }
        prompt = flatten_prompt(AnthropicRequest(**transform_request(body)))
        self.assertIn("<system>", prompt)
        self.assertIn("<name>Bash</name>", prompt)
        self.assertIn("<observation>\na.txt\n</observation>", prompt)
        self.assertIn("[assistant]", prompt)
        self.assertTrue(prompt.rstrip().endswith("stop."))


class EndToEndSseTests(unittest.TestCase):
    def _run_through_react(self, deltas):
        async def upstream():
            for e in anthropic_events_from_text(deltas, "deepseek-web"):
                yield e

        async def collect():
            return [e async for e in transform_stream(upstream(), "claude-test")]

        events = asyncio.run(collect())
        parsed = []
        for e in events:
            for line in e.split("\n"):
                if line.startswith("data: "):
                    parsed.append(json.loads(line[6:]))
        return parsed

    def test_tool_call_text_becomes_tool_use_block(self):
        text = "Let me look.\n<tool_call>\n<name>Bash</name>\n<parameters>\n{\"command\": \"ls\"}\n</parameters>\n</tool_call>"
        parsed = self._run_through_react([text[:20], text[20:]])
        tool_blocks = [p for p in parsed if p["type"] == "content_block_start" and p["content_block"]["type"] == "tool_use"]
        self.assertEqual(len(tool_blocks), 1)
        self.assertEqual(tool_blocks[0]["content_block"]["name"], "Bash")
        args = "".join(p["delta"]["partial_json"] for p in parsed if p["type"] == "content_block_delta" and p["delta"]["type"] == "input_json_delta")
        self.assertEqual(json.loads(args), {"command": "ls"})
        stop = [p for p in parsed if p["type"] == "message_delta"][0]
        self.assertEqual(stop["delta"]["stop_reason"], "tool_use")

    def test_plain_text_stays_end_turn(self):
        parsed = self._run_through_react(["Hello ", "there"])
        text = "".join(p["delta"]["text"] for p in parsed if p["type"] == "content_block_delta")
        self.assertEqual(text, "Hello there")
        self.assertEqual([p for p in parsed if p["type"] == "message_delta"][0]["delta"]["stop_reason"], "end_turn")


B = "\uff5c\uff5c"  # full-width bars as seen in the wild
DSML_WRITE = (
    f'<{B}DSML{B} calls>\n<{B}DSML{B} invoke name="Write">\n'
    f'<{B}DSML{B} parameter name="file_path" string="true">C:/x/hello.py</{B}DSML{B} parameter>\n'
    f'<{B}DSML{B} parameter name="content" string="true">def add(a, b):\n    return a + b\n</{B}DSML{B} parameter>\n'
    f'</{B}DSML{B} invoke>\n</{B}DSML{B} calls>'
)



class DsmlTests(unittest.TestCase):
    def test_block_converts_to_parseable_tool_call(self):
        xml = dsml_to_tool_call(DSML_WRITE)
        calls = parse_tool_calls(xml)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "Write")
        self.assertEqual(calls[0]["arguments"]["file_path"], "C:/x/hello.py")
        self.assertEqual(calls[0]["arguments"]["content"], "def add(a, b):\n    return a + b\n")

    def test_malformed_close_tags_still_parse(self):
        # Real sample: invoke closed by a bare </DSML>, stray </parameter>.
        block = (
            f'<{B}DSML{B} calls>\n<{B}DSML{B} invoke name="Read">\n'
            f'<{B}DSML{B} parameter name="file_path" string="true">C:/a/config.py</{B}DSML{B} parameter>\n'
            f'<{B}DSML{B} parameter name="limit" string="false">120</{B}DSML{B} parameter>\n'
            f'</{B}DSML{B} parameter>\n</{B}DSML{B}>\n</{B}DSML{B} calls>'
        )
        calls = parse_tool_calls(dsml_to_tool_call(block))
        self.assertEqual(calls[0]["name"], "Read")
        self.assertEqual(calls[0]["arguments"], {"file_path": "C:/a/config.py", "limit": 120})

    def test_hybrid_tool_call_header_with_dsml_params(self):
        text = (
            "Looking.\n<tool_call>\n<name>Grep</name>\n"
            '<parameter name="pattern">^from x</' + B + "DSML" + B + " parameter>\n"
            f'<{B}DSML{B} parameter name="-n" string="false">true</{B}DSML{B} parameter>\n'
            f"</{B}DSML{B} invoke>\n</{B}DSML{B} calls>"
        )
        n = DsmlNormalizer()
        out, done = "", False
        for i in range(0, len(text), 6):
            piece, done = n.feed(text[i:i + 6])
            out += piece
            if done:
                break
        self.assertTrue(done)
        calls = parse_tool_calls(out)
        self.assertEqual(calls[0]["name"], "Grep")
        self.assertEqual(calls[0]["arguments"], {"pattern": "^from x", "-n": True})

    def test_normal_tool_call_not_held_back(self):
        n = DsmlNormalizer()
        out = ""
        for piece in ["<tool_call>\n<name>Bash</name>\n<parameters>", '{"command":"ls"}</parameters>\n</tool_call>']:
            p, done = n.feed(piece)
            self.assertFalse(done)
            out += p
        out += n.flush()
        self.assertIn("<parameters>", out)
        self.assertTrue(out.endswith("</tool_call>"))

    def test_non_string_parameter_is_json(self):
        block = f'<{B}DSML{B} invoke name="T"><{B}DSML{B} parameter name="n" string="false">3</{B}DSML{B} parameter></{B}DSML{B} invoke>'
        self.assertIn('"n": 3', dsml_to_tool_call(block))

    def test_streaming_normalizer_across_arbitrary_splits(self):
        text = "Sure.\n" + DSML_WRITE + "\nignored tail"
        n = DsmlNormalizer()
        out, done = "", False
        for i in range(0, len(text), 5):
            piece, done = n.feed(text[i:i + 5])
            out += piece
            if done:
                break
        self.assertTrue(done)
        self.assertTrue(out.startswith("Sure."))
        self.assertIn("<name>Write</name>", out)
        self.assertNotIn("DSML", out)
        self.assertNotIn("ignored", out)

    def test_plain_text_and_lone_angle_bracket_pass_through(self):
        n = DsmlNormalizer()
        out = ""
        for piece in ["a < b ", "and <", "tool_call> ok"]:
            p, done = n.feed(piece)
            self.assertFalse(done)
            out += p
        out += n.flush()
        self.assertEqual(out, "a < b and <tool_call> ok")


class ConversationReuseTests(unittest.TestCase):
    SYS = "sys <tool_call>"

    def _turns(self, *pairs):
        return list(pairs)

    def test_next_agent_step_matches_and_sends_only_new_user_turn(self):
        conv = Conversation(url="u", system=self.SYS, user_turns=["task"], n_messages=1)
        turns = self._turns(("user", "task"), ("assistant", "xml (re-serialised)"), ("user", "<observation>ok</observation>"))
        self.assertIs(match_conversation([conv], self.SYS, turns), conv)
        inc = build_incremental_prompt(self.SYS, turns)
        self.assertTrue(inc.startswith("<observation>ok</observation>"))
        self.assertNotIn("task", inc)

    def test_mismatches_fall_back_to_new_chat(self):
        conv = Conversation(url="u", system=self.SYS, user_turns=["task"], n_messages=1)
        ok = self._turns(("user", "task"), ("assistant", "a"), ("user", "obs"))
        self.assertIsNone(match_conversation([conv], "other system", ok))
        self.assertIsNone(match_conversation([conv], self.SYS, self._turns(("user", "different"), ("assistant", "a"), ("user", "obs"))))
        self.assertIsNone(match_conversation([conv], self.SYS, self._turns(("user", "task"))))  # retry of same request
        self.assertIsNone(match_conversation([], self.SYS, ok))

    def test_picks_the_right_chat_among_several(self):
        a = Conversation(url="a", system=self.SYS, user_turns=["A"], n_messages=1)
        b = Conversation(url="b", system="subagent", user_turns=["B"], n_messages=1)
        turns = self._turns(("user", "A"), ("assistant", "x"), ("user", "obs"))
        self.assertIs(match_conversation([b, a], self.SYS, turns), a)


class _FakeProvider:
    """Minimal stand-in for ProviderConfig in account-selection tests."""

    def __init__(self, keys, active=""):
        self._keys = list(keys)
        self.active_account = active

    def resolved_api_keys(self):
        return list(self._keys)


class AccountTests(unittest.TestCase):
    def test_account_id_uses_active_account(self):
        p = _FakeProvider(["alice", "bob"], "bob")
        self.assertEqual(_account_id(p), "bob")

    def test_account_id_falls_back_when_active_removed(self):
        p = _FakeProvider(["alice", "bob"], "carol")
        self.assertEqual(_account_id(p), "alice")

    def test_account_id_is_empty_without_config(self):
        self.assertEqual(_account_id(None), "")
        self.assertEqual(_account_id(_FakeProvider([])), "")

    def test_profile_dir_is_per_account(self):
        self.assertEqual(_profile_dir("alice"), str(_profile_root() / "alice"))
        self.assertNotEqual(_profile_dir("alice"), _profile_dir("bob"))
        # The default account keeps the pre-multi-account path so its login survives.
        self.assertEqual(_profile_dir(""), str(LEGACY_PROFILE_DIR))

    def test_conversations_are_tracked_per_account(self):
        alice = _convs_for("alice")
        bob = _convs_for("bob")
        self.assertIsNot(alice, bob)
        alice.append(Conversation(url="u", system="s", user_turns=["t"], n_messages=1))
        self.assertEqual(len(_convs_for("alice")), 1)
        self.assertEqual(len(_convs_for("bob")), 0)
        self.assertIs(_convs_for("alice"), alice)  # stable identity

    def test_on_emit_routes_to_the_right_account_queue(self):
        alice, bob = _state_for("alice"), _state_for("bob")
        alice.queue = asyncio.Queue()
        bob.queue = asyncio.Queue()
        try:
            _on_emit(json.dumps({"__account": "bob", "kind": "chunk", "text": "hi"}))
            self.assertTrue(bob.queue.qsize() == 1)
            self.assertTrue(alice.queue.empty())
            self.assertEqual(bob.queue.get_nowait()["text"], "hi")
        finally:
            alice.queue = bob.queue = None
            _states.pop("alice", None)
            _states.pop("bob", None)

    def test_on_emit_drops_messages_without_a_queue(self):
        _states.pop("ghost", None)
        _on_emit(json.dumps({"__account": "ghost", "kind": "chunk", "text": "x"}))
        self.assertNotIn("ghost", _states)


if __name__ == "__main__":
    unittest.main()
