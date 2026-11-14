import unittest

from app.config import merge_config_files, split_config_files


class GlobalConfigSplitTests(unittest.TestCase):
    def test_disabled_routes_stay_local(self):
        merged = merge_config_files(
            {
                "server": {"host": "127.0.0.1", "port": 10388, "local_only": True},
                "upstream": {
                    "enabled": False,
                    "retry": {"max_attempts": 3, "interval_ms": 1000, "total_timeout_ms": 45000},
                },
                "billing": {"enabled": True, "db_file": "logs/billing.db"},
            },
            {
                "upstream": {"disabled_routes": ["gemini/gemini-3.7-flash"]},
                "models": {"claude-sonnet-5": "opencode/deepseek-v4-flash"},
                "providers": {"opencode": {"provider_type": "openai", "base_url": "http://x"}},
            },
        )
        self.assertEqual(merged["server"]["port"], 10388)
        self.assertFalse(merged["upstream"]["enabled"])
        self.assertEqual(merged["upstream"]["disabled_routes"], ["gemini/gemini-3.7-flash"])
        self.assertEqual(merged["billing"]["enabled"], True)

        global_out, local_out = split_config_files(merged)
        self.assertNotIn("disabled_routes", global_out["upstream"])
        self.assertEqual(local_out["upstream"]["disabled_routes"], ["gemini/gemini-3.7-flash"])
        self.assertNotIn("server", local_out)
        self.assertNotIn("billing", local_out)
        self.assertNotIn("models", global_out)
        self.assertNotIn("providers", global_out)

    def test_price_bindings_stay_local(self):
        merged = merge_config_files(
            {
                "server": {"host": "127.0.0.1", "port": 10388},
                "upstream": {"enabled": True},
                "billing": {"enabled": True, "db_file": "logs/billing.db"},
            },
            {
                "models": {"claude-sonnet-5": "opencode/deepseek-v4-flash"},
                "providers": {"opencode": {"provider_type": "openai", "base_url": "http://x"}},
                "price_bindings": {
                    "opencode/deepseek-v4-flash": "deepseek-v4-flash",
                },
            },
        )
        self.assertEqual(
            merged["price_bindings"],
            {"opencode/deepseek-v4-flash": "deepseek-v4-flash"},
        )
        self.assertEqual(
            merged["billing"]["price_bindings"],
            {"opencode/deepseek-v4-flash": "deepseek-v4-flash"},
        )

        global_out, local_out = split_config_files(merged)
        self.assertNotIn("price_bindings", global_out)
        self.assertNotIn("price_bindings", global_out.get("billing", {}))
        self.assertEqual(
            local_out["price_bindings"],
            {"opencode/deepseek-v4-flash": "deepseek-v4-flash"},
        )

    def test_proxy_lives_in_global_file(self):
        merged = merge_config_files(
            {
                "server": {"host": "127.0.0.1", "port": 9223},
                "proxy": {"enabled": True, "url": "http://192.168.10.91:7890"},
            },
            {
                "models": {"claude-sonnet-5": "opencode/deepseek-v4-flash"},
                "providers": {"opencode": {"provider_type": "openai", "base_url": "http://x"}},
            },
        )
        self.assertEqual(merged["proxy"]["enabled"], True)
        self.assertEqual(merged["proxy"]["url"], "http://192.168.10.91:7890")

        global_out, local_out = split_config_files(merged)
        self.assertEqual(global_out["proxy"], {"enabled": True, "url": "http://192.168.10.91:7890"})
        self.assertNotIn("proxy", local_out)

    def test_local_overlay_keeps_old_all_in_one_config(self):
        merged = merge_config_files(
            {},
            {
                "server": {"host": "127.0.0.1", "port": 9223},
                "upstream": {
                    "enabled": True,
                    "disabled_routes": ["glm2/*"],
                    "retry": {"max_attempts": 2},
                },
                "models": {},
                "providers": {},
            },
        )
        self.assertEqual(merged["server"]["port"], 9223)
        self.assertTrue(merged["upstream"]["enabled"])
        self.assertEqual(merged["upstream"]["retry"]["max_attempts"], 2)
        self.assertEqual(merged["upstream"]["disabled_routes"], ["glm2/*"])


if __name__ == "__main__":
    unittest.main()


class EndpointUrlTests(unittest.TestCase):
    """Where ``base_url`` ends up — the passthrough type is ``claude``, not ``anthropic``."""

    @staticmethod
    def _route(provider_type, base_url, model="m"):
        from app.config import ModelConfig, ProviderConfig, ResolvedRoute, UpstreamConfig

        provider = ProviderConfig(provider_type=provider_type, base_url=base_url, models={})
        return ResolvedRoute("p", provider, model, ModelConfig(name=model), UpstreamConfig())

    def test_claude_passthrough_appends_messages(self):
        route = self._route("claude", "https://api.xiaomimimo.com/anthropic/v1")
        self.assertEqual(route.endpoint_url, "https://api.xiaomimimo.com/anthropic/v1/messages")

    def test_openai_appends_chat_completions(self):
        route = self._route("openai", "https://api.xiaomimimo.com/v1")
        self.assertEqual(route.endpoint_url, "https://api.xiaomimimo.com/v1/chat/completions")

    def test_trailing_slash_on_base_url_is_not_doubled(self):
        route = self._route("openai", "https://x.test/v1/")
        self.assertEqual(route.endpoint_url, "https://x.test/v1/chat/completions")
