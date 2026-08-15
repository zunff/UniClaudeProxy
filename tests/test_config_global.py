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
