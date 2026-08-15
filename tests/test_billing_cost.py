import unittest

from app.billing import _compute_cost


class ComputeCostTests(unittest.TestCase):
    def test_usd_official_price_stored_as_cny(self):
        prices = {
            "gemini-3.7-flash": {
                "currency": "USD",
                "input": 0.75,
                "input_cached": 0.075,
                "output": 3.75,
            }
        }
        bindings = {"gemini/gemini-3.7-flash": "gemini-3.7-flash"}
        cost, currency = _compute_cost(
            prices,
            "gemini/gemini-3.7-flash",
            1_000_000,
            0,
            0,
            bindings,
            {"USD": 7.2},
        )
        self.assertEqual(currency, "CNY")
        self.assertAlmostEqual(cost, 5.4, places=6)

    def test_usd_cached_and_output_converted(self):
        prices = {
            "gemini-3.7-flash": {
                "currency": "USD",
                "input": 0.75,
                "input_cached": 0.075,
                "output": 3.75,
            }
        }
        bindings = {"gemini/gemini-3.7-flash": "gemini-3.7-flash"}
        # 800k cached + 200k miss + 500k output
        cost, currency = _compute_cost(
            prices,
            "gemini/gemini-3.7-flash",
            1_000_000,
            500_000,
            800_000,
            bindings,
            {"USD": 7.2},
        )
        native = 800_000 / 1e6 * 0.075 + 200_000 / 1e6 * 0.75 + 500_000 / 1e6 * 3.75
        self.assertEqual(currency, "CNY")
        self.assertAlmostEqual(cost, round(native * 7.2, 6), places=6)

    def test_cny_price_unchanged(self):
        prices = {
            "deepseek-v4-flash": {
                "currency": "CNY",
                "input": 3.0,
                "input_cached": 0.10,
                "output": 9.0,
            }
        }
        cost, currency = _compute_cost(
            prices,
            "opencode/deepseek-v4-flash",
            1_000_000,
            0,
            0,
            {"opencode/deepseek-v4-flash": "deepseek-v4-flash"},
        )
        self.assertEqual(currency, "CNY")
        self.assertAlmostEqual(cost, 3.0, places=6)


if __name__ == "__main__":
    unittest.main()
