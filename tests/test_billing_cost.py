import unittest
from datetime import datetime, timedelta, timezone

from app.billing import _compute_cost, _is_peak

_BJ = timezone(timedelta(hours=8))

_DEEPSEEK_PRICES = {
    "deepseek-v4-flash": {
        "currency": "CNY",
        "peak": {"input": 3.0, "input_cached": 0.1, "output": 9.0},
        "offpeak": {"input": 1.5, "input_cached": 0.05, "output": 4.5},
        "peak_hours": [[9, 12], [14, 18]],
        "peak_weekdays": [1, 2, 3, 4, 5],
    }
}
_DEEPSEEK_BINDINGS = {"opencode/deepseek-v4-flash": "deepseek-v4-flash"}


def _bj(year: int, month: int, day: int, hour: int) -> datetime:
    return datetime(year, month, day, hour, 0, 0, tzinfo=_BJ)


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

    def _cost_at(self, when: datetime) -> float:
        cost, currency = _compute_cost(
            _DEEPSEEK_PRICES,
            "opencode/deepseek-v4-flash",
            1_000_000,
            0,
            0,
            _DEEPSEEK_BINDINGS,
            now_bj=when,
        )
        self.assertEqual(currency, "CNY")
        assert cost is not None
        return cost

    def test_weekday_peak_hour_uses_peak_price(self):
        # 2026-08-28 is Friday; 10:00 is inside 9-12.
        self.assertAlmostEqual(self._cost_at(_bj(2026, 8, 28, 10)), 3.0, places=6)

    def test_weekday_off_hour_uses_offpeak_price(self):
        # Friday 13:00 is lunch / off-peak.
        self.assertAlmostEqual(self._cost_at(_bj(2026, 8, 28, 13)), 1.5, places=6)

    def test_saturday_peak_clock_uses_offpeak_price(self):
        # 2026-08-29 is Saturday; 10:00 would be peak on a weekday.
        self.assertAlmostEqual(self._cost_at(_bj(2026, 8, 29, 10)), 1.5, places=6)

    def test_sunday_afternoon_uses_offpeak_price(self):
        # 2026-08-30 is Sunday; 16:00 would be peak on a weekday.
        self.assertAlmostEqual(self._cost_at(_bj(2026, 8, 30, 16)), 1.5, places=6)

    def test_missing_peak_weekdays_still_peaks_on_weekend(self):
        prices = {
            "deepseek-v4-flash": {
                "currency": "CNY",
                "peak": {"input": 3.0, "input_cached": 0.1, "output": 9.0},
                "offpeak": {"input": 1.5, "input_cached": 0.05, "output": 4.5},
                "peak_hours": [[9, 12], [14, 18]],
            }
        }
        cost, _ = _compute_cost(
            prices,
            "opencode/deepseek-v4-flash",
            1_000_000,
            0,
            0,
            _DEEPSEEK_BINDINGS,
            now_bj=_bj(2026, 8, 29, 10),
        )
        self.assertAlmostEqual(cost, 3.0, places=6)


class IsPeakTests(unittest.TestCase):
    def test_weekend_excluded_by_weekdays(self):
        hours = [[9, 12], [14, 18]]
        weekdays = [1, 2, 3, 4, 5]
        self.assertTrue(_is_peak(hours, _bj(2026, 8, 28, 10), weekdays))
        self.assertFalse(_is_peak(hours, _bj(2026, 8, 29, 10), weekdays))
        self.assertFalse(_is_peak(hours, _bj(2026, 8, 30, 15), weekdays))

    def test_omitted_weekdays_means_every_day(self):
        hours = [[9, 12]]
        self.assertTrue(_is_peak(hours, _bj(2026, 8, 29, 10), None))

    def test_empty_weekdays_means_never_peak(self):
        hours = [[9, 12]]
        self.assertFalse(_is_peak(hours, _bj(2026, 8, 29, 10), []))


if __name__ == "__main__":
    unittest.main()
