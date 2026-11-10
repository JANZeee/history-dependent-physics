"""Hand-computable metric tests; does not read generated test data or models."""
import unittest
import numpy as np
from evaluate import compute_metrics, first_crossing


def example(y, initial, active=None):
    y = np.asarray(y, dtype=np.float32)[..., None]
    return {"y": y, "initial": np.asarray(initial, dtype=np.float32)[:, None],
            "active": np.ones(y.shape[:2], dtype=bool) if active is None else np.asarray(active, dtype=bool),
            "failed": y[..., 0] >= 1.0}


class EvaluationTests(unittest.TestCase):
    def test_censored_events_and_return(self):
        d = example([[.2, .5, 1, 1], [.2, .3, .4, .5], [.2, .5, .8, 1]], [.1] * 3)
        p = np.array([[.2, 1, 1, .8], [.2, .3, 1, 1], [.2, .3, .4, .5]])[..., None]
        m = compute_metrics("fatigue", d, p)
        self.assertAlmostEqual(m["event_time_capped_mae_all_steps"], 4 / 3)
        self.assertEqual(m["false_positive_rate"], 1.0)
        self.assertEqual(m["false_negative_rate"], .5)
        self.assertEqual(m["matched_failure_coverage"], .5)
        self.assertEqual(m["matched_failure_time_count"], 1)
        self.assertEqual(m["post_failure_return_fraction_count"], 1)
        self.assertEqual(m["post_failure_return_fraction_denominator"], 3)
        self.assertIsNone(m["rest_drift_fraction"])

    def test_first_crossing_absent(self):
        np.testing.assert_array_equal(first_crossing([[0, 0, 0], [0, 1, 1], [1, 0, 0]]), [4, 2, 1])

    def test_perfect_cooling(self):
        d = example([[.4, .3, .2, .1]], [.5], [[False] * 4])
        m = compute_metrics("thermal", d, d["y"])
        self.assertEqual(m["state_rmse"], 0)
        self.assertEqual(m["passive_increase_fraction"], 0)
        self.assertEqual(m["ambient_undershoot_fraction"], 0)

    def test_hold_constraint_does_not_remove_numeric_error(self):
        d = example(np.full((1, 128), .3), [.3], np.zeros((1, 128), dtype=bool))
        p = np.full((1, 128, 1), .25)
        m = compute_metrics("fatigue", d, p, passive=True)
        self.assertEqual(m["tail_own_state_final_change_abs_mean"], 0)
        self.assertAlmostEqual(m["tail_state_rmse"], .05, places=6)
        self.assertEqual(m["tail_initial_survivor_count"], 1)


if __name__ == "__main__":
    unittest.main()
