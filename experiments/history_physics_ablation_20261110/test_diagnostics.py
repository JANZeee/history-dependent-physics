"""Diagnostic correctness checks; no trained checkpoints or test datasets needed."""
import unittest
from unittest.mock import patch

import numpy as np
import torch

from data import make_dataset, prefix
from diagnostics import (check_capoff, coverage_diagnostics, event_metrics,
                         finite_rate_summary, fit_alpha, history_decomposition,
                         grid_diagnostics, make_oracle_grid, one_step, previous_truth, rollout)
from models import make_model
import diagnostics


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(12345)

    def test_previous_state_alignment(self):
        data = make_dataset("fatigue", 3, 256, 700)
        previous = previous_truth(data)
        np.testing.assert_array_equal(previous[:, 0], data["initial"])
        np.testing.assert_array_equal(previous[:, 1:], data["y"][:, :-1])
        np.testing.assert_array_equal(prefix(data, 128)["x"], data["x"][:, :128])

    def test_grid_and_zero_rate_denominator(self):
        grid = make_oracle_grid()
        self.assertEqual(grid["x"].shape, (1620, 5))
        self.assertTrue(grid["no_true_crossing"].any())
        self.assertTrue((~grid["no_true_crossing"]).any())
        rest = ~grid["active"]
        np.testing.assert_allclose(grid["truth_rate"][rest], 0., atol=1e-14)
        summary = finite_rate_summary(np.ones(5), np.zeros(5), np.ones(5, dtype=bool))
        self.assertIsNone(summary["relative_l1"])
        self.assertEqual(summary["signed_bias"], 1.)
        rest_summary = grid_diagnostics(make_model("state_full", "fatigue"), grid)[0]["rest"]
        self.assertEqual(rest_summary["mean_predicted_rate"], 0.)
        self.assertEqual(rest_summary["mean_true_rate"], 0.)

    def test_alpha_least_squares_and_clipping(self):
        truth = np.array([1., 2., 3.])
        result = fit_alpha(truth * 2, truth, np.ones(3, dtype=bool))
        self.assertAlmostEqual(result["alpha"], .5)
        self.assertAlmostEqual(result["increment_mse_after"], 0.)
        clipped = fit_alpha(truth * .1, truth, np.ones(3, dtype=bool))
        self.assertEqual(clipped["alpha"], 2.)
        self.assertTrue(clipped["clipped"])

    def test_oracle_does_not_reuse_model_history(self):
        model = make_model("state_full", "fatigue")
        data = make_dataset("fatigue", 4, 32, 701)
        previous = previous_truth(data)
        all_at_once = one_step(model, data["x"], previous)
        at_step_9 = one_step(model, data["x"][:, 8], previous[:, 8])
        np.testing.assert_allclose(all_at_once[:, 8], at_step_9, atol=1e-7)

    def test_decomposition_identity_and_masks(self):
        model = make_model("state_gated", "fatigue")
        data = make_dataset("fatigue", 4, 256, 702, mode="low_load")
        predicted = rollout(model, data)
        result, oracle = history_decomposition(model, data, predicted)
        self.assertLess(result["max_decomposition_residual"], 1e-6)
        self.assertEqual([r["horizon"] for r in result["rows"]], [32, 128, 256])
        for row in result["rows"]:
            length = row["horizon"]
            count = int((data["active"][:, :length] & ~data["failed"][:, :length]).sum())
            self.assertEqual(row["local_oracle_bias"]["count"], count)
            self.assertAlmostEqual(row["history_feedback_term"]["mean"],
                                   row["inherited_previous_error"]["mean"] +
                                   row["feedback_increment"]["mean"], places=12)

    def test_capoff_first_event_identity(self):
        model = make_model("state_full", "fatigue")
        # Fixed positive raw head produces early events without needing training.
        with torch.no_grad():
            for p in model.parameters():
                p.zero_()
            model.net[-1].bias.fill_(8.)
        data = make_dataset("fatigue", 8, 128, 703)
        capped = rollout(model, data)
        result, uncapped = check_capoff(model, data, capped)
        self.assertEqual(result["first_crossing_mismatch_count"], 0)
        self.assertEqual(result["capped_crossing_count"], 8)
        self.assertGreater(float(uncapped.max()), 1.)
        for row in uncapped[..., 0]:
            start = np.flatnonzero(row >= .9999)[0]
            np.testing.assert_array_equal(row[start:], row[start])

    def test_occupation_accounting(self):
        train = make_dataset("fatigue", 8, 32, 704)
        low = make_dataset("fatigue", 8, 256, 705, mode="low_load")
        result = coverage_diagnostics(train, low)
        for row in [result["training"], *result["low_load"]]:
            self.assertEqual(sum(map(sum, row["counts"])), row["eligible_pre_state_points"])
            self.assertAlmostEqual(sum(map(sum, row["fraction"])), 1.)

    def test_event_truth_has_zero_error(self):
        data = make_dataset("fatigue", 16, 256, 706)
        result = event_metrics(data, data["y"])
        self.assertEqual(result["state_rmse"], 0.)
        self.assertEqual(result["false_positive_count"], 0)
        self.assertEqual(result["false_negative_count"], 0)
        self.assertEqual(result["event_time_capped_mae_all_steps"], 0.)

    def test_calibrate_only_never_constructs_test_data(self):
        with patch("sys.argv", ["diagnostics.py", "--output", "/unused", "--calibrate-only"]), \
                patch.object(diagnostics, "load_models", return_value=({}, {}, {})), \
                patch.object(diagnostics, "fit_and_lock_calibration", return_value={"rows": []}), \
                patch.object(diagnostics, "sha_file", return_value="locked-hash"), \
                patch.object(diagnostics, "make_dataset", side_effect=AssertionError("test constructed")), \
                patch("builtins.print"):
            diagnostics.main()


if __name__ == "__main__":
    unittest.main()
