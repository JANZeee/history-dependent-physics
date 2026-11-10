"""Numerical invariants for the frozen synthetic generators (no model training)."""
import unittest

import numpy as np

from data import fatigue_step, make_dataset, prefix, thermal_step


class PhysicsDataTests(unittest.TestCase):
    def test_thermal_closed_form_and_semigroup(self):
        expected = 100 * (1 - np.exp(-0.4))
        self.assertAlmostEqual(float(thermal_step(0., 200., 20., 2., 100.)), expected)
        full = thermal_step(12., 120., 3., 1.5, 100.)
        half = thermal_step(thermal_step(12., 120., 1.5, 1.5, 100.), 120., 1.5, 1.5, 100.)
        self.assertAlmostEqual(float(full), float(half), places=12)

    def test_thermal_rest_recovers_to_ambient(self):
        cold = thermal_step(0., 0., 10., 2., 100.)
        warm = thermal_step(40., 0., 10., 2., 100.)
        self.assertEqual(float(cold), 0.)
        self.assertTrue(0 < warm < 40)
        self.assertAlmostEqual(float(warm), 40 * np.exp(-0.2))

    def test_fatigue_closed_form_and_semigroup(self):
        expected = (0.1 ** -0.5 - 0.5 * 0.08 * 1.2 ** 3 * 2.) ** -2
        full = fatigue_step(0.1, 1.2, 2., 0.08)
        half = fatigue_step(fatigue_step(0.1, 1.2, 1., 0.08), 1.2, 1., 0.08)
        self.assertAlmostEqual(float(full), expected, places=12)
        self.assertAlmostEqual(float(full), float(half), places=12)

    def test_fatigue_no_healing_and_failure_absorbs(self):
        self.assertEqual(float(fatigue_step(0.4, 0., 100., 0.12)), 0.4)
        self.assertEqual(float(fatigue_step(0.4, 1., 0., 0.12)), 0.4)
        self.assertEqual(float(fatigue_step(0.99, 1.5, 1.5, 0.12)), 1.)
        self.assertEqual(float(fatigue_step(1., 0., 100., 0.12)), 1.)

    def test_shapes_finiteness_and_public_features(self):
        for task in ("thermal", "fatigue"):
            d = make_dataset(task, 12, 32, 19)
            self.assertEqual(d["x"].shape, (12, 32, 5))
            self.assertEqual(d["y"].shape, (12, 32, 1))
            self.assertEqual(d["initial"].shape, (12, 1))
            self.assertEqual(d["x"].dtype, np.float32)
            self.assertTrue(np.isfinite(d["y"]).all())
            self.assertFalse(d["meta"]["current_or_future_state_in_x"])
            # Parameter/initial-state channels remain constant within an episode.
            np.testing.assert_array_equal(d["x"][:, :, 2:],
                np.broadcast_to(d["x"][:, :1, 2:], d["x"][:, :, 2:].shape))
            if task == "thermal":
                np.testing.assert_array_equal(d["x"][:, :, 4], 0.)
                prev = d["initial"][:, 0] * 100
                for t in range(32):
                    row = d["x"][:, t]
                    prev = thermal_step(prev, row[:, 0] * 200, row[:, 1] * 5,
                                        row[:, 2] * 4, row[:, 3] * 100)
                    np.testing.assert_allclose(prev / 100, d["y"][:, t, 0], atol=2e-7)
            else:
                np.testing.assert_array_equal(d["x"][:, :, 4], 1.)
                np.testing.assert_allclose(d["x"][:, 0, 3] * .2, d["initial"][:, 0])
                prev = d["initial"][:, 0]
                for t in range(32):
                    row = d["x"][:, t]
                    prev = fatigue_step(prev, row[:, 0] * 1.5, row[:, 1], row[:, 2] * .12)
                    np.testing.assert_allclose(prev, d["y"][:, t, 0], atol=2e-6)

    def test_prefix_and_independent_split(self):
        for task in ("thermal", "fatigue"):
            full = make_dataset(task, 8, 128, 20)
            short = make_dataset(task, 8, 32, 20)
            cut = prefix(full, 32)
            for k in ("x", "y", "initial", "active", "failed"):
                np.testing.assert_array_equal(short[k], cut[k])
            self.assertFalse(np.array_equal(short["x"], make_dataset(task, 8, 32, 21)["x"]))

    def test_passive_tail_invariants(self):
        for task in ("thermal", "fatigue"):
            d = make_dataset(task, 64, 128, 22, "passive_tail")
            self.assertFalse(d["active"][:, 32:].any())
            delta = np.diff(d["y"][:, 31:, 0], axis=1)
            if task == "thermal":
                self.assertTrue((delta <= 0).all())
                self.assertTrue((d["y"] >= 0).all())
            else:
                np.testing.assert_array_equal(delta, 0.)
                self.assertFalse(d["failed"].any())
                self.assertTrue((np.diff(d["y"][:, :, 0], axis=1) >= 0).all())

    def test_fatigue_long_low_load_is_not_all_saturated(self):
        d = make_dataset("fatigue", 128, 128, 23, "low_load")
        self.assertGreater(int((~d["failed"][:, -1]).sum()), 64)
        self.assertTrue((d["y"] <= 1).all())


if __name__ == "__main__":
    unittest.main()
