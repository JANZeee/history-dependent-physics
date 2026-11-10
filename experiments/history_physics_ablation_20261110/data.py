"""Auditable synthetic history systems; no real-material calibration is claimed.

Thermal state is excess temperature / 100 K. Fatigue state is normalized
existing crack length, capped at the synthetic failure threshold 1. The latter
uses the separable Paris m=3 stable-growth approximation and an absorbing event;
it does not model crack initiation, plastic folding, closure, or the accelerated
growth regime. Numeric parameters are intentionally synthetic.
"""
from __future__ import annotations

import numpy as np


def thermal_step(excess, power, dt, loss, capacity):
    """Exact step for C*d(excess)/dt = power - loss*excess (SI units)."""
    decay = np.exp(-np.asarray(loss) * np.asarray(dt) / np.asarray(capacity))
    return np.asarray(excess) * decay + np.asarray(power) / np.asarray(loss) * (1 - decay)


def fatigue_step(crack, stress, cycles, coefficient):
    """Exact normalized da/dN=k*s^3*a^(3/2), with a=1 absorbing.

The normalized stress amplitude varies while the critical crack is fixed;
this defines the benchmark, not a complete general fatigue/fracture law.
"""
    crack = np.asarray(crack)
    dose = np.asarray(coefficient) * np.asarray(stress) ** 3 * np.asarray(cycles)
    denominator = crack ** (-0.5) - 0.5 * dose
    next_crack = np.maximum(denominator, 1.0) ** (-2)
    return np.where((dose == 0) | (crack >= 1.0), crack, next_crack)


def make_dataset(task, n, length, seed, mode="random", active_length=32):
    """Return x[N,L,5], y[N,L,1], initial[N,1], active/failed[N,L], meta.

    x[t] describes the action interval ending at y[t]. No current/future state
    is an input. Initial state and episode parameters are public. Arrays use
    float32 (inputs/targets) or bool (masks); integration uses float64.

    Calls with the same task/n/seed/mode and length <=128 are exact prefixes of
    the same generated 128-step episodes. Training/validation/test seeds must
    be distinct. For larger horizons explicitly generate once and slice.

    Modes:
      random: declared training distribution;
      passive_tail: first active_length intervals, then zero power/stress;
      low_load: fatigue-only lower stress within the training amplitude range.
    """
    if task not in {"thermal", "fatigue"}:
        raise ValueError(f"Unknown task: {task}")
    if mode not in {"random", "passive_tail", "low_load"}:
        raise ValueError(f"Unknown mode: {mode}")
    if task == "thermal" and mode == "low_load":
        raise ValueError("low_load is defined only for fatigue")
    if n < 1 or length < 1 or active_length < 0:
        raise ValueError("n and length must be positive; active_length nonnegative")
    rng = np.random.default_rng(seed)
    full_length = max(128, length)
    shape = (n, full_length)
    failed = np.zeros(shape, dtype=bool)
    target = np.empty((*shape, 1), dtype=np.float64)
    if task == "thermal":
        power = rng.uniform(0, 200, shape)
        power *= rng.random(shape) > 0.35
        dt = rng.uniform(0.5, 5.0, shape)
        loss = rng.uniform(0.5, 4.0, (n, 1))
        capacity = rng.uniform(80.0, 120.0, (n, 1))
        if mode == "passive_tail":
            power[:, active_length:] = 0.0
        initial = np.zeros((n, 1), dtype=np.float64)
        state = initial[:, 0].copy()
        for t in range(full_length):
            state = thermal_step(state, power[:, t], dt[:, t], loss[:, 0], capacity[:, 0])
            target[:, t, 0] = state / 100.0
        x = np.stack((power / 200.0, dt / 5.0,
                      np.broadcast_to(loss / 4.0, shape),
                      np.broadcast_to(capacity / 100.0, shape),
                      np.zeros(shape)), axis=-1)
        active = power > 0
        features = ["power_W/200", "dt_s/5", "loss_W_per_K/4",
                    "capacity_J_per_K/100", "initial_excess_K/100"]
        definition = "C*dT/dt=P-H*(T-20); exact constant-input steps; target=(T-20)/100"
        ranges = {"power_W": [0, 200], "power_zero_probability": 0.35,
                  "dt_s": [0.5, 5], "loss_W_per_K": [0.5, 4],
                  "capacity_J_per_K": [80, 120], "initial_temperature_C": 20}
    else:
        stress_high = 0.8 if mode in {"low_load", "passive_tail"} else 1.5
        stress = rng.uniform(0.3, stress_high, shape)
        stress *= rng.random(shape) > 0.25
        cycles = rng.uniform(0.5, 1.5, shape)
        coefficient = rng.uniform(0.04, 0.12, (n, 1))
        initial = rng.uniform(0.08, 0.2, (n, 1))
        if mode == "passive_tail":
            stress[:, active_length:] = 0.0
        state = initial[:, 0].copy()
        for t in range(full_length):
            state = fatigue_step(state, stress[:, t], cycles[:, t], coefficient[:, 0])
            target[:, t, 0] = state
            failed[:, t] = state >= 1.0
        x = np.stack((stress / 1.5, cycles,
                      np.broadcast_to(coefficient / 0.12, shape),
                      np.broadcast_to(initial / 0.2, shape), np.ones(shape)), axis=-1)
        active = stress > 0
        features = ["normalized_stress_amplitude/1.5", "normalized_cycle_count",
                    "growth_coefficient/0.12", "initial_crack/0.2", "constant_one"]
        definition = "da/dN=k*s^3*a^(3/2); exact steps; a>=1 triggers absorbing failure"
        ranges = {"stress_amplitude": [0.3, stress_high], "stress_zero_probability": 0.25,
                  "cycle_count": [0.5, 1.5], "growth_coefficient": [0.04, 0.12],
                  "initial_crack": [0.08, 0.2], "critical_crack": 1.0}
    sl = slice(0, length)
    return {
        "x": np.ascontiguousarray(x[:, sl], dtype=np.float32),
        "y": np.ascontiguousarray(target[:, sl], dtype=np.float32),
        "initial": np.ascontiguousarray(initial, dtype=np.float32),
        "active": np.ascontiguousarray(active[:, sl]),
        "failed": np.ascontiguousarray(failed[:, sl]),
        "meta": {"task": task, "n": n, "length": length, "seed": int(seed),
                 "mode": mode, "active_length": active_length,
                 "features": features, "definition": definition, "ranges": ranges,
                 "current_or_future_state_in_x": False,
                 "ambient_temperature_C": 20 if task == "thermal" else None,
                 "target_scale_to_physical": 100 if task == "thermal" else 1},
    }


def prefix(data, length):
    """Slice already generated episodes without changing their initial conditions."""
    if not 1 <= length <= data["x"].shape[1]:
        raise ValueError("Invalid prefix length")
    return {"x": data["x"][:, :length].copy(), "y": data["y"][:, :length].copy(),
            "initial": data["initial"].copy(), "active": data["active"][:, :length].copy(),
            "failed": data["failed"][:, :length].copy(),
            "meta": {**data["meta"], "length": length}}
