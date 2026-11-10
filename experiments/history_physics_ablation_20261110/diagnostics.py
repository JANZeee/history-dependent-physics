"""Frozen fatigue diagnostics; run only after the entire training sweep finishes.

Calibration is fitted/saved on independent low-load validation data before this
entry point constructs any new test data. Oracle-state injection is restricted
to a one-step diagnostic and never used in ordinary autoregressive rollouts.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from data import fatigue_step, make_dataset, prefix
from models import make_model

SEEDS = (101, 202, 303)
STATE_MODELS = ("state_free", "state_gated", "state_positive", "state_full", "state_cubic")
FAIL_THRESHOLD = 1 - 1e-4
STATE_BINS = np.array([0., .2, .35, .5, .65, .8, .95, 1.000001])
STRESS_BINS = np.array([-.01, .01, .4, .5, .6, .7, .8, 1., 1.2, 1.5001])
GRID_AXES = dict(a0=[.08, .14, .20], a=[.20, .35, .50, .65, .80, .95],
                 stress=[0., .3, .4, .5, .6, .7, .8, 1., 1.2, 1.5],
                 coefficient=[.04, .08, .12], cycles=[.5, 1., 1.5])
CALIBRATION_CONFIG = dict(seed=21002, n=512, generated_length=256,
                          fitted_prefix=128, mode="low_load", alpha_clip=[.1, 2.])


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dataset_hash(data):
    return {key: hashlib.sha256(data[key].tobytes()).hexdigest()
            for key in ("x", "y", "initial", "active", "failed")}


def write_locked_json(path, value):
    """Permit deterministic re-entry after interruption, never change a lock."""
    encoded = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if path.exists() and json.loads(path.read_text()) != value:
        raise RuntimeError(f"Refusing to replace different locked content: {path}")
    if not path.exists():
        path.write_text(encoded)


def previous_truth(data):
    return np.concatenate((data["initial"][:, None, :], data["y"][:, :-1]), axis=1)


def first_crossing(mask):
    return np.where(mask.any(axis=1), mask.argmax(axis=1) + 1, mask.shape[1] + 1)


def finite_rate_summary(predicted, truth, mask):
    """Relative L1 is ratio of sums, not unstable mean per-sample percentages."""
    p = np.asarray(predicted, dtype=np.float64)[mask]
    y = np.asarray(truth, dtype=np.float64)[mask]
    if not np.isfinite(p).all() or not np.isfinite(y).all():
        raise FloatingPointError("Nonfinite oracle-rate diagnostic")
    if not len(p):
        return {"count": 0, "signed_bias": None, "mae": None, "rmse": None,
                "mean_predicted_rate": None, "mean_true_rate": None,
                "relative_l1": None, "relative_signed_bias": None}
    e = p - y
    denominator = float(np.abs(y).sum())
    return dict(count=int(len(p)), signed_bias=float(e.mean()),
                mae=float(np.abs(e).mean()), rmse=float(np.sqrt(np.mean(e ** 2))),
                mean_predicted_rate=float(p.mean()), mean_true_rate=float(y.mean()),
                relative_l1=float(np.abs(e).sum() / denominator) if denominator else None,
                relative_signed_bias=float(e.sum() / denominator) if denominator else None)


@torch.inference_mode()
def one_step(model, inputs, states, *, disable_cap=False, rate_scale=1.):
    flat_x = np.asarray(inputs, dtype=np.float32).reshape(-1, 5)
    flat_state = np.asarray(states, dtype=np.float32).reshape(-1, 1)
    predictions = []
    for start in range(0, len(flat_x), 4096):
        p = model.step(torch.from_numpy(flat_x[start:start + 4096]),
                       torch.from_numpy(flat_state[start:start + 4096]),
                       disable_cap=disable_cap, rate_scale=rate_scale)
        predictions.append(p.numpy())
    return np.concatenate(predictions).reshape(np.asarray(states).shape)


@torch.inference_mode()
def rollout(model, data, *, rate_scale=1.):
    output = []
    for start in range(0, len(data["x"]), 64):
        output.append(model(torch.from_numpy(data["x"][start:start + 64]),
                            torch.from_numpy(data["initial"][start:start + 64]),
                            rate_scale=rate_scale).numpy())
    result = np.concatenate(output)
    if not np.isfinite(result).all():
        raise FloatingPointError(f"Nonfinite rollout: {int((~np.isfinite(result)).sum())} state points")
    return result


@torch.inference_mode()
def rollout_capoff_until_crossing(model, data):
    """Uncap until first predicted crossing; then freeze that crossing value.

    This intentionally is not an uncapped post-fracture simulator. It prevents
    irrelevant post-event explosions while testing the first-event identity.
    """
    output = []
    for start in range(0, len(data["x"]), 64):
        x = torch.from_numpy(data["x"][start:start + 64])
        state = torch.from_numpy(data["initial"][start:start + 64]).clone()
        crossed = state >= FAIL_THRESHOLD
        rows = []
        for u in x.unbind(1):
            # Preserve the normal rollout's batch shape and matrix products;
            # subselecting unfinished rows can introduce avoidable roundoff.
            candidate = model.step(u, state, disable_cap=True)
            next_state = torch.where(crossed, state, candidate)
            if not torch.isfinite(next_state).all():
                raise FloatingPointError("Nonfinite pre-first-crossing capoff state")
            state = next_state
            crossed |= state >= FAIL_THRESHOLD
            rows.append(state)
        output.append(torch.stack(rows, 1).numpy())
    return np.concatenate(output)


def make_oracle_grid():
    values = np.asarray(list(itertools.product(*GRID_AXES.values())), dtype=np.float64)
    a0, a, stress, coefficient, cycles = values.T
    valid = a >= a0
    values = values[valid]
    a0, a, stress, coefficient, cycles = values.T
    x = np.stack((stress / 1.5, cycles, coefficient / .12, a0 / .2, np.ones_like(a)), -1)
    state = a[:, None].astype(np.float32)
    # Use exactly the state supplied to the model, so float32 representation
    # cannot masquerade as a nonzero rest increment.
    supplied_a = state[:, 0].astype(np.float64)
    truth = fatigue_step(supplied_a, stress, cycles, coefficient)
    return dict(x=x.astype(np.float32), state=state,
                values=values, truth=truth, truth_rate=(truth - supplied_a) / cycles,
                no_true_crossing=truth < 1., active=stress > 0.)


def grid_diagnostics(model, grid):
    raw_next = one_step(model, grid["x"], grid["state"], disable_cap=True)[:, 0]
    a0, a, stress, coefficient, cycles = grid["values"].T
    predicted_rate = (raw_next.astype(np.float64) - grid["state"][:, 0]) / cycles
    selected = grid["no_true_crossing"]
    active = selected & grid["active"]
    result = dict(
        all_points=len(a), true_crossing_excluded_count=int((~selected).sum()),
        rate_uses_uncapped_model_increment=True,
        active=finite_rate_summary(predicted_rate, grid["truth_rate"], active),
        rest=finite_rate_summary(predicted_rate, grid["truth_rate"], selected & ~grid["active"]),
        by_stress={str(s): finite_rate_summary(predicted_rate, grid["truth_rate"], selected & (stress == s))
                   for s in GRID_AXES["stress"]},
        by_state={str(v): finite_rate_summary(predicted_rate, grid["truth_rate"], active & (a == v))
                  for v in GRID_AXES["a"]},
        by_state_and_stress=[dict(state=float(v), stress=float(s),
            **finite_rate_summary(predicted_rate, grid["truth_rate"], selected & (a == v) & (stress == s)))
            for v in GRID_AXES["a"] for s in GRID_AXES["stress"]],
        false_crossing_count=int((selected & (raw_next >= FAIL_THRESHOLD)).sum()),
        false_crossing_denominator=int(selected.sum()),
    )
    return result, raw_next


def fit_alpha(predicted_delta, true_delta, mask):
    p = np.asarray(predicted_delta, dtype=np.float64)[mask]
    y = np.asarray(true_delta, dtype=np.float64)[mask]
    numerator, denominator = float(np.dot(p, y)), float(np.dot(p, p))
    if not np.isfinite(numerator + denominator) or denominator <= 0:
        raise ValueError("Invalid calibration least-squares denominator")
    raw = numerator / denominator
    alpha = float(np.clip(raw, .1, 2.))
    return dict(alpha=alpha, unclipped_alpha=float(raw), clipped=bool(alpha != raw),
                numerator=numerator, denominator=denominator, fitted_count=len(p),
                increment_mse_before=float(np.mean((p - y) ** 2)),
                increment_mse_after=float(np.mean((alpha * p - y) ** 2)))


def fit_and_lock_calibration(models, outdir, checkpoint_hashes):
    cfg = CALIBRATION_CONFIG
    full = make_dataset("fatigue", cfg["n"], cfg["generated_length"], cfg["seed"], mode=cfg["mode"])
    val = prefix(full, cfg["fitted_prefix"])
    prev = previous_truth(val)
    mask = val["active"] & ~val["failed"]
    true_delta = (val["y"] - prev)[..., 0]
    rows = []
    for seed in SEEDS:
        pred = one_step(models[("state_full", seed)], val["x"], prev, disable_cap=True)
        row = fit_alpha((pred - prev)[..., 0], true_delta, mask)
        rows.append(dict(task="fatigue", seed=seed, model="state_full", **row))
    result = dict(config=cfg, hashes=dataset_hash(val), rows=rows,
                  checkpoint_hashes={str(s): checkpoint_hashes[("state_full", s)] for s in SEEDS},
                  mask="active and true post-step crack < 1; excludes true crossing steps",
                  objective="sum((alpha * uncapped_oracle_predicted_increment - true_increment)^2)",
                  target_is_increment_not_rate=True,
                  implementation_sha256=sha_file(Path(__file__)),
                  test_data_constructed_before_this_lock=False)
    write_locked_json(outdir / "calibration.json", result)
    return result


def scalar_summary(values, mask):
    v = np.asarray(values, dtype=np.float64)[mask]
    return dict(count=int(v.size), mean=float(v.mean()) if v.size else None,
                mae=float(np.abs(v).mean()) if v.size else None,
                rmse=float(np.sqrt(np.mean(v ** 2))) if v.size else None)


def history_decomposition(model, data, predictions):
    previous = previous_truth(data)
    oracle_next = one_step(model, data["x"], previous)
    y, p, o = (arr[..., 0].astype(np.float64) for arr in (data["y"], predictions, oracle_next))
    local = o - y
    feedback = p - o
    predicted_previous = np.concatenate((data["initial"][:, None, :], predictions[:, :-1]), axis=1)
    inherited = predicted_previous[..., 0].astype(np.float64) - previous[..., 0]
    feedback_increment = feedback - inherited
    total = p - y
    residual = float(np.max(np.abs(total - local - feedback)))
    if residual > 1e-6:
        raise AssertionError("One-step decomposition identity failed")
    all_mask = data["active"] & ~data["failed"]
    rates_true = (y - previous[..., 0]) / data["x"][..., 1]
    rates_pred = (o - previous[..., 0]) / data["x"][..., 1]
    results = []
    for horizon in (32, 128, 256):
        mask = all_mask.copy()
        mask[:, horizon:] = False
        out = dict(horizon=horizon, local_oracle_bias=scalar_summary(local, mask),
                   history_feedback_term=scalar_summary(feedback, mask),
                   inherited_previous_error=scalar_summary(inherited, mask),
                   feedback_increment=scalar_summary(feedback_increment, mask),
                   total_state_error=scalar_summary(total, mask),
                   oracle_rate=finite_rate_summary(rates_pred, rates_true, mask), by_previous_state=[])
        for lower, upper in zip(STATE_BINS[:-1], STATE_BINS[1:]):
            selected = mask & (previous[..., 0] >= lower) & (previous[..., 0] < upper)
            out["by_previous_state"].append(dict(lower=float(lower), upper=float(upper),
                local_oracle_bias=scalar_summary(local, selected),
                history_feedback_term=scalar_summary(feedback, selected),
                inherited_previous_error=scalar_summary(inherited, selected),
                feedback_increment=scalar_summary(feedback_increment, selected),
                total_state_error=scalar_summary(total, selected),
                oracle_rate=finite_rate_summary(rates_pred, rates_true, selected)))
        results.append(out)
    return dict(max_decomposition_residual=residual, rows=results,
                definitions={"local_oracle_bias": "g(true_previous,u) - true_next",
                             "history_feedback_term": "g(predicted_previous,u) - g(true_previous,u)",
                             "inherited_previous_error": "predicted_previous - true_previous",
                             "feedback_increment": "history_feedback_term - inherited_previous_error"},
                model_cap_enabled_for_decomposition=True), oracle_next


def occupation(data, length):
    prev = previous_truth(data)[:, :length, 0]
    stress = data["x"][:, :length, 0].astype(np.float64) * 1.5
    valid = prev < 1.
    hist = np.histogram2d(prev[valid], stress[valid], bins=(STATE_BINS, STRESS_BINS))[0].astype(int)
    return dict(length=length, eligible_pre_state_points=int(valid.sum()),
                active_pre_state_points=int((valid & (stress > 0)).sum()), counts=hist.tolist(),
                fraction=(hist / max(hist.sum(), 1)).tolist())


def coverage_diagnostics(train, lowload):
    base = occupation(train, 32)
    train_counts = np.asarray(base["counts"])
    rows = []
    for length in (32, 128, 256):
        entry = occupation(lowload, length)
        counts = np.asarray(entry["counts"])
        denom = int(counts.sum())
        entry.update(test_points_in_zero_train_cells=int(counts[train_counts == 0].sum()),
                     test_points_in_fewer_than_20_train_cells=int(counts[train_counts < 20].sum()),
                     comparison_denominator=denom)
        rows.append(entry)
    return dict(state_bin_edges=STATE_BINS.tolist(), stress_bin_edges=STRESS_BINS.tolist(),
                rest_is_first_stress_bin=True, training=base, low_load=rows,
                note="Marginal 2D coverage, not a proof of full conditional support or causation.")


def event_metrics(data, predicted):
    p, y = predicted[..., 0], data["y"][..., 0]
    horizon = p.shape[1]
    truth_time = first_crossing(data["failed"])
    pred_time = first_crossing(p >= FAIL_THRESHOLD)
    true_failed, pred_failed = truth_time <= horizon, pred_time <= horizon
    surviving = ~true_failed
    fp, fn = int((pred_failed & surviving).sum()), int((~pred_failed & true_failed).sum())
    pre = ~data["failed"]
    e = p.astype(np.float64) - y
    return dict(sequences=len(p), horizon=horizon, state_rmse=float(np.sqrt(np.mean(e ** 2))),
                pre_failure_rmse=float(np.sqrt(np.mean(e[pre] ** 2))) if pre.any() else None,
                false_positive_count=fp, false_positive_denominator=int(surviving.sum()),
                false_positive_rate=fp / int(surviving.sum()) if surviving.any() else None,
                false_negative_count=fn, false_negative_denominator=int(true_failed.sum()),
                false_negative_rate=fn / int(true_failed.sum()) if true_failed.any() else None,
                event_time_capped_mae_all_steps=float(np.abs(pred_time - truth_time).mean()),
                predicted_failure_count=int(pred_failed.sum()), true_failure_count=int(true_failed.sum()))


def check_capoff(model, data, capped):
    uncapped = rollout_capoff_until_crossing(model, data)
    left = first_crossing(capped[..., 0] >= FAIL_THRESHOLD)
    right = first_crossing(uncapped[..., 0] >= FAIL_THRESHOLD)
    mismatches = int((left != right).sum())
    if mismatches:
        raise AssertionError(f"Capping changed {mismatches} first predicted crossings")
    return dict(sequences=len(left), first_crossing_mismatch_count=mismatches,
                capped_crossing_count=int((left <= capped.shape[1]).sum()),
                uncapped_max_saved_state=float(uncapped.max()),
                post_first_crossing_policy="Freeze uncapped crossing value; do not simulate post-fracture growth"), uncapped


def load_models(output):
    complete_path = output / "training_complete.json"
    if not complete_path.exists():
        raise RuntimeError("All training must finish before diagnostics")
    complete = json.loads(complete_path.read_text())
    manifest = json.loads((output / "training_manifest.json").read_text())
    if complete["fingerprint"] != manifest["fingerprint"]:
        raise RuntimeError("Training fingerprint mismatch")
    folder = Path(__file__).resolve().parent
    for name in ("data.py", "models.py"):
        if name in manifest["sources"] and sha_file(folder / name) != manifest["sources"][name]:
            raise RuntimeError(f"Training source changed: {name}")
    models, hashes = {}, {}
    for name in STATE_MODELS:
        for seed in SEEDS:
            path = output / "fatigue" / name / f"seed_{seed}" / "best.pt"
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            meta = checkpoint["metadata"]
            for key, expected in dict(task="fatigue", model=name, seed=seed,
                                      fingerprint=complete["fingerprint"]).items():
                if meta.get(key) != expected:
                    raise RuntimeError(f"Checkpoint metadata mismatch: {path}: {key}")
            model = make_model(name, "fatigue").eval()
            model.load_state_dict(checkpoint["state_dict"])
            models[(name, seed)] = model
            hashes[(name, seed)] = sha_file(path)
    return models, hashes, complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibrate-only", action="store_true",
                        help="Lock validation-only calibration and exit before constructing test data")
    args = parser.parse_args()
    torch.set_num_threads(2)
    models, hashes, complete = load_models(args.output)
    # This call must stay before every make_dataset call using a test seed.
    calibration = fit_and_lock_calibration(models, args.output, hashes)
    calibration_sha = sha_file(args.output / "calibration.json")
    print(json.dumps({"calibration_locked": calibration["rows"]}), flush=True)
    if args.calibrate_only:
        return
    test_created_utc = datetime.now(timezone.utc).isoformat()
    datasets = {"random": make_dataset("fatigue", 512, 256, 21003),
                "low_load": make_dataset("fatigue", 512, 256, 21005, mode="low_load")}
    train = make_dataset("fatigue", 5000, 32, 11001)
    # Reconstruct the actual 32-step training generator call. Generating 256
    # first and slicing would consume the seeded RNG differently.
    training_hashes = json.loads((args.output / "fatigue_data_manifest.json").read_text())["train"]
    if any(dataset_hash(train)[key] != value for key, value in training_hashes.items()):
        raise RuntimeError("Reconstructed training data differs from the trained data")
    grid = make_oracle_grid()
    arrays = args.output / "diagnostic_arrays"
    arrays.mkdir(exist_ok=True)
    np.savez_compressed(arrays / "oracle_grid.npz", **grid)
    result = dict(scope="Frozen post-training local-update and calibration diagnostics; not real-material validation",
                  training_fingerprint=complete["fingerprint"], calibration_sha256=calibration_sha,
                  calibration_file_mtime_ns=(args.output / "calibration.json").stat().st_mtime_ns,
                  test_data_construction_started_utc=test_created_utc,
                  checkpoint_sha256={f"{n}/{s}": h for (n, s), h in hashes.items()},
                  source_sha256={n: sha_file(Path(__file__).parent / n)
                                 for n in ("diagnostics.py", "models.py", "data.py")},
                  test_hashes={k: dataset_hash(v) for k, v in datasets.items()},
                  training_hashes=dataset_hash(train),
                  grid_axes=GRID_AXES, grid=[], decomposition=[], capoff=[], calibration_tradeoff=[],
                  occupation=coverage_diagnostics(train, datasets["low_load"]), errors=[])
    for name in STATE_MODELS:
        for seed in SEEDS:
            model = models[(name, seed)]
            grid_result, grid_raw = grid_diagnostics(model, grid)
            result["grid"].append(dict(model=name, seed=seed, **grid_result))
            np.savez_compressed(arrays / f"grid_{name}_{seed}.npz", raw_next=grid_raw)
            try:
                pred = rollout(model, datasets["low_load"])
                decomposition, oracle_next = history_decomposition(model, datasets["low_load"], pred)
                result["decomposition"].append(dict(model=name, seed=seed, **decomposition))
                np.savez_compressed(arrays / f"low_load_{name}_{seed}.npz", prediction=pred, oracle_next=oracle_next)
            except FloatingPointError as exc:
                result["errors"].append(dict(model=name, seed=seed, diagnostic="low_load_rollout", error=str(exc)))
            if name == "state_full":
                alpha = next(r["alpha"] for r in calibration["rows"] if r["seed"] == seed)
                for condition, full in datasets.items():
                    baseline = rollout(model, full)
                    cap, cap_values = check_capoff(model, full, baseline)
                    result["capoff"].append(dict(seed=seed, condition=condition, **cap))
                    scaled = rollout(model, full, rate_scale=alpha)
                    np.savez_compressed(arrays / f"calibration_{condition}_{seed}.npz",
                                        baseline=baseline, scaled=scaled, capoff_frozen=cap_values,
                                        truth=full["y"], initial=full["initial"], x=full["x"],
                                        active=full["active"], failed=full["failed"])
                    for horizon in (128, 256):
                        short = prefix(full, horizon)
                        for variant, values, multiplier in (("uncalibrated", baseline, 1.), ("calibrated", scaled, alpha)):
                            result["calibration_tradeoff"].append(dict(seed=seed, condition=condition,
                                variant=variant, alpha=multiplier,
                                metrics=event_metrics(short, values[:, :horizon])))
            print(json.dumps({"diagnosed": name, "seed": seed}), flush=True)
    if sha_file(args.output / "calibration.json") != calibration_sha:
        raise RuntimeError("Calibration changed after test construction")
    (args.output / "diagnostics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"diagnostics": str((args.output / "diagnostics.json").resolve()),
                      "errors": result["errors"]}), flush=True)


if __name__ == "__main__":
    main()
