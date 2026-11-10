"""Locked evaluation; refuses to run before all twelve training runs complete."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from data import make_dataset, prefix
from models import make_model

TOL = 1e-5
FAIL_THRESHOLD = 1 - 1e-4
SEEDS = (101, 202, 303)
MODELS = ("gru", "constrained")
TASKS = ("thermal", "fatigue")


def first_crossing(mask):
    """1-based observation step; right-censored sequences use horizon+1."""
    mask = np.asarray(mask, dtype=bool)
    return np.where(mask.any(axis=1), mask.argmax(axis=1) + 1, mask.shape[1] + 1)


def add_fraction(out, name, numerator_mask, denominator_mask):
    denominator = int(np.asarray(denominator_mask).sum())
    numerator = int(np.asarray(numerator_mask).sum())
    out[name + "_count"] = numerator
    out[name + "_denominator"] = denominator
    out[name] = numerator / denominator if denominator else None


def compute_metrics(task, data, predicted, passive=False):
    p = np.asarray(predicted, dtype=np.float64).squeeze(-1)
    y = np.asarray(data["y"], dtype=np.float64).squeeze(-1)
    initial = np.asarray(data["initial"], dtype=np.float64)
    if p.shape != y.shape or not np.isfinite(p).all():
        raise ValueError("Prediction shape or finite-value check failed")
    n, horizon = p.shape
    scale = 100.0 if task == "thermal" else 1.0
    error = (p - y) * scale
    previous = np.concatenate((initial, p[:, :-1]), axis=1)
    delta = p - previous
    all_steps = np.ones(p.shape, dtype=bool)
    rest = ~np.asarray(data["active"], dtype=bool)
    out = {
        "sequences": n, "horizon_steps": horizon, "state_time_points": n * horizon,
        "state_rmse": float(np.sqrt(np.mean(error ** 2))),
        "final_state_mae": float(np.mean(np.abs(error[:, -1]))),
        "maximum_absolute_state_error": float(np.max(np.abs(error))),
    }
    if task == "thermal":
        add_fraction(out, "passive_increase_fraction", rest & (delta > TOL), rest)
        add_fraction(out, "ambient_undershoot_fraction", p < -TOL, all_steps)
        out["maximum_passive_increase_K"] = float(max(0.0, np.max(delta[rest])) * 100) if rest.any() else None
    else:
        pre_failure = ~np.asarray(data["failed"], dtype=bool)
        out["pre_failure_state_time_points"] = int(pre_failure.sum())
        out["pre_failure_rmse"] = float(np.sqrt(np.mean(error[pre_failure] ** 2))) if pre_failure.any() else None
        add_fraction(out, "damage_decrease_fraction", delta < -TOL, all_steps)
        add_fraction(out, "rest_drift_fraction", rest & (np.abs(delta) > TOL), rest)
        add_fraction(out, "negative_crack_fraction", p < -TOL, all_steps)
        add_fraction(out, "above_failure_state_fraction", p > 1 + TOL, all_steps)
        true_t = first_crossing(np.asarray(data["failed"]))
        pred_t = first_crossing(p >= FAIL_THRESHOLD)
        true_failed = true_t <= horizon
        pred_failed = pred_t <= horizon
        matched = true_failed & pred_failed
        time_error = np.abs(true_t - pred_t).astype(float)
        out.update(
            true_failure_fraction=float(true_failed.mean()),
            predicted_failure_fraction=float(pred_failed.mean()),
            event_accuracy=float(np.mean(true_failed == pred_failed)),
            event_time_capped_mae_all_steps=float(time_error.mean()),
            event_time_true_failure_mae_including_missed_steps=float(time_error[true_failed].mean()) if true_failed.any() else None,
            event_time_matched_failure_mae_steps=float(time_error[matched].mean()) if matched.any() else None,
            matched_failure_time_count=int(matched.sum()),
        )
        add_fraction(out, "false_positive_rate", pred_failed & ~true_failed, ~true_failed)
        add_fraction(out, "false_negative_rate", ~pred_failed & true_failed, true_failed)
        add_fraction(out, "matched_failure_coverage", matched, true_failed)
        after_predicted_failure = np.arange(1, horizon + 1)[None, :] > pred_t[:, None]
        add_fraction(out, "post_failure_return_fraction",
                     after_predicted_failure & (p < FAIL_THRESHOLD - TOL), after_predicted_failure)
        true_post_failure = np.asarray(data["failed"], dtype=bool)
        add_fraction(out, "true_failed_state_undershoot_fraction",
                     true_post_failure & (p < 1 - TOL), true_post_failure)
    if passive:
        start = 32
        tail = error[:, start:]
        own_change = (p[:, -1] - p[:, start - 1]) * scale
        out.update(
            tail_state_rmse=float(np.sqrt(np.mean(tail ** 2))),
            tail_final_mae=float(np.mean(np.abs(error[:, -1]))),
            tail_own_state_final_change_mean=float(own_change.mean()),
            tail_own_state_final_change_abs_mean=float(np.abs(own_change).mean()),
        )
        if task == "thermal":
            out["tail_cooling_change_mae_K"] = float(np.mean(np.abs(own_change - (y[:, -1] - y[:, start - 1]) * scale)))
            tail_mask = np.zeros_like(all_steps)
            tail_mask[:, start:] = True
            add_fraction(out, "tail_passive_increase_fraction", tail_mask & (delta > TOL), tail_mask)
            add_fraction(out, "tail_ambient_undershoot_fraction", tail_mask & (p < -TOL), tail_mask)
        else:
            survivors = y[:, start - 1] < 1.0
            out["tail_initial_survivor_count"] = int(survivors.sum())
            out["tail_survivor_rmse"] = float(np.sqrt(np.mean(tail[survivors] ** 2))) if survivors.any() else None
            out["tail_survivor_own_drift_abs_mean"] = float(np.mean(np.abs(own_change[survivors]))) if survivors.any() else None
            tail_mask = np.zeros_like(all_steps)
            tail_mask[:, start:] = True
            add_fraction(out, "tail_damage_decrease_fraction", tail_mask & (delta < -TOL), tail_mask)
            add_fraction(out, "tail_rest_drift_fraction", tail_mask & (np.abs(delta) > TOL), tail_mask)
    return out


@torch.inference_mode()
def predict(model, dataset):
    rows = []
    model.eval()
    for start in range(0, len(dataset["x"]), 64):
        x = torch.from_numpy(dataset["x"][start:start + 64])
        initial = torch.from_numpy(dataset["initial"][start:start + 64])
        rows.append(model(x, initial).numpy())
    result = np.concatenate(rows)
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite test prediction")
    return result


def summarize(rows):
    summary = []
    keys = sorted({(r["task"], r["model"], r["condition"]) for r in rows})
    for task, name, condition in keys:
        members = sorted([r for r in rows if (r["task"], r["model"], r["condition"]) == (task, name, condition)],
                         key=lambda r: r["seed"])
        assert sorted(r["seed"] for r in members) == list(SEEDS)
        aggregate = {}
        for metric in members[0]["metrics"]:
            values = [r["metrics"][metric] for r in members]
            finite = [v for v in values if v is not None]
            aggregate[metric] = {"mean": float(np.mean(finite)) if finite else None,
                                 "sample_sd": float(np.std(finite, ddof=1)) if len(finite) > 1 else None,
                                 "valid_seed_count": len(finite), "per_seed": dict(zip(map(str, SEEDS), values))}
        summary.append({"task": task, "model": name, "condition": condition, "metrics": aggregate})
    return summary


def failure_examples(datasets, predictions, outdir):
    cases = []
    for task in TASKS:
        for seed in SEEDS:
            for selected_model in MODELS:
                selectors = [("worst_long_rmse", "random_128"), ("worst_passive_tail_rmse", "passive_tail_128")]
                if task == "fatigue":
                    selectors.append(("worst_event_time", "random_128"))
                for criterion, condition in selectors:
                    d = datasets[(task, condition)]
                    p = predictions[(task, selected_model, seed, condition)]
                    y = d["y"]
                    if criterion == "worst_event_time":
                        scores = np.abs(first_crossing(d["failed"]) - first_crossing(p[..., 0] >= FAIL_THRESHOLD))
                    else:
                        start = 32 if "passive" in criterion else 0
                        scores = np.sqrt(np.mean((p[:, start:] - y[:, start:]) ** 2, axis=(1, 2)))
                        if task == "thermal":
                            scores *= 100
                    index = int(np.argmax(scores))
                    case_id = f"{task}_{seed}_{selected_model}_{criterion}_sample{index}"
                    filename = outdir / "failure_cases" / f"{case_id}.npz"
                    filename.parent.mkdir(exist_ok=True)
                    np.savez_compressed(filename, x=d["x"][index], initial=d["initial"][index], truth=y[index],
                                        gru=predictions[(task, "gru", seed, condition)][index],
                                        constrained=predictions[(task, "constrained", seed, condition)][index])
                    cases.append({"id": case_id, "task": task, "seed": seed, "selected_model": selected_model,
                                  "criterion": criterion, "condition": condition, "sample_index": index,
                                  "score": float(scores[index]), "file": str(filename.resolve())})
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/history_physics_benchmark_20261110"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    complete = args.output / "training_complete.json"
    training = json.loads(complete.read_text())
    manifest = json.loads((args.output / "training_manifest.json").read_text())
    source_dir = Path(__file__).resolve().parent
    if training["fingerprint"] != manifest["fingerprint"]:
        raise RuntimeError("Training manifest fingerprint mismatch")
    for name, expected_sha in manifest["sources"].items():
        if hashlib.sha256((source_dir / name).read_bytes()).hexdigest() != expected_sha:
            raise RuntimeError(f"Training source changed after locking: {name}")
    runs = training["runs"]
    expected = {(t, m, s) for t in TASKS for m in MODELS for s in SEEDS}
    if len(runs) != 12 or {(r["task"], r["model"], r["seed"]) for r in runs} != expected:
        raise RuntimeError("All 12 specified training runs must complete before evaluation")
    start = time.perf_counter()
    datasets = {}
    for task in TASKS:
        full = make_dataset(task, 512, 128, 11003)
        for length in (32, 64, 128):
            datasets[(task, f"random_{length}")] = prefix(full, length)
        datasets[(task, "passive_tail_128")] = make_dataset(task, 512, 128, 11004, mode="passive_tail")
        if task == "fatigue":
            datasets[(task, "low_load_128")] = make_dataset(task, 512, 128, 11005, mode="low_load")
    predictions, rows = {}, []
    pred_dir = args.output / "predictions"
    pred_dir.mkdir(exist_ok=True)
    for run in runs:
        task, name, seed = run["task"], run["model"], run["seed"]
        model = make_model(name, task)
        ckpt = torch.load(run["checkpoint"], map_location="cpu", weights_only=False)
        expected_meta = dict(task=task, model=name, seed=seed, fingerprint=training["fingerprint"])
        if any(ckpt["metadata"].get(k) != v for k, v in expected_meta.items()):
            raise RuntimeError("Checkpoint metadata does not match the locked training run")
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        # Predict the 128-step random trajectory once; shorter results are prefixes.
        cache = {"random": predict(model, datasets[(task, "random_128")])}
        for (data_task, condition), d in datasets.items():
            if data_task != task:
                continue
            p = cache["random"][:, :d["x"].shape[1]].copy() if condition.startswith("random_") else predict(model, d)
            predictions[(task, name, seed, condition)] = p
            metrics = compute_metrics(task, d, p, passive=condition.startswith("passive_"))
            filename = pred_dir / f"{task}_{name}_seed{seed}_{condition}.npz"
            np.savez_compressed(filename, prediction=p, truth=d["y"], x=d["x"], initial=d["initial"],
                                active=d["active"], failed=d["failed"])
            rows.append({"task": task, "model": name, "seed": seed, "condition": condition,
                         "parameters": sum(v.numel() for v in model.parameters()),
                         "checkpoint": run["checkpoint"], "predictions": str(filename.resolve()), "metrics": metrics})
        print(json.dumps({"evaluated": task, "model": name, "seed": seed}), flush=True)
    cases = failure_examples(datasets, predictions, args.output)
    result = {
        "scope": "Fixed-parameter synthetic numerical history experiment; no visual or real-material claim",
        "failure_threshold": FAIL_THRESHOLD, "violation_tolerance_normalized": TOL,
        "state_units": {"thermal": "kelvin above ambient for error metrics", "fatigue": "critical crack length units"},
        "notes": ["Constrained-model sign, passivity and absorption constraints are structural guarantees, not learned discoveries.",
                  "No-event time is horizon+1; event times are observation-step indices, not continuous physical time.",
                  "MSE checkpoints selected only on independent ID validation; test set and 3 seeds are fixed.",
                  "Counts and denominators accompany violation fractions; undefined rates are null, not zero."],
        "training_complete_sha256": hashlib.sha256(complete.read_bytes()).hexdigest(),
        "evaluation_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "test_datasets": {f"{task}/{condition}": {
            "meta": d["meta"], "sha256": {k: hashlib.sha256(d[k].tobytes()).hexdigest()
                                          for k in ("x", "y", "initial")}}
            for (task, condition), d in datasets.items()},
        "training_runs": runs, "runs": rows, "aggregate": summarize(rows), "failure_cases": cases,
        "evaluation_seconds": time.perf_counter() - start,
    }
    (args.output / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"evaluation": str((args.output / "evaluation.json").resolve()),
                      "seconds": result["evaluation_seconds"], "failure_examples": len(cases)}), flush=True)


if __name__ == "__main__":
    main()
