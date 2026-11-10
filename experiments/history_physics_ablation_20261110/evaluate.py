"""Fixed independent tests, paired factorial effects and nested contrasts."""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from data import make_dataset, prefix
from models import make_model, TASK_MODELS, GRU_NAMES
from metrics import compute_metrics, first_crossing, FAIL_THRESHOLD, TOL

SEEDS = (101, 202, 303)
EFFECT_METRICS = ("state_rmse", "final_state_mae", "pre_failure_rmse", "tail_state_rmse",
                  "false_positive_rate", "false_negative_rate", "event_time_capped_mae_all_steps",
                  "damage_decrease_fraction", "rest_drift_fraction", "passive_increase_fraction")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_frozen_sources(output, training):
    manifest = json.loads((output / "training_manifest.json").read_text())
    expected = hashlib.sha256(json.dumps({"config": manifest["config"], "sources": manifest["sources"]},
                                        sort_keys=True).encode()).hexdigest()
    if manifest["fingerprint"] != expected or training["fingerprint"] != expected:
        raise RuntimeError("Training manifest/completion fingerprint mismatch")
    folder = Path(__file__).resolve().parent
    for name, sha in manifest["sources"].items():
        if digest(folder / name) != sha:
            raise RuntimeError(f"Frozen source changed: {name}")
    return manifest


def three_seed_summary(values):
    valid = [v for v in values if v is not None and np.isfinite(v)]
    return {"mean": float(np.mean(valid)) if len(valid) == 3 else None,
            "sample_sd": float(np.std(valid, ddof=1)) if len(valid) == 3 else None,
            "valid_seed_count": len(valid), "per_seed": dict(zip(map(str, SEEDS), values))}


def summarize(rows):
    aggregate = []
    keys = sorted({(r["task"], r["model"], r["condition"]) for r in rows})
    for task, model, condition in keys:
        members = sorted([r for r in rows if (r["task"], r["model"], r["condition"]) == (task, model, condition)],
                         key=lambda r: r["seed"])
        assert [r["seed"] for r in members] == list(SEEDS)
        names = sorted({k for r in members if r["valid"] for k in r["metrics"]})
        aggregate.append({"task": task, "model": model, "condition": condition,
                          "valid_run_count": sum(r["valid"] for r in members),
                          "all_seeds_valid": all(r["valid"] for r in members),
                          "metrics": {name: three_seed_summary([r["metrics"].get(name) if r["valid"] else None
                                                               for r in members]) for name in names}})
    return aggregate


def paired_effects(rows):
    lookup = {(r["task"], r["model"], r["seed"], r["condition"]): r for r in rows}
    factorial, nested = [], []
    conditions = sorted({(r["task"], r["condition"]) for r in rows})
    factors = {
        "state_main": {"gru_state": .5, "gru_plain": -.5, "gru_state_projected": .5, "gru_projected": -.5},
        "projection_main": {"gru_projected": .5, "gru_plain": -.5, "gru_state_projected": .5, "gru_state": -.5},
        "interaction": {"gru_state_projected": 1., "gru_state": -1., "gru_projected": -1., "gru_plain": 1.},
        "state_when_unprojected": {"gru_state": 1., "gru_plain": -1.},
        "state_when_projected": {"gru_state_projected": 1., "gru_projected": -1.},
        "projection_without_state_feedback": {"gru_projected": 1., "gru_plain": -1.},
        "projection_with_state_feedback": {"gru_state_projected": 1., "gru_state": -1.},
    }
    for task, condition in conditions:
        for metric in EFFECT_METRICS:
            if not any(metric in (r["metrics"] or {}) for r in rows if r["task"] == task and r["condition"] == condition):
                continue
            for effect, weights in factors.items():
                values = []
                for seed in SEEDS:
                    members = [lookup[(task, m, seed, condition)] for m in weights]
                    available = all(r["valid"] and r["metrics"].get(metric) is not None for r in members)
                    values.append(sum(weights[r["model"]] * r["metrics"][metric] for r in members) if available else None)
                factorial.append({"task": task, "condition": condition, "metric": metric, "effect": effect,
                                  "weights": weights, **three_seed_summary(values)})
            comparisons = [("state_free", "state_full")] if task == "thermal" else [
                ("state_free", "state_gated"), ("state_gated", "state_positive"),
                ("state_positive", "state_full"), ("state_full", "state_cubic")]
            for before, after in comparisons:
                values = []
                for seed in SEEDS:
                    a, b = lookup[(task, before, seed, condition)], lookup[(task, after, seed, condition)]
                    okay = a["valid"] and b["valid"] and a["metrics"].get(metric) is not None and b["metrics"].get(metric) is not None
                    values.append(b["metrics"][metric] - a["metrics"][metric] if okay else None)
                nested.append({"task": task, "condition": condition, "metric": metric, "before": before,
                               "after": after, "difference": "after-minus-before", **three_seed_summary(values)})
    return {"interpretation": "Signed paired differences; negative means lower error/rate. No p-values or 3-seed significance claim.",
            "thermal_nested_limit": "state_full-minus-state_free changes the update form jointly; it is not a single-constraint causal contrast.",
            "factorial": factorial, "nested": nested}


@torch.inference_mode()
def predict(model, dataset):
    results = []
    for start in range(0, len(dataset["x"]), 64):
        results.append(model(torch.from_numpy(dataset["x"][start:start + 64]),
                             torch.from_numpy(dataset["initial"][start:start + 64])).numpy())
    return np.concatenate(results)


def select_cases(task, model, seed, condition, data, p, outdir):
    if not np.isfinite(p).all() or condition not in ("random_256", "passive_tail_256", "low_load_256"):
        return []
    items = []
    if condition != "low_load_256":
        start = 32 if condition.startswith("passive") else 0
        scores = np.sqrt(np.mean((p[:, start:].astype(float) - data["y"][:, start:]) ** 2, axis=(1, 2)))
        if task == "thermal":
            scores *= 100
        chosen = int(np.argmax(scores))
        items.append(("worst_tail_rmse" if start else "worst_long_rmse", chosen, float(scores[chosen])))
    else:
        true_time = first_crossing(data["failed"])
        pred_time = first_crossing(p[..., 0] >= FAIL_THRESHOLD)
        eligible = np.flatnonzero((true_time > 256) & (pred_time <= 256))
        if len(eligible):
            chosen = int(eligible[np.argmin(pred_time[eligible])])
            items.append(("earliest_low_load_false_positive", chosen, float(pred_time[chosen])))
    records = []
    for criterion, index, score in items:
        case_id = f"{task}_{model}_seed{seed}_{condition}_{criterion}_sample{index}"
        filename = outdir / "failure_cases" / f"{case_id}.npz"
        filename.parent.mkdir(exist_ok=True)
        np.savez_compressed(filename, x=data["x"][index], truth=data["y"][index],
                            initial=data["initial"][index], prediction=p[index])
        records.append({"id": case_id, "task": task, "model": model, "seed": seed,
                        "condition": condition, "criterion": criterion, "sample_index": index,
                        "score": score, "file": str(filename.resolve())})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/history_physics_ablation_20261110"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    complete = args.output / "training_complete.json"
    training = json.loads(complete.read_text())
    manifest = verify_frozen_sources(args.output, training)
    runs = training["runs"]
    expected = {(t, m, s) for t in TASK_MODELS for m in TASK_MODELS[t] for s in SEEDS}
    if len(runs) != 45 or {(r["task"], r["model"], r["seed"]) for r in runs} != expected:
        raise RuntimeError("All 45 fixed training runs must finish before any independent test data are constructed")
    source_folder = Path(__file__).resolve().parent
    source_hashes = {name: digest(source_folder / name)
                     for name in ("evaluate.py", "metrics.py", "diagnostics.py", "plot_results.py")}
    source_path = args.output / "evaluation_sources.json"
    if source_path.exists() and json.loads(source_path.read_text()) != source_hashes:
        raise RuntimeError("Evaluation source freeze already exists with different hashes")
    source_path.write_text(json.dumps(source_hashes, indent=2) + "\n")
    calibration_path = args.output / "calibration.json"
    if not calibration_path.exists():
        raise RuntimeError("Run diagnostics --calibrate-only before constructing any independent test data")
    calibration = json.loads(calibration_path.read_text())
    if calibration["implementation_sha256"] != source_hashes["diagnostics.py"]:
        raise RuntimeError("Calibration implementation hash mismatch")
    expected_calibration = {"seed": 21002, "n": 512, "generated_length": 256,
                            "fitted_prefix": 128, "mode": "low_load", "alpha_clip": [.1, 2.]}
    if calibration["config"] != expected_calibration or calibration["test_data_constructed_before_this_lock"]:
        raise RuntimeError("Calibration configuration or before-test lock mismatch")
    calibration_records = calibration["rows"]
    if sorted(r["seed"] for r in calibration_records) != list(SEEDS):
        raise RuntimeError("Frozen calibration must contain exactly the three prescribed seeds")
    if not all(np.isfinite(r["alpha"]) and .1 <= r["alpha"] <= 2 for r in calibration_records):
        raise RuntimeError("Invalid frozen calibration factor")
    if not all(r["task"] == "fatigue" and r["model"] == "state_full" for r in calibration_records):
        raise RuntimeError("Calibration model/task mismatch")
    for run in runs:
        if run["task"] == "fatigue" and run["model"] == "state_full":
            if digest(Path(run["checkpoint"])) != calibration["checkpoint_hashes"][str(run["seed"])]:
                raise RuntimeError("Calibration checkpoint hash mismatch")
    datasets = {}
    for task in TASK_MODELS:
        random = make_dataset(task, 512, 256, 21003)
        passive = make_dataset(task, 512, 256, 21004, mode="passive_tail")
        for length in (32, 64, 128, 256):
            datasets[(task, f"random_{length}")] = prefix(random, length)
        for length in (128, 256):
            datasets[(task, f"passive_tail_{length}")] = prefix(passive, length)
        if task == "fatigue":
            low = make_dataset(task, 512, 256, 21005, mode="low_load")
            for length in (128, 256):
                datasets[(task, f"low_load_{length}")] = prefix(low, length)
    dataset_manifest = [{"task": task, "condition": condition, "meta": data["meta"],
                         "arrays": {k: {"sha256": hashlib.sha256(data[k].tobytes()).hexdigest(),
                                        "shape": list(data[k].shape), "dtype": str(data[k].dtype)}
                                    for k in ("x", "y", "initial", "active", "failed")}}
                        for (task, condition), data in datasets.items()]
    (args.output / "test_data_manifest.json").write_text(json.dumps(dataset_manifest, indent=2) + "\n")
    start = time.perf_counter()
    prediction_dir = args.output / "predictions"
    prediction_dir.mkdir(exist_ok=True)
    rows, cases = [], []
    for run in runs:
        task, name, seed = run["task"], run["model"], run["seed"]
        load_error, cache = None, {}
        try:
            if run["status"] != "complete":
                raise RuntimeError(f"Training did not complete its fixed budget: {run.get('failure', run['status'])}")
            model = make_model(name, task).eval()
            ckpt = torch.load(run["checkpoint"], map_location="cpu", weights_only=False)
            for field, value in (("task", task), ("model", name), ("seed", seed),
                                 ("fingerprint", manifest["fingerprint"]), ("status", "complete")):
                if ckpt["metadata"].get(field) != value or run.get(field) != value:
                    raise RuntimeError(f"Checkpoint/run metadata mismatch: {field}")
            model.load_state_dict(ckpt["state_dict"])
            parameters = sum(v.numel() for v in model.parameters())
        except Exception as exc:
            load_error = f"checkpoint/model load failed: {type(exc).__name__}: {exc}"
            parameters = None
        for (data_task, condition), data in datasets.items():
            if data_task != task:
                continue
            family = condition.rsplit("_", 1)[0]
            invalid = load_error
            if not invalid and family not in cache:
                try:
                    cache[family] = predict(model, datasets[(task, f"{family}_256")])
                except Exception as exc:
                    cache[family] = f"prediction failed: {type(exc).__name__}: {exc}"
            if not invalid and isinstance(cache[family], str):
                invalid = cache[family]
            p = None if invalid else cache[family][:, :data["x"].shape[1]].copy()
            filename = prediction_dir / f"{task}_{name}_seed{seed}_{condition}.npz"
            if p is not None:
                np.savez_compressed(filename, prediction=p, truth=data["y"], x=data["x"],
                                    initial=data["initial"], active=data["active"], failed=data["failed"])
                bad = int((~np.isfinite(p)).sum())
                if bad:
                    invalid = f"nonfinite prediction entries: {bad}/{p.size}"
            metrics = None if invalid else compute_metrics(task, data, p, passive=condition.startswith("passive_"))
            rows.append({"task": task, "model": name, "seed": seed, "condition": condition,
                         "parameters": parameters, "checkpoint": run.get("checkpoint"),
                         "predictions": str(filename.resolve()) if p is not None else None,
                         "valid": invalid is None, "invalid_reason": invalid, "metrics": metrics})
            if invalid is None:
                cases.extend(select_cases(task, name, seed, condition, data, p, args.output))
        print(json.dumps({"evaluated": task, "model": name, "seed": seed, "load_error": load_error}), flush=True)
    effects = paired_effects(rows)
    for name, sha in source_hashes.items():
        if digest(source_folder / name) != sha:
            raise RuntimeError(f"Evaluation source changed during test: {name}")
    result = {"scope": "Independent synthetic numerical state-feedback and constraint ablation; no visual/real-material claim",
              "seeds": list(SEEDS), "failure_threshold": FAIL_THRESHOLD, "violation_tolerance_normalized": TOL,
              "training_complete_sha256": hashlib.sha256(complete.read_bytes()).hexdigest(),
              "training_fingerprint": manifest["fingerprint"],
              "evaluation_source_sha256": digest(Path(__file__)),
              "evaluation_sources": source_hashes,
              "calibration_sha256_before_test_construction": digest(calibration_path),
              "test_datasets": dataset_manifest,
              "training_runs": runs, "runs": rows, "aggregate": summarize(rows), "failure_cases": cases,
              "evaluation_seconds": time.perf_counter() - start,
              "invalid_run_conditions": sum(not r["valid"] for r in rows),
              "notes": ["Nonfinite and failed predictions are retained as invalid; not silently excluded.",
                        "Aggregate mean/SD are null unless all 3 seed metrics are defined and valid.",
                        "Absence of hard-coded constraint violations is not evidence that those constraints were learned.",
                        "Censored event-time index is horizon+1; event times use observation steps."]}
    (args.output / "evaluation.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (args.output / "paired_effects.json").write_text(json.dumps(effects, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"evaluation_seconds": result["evaluation_seconds"], "invalid_run_conditions": result["invalid_run_conditions"]}), flush=True)


if __name__ == "__main__":
    main()
