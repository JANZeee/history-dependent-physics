"""Independently audit saved evaluation arrays without rerunning model inference.

This script deliberately does not import the original evaluation functions.
It recomputes state/event metrics, verifies aggregate statistics and validation
checkpoint selection, and checks matched data/minibatch identities.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def crossing(rows, threshold=None):
    """Independent loop implementation; absent events are censored at L+1."""
    times = []
    for row in rows:
        positions = np.flatnonzero(row if threshold is None else row >= threshold)
        times.append(int(positions[0] + 1) if len(positions) else len(row) + 1)
    return np.asarray(times)


def audit(output):
    started = time.perf_counter()
    evaluation_path = output / "evaluation.json"
    evaluation = json.loads(evaluation_path.read_text())
    complete_path = output / "training_complete.json"
    training = json.loads(complete_path.read_text())["runs"]
    issues, event_counts, batch_pairs = [], [], []
    reference_hashes = {}
    numeric_checks = 0
    test_identity_checks = 0

    def check(label, recomputed, recorded):
        nonlocal numeric_checks
        numeric_checks += 1
        if recomputed is None or recorded is None:
            okay = recomputed is None and recorded is None
        else:
            okay = np.isclose(recomputed, recorded, rtol=1e-10, atol=1e-12)
        if not okay:
            issues.append({"check": label,
                           "recomputed": recomputed.item() if isinstance(recomputed, np.generic) else recomputed,
                           "recorded": recorded})

    for row in evaluation["runs"]:
        with np.load(row["predictions"]) as data:
            prediction = data["prediction"][..., 0].astype(np.float64)
            truth = data["truth"][..., 0].astype(np.float64)
            task, condition, model, seed = (row[k] for k in ("task", "condition", "model", "seed"))
            label = f"{task}/{model}/{seed}/{condition}"
            metrics = row["metrics"]
            _, horizon = truth.shape
            scale = 100 if task == "thermal" else 1
            error = (prediction - truth) * scale
            check(label + "/rmse", np.sqrt(np.mean(error ** 2)), metrics["state_rmse"])
            check(label + "/final_mae", np.abs(error[:, -1]).mean(), metrics["final_state_mae"])
            for field in ("truth", "x", "initial", "active", "failed"):
                value_hash = hashlib.sha256(data[field].tobytes()).hexdigest()
                identity_key = (task, condition, field)
                if identity_key in reference_hashes:
                    test_identity_checks += 1
                    if reference_hashes[identity_key] != value_hash:
                        issues.append({"check": label + "/test_data_identity", "field": field})
                reference_hashes[identity_key] = value_hash
            previous = np.concatenate([data["initial"].astype(np.float64), prediction[:, :-1]], axis=1)
            delta = prediction - previous
            if task == "fatigue":
                truth_event = crossing(data["failed"])
                predicted_event = crossing(prediction, 1 - 1e-4)
                truth_failed = truth_event <= horizon
                predicted_failed = predicted_event <= horizon
                fp = int((predicted_failed & ~truth_failed).sum())
                fn = int((~predicted_failed & truth_failed).sum())
                negative_count, positive_count = int((~truth_failed).sum()), int(truth_failed.sum())
                capped_time_mae = float(np.abs(truth_event - predicted_event).mean())
                check(label + "/fp_count", fp, metrics["false_positive_rate_count"])
                check(label + "/fp_denominator", negative_count, metrics["false_positive_rate_denominator"])
                check(label + "/fp_rate", fp / negative_count if negative_count else None, metrics["false_positive_rate"])
                check(label + "/fn_count", fn, metrics["false_negative_rate_count"])
                check(label + "/fn_denominator", positive_count, metrics["false_negative_rate_denominator"])
                check(label + "/fn_rate", fn / positive_count if positive_count else None, metrics["false_negative_rate"])
                check(label + "/event_mae", capped_time_mae, metrics["event_time_capped_mae_all_steps"])
                check(label + "/failure_fraction", truth_failed.mean(), metrics["true_failure_fraction"])
                alive = ~data["failed"]
                check(label + "/pre_failure_rmse", np.sqrt(np.mean(error[alive] ** 2)), metrics["pre_failure_rmse"])
                if condition in ("random_128", "low_load_128"):
                    event_counts.append({"task": task, "model": model, "seed": seed,
                                         "condition": condition, "false_positives": fp,
                                         "true_nonfailure_count": negative_count,
                                         "false_negatives": fn, "true_failure_count": positive_count,
                                         "capped_event_time_mae_steps": capped_time_mae})
            if condition == "passive_tail_128":
                tail = error[:, 32:]
                check(label + "/tail_rmse", np.sqrt((tail ** 2).mean()), metrics["tail_state_rmse"])
                own_change = (prediction[:, -1] - prediction[:, 31]) * scale
                check(label + "/tail_own_abs_change", np.abs(own_change).mean(), metrics["tail_own_state_final_change_abs_mean"])
                if task == "thermal":
                    check(label + "/tail_increase_fraction", (delta[:, 32:] > 1e-5).mean(), metrics["tail_passive_increase_fraction"])
                else:
                    check(label + "/tail_drift_fraction", (np.abs(delta[:, 32:]) > 1e-5).mean(), metrics["tail_rest_drift_fraction"])

    for aggregate in evaluation["aggregate"]:
        members = [row for row in evaluation["runs"]
                   if all(row[key] == aggregate[key] for key in ("task", "model", "condition"))]
        for metric, recorded in aggregate["metrics"].items():
            values = [row["metrics"][metric] for row in members if row["metrics"][metric] is not None]
            check("aggregate/" + metric, np.mean(values) if values else None, recorded["mean"])
            check("aggregate_sd/" + metric, np.std(values, ddof=1) if len(values) > 1 else None, recorded["sample_sd"])

    for task in ("thermal", "fatigue"):
        for seed in (101, 202, 303):
            pair = [row for row in training if row["task"] == task and row["seed"] == seed]
            hashes = {row["model"]: row["schedule_sha256"] for row in pair}
            matched = set(hashes) == {"gru", "constrained"} and len(set(hashes.values())) == 1
            batch_pairs.append({"task": task, "seed": seed, "schedule_sha256": hashes, "matched": matched})
            if not matched:
                issues.append({"check": "paired_minibatch_hash", "task": task, "seed": seed})
            for row in pair:
                trace = json.loads((Path(row["checkpoint"]).parent / "trace.json").read_text())
                selected = min(trace, key=lambda entry: entry["val_mse"])
                check("best_step", selected["step"], row["best_step"])
                check("best_mse", selected["val_mse"], row["best_val_mse"])

    result = {
        "status": "pass" if not issues else "fail",
        "scope": "Independent array-based recomputation; no model retraining or inference",
        "numeric_checks": numeric_checks, "test_array_identity_checks": test_identity_checks,
        "paired_minibatch_checks": len(batch_pairs),
        "total_checks": numeric_checks + test_identity_checks + len(batch_pairs),
        "training_runs": len(training), "prediction_files": len(evaluation["runs"]),
        "saved_failure_cases": len(evaluation["failure_cases"]),
        "numeric_tolerances": {"rtol": 1e-10, "atol": 1e-12},
        "paired_minibatch_hashes": batch_pairs,
        "recomputed_fatigue_event_counts": event_counts,
        "evaluation_sha256": sha256(evaluation_path),
        "training_complete_sha256": sha256(complete_path),
        "audit_script_sha256": sha256(Path(__file__)),
        "issues": issues, "elapsed_seconds": time.perf_counter() - started,
        "interpretation_notes": [
            "Random-128 has 497 true failures and only 15 true nonfailures; its FP denominator is small.",
            "Low-load-128 has 511 true nonfailures and one true failure; report FP counts, not only state RMSE.",
            "Zero structural violations do not guarantee accurate learned damage increments or event times.",
            "Censored event times use horizon+1; capped MAE is not an uncensored failure-time error.",
        ],
    }
    (output / "audit_results.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/history_physics_benchmark_20261110"))
    args = parser.parse_args()
    result = audit(args.output)
    print(json.dumps({key: result[key] for key in ("status", "total_checks", "numeric_checks",
                                                  "test_array_identity_checks", "paired_minibatch_checks", "issues")}))
    if result["issues"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
