"""Independent post-training audit of pairing, frozen sources and saved arrays.

Does not import training/evaluation/model/metric code, train a model, or perform
model inference. Event times are independently reconstructed by per-row loops.
Run only after evaluation.json and paired_effects.json have been written.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np


SEEDS = (101, 202, 303)
GRUS = ("gru_plain", "gru_state", "gru_projected", "gru_state_projected")
MODELS = {
    "thermal": GRUS + ("state_free", "state_full"),
    "fatigue": GRUS + ("state_free", "state_gated", "state_positive", "state_full", "state_cubic"),
}
FACTORS = {
    "state_main": {"gru_state": .5, "gru_plain": -.5, "gru_state_projected": .5, "gru_projected": -.5},
    "projection_main": {"gru_projected": .5, "gru_plain": -.5, "gru_state_projected": .5, "gru_state": -.5},
    "interaction": {"gru_state_projected": 1., "gru_state": -1., "gru_projected": -1., "gru_plain": 1.},
    "state_when_unprojected": {"gru_state": 1., "gru_plain": -1.},
    "state_when_projected": {"gru_state_projected": 1., "gru_projected": -1.},
    "projection_without_state_feedback": {"gru_projected": 1., "gru_plain": -1.},
    "projection_with_state_feedback": {"gru_state_projected": 1., "gru_state": -1.},
}
COMPARISONS = {
    "thermal": (("state_free", "state_full"),),
    "fatigue": (("state_free", "state_gated"), ("state_gated", "state_positive"),
                ("state_positive", "state_full"), ("state_full", "state_cubic")),
}
EFFECT_METRICS = ("state_rmse", "final_state_mae", "pre_failure_rmse", "tail_state_rmse",
                  "false_positive_rate", "false_negative_rate", "event_time_capped_mae_all_steps",
                  "damage_decrease_fraction", "rest_drift_fraction", "passive_increase_fraction")
DATA_FIELDS = ("truth", "x", "initial", "active", "failed")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def signature(array):
    a = np.asarray(array)
    return {"shape": list(a.shape), "dtype": str(a.dtype),
            "sha256": hashlib.sha256(a.tobytes()).hexdigest()}


def load(path):
    return json.loads(Path(path).read_text())


def event_times(mask):
    """1-based first event or horizon+1; deliberately not argmax-based."""
    return np.asarray([next((j + 1 for j, hit in enumerate(row) if hit), len(row) + 1)
                       for row in np.asarray(mask, dtype=bool)], dtype=np.int64)


class Checks:
    def __init__(self):
        self.counts = Counter()
        self.issues = []

    def equal(self, category, label, actual, expected):
        self.counts[category] += 1
        if actual != expected:
            self.issues.append({"category": category, "check": label,
                                "recomputed": actual, "recorded": expected})

    def numeric(self, category, label, actual, expected):
        self.counts[category] += 1
        if actual is None or expected is None:
            okay = actual is None and expected is None
        else:
            okay = bool(np.isclose(actual, expected, rtol=1e-10, atol=1e-12))
        if not okay:
            self.issues.append({"category": category, "check": label,
                                "recomputed": None if actual is None else float(actual),
                                "recorded": expected})

    def summary(self, category, label, values, recorded):
        finite = [v for v in values if v is not None and np.isfinite(v)]
        self.numeric(category, label + "/mean", np.mean(finite) if len(finite) == 3 else None, recorded["mean"])
        self.numeric(category, label + "/sample_sd", np.std(finite, ddof=1) if len(finite) == 3 else None,
                     recorded["sample_sd"])
        self.equal(category, label + "/valid_seed_count", len(finite), recorded["valid_seed_count"])
        self.equal(category, label + "/seed_keys", sorted(map(str, SEEDS)), sorted(recorded["per_seed"]))
        for seed, value in zip(SEEDS, values):
            self.numeric(category, label + f"/seed{seed}", value, recorded["per_seed"][str(seed)])


def array_metrics(task, condition, data):
    """Recompute state/event/constraint metrics directly from stored arrays."""
    p = data["prediction"][..., 0].astype(np.float64)
    y = data["truth"][..., 0].astype(np.float64)
    n, horizon = p.shape
    scale = 100. if task == "thermal" else 1.
    error = (p - y) * scale
    previous = np.column_stack((data["initial"][:, 0], p[:, :-1]))
    delta = p - previous
    rest = ~data["active"].astype(bool)
    out = {"sequences": n, "horizon_steps": horizon, "state_time_points": n * horizon,
           "state_rmse": float(np.sqrt(np.mean(np.square(error)))),
           "final_state_mae": float(np.mean(np.abs(error[:, -1]))),
           "maximum_absolute_state_error": float(np.max(np.abs(error)))}
    fractions = {}
    if task == "thermal":
        fractions["passive_increase_fraction"] = (rest & (delta > 1e-5), rest)
        fractions["ambient_undershoot_fraction"] = (p < -1e-5, np.ones_like(rest))
        out["maximum_passive_increase_K"] = float(max(0., np.max(delta[rest])) * 100) if rest.any() else None
    else:
        pre = ~data["failed"].astype(bool)
        out["pre_failure_state_time_points"] = int(pre.sum())
        out["pre_failure_rmse"] = float(np.sqrt(np.mean(error[pre] ** 2))) if pre.any() else None
        fractions["damage_decrease_fraction"] = (delta < -1e-5, np.ones_like(rest))
        fractions["rest_drift_fraction"] = (rest & (np.abs(delta) > 1e-5), rest)
        fractions["negative_crack_fraction"] = (p < -1e-5, np.ones_like(rest))
        fractions["above_failure_state_fraction"] = (p > 1 + 1e-5, np.ones_like(rest))
        true_time, pred_time = event_times(data["failed"]), event_times(p >= .9999)
        true_event, pred_event = true_time <= horizon, pred_time <= horizon
        matched = true_event & pred_event
        time_error = np.abs(true_time - pred_time)
        out.update(true_failure_fraction=float(true_event.mean()), predicted_failure_fraction=float(pred_event.mean()),
                   event_accuracy=float((true_event == pred_event).mean()),
                   event_time_capped_mae_all_steps=float(time_error.mean()),
                   event_time_true_failure_mae_including_missed_steps=float(time_error[true_event].mean()) if true_event.any() else None,
                   event_time_matched_failure_mae_steps=float(time_error[matched].mean()) if matched.any() else None,
                   matched_failure_time_count=int(matched.sum()))
        fractions["false_positive_rate"] = (pred_event & ~true_event, ~true_event)
        fractions["false_negative_rate"] = (~pred_event & true_event, true_event)
        fractions["matched_failure_coverage"] = (matched, true_event)
        after = np.arange(1, horizon + 1)[None, :] > pred_time[:, None]
        fractions["post_failure_return_fraction"] = (after & (p < .9999 - 1e-5), after)
        fractions["true_failed_state_undershoot_fraction"] = (data["failed"] & (p < 1 - 1e-5), data["failed"])
    if condition.startswith("passive_tail_"):
        tail = error[:, 32:]
        change = (p[:, -1] - p[:, 31]) * scale
        out.update(tail_state_rmse=float(np.sqrt(np.mean(tail ** 2))),
                   tail_final_mae=float(np.abs(error[:, -1]).mean()),
                   tail_own_state_final_change_mean=float(change.mean()),
                   tail_own_state_final_change_abs_mean=float(np.abs(change).mean()))
        tail_mask = np.zeros_like(rest)
        tail_mask[:, 32:] = True
        if task == "thermal":
            out["tail_cooling_change_mae_K"] = float(np.abs(change - (y[:, -1] - y[:, 31]) * scale).mean())
            fractions["tail_passive_increase_fraction"] = (tail_mask & (delta > 1e-5), tail_mask)
            fractions["tail_ambient_undershoot_fraction"] = (tail_mask & (p < -1e-5), tail_mask)
        else:
            survivors = y[:, 31] < 1
            out["tail_initial_survivor_count"] = int(survivors.sum())
            out["tail_survivor_rmse"] = float(np.sqrt(np.mean(tail[survivors] ** 2))) if survivors.any() else None
            out["tail_survivor_own_drift_abs_mean"] = float(np.abs(change[survivors]).mean()) if survivors.any() else None
            fractions["tail_damage_decrease_fraction"] = (tail_mask & (delta < -1e-5), tail_mask)
            fractions["tail_rest_drift_fraction"] = (tail_mask & (np.abs(delta) > 1e-5), tail_mask)
    for name, (numerator, denominator) in fractions.items():
        a, b = int(numerator.sum()), int(denominator.sum())
        out.update({name: a / b if b else None, name + "_count": a, name + "_denominator": b})
    return out


def conditions(task):
    result = [f"random_{n}" for n in (32, 64, 128, 256)] + [f"passive_tail_{n}" for n in (128, 256)]
    return result + (["low_load_128", "low_load_256"] if task == "fatigue" else [])


def audit_training(output, checks):
    import torch  # Read checkpoints only; no model is constructed or executed.
    manifest, complete = load(output / "training_manifest.json"), load(output / "training_complete.json")
    config = manifest["config"]
    source_dir = Path(__file__).resolve().parent
    checks.equal("source", "training fingerprint", complete["fingerprint"], manifest["fingerprint"])
    fingerprint = hashlib.sha256(json.dumps(dict(config=config, sources=manifest["sources"]), sort_keys=True).encode()).hexdigest()
    checks.equal("source", "recomputed fingerprint", fingerprint, manifest["fingerprint"])
    for name, expected in manifest["sources"].items():
        checks.equal("source", name, digest(source_dir / name), expected)
    checks.equal("source", "locked protocol", digest(output / "locked_protocol.md"), digest(source_dir / "PROTOCOL.md"))
    for key, expected in (("seeds", list(SEEDS)), ("updates", 1500), ("batch_size", 64), ("length", 32),
                          ("train_seed", 11001), ("val_seed", 11002), ("validate_every", 100)):
        checks.equal("training", "fixed config/" + key, config[key], expected)
    runs = complete["runs"]
    expected_keys = {(t, m, s) for t in MODELS for m in MODELS[t] for s in SEEDS}
    actual_keys = [(r["task"], r["model"], r["seed"]) for r in runs]
    checks.equal("training", "45 unique runs", len(actual_keys), 45)
    checks.equal("training", "complete run grid", sorted(actual_keys), sorted(expected_keys))
    pairing, checkpoints = [], []
    for task in MODELS:
        prior_path = output.parent / "history_physics_benchmark_20261110" / f"{task}_data_manifest.json"
        if prior_path.exists():
            prior_manifest = load(prior_path)
            checks.equal("training", task + "/unchanged train-val data", load(output / f"{task}_data_manifest.json"),
                         prior_manifest.get("hashes", prior_manifest))
        for seed in SEEDS:
            members = [r for r in runs if (r["task"], r["seed"]) == (task, seed)]
            schedule = np.random.default_rng(seed).integers(0, config["train_n"], (1500, 64))
            schedule_hash = hashlib.sha256(schedule.tobytes()).hexdigest()
            hashes = {r["model"]: r["schedule_sha256"] for r in members}
            for name, actual in hashes.items():
                checks.equal("pairing", f"{task}/{name}/{seed}/minibatches", actual, schedule_hash)
            initial_groups = {}
            for family, names in (("GRU", GRUS), ("state", tuple(m for m in MODELS[task] if m not in GRUS))):
                group = {r["model"]: r["initial_weights_sha256"] for r in members if r["model"] in names}
                checks.equal("pairing", f"{task}/{seed}/{family}/members", sorted(group), sorted(names))
                checks.equal("pairing", f"{task}/{seed}/{family}/identical initial weights", len(set(group.values())), 1)
                initial_groups[family] = group
            pairing.append(dict(task=task, seed=seed, expected_schedule_sha256=schedule_hash,
                                minibatch_sha256=hashes, initial_weights_sha256=initial_groups))
    for run in runs:
        task, model, seed = (run[k] for k in ("task", "model", "seed"))
        label = f"{task}/{model}/{seed}"
        folder = output / task / model / f"seed_{seed}"
        checks.equal("training", label + "/metadata identity", load(folder / "metadata.json"), run)
        checks.equal("training", label + "/status", run["status"], "complete")
        checks.equal("training", label + "/updates", run["updates_completed"], 1500)
        checks.equal("source", label + "/fingerprint", run["fingerprint"], fingerprint)
        trace = load(folder / "trace.json")
        checks.equal("checkpoint", label + "/validation steps", [r["step"] for r in trace], list(range(100, 1501, 100)))
        valid = [r for r in trace if r["val_mse"] is not None and np.isfinite(r["val_mse"])]
        selected = min(valid, key=lambda r: r["val_mse"]) if valid else None
        checks.numeric("checkpoint", label + "/best step", selected["step"] if selected else None, run["best_step"])
        checks.numeric("checkpoint", label + "/best val MSE", selected["val_mse"] if selected else None, run["best_val_mse"])
        for name in ("best.pt", "final.pt"):
            path = folder / name
            checks.equal("checkpoint", label + "/" + name + " exists", path.exists(), True)
            if path.exists():
                checkpoint = torch.load(path, map_location="cpu", weights_only=False)
                checks.equal("checkpoint", label + "/" + name + " metadata", checkpoint["metadata"], run)
                checkpoints.append({"task": task, "model": model, "seed": seed, "file": name, "sha256": digest(path)})
    return complete, pairing, checkpoints


def audit_cap_diagnostics(output, checks, lookup):
    path = output / "diagnostics.json"
    if not path.exists():
        return {"status": "not_yet_available", "rows": []}
    diagnostics = load(path)
    calibration = load(output / "calibration.json")
    checks.equal("source", "diagnostic calibration identity", digest(output / "calibration.json"), diagnostics["calibration_sha256"])
    for name, expected in diagnostics["source_sha256"].items():
        checks.equal("source", "diagnostics/" + name, digest(Path(__file__).resolve().parent / name), expected)
    for row in calibration["rows"]:
        alpha = float(np.clip(row["numerator"] / row["denominator"], .1, 2.))
        checks.numeric("calibration", f"{row['seed']}/least-squares factor", alpha, row["alpha"])
        checks.numeric("calibration", f"{row['seed']}/raw least-squares factor",
                       row["numerator"] / row["denominator"], row["unclipped_alpha"])
        checks.equal("source", f"{row['seed']}/calibration checkpoint",
                     digest(lookup[("fatigue", "state_full", row["seed"], "random_256")]["checkpoint"]),
                     calibration["checkpoint_hashes"][str(row["seed"])])
    cap_rows = diagnostics["capoff"]
    checks.equal("sameweights_cap", "six diagnostic conditions",
                 sorted((r["seed"], r["condition"]) for r in cap_rows),
                 sorted((seed, condition) for seed in SEEDS for condition in ("random", "low_load")))
    results, tradeoff_results, fp_profiles = [], [], []
    for row in cap_rows:
        seed, condition = row["seed"], row["condition"]
        path = output / "diagnostic_arrays" / f"calibration_{condition}_{seed}.npz"
        with np.load(path) as data:
            baseline = event_times(data["baseline"][..., 0] >= .9999)
            uncapped = event_times(data["capoff_frozen"][..., 0] >= .9999)
            mismatches = int((baseline != uncapped).sum())
            checks.numeric("sameweights_cap", f"{seed}/{condition}/mismatch_count", mismatches,
                           row["first_crossing_mismatch_count"])
            checks.numeric("sameweights_cap", f"{seed}/{condition}/zero mismatches", mismatches, 0)
            checks.numeric("sameweights_cap", f"{seed}/{condition}/sequence count", len(baseline), row["sequences"])
            checks.numeric("sameweights_cap", f"{seed}/{condition}/capped crossing count",
                           int((baseline <= data["baseline"].shape[1]).sum()), row["capped_crossing_count"])
            checks.numeric("sameweights_cap", f"{seed}/{condition}/uncapped maximum",
                           float(data["capoff_frozen"].max()), row["uncapped_max_saved_state"])
            formal = lookup[("fatigue", "state_full", seed, f"{condition}_256")]
            with np.load(formal["predictions"]) as saved:
                for diagnostic_field, formal_field in (("baseline", "prediction"), ("truth", "truth"),
                                                        ("initial", "initial"), ("x", "x")):
                    checks.equal("sameweights_cap", f"{seed}/{condition}/formal identity/{diagnostic_field}",
                                 signature(data[diagnostic_field]), signature(saved[formal_field]))
            for tradeoff in diagnostics["calibration_tradeoff"]:
                if (tradeoff["seed"], tradeoff["condition"]) != (seed, condition):
                    continue
                horizon = tradeoff["metrics"]["horizon"]
                field = "scaled" if tradeoff["variant"] == "calibrated" else "baseline"
                selected = {k: data[k] if k == "initial" else data[k][:, :horizon] for k in DATA_FIELDS}
                selected["prediction"] = data[field][:, :horizon]
                metrics = array_metrics("fatigue", f"{condition}_{horizon}", selected)
                label = f"{seed}/{condition}/{horizon}/{tradeoff['variant']}"
                aliases = {"horizon": "horizon_steps", "false_positive_count": "false_positive_rate_count",
                           "false_positive_denominator": "false_positive_rate_denominator",
                           "false_negative_count": "false_negative_rate_count",
                           "false_negative_denominator": "false_negative_rate_denominator"}
                counts = {"predicted_failure_count": int((event_times(selected["prediction"][..., 0] >= .9999) <= horizon).sum()),
                          "true_failure_count": int((event_times(selected["failed"]) <= horizon).sum())}
                for metric, expected in tradeoff["metrics"].items():
                    actual = counts[metric] if metric in counts else metrics[aliases.get(metric, metric)]
                    checks.numeric("calibration", label + "/" + metric, actual, expected)
                expected_alpha = next(r["alpha"] for r in calibration["rows"] if r["seed"] == seed) if field == "scaled" else 1.
                checks.numeric("calibration", label + "/alpha", tradeoff["alpha"], expected_alpha)
                tradeoff_results.append(dict(seed=seed, condition=condition, horizon=horizon, variant=tradeoff["variant"],
                    alpha=expected_alpha, false_positives=metrics["false_positive_rate_count"],
                    true_nonfailure_count=metrics["false_positive_rate_denominator"],
                    false_negatives=metrics["false_negative_rate_count"], true_failure_count=metrics["false_negative_rate_denominator"],
                    state_rmse=metrics["state_rmse"], capped_event_time_mae_steps=metrics["event_time_capped_mae_all_steps"]))
            results.append(dict(seed=seed, condition=condition, sequences=len(baseline),
                                first_crossing_mismatch_count=mismatches,
                                capped_crossing_count=int((baseline <= data["baseline"].shape[1]).sum())))
            if condition == "low_load":
                with np.load(output / "diagnostic_arrays" / f"low_load_state_full_{seed}.npz") as saved_oracle:
                    oracle = saved_oracle["oracle_next"][..., 0].astype(np.float64)
                    checks.equal("test_data_identity", f"{seed}/low_load/diagnostic rollout identity",
                                 signature(saved_oracle["prediction"]), signature(data["baseline"]))
                for horizon in (128, 256):
                    p = data["baseline"][:, :horizon, 0].astype(np.float64)
                    y = data["truth"][:, :horizon, 0].astype(np.float64)
                    pred_time = event_times(p >= .9999)
                    true_failure = data["failed"][:, :horizon].any(axis=1)
                    fp = (pred_time <= horizon) & ~true_failure
                    index = np.flatnonzero(fp)
                    if not len(index):
                        fp_profiles.append(dict(seed=seed, horizon=horizon, false_positive_count=0))
                        continue
                    local = oracle[:, :horizon] - y
                    before = np.arange(1, horizon + 1)[None, :] <= pred_time[:, None]
                    selected = before & data["active"][:, :horizon] & fp[:, None]
                    true_delta = y - np.column_stack((data["initial"][:, 0], y[:, :-1]))
                    cumulative_bias = (local * before).sum(axis=1)
                    true_gap = 1 - y[index, pred_time[index] - 1]
                    fp_profiles.append(dict(seed=seed, horizon=horizon, false_positive_count=len(index),
                        pre_crossing_oracle_increment_signed_mean=float(local[selected].mean()),
                        pre_crossing_oracle_increment_relative_bias=float(local[selected].sum() / true_delta[selected].sum()),
                        fraction_with_positive_cumulative_oracle_bias=float((cumulative_bias[fp] > 0).mean()),
                        truth_gap_to_failure_at_prediction_min=float(true_gap.min()),
                        truth_gap_to_failure_at_prediction_median=float(np.median(true_gap))))
    checks.numeric("calibration", "complete calibration tradeoff grid", len(tradeoff_results), 24)
    return {"status": "checked", "rows": results, "calibration_tradeoffs": tradeoff_results,
            "exploratory_false_positive_profiles": fp_profiles,
            "profile_caveat": "Post-hoc analysis conditioned on false positives, using saved true-state one-step diagnostics; not a population-wide or independently randomized causal claim.",
            "diagnostics_sha256": digest(output / "diagnostics.json")}


def audit(output):
    started, checks = time.perf_counter(), Checks()
    training, pairing, checkpoints = audit_training(output, checks)
    evaluation = load(output / "evaluation.json")
    effects = load(output / "paired_effects.json")
    checks.numeric("evaluation", "locked failure threshold", .9999, evaluation["failure_threshold"])
    checks.numeric("evaluation", "locked violation tolerance", 1e-5, evaluation["violation_tolerance_normalized"])
    checks.equal("source", "evaluation training complete SHA", digest(output / "training_complete.json"), evaluation["training_complete_sha256"])
    checks.equal("evaluation", "training metadata repeated faithfully", evaluation["training_runs"], training["runs"])
    checks.equal("source", "evaluation fingerprint", evaluation["training_fingerprint"], training["fingerprint"])
    frozen_eval_sources = load(output / "evaluation_sources.json")
    checks.equal("source", "evaluation source manifest", frozen_eval_sources, evaluation["evaluation_sources"])
    for name, expected in frozen_eval_sources.items():
        checks.equal("source", "evaluation/" + name, digest(Path(__file__).resolve().parent / name), expected)
    checks.equal("source", "frozen calibration", digest(output / "calibration.json"), evaluation["calibration_sha256_before_test_construction"])
    dataset_manifest = load(output / "test_data_manifest.json")
    checks.equal("test_data_identity", "test data manifest", dataset_manifest, evaluation["test_datasets"])
    dataset_lookup = {(r["task"], r["condition"]): r for r in dataset_manifest}
    for dataset in dataset_manifest:
        family = dataset["condition"].rsplit("_", 1)[0]
        checks.equal("test_data_identity", dataset["task"] + "/" + dataset["condition"] + "/fresh seed",
                     dataset["meta"]["seed"], {"random": 21003, "passive_tail": 21004, "low_load": 21005}[family])
    rows = evaluation["runs"]
    expected_rows = {(t, m, s, c) for t in MODELS for m in MODELS[t] for s in SEEDS for c in conditions(t)}
    checks.equal("evaluation", "all run conditions present", sorted((r["task"], r["model"], r["seed"], r["condition"]) for r in rows), sorted(expected_rows))
    checks.equal("evaluation", "invalid count", sum(not r["valid"] for r in rows), evaluation["invalid_run_conditions"])
    lookup, independent, references, short_hashes, expected_prefixes, event_counts = {}, {}, {}, {}, {}, []
    for row in rows:
        key = tuple(row[k] for k in ("task", "model", "seed", "condition"))
        task, model, seed, condition = key
        label, lookup[key] = "/".join(map(str, key)), row
        if row["predictions"] is None:
            checks.equal("evaluation", label + "/missing prediction invalid", row["valid"], False)
            checks.equal("evaluation", label + "/invalid metric exclusion", row["metrics"], None)
            continue
        with np.load(row["predictions"]) as archive:
            data = {k: archive[k] for k in (*DATA_FIELDS, "prediction")}
        checks.equal("evaluation", label + "/prediction shape", data["prediction"].shape, data["truth"].shape)
        horizon = int(condition.rsplit("_", 1)[1])
        checks.equal("evaluation", label + "/horizon", data["truth"].shape[1], horizon)
        checks.equal("evaluation", label + "/episodes", data["truth"].shape[0], 512)
        finite = bool(np.isfinite(data["prediction"]).all())
        checks.equal("evaluation", label + "/valid iff finite prediction", row["valid"], finite)
        for field in DATA_FIELDS:
            sig = signature(data[field])
            ref_key = (task, condition, field)
            if ref_key in references:
                checks.equal("test_data_identity", label + "/" + field, sig, references[ref_key])
            else:
                references[ref_key] = sig
                manifest_field = "y" if field == "truth" else field
                checks.equal("test_data_identity", f"{task}/{condition}/{field}/manifest",
                             sig, dataset_lookup[(task, condition)]["arrays"][manifest_field])
        family = condition.rsplit("_", 1)[0]
        for field, array in data.items():
            prefix_key = (task, model, seed, family, horizon, field)
            short_hashes[prefix_key] = signature(array)
            if horizon == 256:
                lengths = (32, 64, 128) if family == "random" else (128,)
                for length in lengths:
                    expected_prefixes[(task, model, seed, family, length, field)] = signature(array if field == "initial" else array[:, :length])
        if not row["valid"]:
            checks.equal("evaluation", label + "/invalid metric exclusion", row["metrics"], None)
            continue
        metrics = array_metrics(task, condition, data)
        independent[key] = metrics
        checks.equal("metric", label + "/metric keys", sorted(metrics), sorted(row["metrics"]))
        for metric, value in metrics.items():
            checks.numeric("metric", label + "/" + metric, value, row["metrics"].get(metric))
        if task == "fatigue" and (condition.startswith("low_load") or condition in ("random_128", "random_256")):
            event_counts.append(dict(task=task, model=model, seed=seed, condition=condition,
                false_positives=metrics["false_positive_rate_count"], true_nonfailure_count=metrics["false_positive_rate_denominator"],
                false_negatives=metrics["false_negative_rate_count"], true_failure_count=metrics["false_negative_rate_denominator"],
                capped_event_time_mae_steps=metrics["event_time_capped_mae_all_steps"],
                state_rmse=metrics["state_rmse"], pre_failure_rmse=metrics["pre_failure_rmse"]))
    for key, expected in expected_prefixes.items():
        checks.equal("nested_prefix", "/".join(map(str, key)), short_hashes.get(key), expected)
    for group in evaluation["aggregate"]:
        task, model, condition = (group[k] for k in ("task", "model", "condition"))
        members = [lookup[(task, model, seed, condition)] for seed in SEEDS]
        label = f"{task}/{model}/{condition}"
        checks.equal("aggregate", label + "/valid runs", sum(r["valid"] for r in members), group["valid_run_count"])
        checks.equal("aggregate", label + "/all seeds", all(r["valid"] for r in members), group["all_seeds_valid"])
        for metric, summary in group["metrics"].items():
            values = [independent.get((task, model, seed, condition), {}).get(metric) for seed in SEEDS]
            checks.summary("aggregate", label + "/" + metric, values, summary)
    factorial_keys, nested_keys = set(), set()
    for effect in effects["factorial"]:
        task, condition, metric, name = (effect[k] for k in ("task", "condition", "metric", "effect"))
        label = f"{task}/{condition}/{metric}/{name}"
        factorial_keys.add((task, condition, metric, name))
        weights = FACTORS[name]
        checks.equal("factorial", label + "/weights", effect["weights"], weights)
        values = []
        for seed in SEEDS:
            components = [independent.get((task, model, seed, condition), {}).get(metric) for model in weights]
            values.append(sum(w * v for w, v in zip(weights.values(), components)) if all(v is not None for v in components) else None)
        checks.summary("factorial", label, values, effect)
    for effect in effects["nested"]:
        task, condition, metric, before, after = (effect[k] for k in ("task", "condition", "metric", "before", "after"))
        label = f"{task}/{condition}/{metric}/{before}->{after}"
        nested_keys.add((task, condition, metric, before, after))
        checks.equal("nested_contrast", label + "/direction", effect["difference"], "after-minus-before")
        values = []
        for seed in SEEDS:
            a = independent.get((task, before, seed, condition), {}).get(metric)
            b = independent.get((task, after, seed, condition), {}).get(metric)
            values.append(b - a if a is not None and b is not None else None)
        checks.summary("nested_contrast", label, values, effect)
    expected_factorial, expected_nested = set(), set()
    for task in MODELS:
        for condition in conditions(task):
            available = {metric for key, values in independent.items() if key[0] == task and key[3] == condition for metric in values}
            for metric in set(EFFECT_METRICS) & available:
                expected_factorial.update((task, condition, metric, factor) for factor in FACTORS)
                expected_nested.update((task, condition, metric, before, after) for before, after in COMPARISONS[task])
    checks.equal("factorial", "complete factorial contrast grid", sorted(factorial_keys), sorted(expected_factorial))
    checks.equal("nested_contrast", "complete nested contrast grid", sorted(nested_keys), sorted(expected_nested))
    cap_audit = audit_cap_diagnostics(output, checks, lookup)
    result = dict(status="pass" if not checks.issues else "fail",
                  scope="Independent frozen-source, recorded training/checkpoint, array-metric and paired-contrast audit; no retraining or inference",
                  training_runs=len(training["runs"]), prediction_conditions=len(rows),
                  invalid_run_conditions=evaluation["invalid_run_conditions"],
                  total_checks=sum(checks.counts.values()), checks_by_category=dict(checks.counts),
                  paired_hashes=pairing, checkpoint_sha256=checkpoints,
                  recomputed_fatigue_event_counts=event_counts,
                  sameweights_cap_audit=cap_audit,
                  numeric_tolerances={"rtol": 1e-10, "atol": 1e-12},
                  evaluation_sha256=digest(output / "evaluation.json"),
                  paired_effects_sha256=digest(output / "paired_effects.json"),
                  training_complete_sha256=digest(output / "training_complete.json"),
                  audit_script_sha256=digest(__file__), issues=checks.issues,
                  elapsed_seconds=time.perf_counter() - started,
                  limitations=["Initialization pairing is verified from recorded hashes; initial parameter tensors were not saved.",
                               "Checkpoint selection is checked against recorded validation traces, without replaying validation inference.",
                               "Three-seed variability and fixed synthetic data do not establish real-material validity or statistical significance."])
    (output / "audit_results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/history_physics_ablation_20261110"))
    args = parser.parse_args()
    result = audit(args.output)
    print(json.dumps({k: result[k] for k in ("status", "training_runs", "prediction_conditions", "total_checks", "checks_by_category", "issues")}))
    if result["issues"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
