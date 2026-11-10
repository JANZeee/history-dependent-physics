"""Unchanged pure metrics copied from the previous locked numerical experiment.
Source evaluate.py SHA256: 4ec4be6c30a387de0914a40eae791948beefcc813fe5865245b842ae232f7082
"""
import numpy as np

TOL = 1e-5
FAIL_THRESHOLD = 1 - 1e-4

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
