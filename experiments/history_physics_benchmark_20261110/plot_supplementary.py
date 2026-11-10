"""Post-test descriptive failure figure using saved predictions only.

Does not retrain, rescore aggregate metrics, or change the locked protocol.
Selection details and source hashes are written beside the figure.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluate import FAIL_THRESHOLD, first_crossing

COLORS = {"gru": "#2563eb", "constrained": "#d97706"}
LABELS = {"gru": "Plain GRU", "constrained": "Constrained state"}
SEEDS = (101, 202, 303)


def load_pair(directory, seed, condition):
    files = {model: directory / "predictions" / f"fatigue_{model}_seed{seed}_{condition}.npz"
             for model in COLORS}
    data = {}
    for model, filename in files.items():
        with np.load(filename) as z:
            data[model] = {key: z[key].copy() for key in z.files}
    np.testing.assert_array_equal(data["gru"]["truth"], data["constrained"]["truth"])
    return data, files


def select_case(directory, condition, kind):
    candidates = []
    for seed in SEEDS:
        data, files = load_pair(directory, seed, condition)
        truth = data["gru"]["truth"][..., 0]
        true_time = first_crossing(data["gru"]["failed"])
        horizon = truth.shape[1]
        selected_model = "constrained" if kind == "false_positive" else "gru"
        pred_time = first_crossing(data[selected_model]["prediction"][..., 0] >= FAIL_THRESHOLD)
        if kind == "false_positive":
            eligible = np.flatnonzero((true_time > horizon) & (pred_time <= horizon))
            for index in eligible:
                candidates.append((int(pred_time[index]), seed, int(index)))
        else:
            eligible = np.flatnonzero((true_time <= horizon) & (pred_time > horizon))
            for index in eligible:
                # The largest censored event-time error is the earliest missed
                # true fracture. Ties use ascending seed then sample index.
                candidates.append((int(true_time[index]), seed, int(index)))
    if not candidates:
        raise RuntimeError(f"No eligible {kind} exists; selection must not be silently substituted")
    _, seed, index = min(candidates)
    data, files = load_pair(directory, seed, condition)
    true_time = first_crossing(data["gru"]["failed"])[index]
    record = {
        "id": f"fatigue_{condition}_seed{seed}_sample{index}",
        "condition": condition, "seed": seed, "sample_index": index,
        "post_test_descriptive_selection": True,
        "selection": "Earliest constrained-model false positive among true nonfailures; ties by seed/sample"
                     if kind == "false_positive" else
                     "Largest capped event-time error among GRU missed true failures; ties by seed/sample",
        "true_event_step_or_horizon_plus_one": int(true_time),
        "true_final_crack": float(data["gru"]["truth"][index, -1, 0]),
        "models": {}, "sources": {},
    }
    for model in COLORS:
        p = data[model]["prediction"][index, :, 0]
        record["models"][model] = {
            "event_step_or_horizon_plus_one": int(first_crossing((p >= FAIL_THRESHOLD)[None])[0]),
            "final_crack": float(p[-1]),
        }
        record["sources"][model] = {"file": str(files[model].resolve()),
                                    "sha256": hashlib.sha256(files[model].read_bytes()).hexdigest()}
    return record, {"truth": data["gru"]["truth"][index, :, 0],
                    **{model: data[model]["prediction"][index, :, 0] for model in COLORS}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/history_physics_benchmark_20261110"))
    args = parser.parse_args()
    evaluation = args.output / "evaluation.json"
    report = json.loads(evaluation.read_text())
    runs = {(r["task"], r["model"], r["seed"], r["condition"]): r for r in report["runs"]}
    fp_case, fp_lines = select_case(args.output, "low_load_128", "false_positive")
    missed_case, missed_lines = select_case(args.output, "random_128", "missed_failure")
    figures = args.output / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    bars = []
    positions = np.arange(3)
    width = .36
    for j, model in enumerate(COLORS):
        metrics = [runs[("fatigue", model, seed, "low_load_128")]["metrics"] for seed in SEEDS]
        values = [m["false_positive_rate"] * 100 for m in metrics]
        counts = [m["false_positive_rate_count"] for m in metrics]
        denominators = [m["false_positive_rate_denominator"] for m in metrics]
        axes[0].bar(positions + (j - .5) * width, values, width, label=LABELS[model], color=COLORS[model])
        for x, y, count, den in zip(positions + (j - .5) * width, values, counts, denominators):
            axes[0].text(x, y + .12, f"{count}/{den}", ha="center", fontsize=8)
        bars.append({"model": model, "seeds": list(SEEDS), "rates_percent": values,
                     "false_positive_counts": counts, "true_nonfailure_denominators": denominators})
    axes[0].set(title="A. False fracture predictions under low load",
                xticks=positions, xticklabels=SEEDS, xlabel="Training seed",
                ylabel="False-positive rate among true nonfailures (%)", ylim=(0, 8.3))
    axes[0].legend(fontsize=9, frameon=False)
    axes[0].grid(axis="y", alpha=.2)
    for ax, record, lines, label in zip(axes[1:], [fp_case, missed_case], [fp_lines, missed_lines],
                                       ["B. Constrained model: premature fracture", "C. GRU: missed long-horizon fracture"]):
        t = np.arange(1, len(lines["truth"]) + 1)
        ax.plot(t, lines["truth"], color="#111827", lw=2.5, label="Reference")
        for model in COLORS:
            ax.plot(t, lines[model], color=COLORS[model], lw=1.7, label=LABELS[model])
        ax.axhline(FAIL_THRESHOLD, color="#64748b", ls=":", lw=1)
        ax.axvline(32.5, color="#64748b", ls="--", lw=1, alpha=.7)
        ax.set(title=f"{label}\n{record['condition']} | seed {record['seed']} | sample {record['sample_index']}",
               xlabel="Control step (training length: 32)", ylabel="Normalized crack length", ylim=(0, 1.12))
        ax.grid(alpha=.2)
    axes[2].legend(loc="lower right", fontsize=8, frameon=False)
    fig.suptitle("Fatigue failure diagnostics — fixed tests, 3 seeds\nLower state RMSE does not ensure fewer false fracture predictions", fontsize=14)
    fig.savefig(figures / "low_load_failure.png", dpi=180, bbox_inches="tight")
    fig.savefig(figures / "low_load_failure.pdf", bbox_inches="tight")
    plt.close(fig)
    selected = {
        "scope": "Post-test descriptive visualization only; no retraining, tuning, or aggregate changes",
        "evaluation_sha256": hashlib.sha256(evaluation.read_bytes()).hexdigest(),
        "failure_threshold": FAIL_THRESHOLD,
        "event_time_note": "129 means no fracture predicted/observed within the 128-step horizon",
        "false_positive_comparison": bars, "selected_cases": [fp_case, missed_case],
    }
    filename = figures / "low_load_failure_selection.json"
    filename.write_text(json.dumps(selected, indent=2) + "\n")
    print(json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()
