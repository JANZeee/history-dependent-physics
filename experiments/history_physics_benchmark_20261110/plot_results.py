"""Static scientific figures from completed, locked evaluation artifacts."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"gru": "#2563eb", "constrained": "#d97706"}
LABELS = {"gru": "Plain GRU", "constrained": "Constrained state"}


def save(fig, output, stem):
    fig.savefig(output / f"{stem}.png", dpi=180, bbox_inches="tight")
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def aggregate_plots(report, output):
    lookup = {(r["task"], r["model"], r["condition"]): r["metrics"] for r in report["aggregate"]}
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.8), constrained_layout=True)
    specs = [
        ("thermal", "state_rmse", "Temperature trajectory", "RMSE (K)"),
        ("fatigue", "state_rmse", "Fatigue trajectory (all states)", "RMSE (critical-crack units)"),
        ("fatigue", "pre_failure_rmse", "Fatigue before true failure", "RMSE (critical-crack units)"),
        ("fatigue", "event_time_capped_mae_all_steps", "First failure, including censored cases", "Capped event-time MAE (steps)"),
    ]
    for ax, (task, metric, title, ylabel) in zip(axes.flat, specs):
        for model in COLORS:
            records = [lookup[(task, model, f"random_{h}")][metric] for h in (32, 64, 128)]
            ax.errorbar([32, 64, 128], [r["mean"] for r in records], yerr=[r["sample_sd"] for r in records],
                        marker="o", capsize=4, color=COLORS[model], label=LABELS[model])
        ax.set(title=title, xlabel="Rollout length (control steps)", ylabel=ylabel, xticks=[32, 64, 128])
        ax.grid(alpha=.2)
        ax.set_ylim(bottom=0)
    for ax, task in zip(axes.flat[4:], ["thermal", "fatigue"]):
        for i, model in enumerate(COLORS):
            rec = lookup[(task, model, "passive_tail_128")]["tail_state_rmse"]
            ax.bar(i, rec["mean"], yerr=rec["sample_sd"], color=COLORS[model], capsize=5, width=.6)
        ax.set(xticks=[0, 1], xticklabels=[LABELS[m] for m in COLORS],
               title=f"{'Cooling' if task == 'thermal' else 'Damage hold'}: 96-step passive tail",
               ylabel="Tail RMSE (K)" if task == "thermal" else "Tail RMSE (critical-crack units)")
        ax.grid(axis="y", alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2)
    fig.suptitle("Local synthetic history experiment — 3 initialization seeds\nError bars: sample SD; training history length: 32 steps", fontsize=14)
    save(fig, output, "aggregate_metrics")


def failure_plots(report, output):
    # Both selection models and all seeds are represented in the files. Each
    # sheet contains twelve predetermined worst cases (6 per physical task).
    for criterion in ("worst_long_rmse", "worst_passive_tail_rmse", "worst_event_time"):
        cases = [c for c in report["failure_cases"] if c["criterion"] == criterion]
        rows = (len(cases) + 2) // 3
        fig, axes = plt.subplots(rows, 3, figsize=(15, 3.1 * rows), squeeze=False, constrained_layout=True)
        for ax, case in zip(axes.flat, cases):
            data = np.load(case["file"])
            factor = 100 if case["task"] == "thermal" else 1
            offset = 20 if case["task"] == "thermal" else 0
            t = np.arange(1, len(data["truth"]) + 1)
            ax.plot(t, data["truth"].ravel() * factor + offset, color="#111827", lw=2, label="Reference")
            for model in COLORS:
                ax.plot(t, data[model].ravel() * factor + offset, color=COLORS[model], lw=1.5, label=LABELS[model])
            if "passive" in criterion:
                ax.axvline(32.5, color="#64748b", linestyle="--", lw=1)
                ax.axvspan(32.5, 128, color="#cbd5e1", alpha=.15)
            if case["task"] == "fatigue":
                ax.axhline(1, color="#6b7280", ls=":", lw=1)
            ax.set_title(f"{case['task'].title()} | seed {case['seed']} | sample {case['sample_index']}\nSelected by {LABELS[case['selected_model']]}", fontsize=10)
            ax.set_xlabel("Control step")
            ax.set_ylabel("Temperature (C)" if case["task"] == "thermal" else "Normalized crack length")
            ax.grid(alpha=.18)
        for ax in list(axes.flat)[len(cases):]:
            ax.set_visible(False)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=3)
        fig.suptitle(criterion.replace("_", " ").title() + " — both models, every seed", fontsize=15)
        save(fig, output, criterion)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/history_physics_benchmark_20261110"))
    args = parser.parse_args()
    report = json.loads((args.output / "evaluation.json").read_text())
    output = args.output / "figures"
    output.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    aggregate_plots(report, output)
    failure_plots(report, output)
    print(json.dumps({"figures": str(output.resolve())}))


if __name__ == "__main__":
    main()
