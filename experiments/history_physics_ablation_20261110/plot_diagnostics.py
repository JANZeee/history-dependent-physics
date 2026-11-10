"""Plot saved diagnostic summaries without constructing data or predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

COLORS = {"state_full": "#EE6677", "state_cubic": "#228833"}
LABELS = {"state_full": "Full: linear stress gate", "state_cubic": "Full: cubic stress gate"}
SEED_COLORS = {101: "#4477AA", 202: "#EE6677", 303: "#228833"}


def save(fig, output, stem):
    fig.savefig(output / f"{stem}.png", dpi=180, bbox_inches="tight")
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def mechanism_figure(result, output):
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8))
    for model in COLORS:
        records = sorted((r for r in result["grid"] if r["model"] == model), key=lambda r: r["seed"])
        stress = sorted(float(s) for s in records[0]["by_stress"] if float(s) > 0)
        for ax, metric, factor in zip(axes, ("signed_bias", "relative_signed_bias"), (1., 100.)):
            points = np.array([[r["by_stress"][str(s)][metric] * factor for s in stress] for r in records])
            for seed_values in points:
                ax.plot(stress, seed_values, color=COLORS[model], alpha=.23, linewidth=.8)
            ax.errorbar(stress, points.mean(0), yerr=points.std(0, ddof=1), color=COLORS[model],
                        marker="o", markersize=4, capsize=3, label=LABELS[model])
    for ax in axes:
        ax.axhline(0, color="black", linewidth=.8)
        ax.axvspan(.3, .8, color="#888888", alpha=.08)
        ax.set_xlabel("Stress amplitude")
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Signed finite-step rate bias (crack / cycle)")
    axes[1].set_ylabel("Signed bias / total true rate (%)")
    axes[0].set_title("a  Oracle true state: absolute rate bias", loc="left", fontsize=10)
    axes[1].set_title("b  Oracle true state: relative rate bias", loc="left", fontsize=10)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Does the learned update overestimate low-load damage?", fontsize=13)
    fig.text(.5, .02, "Thin lines: individual seeds; thick lines and bars: mean +/- sample SD across 3 seeds.\n"
             "Fixed (initial crack, current crack, stress, coefficient, cycles) grid; true crossing steps excluded.\n"
             "Shading marks low-load range. Finite-step rates use uncapped updates; this is a synthetic-material diagnostic.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .14, 1, .95))
    save(fig, output, "oracle_rate_bias")


def calibration_figure(result, calibration, output):
    fig, axes = plt.subplots(2, 3, figsize=(12.7, 7.3))
    lookup = {(r["condition"], r["seed"], r["variant"], r["metrics"]["horizon"]): r
              for r in result["calibration_tradeoff"]}
    specs = [("false_positive_rate", "False positives / true nonfailures (%)", 100.),
             ("false_negative_rate", "False negatives / true failures (%)", 100.),
             ("state_rmse", "Crack-state RMSE", 1.)]
    for row, condition in enumerate(("random", "low_load")):
        for col, (metric, ylabel, factor) in enumerate(specs):
            ax = axes[row, col]
            for hi, horizon in enumerate((128, 256)):
                undefined = 0
                for si, seed in enumerate(SEED_COLORS):
                    before = lookup[(condition, seed, "uncalibrated", horizon)]["metrics"][metric]
                    after = lookup[(condition, seed, "calibrated", horizon)]["metrics"][metric]
                    if before is None or after is None:
                        undefined += 1
                        continue
                    x = hi + (si - 1) * .06
                    ax.plot([x - .12, x + .12], [before * factor, after * factor],
                            color=SEED_COLORS[seed], alpha=.7, linewidth=1)
                    ax.scatter(x - .12, before * factor, s=24, edgecolor=SEED_COLORS[seed],
                               facecolor="white", zorder=4)
                    ax.scatter(x + .12, after * factor, s=24, color=SEED_COLORS[seed], zorder=4)
                if undefined:
                    ax.text(hi, .94, f"{undefined}/3 undefined", transform=ax.get_xaxis_transform(),
                            ha="center", va="top", fontsize=7, color="#993333")
            ax.set_xticks([0, 1], ["128", "256"])
            ax.set_xlim(-.4, 1.4)
            ax.set_ylim(bottom=0)
            ax.set_xlabel("Rollout horizon (steps)")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{chr(97+row*3+col)}  {condition.replace('_', ' ').title()}", loc="left", fontsize=10)
            ax.grid(axis="y", alpha=.2)
    handles = [Line2D([], [], color=SEED_COLORS[r["seed"]], label=f"Seed {r['seed']}: alpha = {r['alpha']:.3f}")
               for r in calibration["rows"]]
    handles += [Line2D([], [], marker="o", color="#555555", linestyle="none", markerfacecolor="white", label="Original"),
                Line2D([], [], marker="o", color="#555555", linestyle="none", label="Calibrated")]
    fig.legend(handles=handles, ncol=5, frameon=False, loc="lower center", bbox_to_anchor=(.5, .062), fontsize=8)
    fig.suptitle("One validation-fitted rate multiplier: error and event tradeoffs", fontsize=13)
    fig.text(.5, .012, "Same state_full checkpoints before/after scaling; points are paired within seed and test trajectory set.\n"
             "Alpha was fitted to oracle increments on independent low-load validation (128 steps), then frozen before tests.\n"
             "Undefined rates have zero eligible denominator; they are not plotted as zero. No significance test is implied.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .15, 1, .95))
    save(fig, output, "calibration_tradeoffs")


def history_figure(result, output):
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8), sharey=True)
    fields = [("local_oracle_bias", "Local update bias"),
              ("inherited_previous_error", "Inherited previous error"),
              ("feedback_increment", "Extra history-dependent increment")]
    colors = ("#4477AA", "#CCBB44", "#AA3377")
    for ax, model in zip(axes, COLORS):
        records = [r for r in result["decomposition"] if r["model"] == model]
        for (field, label), color in zip(fields, colors):
            values = np.array([[next(v for v in r["rows"] if v["horizon"] == h)[field]["mean"]
                                for h in (32, 128, 256)] for r in records])
            if len(values) != 3:
                ax.text(.02, .96, f"{len(values)}/3 finite seeds", transform=ax.transAxes, va="top")
                continue
            for points in values:
                ax.scatter([0, 1, 2], points, color=color, s=12, alpha=.3)
            ax.errorbar([0, 1, 2], values.mean(0), yerr=values.std(0, ddof=1), color=color,
                        marker="o", markersize=3, capsize=3, label=label)
        ax.axhline(0, color="black", linewidth=.8)
        ax.set_xticks([0, 1, 2], ["32", "128", "256"])
        ax.set_xlabel("Prefix horizon (steps)")
        ax.set_ylabel("Mean signed crack error / increment")
        ax.set_title(LABELS[model], loc="left", fontsize=10)
        ax.grid(alpha=.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Separate local bias, inherited error, and additional feedback", fontsize=13)
    fig.text(.5, .015, "Low-load rollouts; active steps with true post-step crack < 1 only. Mean +/- sample SD across seeds.\n"
             "Total error = local oracle bias + inherited previous error + extra history-dependent increment.\n"
             "A large inherited error alone is not evidence of feedback amplification.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .13, 1, .95))
    save(fig, output, "history_error_decomposition")


def occupied_state_figure(result, output):
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.5), sharey=True)
    for ax, horizon in zip(axes, (128, 256)):
        bin_counts = None
        bins = None
        for mi, model in enumerate(COLORS):
            records = [next(row for row in entry["rows"] if row["horizon"] == horizon)
                       for entry in result["decomposition"] if entry["model"] == model]
            bins = records[0]["by_previous_state"]
            bin_counts = [v["oracle_rate"]["count"] for v in bins]
            values = np.array([[np.nan if v["oracle_rate"]["relative_signed_bias"] is None
                                else v["oracle_rate"]["relative_signed_bias"] * 100
                                for v in r["by_previous_state"]] for r in records])
            mean, sd = [], []
            for col in values.T:
                mean.append(float(col.mean()) if np.isfinite(col).all() else np.nan)
                sd.append(float(col.std(ddof=1)) if np.isfinite(col).all() else np.nan)
            x = np.arange(len(bins)) + (mi - .5) * .14
            for seed_values in values:
                ax.scatter(x, seed_values, color=COLORS[model], s=14, alpha=.3)
            ax.errorbar(x, mean, yerr=sd, color=COLORS[model], marker="o", markersize=4,
                        capsize=3, label=LABELS[model])
        labels = [f"{v['lower']:g}-{min(v['upper'], 1.):g}\nn={n:,}" for v, n in zip(bins, bin_counts)]
        ax.set_xticks(np.arange(len(labels)), labels, fontsize=8)
        for tick, count in zip(ax.get_xticklabels(), bin_counts):
            if count < 100:
                tick.set_color("#993333")
        ax.axhline(0, color="black", linewidth=.8)
        ax.set_title(f"{'a' if horizon == 128 else 'b'}  Actual low-load trajectories: {horizon} steps", loc="left", fontsize=10)
        ax.set_xlabel("True previous crack bin; n = active transitions")
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Oracle signed rate bias / total true rate (%)")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Local rate bias changes with damage state", fontsize=13)
    fig.text(.5, .025, "Oracle uses each test trajectory's true previous state for one step; normal rollouts receive no oracle states.\n"
             "Lines and bars: mean +/- sample SD across 3 model seeds. Counts are shared data transitions, not independent seed replicates.\n"
             "Active true pre-failure transitions only; relative bias = sum(rate error) / sum(true rate).\n"
             "Red counts indicate fewer than 100 points; at 128 steps the final bin has no data and its rate is undefined.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .19, 1, .95))
    save(fig, output, "occupied_state_rate_bias")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = json.loads((args.output / "diagnostics.json").read_text())
    calibration = json.loads((args.output / "calibration.json").read_text())
    folder = args.output / "figures"
    folder.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})
    mechanism_figure(result, folder)
    calibration_figure(result, calibration, folder)
    history_figure(result, folder)
    occupied_state_figure(result, folder)
    print(json.dumps({"diagnostic_figures": str(folder.resolve()), "source": "Saved summaries only"}))


if __name__ == "__main__":
    main()
