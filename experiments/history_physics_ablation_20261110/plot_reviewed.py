"""Render-only revision of frozen plot_results.py: constrain nonnegative performance axes.
Statistics, predictions and signed paired-effect axes are unchanged; original figures are retained.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

GRU_MODELS = ("gru_plain", "gru_state", "gru_projected", "gru_state_projected")
STATE_MODELS = ("state_free", "state_gated", "state_positive", "state_full", "state_cubic")
LABELS = {"gru_plain": "GRU: neither", "gru_state": "GRU: feedback", "gru_projected": "GRU: projection",
          "gru_state_projected": "GRU: feedback + projection", "state_free": "Free update",
          "state_gated": "+ stress gate", "state_positive": "+ positive increment",
          "state_full": "+ cap (full)", "state_cubic": "Full + cubic gate"}
COLORS = dict(zip(GRU_MODELS, ("#4477AA", "#EE6677", "#228833", "#AA3377")))
COLORS.update(dict(zip(STATE_MODELS, ("#4477AA", "#66CCEE", "#CCBB44", "#EE6677", "#228833"))))
SEEDS = ("101", "202", "303")


def savedata(fig, folder, stem):
    fig.savefig(folder / f"{stem}.png", dpi=180, bbox_inches="tight")
    fig.savefig(folder / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def curve(ax, lookup, task, models, family, horizons, metric, ylabel, title, factor=1., log=False):
    notes = []
    for mi, model in enumerate(models):
        means, deviations = [], []
        for hi, horizon in enumerate(horizons):
            group = lookup[(task, model, f"{family}_{horizon}")]
            m = group["metrics"].get(metric, {})
            mean = m.get("mean")
            means.append(np.nan if mean is None else mean * factor)
            deviations.append(np.nan if m.get("sample_sd") is None else m["sample_sd"] * factor)
            # Individual seeds remain visible, including when no 3-seed mean is valid.
            for si, seed in enumerate(SEEDS):
                value = m.get("per_seed", {}).get(seed)
                if value is not None and np.isfinite(value):
                    dx = (si - 1) * .025 + (mi - (len(models)-1)/2) * .012
                    ax.scatter(hi + dx, value * factor, s=13, color=COLORS[model], alpha=.55, zorder=4)
            if m.get("valid_seed_count", 0) != 3:
                notes.append(f"{LABELS[model]}@{horizon}: {m.get('valid_seed_count', 0)}/3")
        ax.errorbar(range(len(horizons)), means, yerr=deviations, color=COLORS[model],
                    marker="o", markersize=3.5, linewidth=1.6, capsize=3, label=LABELS[model])
    ax.set_xticks(range(len(horizons)), horizons)
    ax.set_xlabel("Rollout horizon (steps)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=10)
    ax.grid(True, alpha=.2)
    if log:
        # A symmetric logarithmic axis retains zero and negative SD error-bar endpoints.
        ax.set_yscale("symlog", linthresh=.001 if factor == 1. else .01, linscale=.8)
        ax.set_ylabel(ylabel + " (symlog)")
    # RMSE/rates cannot be negative. Only the displayed SD-bar portion outside
    # this physical domain is clipped; saved statistics are unchanged.
    ax.set_ylim(bottom=0)
    if notes:
        ax.text(.02, .98, "Defined seeds: " + "; ".join(notes), transform=ax.transAxes,
                ha="left", va="top", fontsize=6.2, color="#993333", wrap=True)


def family_figure(result, folder, group):
    lookup = {(r["task"], r["model"], r["condition"]): r for r in result["aggregate"]}
    models = GRU_MODELS if group == "factorial" else STATE_MODELS
    thermal_models = models if group == "factorial" else ("state_free", "state_full")
    fig, axes = plt.subplots(3, 2, figsize=(11.6, 11.5))
    specs = [
        ("thermal", thermal_models, "random", [32,64,128,256], "state_rmse", "Temperature RMSE (K)", "a  Thermal: random inputs", 1., True),
        ("thermal", thermal_models, "passive_tail", [128,256], "tail_state_rmse", "Cooling-tail RMSE (K)", "b  Thermal: zero power after step 32", 1., True),
        ("fatigue", models, "random", [32,64,128,256], "pre_failure_rmse", "Pre-failure crack RMSE", "c  Fatigue: random inputs", 1., True),
        ("fatigue", models, "passive_tail", [128,256], "tail_state_rmse", "Rest-tail crack RMSE", "d  Fatigue: zero load after step 32", 1., True),
        ("fatigue", models, "low_load", [128,256], "pre_failure_rmse", "Pre-failure crack RMSE", "e  Fatigue: sustained low load", 1., True),
        ("fatigue", models, "low_load", [128,256], "false_positive_rate", "False positives / true nonfailures (%)", "f  Fatigue: low-load false positives", 100., False),
    ]
    for ax, spec in zip(axes.flat, specs):
        curve(ax, lookup, *spec)
    handles = [Line2D([], [], color=COLORS[m], marker="o", label=LABELS[m]) for m in models]
    if group == "state":
        # In the thermal bridge, full changes source/loss positivity and the update form jointly.
        handles[3].set_label("Full physical update / + cap")
    title = "A: identical GRUs, state feedback x output projection" if group == "factorial" else "B: explicit-state update ablations"
    fig.suptitle(title, y=.995, fontsize=13)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .026), ncol=2 if group == "factorial" else 3,
               frameon=False, fontsize=9)
    fig.text(.5, .008, "Points: 3 seeds; lines and bars: mean +/- sample SD. No mean is plotted for incomplete seed sets.\n"
             "Train horizon = 32; tests = 512 fixed episodes. SD bars below zero are clipped to the nonnegative domain; statistics unchanged.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0,.095,1,.98))
    savedata(fig, folder, f"{group}_performance_reviewed")


def effects_figure(effects, folder):
    effect_order = ("state_main", "projection_main", "interaction")
    labels = ("Feedback\nmain effect", "Projection\nmain effect", "Interaction")
    specs = [("thermal", "random_256", "state_rmse", "Temperature RMSE difference (K)", 1.),
             ("fatigue", "random_256", "pre_failure_rmse", "Pre-failure crack RMSE difference", 1.),
             ("fatigue", "low_load_256", "false_positive_rate", "False-positive rate difference (pp)", 100.)]
    lookup = {(r["task"],r["condition"],r["metric"],r["effect"]): r for r in effects["factorial"]}
    fig, axes = plt.subplots(1,3,figsize=(12,4.5))
    for ax, (task, condition, metric, ylabel, factor) in zip(axes,specs):
        ax.axhline(0,color="black",linewidth=.8)
        for i, effect in enumerate(effect_order):
            row = lookup.get((task,condition,metric,effect),
                             {"mean": None, "valid_seed_count": 0, "per_seed": dict.fromkeys(SEEDS)})
            for si, seed in enumerate(SEEDS):
                v = row["per_seed"][seed]
                if v is not None:
                    ax.scatter(i + (si-1)*.08,v*factor,color="#4477AA",s=25,alpha=.7)
            if row["mean"] is not None:
                ax.errorbar(i,row["mean"]*factor,yerr=row["sample_sd"]*factor,fmt="_",color="black",capsize=4,markersize=15)
            else:
                ax.text(i,.97,f"{row['valid_seed_count']}/3 defined",transform=ax.get_xaxis_transform(),ha="center",fontsize=7,color="#993333")
        ax.set_xticks(range(3),labels)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{task.title()}: {condition.replace('_', ' ')}", fontsize=10)
        ax.grid(axis="y",alpha=.2)
    fig.suptitle("Paired factorial effects at 256 steps",fontsize=13)
    fig.text(.5,.018,"Within-seed contrasts; black marks = mean +/- sample SD across all 3 seeds. Negative = lower error/rate.\n"
             "Interaction = M11 - M10 - M01 + M00. These are descriptive effects, not significance tests.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.11,1,.95))
    savedata(fig,folder,"paired_factorial_effects")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/history_physics_ablation_20261110"))
    args = parser.parse_args()
    result = json.loads((args.output/"evaluation.json").read_text())
    effects = json.loads((args.output/"paired_effects.json").read_text())
    folder = args.output/"figures"
    folder.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":9, "axes.spines.top":False, "axes.spines.right":False})
    family_figure(result,folder,"factorial")
    family_figure(result,folder,"state")
    # The signed paired-effect figure already rendered correctly; retain it unchanged.
    print(json.dumps({"figures":str(folder.resolve()),"source":"Saved evaluation.json and paired_effects.json only"}))


if __name__ == "__main__":
    main()
