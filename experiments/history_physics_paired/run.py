"""Compare information available at the forecast boundary of a fatigue pair."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn

from experiments.history_physics_ablation_20261110.data import fatigue_step


INITIAL = .12
COEFFICIENT = .065
PAST = 60
FUTURE = 128
SEEDS = (101, 202, 303)
MODES = ("current_frame", "observations_and_load_history", "true_damage")


def observed_frame(shade):
    """The camera sees an opaque cover over the crack plane."""
    image = Image.new("RGB", (96, 48), (22, 34, 50))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((19, 11, 77, 37), 4, fill=(shade, shade + 6, shade + 10))
    for x in (12, 72):
        draw.rounded_rectangle((x, 8, x + 12, 40), 3, fill=(77, 95, 116))
    draw.line((25, 15, 70, 15), fill=(shade + 18, shade + 21, shade + 24))
    return image


def propagate(initial, loads):
    state = np.broadcast_to(initial, np.broadcast_shapes(np.shape(initial), loads.shape[:-1]))
    states = [state]
    for load in np.moveaxis(loads, -1, 0):
        state = fatigue_step(state, load, 1, COEFFICIENT)
        states.append(state)
    return np.stack(states, axis=-1)


def first_break(states):
    """Return the first 1-based future cycle, or horizon+1 if censored."""
    crossed = states[..., 1:] >= 1.
    return np.where(crossed.any(axis=-1), crossed.argmax(axis=-1) + 1, states.shape[-1])


def dataset(count, seed, blocked=False):
    rng = np.random.default_rng(seed)
    center = rng.uniform(.67, .82, (count, 1, 1))
    first = rng.uniform(.05, .33, count)
    second = rng.uniform(.05, .33, count)
    while (close := abs(first - second) < .06).any():
        second[close] = rng.uniform(.05, .33, close.sum())
    spread = np.stack((first, second), axis=1)[..., None]
    signs = np.tile([-1., 1.], PAST // 2)
    if blocked:
        # Hold out pulses of ten consecutive cycles in the test split.
        pulses = np.tile(np.repeat([-1., 1.], 10), PAST // 20)
        phase = rng.choice([-1., 1.], (count, 2, 1))
        signs = phase * pulses
    else:
        orders = rng.random((count, 2, PAST)).argsort(axis=-1)
        signs = signs[orders]
    # End both histories at rest so the final camera view is aligned.
    past = np.concatenate((center + spread * signs, np.zeros((count, 2, 1))), axis=-1)
    future = rng.uniform(.76, .94, (count, FUTURE))
    future[rng.random(future.shape) < .12] = 0.
    past_truth = propagate(INITIAL, past)
    truth = propagate(past_truth[..., -1], future[:, None, :])
    events = first_break(truth)
    shades = rng.integers(130, 191, count)
    # This stationary camera view is shared by both specimens and by all
    # history frames. It provides no damage information by construction.
    frames = np.stack([np.asarray(observed_frame(int(shade))) for shade in shades])
    np.testing.assert_allclose(past[:, 0].mean(-1), past[:, 1].mean(-1))
    assert (past_truth < 1).all()
    assert (events <= FUTURE).all() and (events[:, 0] != events[:, 1]).all()
    return dict(past=past, future=future, past_truth=past_truth, truth=truth, events=events, frames=frames)


def observations(data, history):
    # Small RGB descriptors of the actual raster frames. Repeat the stationary
    # observation through time and for each member of the pair.
    frame = data["frames"].reshape(-1, 4, 12, 8, 12, 3).mean(axis=(2, 4))
    frame = frame.reshape(len(frame), 1, 1, -1) / 255.
    visual = np.broadcast_to(frame, (*data["past"].shape, frame.shape[-1]))
    loads = data["past"] if history else np.zeros_like(data["past"])
    features = np.concatenate((visual, loads[..., None]), axis=-1)
    return torch.from_numpy(features.astype(np.float32)).flatten(0, 1)


class StateEstimator(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.step = nn.Sequential(nn.Linear(features, 32), nn.SiLU(), nn.Linear(32, 24), nn.SiLU())
        self.readout = nn.Sequential(nn.Linear(24, 32), nn.SiLU(), nn.Linear(32, 1))

    def forward(self, history):
        # Pool per-cycle features, matching this simulator's additive dose.
        return self.readout(self.step(history).mean(dim=1)).squeeze(-1).sigmoid()


def fit(train, val, tests, history, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    x, vx = observations(train, history), observations(val, history)
    target = torch.tensor(train["past_truth"][..., -1].ravel(), dtype=torch.float32)
    vtarget = torch.tensor(val["past_truth"][..., -1].ravel(), dtype=torch.float32)
    model = StateEstimator(x.shape[-1])
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    best_loss = float("inf")
    for epoch in range(80):
        for batch in np.array_split(rng.permutation(len(x)), len(x) // 128):
            optimizer.zero_grad()
            loss = (model(x[batch]) - target[batch]).square().mean()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            loss = (model(vx) - vtarget).square().mean().item()
            if loss < best_loss:
                best_loss = loss
                best = {key: value.clone() for key, value in model.state_dict().items()}
                best_epoch = epoch + 1
    model.load_state_dict(best)
    with torch.no_grad():
        pred = [model(observations(test, history)).numpy().reshape(-1, 2) for test in tests]
    return pred, best_epoch


def evaluate(data, initial):
    trajectory = propagate(initial, data["future"][:, None, :])
    events = first_break(trajectory)
    truth = data["events"]
    order = np.sign(events[:, 0] - events[:, 1])
    expected = np.sign(truth[:, 0] - truth[:, 1])
    return dict(
        correct=int((order == expected).sum()),
        wrong=int((order == -expected).sum()),
        ties=int((order == 0).sum()),
        censored=int((events > FUTURE).sum()),
        fracture_step_mae=float(np.abs(events - truth).mean()),
        damage_mae=float(np.abs(initial - data["past_truth"][..., -1]).mean()),
    ), events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    train = dataset(1024, 31001)
    val = dataset(256, 31002)
    tests = {"shuffled": dataset(512, 31003), "held_out_blocks": dataset(512, 31004, blocked=True)}
    held_out = tests["held_out_blocks"]
    gaps = np.abs(np.diff(held_out["events"], axis=1)).ravel()
    # Choose the demo using only reference outcomes before fitting any model.
    demo_index = int(np.argmin(np.abs(gaps - np.median(gaps))))
    report = dict(pairs_per_test=512, seeds=list(SEEDS), future_steps=FUTURE,
                  demo_seed=SEEDS[0], demo_index=demo_index, results=[])
    demo_events = []
    for seed in SEEDS:
        for mode in MODES:
            if mode == "true_damage":
                predictions = [data["past_truth"][..., -1] for data in tests.values()]
                epoch = None
            else:
                predictions, epoch = fit(train, val, tests.values(), mode != "current_frame", seed)
            for (split, data), initial in zip(tests.items(), predictions):
                metrics, events = evaluate(data, initial)
                if mode == "current_frame":
                    np.testing.assert_array_equal(initial[:, 0], initial[:, 1])
                report["results"].append(dict(seed=seed, mode=mode, split=split, val_epoch=epoch, **metrics))
                if seed == SEEDS[0] and split == "held_out_blocks":
                    demo_events.append(events[demo_index])
        print(f"Completed seed {seed}", flush=True)
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(args.output / "demo_case.npz",
                       **{key: value[demo_index] for key, value in held_out.items()},
                       predicted_events=np.stack(demo_events))


if __name__ == "__main__":
    main()
