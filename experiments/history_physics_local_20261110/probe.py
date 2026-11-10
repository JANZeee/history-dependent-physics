"""Bounded local training probe; synthetic thermal learning and optional visual throughput.

No external data/weights or installation. Thermal results are a single-seed
feasibility check, not evidence of general physical reasoning. Visual clips
are synthetic moving discs and measure throughput, not physical correctness.
"""
import argparse
from datetime import datetime, timezone
import gc
import json
import os
import platform
import resource
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch import nn


class ThermalGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(3, 64, batch_first=True)
        self.head = nn.Linear(64, 1)

    def forward(self, x):
        return self.head(self.gru(x)[0])


class VisualGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 32, 4, 2, 1), nn.ReLU(),
            nn.Flatten(), nn.Linear(32 * 8 * 8, 64), nn.ReLU(),
        )
        self.gru = nn.GRU(64, 64, batch_first=True)
        self.project = nn.Linear(64, 32 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 4, 2, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        batch, length = x.shape[:2]
        z = self.encoder(x.reshape(-1, 3, 64, 64)).reshape(batch, length, 64)
        z = self.gru(z)[0]
        return self.decoder(self.project(z).reshape(-1, 32, 8, 8)).reshape(
            batch, length, 3, 64, 64)


def thermal_data(seed, count, length=32):
    """Exact piecewise-constant heat balance, C=100 J/K, initial T=ambient=20 C."""
    rng = np.random.default_rng(seed)
    power = rng.uniform(0, 200, (count, length))
    power *= rng.random((count, length)) > 0.35
    dt = rng.uniform(0.5, 5, (count, length))
    loss = rng.uniform(0.5, 4, (count, 1))
    excess = np.zeros(count)
    y = np.empty((count, length, 1), dtype=np.float32)
    for t in range(length):
        decay = np.exp(-loss[:, 0] * dt[:, t] / 100)
        excess = excess * decay + power[:, t] / loss[:, 0] * (1 - decay)
        y[:, t, 0] = excess / 100  # Target is (T - 20 C) / 100 K.
    x = np.stack((power / 200, dt / 5, np.broadcast_to(loss / 4, power.shape)), -1)
    return torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y)


def visual_data(seed, count, length=8):
    """Small fixed train/validation sets; full observed clip -> next-frame prediction."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:64, :64]
    clips = np.zeros((count, length + 1, 3, 64, 64), dtype=np.float32)
    for i in range(count):
        origin = rng.uniform(16, 48, 2)
        velocity = rng.uniform(-1, 1, 2)
        color = rng.uniform(0.2, 1, 3)
        for t in range(length + 1):
            cx, cy = origin + t * velocity
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 < 7 ** 2
            clips[i, t] = color[:, None, None] * mask[None]
    return torch.from_numpy(clips[:, :-1].copy()), torch.from_numpy(clips[:, 1:].copy())


def sync(device):
    if device == "mps":
        torch.mps.synchronize()


def memory_sample(device):
    # ru_maxrss is a cumulative process high-water mark, not a per-case delta.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    out = {"process_max_rss_bytes": int(rss if platform.system() == "Darwin" else rss * 1024)}
    if device == "mps":
        out.update(mps_tensor_bytes=torch.mps.current_allocated_memory(),
                   mps_driver_bytes=torch.mps.driver_allocated_memory())
    return out


@torch.no_grad()
def evaluate(model, data, device, batch_size):
    model.eval()
    squared, elements = 0.0, 0
    for start in range(0, len(data[0]), batch_size):
        x, y = (v[start:start + batch_size].to(device) for v in data)
        error = (model(x) - y).square()
        squared += float(error.sum().cpu())
        elements += error.numel()
    return squared / elements


def run_case(kind, device, steps, outdir):
    wall_start = datetime.now(timezone.utc).isoformat()
    case_start = time.perf_counter()
    torch.manual_seed(20261110)
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; no CPU fallback was requested")
    if device == "mps":
        torch.mps.set_per_process_memory_fraction(0.30)
        torch.mps.manual_seed(20261110)
    is_thermal = kind == "thermal"
    batch_size = 32 if is_thermal else 4
    train = thermal_data(310, 1024) if is_thermal else visual_data(310, 32)
    val = thermal_data(311, 256) if is_thermal else visual_data(311, 8)
    model = (ThermalGRU() if is_thermal else VisualGRU()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    # Identical minibatch schedules and initial parameters on both backends.
    rng = np.random.default_rng(312)
    schedule = rng.integers(0, len(train[0]), (steps, batch_size))
    before = evaluate(model, val, device, batch_size)
    baseline = float(val[1].square().mean())
    timing, losses, memory = [], [], []
    warmup = min(5, max(steps - 1, 0))
    sync(device)
    train_start = time.perf_counter()
    model.train()
    for step, indices in enumerate(schedule):
        sync(device)
        start = time.perf_counter()
        x, y = (v[indices].to(device) for v in train)
        optimizer.zero_grad(set_to_none=True)
        loss = (model(x) - y).square().mean()
        loss.backward()
        optimizer.step()
        sync(device)
        timing.append(time.perf_counter() - start)
        value = float(loss.detach().cpu())
        if not np.isfinite(value):
            raise FloatingPointError(f"Non-finite loss at step {step}")
        losses.append(value)
        memory.append(memory_sample(device))
        if (step + 1) % 25 == 0 or step + 1 == steps:
            print(json.dumps({"kind": kind, "device": device, "step": step + 1,
                              "steps": steps, "loss": value}), flush=True)
    train_seconds = time.perf_counter() - train_start
    after = evaluate(model, val, device, batch_size)
    sync(device)
    memory.append(memory_sample(device))
    avg = float(np.mean(timing[warmup:]))
    report = {
        "kind": kind, "device": device, "status": "ok",
        "parameters": sum(p.numel() for p in model.parameters()),
        "train_sequences": len(train[0]), "val_sequences": len(val[0]),
        "batch_size": batch_size, "sequence_length": train[0].shape[1],
        "train_steps": steps, "timing_warmup_steps_excluded": warmup,
        "synchronized_ms_per_step_after_warmup": avg * 1000,
        "sequences_per_second_after_warmup": batch_size / avg,
        "train_seconds_including_warmup": train_seconds,
        "val_mse_before": before, "val_mse_after": after,
        "zero_prediction_val_mse": baseline,
        "first_train_loss": losses[0], "last_train_loss": losses[-1],
        "memory_max_over_samples": {k: max(s.get(k, 0) for s in memory) for k in memory[-1]},
        "memory_notes": "MPS values sampled after steps, not allocator peaks; process RSS is cumulative across cases.",
        "timing_notes": "FP32; includes CPU indexing, transfer, forward, backward, optimizer and synchronization; excludes validation.",
    }
    if is_thermal:
        report.update(val_rmse_kelvin_before=100 * before ** 0.5,
                      val_rmse_kelvin_after=100 * after ** 0.5)
    else:
        report["scope"] = "Throughput probe only: observed synthetic disc frames, not open-loop physics evaluation."
    checkpoint = outdir / f"{kind}_{device}.pt"
    torch.save({"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "report": report, "seed": 20261110}, checkpoint)
    report["checkpoint"] = str(checkpoint.resolve())
    report["wall_start_utc"] = wall_start
    report["wall_end_utc"] = datetime.now(timezone.utc).isoformat()
    report["case_seconds_including_data_validation_and_save"] = time.perf_counter() - case_start
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--visual", action="store_true")
    parser.add_argument("--visual-steps", type=int, default=20)
    parser.add_argument("--devices", default="cpu,mps")
    args = parser.parse_args()
    if not 1 <= args.steps <= 1000 or not 1 <= args.visual_steps <= 100:
        parser.error("Use 1..1000 thermal steps and 1..100 visual steps for this bounded probe")
    devices = args.devices.split(",")
    if any(d not in ("cpu", "mps") for d in devices):
        parser.error("--devices must be a comma-separated subset of cpu,mps")
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    result = {"scope": "Single-seed local feasibility, not a physics benchmark result",
              "python": platform.python_version(), "torch": torch.__version__,
              "platform": platform.platform(), "cpu_threads": torch.get_num_threads(),
              "PYTORCH_ENABLE_MPS_FALLBACK": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
              "mps_available": torch.backends.mps.is_available(), "runs": []}
    if result["mps_available"]:
        result["mps_recommended_working_set_bytes"] = torch.mps.recommended_max_memory()
        result["mps_process_memory_fraction"] = 0.30
    for kind in (["thermal", "visual"] if args.visual else ["thermal"]):
        for device in devices:
            print(json.dumps({"starting": kind, "device": device}), flush=True)
            try:
                report = run_case(kind, device, args.steps if kind == "thermal" else args.visual_steps, args.output)
            except Exception as exc:
                report = {"kind": kind, "device": device, "status": "error",
                          "error": str(exc), "traceback": traceback.format_exc(limit=4)}
            result["runs"].append(report)
            (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(report), flush=True)
            gc.collect()
            if device == "mps" and torch.backends.mps.is_available():
                torch.mps.empty_cache()
    print(json.dumps({"report": str((args.output / "metrics.json").resolve())}), flush=True)


if __name__ == "__main__":
    main()
