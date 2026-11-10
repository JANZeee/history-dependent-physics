"""Locked CPU training. Test data are never constructed by this entry point."""
import argparse
import hashlib
import json
import platform
import resource
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from data import make_dataset
from models import make_model


CONFIG = dict(seeds=[101, 202, 303], tasks=["thermal", "fatigue"],
              models=["gru", "constrained"], train_seed=11001, val_seed=11002,
              train_n=5000, val_n=512, length=32, updates=1500, batch_size=64,
              learning_rate=0.001, grad_clip=1.0, validate_every=100, threads=2)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensors(raw):
    return {k: torch.from_numpy(raw[k]) for k in ("x", "y", "initial")}


@torch.no_grad()
def validation(model, batch):
    model.eval()
    total, count = 0., 0
    for start in range(0, len(batch["x"]), 64):
        pred = model(batch["x"][start:start+64], batch["initial"][start:start+64])
        error = (pred - batch["y"][start:start+64]).square()
        if not torch.isfinite(error).all():
            raise FloatingPointError("Non-finite validation prediction")
        total += error.sum().item()
        count += error.numel()
    return total / count


def run(task, name, seed, train, val, out, fingerprint):
    torch.manual_seed(seed)
    model = make_model(name, task)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    rng = np.random.default_rng(seed)
    schedule = rng.integers(0, CONFIG["train_n"],
                            (CONFIG["updates"], CONFIG["batch_size"]))
    run_dir = out / task / name / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=False)
    meta = dict(task=task, model=name, seed=seed, fingerprint=fingerprint,
                parameters=sum(p.numel() for p in model.parameters()),
                schedule_sha256=hashlib.sha256(schedule.tobytes()).hexdigest(),
                initial_val_mse=validation(model, val))
    trace = []
    best, best_step, best_state = float("inf"), None, None
    started = time.perf_counter()
    print(json.dumps({"start": meta}), flush=True)
    for step, ids in enumerate(schedule, start=1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred = model(train["x"][ids], train["initial"][ids])
        loss = (pred - train["y"][ids]).square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{task}/{name}/{seed} loss at update {step}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
        if not torch.isfinite(grad_norm):
            raise FloatingPointError("Non-finite gradient")
        optimizer.step()
        if step % CONFIG["validate_every"] == 0:
            val_mse = validation(model, val)
            row = dict(step=step, train_batch_mse=loss.item(), val_mse=val_mse,
                       elapsed_seconds=time.perf_counter()-started,
                       unclipped_gradient_norm=grad_norm.item())
            trace.append(row)
            if val_mse < best:
                best, best_step = val_mse, step
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if step % 500 == 0:
                print(json.dumps(dict(task=task, model=name, seed=seed, **row)), flush=True)
    meta.update(best_step=best_step, best_val_mse=best,
                training_seconds=time.perf_counter()-started,
                process_max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                checkpoint=str((run_dir / "best.pt").resolve()))
    torch.save(dict(state_dict=model.state_dict(), metadata=meta), run_dir / "final.pt")
    torch.save(dict(state_dict=best_state, metadata=meta), run_dir / "best.pt")
    (run_dir / "trace.json").write_text(json.dumps(trace, indent=2)+"\n")
    (run_dir / "metadata.json").write_text(json.dumps(meta, indent=2)+"\n")
    print(json.dumps({"finished": meta}), flush=True)
    return meta


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "training_manifest.json").exists():
        raise RuntimeError("Refusing to overwrite an existing locked run; use a new output directory")
    torch.set_num_threads(CONFIG["threads"])
    folder = Path(__file__).resolve().parent
    sources = {name: digest(folder/name) for name in ("models.py", "data.py", "train.py", "PROTOCOL.md")}
    fingerprint = hashlib.sha256(json.dumps(dict(config=CONFIG, sources=sources), sort_keys=True).encode()).hexdigest()
    manifest = dict(config=CONFIG, sources=sources, fingerprint=fingerprint,
                    start_utc=datetime.now(timezone.utc).isoformat(),
                    python=platform.python_version(), torch=torch.__version__,
                    platform=platform.platform(), device="cpu", dtype="float32")
    (args.output / "training_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    (args.output / "locked_protocol.md").write_text((folder/"PROTOCOL.md").read_text())
    runs = []
    for task in CONFIG["tasks"]:
        raw_train = make_dataset(task, CONFIG["train_n"], CONFIG["length"], CONFIG["train_seed"])
        raw_val = make_dataset(task, CONFIG["val_n"], CONFIG["length"], CONFIG["val_seed"])
        data_hash = {split: {k: hashlib.sha256(raw[k].tobytes()).hexdigest()
                            for k in ("x", "y", "initial")}
                     for split, raw in (("train", raw_train), ("val", raw_val))}
        (args.output/f"{task}_data_manifest.json").write_text(json.dumps(dict(hashes=data_hash, meta=raw_train["meta"]), indent=2)+"\n")
        train, val = tensors(raw_train), tensors(raw_val)
        for seed in CONFIG["seeds"]:
            for name in CONFIG["models"]:
                runs.append(run(task, name, seed, train, val, args.output, fingerprint))
                (args.output/"training_progress.json").write_text(json.dumps(dict(runs=runs), indent=2)+"\n")
    for name, sha in sources.items():
        if digest(folder/name) != sha:
            raise RuntimeError(f"Training source changed during the run: {name}")
    (args.output/"training_complete.json").write_text(json.dumps(dict(
        runs=runs, fingerprint=fingerprint, completed_utc=datetime.now(timezone.utc).isoformat()), indent=2)+"\n")
    print(json.dumps({"complete": str(args.output.resolve()), "runs": len(runs)}), flush=True)


if __name__ == "__main__":
    main()
