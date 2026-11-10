"""Paired, fixed-budget ablation training; never constructs evaluation data."""
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
from models import GRU_NAMES, TASK_MODELS, make_model

CONFIG = dict(seeds=[101, 202, 303], tasks=['thermal', 'fatigue'],
              models={k: list(v) for k, v in TASK_MODELS.items()},
              train_seed=11001, val_seed=11002, train_n=5000, val_n=512,
              length=32, updates=1500, batch_size=64, learning_rate=.001,
              grad_clip=1., validate_every=100, threads=2)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.no_grad()
def validation(model, batch):
    model.eval()
    total, count = 0., 0
    for start in range(0, len(batch['x']), 64):
        pred = model(batch['x'][start:start+64], batch['initial'][start:start+64])
        error = (pred - batch['y'][start:start+64]).square()
        if not torch.isfinite(error).all():
            return None
        total += error.sum().item()
        count += error.numel()
    return total / count


@torch.no_grad()
def projection_diagnostics(model, batch):
    if model.name not in GRU_NAMES:
        return None
    x, initial = batch['x'][:64], batch['initial'][:64]
    pred, raw = model(x, initial, return_candidates=True)
    prev = torch.cat((initial.unsqueeze(1), pred[:, :-1]), 1)
    active = x[..., :1] > 0
    if model.task == 'fatigue':
        interior = active & (raw > prev) & (raw < 1)
        feasible = torch.where(active, (raw >= prev) & (raw <= 1), raw == prev)
    else:
        interior = (raw > 0) & (active | (raw < prev))
        feasible = (raw >= 0) & (active | (raw <= prev))
    return dict(raw_candidate_feasible_fraction=feasible.float().mean().item(),
                raw_candidate_differentiable_interior_fraction=interior.float().mean().item(),
                actually_changed_fraction=(raw != pred).float().mean().item(),
                raw_candidate_min=raw.min().item(), raw_candidate_max=raw.max().item())


@torch.no_grad()
def initial_increment_diagnostics(model, batch):
    # Same held-out input slice for every model; descriptive, never selected on.
    pred = model(batch['x'][:64, :1], batch['initial'][:64])[:, 0]
    delta = pred - batch['initial'][:64]
    return dict(mean=delta.mean().item(), minimum=delta.min().item(), maximum=delta.max().item(),
                negative_fraction=(delta < 0).float().mean().item())


def run(task, name, seed, train, val, output, fingerprint):
    torch.manual_seed(seed)
    model = make_model(name, task)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    schedule = np.random.default_rng(seed).integers(0, CONFIG['train_n'],
                                                  (CONFIG['updates'], CONFIG['batch_size']))
    directory = output / task / name / f'seed_{seed}'
    directory.mkdir(parents=True, exist_ok=False)
    meta = dict(task=task, model=name, seed=seed, fingerprint=fingerprint,
                parameters=sum(p.numel() for p in model.parameters()),
                schedule_sha256=hashlib.sha256(schedule.tobytes()).hexdigest(),
                initial_weights_sha256=hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest(),
                initial_val_mse=validation(model, val),
                initial_increment=initial_increment_diagnostics(model, val), status='running')
    started = time.perf_counter()
    trace, best, best_step, best_state = [], float('inf'), None, None
    zero_grad_steps = 0
    print(json.dumps({'start': {k: meta[k] for k in ('task', 'model', 'seed', 'parameters')}}), flush=True)
    for step, ids in enumerate(schedule, 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred = model(train['x'][ids], train['initial'][ids])
        loss = (pred - train['y'][ids]).square().mean()
        if not torch.isfinite(loss):
            meta.update(status='failed', failure=f'Nonfinite training loss at update {step}')
            break
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])
        if not torch.isfinite(grad_norm):
            meta.update(status='failed', failure=f'Nonfinite gradient at update {step}')
            break
        zero_grad_steps += int(grad_norm.item() <= 1e-12)
        optimizer.step()
        if step % CONFIG['validate_every'] == 0:
            val_mse = validation(model, val)
            row = dict(step=step, train_batch_mse=loss.item(), val_mse=val_mse,
                       elapsed_seconds=time.perf_counter()-started,
                       unclipped_gradient_norm=grad_norm.item(),
                       zero_gradient_updates_so_far=zero_grad_steps,
                       projection=projection_diagnostics(model, val))
            trace.append(row)
            if val_mse is not None and val_mse < best:
                best, best_step = val_mse, step
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if step % 500 == 0:
                print(json.dumps(dict(task=task, model=name, seed=seed, **row)), flush=True)
    else:
        meta['status'] = 'complete'
    meta.update(best_step=best_step, best_val_mse=best if best_state is not None else None,
                updates_completed=step if meta['status'] == 'complete' else step-1,
                zero_gradient_updates=zero_grad_steps,
                training_seconds=time.perf_counter()-started,
                process_max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                checkpoint=str((directory/'best.pt').resolve()) if best_state is not None else None)
    torch.save(dict(state_dict=model.state_dict(), metadata=meta), directory/'final.pt')
    if best_state is not None:
        torch.save(dict(state_dict=best_state, metadata=meta), directory/'best.pt')
    (directory/'trace.json').write_text(json.dumps(trace, indent=2)+'\n')
    (directory/'metadata.json').write_text(json.dumps(meta, indent=2)+'\n')
    print(json.dumps({'finished': {k: meta[k] for k in ('task','model','seed','status','best_step','best_val_mse','training_seconds')}}), flush=True)
    return meta


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'training_manifest.json').exists():
        raise RuntimeError('Refusing to overwrite an existing experiment')
    torch.set_num_threads(CONFIG['threads'])
    folder = Path(__file__).resolve().parent
    names = ('models.py','data.py','train.py','PROTOCOL.md')
    sources = {name: digest(folder/name) for name in names}
    fingerprint = hashlib.sha256(json.dumps(dict(config=CONFIG, sources=sources), sort_keys=True).encode()).hexdigest()
    manifest = dict(config=CONFIG, sources=sources, fingerprint=fingerprint,
                    start_utc=datetime.now(timezone.utc).isoformat(),
                    python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                    platform=platform.platform(), device='cpu', dtype='float32')
    (args.output/'training_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (args.output/'locked_protocol.md').write_text((folder/'PROTOCOL.md').read_text())
    runs = []
    for task in CONFIG['tasks']:
        raw = {kind: make_dataset(task, CONFIG[f'{kind}_n'], CONFIG['length'], CONFIG[f'{kind}_seed'])
               for kind in ('train','val')}
        hashes = {kind: {k: hashlib.sha256(d[k].tobytes()).hexdigest() for k in ('x','y','initial')}
                  for kind, d in raw.items()}
        (args.output/f'{task}_data_manifest.json').write_text(json.dumps(hashes, indent=2)+'\n')
        batches = {kind: {k: torch.from_numpy(d[k]) for k in ('x','y','initial')} for kind,d in raw.items()}
        for seed in CONFIG['seeds']:
            for name in TASK_MODELS[task]:
                runs.append(run(task, name, seed, batches['train'], batches['val'], args.output, fingerprint))
                (args.output/'training_progress.json').write_text(json.dumps(dict(runs=runs), indent=2)+'\n')
    for name, sha in sources.items():
        if digest(folder/name) != sha:
            raise RuntimeError(f'Frozen source changed: {name}')
    (args.output/'training_complete.json').write_text(json.dumps(dict(
        runs=runs, fingerprint=fingerprint, completed_utc=datetime.now(timezone.utc).isoformat()), indent=2)+'\n')
    print(json.dumps({'complete':str(args.output.resolve()),'runs':len(runs)}),flush=True)


if __name__ == '__main__':
    main()
