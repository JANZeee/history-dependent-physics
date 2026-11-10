# Local training feasibility probe — 2026-11-10

This isolated probe uses the existing repository `.venv`; it downloads no data
or weights and does not use or alter the OASIS checkpoints.

Measured host: Apple M2 Pro, 10 CPU cores, 16 GPU cores, 16 GiB unified memory,
macOS 27.2, Python 3.12.13, PyTorch 2.12.1, MPS available. Approximately 90 GiB
disk space was available at preflight. Swap use was already about 10.47 GiB;
that is not a measurement of available GPU memory.

## Reproduce

From the repository root:

```bash
.venv/bin/python experiments/history_physics_local_20261110/probe.py \
  --steps 100 --visual --visual-steps 20 \
  --output outputs/history_physics_local_rerun
```

The recorded run is in `outputs/history_physics_local_20261110/metrics.json`.
Four small checkpoints are saved beside it. CPU and MPS execute sequentially,
with identical initial weights, data, minibatch schedule and FP32 settings.
The recorded run had `PYTORCH_ENABLE_MPS_FALLBACK` unset.

## Measured results

| Probe | Parameters | Batch / sequence | CPU ms/update | MPS ms/update |
|---|---:|---:|---:|---:|
| Thermal history GRU | 13,313 | 32 / 32 | 3.108 | 17.889 |
| 64×64 CNN–GRU–decoder | 340,035 | 4 / 8 | 53.232 | 10.944 |

Each measured update includes indexing, transfer, forward, backward, optimizer,
and explicit synchronization. The first five updates are excluded from these
means. There were 100 thermal updates and 20 visual updates. These are short,
single-seed measurements under the current host workload, not sustained
throughput or convergence estimates.

The thermal model used 1,024 training and 256 independent validation sequences.
Its validation temperature RMSE fell from 13.345 K to 2.664 K on both backends.
Inputs are power, time interval, and heat-loss coefficient. Initial temperature
and heat capacity are fixed; the current or future temperature is not supplied
as an input. This tests learning a simple history integral with known
parameters, not visual physics reasoning, unknown-material identification,
fatigue, fracture, or long-horizon generalization.

The visual probe uses observed moving-disc clips to predict next frames. Its
20-update validation MSE remained worse than predicting a black frame, so it
only establishes that forward/backward training runs at the measured speed.
It is not evidence of learned motion or open-loop video prediction.

The process RSS high-water mark across all four cases was about 454 MiB.
The maximum sampled MPS driver allocation was about 70.6 MiB. MPS figures are
sampled after updates, not true allocation peaks or total system memory;
their definitions follow the PyTorch MPS API. The per-process allocator limit
was 30% of the recommended 11.84 GiB working set, approximately 3.552 GiB.

## Local experiment scope

1. Prefer CPU for small numerical history models: thermal accumulation,
   stable fatigue-crack growth and threshold events, with matched recurrent
   and constrained-state baselines.
2. Prefer MPS for a small 64×64 visual encoder/decoder. Start at batch 4,
   8-frame chunks; precompute visual features for repeated dynamics training.
3. Keep large frozen OASIS forward passes outside the small-model training
   loop. Its historical adapter runtimes are not estimates for these probes.

The three-seed numerical benchmark and follow-up ablations are now complete;
see the [project overview](../../README.md). Visual world-model validation and
real-material transfer remain future work.
