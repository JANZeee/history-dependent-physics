# Locked local protocol — 2026-11-10

This file is fixed before training or test scoring. This is a CPU numerical
history experiment, not a visual world-model or real-material validation.
Previous OASIS experiments and local feasibility-probe outputs are unchanged.

## Questions and limits

Compare an ordinary GRU and a similarly sized constrained-state network on
thermal accumulation/cooling and normalized fatigue crack growth/fracture.
Both receive the same actions, elapsed times/cycles, public material parameters
and initial conditions. Neither receives current true state during rollouts.
The comparison changes architecture and prior jointly; it does not identify
the causal effect of constraints alone. The constrained architecture's passive
cooling and irreversible damage are guaranteed by construction, not learned
physical discoveries. Numeric accuracy and event timing are separate outcomes.

## Data and fixed computation budget

- NumPy synthetic training seed 11001: 5,000 trajectories per task, 32 steps.
- Independent validation seed 11002: 512 trajectories per task, 32 steps.
- Test random seed 11003: 512 trajectories, generated once at length 128;
  report nested 32/64/128 prefixes.
- Passive-tail seed 11004: 512 trajectories; 32 active then 96 zero-input steps.
- Fatigue low-load seed 11005: 512 trajectories of 128 in-range low-amplitude
  steps, so long-horizon results cannot rely entirely on absorbing failure.
- Initialization/minibatch seeds: 101, 202, 303. Same minibatch schedule within
  each seed for both architectures. Same training/validation/test arrays for
  all runs. No test-selected hyperparameters, early stopping, or retries.
- CPU FP32, two CPU threads, batch 64, 1,500 Adam updates, learning rate 0.001,
  no weight decay; global gradient norm clipped to 1 for both architectures.
- Objective: identical mean squared state error at all 32 training steps.
  No teacher forcing. Full backpropagation through training history.
- Validate every 100 updates and retain the checkpoint with lowest ID validation
  MSE (ties retain the earlier checkpoint). No test scoring until all 12 runs
  finish. Save final and validation-selected checkpoints and training traces.

## Reference physics

Thermal: exact constant-input solution of C*dT/dt=P-H*(T-20), with C in
[80,120] J/K, H in [0.5,4] W/K, P in [0,200] W with 35% zero inputs,
dt in [0.5,5] s and initial T=20 C. Target is (T-20)/100.

Fatigue: dimensionless Paris-law illustration da/dn=k*s^3*a^(3/2), with
exact stable-flow integration and an absorbing threshold a=1. Initial a in
[0.08,0.2], k in [0.04,0.12], stress range s in [0.3,1.5] with 25% rests,
cycle increment dn in [0.5,1.5]. A fixed maximum stress with changing stress
range permits a fixed critical crack size. Crack closure, short-crack physics,
plasticity and near-instability acceleration are omitted. This is a synthetic
stable-growth-plus-threshold model, not a validated real fracture simulator.
Fatigue passive prefixes and low-load tests use s in [0.3,0.8]; these are
within training amplitude support. No recovery or self-healing is modeled.

## Locked evaluation

- State trajectory RMSE; thermal reports K and fatigue critical-crack units.
- Final-state MAE and true pre-failure RMSE for fatigue; also report true
  failure prevalence and surviving-state count.
- Passive-tail metrics: tail RMSE, final MAE, change relative to the model's
  own predicted state at step 32, plus violations relative to physical truth.
- Thermal passive temperature-increase/ambient-undershoot rates; fatigue
  decrease/rest-drift/post-failure-return rates. Tolerance is 1e-5 in normalized
  state units (0.001 K for thermal). All fractions have explicit denominators.
- Common predicted failure crossing threshold: 1-1e-4. True failure is reference
  a=1. Report FP/FN among true nonfailures/failures and all-case accuracy.
- Event time: first-crossing observation step; no crossing uses horizon+1.
  Report all-case capped absolute error, true-failure error including missed
  events, and matched-failure-only MAE with its coverage. Endpoint event times
  are quantized to control steps, not continuous physical seconds/cycles.
- Per-seed and mean/sample-SD; three seeds are not a broad significance claim.
- Failure examples: for both models and every seed select worst long-rollout
  RMSE, worst passive-tail error, and worst fatigue event-time error; retain IDs,
  inputs, true state and both models' predictions. Report constrained failures
  even when their physical violation counts are zero.

No evaluation thresholds, training budgets or dataset ranges may be revised
after viewing test results. Correctness fixes must be logged separately and
the affected runs invalidated and rerun without selective reuse.
