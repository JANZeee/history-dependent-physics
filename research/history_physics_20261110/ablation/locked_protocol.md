# Frozen ablation protocol — 2026-11-10

This protocol follows the completed first-round benchmark. Its hypotheses use
that result, so all new final tests use independent seeds. Old results are not
retuned, overwritten, or pooled into the new numerical comparisons.

## Identification and models

A: a 2-by-2 factorial with identical GRUCell(6,64)+Linear(64,1) parameters and
initial weights within seed. The sixth input is either zero or the model's own
previous predicted physical state. A direct next-state candidate is either
unmodified or physically projected. Names: gru_plain, gru_state,
gru_projected, gru_state_projected. The initial state is public to all models.
Head bias is .1 for thermal and .2 for fatigue in all four variants, fixed
before fitting to reduce initially fully blocked projection gradients.

Thermal projection: nonnegative excess temperature; with zero power the next
state cannot exceed the previous state. Fatigue projection:
min(max(candidate, previous_state),1), with exact hold at zero load.
This factorial identifies explicit predicted-state **feedback** and this hard
output projection, not every possible meaning of explicit state representation.
All four keep previous output bookkeeping; unexposed state is not a feature.
Record candidate feasibility/interior fraction and gradient norms. A failed
hard-projection implementation is not evidence against all physical priors.

B: identical explicit-state MLP(6,112,112,1 or 2) within physical task and seed.
Fatigue starts with signed state increment .02*dn*raw (state_free), adds a
normalized stress gate s/1.5 (state_gated), replaces raw with softplus(raw)
(state_positive), adds upper cap 1 (state_full). A separate stronger-prior
intervention state_cubic replaces the linear stress gate with (s/1.5)^3.
No compensating initialization scale: initial delta statistics are recorded.
The cubic prior is deliberately informed by the simulator, not a discovered law.
The gate also changes amplitude dependence; softplus also changes function
shape and initial scale. These contrasts must not be overinterpreted as pure
single-property causal effects. state_positive versus state_full changes only
the cap operator, including its effect during training.

Thermal state_free is a signed affine-in-state residual update using the same
2-output MLP; state_full is the previous positive-source/positive-loss exponential
update. This thermal bridge changes several priors jointly; use A for clean
feedback/projection factor effects.

6 thermal + 9 fatigue models, 3 seeds = 45 independent fixed-budget fits.
Unconstrained models may predict negative/unbounded states. Do not secretly
clamp or discard them. Record nonfinite training/evaluation as failure; no
retries or selective replacement, no 3-seed superiority claim on partial seeds.

## Training and selection

Reuse the original, unmodified training distribution/data seeds 11001/11002:
5,000 train, 512 independent ID validation trajectories per task, length32.
Initialization and minibatch seeds101,202,303. Same 1,500-update minibatch
schedule within seed for all models. Adam lr.001, batch64, clipgrad1, no decay,
CPU FP32 2threads. Same all-step state MSE and full backpropagation, no true
state feedback during training/ordinary prediction. Evaluate ID validation
every100 updates; select minimum ID validation MSE only; train all1500.
Match parameter count within each family exactly; do not claim equal FLOPs.

Record source/config/dataset/minibatch/initial-weight hashes, every selected
checkpoint and trace. All45 training attempts must finish before final tests.
No test-dependent threshold, architecture, update-budget, or data-range changes.

## New tests and metrics

Each condition512 episodes. Generate full256 once and slice nested horizons.
Seed21003 random:32/64/128/256. Seed21004 passive:32 allowed-input intervals
then224 zero input, report128/256. Seed21005 fatigue low-load: s in[.3,.8],
report128/256. Material parameter ranges are unchanged; no claim of material OOD.
Use the reference analytic steps from the previous round without modification.

Reuse the previous state-RMSE, true-pre-failure RMSE, final-state error,
violation tolerance1e-5, predicted fracture threshold.9999, true threshold1,
first-crossing censor time horizon+1, FP/FN with explicit denominators, passive
tail absolute drift and error definitions. 3-seed mean/sampleSD and each seed.
For A report paired simple effects, average main effects and interaction
M11-M10-M01+M00. Negative differences mean lower error, not automatically benefit
for every diagnostic. Report B adjacent paired contrasts and cubic-full.
Retain predeclared worst256-state-RMSE and earliest false-positive cases per
model/seed, showing reference and relevant paired predictions where possible.

## Predeclared mechanism diagnostics

Only explicit-state MLPs admit scalar true-state replacement without changing
the meaning of the hidden representation. Give true previous state to a single
step for oracle local-increment diagnostics; never mix this into ordinary rollout.
Compare exact finite-step increments, not instantaneous differential rates.
Fixed grid: initial a0 in{.08,.14,.20}, current a in{.20,.35,.50,.65,.80,.95},
stress in{0,.3,.4,.5,.6,.7,.8,1,1.2,1.5}, k in{.04,.08,.12},
dn in{.5,1,1.5}; require a>=a0, distinguish true threshold crossings.
Report signed growth-rate bias and error by stress/current crack, and the
training occupancy of true previous crack versus stress. Marginal in-range
load does not guarantee jointly covered state-action history.

On new low-load histories decompose exact one-step prediction error into
g(predicted_previous,u)-g(true_previous,u) plus
g(true_previous,u)-reference_next. These are inherited-error contribution and
oracle local bias, not additive per-step error totals over an entire sequence.

Same trained state_full weights: disable only the upper cap and compare FIRST
threshold crossing. It must be identical; freeze after each first crossing to
avoid irrelevant post-crossing explosion. This can exclude cap inference as
the source of first false positives, not exclude training effects of the cap.

Diagnostic calibration: independently generate low-load validation seed21002,
512x256 then first128. For state_full, on active true-pre-failure non-crossing
steps, fit alpha=sum(predicted_uncapped_delta*true_delta)/sum(predicted_delta^2),
clipped to[.1,2]. Save all3 calibration factors BEFORE constructing final test
data. Compare original weights with alpha1 versus frozen-alpha updates on new
random and low-load128/256, including FP, FN and RMSE. This is a supervised
diagnostic intervention, not part of the equally trained primary models; it
uses extra low-load validation information and may trade false alarms for misses.

No hypothesis that softplus necessarily causes bias: it has no positive hard
floor. A cubic or calibrated-rate improvement establishes only the effect of
that specified intervention, not a universal explanation for neural fatigue.

This is synthetic numerical model research, not video or real-material validation.
