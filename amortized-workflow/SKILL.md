---
name: amortized-workflow
description: >
  Opinionated amortized Bayesian workflow with BayesFlow for simulation-based inference (SBI).
  Contains critical guardrails that agents will usually not apply unprompted — always consult
  before writing BayesFlow code. Trigger on: simulation-based inference, amortized inference, approximate Bayesian inference,
  BayesFlow, neural posterior estimation, posterior amortization, simulator design, prior design
  for SBI, offline/online simulation pipelines,
  uncertainty quantification from simulators, structured data encoders (sets, time series, images),
  or mentions of BasicWorkflow, fit_online, fit_offline, fit_disk, flow matching, diffusion model, consistency model, normalizing flow, summary networks, adapters, or simulation budgets.
license: MIT
metadata:
  author:
    - "[Stefan T. Radev](https://bayesops.com)"
    - "[Alexandre Andorra](https://alexandorra.github.io/)"
  version: "2.0"
---

# Amortized Bayesian Workflow

## Workflow overview

Every amortized Bayesian analysis follows this sequence. Do not skip steps — especially simulator validation and model criticism.

1. **Formulate** — Define the generative story. What latent variables or parameters generated the observations?
2. **Specify the simulator regime** — The first iteration always uses **offline training** for fast turnaround, regardless of simulator speed. The simulator regime only determines the simulation budget for the pilot run:
   - **Fast simulator** (< 0.05 s per draw): pre-simulate **20 000** datasets, train for **100 epochs**
   - **Slow simulator** (> 1 s – minutes per draw): pre-simulate **3 000–5 000** datasets, train for **100 epochs**
   - **No simulator / pre-existing bank**: use whatever is available; switch to **disk training** if it does not fit in memory
   Online training is a refinement step — use it only after the first offline pass shows healthy diagnostics and you want to squeeze out more performance.
3. **Define prior + observation model or simulation bank**
   - Implement prior and observation model and wrap them in a simulator
   - Pre-simulate the pilot budget into a dict (using `workflow.simulate(N)`) for offline training
   - If the simulator is external or proprietary, ensure simulations are already generated from the intended prior and data-generating process
4. **Choose the architecture** — this step is critical; getting it wrong ruins inference. See `references/conditioning.md` for the full conditioning logic and decision table.
   - **"Simple vector"** means the observation is a **single fixed-length feature vector** whose element order is meaningful (e.g., 5 named sensor readings, a pre-computed summary statistic). Only then: route through `inference_conditions` with no summary network.
   - **Set-based / exchangeable data** — If the simulator produces **N observations that are exchangeable**, the data is a **set**, not a vector. This includes: N i.i.d. draws, regression datasets with (x, y) pairs, repeated measurements, trial-level data, cross-sectional samples. Route through `summary_variables` with a `SetTransformer`. **Never put this in `inference_conditions`.**
   - **Time series** — ordered sequences: route through `summary_variables` with `TimeSeriesTransformer` or `TimeSeriesNetwork`.
  - **Images as conditions / observations for parameter inference** — route through `summary_variables` with `ConvolutionalNetwork`.
  - **Images as inferential targets** — conditional image generation, spatial field generation, denoising, and other image-valued outputs require an **image-capable diffusion inference network**. Use `bf.networks.DiffusionModel(subnet=...)` with `UNet`, `UViT`, or `ResidualUViT`; see `references/image-generation.md`.
   - **A workflow can use both slots simultaneously.** Fixed-length metadata (e.g., sample size N, scalar design variables) can go in `inference_conditions` while structured observations go in `summary_variables`.
   - **When in doubt, use a summary network.** It is always safer to include one than to omit one; a summary network will always be needed if the data has more than one axis.
5. **Build the workflow** — Prefer `bf.BasicWorkflow(...)`
    - Decide on which variables to auto-standardize. Prefer `standardize="all"` unless you have verfied that the simulator outputs are already in a good range for the networks.
6. **Run simulation sanity checks** — Before training, verify that simulated data look plausible and span the relevant range of real observations. Again, pay attention to what needs to be standardized.
7. **Train the amortizer** — First iteration always uses offline training for fast feedback:
   - `workflow.fit_offline(...)` with the pre-simulated pilot budget (default first pass)
   - `workflow.fit_online(...)` only as a refinement step after offline diagnostics look healthy, or when the user explicitly requests it
   - `workflow.fit_disk(...)` if streaming simulations from disk
   **Always offer to run training in the terminal** so the user can monitor progress interactively.
8. **Diagnose in silico** — Use held-out simulations with known ground truth using the workflow's built-in diagnostics: `workflow.compute_default_diagnostics(...)` for numerical results and `workflow.plot_default_diagnostics(...)` for visual diagnostics.
9. **Amortized inference on real data** — Use `workflow.sample(...)`
10. **Posterior predictive checks (PPCs)** — Re-simulate data from posterior samples and compare to the real data using model-specific test quantities
11. **Write a report** — Use `references/reporting.md` to generate a structured report outlining results and next steps.

## Hard rules — MUST and NEVER

These rules are non-negotiable. Violating any of them will silently produce wrong results.

- **MUST use `bf.Adapter()` for data routing.** Build an explicit adapter chain with `.as_set()`, `.constrain()`, `.concatenate()`, etc. and pass `adapter=` to `BasicWorkflow`, as described in `references/adapter.md`. Do NOT do manual preprocessing — the adapter handles training and inference identically. The naming shorthand (`inference_variables=`, `summary_variables=` as kwargs to `BasicWorkflow`) is ONLY acceptable when the simulator output already has the exact shapes/dtypes the networks expect AND no parameter has bounded support. When in doubt, use an explicit adapter.
- **MUST start with the Base network configuration** from `references/model-sizes.md`. Scale up to Large or XL ONLY if diagnostics show poor recovery or calibration after sufficient training. Oversized networks waste compute and can hurt calibration on simple problems.
- **MUST use `workflow.simulate(N)` to generate train/test data** — not a Python for-loop over `simulator()`. The simulator returned by `bf.make_simulator` is a batched object; `workflow.simulate(N)` calls it efficiently and returns data in the format the workflow expects.
- **MUST use `workflow.compute_default_diagnostics(test_data=...)` and `workflow.plot_default_diagnostics(test_data=...)`** for in-silico diagnostics. NEVER hand-roll coverage, bias, or calibration computations — the built-in methods are correct, complete, and consistent with the house thresholds.
- **For image-valued inference targets, follow `references/image-generation.md`.** MUST use `bf.networks.DiffusionModel(subnet=...)` with `UNet`, `UViT`, or `ResidualUViT` — not the default low-dimensional setup. Conditions must be spatially concatenable with the image target (broadcast `(B, D)` to `(B, H, W, D)`). The standard diagnostic report does not apply; use visual sample grids instead.
- **`workflow.sample()` returns original parameter names, NOT `"inference_variables"`.** The adapter's reverse transform restores the original keys from the simulator (e.g., `"alpha"`, `"beta"`, `"sigma"`). Each parameter has shape `(batch, num_samples)` for scalars or `(batch, num_samples, d)` for vectors. NEVER index into `"inference_variables"` — that key does not exist in the output.
- **MUST reuse the existing simulator functions for PPCs.** NEVER re-implement the generative model by hand for posterior predictive checks. Loop over a subset of posterior draws (50 is a good default), indexing over the `num_samples` axis, and pass each draw through the simulator's forward model.
- **MUST save `history.history` as JSON** (not CSV, not a DataFrame — it is a plain dict). Then run `scripts/inspect_training.py` or call `inspect_history()` in-process.
- **MUST pass `validation_data=` to all `fit_*` calls.** For offline training, hold out ~300 simulations as a separate validation dict. For online training (refinement step only), pass an integer (e.g., `validation_data=300`) to auto-simulate.
- **NEVER mix an explicit `adapter=` with the naming shorthand** (`inference_variables=`, `summary_variables=`, `inference_conditions=` as kwargs). They are mutually exclusive. Passing both causes silent conflicts.
- **NEVER flatten structured data into `inference_conditions`.** Sets, time series, and images MUST go through `summary_variables` with an appropriate summary network.
- **`workflow.plot_default_diagnostics()` ALWAYS returns a `dict[str, Figure]`.** Iterate directly over `.items()` to save figures. Do not type-check or branch on the return type.
- **NEVER skip in-silico diagnostics.** Good training loss does not imply good inference.
- **MUST generate `report.md` after every training + diagnostics run.** Store all artifacts in `<slug>/` (see `references/reporting.md` for naming and structure). Save all diagnostic figures with their standard names, save `metrics.csv`, and produce a self-contained markdown diagnostic report. If real data is available, include the optional real-data sections in the same report. For image-valued targets, skip the standard report and use visual sample grids instead.
- **NEVER use `fit_online` as the first training pass** unless the user explicitly requests it. The first iteration MUST use `fit_offline` with a pre-simulated pilot budget (10k sims for fast simulators, 3k–5k for slow ones) to maximize iteration speed. Online training is a refinement step for subsequent iterations.
- **MUST offer to run training in the terminal** so the user can monitor progress. Training scripts should be runnable standalone; do not silently execute long training runs without giving the user access to the live output.

## Installation

Install BayesFlow and a backend. Prefer JAX unless the user has a strong reason to use PyTorch or TensorFlow.

```bash
pip install "bayesflow"
```

## BayesFlow workflow template

```python
import bayesflow as bf
import numpy as np

# --------------------------------------------------
# 1. Define prior + observation model
# --------------------------------------------------

def my_prior():
    theta = ...
    return {"parameters": theta}

def my_observation_model(parameters):
    x = ...
    return {"observables": x}

# bf.make_simulator returns a BATCHED simulator object.
# Use simulator.sample(batch_size) or workflow.simulate(N) — NEVER loop with simulator() in Python.
simulator = bf.make_simulator([my_prior, my_observation_model])

# --------------------------------------------------
# 2. Choose architecture
# --------------------------------------------------

# See references/conditioning.md for the full conditioning logic and decision table.

# See references/model-sizes.md for different configurations — always start with Base.

# Example summary network for set-based data (exchangeable observations):
summary_net = bf.networks.SetTransformer(...) 


inference_net = bf.networks.FlowMatching(...)
# alternatives:
# bf.networks.StableConsistencyModel() # faster sampler, less performant
# bf.networks.DiffusionModel() # slower sampler, good for image generation
# bf.networks.CouplingFlow(depth=4, transform="spline") # good-old normalizing flow

# --------------------------------------------------
# 3. Build the adapter (MUST use bf.Adapter)
# --------------------------------------------------

# See references/adapter.md for the full API and a step-by-step example.
# The adapter routes simulator output to the correct network slots and handles
# parameter constraints, set assembly, dtype conversion, and concatenation.
adapter = (
    bf.Adapter()
    .as_set(["observables"])              # (N,) -> (N, 1) for SetTransformer
    .constrain("parameters", lower=0)     # if parameters have bounded support
    .convert_dtype("float64", "float32")
    .concatenate(["observables"], into="summary_variables")
    .concatenate(["parameters"], into="inference_variables")
)

# --------------------------------------------------
# 4. Create results folder and workflow
# --------------------------------------------------

import os

results_dir = "<slug>"  # e.g., "churn-model" or "churn-model-v2" for iterations
os.makedirs(results_dir, exist_ok=True)

workflow = bf.BasicWorkflow(
    simulator=simulator,
    inference_network=inference_net,
    summary_network=summary_net,
    adapter=adapter,
    checkpoint_filepath=results_dir,
)

# --------------------------------------------------
# 5. Pre-simulate pilot budget (ALWAYS offline first)
# --------------------------------------------------
# First iteration: pre-simulate a fixed budget for fast turnaround.
# - Fast simulator (< 0.05 s/draw): 20 000 datasets
# - Slow simulator (1 s+ /draw): 3 000–5 000 datasets
# Online training is a REFINEMENT step — only use it after offline
# diagnostics are healthy and you want to squeeze out more performance.

N_PILOT = 20_000  # adjust down for slow simulators
N_VAL = 300

all_sims = workflow.simulate(N_PILOT + N_VAL)

# Split into training and validation sets
train_data = {k: v[:N_PILOT] for k, v in all_sims.items()}
val_data = {k: v[N_PILOT:] for k, v in all_sims.items()}

# --------------------------------------------------
# 6. Train (offline first — fast iteration)
# --------------------------------------------------

history = workflow.fit_offline(
    data=train_data,
    epochs=100, # typically between 100 and 300
    batch_size=32,
    validation_data=val_data,
)

# --- Mandatory: save history and inspect training convergence ---
import json
from scripts.inspect_training import inspect_history

with open(os.path.join(results_dir, "history.json"), "w") as f:
    json.dump(history.history, f)

training_report = inspect_history(history.history)
print(json.dumps(training_report, indent=2))

if not training_report["overall"]["ok"]:
    print("TRAINING ISSUES — address before continuing:")
    for issue in training_report["overall"]["issues"]:
        print(f"  - {issue}")

# --------------------------------------------------
# 7. In-silico diagnostics and reporting
# --------------------------------------------------

test_data = workflow.simulate(300)

# --- Save diagnostic figures (standard names from references/reporting.md) ---
import matplotlib.pyplot as plt

figures = workflow.plot_default_diagnostics(test_data=test_data)
figure_names = {
    "losses": "loss.png",
    "recovery": "recovery.png",
    "calibration_ecdf": "calibration_ecdf.png",
    "coverage": "coverage.png",
    "z_score_contraction": "z_score_contraction.png",
}
for key, fig in figures.items():
    fig.savefig(os.path.join(results_dir, figure_names[key]), dpi=150, bbox_inches="tight")
    plt.close(fig)

# --- Save numerical diagnostics ---
metrics = workflow.compute_default_diagnostics(test_data=test_data, as_data_frame=True)
metrics.to_csv(os.path.join(results_dir, "metrics.csv"))
print(metrics)

# --- Assess and generate report ---
from scripts.check_diagnostics import check_diagnostics, suggest_next_steps

diag_report = check_diagnostics(metrics)
next_steps = suggest_next_steps(training_report, diag_report)

# Generate results_dir/report.md following the template in references/reporting.md.
# Use training_report for the Convergence assessment.
# Use diag_report["summary"] for the Recovery, Calibration, Contraction,
# and Numerical Summary assessments (plain-language ratings per parameter).
# Use next_steps for the Suggested Next Steps section.
# If real data is available, also include the optional real-data sections.

# --------------------------------------------------
# 8. Amortized inference on real data (if any)
# --------------------------------------------------

# The adapter is applied in REVERSE after sampling: the returned dict
# contains the ORIGINAL parameter names from the simulator (e.g.,
# "alpha", "beta", "sigma"), NOT "inference_variables".
# Each parameter has shape (batch, num_samples) for scalars or
# (batch, num_samples, d) for vectors.
real_data = {"observables": x_obs}
samples = workflow.sample(
    conditions=real_data,
    num_samples=1000
)
# e.g. samples["alpha"].shape == (1, 1000, 1)
#      samples["beta"].shape  == (1, 1000, 1)

# --------------------------------------------------
# 9. Posterior predictive checks (custom)
# --------------------------------------------------

# MUST reuse the existing simulator functions for PPCs.
# NEVER re-implement the generative model by hand.
# Loop over a subset of posterior draws (50 is a good default),
# indexing over the num_samples axis:
#
# n_ppc = 50
# for s in range(n_ppc):
#     # Extract single draw (index over num_samples dim)
#     theta_s = {k: samples[k][0, s] for k in ["alpha", "beta", ...]}
#     # Pass through the simulator's forward model
#     x_rep = my_observation_model(**theta_s)
#     # Compare x_rep to x_obs using domain-specific summaries
```

## Online training (refinement step)

Online training generates fresh simulations on the fly during training. **Use it only after the first offline pass shows healthy diagnostics** and you want to squeeze out additional performance with a larger effective simulation budget. It is also the natural choice when the user explicitly requests it.

```python
# Refinement: switch to online training after successful offline iteration
history = workflow.fit_online(
    epochs=200,
    batch_size=32,
    num_batches_per_epoch=100,
    validation_data=300,
)
```

## Disk training

Use disk training when the simulation bank is too large for memory.

```python
def custom_load(path_to_file):
    # load file + preprocessing
    return {"observables": x, "parameters": params}


workflow = bf.BasicWorkflow(
    inference_network=bf.networks.FlowMatching(...),
    summary_network=summary_net,
    inference_variables=["parameters"],
    summary_variables=["observables"],
    ...
)

history = workflow.fit_disk(root="path/to/simulation_bank", load_fn=custom_load, epochs=100, batch_size=32, validation_data=validation_data)
```

## Augmentations

See `references/augmentations.md` for a guide on applying optional transformations during training.

## Architecture defaults

### Adapters

See `references/adapter.md` for the full adapter API and a step-by-step example.

When `BasicWorkflow` receives `inference_variables=`, `summary_variables=`, etc. as keyword arguments, it constructs a **minimal implicit adapter** that only renames and routes those keys. This is sufficient when the simulator output already has the right shapes and dtypes. Use an explicit `bf.Adapter()` chain and pass `adapter=` to the workflow whenever you need structural transforms (`.as_set`, `as_time_series`, `.broadcast`), parameter constraints (`.constrain`), feature engineering (`.sqrt`, `.log`), dtype coercion (`.convert_dtype`), or custom concatenation. The explicit adapter and the naming shorthand are mutually exclusive — do not use both.

### Summary networks

See `references/conditioning.md` for the full conditioning model `p(inference_variables | summary_net(summary_variables), inference_conditions)` and a decision table.

Use a summary network (and route data through `summary_variables`) whenever observations are not a single fixed-length vector with meaningful element order. **Most statistical models produce set-based (exchangeable) data** — N i.i.d. draws, regression datasets, repeated measurements, cross-sectional samples. These MUST use a `SetTransformer` via `summary_variables`, never be flattened and placed in `inference_conditions`.

- **Set-based / exchangeable data (most common case)**:
  - `bf.networks.SetTransformer` — **required default** for any model that generates N exchangeable observations
  - `bf.networks.DeepSet` — simpler baseline (discouraged)
- **Images as conditions / observations for parameter inference**:
  - `bf.networks.ConvolutionalNetwork` - extensible default
- **Time series**:
  - `bf.networks.TimeSeriesNetwork` — simplest default
  - `bf.networks.TimeSeriesTransformer` — stronger sequence model
  - `bf.networks.FusionTransformer` — for more complex sequential structure
As a heuristic, always start by setting the `summary_dim` argument to 2x the number of parameters to be estimated.

- **Custom summary networks** — Only when the user explicitly requests one OR the data modality is non-typical (not images, time series, sets, or vectors). See `references/custom-summary.md`.

### Inference networks

For most posterior approximation tasks, default to one of:

- `bf.networks.FlowMatching()` - multi-step sampling, recommended for a first iteration
- `bf.networks.DiffusionModel()` - multi-step sampling, recommended for high-dimensional targets
- `bf.networks.StableConsistencyModel()` - few-step sampling, less performant, can lose information
- `bf.networks.CouplingFlow()` - single-step sampling, recommended if very fast inference is important

### Image-valued inference targets

If the inferential target is an image, spatial field, or other grid-valued object, use the dedicated image-generation workflow in `references/image-generation.md`.

- Default to `bf.networks.DiffusionModel(...)` with an image-capable subnet.
- Choose subnet complexity in this order: `UNet` < `UViT` < `ResidualUViT`.
- `ConvolutionalNetwork` is a **summary network for image conditions**, not the default choice for image-valued targets.
- Ensure all conditioning information is channel-wise concatenable with the image target; broadcast global conditions from `(B, D)` to `(B, H, W, D)` before concatenation.

## Training inspection

After every `fit_online`, `fit_offline`, or `fit_disk` call, you **must** perform the following steps. Do not skip any of them.

### Always pass `validation_data`

All three `fit_*` methods accept a `validation_data` argument. When a simulator is available, pass an integer (e.g., `validation_data=300`) to auto-simulate validation sets. When training offline or from disk without a simulator, pass a held-out dict of arrays. This enables `val_loss` tracking and overfitting detection.

### Always save and inspect the returned History

`fit_*` returns a `keras.callbacks.History` object. `history.history` is a dict with `"loss"` (always present) and `"val_loss"` (present when `validation_data` is provided). **BayesFlow does not save the history automatically** — you must save it yourself.

After training, always:

1. **Save the history** to a JSON file so it survives the session:
   ```python
   import json
   with open("history.json", "w") as f:
       json.dump(history.history, f)
   ```
2. **Run `scripts/inspect_training.py`** to get a structured convergence report:
   ```bash
   python scripts/inspect_training.py --history history.json
   ```
   Or call it in-process:
   ```python
   from scripts.inspect_training import inspect_history
   report = inspect_history(history.history)
   ```
3. The script checks for: NaN in losses, overfitting (val_loss ratio > 1.1×), under-training (loss still decreasing), and prints a JSON report with go/no-go recommendation.

### Controlling terminal output

All `fit_*` methods pass `**kwargs` through to Keras `model.fit()`. Use `verbose=1` (default) for progress bars, `verbose=2` for one line per epoch, or `verbose=0` to suppress output. Prefer `verbose=1` and remind the user to focus the terminal to follow how the script progresses.

## Diagnostics and reporting

After every training + diagnostics run, you **must** generate a self-contained diagnostic report. See `references/reporting.md` for the full template.

## Scripts

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `scripts/inspect_training.py` | Check training convergence | `--history history.json` | JSON report: NaN, overfitting, under-training |
| `scripts/check_diagnostics.py` | Produce qualitative per-parameter assessments for the report | `--metrics metrics.csv [--history history.json]` | JSON: per-parameter ratings (calibration, recovery, contraction) + summary + next steps |

Both scripts can be run from the command line or imported as Python modules:

```python
from scripts.inspect_training import inspect_history
from scripts.check_diagnostics import check_diagnostics, suggest_next_steps

training_report = inspect_history(history.history)
diag_report = check_diagnostics(metrics)
next_steps = suggest_next_steps(training_report, diag_report)
```

## Diagnostic interpretation

Use `workflow.compute_default_diagnostics(...)` as the **primary** diagnostic interface. Use `workflow.plot_default_diagnostics(...)` as supporting visual evidence for the report.

`check_diagnostics()` converts numeric diagnostics into qualitative per-parameter ratings:

- **calibration** — rated from ECE: `excellent`, `fair`, or `poor`
- **recovery** — rated from NRMSE: `excellent`, `good`, `fair`, or `poor`
- **contraction** — rated from posterior contraction: `high`, `medium`, `low`, or `poor — overconfident` (high contraction + poor calibration)

The output also includes a plain-language `summary` per parameter (e.g., `"excellent calibration; good recovery; high contraction"`) ready to paste into the report.

If diagnostics disagree, trust **calibration** first. A narrow but miscalibrated posterior is worse than a wider calibrated one.

Numeric thresholds are internal to `check_diagnostics()` — do not expose them in the report. Use only the qualitative ratings.

## Posterior predictive checks

PPCs in BayesFlow are always custom and model-dependent. **MUST reuse the existing simulator functions** — NEVER re-implement the generative model by hand.

General recipe:

1. Draw posterior samples `theta_s ~ q(theta | x_obs)` via `workflow.sample()`. The returned dict has the **original parameter names** (e.g., `alpha`, `beta`), each with shape `(batch, num_samples)` or `(batch, num_samples, d)`.
2. Loop over a subset of posterior draws (50 is a good default), indexing over the `num_samples` axis:
   ```python
   n_ppc = 50
   for s in range(n_ppc):
       theta_s = {k: float(samples[k][0, s]) for k in ["alpha", "beta", ...]}
       x_rep = my_observation_model(**theta_s)
       # overlay / compare x_rep to x_obs
   ```
3. Compare `x_rep` to `x_obs` using:
   - raw overlays
   - domain-relevant summary statistics
   - discrepancy measures
   - tail behavior
   - event frequencies
   - temporal or spatial structure
4. If replicated data systematically miss the observed data, improve the simulator before trusting inference

## When things go wrong

| Symptom                          | Likely Cause                                              | Fix                                                                 |
|-----------------------------------|-----------------------------------------------------------|---------------------------------------------------------------------|
| Loss becomes NAN                       | Simulator outputs contain inf/nan or large values         | Inspect simulator outputs; if nan/inf, explore root cause; if large values, ensure workflow uses the proper `standardize` flag or the simulator normalizes outputs (e.g., divide by 255 for images) |
| Recovery good / Calibration bad   | Networks are underexpressive or training is too short       | Train for twice the number of epochs; if not fixed, increase summary capacity to Large |
| Recovery bad / Calibration good  | Some parameters are non-identifiable or same issue as bad recovery | Increase network capacity by two and train for twice as long; if no improvement, parameters may be non-identifiable |
| Nans/inf in samples on real data  | Real data preprocessed differently, contains outliers, or model misspecified | Look at scale of real data; prompt user to test for outliers; check for potential model mis-specification |
| Online training is slow           | Batch size or simulator calls take too long               | Switch to offline /disk training or speed up the simulator |
| Training loss improves but val_loss stays worse or diverges | Overfitting / small simulation budget / excessive capacity | simulate more data, if possible, add `dropout=0.1` or even `dropout=0.2` in `subnet_kwargs` for the inference net and to the init of the summary net (if any); retrain and re-check diagnostics |

## Model sizes

Always check `references/model-sizes.md` for rules on model sizes when choosing a summary backbone and an inference net for a particular problem.