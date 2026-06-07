---
name: bayesian-workflow
description: >
  Opinionated Bayesian modeling workflow with PyMC and ArviZ. Contains critical guardrails
  (nutpie sampler, prior/posterior predictive checks, LOO-PIT calibration, prior sensitivity
  checks, 94% HDI, non-centered parameterizations, reproducible seeds) that agents won't
  apply unprompted — always consult before writing Bayesian model code. Trigger on: building
  probabilistic/Bayesian models, prior elicitation, MCMC inference, convergence diagnostics
  (divergences, R-hat, ESS), model comparison (LOO-CV, ELPD, stacking weights),
  hierarchical/multilevel models, count regressions, logistic regression with uncertainty,
  prior sensitivity analysis, reporting Bayesian results, or mentions of PyMC, ArviZ,
  InferenceData, credible intervals, posterior distributions, shrinkage, uncertainty
  quantification. Also trigger for model comparison, diagnosing sampling problems, choosing
  priors, or presenting stats to non-technical audiences.
license: MIT
metadata:
  author: [Alexandre Andorra](https://alexandorra.github.io/)
  version: "1.5"
---

# Bayesian Workflow

## Workflow overview

Every Bayesian analysis follows this sequence. Do not skip steps -- especially model criticism.

1. **Formulate** — Define the generative story. What underlying process, that we're precisely trying to model, created the data?
2. **Specify priors** — See [references/priors.md](references/priors.md)
3. **Implement in PyMC** — Write the model. Prefer PyMC 5+ syntax. Use the latest version possible.
4. **Run prior predictive checks** — `pm.sample_prior_predictive()`. Verify priors produce plausible data ranges before fitting
5. **Inference** — `pm.sample(nuts_sampler="nutpie")`. Always use nutpie for speed (the nutpie python package provides cutting-edge sampling). Don't hardcode the number of chains — let the sampler pick the best default for the platform.
6. **Diagnose convergence** — Use `arviz_stats.diagnose(idata)` as the first check (requires arviz-stats >= 1.0.0). It covers R-hat, ESS, divergences, tree depth, and E-BFMI in one call. See [references/diagnostics.md](references/diagnostics.md)
7. **Criticize the model** — See [references/model-criticism.md](references/model-criticism.md)
8. **Check prior sensitivity** — Run `psense_summary(idata)` to verify conclusions are robust to prior choices. Visualize with `plot_psense_dist(idata)` from `arviz_plots`. Requires `log_likelihood` and `log_prior` in the InferenceData — compute them after sampling if needed. See [references/sensitivity.md](references/sensitivity.md)
9. **Compare models** (if applicable) — See [references/model-comparison.md](references/model-comparison.md)
10. **Report results** — Generate `<slug>/report.md` using the canonical template in [references/reporting.md](references/reporting.md). Run `scripts/check_diagnostics.py` to turn raw diagnostics into qualitative ratings + an ordered next-steps list, and use that output to fill the Assessment lines and Suggested Next Steps section. When the user mentions a non-technical audience or is new to Bayesian stats, additionally adapt the prose to plain language and include a glossary — but keep the canonical report structure as the audit trail.

## Installation

Prefer conda-forge / mamba-forge to install PyMC and its dependencies — pip can cause issues with
compiled backends (nutpie, JAX). Example:

```bash
mamba install -c conda-forge pymc nutpie arviz arviz-stats preliz
```

The repo ships two pinned environments: `environment.yml` (`baygent`, PyMC 5) and
`environment-pymc6.yml` (`baygent6`, PyMC 6 + ArviZ 1.x) — see Stack compatibility below.

## Stack compatibility (PyMC 5.x and 6.x)

This skill teaches the **latest** PyMC 6 / ArviZ 1.x idioms and stays runnable on
PyMC 5.x during the transition (regulated/corporate environments can't always
upgrade freely). The reporting-harness smoke test and
`evals/smoke/cross_env_equivalence.py` are run on **both** stacks — that dual run
is the compatibility guarantee.

Most code is identical across versions. Where an API genuinely diverges, prefer the
form that runs on **both**:

| Task | Modern (PyMC 6 / ArviZ 1.x) | Legacy (PyMC 5 / ArviZ 0.23) |
|---|---|---|
| Posterior-predictive plot | `arviz_plots.plot_ppc_dist(idata)` — runs on both | `az.plot_ppc(idata)` (removed in ArviZ 1.x) |
| Trace / rank plot | `az.plot_trace(idata, var_names=[...])`; rank view `az.plot_rank(idata, var_names=[...])` — both run on both, but **pass `var_names`**: ArviZ 1.x errors when the auto-selected set exceeds its subplot cap (e.g. a vector `Deterministic` like `mu` with an `obs` dim) | `az.plot_trace(idata, kind="rank_vlines")` (the `kind=` arg is 0.23-only) |
| Prior-predictive draws | `pm.sample_prior_predictive(draws=500)` — runs on both | `samples=500` (removed in PyMC 6) |
| Summary interval | `az.summary(idata, ci_prob=0.94, ci_kind="hdi")` (ArviZ 1.x **only**) | `az.summary(idata, hdi_prob=0.94)` (ArviZ 0.23 **only**) |
| Sampler output type | `DataTree` | `InferenceData` |

`arviz_plots` (the ArviZ 1.x plotting package, imported as `azp`) installs on **both**
stacks, so leading with `azp.*` plots is the most portable choice. Bare `az.summary(idata)`
works on both, but there is no single interval kwarg that does — `ci_prob`/`ci_kind` is
ArviZ 1.x and `hdi_prob` is ArviZ 0.23 (the bare default also differs: 89% ETI vs 94% HDI).
`arviz_stats.diagnose`, `az.loo`, `az.plot_forest/energy/pair/khat`,
and `idata.to_netcdf()` are unchanged on both; `InferenceData` and `DataTree` are both
xarray-backed, so downstream `.sel()` / `az.*` calls are identical.

## PyMC model template

```python
import pymc as pm
import arviz as az
import numpy as np

RANDOM_SEED = sum(map(ord, "churn-logistic-v1"))
rng = np.random.default_rng(RANDOM_SEED)

# always use dimensions and coordinates in PyMC models
with pm.Model(coords=coords) as model:
    # use Data containers when working on a PyMC model
    data = pm.Data("data", df["y"].to_numpy(), dims="obs")

    # --- Priors ---
    # Always document WHY each prior was chosen
    mu = pm.Normal("mu", mu=0, sigma=10)  # Weakly informative: allows wide range

    # --- Data model ---
    pm.Normal("obs", mu=mu, sigma=1, observed=data, dims="obs")

    # --- Prior predictive check ---
    prior_pred = pm.sample_prior_predictive(random_seed=rng)

    # --- Inference ---
    idata = pm.sample(nuts_sampler="nutpie", random_seed=rng)
    idata.extend(prior_pred)

    # --- Posterior predictive check ---
    idata.extend(pm.sample_posterior_predictive(idata, random_seed=rng))

    # --- Compute log-likelihood and log-prior for sensitivity checks & LOO ---
    pm.compute_log_likelihood(idata, model=model)
    pm.compute_log_prior(idata, model=model)

    # --- Save immediately after sampling ---
    # Late crashes can destroy valid results. Save to disk before any post-processing.
    idata.to_netcdf("model_output.nc")
```

## Critical rules

- **Always run prior predictive checks** before sampling. If prior predictions span implausible ranges, fix priors first. If you have issues or doubts for some parameters, use the [PreliZ](https://preliz.readthedocs.io/en/latest/) package to elicit priors from the user.
- **Always check convergence** before interpreting results. R-hat > 1.01 or ESS < 100 * nbr_chains means the results are unreliable.
- **Always run posterior predictive checks**. A model that fits well numerically but cannot reproduce the data is useless.
- **Always run calibration checks** (PIT / coverage). Use ArviZ's `plot_ppc_pit` for this — it handles all data types (continuous, binary, count) correctly. See [references/model-criticism.md](references/model-criticism.md).
- **Document every prior choice** with a brief justification in a code comment.
- **Never report point estimates alone**. Always include credible intervals — a 94% HDI is a fine default, but no interval width is magic (see [references/reporting.md](references/reporting.md)).
- **Use `arviz_stats.diagnose(idata)` as the first diagnostic on every model** (arviz-stats >= 1.0.0). It checks R-hat, ESS, divergences, tree depth saturation, and E-BFMI in one call. Follow up with `az.plot_trace(idata, var_names=[...])` for visual inspection, or `az.plot_rank(idata, var_names=[...])` for rank-based convergence views (both run on both stacks). Pass `var_names` to focus on the parameters — ArviZ 1.x errors if the auto-selected set (e.g. a vector `Deterministic` over an `obs` dim) exceeds its subplot cap. The older `az.plot_trace(idata, kind="rank_vlines")` is ArviZ-0.23-only.
- **Don't hardcode number of chains.** Let PyMC / nutpie choose the optimal default for the user's platform. Just call `pm.sample()` without specifying `chains`.
- **Use reproducible, descriptive seeds.** Never use magic numbers like `42`. Instead, derive a seed from the analysis name: `RANDOM_SEED = sum(map(ord, "my-analysis-name"))`. Pass it to `pm.sample(random_seed=rng)`, `pm.sample_prior_predictive(random_seed=rng)`, and numpy via `rng = np.random.default_rng(RANDOM_SEED)`.
- **Save InferenceData immediately after sampling** with `idata.to_netcdf("model_output.nc")`. Late crashes or kernel restarts can destroy valid MCMC results — save before any post-processing.
- **Use ArviZ for all plots and calibration.** Don't write custom plotting code when ArviZ already handles it — including for binary data, count data, and calibration. ArviZ developers have thought through edge cases so you don't have to.
- **Prefer xarray over numpy for InferenceData operations.** `InferenceData` and `DataTree` objects are backed by xarray — use xarray's labeled indexing (`.sel()`, `.mean(dim=...)`, etc.) instead of converting to numpy arrays. This preserves dimension labels, avoids shape bugs, and makes code more readable. Fall back to numpy only when xarray can't do what you need.
- **Always generate `<slug>/report.md` after a full analysis run.** Store all artifacts (`inference_data.nc`, `trace.png`, `forest.png`, `posterior_predictive.png`, `pit_ecdf.png`, `summary.csv`, `diagnostics.json`, `calibration.json`) in a slug-named results folder, and produce `report.md` from the canonical template in [references/reporting.md](references/reporting.md). Code without an interpreted, fixed-shape report is incomplete.
- **Use `scripts/check_diagnostics.py` to interpret diagnostics, not hand-rolled prose.** Pipe the JSON outputs of `diagnose_model.py` (and optionally `calibration_check.py` and `psense_summary`) into `check_diagnostics.py` to get per-section qualitative ratings and an ordered, actionable Suggested Next Steps list. Use those outputs verbatim in the report's Assessment lines; expand only with problem-specific context.
- **Always use the posterior mean (not median) for predictive probabilities.** The proper Bayesian predictive distribution averages over the posterior: `P(Y=k|x) = (1/S) Σ P(Y=k|x,θₛ)`. This is the mean, not the median. The median does not correspond to the posterior predictive distribution, can violate probability coherence (probabilities may not sum to 1), and biases calibration due to Jensen's inequality. In code: use `np.mean(probs, axis=sample_axis)`, never `np.median(...)`.
- **Use `pm.set_data()` + `pm.sample_posterior_predictive()` for out-of-sample predictions.** Don't manually extract posterior samples and recompute predictions — let PyMC propagate uncertainty properly. Define predictors as `pm.Data(...)` during model building, then swap in new data:

```python
# After fitting the model:
with model:
    pm.set_data({"X": X_new, "group_idx": group_idx_new})
    oos_preds = pm.sample_posterior_predictive(idata, predictions=True, random_seed=rng)
```

- **Check model identifiability before interpreting components.** If two model components always appear together in the likelihood (e.g., a league intercept and a home advantage term when every observation is from home perspective), their individual posteriors reflect prior assumptions, not data signal — only their sum is identified. Use `az.plot_pair()` to check for strong posterior correlations between components. If correlation is near ±1, the components are not separately identifiable — either merge them or restructure the data.

## Common model families

| Problem | Data model | Typical priors | Reference |
|---|---|---|---|
| Continuous outcome | Normal / StudentT | Normal, Gamma avoiding 0 for positive-constrained parameters | [references/priors.md](references/priors.md) |
| Binary outcome | Bernoulli or Binomial if aggregated, with logit inverse-link | Normal(0, 1.5) on coeffs | [references/priors.md](references/priors.md) |
| Count data | Poisson / NegBinomial | Gamma on rate, avoiding 0 | [references/priors.md](references/priors.md) |
| Count data with excess zeros | ZeroInflatedPoisson / ZeroInflatedNegBinomial | Gamma on rate; Beta or Normal+logit on zero-inflation prob | [references/priors.md](references/priors.md) |
| Positive count data (no zeros) | Hurdle Poisson / Hurdle NegBinomial | Separate zero-gate (Bernoulli) and count (Truncated) components | [references/priors.md](references/priors.md) |
| Ordinal outcome | OrderedLogistic (cumulative link) | Normal on coeffs; Normal with ordered transform on cutpoints | [references/priors.md](references/priors.md) |
| Censored data (survival, limits of detection) | `pm.Censored(dist, lower, upper)` | Same as uncensored, applied to underlying distribution | [references/priors.md](references/priors.md) |
| Truncated data | `pm.Truncated(dist, lower, upper)` | Same as underlying distribution | [references/priors.md](references/priors.md) |
| High-dimensional / sparse regression | Normal / StudentT with sparsity prior on coefficients | Regularized Horseshoe or R2-D2 on coeffs | [references/priors.md](references/priors.md) |
| Hierarchical / multilevel | Varies | See partial pooling pattern | [references/hierarchical.md](references/hierarchical.md) |
| Time series | state space models / Gaussian Processes | Problem-specific | [references/priors.md](references/priors.md) |

## Utility scripts

Run these in order — each script's output feeds the next.

```bash
# 1. Run convergence + LOO + PPC checks (writes diagnostics.json)
python scripts/diagnose_model.py --idata <slug>/inference_data.nc --output <slug>/diagnostics.json

# 2. Run calibration check (writes calibration.json + pit_ecdf.png + pit_coverage.png)
python scripts/calibration_check.py --idata <slug>/inference_data.nc --output <slug>/calibration.json --save-plots --plot-dir <slug>/

# 3. Interpret the JSON outputs into qualitative ratings + suggested next steps
python scripts/check_diagnostics.py --diagnostics <slug>/diagnostics.json --calibration <slug>/calibration.json --output <slug>/check_report.json
```

Step 3 is what powers the `report.md` Assessment lines and Suggested Next Steps section — never hand-roll those interpretations from raw R-hat / ESS / pareto-k numbers when the harness can produce them consistently.

See [scripts/](scripts/) for all available utilities.

## Common gotchas

These are battle-tested lessons that save hours of debugging:

- **nutpie silently ignores `idata_kwargs`** for `log_likelihood` and `log_prior`. Always compute them explicitly after sampling: `pm.compute_log_likelihood(idata, model=model)` (needed for LOO-CV) and `pm.compute_log_prior(idata, model=model)` (needed for prior sensitivity checks). Don't assume they're stored automatically.
- **`az.plot_khat()` requires the LOO object**, not InferenceData. Pass the output of `az.loo(idata, pointwise=True)` to it.
- **Flat priors on scale parameters** (`HalfCauchy`, `HalfFlat`) cause funnels in hierarchical models. Use `Gamma(2, ...)` or `Exponential` — these avoid the near-zero region that creates sampling problems. If there's no group-level variation to detect, you don't need the hierarchy.
- **Python conditionals in models** (`if x > 0`) don't work inside PyMC. Use `pm.math.switch` or `pytensor.tensor.where` instead.
- **Forgetting to standardize predictors** makes shared priors inappropriate and slows sampling. Always standardize before fitting, then back-transform for interpretation.
- **Horseshoe priors create a double-funnel geometry** that standard NUTS can struggle with. Always use the **regularized (Finnish) horseshoe** (Piironen & Vehtari, 2017), which adds a slab component that smooths the geometry. Set `target_accept=0.95` or higher. If you see divergences with a horseshoe model, this is almost certainly the cause.
- **`np.median` on posterior predictive probabilities is a silent bug.** It does not produce the Bayesian predictive distribution and can yield probabilities that don't sum to 1 across categories. Always use `np.mean` over the posterior samples dimension.
- **Discrete latents: marginalize, don't plug in.** NUTS-only samplers (nutpie, Pathfinder) cannot sample discrete or `ordered` variables at all; PyMC's default `pm.sample()` falls back to a compound Metropolis step for them, which mixes poorly. Prefer to **marginalize** the discrete latents — `pmx.marginalize(model, [...])` or analytic integration — and feed a **true mixture likelihood** downstream (exact, O(K) per observation, NUTS-compatible). Plugging a soft relaxation (soft-min/argmax, or `E[z]`) into a nonlinear function is mathematically wrong: it is not the marginal and can return out-of-bounds values. Any mixture also needs an identification constraint (e.g. `ordered` components) or chains will label-switch.
- **Overlapping data subsets in a likelihood double-count.** When a likelihood is assembled from per-subset terms, the subsets must partition the data *disjointly* — an observation that lands in two terms is counted twice, silently over-shrinking the posterior. Partition disjointly, or model the overlap explicitly.

## When things go wrong

| Symptom | Likely cause | Fix |
|---|---|---|
| Divergences | Posterior geometry issue | Reparameterize (non-centered), increase `target_accept` to 0.95-0.99 |
| Low ESS | High autocorrelation | More tuning steps, reparameterize, reduce correlations |
| R-hat > 1.01 | Chains haven't mixed | More draws, better initialization, check for multimodality |
| Prior pred. looks wrong | Bad priors | Tighten or shift priors, use domain knowledge |
| Post. pred. misses data | Model misspecification | Add complexity (varying slopes, different data model, interaction terms) |
| `log_likelihood` missing | nutpie doesn't auto-store it | Call `pm.compute_log_likelihood(idata, model=model)` after sampling |
| Slow model | Large Deterministics or recompilation | Profile with `model.profile(model.logp())`, avoid large `Deterministic` arrays |
| Slow to initialize / poor warmup | Bad starting point | Try `init="adapt_diag_grad"` in `pm.sample()`, or run `pmx.fit(method="pathfinder")` first (`import pymc_extras as pmx`) and pass its estimates as `initvals` |
| Prior sensitivity flag | Prior-data conflict or strong prior | Check `psense_summary(idata)` — see [references/sensitivity.md](references/sensitivity.md). Justify or revise the flagged prior |
