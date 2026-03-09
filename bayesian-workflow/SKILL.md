---
name: bayesian-workflow
description: >
  Opinionated Bayesian modeling workflow with PyMC and ArviZ. Contains critical guardrails
  (nutpie sampler, prior/posterior predictive checks, LOO-PIT calibration, 94% HDI, non-centered
  parameterizations, reproducible seeds) that agents won't apply unprompted — always consult
  before writing Bayesian model code. Trigger on: building probabilistic/Bayesian models, prior
  elicitation, MCMC inference, convergence diagnostics (divergences, R-hat, ESS), model comparison
  (LOO-CV, ELPD, stacking weights), hierarchical/multilevel models, count regressions, logistic
  regression with uncertainty, reporting Bayesian results, or mentions of PyMC, ArviZ,
  InferenceData, credible intervals, posterior distributions, shrinkage, uncertainty quantification.
  Also trigger for model comparison, diagnosing sampling problems, choosing priors, or presenting
  stats to non-technical audiences.
license: MIT
metadata:
  author: [Alexandre Andorra](https://alexandorra.github.io/)
  version: "1.0"
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
8. **Compare models** (if applicable) — See [references/model-comparison.md](references/model-comparison.md)
9. **Report results** — See [references/reporting.md](references/reporting.md). When the user asks for a report or mentions a non-technical audience, generate a **standalone markdown report file** (not just code comments) using the template in reporting.md. Adapt the language to the audience — if they're new to Bayesian stats, include a glossary and plain-language explanations of key concepts.

## Installation

Prefer conda-forge / mamba-forge to install PyMC and its dependencies — pip can cause issues with
compiled backends (nutpie, JAX). Example:

```bash
mamba install -c conda-forge pymc nutpie arviz arviz-stats preliz
```

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

    # --- Likelihood ---
    pm.Normal("obs", mu=mu, sigma=1, observed=data, dims="obs")

    # --- Prior predictive check ---
    prior_pred = pm.sample_prior_predictive(random_seed=rng)

    # --- Inference ---
    idata = pm.sample(nuts_sampler="nutpie", random_seed=rng)
    idata.extend(prior_pred)

    # --- Posterior predictive check ---
    idata.extend(pm.sample_posterior_predictive(idata, random_seed=rng))

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
- **Never report point estimates alone**. Always include credible intervals (default: 94% HDI).
- **Use `arviz_stats.diagnose(idata)` as the first diagnostic on every model** (arviz-stats >= 1.0.0). It checks R-hat, ESS, divergences, tree depth saturation, and E-BFMI in one call. Follow up with `az.plot_trace(idata, kind="rank_vlines")` for visual inspection.
- **Don't hardcode number of chains.** Let PyMC / nutpie choose the optimal default for the user's platform. Just call `pm.sample()` without specifying `chains`.
- **Use reproducible, descriptive seeds.** Never use magic numbers like `42`. Instead, derive a seed from the analysis name: `RANDOM_SEED = sum(map(ord, "my-analysis-name"))`. Pass it to `pm.sample(random_seed=rng)`, `pm.sample_prior_predictive(random_seed=rng)`, and numpy via `rng = np.random.default_rng(RANDOM_SEED)`.
- **Save InferenceData immediately after sampling** with `idata.to_netcdf("model_output.nc")`. Late crashes or kernel restarts can destroy valid MCMC results — save before any post-processing.
- **Use ArviZ for all plots and calibration.** Don't write custom plotting code when ArviZ already handles it — including for binary data, count data, and calibration. ArviZ developers have thought through edge cases so you don't have to.
- **Prefer xarray over numpy for InferenceData operations.** `InferenceData` and `DataTree` objects are backed by xarray — use xarray's labeled indexing (`.sel()`, `.mean(dim=...)`, etc.) instead of converting to numpy arrays. This preserves dimension labels, avoids shape bugs, and makes code more readable. Fall back to numpy only when xarray can't do what you need.
- **Always generate analysis notes alongside code.** When producing a model script, also produce a companion markdown file (`analysis_notes.md` or similar) that interprets the results — what the diagnostics mean, what the posteriors tell us, what the calibration plots show. Code without interpretation is incomplete.

## Common model families

| Problem | Likelihood | Typical priors | Reference |
|---|---|---|---|
| Continuous outcome | Normal / StudentT | Normal, Gamma avoiding 0 for positive-constrained parameters | [references/priors.md](references/priors.md) |
| Binary outcome | Bernoulli or Binomial if aggregted, with logit inverse-link | Normal(0, 1.5) on coeffs | [references/priors.md](references/priors.md) |
| Count data | Poisson / NegBinomial | Gamma on rate, avoiding 0 | [references/priors.md](references/priors.md) |
| Hierarchical / multilevel | Varies | See partial pooling pattern | [references/hierarchical.md](references/hierarchical.md) |
| Time series | state space models / Gaussian Processes | Problem-specific | [references/priors.md](references/priors.md) |

## Utility scripts

Run `diagnose_model.py` after sampling to get a structured convergence + diagnostics report:

```bash
python scripts/diagnose_model.py --idata path/to/inference_data.nc
```

Run `calibration_check.py` to generate calibration plots:

```bash
python scripts/calibration_check.py --idata path/to/inference_data.nc
```

See [scripts/](scripts/) for all available utilities.

## Common gotchas

These are battle-tested lessons that save hours of debugging:

- **nutpie silently ignores `idata_kwargs={"log_likelihood": True}`**. If you need log-likelihood (for LOO-CV), call `pm.compute_log_likelihood(idata, model=model)` after sampling. This is a known issue — don't assume it's stored automatically.
- **`az.plot_khat()` requires the LOO object**, not InferenceData. Pass the output of `az.loo(idata, pointwise=True)` to it.
- **Flat priors on scale parameters** (`HalfCauchy`, `HalfFlat`) cause funnels in hierarchical models. Use `Gamma(2, ...)` or `Exponential` — these avoid the near-zero region that creates sampling problems. If there's no group-level variation to detect, you don't need the hierarchy.
- **Python conditionals in models** (`if x > 0`) don't work inside PyMC. Use `pm.math.switch` or `pytensor.tensor.where` instead.
- **Forgetting to standardize predictors** makes shared priors inappropriate and slows sampling. Always standardize before fitting, then back-transform for interpretation.

## When things go wrong

| Symptom | Likely cause | Fix |
|---|---|---|
| Divergences | Posterior geometry issue | Reparameterize (non-centered), increase `target_accept` to 0.95-0.99 |
| Low ESS | High autocorrelation | More tuning steps, reparameterize, reduce correlations |
| R-hat > 1.01 | Chains haven't mixed | More draws, better initialization, check for multimodality |
| Prior pred. looks wrong | Bad priors | Tighten or shift priors, use domain knowledge |
| Post. pred. misses data | Model misspecification | Add complexity (varying slopes, different likelihood, interaction terms) |
| `log_likelihood` missing | nutpie doesn't auto-store it | Call `pm.compute_log_likelihood(idata, model=model)` after sampling |
| Slow model | Large Deterministics or recompilation | Profile with `model.profile(model.logp())`, avoid large `Deterministic` arrays |
