# Benchmark — bayesian-workflow PyMC 5/6 dual-compatibility

Goal: make the **bayesian-workflow** skill teach the latest PyMC 6 / ArviZ 1.x idioms
while staying runnable on PyMC 5.x for the transition. Method: build both envs, run the
harness on each, and let breakage reveal the real divergences (not a changelog).

## Environments

| Env | PyMC | ArviZ umbrella | arviz-stats/plots | nutpie | pymc-extras | sampler output |
|-----|------|----------------|-------------------|--------|-------------|----------------|
| `baygent`  | 5.28.1 | 0.23.4 (classic) | 1.0.0 | 0.16.7  | 0.10.0 | `InferenceData` |
| `baygent6` | 6.0.1  | 1.1.0 (umbrella) | 1.1.0 | 0.16.10 | 0.12.0 | `DataTree` |

`environment-pymc6.yml` resolves PyMC 6 + the full ArviZ 1.x stack + nutpie + pymc-extras
cleanly. It also pins `netcdf4` + `h5netcdf`: a pip-only ArviZ-1 install pulls no
group-aware netcdf engine, so without them `az.from_netcdf` / `convert_to_datatree(path)`
can't read a `.nc` (the scripts' first step).

## Divergences found by running on both stacks, and the fixes

| Divergence | Symptom on PyMC 6 / ArviZ 1.x | Fix |
|---|---|---|
| `difference_ecdf_pit` relocated (arviz_plots 1.0 → arviz_stats 1.1) | `calibration_check.py` failed to **import** | import from the stable `arviz_stats.ecdf_utils` home (both stacks), with a fallback |
| `InferenceData.groups()` (method) vs `DataTree.groups` (property of paths) | LOO **silently dropped** — `"'tuple' object is not callable"`, swallowed by check_loo's except | `_group_names()` normalizes both to bare names |
| ELPDData field renames (`elpd_loo→elpd`, `p_loo→p`) | `check_loo` would `AttributeError` after the groups fix | `_loo_field()` tries each name |
| `Dataset.to_array()` removed in modern xarray | R-hat "max" perpetually `null` (both envs, latent) | reduce per-variable; empty (all-converged) → `None` |
| arviz 1.x returns **NaN** Pareto-k for degenerate points; 0.23 smoothed | a bare `NaN` would leak into the JSON report (invalid JSON) | take the max over finite k only; count non-finite as bad |
| no netcdf engine in a fresh pip ArviZ-1 env | `.nc` read/write fails entirely | add `netcdf4` + `h5netcdf` to the env |

`arviz_stats.diagnose` (the primary convergence call), `az.loo`, `az.summary`,
`az.from_netcdf`, `convert_to_datatree`, and the diagnose→check schema are unchanged.

## Cross-env equivalence gate (`evals/smoke/cross_env_equivalence.py`)

One shared idata is fed to **both** envs; the comparison is partitioned by who owns each
difference: `strict` (exact) for the safety-critical verdict, `numeric` (tolerance) for
quantities arviz estimates differently, `info` (reported, non-gating) for the
threshold-sensitive LOO Pareto-k rating.

| Fixture | convergence | calibration | result |
|---|---|---|---|
| healthy (well-behaved regression) | excellent = excellent | excellent = excellent | **identical** |
| pathological (centered eight-schools funnel) | poor = poor (divergences, same flagged params) | excellent = excellent | **strict identical** |

Documented limitation (surfaced by the pathological fixture, not hidden): arviz 0.23 and
1.x use different PSIS tail estimators, so on a divergent fit the **finite** Pareto-k
values match but 1.x marks one degenerate point `NaN` where 0.23 smoothed it — flipping
the qualitative LOO rating (excellent ↔ poor). This is upstream, not our bug; the
convergence verdict (the "don't interpret this posterior" guidance) agrees exactly, and
LOO is not trustworthy on a non-converged model anyway.

## Taught-API resolution sweep (checkpoint: every call resolves on both)

Verified identical on both envs: `arviz_stats.diagnose`, `psense_summary`,
`plot_psense_dist` / `plot_psense_quantities`, `pm.compute_log_likelihood` /
`compute_log_prior`, `pmx.marginalize` / `MarginalModel` / `fit`, `preliz`, and the
distribution families (`Censored`, `Truncated`, `OrderedLogistic`, `ZeroInflatedPoisson`,
`HurdlePoisson`, `NegativeBinomial`). The only APIs that genuinely differ are the handful
documented in SKILL.md → "Stack compatibility" (`az.plot_ppc` → `arviz_plots.plot_ppc_dist`;
`plot_trace(kind="rank_vlines")` → `plot_trace`/`plot_rank`; `summary` interval kwargs;
`sample_prior_predictive` `samples=`→`draws=`; `az.compare` `elpd_loo`→`elpd`).

## Scope

- **amortized-workflow** imports only `bayesflow` / `keras` / `numpy` — no pymc/arviz
  coupling, so it is independent of the PyMC major (its own backend env, unaffected).
- **causal-inference** stays on PyMC 5 (`baygent`): CausalPy caps `pymc<6`. Migrate when
  CausalPy ships PyMC-6 support.

## Verdict

The bayesian-workflow diagnostics scripts run on both PyMC 5 and PyMC 6 and agree on the
**safety-critical verdict** (convergence + calibration ratings, structural flags, and
`loo_computed`) for the same idata. The one place they can differ — the LOO Pareto-k
qualitative rating on a degenerate fit — is an upstream PSIS difference (arviz 1.x marks a
point's k non-finite where 0.23 smoothed it), is reported honestly on each stack, and is
immaterial because LOO is untrustworthy on a non-converged model (the convergence verdict,
which agrees, already says "don't interpret"). Gates green on both envs:
`test_reporting_harness.py` (16/16 each) and `cross_env_equivalence.py` (healthy +
pathological). Every API the skill teaches resolves on both — with the version-specific
forms (the `az.summary` interval kwargs; `var_names=` on `plot_trace`/`plot_rank` to stay
under ArviZ 1.x's subplot cap) spelled out in SKILL.md → "Stack compatibility".
