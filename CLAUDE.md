# CLAUDE.md

## Project overview

**baygent-skills** is a collection of Agent Skills for Bayesian modeling, causal inference, and probabilistic thinking. Each skill is a self-contained subfolder following the [Agent Skills spec](https://agentskills.io/specification).

## Repo structure

```
baygent-skills/
├── bayesian-workflow/          # Shipped skill (v1.5, dual PyMC 5/6)
│   ├── SKILL.md                # Main workflow instructions
│   ├── references/             # Detailed reference docs (priors, diagnostics, sensitivity, reporting)
│   └── scripts/                # diagnose_model.py, calibration_check.py, check_diagnostics.py
├── causal-inference/           # Shipped skill (v1.2)
│   ├── SKILL.md                # Main workflow instructions (depends on bayesian-workflow)
│   ├── references/             # DAGs, quasi-experiments, structural models, refutation, reporting
│   └── scripts/                # check_refutation.py (calibrated causal language harness)
├── amortized-workflow/         # Shipped skill (v2.0, co-authored with Stefan Radev)
│   ├── SKILL.md                # Amortized Bayesian workflow with BayesFlow
│   ├── references/             # Adapters, conditioning logic, model sizes, reporting
│   └── scripts/                # check_diagnostics.py, inspect_training.py
├── evals/                         # Eval scenarios and benchmarks
│   ├── bayesian-workflow/         # 6 scenarios, 3 iterations
│   ├── causal-inference/          # 6 scenarios
│   ├── amortized-workflow/        # 6 scenarios + trigger set + benchmark results
│   └── smoke/                     # Reporting-harness smoke test + cross-env (PyMC 5/6) equivalence gate
├── environment.yml             # Mamba/conda env (env name: baygent, PyMC 5)
├── environment-pymc6.yml       # Mamba/conda env (env name: baygent6, PyMC 6 / ArviZ 1.x)
├── LICENSE                     # MIT
└── CLAUDE.md                   # This file
```

## Development conventions

### Python environment
- **Two envs during the PyMC 5 → 6 transition:**
  - `baygent` (PyMC 5.28 / arviz 0.23 + arviz-stats/plots 1.0) — `environment.yml`. The **causal-inference** skill is pinned here (CausalPy caps `pymc<6`).
  - `baygent6` (PyMC 6.0.1 / arviz 1.x + pymc-extras 0.12) — `environment-pymc6.yml`. The **bayesian-workflow** scripts run on **both**; that dual run is the compatibility guarantee.
- Run `conda run -n baygent python <script>` (or `-n baygent6`). Recreate with `mamba env create -f environment.yml` / `mamba env create -f environment-pymc6.yml`
- Never use system Python

### Skill structure
Every skill follows the Agent Skills spec:
- `SKILL.md` with YAML frontmatter (name, description, license, metadata)
- `references/` for detailed docs (plural, not `reference/`)
- `scripts/` for utility scripts
- Description must be agent-neutral (no "Claude"-specific language)

### Skill authoring heuristics
- **The `description` is the only thing the agent sees when deciding to load the skill.** Lead with what it does, then an explicit "Use when …" trigger sentence listing concrete keywords/situations. This is the single highest-leverage field.
- **Keep `SKILL.md` lean; push depth to `references/`.** The main file is the always-loaded budget — progressive disclosure, link out for detail.
- **Prefer a script over generated code for deterministic, repeated, or error-prone operations.** Scripts save tokens and run consistently; reserve inline code for one-off, model-specific logic.

### Code style
- PyMC 5+ syntax with coords and dims; the bayesian-workflow scripts are dual-compatible with PyMC 5 and 6 via capability detection (`hasattr` / try-import / field-name fallbacks), not version branches
- nutpie sampler by default
- Descriptive seeds: `RANDOM_SEED = sum(map(ord, "analysis-name"))`
- xarray-first for InferenceData operations

### Testing
- All evals now in `evals/` (bayesian-workflow, causal-inference, amortized-workflow)
- Benchmark target: 100% with skill vs ~90% without
- Each eval has: `eval_metadata.json` (prompt + assertions), `with_skill/` and `without_skill/` outputs + grading
- **Reporting harness smoke test** (`evals/smoke/test_reporting_harness.py`): runs the bayesian diagnostics pipeline (`diagnose_model → calibration_check → check_diagnostics`) end-to-end on a tiny model and the causal `check_refutation` harness on fixtures. Run after any change to the `scripts/` of either skill — on **both** envs: `conda run -n baygent python evals/smoke/test_reporting_harness.py` and `conda run -n baygent6 python evals/smoke/test_reporting_harness.py`. Guards JSON-serializability, the diagnose→check schema contract, and refutation metric direction.
- **Cross-env equivalence gate** (`evals/smoke/cross_env_equivalence.py`): feeds one shared idata to both `baygent` (PyMC 5) and `baygent6` (PyMC 6) and asserts identical user-facing diagnostics/ratings across a healthy and a pathological fixture — this is the dual-compat guarantee. Run: `python evals/smoke/cross_env_equivalence.py` (needs conda on PATH; skips loudly if `baygent6` is absent, fails with `--require-both`). Run after any change to the bayesian-workflow `scripts/`.
