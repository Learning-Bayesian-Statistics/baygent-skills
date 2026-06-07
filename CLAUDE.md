# CLAUDE.md

## Project overview

**baygent-skills** is a collection of Agent Skills for Bayesian modeling, causal inference, and probabilistic thinking. Each skill is a self-contained subfolder following the [Agent Skills spec](https://agentskills.io/specification).

## Repo structure

```
baygent-skills/
├── bayesian-workflow/          # Shipped skill (v1.4)
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
│   └── smoke/                     # Integration smoke test for the reporting harness pipeline
├── environment.yml             # Mamba/conda env definition (env name: baygent)
├── LICENSE                     # MIT
└── CLAUDE.md                   # This file
```

## Development conventions

### Python environment
- Use the `baygent` mamba env: `conda run -n baygent python <script>`
- Recreate with: `mamba env create -f environment.yml`
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
- PyMC 5+ syntax with coords and dims
- nutpie sampler by default
- Descriptive seeds: `RANDOM_SEED = sum(map(ord, "analysis-name"))`
- xarray-first for InferenceData operations

### Testing
- All evals now in `evals/` (bayesian-workflow, causal-inference, amortized-workflow)
- Benchmark target: 100% with skill vs ~90% without
- Each eval has: `eval_metadata.json` (prompt + assertions), `with_skill/` and `without_skill/` outputs + grading
- **Reporting harness smoke test** (`evals/smoke/test_reporting_harness.py`): runs the bayesian diagnostics pipeline (`diagnose_model → calibration_check → check_diagnostics`) end-to-end on a tiny model and the causal `check_refutation` harness on fixtures. Run after any change to the `scripts/` of either skill: `conda run -n baygent python evals/smoke/test_reporting_harness.py`. Guards JSON-serializability, the diagnose→check schema contract, and refutation metric direction.
