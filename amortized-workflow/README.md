# amortized-workflow

An opinionated [Agent Skill](https://agentskills.io) for [amortized Bayesian inference](https://learnbayesstats.com/episode/amortized-bayesian-inference-deep-neural-networks-bayesflow-marvin-schmitt) using [BayesFlow](https://bayesflow.org/v2.0.8/index.html) for simulation-based inference (SBI). [Full breakdown](https://learnbayesstats.com/blog-posts/amortized-bayesian-inference-agent-skill-bayesflow).

Compatible with Claude Code, Kimi Code, Cursor, Gemini CLI, and any agent that supports the [Agent Skills spec](https://agentskills.io/specification).

## What it does

Guides your coding agent through the full amortized Bayesian workflow with BayesFlow, enforcing architecture decisions and diagnostic gates that agents skip without prompting:

1. **Formulate** — Define the generative story (latent variables, observation model)
2. **Specify the simulator regime** — Online, offline, or disk training
3. **Choose the architecture** — Summary network selection based on data structure (sets, time series, images), inference network, and model sizing
4. **Build the adapter** — Explicit data routing with parameter constraints, set/time-series marking, and dtype conversion
5. **Train** — With mandatory validation data and convergence inspection
6. **Diagnose in silico** — Built-in diagnostics with house threshold gating (ECE, NRMSE, contraction)
7. **Infer** — Amortized posterior sampling with original parameter names
8. **Posterior predictive checks** — Reusing the simulator's forward model
9. **Report** — Architecture, simulation budget, diagnostics, and limitations

The skill enforces guardrails that agents won't apply on their own: start with small model sizes, gate on diagnostic thresholds before real-data inference, use `SetTransformer` for exchangeable data (not `inference_conditions`), use `DiffusionModel` with `UNet` for image-valued targets, and never hand-roll coverage or calibration metrics.

## Install

This skill is standalone — it does not depend on bayesian-workflow.

### Claude Code

```bash
git clone https://github.com/Learning-Bayesian-Statistics/baygent-skills.git /tmp/baygent-skills
mkdir -p ~/.claude/skills
cp -r /tmp/baygent-skills/amortized-workflow ~/.claude/skills/
```

For project-level installation (available only in that project), copy into `.claude/skills/` at the project root instead.

### Other compatible agents (Kimi Code, Cursor, etc.)

```bash
git clone https://github.com/Learning-Bayesian-Statistics/baygent-skills.git /tmp/baygent-skills
cp -r /tmp/baygent-skills/amortized-workflow/ ~/.config/agents/skills/amortized-workflow/
```

### Python dependencies

```bash
pip install "bayesflow>=2.0"
```

BayesFlow requires a backend (JAX recommended, PyTorch and TensorFlow also supported).

## Example prompts

Once installed, just ask your agent naturally:

- *"I have a simulator for a Gaussian process with 3 hyperparameters. Train an amortized estimator so I can do instant inference on new datasets."*
- *"I want to do amortized inference for a linear regression where different datasets have different sample sizes N, ranging from 10 to 200."*
- *"I have a compartmental SIR model with 4 parameters. Two are positive rates, two are probabilities between 0 and 1. Set up BayesFlow with proper constraint handling."*
- *"I have 20,000 pre-simulated parameter-data pairs as .npz files. The original simulator is gone. Train an amortized estimator offline."*
- *"I want to do Bayesian image denoising — clean images are the targets, blurry images are conditions. Set up BayesFlow with the right architecture for image-valued outputs."*

## Architecture selection guide

| Data structure | Summary network | Route through |
|---|---|---|
| N i.i.d. observations (most common) | `SetTransformer` | `summary_variables` |
| Time series / ordered sequences | `TimeSeriesNetwork` / `TimeSeriesTransformer` / `FusionTransformer` | `summary_variables` |
| Images as observations (target is low-dim) | `ConvolutionalNetwork` | `summary_variables` |
| Images as targets (denoising, generation) | None — use `DiffusionModel(subnet=UNet)` | `inference_conditions` |
| Single fixed-length vector | None | `inference_conditions` |

## What's included

amortized-workflow/
├── SKILL.md                          # Main workflow instructions
├── references/
│   ├── adapter.md                    # Adapter API and step-by-step examples
│   ├── augmentation.md               # Data augmentation policies and examples
│   ├── conditioning.md               # Conditioning logic and decision table
│   ├── custom-summary.md             # Custom summary network guidelines
│   ├── image-generation.md           # Image-valued targets (UNet, DiffusionModel)
│   ├── model-sizes.md                # Small/Base/Large network configurations
│   └── reporting.md                  # Reporting outputs, metrics, and conventions
└── scripts/
    ├── check_diagnostics.py          # Diagnostics checker
    └── inspect_training.py           # Training convergence inspector

## Authors

[Stefan Radev](https://github.com/stefanradev93) and [Alexandre Andorra](https://github.com/alexandorra)

## License

MIT - see [LICENSE](../LICENSE).
