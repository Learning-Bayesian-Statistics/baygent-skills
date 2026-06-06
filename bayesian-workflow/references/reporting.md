# Reporting Bayesian Analyses

## Contents
- Canonical report artifact (`<slug>/report.md`)
- Reporting principles
- Analysis report template (legacy / inline)
- Presentation template
- Visualization standards
- Common reporting mistakes

## Canonical report artifact

After every full analysis run, generate `report.md` inside a dedicated results folder. Static descriptions are verbatim — copy them as-is. `<placeholders>` are dynamic — fill them in from the actual run.

### Results folder naming

All artifacts for a single analysis go into `<slug>/`, where `<slug>` is a short lowercase-hyphenated descriptor of the analysis (e.g., `churn-logistic`, `school-funding-hierarchical`). Choose the most informative 1–3 word name. When iterating on the same problem, append a version: `churn-logistic-v2/`.

```python
import os

results_dir = "<slug>"  # e.g., "churn-logistic"
os.makedirs(results_dir, exist_ok=True)
```

### Output structure

```
<slug>/
├── inference_data.nc            # full InferenceData (idata.to_netcdf)
├── model_graph.png              # pm.model_to_graphviz output
├── prior_predictive.png         # az.plot_ppc(..., group="prior")
├── trace.png                    # az.plot_trace(idata, kind="rank_vlines")
├── forest.png                   # az.plot_forest of posteriors
├── posterior_predictive.png     # az.plot_ppc(idata)
├── pit_ecdf.png                 # azp.plot_ppc_pit (or loo_pit)
├── pit_coverage.png             # azp.plot_ppc_pit(coverage=True)
├── psense.png                   # azp.plot_psense_dist (if sensitivity ran)
├── summary.csv                  # az.summary(idata).to_csv
├── diagnostics.json             # diagnose_model.py output
├── calibration.json             # calibration_check.py output
├── psense.json                  # psense_summary().to_json (if available)
└── report.md                    # this template, filled in
```

### Figure naming convention

Save figures using these exact names so the report template can reference them statically:

```python
import arviz as az
import matplotlib.pyplot as plt

# Trace
ax = az.plot_trace(idata, kind="rank_vlines")
plt.gcf().savefig(os.path.join(results_dir, "trace.png"), dpi=150, bbox_inches="tight")
plt.close()

# Forest
ax = az.plot_forest(idata, combined=True)
plt.gcf().savefig(os.path.join(results_dir, "forest.png"), dpi=150, bbox_inches="tight")
plt.close()

# Posterior predictive
ax = az.plot_ppc(idata)
plt.gcf().savefig(os.path.join(results_dir, "posterior_predictive.png"), dpi=150, bbox_inches="tight")
plt.close()
```

For ArviZ 1.0+ calibration plots (`arviz_plots.plot_ppc_pit`), use `pc.savefig(...)` directly — `scripts/calibration_check.py --save-plots --plot-dir <slug>` does this automatically with the right filenames.

### Report template

Copy this template verbatim into `<slug>/report.md` and fill in the `<placeholders>`. Every paragraph that is not a placeholder is **static** — keep it as-is in every report.

Sections marked **[IF …]** are optional — include them only when the relevant analysis ran. Otherwise omit the entire block (header included).

````markdown
# <Analysis Title> — Bayesian Analysis Report

## Executive Summary

<2–3 sentences summarizing the key finding with credible intervals and the most important caveat. Lead with the substantive conclusion, not the model. If the audience is non-technical, also translate to natural frequencies (e.g., "roughly 9 in 10 chance the effect is positive").>

## Data and Question

| | |
|---|---|
| Source | <where the data came from> |
| Sample size | <N> |
| Key variables | <comma-separated list with brief descriptions> |
| Question | <one-sentence statement of what we want to learn> |

<Brief description of any notable features: missingness, outliers, grouping structure, or transformations applied before modeling.>

## Model Specification

![Model graph](model_graph.png)

The model graph shows the directed structure of the generative process. Plates indicate replicated structure (e.g., over observations or groups). Shaded nodes are observed; unshaded nodes are latent parameters or hyperparameters with priors.

**Generative story.** <1–3 sentences in plain language describing the assumed data-generating process. What was sampled first, what depended on what, and what was finally observed?>

| Parameter | Prior | Justification |
|-----------|-------|---------------|
| <param_1> | <e.g., Normal(0, 2.5)> | <e.g., Weakly informative on standardized predictors> |
| <param_2> | <prior> | <justification> |

## Prior Predictive Check

![Prior predictive](prior_predictive.png)

The prior predictive distribution shows the data the model would generate before seeing any observations, using only the priors. Plausible Bayesian models produce prior predictive samples that span — but do not wildly exceed — the range of the observed data. Tightly bunched priors that exclude the observed range indicate priors that are too narrow; priors that produce wildly implausible values (e.g., negative blood pressure, billion-dollar daily revenue) indicate priors that are too wide and should be tightened before sampling.

**Assessment:** <1–3 sentences — do prior predictive samples span the plausible range of the data? Any signs of over-tight or over-wide priors? If priors were revised after this check, note that here.>

## Sampling and Convergence

| Diagnostic | Value | Threshold | Status |
|------------|-------|-----------|--------|
| Max R-hat | <e.g., 1.003> | ≤ 1.01 | <✓ / ✗> |
| Min ESS (bulk) | <e.g., 1840> | ≥ 100 × n_chains | <✓ / ✗> |
| Min ESS (tail) | <e.g., 1620> | ≥ 100 × n_chains | <✓ / ✗> |
| Divergences | <e.g., 0> | 0 (or near zero) | <✓ / ✗> |

![Trace](trace.png)

Rank-vline trace plots check chain mixing. Well-mixed chains show overlapping rank distributions across chains — the vertical lines (one per chain) sit close to the uniform expectation. Visible separation between chains, monotone drift, or stuck chains indicate non-convergence and the posterior should not be interpreted.

**Assessment:** <1–2 sentences from `check_diagnostics()` convergence section — state whether all diagnostics pass and flag any parameters with R-hat > 1.01, ESS < threshold, or divergences concentrated in their posterior.>

## Posterior

![Forest](forest.png)

The forest plot shows posterior medians (points) and credible intervals (lines) for the parameters of interest. Wide intervals indicate parameters the data are only weakly informative for; narrow intervals concentrated away from zero indicate strong evidence in a direction.

| Parameter | Mean | SD | 94% HDI | P(>0) |
|-----------|------|----|---------|-------|
| <param_1> | <m> | <s> | [<lo>, <hi>] | <prob> |
| <param_2> | <m> | <s> | [<lo>, <hi>] | <prob> |

**Substantive interpretation.** <2–4 sentences on what the posteriors mean in domain terms — effect sizes in original units, practical significance, what the posterior probability of direction implies for the question. Avoid frequentist language ("significant", "rejected").>

## Posterior Predictive Check

![Posterior predictive](posterior_predictive.png)

The posterior predictive distribution shows what the fitted model implies the data should look like. A well-fitting model produces replicated samples that closely overlap the observed data across the full range. Systematic discrepancies — the model under-predicts the tails, misses a mode, or over-disperses — indicate model misspecification and should be addressed before drawing conclusions.

**Assessment:** <1–3 sentences — does the posterior predictive cover the observed data? Any systematic miss (under-dispersed, missing tails, missing modes)? If misspecification is visible, state which aspect of the data the model fails to reproduce.>

## Calibration

![PIT ECDF](pit_ecdf.png)

The PIT-ECDF plot tests whether the model's predictive distribution is calibrated — that is, whether stated credible levels match empirical coverage. The empirical CDF of probability integral transform values should fall within the simultaneous confidence bands. Lines outside the bands above the diagonal indicate under-confident predictions (intervals wider than they should be); lines below indicate over-confident predictions (intervals too narrow).

![Coverage](pit_coverage.png)

The coverage plot tests the same idea in coverage units: it asks whether nominal central credible intervals (50%, 80%, 95%) actually contain the stated fraction of the observed data. A well-calibrated model lies on the diagonal.

**Assessment:** <1–2 sentences from `check_diagnostics()` calibration section — well-calibrated, over-confident, or under-confident, with the mean coverage deviation if available.>

## [IF MODEL_COMPARISON] Model Comparison

| Model | ELPD | SE | ΔELPD | Weight |
|-------|------|----|-------|--------|
| <model_a> | <elpd> | <se> | <delta> | <weight> |
| <model_b> | <elpd> | <se> | <delta> | <weight> |

**Assessment:** <1–3 sentences — which model is preferred, by how much, and whether the preference is robust (ΔELPD > 2 × SE) or marginal.>

## [IF SENSITIVITY] Prior Sensitivity

![Prior sensitivity](psense.png)

| Parameter | Prior CJS | Likelihood CJS | Diagnostic |
|-----------|-----------|----------------|------------|
| <param_1> | <c> | <c> | <Low / Strong prior — justified by …> |

**Assessment:** <1–3 sentences — which parameters are sensitive to prior choice, whether the substantive conclusions depend on those priors, and whether informative priors that flag as sensitive are explicitly justified.>

## Limitations and Threats

This section is mandatory. Rank threats by severity. For each: state the assumption that might be violated, the direction of bias if violated, and what additional data or design would resolve the threat.

1. <threat 1>
2. <threat 2>

## Suggested Next Steps

<Provide 1–5 concrete, actionable steps. Use the output of `scripts/check_diagnostics.py` `suggest_next_steps()` as the starting point; expand with problem-specific context. Don't dump generic advice — tailor it to what this run actually showed.>

1. <step>
2. <step>

## Appendix

<Full ArviZ summary table from `summary.csv`. Code repository link. Random seed used. Software versions (PyMC, ArviZ, nutpie).>
````

### Common "Suggested Next Steps" patterns

Use these as a reference when filling in the final section. The harness in `scripts/check_diagnostics.py` emits these automatically based on what it finds — only override when problem-specific context warrants it.

- All diagnostics healthy, calibration good → "Proceed to interpretation. If decision-stakes warrant, run prior sensitivity (`psense_summary`) and report alongside results."
- Divergences concentrated near a boundary parameter → "Reparameterize the affected parameter as non-centered, or replace `HalfCauchy` with `Gamma(2, …)` for scale priors."
- High R-hat with bimodal trace → "Run more draws and use `init=\"adapt_diag_grad\"`; consider whether the posterior is genuinely multimodal (label-switching, ordering identifiability)."
- Poor PIT calibration with good convergence → "Likelihood is misspecified — consider StudentT for heavy tails, NegBinomial for overdispersed counts, or hierarchical structure for grouped variation."
- LOO Pareto k > 0.7 for some observations → "Investigate the influential points (often outliers or leverage). Consider a more robust likelihood or refit excluding the worst points to test sensitivity."
- Strong prior sensitivity on a parameter → "Either justify the informative prior with domain knowledge, or widen the prior and refit. Report both runs if the substantive conclusion changes."
- Posterior shows non-identifiability (parameters perfectly correlated) → "Either combine the components or restructure data so they're separately identified — see `references/diagnostics.md` on identifiability."

## Reporting principles

Bayesian results are inherently richer than frequentist results -- use that richness.

1. **Report full posteriors**, not just point estimates, with the HDI (Highest Density Interval) as the default summary. **No interval width is magic — 95% least of all.** Choose the width from the decision context rather than convention, and report several when it matters (e.g. a 50% interval for the typical case alongside 89%/94% for a robust range). The default 94% is deliberately off-95 to signal that the number is a *choice*, not a law of nature; a wider interval isn't "more rigorous," it just trades a tighter claim for more coverage.
2. **Visualize uncertainty**. Every parameter estimate should have a visual representation of its posterior.
3. **Show the model**. Include a model specification section -- readers should know exactly what was assumed. Use `pm.model_to_graphviz` to visualize the model.
4. **Report diagnostics**. Convergence and model criticism results build trust.
5. **Use probability language**, not p-value language. "There is a 94% probability that θ lies in [a, b]" — not "the interval [a, b] is significant."

## Analysis report template

Use this structure for written reports. Adapt sections as needed.

```markdown
# [Analysis Title]

## Executive summary
[2-3 sentence summary of key findings with credible intervals]

## Data description
- Source: [where the data came from]
- Sample size: N = [n]
- Key variables: [list with descriptions]
- Notable features: [missingness, outliers, grouping structure]

## Model specification

### Generative story
[Plain-language description of the assumed data-generating process]

### Mathematical notation
[Model equations using standard notation]

### Prior choices
| Parameter | Prior | Justification |
|-----------|-------|---------------|
| β | Normal(0, 2.5) | Weakly informative on standardized predictors |
| σ | Gamma(2, 2) | Allows wide range of residual variation |

### Model graph
[Figure: model graph, probably done with `pm.model_to_graphviz`]

### Prior predictive check
[Figure: prior predictive distribution vs. plausible data range]
[Brief assessment: "Priors generate data in the range [a, b], consistent with domain knowledge."]

## Results

### Convergence diagnostics
- R-hat: all ≤ 1.01 ✓
- ESS (bulk): minimum [X] ✓
- ESS (tail): minimum [X] ✓
- Divergences: [N] [✓ or ✗]
[Figure: trace/rank plots for key parameters]

### Parameter estimates
| Parameter | Mean | SD | 94% HDI |
|-----------|------|----|---------|
| β₁ | 0.45 | 0.12 | [0.22, 0.68] |
| σ | 1.23 | 0.08 | [1.08, 1.38] |

[Figure: forest plot of key parameters]

### Model criticism
- **Posterior predictive check**: [Figure + assessment]
- **LOO-CV**: ELPD = [X] (SE = [Y]), p_loo = [Z]
- **Calibration**: [Figure + assessment]
- **Pareto k**: all < 0.5 ✓ (or list problematic observations)

### Model comparison (if applicable)
| Model | ELPD | SE | ΔELPD | Weight |
|-------|------|----|-------|--------|
[comparison table]

### Prior sensitivity
| Parameter | Prior CJS | Likelihood CJS | Diagnostic |
|-----------|-----------|----------------|------------|
| β₁ | 0.02 | 0.01 | Low sensitivity ✓ |
| σ | 0.08 | 0.03 | Strong prior / weak likelihood — justified by [domain rationale] |

[Brief interpretation: which parameters are sensitive, whether this affects conclusions,
and justification for any retained informative priors.]

## Interpretation
[What do the results mean substantively? Discuss effect sizes, practical significance,
and how uncertainty affects conclusions. Be explicit about what the model does NOT tell us.]

## Limitations
[Model assumptions that may not hold. Sensitivity to prior choices. Data limitations.]

## Appendix
[Full ArviZ summary table. Additional diagnostic plots. Code repository link.]
```

## Presentation template

For slide-based presentations, use this structure:

```
Slide 1: Title + one-sentence finding
Slide 2: The question -- what are we trying to learn?
Slide 3: Data overview (1 figure, minimal text)
Slide 4: Model diagram or generative story (visual, not equations)
Slide 5: Prior predictive check ("Our assumptions produce plausible data")
Slide 6: Key results -- posterior distributions (forest plot or ridgeplot)
Slide 7: Posterior predictive check ("The data could have been credibly produced by the model")
Slide 8: Practical implications -- translate posteriors into decisions
Slide 9: Limitations and next steps
```

**Presentation rules**:
- One idea per slide
- Visualize posteriors, do not just show tables
- Use ridgeplots or forest plots for multiple parameters
- Show uncertainty in predictions (fan charts, spaghetti plots)
- For non-technical audiences: translate credible intervals into natural language ("There is a 5-in-6 chance that the effect is between X and Y")

## Visualization standards

### Parameter posteriors

```python
# Forest plot (multiple parameters)
az.plot_forest(idata, var_names=["beta"], combined=True)

# Ridge plot (distributions)
az.plot_forest(idata, kind='ridgeplot', var_names=["beta"], combined=True)

# Pair plot (correlations between parameters)
az.plot_pair(idata, var_names=["beta", "sigma"])
```

### Predictions with uncertainty

The following are very simple examples (most of the time you will use ArviZ's built-in functions, as well as
work directly with the `InferenceData` object through xarray).
Nevertheless, the following will give you an idea of the *concepts* we're interested in:

```python
# Fan chart for time series predictions
percentiles = [5, 25, 50, 75, 95]
for lo, hi in [(5, 95), (25, 75)]:
    plt.fill_between(
        x,
        np.percentile(preds, lo, axis=0),
        np.percentile(preds, hi, axis=0),
        alpha=0.3,
    )
plt.plot(
    x,
    np.percentile(preds, 50, axis=0),
    color="blue",
    label="Median",
)
plt.scatter(
    x_obs,
    y_obs,
    color="black",
    s=10,
    label="Observed",
)

# Spaghetti plot (individual posterior draws)
for i in range(50):
    plt.plot(x, preds[i], alpha=0.05, color="blue")
```

### Calibration plot

Use the new ArviZ's (1.0+) `plot_ppc_pit` [function](https://python.arviz.org/projects/plots/en/latest/api/generated/arviz_plots.plot_ppc_pit.html). Refer to [this guide](https://arviz-devs.github.io/EABM/Chapters/Prior_posterior_predictive_checks.html#coverage) for guidance about coverage interpretation, but also across the whole Bayesian workflow in general -- it's a treasure trove!

## Adapting for your audience

The report template above is for a technical audience. When the user mentions a non-technical audience (a boss, a medical board, stakeholders, executives), or says they're new to Bayesian stats, adapt the report:

**For non-technical audiences:**
- Generate a **standalone markdown report file** — not just code with inline comments. The report should be readable on its own, without looking at any code.
- Replace jargon with plain language: "There is roughly a 19-in-20 chance the effect is between X and Y" instead of "94% HDI: [X, Y]"
- Include a **glossary** defining terms like posterior, credible interval, prior, MCMC in everyday language
- Lead with the practical conclusion ("The drug lowers blood pressure by about 10 mmHg"), then support it with the statistical evidence
- Move technical details (convergence diagnostics, model equations) to an appendix
- Use the section "How the Analysis Works (Plain Language)" to explain the generative story as a narrative, not equations

**For Bayesian beginners:**
- Explain *why* each step matters, not just what it does: "We check our assumptions first (prior predictive check) to make sure our model doesn't predict impossible values like negative blood pressure"
- Contrast with frequentist approaches where helpful: "Unlike a p-value, a credible interval directly tells you where the parameter probably lies"
- Define acronyms on first use: "HDI (Highest Density Interval — the narrowest range containing 94% of the plausible values)"

## Common reporting mistakes

1. **Reporting only posterior means**: Always include credible intervals. The uncertainty IS the result.
2. **Using frequentist language**: Avoid "significant", "p < 0.05", "fail to reject". Use "probability", "credible interval", "posterior probability of direction."
3. **Hiding diagnostics**: If convergence was imperfect, say so. If you had to fix divergences, describe how.
4. **Ignoring practical significance**: A posterior that excludes zero is not automatically important. Discuss effect sizes in context.
5. **Not showing prior sensitivity**: Run `psense_summary(idata)` and report the results — especially for policy-relevant or controversial conclusions. Show the CJS values table, flag any parameters with CJS > 0.05, and briefly explain whether the sensitivity affects your conclusions. If you have an intentionally informative prior that flags as sensitive, justify it explicitly rather than hiding the diagnostic. Readers should know which conclusions depend on prior choices and which are robust.
6. **Skipping the generative story**: The model specification should make clear what process is assumed to have generated the data.
