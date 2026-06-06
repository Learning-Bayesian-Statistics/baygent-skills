# Reporting Causal Analyses

## Contents

1. [Canonical report artifact](#canonical-report-artifact)
2. [Causal analysis report template (legacy / inline)](#causal-analysis-report-template)
3. [Causal language guardrails](#causal-language-guardrails)
4. [Decision-relevant HDIs](#decision-relevant-hdis)
5. [Audience adaptation](#audience-adaptation)
6. [Common reporting mistakes](#common-reporting-mistakes)

---

## Canonical report artifact

Every causal analysis writes `report.md` inside a dedicated results folder. Static descriptions are verbatim — copy them as-is. `<placeholders>` are dynamic — fill them in from the actual run. Sections marked **[IF …]** are design-specific and should be included only when applicable.

### Results folder naming

Slug pattern: `<treatment>-on-<outcome>/`. Use lowercase-hyphenated, 1–4 words. For example: `policy-on-test-scores`, `exercise-on-bp`, `tariff-on-export-volume`. When iterating, append a version: `policy-on-test-scores-v2/`.

```python
import os

results_dir = "<treatment>-on-<outcome>"  # e.g., "policy-on-test-scores"
os.makedirs(results_dir, exist_ok=True)
```

### Output structure

```
<treatment>-on-<outcome>/
├── inference_data.nc            # full InferenceData from estimation
├── dag.png                      # rendered DAG (CausalModel.plot or graphviz)
├── effect_posterior.png         # posterior of the causal effect
├── forest.png                   # forest of all relevant parameters
├── parallel_trends.png          # [IF DiD]
├── pre_treatment_fit.png        # [IF SC]
├── placebo_density.png          # [IF SC] distribution of placebo effects
├── density_test.png             # [IF RDD] McCrary density
├── bandwidth_sensitivity.png    # [IF RDD]
├── autocorrelation.png          # [IF ITS]
├── sensitivity_tipping.png      # unobserved confounder tipping point
├── refutation.json              # structured refutation outcomes
├── identification.json          # identified estimand + adjustment set
├── effect_summary.csv           # effect estimates + HDIs at multiple widths
└── report.md                    # this template, filled in
```

### Report template

Copy this template verbatim into `<results-folder>/report.md` and fill in the `<placeholders>`. Keep static paragraphs as-is.

````markdown
# <Causal Question> — Causal Analysis Report

## 1. Causal Question

<One sentence: "What is the effect of <treatment> on <outcome> in <population>?" If you cannot write this sentence, you do not have a causal question yet — you have a dataset.>

**Estimand:** <ATE / ATT / LATE / CATE — be specific, including the conditioning set if applicable.>

## 2. DAG and Assumptions

![Causal graph](dag.png)

The directed acyclic graph encodes which variables are assumed to cause which others. Edges express direct causal effects assumed by the analyst. The *absence* of an edge between two variables is the most consequential assumption — it asserts there is no direct causal effect, even after conditioning. Identification of the causal effect depends entirely on this graph being correct.

| Assumption | Testable? | Fragility | What if violated? |
|------------|-----------|-----------|-------------------|
| <e.g., No unobserved confounders> | <No / Partially / Yes> | <Robust / Moderate / Fragile> | <e.g., Effect biased in unknown direction> |

<For each fragile assumption listed above, briefly state what evidence (if any) supports it. Untestable assumptions still need a defensible argument.>

## 3. Identification Strategy

<"We use <method (backdoor / frontdoor / DiD parallel trends / RDD continuity / IV exclusion+relevance / structural)> to identify the causal effect. This is valid because <justification, in 1–2 sentences>.">

**Adjustment set / instrument / running variable:** <list the variables>

**[IF NOT POINT-IDENTIFIED]** The effect is not point-identified under the available assumptions. We report bounds rather than a point estimate.

## 4. Estimation

**Design:** <DiD / Synthetic Control / RDD / ITS / IV / IPSW / Structural>

**Model:** <likelihood, link, key priors — link to bayesian-workflow's diagnostics standards rather than restating them>

**Convergence summary:** <one line — R-hat, ESS, divergences. Use the bayesian-workflow harness if available.>

## 5. Results

![Effect posterior](effect_posterior.png)

The posterior distribution of the causal effect is the result. Point estimates and intervals are summaries of this posterior.

| | Value |
|---|---|
| Posterior median | <value in domain units> |
| 50% HDI | [<lo>, <hi>] |
| 89% HDI | [<lo>, <hi>] |
| 95% HDI | [<lo>, <hi>] |
| P(effect > 0) | <prob> |
| P(\|effect\| > <decision threshold>) | <prob> |

**Interpretation.** <2–4 sentences in domain language. Translate the effect to natural frequencies for non-technical readers (e.g., "for every 100 people exposed, we estimate 8 more would <outcome>"). State whether the result depends on a particular HDI width.>

## 6. Refutation

The strength of causal language must match the strength of refutation results. We report every test, including failures.

| Test | Outcome | Rating |
|------|---------|--------|
| <e.g., Placebo treatment time> | <result> | <PASS / MARGINAL / FAIL> |
| <e.g., Random common cause (DoWhy)> | <result> | <PASS / MARGINAL / FAIL> |
| <e.g., Data subset stability> | <result> | <PASS / MARGINAL / FAIL> |
| Unobserved confounding sensitivity | tipping point ≈ <value> | <PASS / MARGINAL / FAIL> |

**[IF DiD]**

![Parallel trends](parallel_trends.png)

Pre-treatment trends in treatment and control groups should be approximately parallel — same slope, possibly different levels. Visible divergence in the pre-period is direct evidence that parallel trends does not hold and the DiD estimate is biased.

**Assessment:** <1–2 sentences — are pre-trends parallel? Any visible divergence?>

**[IF SC]**

![Pre-treatment fit](pre_treatment_fit.png) ![Placebo density](placebo_density.png)

Synthetic control validity rests on close pre-treatment fit between treated unit and synthetic control. Placebo distributions show whether the post-treatment gap is unusually large compared to the same procedure applied to control units.

**Assessment:** <1–2 sentences on fit quality and placebo distribution.>

**[IF RDD]**

![Density test](density_test.png) ![Bandwidth sensitivity](bandwidth_sensitivity.png)

The McCrary density test checks for manipulation of the running variable around the threshold. Bandwidth sensitivity checks whether the estimate is stable across reasonable bandwidth choices.

**Assessment:** <1–2 sentences — bunching at threshold? Estimate stability across bandwidths?>

**[IF ITS]**

![Autocorrelation](autocorrelation.png)

ITS regressions on time-series data have correlated residuals. Spikes outside the confidence band on the ACF indicate residual autocorrelation that, if unmodeled, deflates standard errors.

**Assessment:** <1–2 sentences — Durbin-Watson, residual autocorrelation, confounding events flagged by user.>

### Sensitivity to Unobserved Confounding

![Sensitivity tipping point](sensitivity_tipping.png)

The tipping point is the unobserved confounder strength at which the estimated effect collapses to zero. Compare to the strongest *measured* confounder: a tipping point much larger than the strongest observed confounder is reassuring; a tipping point similar to or smaller than measured confounders means the result is fragile.

**Tipping point:** <value>
**Strongest observed confounder strength:** <value>
**Ratio:** <tipping / observed>×

**Assessment:** <one sentence — robust / marginal / fragile.>

## 7. Causal Language Calibration

Based on the refutation outcomes above, this analysis supports the following level of claim:

> **<causal / suggestive / associational / descriptive>**

<Use the harness output from `scripts/check_refutation.py`. Justify the chosen level in one sentence — which test result drove this calibration. If associational or descriptive, explicitly state that causal interpretation is not supported.>

## 8. Limitations and Threats

This section is **mandatory** and must be prominent — not buried in an appendix. Decision-makers must see it.

Threats ranked by severity:

1. **<biggest threat>** — <direction of bias if violated> <quantification (E-value, sensitivity tipping)> <what data or design would resolve it>
2. **<next threat>** — <same structure>

## 9. Plain-Language Conclusion

> "We estimate <treatment> causes <outcome> to change by <effect> (<HDI>), assuming <key assumptions>. There is a <P>% probability the effect is positive. The main threat to this conclusion is <biggest weakness>. If that assumption is violated, the true effect could be <direction and magnitude of bias>."

<If refutation downgraded the language to "associational" or "descriptive", rewrite the above sentence to match — do not use "causes" when you have not earned it.>

## Suggested Next Steps

<From `scripts/check_refutation.py` `suggest_next_steps()`. Tailor with problem-specific context.>

1. <step>
2. <step>

## Appendix

<Effect summary CSV, identification.json, refutation.json, code repository link, software versions.>
````

### Common "Suggested Next Steps" patterns

The harness in `scripts/check_refutation.py` emits these automatically. Override only with problem-specific context.

- All refutation passes, sensitivity tipping high → "Proceed with causal language. Communicate the effect with decision-relevant HDIs and the biggest threat in plain language."
- One critical test fails (placebo non-zero, McCrary bunching, poor SC pre-fit) → "Downgrade to associational language. Investigate the failure: is it fixable (e.g., add donors, restrict bandwidth) or fundamental (manipulation, parallel trends violated)?"
- Sensitivity tipping comparable to strongest observed confounder → "Result is fragile. Either collect more covariates, switch to a stronger design (find an instrument or discontinuity), or report explicitly as suggestive."
- Multiple marginal failures, no critical fails → "Use suggestive language. Run alternative designs as cross-checks; if they agree, claim convergent evidence rather than identification from any single design."
- Parallel trends violated in DiD → "Switch to synthetic control or trend-adjusted DiD. Quantify the bias the naive DiD introduced (often the most informative thing you can report)."
- Poor SC pre-treatment fit → "Expand donor pool, add predictors, or switch to BSTS. Do not interpret the post-treatment gap until pre-fit is acceptable."

### Refutation metric directions

`check_refutation.py` selects the correct pass-direction from each canonical test name, so you usually do **not** set `metric` in `refutation.json` — just name the test. The non-obvious directions it handles for you:

| Test(s) | What PASS means | Auto metric |
|---------|-----------------|-------------|
| `placebo_treatment_time`, `dowhy_placebo_treatment` | placebo effect brackets 0 | `effect` |
| `parallel_trends_pretest`, `mccrary_density`, `autocorrelation`, `covariate_balance` | **high** p-value — fail to reject the *good* null | `pvalue_high_good` |
| `sc_pre_treatment_fit` | low pre-fit RMSE ratio | `rmse_ratio` |
| `iv_first_stage_strength` | high first-stage F | `strength` |
| `random_common_cause`, `data_subset`, `leave_one_out_donors` | small shift in the estimate | `shift` |

**The trap this avoids:** for parallel-trends and McCrary, a *low* p-value is bad — it rejects parallel pre-trends or signals running-variable manipulation. Labeling those with a generic `pvalue` metric (which treats low as good, for "fraction of placebos as extreme as treated" style scores) silently flips a FAIL into a PASS and lets the report claim a causal effect it has not earned. When unsure, omit `metric` and let the test name drive it.

---

## Causal analysis report template

> The canonical artifact above is the source of truth. The inline structure below is kept for reference and for cases where a one-off prose report is more useful than a slug-folder artifact.

Every causal analysis produces a report with this mandatory structure. Adapt sections as needed, but do not drop sections 1, 7, or 8 — they are non-negotiable.

### 1. Causal question

One sentence: "What is the effect of [treatment] on [outcome] in [population]?"

Write this before touching data. If you cannot write this sentence, you do not yet have a causal question — you have a dataset.

### 2. DAG and assumptions

Include the causal graph (generated with `model.plot()` or drawn explicitly) and an assumption transparency table:

| Assumption | Testable? | How fragile? | What if violated? |
|-----------|-----------|-------------|-------------------|
| No unobserved confounders | No | Often fragile | Effect estimate biased in unknown direction |
| Parallel trends (DiD) | Partially (pre-treatment only) | Moderate | Effect estimate biased |
| No anticipation (DiD) | No | Robust if policy unexpected | Effect diluted pre-treatment |
| SUTVA / no spillovers | No | Fragile if units interact | Estimate includes spillovers |
| Exclusion restriction (IV) | No | Very fragile | IV estimate inconsistent |

Every assumption in the table must be discussed — not just listed. For each fragile assumption, state what evidence, if any, supports it.

### 3. Identification strategy

"We use [method] to identify the causal effect. This is valid because [justification]."

Be explicit: which identification result applies (backdoor, frontdoor, IV, RDD continuity, DiD parallel trends)? Which variables are adjusted for and why? Reference `dags-and-identification.md` for identification criteria and `quasi-experiments.md` for design-based methods.

If the effect is not point-identified, say so. Partial identification — bounding the effect rather than pinning it down — is a legitimate and honest result.

### 4. Estimation

State the model specification: likelihood, priors, and any structural constraints. Summarize diagnostics (R-hat, ESS, divergences). Defer to `bayesian-workflow/references/diagnostics.md` for diagnostic standards and thresholds. Do not re-explain those standards here; link to them.

For structural causal models, report the DoWhy estimand and estimation method:

```python
estimate = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.linear_regression",
)
```

### 5. Results with uncertainty

Report effect size with full posterior distribution and multiple HDIs. Never report only a point estimate.

Example:

> "We estimate the policy increased test scores by 4.2 points (50% HDI: [3.1, 5.3]; 95% HDI: [-0.3, 8.9]). The most likely effect is 3–5 points, but we cannot rule out a null effect at 95% credibility. There is an 87% posterior probability the effect is positive."

Include:
- A posterior density plot or forest plot of the causal effect
- The probability of direction: `P(effect > 0)` or `P(effect < threshold)`
- Effect size in domain-relevant units, not just standardized coefficients

### 6. Refutation results

Run all applicable refutation tests and report every result — including failures. Do not cherry-pick passing tests.

| Test | Result | Interpretation |
|------|--------|---------------|
| Placebo treatment (random treatment) | PASS | Random assignment gives near-zero effect |
| Placebo treatment time | PASS | No effect at a time when none should exist |
| Parallel trends (pre-treatment) | PASS | Pre-treatment trends are parallel |
| Random common cause | PASS | Adding a random confounder does not change estimate meaningfully |
| Data subset refutation | PASS | Effect is stable across random subsets |
| Sensitivity to unobserved confounding | E-value = 2.3 | A confounder with RR > 2.3 on both treatment and outcome would explain away the effect |

For DoWhy refutations:

```python
refute = model.refute_estimate(
    identified_estimand,
    estimate,
    method_name="placebo_treatment_refuter",
)
print(refute)
```

If a refutation test fails, downgrade the causal language accordingly — see [Causal language guardrails](#causal-language-guardrails).

### 7. Limitations and threats to validity

This section is mandatory and must be prominent — not buried in an appendix. Decision-makers must see it.

Rank threats by severity. For each:
- State the assumption that might be violated
- Explain the direction of bias if it is violated
- Quantify if possible (E-value, sensitivity analysis, bounding)
- State what additional data or design would resolve the threat

Example:

> "The main threat to this conclusion is unobserved confounding. An unobserved variable would need to be associated with both treatment assignment and outcomes with a relative risk greater than 2.3 to explain away the estimated effect (E-value = 2.3). Given the rich covariate set and the DiD design, we consider this unlikely but cannot rule it out without experimental data."

### 8. Plain-language conclusion

Close every report with one paragraph in plain language, regardless of audience:

> "We estimate [treatment] causes [outcome] to change by [effect] ([HDI]), assuming [key assumptions hold]. There is a [P]% probability the effect is positive. The main threat to this conclusion is [biggest weakness]. If that assumption is violated, the true effect could be [direction and magnitude of bias]."

---

## Causal language guardrails

The strength of your language must match the strength of your identification. Using causal language without identification is not just imprecise — it is misleading.

| Analysis state | Language | Example |
|---------------|----------|---------|
| ID + estimation + all refutations pass | Causal | "X causes Y to increase by Z" |
| ID + estimation pass, some refutations marginal | Suggestive | "Evidence suggests X causes Y to increase by Z, though [caveat]" |
| Critical refutation fails | Associational | "X is associated with a Z-unit increase in Y, but causal interpretation is limited because [reason]" |
| No identification strategy | Descriptive | "We observe X and Y are correlated. We cannot assign a causal interpretation without a credible identification strategy." |

Default to the more conservative language when in doubt. Overclaiming in a causal analysis is a more serious error than underclaiming — it can drive bad decisions.

Never use the word "effect" when the analysis state is Descriptive. "Association," "correlation," and "relationship" are correct.

---

## Decision-relevant HDIs

Do not default to a fixed HDI width. Choose widths that map to intuitive probabilities for the decision context.

| HDI width | Natural frequency | When to use |
|-----------|------------------|-------------|
| 50% | "roughly 1 in 2 chance" | Most likely range; good for communicating typical effect |
| 75% | "roughly 3 in 4 chance" | Good default for moderate-stakes decisions |
| 89% | "roughly 9 in 10 chance" | Moderate-to-high stakes |
| 95% | "roughly 19 in 20 chance" | High stakes; safety-critical decisions |

Report multiple widths when useful, especially when the conclusion changes across them:

> "The effect is 4.2 points (50% HDI: [3.1, 5.3]; 95% HDI: [-0.3, 8.9]). The most likely effect is 3–5 points, but we cannot rule out a null at 95% credibility."

Always state why you chose the reported HDI width. "We report the 75% HDI because this decision requires us to act if the effect is positive with 3-in-4 confidence" is better than silently presenting a number.

For causal analyses specifically, also report the probability of direction:

```python
# Probability effect is positive
p_positive = (idata.posterior["causal_effect"] > 0).mean().item()
print(f"P(effect > 0) = {p_positive:.2f}")
```

This is more interpretable than any fixed HDI when the question is directional ("does the policy help or hurt?").

---

## Audience adaptation

### Technical audience (researchers, analysts)

- Full DAG with node and edge justifications
- Formal identification result (backdoor/frontdoor/IV/RDD/DiD) with citation if applicable
- Posterior plots and full diagnostic summary
- Complete refutation table with test statistics
- Sensitivity analysis (E-values, partial R² bounds, or Rosenbaum bounds)
- Code or link to code repository in appendix

### Decision-makers (executives, policymakers)

- Causal question in plain language — one sentence
- Effect size translated to natural frequencies: "For every 100 people exposed, we estimate 8 more would [outcome]"
- Key threats in 1–2 sentences, stated simply: "The main reason this estimate could be wrong is [X]. If so, the true effect is likely [smaller/larger]."
- Actionable recommendation with explicit uncertainty: "Given the uncertainty, we recommend [action] if the cost of a false positive is less than [threshold]."
- Technical details (DAG, model specification, diagnostics, refutation table) in a clearly labeled appendix

**Both audiences get the limitations section.** Never hide limitations from decision-makers on the grounds that they are "too technical." Translate, do not omit.

### Presenting to mixed audiences

Open with the plain-language conclusion and effect size. Then walk through the DAG visually — most people understand arrows even without training. Reserve equations and diagnostic plots for Q&A or written appendices.

---

## Common reporting mistakes

1. **Using causal language without identification.** If there is no identification strategy, the word "effect" is wrong. Use "association."

2. **Reporting only the point estimate.** The posterior is the result. Always show the full distribution or at minimum multiple HDIs.

3. **Hiding refutation failures.** A failed refutation is information. Report it, downgrade your language, and explain what the failure means for the conclusion.

4. **Burying limitations.** Threats to validity belong in the body of the report, ranked by severity — not in an appendix labeled "caveats."

5. **Conflating LATE with ATE.** IV estimates give the Local Average Treatment Effect for compliers only. DiD estimates are often ATT (Average Treatment Effect on the Treated). Be explicit about whose effect you are estimating and whether it answers the question.

6. **Ignoring spillovers.** If SUTVA is violated — units affect each other — the estimated effect conflates direct and spillover effects. State whether spillovers are plausible and, if so, what direction they push the estimate.

7. **Omitting the E-value or sensitivity analysis.** For observational studies, always quantify how much unobserved confounding would be needed to overturn the conclusion. This anchors the limitations discussion in something concrete rather than vague hedging.
