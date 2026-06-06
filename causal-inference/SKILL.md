---
name: causal-inference
description: >
  Production-grade Bayesian causal inference with PyMC, CausalPy, and DoWhy. Enforces DAG-first
  thinking, mandatory user checkpoints for assumptions, design-specific refutation, and defensible
  reporting with causal language guardrails. Trigger on: causal inference, causal effect estimation,
  treatment effects, counterfactuals, difference-in-differences (DiD), synthetic control, regression
  discontinuity (RDD), interrupted time series (ITS), instrumental variables (IV), propensity scores,
  DAGs, causal graphs, confounders, backdoor criterion, do-calculus, interventional distributions,
  pm.do(), pm.observe(), CausalPy, DoWhy, mediation analysis, refutation, sensitivity analysis,
  parallel trends, placebo tests, or any question of the form "does X cause Y" or "what is the
  effect of X on Y."
license: MIT
metadata:
  author: "[Alexandre Andorra](https://alexandorra.github.io/)"
  version: "1.2"
---

# Causal Inference

## Dependencies

This skill requires the **bayesian-workflow** skill for all PyMC modeling steps (priors, sampling,
diagnostics, calibration, reporting).

Detect it:

```bash
ls ~/.claude/skills/bayesian-workflow/SKILL.md 2>/dev/null || ls .claude/skills/bayesian-workflow/SKILL.md 2>/dev/null
```

If not found, install it:

```bash
git clone https://github.com/Learning-Bayesian-Statistics/baygent-skills.git /tmp/baygent-skills
cp -r /tmp/baygent-skills/bayesian-workflow ~/.claude/skills/
```

For all PyMC modeling steps (priors, sampling, diagnostics, calibration, reporting), follow the
bayesian-workflow skill.

## Workflow overview

Every causal analysis follows this sequence. Steps 1-4 are the thinking phase (no code). Steps 5-8
are the doing phase. Think before you do.

1. **Formulate the causal question** — Propose precise estimand (ATE, ATT, LATE, etc.). ⚠️ ASK USER TO CONFIRM.
2. **Draw the DAG** — Propose causal graph with nodes, edges, and explicit non-edges. ⚠️ ASK USER TO CONFIRM. See [references/dags-and-identification.md](references/dags-and-identification.md)
3. **Identify** — Determine identification strategy (backdoor, front-door, IV, RDD, DiD). ⚠️ ASK USER TO CONFIRM untestable assumptions. See [references/dags-and-identification.md](references/dags-and-identification.md)
4. **Choose design** — Match problem to method using table below. ⚠️ ASK USER TO CONFIRM. See [references/quasi-experiments.md](references/quasi-experiments.md) or [references/structural-models.md](references/structural-models.md)
5. **Estimate** — Build and fit the model. Delegate all PyMC mechanics to bayesian-workflow skill.
6. **Refute** — MANDATORY. Run design-specific robustness checks. See [references/refutation.md](references/refutation.md)
7. **Interpret** — Effect size + decision-relevant HDIs + probability of direction.
8. **Report** — Generate `<treatment>-on-<outcome>/report.md` using the canonical template in [references/reporting.md](references/reporting.md). Run `scripts/check_refutation.py` to turn refutation outcomes into pass/marginal/fail ratings, calibrated causal language (causal / suggestive / associational / descriptive), and an ordered next-steps list. Use that output to fill the report's section 7 (causal language calibration) and Suggested Next Steps.

## Design selection guide

Before reading the table, walk the **intake** — it routes you to a design *and* surfaces the assumptions you'll have to defend later:

1. **Restate the estimand.** ATE, ATT, LATE, or a CATE? On whom, over what period? If you can't write the one-sentence estimand, you don't have a causal question yet — you have a dataset.
2. **Identify the data shape.** Single time series, wide panel of units, long unit–time panel, cross-section, or pre/post group data?
3. **Identify the treatment assignment.** Known intervention time, staggered adoption, a threshold/cutoff, a kink, an instrument, or observed treatment with confounders?
4. **State the identifying story.** What makes the comparison causal — parallel trends, no anticipation, no manipulation at the cutoff, a valid instrument, overlap/positivity, donor support? This is exactly what you'll refute later.
5. **Recommend one primary design + alternatives — and reconcile with the estimand.** Name the best-fit design from the table plus any plausible cross-check. Then check that the estimand from step 1 is the one the design *actually identifies*: DiD → ATT (on the treated); RDD → a *local* effect at the cutoff; IV → LATE for compliers (under monotonicity); SC → the treated-unit effect. If they differ, change the design or re-scope the estimand and say so. If two designs agree later, that's convergent evidence rather than reliance on a single identification.

| Design | Use when | Key assumption | Tool |
|---|---|---|---|
| DiD | Treatment at known time, control group available | Parallel trends | CausalPy |
| Staggered DiD | Treatment rolls out at different times | Parallel trends per cohort | CausalPy |
| Synthetic Control | Single treated unit, donor pool available | Weighted donors approximate counterfactual | CausalPy |
| ITS | Time series, intervention at known time, no control | No confounding event at treatment time | CausalPy |
| RDD | Treatment by threshold on running variable | No manipulation at threshold | CausalPy |
| IV | Endogenous treatment, valid instrument | Exclusion restriction, relevance | CausalPy |
| IPSW | Observational data, treatment modeled | No unmeasured confounders, positivity | CausalPy |
| Structural (do/observe) | Full causal theory, model mechanisms | Correct DAG specification | PyMC |
| Counterfactual | "What would Y have been if X differed?" | Correct structural model | PyMC |

## Critical rules

- **No estimation without a confirmed DAG.** A causal graph is not optional decoration — it makes
  assumptions explicit and determines the adjustment set. If the user resists, explain why the DAG
  is non-negotiable before proceeding.
- **The DAG licenses adjustment, not timing — pre-treatment is necessary, not sufficient.** Pre-treatment
  timing only rules out two *post*-treatment hazards: collider bias (conditioning on a common effect of
  treatment and outcome) and over-control (conditioning on a mediator). It does **not** by itself make a
  variable safe to adjust for. A *pre-treatment* variable can still be a collider on a back-door path —
  **M-bias**: a common effect of an unobserved cause of treatment and an unobserved cause of the outcome —
  and adjusting for it *induces* confounding that wasn't there. So include a covariate only if (a) it is
  pre-treatment **and** (b) the DAG shows it blocks a back-door path and is not itself a collider (or a
  descendant of one) on an otherwise-blocked path — i.e. it satisfies the back-door/adjustment criterion.
  Timing is the quick screen; the graph is the authority. (Deliberate **mediation analysis** is the
  principled exception to the post-treatment ban: it conditions on a post-treatment mediator on purpose to
  estimate a *different* estimand — direct vs. indirect effects — under stronger assumptions, including
  **no treatment-induced mediator–outcome confounding**; declare it as mediation, don't slip it in as
  confounder control.) If the covariates that actually satisfy the back-door criterion are too thin to
  identify the effect, say so and frame the result as **descriptive, not causal**.
- **No causal claims without refutation.** Every design has failure modes. Run at minimum one
  design-specific robustness check (placebo test, sensitivity analysis, falsification test) before
  reporting results. See [references/refutation.md](references/refutation.md).
- **State assumptions before results.** Lead with what must be true for the estimate to be causal.
  Bury the estimate after the assumptions, not before. This is not optional politeness — it prevents
  misuse of results.
- **Adapt HDIs to the decision context.** The bayesian-workflow skill's 94% HDI is a sensible
  default; adapt it with explicit explanation when the decision stakes warrant it (e.g., 89% for
  exploratory, 97% for high-stakes policy). Report multiple intervals when the decision threshold
  matters.
- **Downgrade causal language when warranted.** If identification assumptions are unverifiable or
  refutation raises flags, soften claims: "consistent with a causal effect" not "causes", "estimated
  effect" not "true effect". Flag uncertainty loudly in the report.
- **Use `scripts/check_refutation.py` to calibrate language, not human judgment.** The script takes
  a structured `refutation.json` (see its docstring for schema) and returns a calibrated level —
  causal / suggestive / associational / descriptive — plus an ordered list of next steps. Use the
  script's output verbatim in the report; expand only with problem-specific context. Hand-rolled
  language calibration is exactly where overclaiming creeps in.
- **Always generate `<treatment>-on-<outcome>/report.md` after the analysis.** Store all artifacts
  (`inference_data.nc`, `dag.png`, `effect_posterior.png`, `forest.png`, design-specific figures,
  `refutation.json`, `effect_summary.csv`) in the slug-named results folder, and produce `report.md`
  from the canonical template in [references/reporting.md](references/reporting.md). The report is
  the audit trail; code without an interpreted, fixed-shape report is incomplete.
- **Ask the user when domain knowledge is needed.** You cannot know whether an instrument is valid,
  whether parallel trends holds, or whether a confounder exists without domain expertise. Ask
  before assuming.
- **Delegate PyMC mechanics to bayesian-workflow.** This skill handles causal structure and design.
  The bayesian-workflow skill handles priors, sampling, diagnostics, calibration, and reporting
  format. Don't duplicate those rules here.

## Common gotchas

These are battle-tested lessons that save hours of debugging:

- **CausalPy formula syntax uses `C()` for categoricals.** Passing a string column directly without
  `C()` will silently produce wrong dummy coding. Always wrap categorical treatment and group
  variables: `"y ~ C(treatment) + C(group)"`.
- **DoWhy requires explicit `U` nodes for unobserved confounders.** Omitting them from the graph
  will make DoWhy treat your model as fully identified when it isn't. Add latent nodes explicitly
  and mark them as unobserved.
- **CausalPy's PyMC models don't auto-store log-likelihood.** Same issue as bayesian-workflow:
  nutpie silently drops it. Call `pm.compute_log_likelihood(idata, model=model)` after sampling if
  you need it for model comparison.
- **Parallel trends is untestable in the post-treatment period.** Pre-treatment trend tests are
  necessary but not sufficient — passing them doesn't prove the assumption holds after treatment.
  State this explicitly in every DiD report.
- **Synthetic control requires the treated unit to lie within the convex hull of donors.** If the
  treated unit is an outlier (highest GDP, largest city), no weighted combination of donors can
  approximate its counterfactual. Check this before running — if violated, the design is invalid.
- **DiD group variable must be dummy-coded (0/1).** CausalPy rejects string labels like "treatment"/"control". Use integers: 1 = treatment, 0 = control. Data also requires a `unit` column.
- **SyntheticControl expects wide-format data.** Index = time, columns = unit names, values = outcome. If your data is long format, pivot first: `df.pivot(index="date", columns="unit", values="outcome")`.

## Utility scripts

```bash
# Interpret refutation outcomes and calibrate causal language
python scripts/check_refutation.py --refutation <slug>/refutation.json --output <slug>/check_report.json
```

The `refutation.json` is written by the analyst from DoWhy/CausalPy refutation outputs. See the
[script docstring](scripts/check_refutation.py) for the JSON schema, the test categories
(critical vs. marginal), and the language calibration rules.

## When things go wrong

| Symptom | Likely cause | Fix |
|---|---|---|
| Refutation fails | Assumption violated | Diagnose which assumption, try alternative design or sensitivity bounds |
| DiD effect at placebo time | Parallel trends violated | Try synthetic control or add group-specific time trends |
| RDD: bunching at threshold | Manipulation of running variable | Design is invalid for this threshold — report and stop |
| SC: poor pre-treatment fit | Donors don't span treated unit | Add donors, expand donor pool, or reconsider design |
| DoWhy says "not identifiable" | Insufficient adjustment set | Revise DAG, add measured variables, or change design |
| CausalPy formula error | Wrong formula syntax | Use `C()` for categoricals, check variable names match dataframe columns |
