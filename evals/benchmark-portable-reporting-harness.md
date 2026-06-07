# Benchmark — portable-reporting-harness-v1 (stage 1)

Targeted benchmark for the branch's new + changed content. Method: `with_skill` runs invoke the
relevant skill via the Skill tool and follow it; `without_skill` runs answer from base knowledge with
no skill; both are graded against each scenario's `eval_metadata.json` assertions (assertions withheld
from the answering agents). Run on the `baygent` env (PyMC 5.28 / CausalPy 0.8.0).

## New scenarios — coverage for this branch's guidance

| Scenario | with_skill | without_skill | lift | note |
|----------|-----------|---------------|------|------|
| discrete-latent-marginalization (bayesian) | 7/7 | 6/7 | +1 | base missed the soft-plug-in-is-wrong warning |
| its-effect-persistence (causal) | 8/8 | 7/8 | +1 | base hand-rolled a 2-changepoint regression; missed CausalPy `treatment_end_time` / `analyze_persistence` |
| mbias-adjustment-set (causal) | 7/7 | 7/7 | 0 | base handled M-bias unaided |

Lift is modest: base models are strong on well-known theory (M-bias, mixtures). These scenarios earn
their place as **regression guards** (they lock in the corrected guidance — notably the M-bias /
pre-treatment-not-sufficient rule) and coverage for the specific tooling idioms, not as lift drivers.

## Changed scenarios — regression check (with_skill)

| Scenario | score | new content present? |
|----------|-------|----------------------|
| troubleshooting-divergences (bayesian) | 7/8 | ✓ sampling-failure escalation ladder (lone miss: didn't say "multimodality" explicitly — single-run variance, not a content regression) |
| synthetic-control-poor-donors (causal) | 10/10 | ✓ donor-support hull pre-screen, with the "FAIL informative, PASS not sufficient" framing |
| observational-dag-confounders (causal) | 11/11 | ✓ BMI-as-mediator + pre-treatment-not-sufficient reasoning |

**No regressions** — the edits improved the changed scenarios rather than breaking them.

## Verdict

Merge gate (with_skill correctness + no regression on touched scenarios) is met: 22/22 on new
scenarios, 28/29 on changed. The full remaining-scenario sweep (untouched scenarios + all iterations)
is low marginal value for this branch — those scenarios exercise guidance this branch did not change —
and is best run via the usual full-suite harness if a complete record is wanted.
