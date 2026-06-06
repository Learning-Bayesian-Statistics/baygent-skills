"""
Interpret causal refutation outcomes and calibrate causal language.

Reads a structured refutation JSON (the analyst writes this from DoWhy /
CausalPy results) and produces:

- Per-test pass/marginal/fail ratings using house thresholds
- A causal-language calibration: causal / suggestive / associational / descriptive
- An ordered list of suggested next steps

This is the causal-inference analog of bayesian-workflow's check_diagnostics.py
and amortized-workflow's check_diagnostics.py. The script does not run
refutations — it interprets results that the analyst has already produced.

Input format (refutation.json):
    {
      "design": "DiD" | "SC" | "RDD" | "ITS" | "IV" | "IPSW" | "Structural",
      "tests": {
        "<test_name>": {
          "result": <number or null>,         # e.g., placebo effect mean, p-value, RMSE ratio
          "hdi": [<lo>, <hi>],                 # optional, when applicable
          "passes_zero": true | false,         # for HDIs; null if not applicable
          "metric": "<optional — see below>",  # omit to auto-select by test name
          "notes": "<optional analyst note>"
        }
      },

    The "metric" field is OPTIONAL. When omitted, the correct pass-direction is
    chosen from the canonical test name via DEFAULT_METRIC_BY_TEST — this is the
    safe default and avoids the direction-inversion footgun below. Supported:

      "effect"           PASS when the HDI brackets zero (placebo/falsification
                         effect should be null).
      "shift"            PASS when |relative shift| is small (estimate barely
                         moves under perturbation: random common cause, subset).
      "rmse_ratio"       PASS when low (SC pre-treatment fit RMSE / effect).
      "fraction"         PASS when low (e.g., fraction of placebos as extreme as
                         treated, fraction of bandwidths that flip sign).
      "pvalue"           PASS when LOW (< 0.10). For "badness" scores where a
                         small value is good — NOT for ordinary GOF p-values.
      "pvalue_high_good" PASS when HIGH (>= 0.10). For falsification tests where
                         failing to reject the GOOD null is what you want:
                         parallel-trends pre-test, McCrary density, residual
                         autocorrelation, covariate balance.
      "strength"         PASS when HIGH (>= 10). Instrument first-stage F.
                         ("auc" is accepted as a back-compat alias.)

    Direction trap: for parallel-trends and McCrary, a LOW p-value is BAD (it
    rejects parallel pre-trends / signals manipulation). Labeling them "pvalue"
    would invert the verdict. Prefer omitting "metric" so the test name drives it.
      "sensitivity": {
        "tipping_point": <number>,
        "strongest_observed_confounder": <number>,
        "notes": "<optional>"
      }
    }

Usage:
    python check_refutation.py --refutation refutation.json
    python check_refutation.py --refutation refutation.json --output check_report.json
"""

import argparse
import json
import sys


# House thresholds — internal reference levels. Use the qualitative labels
# (PASS / MARGINAL / FAIL) in reports, not these raw numbers.

# Critical tests: failure of any one is disqualifying for causal language.
CRITICAL_TESTS = {
    "placebo_treatment_time",
    "dowhy_placebo_treatment",
    "mccrary_density",
    "sc_pre_treatment_fit",
    "parallel_trends_pretest",
    "iv_first_stage_strength",
}

# Marginal tests: failure warrants caveats but not full disqualification.
MARGINAL_ONLY_TESTS = {
    "bandwidth_sensitivity",
    "data_subset",
    "leave_one_out_donors",
    "autocorrelation",
    "covariate_balance",
    "random_common_cause",
}

# Default metric per canonical test — picks the correct pass-direction so the
# analyst never has to. An explicit "metric" in the test dict overrides this.
# This is the primary guard against direction inversion (e.g., rating a
# significant parallel-trends pre-test as PASS).
DEFAULT_METRIC_BY_TEST = {
    "placebo_treatment_time": "effect",          # placebo effect should bracket 0
    "dowhy_placebo_treatment": "effect",
    "random_common_cause": "shift",              # estimate should barely move
    "data_subset": "shift",
    "leave_one_out_donors": "shift",
    "sc_pre_treatment_fit": "rmse_ratio",        # pre-RMSE / effect, lower better
    "bandwidth_sensitivity": "fraction",         # fraction of bw sign-flips, lower better
    "mccrary_density": "pvalue_high_good",       # high p = no manipulation
    "parallel_trends_pretest": "pvalue_high_good",  # high p = parallel pre-trends
    "autocorrelation": "pvalue_high_good",       # high p = no residual autocorrelation
    "covariate_balance": "pvalue_high_good",     # high p = balanced
    "iv_first_stage_strength": "strength",       # first-stage F, higher better
}

# Sensitivity ratio thresholds (tipping_point / strongest_observed_confounder)
SENSITIVITY_ROBUST = 3.0  # tipping point ≥ 3× strongest observed → robust
SENSITIVITY_MARGINAL = 1.5  # 1.5–3× → marginal


def _rate_effect_test(test: dict) -> str:
    """Tests whose 'PASS' means: effect is indistinguishable from zero
    (placebo treatment, placebo time, placebo threshold, random common cause)."""
    hdi = test.get("hdi")
    passes_zero = test.get("passes_zero")
    if passes_zero is True or (hdi and hdi[0] <= 0 <= hdi[1]):
        return "PASS"
    # HDI excludes zero — fail if even its nearest edge sits well away from zero
    # (a clearly non-null placebo/falsification effect), else marginal.
    result = test.get("result")
    if result is None:
        return "MARGINAL"
    if hdi:
        nearest_edge = min(abs(hdi[0]), abs(hdi[1]))
        if nearest_edge > 0.3 * abs(result):
            return "FAIL"
        return "MARGINAL"
    return "MARGINAL"


def _rate_pvalue_test(test: dict) -> str:
    """Tests reporting a p-value-analogue (e.g., SC placebo fraction).
    Lower is better. PASS if < 0.10, MARGINAL if < 0.20, FAIL otherwise."""
    p = test.get("result")
    if p is None:
        return "MARGINAL"
    if p < 0.10:
        return "PASS"
    if p < 0.20:
        return "MARGINAL"
    return "FAIL"


def _rate_rmse_ratio_test(test: dict) -> str:
    """SC pre-treatment fit: ratio of pre-RMSE to post-treatment effect.
    PASS if < 0.10, MARGINAL if < 0.30, FAIL otherwise."""
    r = test.get("result")
    if r is None:
        return "MARGINAL"
    if r < 0.10:
        return "PASS"
    if r < 0.30:
        return "MARGINAL"
    return "FAIL"


def _rate_shift_test(test: dict) -> str:
    """Tests reporting a shift in the estimate (random common cause, data subset).
    The reported number is typically the relative shift |new - original| / |original|.
    PASS if < 0.10, MARGINAL if < 0.25, FAIL otherwise."""
    s = test.get("result")
    if s is None:
        return "MARGINAL"
    if abs(s) < 0.10:
        return "PASS"
    if abs(s) < 0.25:
        return "MARGINAL"
    return "FAIL"


def _rate_fraction_test(test: dict) -> str:
    """Generic fraction-style metrics where lower is better.
    e.g., fraction of placebos as extreme as treated (SC), or fraction of
    bandwidth values where estimate flips sign (RDD)."""
    return _rate_pvalue_test(test)


def _rate_pvalue_high_good(test: dict) -> str:
    """Falsification / goodness-of-fit p-values where FAILING to reject the
    (good) null is the desired outcome: parallel-trends pre-test, McCrary
    density, residual autocorrelation, covariate balance.

    HIGH p = PASS. This is the opposite direction from _rate_pvalue_test, and
    using the wrong one is exactly how a violated parallel-trends assumption
    gets silently rated PASS."""
    p = test.get("result")
    if p is None:
        return "MARGINAL"
    if p >= 0.10:
        return "PASS"
    if p >= 0.05:
        return "MARGINAL"
    return "FAIL"


def _rate_strength_test(test: dict) -> str:
    """Instrument strength / first-stage F-statistic (higher is better).
    F > 10 is the conventional 'not weak' threshold."""
    v = test.get("result")
    if v is None:
        return "MARGINAL"
    if v >= 10:
        return "PASS"
    if v >= 5:
        return "MARGINAL"
    return "FAIL"


def _effective_metric(name: str, test: dict) -> str:
    """Explicit 'metric' wins; otherwise auto-select by canonical test name."""
    return test.get("metric") or DEFAULT_METRIC_BY_TEST.get(name, "effect")


def _rate_test(name: str, test: dict) -> str:
    """Dispatch on the effective metric (explicit, or auto-selected by name)."""
    metric = _effective_metric(name, test)
    if metric == "effect":
        return _rate_effect_test(test)
    if metric == "pvalue":
        return _rate_pvalue_test(test)
    if metric == "pvalue_high_good":
        return _rate_pvalue_high_good(test)
    if metric == "rmse_ratio":
        return _rate_rmse_ratio_test(test)
    if metric == "shift":
        return _rate_shift_test(test)
    if metric == "fraction":
        return _rate_fraction_test(test)
    if metric in ("strength", "auc"):  # "auc" kept as a back-compat alias
        return _rate_strength_test(test)
    return "MARGINAL"


def _rate_sensitivity(sensitivity: dict | None) -> tuple[str, float | None]:
    """Rate the unobserved confounding sensitivity tipping point.

    Returns (rating, ratio). Ratio is tipping / strongest_observed.
    """
    if not sensitivity:
        return "MARGINAL", None

    tip = sensitivity.get("tipping_point")
    obs = sensitivity.get("strongest_observed_confounder")

    if tip is None:
        return "MARGINAL", None
    if obs is None or obs <= 0:
        # No reference point — only the tipping point itself
        if tip > 0.5:
            return "PASS", None
        if tip > 0.2:
            return "MARGINAL", None
        return "FAIL", None

    ratio = tip / obs
    if ratio >= SENSITIVITY_ROBUST:
        return "PASS", ratio
    if ratio >= SENSITIVITY_MARGINAL:
        return "MARGINAL", ratio
    return "FAIL", ratio


def check_refutation(refutation: dict) -> dict:
    """Interpret refutation outcomes into per-test ratings + summary.

    Parameters
    ----------
    refutation : dict
        Schema as documented in the module docstring.

    Returns
    -------
    dict
        Structured assessment with per-test ratings, sensitivity rating,
        and a summary suitable for the report's pass/fail table.
    """
    report: dict = {
        "design": refutation.get("design", "unknown"),
        "tests": {},
        "sensitivity": {},
    }

    for name, test in refutation.get("tests", {}).items():
        rating = _rate_test(name, test)
        is_critical = name in CRITICAL_TESTS
        report["tests"][name] = {
            "rating": rating,
            "critical": is_critical,
            # Record the metric actually used (auto-selected when omitted) so the
            # report is transparent about how each verdict was reached.
            "metric": _effective_metric(name, test),
            "result": test.get("result"),
            "notes": test.get("notes", ""),
        }

    sens_rating, sens_ratio = _rate_sensitivity(refutation.get("sensitivity"))
    report["sensitivity"] = {
        "rating": sens_rating,
        "ratio": sens_ratio,
        "tipping_point": (refutation.get("sensitivity") or {}).get("tipping_point"),
        "strongest_observed": (refutation.get("sensitivity") or {}).get(
            "strongest_observed_confounder"
        ),
    }

    return report


def calibrate_causal_language(report: dict) -> dict:
    """Map the refutation report to one of:
    causal / suggestive / associational / descriptive.

    Returns a dict with the level, the driving test, and a justification.
    """
    tests = report.get("tests", {})
    sens = report.get("sensitivity", {})

    # Identify failures by tier
    critical_fails = [n for n, t in tests.items() if t["critical"] and t["rating"] == "FAIL"]
    critical_marginal = [
        n for n, t in tests.items() if t["critical"] and t["rating"] == "MARGINAL"
    ]
    non_critical_fails = [
        n for n, t in tests.items() if not t["critical"] and t["rating"] == "FAIL"
    ]
    non_critical_marginal = [
        n for n, t in tests.items() if not t["critical"] and t["rating"] == "MARGINAL"
    ]

    sens_rating = sens.get("rating", "MARGINAL")

    # ── Critical failure → associational ──────────────────────────────
    if critical_fails:
        return {
            "level": "associational",
            "driving_test": critical_fails[0],
            "justification": (
                f"A critical refutation test failed ({critical_fails[0]}). "
                "Causal interpretation is not supported; report as associational."
            ),
        }

    # ── Sensitivity FAIL → associational ──────────────────────────────
    if sens_rating == "FAIL":
        return {
            "level": "associational",
            "driving_test": "sensitivity_to_unobserved_confounding",
            "justification": (
                "Tipping point for unobserved confounding is comparable to or "
                "smaller than the strongest observed confounder. Effect is "
                "fragile; causal interpretation is not supported."
            ),
        }

    # ── Critical marginal OR multiple non-critical fails → suggestive ─
    if critical_marginal or len(non_critical_fails) >= 2:
        driver = (critical_marginal or non_critical_fails)[0]
        return {
            "level": "suggestive",
            "driving_test": driver,
            "justification": (
                f"Some refutation tests are marginal or failing ({driver} and "
                f"others). Use hedged language: 'evidence is suggestive of a "
                "causal effect, but cannot be considered definitive.'"
            ),
        }

    # ── Sensitivity MARGINAL → suggestive ─────────────────────────────
    if sens_rating == "MARGINAL":
        ratio = sens.get("ratio")
        ratio_text = f" (tipping point ≈ {ratio:.1f}× strongest observed)" if ratio else ""
        return {
            "level": "suggestive",
            "driving_test": "sensitivity_to_unobserved_confounding",
            "justification": (
                f"Sensitivity to unobserved confounding is marginal{ratio_text}. "
                "Use hedged causal language."
            ),
        }

    # ── Many non-critical marginal → suggestive ───────────────────────
    if len(non_critical_marginal) >= 2:
        return {
            "level": "suggestive",
            "driving_test": non_critical_marginal[0],
            "justification": (
                f"Multiple non-critical tests are marginal ({', '.join(non_critical_marginal[:2])}). "
                "Use hedged causal language."
            ),
        }

    # ── Otherwise: causal ─────────────────────────────────────────────
    return {
        "level": "causal",
        "driving_test": None,
        "justification": (
            "All critical refutation tests pass and sensitivity to unobserved "
            "confounding is robust. Causal language is supported."
        ),
    }


def suggest_next_steps(report: dict, language: dict) -> list[str]:
    """Return an ordered, actionable list of next steps based on the
    refutation outcomes and the calibrated causal language."""
    steps: list[str] = []
    tests = report.get("tests", {})
    sens = report.get("sensitivity", {})
    design = report.get("design", "unknown")

    # ── Critical failures (highest priority) ──────────────────────────
    for name, t in tests.items():
        if not t["critical"] or t["rating"] != "FAIL":
            continue

        if name == "placebo_treatment_time":
            steps.append(
                "Placebo treatment time produces a non-zero effect — parallel "
                "trends or pre-treatment dynamics are likely violated. Inspect "
                "the pre-period plot, consider unit-specific time trends, or "
                "switch to synthetic control for heterogeneous pre-trends."
            )
        elif name == "mccrary_density":
            steps.append(
                "McCrary density test shows bunching at the threshold — units "
                "are manipulating their position. The RDD design is invalid for "
                "this threshold; do not interpret the discontinuity causally. "
                "Consider a fuzzy RDD restricted to a wide bandwidth or a "
                "different identification strategy."
            )
        elif name == "sc_pre_treatment_fit":
            steps.append(
                "Pre-treatment fit of synthetic control is poor — the synthetic "
                "is not a credible counterfactual. Expand the donor pool, add "
                "predictors to the formula, or switch to a Bayesian structural "
                "time series model."
            )
        elif name == "parallel_trends_pretest":
            steps.append(
                "Pre-treatment trends are not parallel — the DiD parallel trends "
                "assumption is not supported by the data. Switch to synthetic "
                "control, trend-adjusted DiD, or explicitly bound the bias."
            )
        elif name == "dowhy_placebo_treatment":
            steps.append(
                "DoWhy placebo treatment refuter does not drive effect to zero — "
                "the model is fitting noise. Re-examine the DAG and the "
                "adjustment set; suspect residual confounding."
            )
        elif name == "iv_first_stage_strength":
            steps.append(
                "IV first-stage F-statistic is too weak — the instrument is "
                "weak and IV estimates are biased and high-variance. Find a "
                "stronger instrument or switch designs."
            )
        else:
            steps.append(
                f"Critical test {name} failed — investigate and either fix the "
                "design or downgrade the language to associational."
            )

    # ── Sensitivity ───────────────────────────────────────────────────
    if sens.get("rating") == "FAIL":
        ratio = sens.get("ratio")
        ratio_text = f" (tipping point ≈ {ratio:.1f}× strongest observed)" if ratio else ""
        steps.append(
            "Sensitivity to unobserved confounding is fragile" + ratio_text
            + " — collect more covariates, switch to a quasi-experimental design "
            "(find an instrument or discontinuity), or report explicitly as "
            "associational."
        )
    elif sens.get("rating") == "MARGINAL":
        steps.append(
            "Sensitivity to unobserved confounding is marginal — quantify the "
            "minimum confounder strength needed to overturn the result and "
            "compare to plausible unmeasured variables (see refutation.md). "
            "Use hedged language in the report."
        )

    # ── Critical marginal ─────────────────────────────────────────────
    for name, t in tests.items():
        if not t["critical"] or t["rating"] != "MARGINAL":
            continue
        steps.append(
            f"Critical test {name} is marginal — strengthen the analysis "
            "(more data, refined design, alternative estimator) or downgrade "
            "language to 'suggestive'."
        )

    # ── Non-critical failures ─────────────────────────────────────────
    nc_fails = [n for n, t in tests.items() if not t["critical"] and t["rating"] == "FAIL"]
    if nc_fails:
        steps.append(
            f"Non-critical failures: {', '.join(nc_fails)}. Each warrants a "
            "short paragraph in the limitations section explaining the failure "
            "and what it implies for the conclusion."
        )

    # ── Design-specific reminders when no fails ───────────────────────
    if not steps:
        steps.append(
            f"All refutation tests pass for the {design} design. Communicate "
            "the effect with decision-relevant HDIs (50% / 89% / 95%) and the "
            "biggest remaining threat in plain language. Run an alternative "
            "design as a cross-check if feasible."
        )

    # ── Always: language calibration reminder ─────────────────────────
    level = language.get("level")
    if level in ("suggestive", "associational", "descriptive"):
        steps.append(
            f"Calibrated causal language is '{level}' — rewrite the conclusion "
            "and abstract to match. Replace 'causes' with the appropriate "
            "softer phrasing (see causal language guardrails in reporting.md)."
        )

    return steps


def main():
    parser = argparse.ArgumentParser(
        description="Interpret causal refutation outcomes and calibrate causal language"
    )
    parser.add_argument(
        "--refutation",
        required=True,
        help="Path to refutation.json (schema in module docstring)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save full JSON report (default: print to stdout)",
    )
    args = parser.parse_args()

    try:
        with open(args.refutation) as f:
            refutation = json.load(f)
    except FileNotFoundError:
        print(json.dumps({"error": f"File not found: {args.refutation}"}))
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"Could not parse {args.refutation}: {e}"}))
        sys.exit(1)

    report = check_refutation(refutation)
    language = calibrate_causal_language(report)
    next_steps = suggest_next_steps(report, language)

    print("=== Refutation Ratings ===")
    for name, t in report["tests"].items():
        crit = "[CRITICAL]" if t["critical"] else ""
        print(f"  {name}: {t['rating']} {crit}")
    sens = report["sensitivity"]
    if sens.get("rating"):
        ratio = sens.get("ratio")
        ratio_text = f" (ratio={ratio:.2f})" if ratio else ""
        print(f"  sensitivity: {sens['rating']}{ratio_text}")
    print("==========================\n")

    print("=== Calibrated Causal Language ===")
    print(f"  Level: {language['level'].upper()}")
    print(f"  Driver: {language['driving_test']}")
    print(f"  Why: {language['justification']}")
    print("=================================\n")

    print("=== Suggested Next Steps ===")
    for i, s in enumerate(next_steps, 1):
        print(f"  {i}. {s}")
    print("============================\n")

    full = {
        "ratings": report,
        "language": language,
        "next_steps": next_steps,
    }
    output = json.dumps(full, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()
