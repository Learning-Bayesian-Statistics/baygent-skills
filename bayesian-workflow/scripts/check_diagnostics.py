"""
Interpret Bayesian model diagnostics and suggest next steps.

Reads structured outputs from ``diagnose_model.py`` and ``calibration_check.py``
and produces qualitative per-section assessments for the diagnostic report,
plus an ordered list of suggested next steps.

This script does not re-run diagnostics — it interprets what is already there.
Run ``diagnose_model.py`` and ``calibration_check.py`` first to produce the JSON
inputs, then this script to turn the numbers into human-readable ratings and
actionable recommendations.

Usage:
    python check_diagnostics.py --diagnostics diagnostics.json
    python check_diagnostics.py --diagnostics diagnostics.json --calibration calibration.json
    python check_diagnostics.py --diagnostics diagnostics.json --psense psense.json --output report.json

The ``diagnostics.json`` is the output of ``diagnose_model.py``.
The ``calibration.json`` is the output of ``calibration_check.py``.
The ``psense.json`` (optional) is a JSON dump of ``psense_summary(idata)``.
"""

import argparse
import json
import sys


# House thresholds — internal reference levels for qualitative interpretation.
# These are NOT exposed in reports — use the qualitative labels instead.
RHAT_OK = 1.01
DIVERGENCE_OK = 0
DIVERGENCE_FAIR = 0.005  # < 0.5% of post-warmup draws
PARETO_K_OK = 0.5
PARETO_K_FAIR = 0.7
COVERAGE_DEVIATION_OK = 0.02
COVERAGE_DEVIATION_FAIR = 0.05
PSENSE_OK = 0.05
PSENSE_FAIR = 0.10


def _rate_convergence(conv: dict) -> tuple[str, list[str]]:
    """Return (rating, problematic_param_names) for the convergence section.

    Consumes the unified schema emitted by ``diagnose_model.check_convergence``
    — both the ``arviz_stats.diagnose`` and manual fallback paths share it, so
    real parameter names flow through (no opaque placeholder).
    """
    if not conv or conv.get("all_ok"):
        return "excellent", []

    issues: list[str] = []

    rhat = conv.get("rhat", {})
    ess_b = conv.get("ess_bulk", {})
    ess_t = conv.get("ess_tail", {})
    div = conv.get("divergences", {})
    bfmi = conv.get("bfmi", {})
    td = conv.get("treedepth", {})

    if not rhat.get("ok", True):
        issues.extend(rhat.get("problematic_params", []))
    if not ess_b.get("ok", True):
        issues.extend(ess_b.get("problematic_params", []))
    if not ess_t.get("ok", True):
        issues.extend(ess_t.get("problematic_params", []))
    if not div.get("ok", True):
        issues.append(f"divergences={div.get('count', 0)}")
    if not bfmi.get("ok", True):
        issues.append("low E-BFMI")
    if not td.get("ok", True):
        issues.append("max treedepth saturation")

    issues = list(dict.fromkeys(issues))  # dedupe, preserve order

    n_div = div.get("count", 0) if div else 0
    div_pct = div.get("pct", 0.0) if div else 0.0
    rhat_max = rhat.get("max") if rhat else None

    if n_div > 0 and div_pct > DIVERGENCE_FAIR * 100:
        rating = "poor"
    elif rhat_max is not None and rhat_max > 1.05:
        rating = "poor"
    elif issues:
        rating = "fair"
    else:
        rating = "good"

    return rating, issues


def _rate_loo(loo: dict) -> str:
    """Return a qualitative rating for the LOO Pareto-k diagnostic."""
    if not loo or not loo.get("computed"):
        return "not computed"

    pk = loo.get("pareto_k", {})
    n_bad = pk.get("n_bad", 0)
    n_marginal = pk.get("n_marginal", 0)
    pk_max = pk.get("max", 0.0)

    if pk_max <= PARETO_K_OK:
        return "excellent"
    elif pk_max <= PARETO_K_FAIR and n_bad == 0:
        return "fair"
    elif n_bad > 0:
        return "poor"
    else:
        return "fair"


def _rate_calibration(cal: dict) -> tuple[str, str]:
    """Return (rating, diagnosis) for calibration."""
    if not cal:
        return "not computed", ""

    assessment = cal.get("assessment", {})
    diagnosis = assessment.get("calibration_diagnosis", "")
    well_cal = assessment.get("well_calibrated", False)
    mean_dev = abs(assessment.get("mean_coverage_deviation", 0.0))

    if well_cal:
        return "excellent", diagnosis
    if mean_dev <= COVERAGE_DEVIATION_FAIR:
        return "fair", diagnosis
    return "poor", diagnosis


def _rate_psense(ps: dict) -> tuple[str, list[str]]:
    """Return (rating, flagged_params) for prior sensitivity."""
    if not ps:
        return "not computed", []

    # Expected: dict of {param: {"prior": <cjs>, "likelihood": <cjs>}}
    flagged: list[str] = []
    max_prior_cjs = 0.0

    for param, vals in ps.items():
        if isinstance(vals, dict):
            prior = float(vals.get("prior", 0.0))
            max_prior_cjs = max(max_prior_cjs, prior)
            if prior > PSENSE_FAIR:
                flagged.append(param)

    if max_prior_cjs <= PSENSE_OK:
        return "low sensitivity", flagged
    if max_prior_cjs <= PSENSE_FAIR:
        return "moderate sensitivity", flagged
    return "strong sensitivity", flagged


def check_diagnostics(
    diagnostics: dict | None = None,
    calibration: dict | None = None,
    psense: dict | None = None,
) -> dict:
    """Interpret diagnostic JSON outputs into qualitative assessments.

    Parameters
    ----------
    diagnostics : dict, optional
        Output of ``diagnose_model.generate_report()``. Provides
        ``convergence``, ``loo``, and ``posterior_predictive`` sub-reports.
    calibration : dict, optional
        Output of ``calibration_check.py``.
    psense : dict, optional
        Output of ``psense_summary(idata).to_dict()`` or equivalent.

    Returns
    -------
    dict
        Structured assessment with sections: ``convergence``, ``loo``,
        ``calibration``, ``psense``, plus ``summary`` (one-line per section).
    """
    report: dict = {}

    if diagnostics is not None:
        conv_rating, conv_issues = _rate_convergence(diagnostics.get("convergence", {}))
        report["convergence"] = {
            "rating": conv_rating,
            "problematic_params": conv_issues,
        }
        report["loo"] = {"rating": _rate_loo(diagnostics.get("loo", {}))}
        ppc = diagnostics.get("posterior_predictive", {})
        report["posterior_predictive"] = {
            "available": bool(ppc.get("available", False)),
        }

    if calibration is not None:
        cal_rating, cal_diagnosis = _rate_calibration(calibration)
        report["calibration"] = {
            "rating": cal_rating,
            "diagnosis": cal_diagnosis,
        }

    if psense is not None:
        ps_rating, ps_flagged = _rate_psense(psense)
        report["psense"] = {
            "rating": ps_rating,
            "flagged_params": ps_flagged,
        }

    report["summary"] = _build_summary(report)
    return report


def _build_summary(report: dict) -> dict:
    """One short sentence per section, suitable for the report's Assessment lines."""
    s: dict = {}

    if "convergence" in report:
        r = report["convergence"]
        if r["rating"] == "excellent":
            s["convergence"] = "All convergence diagnostics passed (R-hat ≤ 1.01, ESS adequate, no divergences)."
        elif r["rating"] == "good":
            s["convergence"] = "Convergence diagnostics broadly pass; minor flags worth noting."
        elif r["rating"] == "fair":
            params = ", ".join(r["problematic_params"][:3]) or "some parameters"
            s["convergence"] = f"Convergence is fair — flags on {params}. Inspect before trusting the posterior."
        else:
            params = ", ".join(r["problematic_params"][:3]) or "multiple parameters"
            s["convergence"] = f"Poor convergence on {params}. Posterior should not be interpreted until resolved."

    if "loo" in report:
        s["loo"] = f"LOO Pareto-k: {report['loo']['rating']}."

    if "calibration" in report:
        r = report["calibration"]
        diag = r.get("diagnosis", "")
        s["calibration"] = (
            f"Calibration is {r['rating']}" + (f" — {diag}." if diag else ".")
        )

    if "psense" in report:
        r = report["psense"]
        flagged = ", ".join(r["flagged_params"]) if r["flagged_params"] else "none"
        s["psense"] = f"Prior sensitivity: {r['rating']} (flagged: {flagged})."

    return s


def suggest_next_steps(report: dict) -> list[str]:
    """Combine the interpreted report into an ordered, actionable list.

    Most-critical issues first. Returns generic guidance — extend with
    problem-specific context when filling in the report.
    """
    steps: list[str] = []

    # ── Convergence (highest priority) ────────────────────────────────
    conv = report.get("convergence", {})
    if conv.get("rating") in ("poor", "fair"):
        params = conv.get("problematic_params", [])
        has_divergences = any("divergence" in str(p).lower() for p in params)
        # Non-parameter flags (divergences, E-BFMI, treedepth) are surfaced
        # separately — keep them out of the per-parameter R-hat/ESS message so
        # it never interpolates a flag phrase where a parameter name belongs.
        _non_param = ("divergence", "e-bfmi", "treedepth", "=")
        named_params = [
            p for p in params if not any(tok in str(p).lower() for tok in _non_param)
        ]

        if has_divergences:
            steps.append(
                "Divergences detected — reparameterize the affected component "
                "(non-centered for hierarchical scales; replace HalfCauchy with "
                "Gamma(2, ...) for scale priors) and increase target_accept to 0.95–0.99."
            )
        if named_params:
            preview = ", ".join(named_params[:3])
            steps.append(
                f"R-hat or ESS flags on {preview} — run more draws, try "
                "init=\"adapt_diag_grad\", or warm-start with `pmx.fit(method=\"pathfinder\")`. "
                "Check for multimodality with az.plot_trace(idata, kind=\"rank_vlines\")."
            )

    # ── Calibration ───────────────────────────────────────────────────
    cal = report.get("calibration", {})
    if cal.get("rating") == "poor":
        diag = cal.get("diagnosis", "")
        if "over-confident" in diag:
            steps.append(
                "Calibration is over-confident — likelihood is too narrow for the "
                "data. Consider StudentT for continuous outcomes with heavy tails, "
                "NegBinomial for overdispersed counts, or hierarchical structure if "
                "groups have distinct variance."
            )
        elif "under-confident" in diag:
            steps.append(
                "Calibration is under-confident — predictions are too uncertain. "
                "Tighten priors that are dominating the likelihood, or check whether "
                "the model is overcomplicated for the data."
            )
        else:
            steps.append(
                "Calibration check failed — re-examine the likelihood and the "
                "prior predictive range before interpreting posteriors."
            )
    elif cal.get("rating") == "fair":
        steps.append(
            "Calibration is fair but not excellent — consider tightening priors, "
            "switching to a heavier-tailed likelihood, or running a sensitivity "
            "check on the most informative observations."
        )

    # ── LOO ───────────────────────────────────────────────────────────
    loo_rating = report.get("loo", {}).get("rating", "not computed")
    if loo_rating == "poor":
        steps.append(
            "LOO Pareto-k > 0.7 for some observations — these points are highly "
            "influential. Inspect them (often outliers or high-leverage), consider "
            "a more robust likelihood (StudentT), or refit excluding them to test "
            "sensitivity. Use az.loo(idata, pointwise=True) and az.plot_khat(loo)."
        )
    elif loo_rating == "fair":
        steps.append(
            "LOO Pareto-k between 0.5 and 0.7 for some observations — borderline. "
            "Identify the points and decide whether they're substantively important; "
            "consider a robust likelihood if they reflect a tail-heavy data process."
        )

    # ── PPC availability ──────────────────────────────────────────────
    if not report.get("posterior_predictive", {}).get("available", True):
        steps.append(
            "No posterior_predictive group found — run "
            "`pm.sample_posterior_predictive(idata)` and re-run diagnostics. "
            "Models without PPC cannot be model-criticized."
        )

    # ── Prior sensitivity ─────────────────────────────────────────────
    ps = report.get("psense", {})
    if ps.get("rating") == "strong sensitivity":
        flagged = ", ".join(ps.get("flagged_params", []))
        steps.append(
            f"Strong prior sensitivity on {flagged} — either justify the "
            "informative prior explicitly with domain knowledge, or widen it "
            "and refit. Report both versions if the substantive conclusion changes."
        )
    elif ps.get("rating") == "moderate sensitivity":
        flagged = ", ".join(ps.get("flagged_params", []))
        steps.append(
            f"Moderate prior sensitivity on {flagged} — note in the report. "
            "Conclusions for those parameters depend partly on the prior."
        )

    # ── Default if nothing is wrong ───────────────────────────────────
    if not steps:
        steps.append(
            "All diagnostics are within acceptable bounds — proceed to "
            "interpretation. Translate posteriors into decision-relevant terms "
            "and document the model assumptions in the report."
        )
        if report.get("psense", {}).get("rating") == "not computed":
            steps.append(
                "Consider running prior sensitivity (`psense_summary(idata)`) "
                "before publication, especially if any conclusions are policy-relevant."
            )

    return steps


def _load_optional(path: str | None) -> dict | None:
    if path is None:
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: {path} not found — skipping.", file=sys.stderr)
        return None
    except json.JSONDecodeError as e:
        print(f"Warning: could not parse {path} ({e}) — skipping.", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Interpret Bayesian diagnostics and suggest next steps"
    )
    parser.add_argument(
        "--diagnostics",
        required=True,
        help="Path to diagnostics JSON (output of diagnose_model.py)",
    )
    parser.add_argument(
        "--calibration",
        default=None,
        help="Path to calibration JSON (output of calibration_check.py)",
    )
    parser.add_argument(
        "--psense",
        default=None,
        help="Path to prior sensitivity JSON (psense_summary output)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save full JSON report (default: print to stdout)",
    )
    args = parser.parse_args()

    diagnostics = _load_optional(args.diagnostics)
    if diagnostics is None:
        print(
            json.dumps(
                {"error": f"Could not load required diagnostics file: {args.diagnostics}"}
            )
        )
        sys.exit(1)

    calibration = _load_optional(args.calibration)
    psense = _load_optional(args.psense)

    report = check_diagnostics(
        diagnostics=diagnostics,
        calibration=calibration,
        psense=psense,
    )
    next_steps = suggest_next_steps(report)
    report["next_steps"] = next_steps

    print("=== Per-Section Assessment ===")
    for section, line in report.get("summary", {}).items():
        print(f"  {section}: {line}")
    print("==============================\n")

    print("=== Suggested Next Steps ===")
    for i, step in enumerate(next_steps, 1):
        print(f"  {i}. {step}")
    print("============================\n")

    output = json.dumps(report, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Report saved to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
