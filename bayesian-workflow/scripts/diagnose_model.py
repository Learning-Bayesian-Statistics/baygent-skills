"""
Automated Bayesian model diagnostics.

Runs convergence checks, posterior predictive checks, and produces
a structured report.

Usage:
    python diagnose_model.py --idata path/to/inference_data.nc
    python diagnose_model.py --idata path/to/inference_data.nc --output report.json
"""

import argparse
import json
import sys
import warnings

import arviz as az
import numpy as np
import pymc as pm

warnings.filterwarnings("ignore", category=FutureWarning)

# arviz_stats >= 1.0.0 provides diagnose() for one-call diagnostics
try:
    import arviz_stats as azs

    HAS_DIAGNOSE = hasattr(azs, "diagnose")
except ImportError:
    HAS_DIAGNOSE = False


def _json_default(o):
    """Coerce stray numpy scalars/arrays so json.dumps never chokes.

    Belt-and-suspenders: check_convergence already returns primitives, but LOO
    and any future additions may carry numpy types. np.int64 in particular is
    not a Python int and is rejected by the default JSON encoder.
    """
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _max_from_dataset(ds):
    """Best-effort scalar maximum from an xarray Dataset of diagnostic values.

    ``arviz_stats.diagnose`` returns the per-parameter R-hat values as an xarray
    Dataset (not JSON-serializable). We only need the scalar max for rating, so
    extract it defensively and drop the Dataset itself.
    """
    if ds is None:
        return None
    try:
        # Reduce each variable independently, then take the overall max. This is
        # robust to (a) modern xarray dropping Dataset.to_array() in favour of
        # to_dataarray(), and (b) variables with mismatched dims (scalar params
        # alongside a vector Deterministic) that break both to_array/to_dataarray.
        # arviz_stats.diagnose only stores the *flagged* params here, so an empty
        # Dataset (everything converged) correctly yields None — no max to report.
        data_vars = getattr(ds, "data_vars", None)
        if data_vars is not None:
            vals = [float(ds[v].max()) for v in data_vars]
        else:
            vals = [float(ds.max())]  # already a DataArray
        # Drop non-finite maxima — a NaN here would serialize as a bare `NaN`,
        # which json.dumps emits but is invalid JSON (rejected by strict parsers).
        finite = [v for v in vals if np.isfinite(v)]
        return max(finite) if finite else None
    except Exception:
        return None


def check_convergence(idata):
    """Run all convergence diagnostics. Returns structured results.

    Both code paths emit the SAME serializable schema so ``check_diagnostics.py``
    has a single contract to consume: ``rhat`` / ``ess_bulk`` / ``ess_tail`` /
    ``divergences`` each carry ``ok`` and ``problematic_params``; the
    ``arviz_stats.diagnose`` path additionally reports ``bfmi`` and ``treedepth``.
    No numpy scalars, numpy arrays, or xarray objects leak into the dict — those
    are not JSON-serializable and would crash the report writer.
    """
    # Use arviz_stats.diagnose() if available — it covers R-hat, ESS,
    # divergences, tree depth saturation, and E-BFMI in one call.
    if HAS_DIAGNOSE:
        # show_diagnostics=False keeps stdout clean so the JSON report can be
        # piped there when --output is omitted.
        has_errors, d = azs.diagnose(
            idata, return_diagnostics=True, show_diagnostics=False
        )
        rhat_bad = [str(p) for p in d.get("rhat", {}).get("bad_params", [])]
        ess_bad = [str(p) for p in d.get("ess", {}).get("bad_params", [])]
        div = d.get("divergent", {})
        n_div = int(div.get("n_divergent", 0))
        td = d.get("treedepth", {})
        n_td = int(td.get("n_max", 0))
        failed_chains = [int(c) for c in d.get("bfmi", {}).get("failed_chains", [])]
        return {
            "all_ok": not has_errors,
            "method": "arviz_stats.diagnose",
            "rhat": {
                "ok": len(rhat_bad) == 0,
                "max": _max_from_dataset(d.get("rhat", {}).get("rhat_values")),
                "problematic_params": rhat_bad,
            },
            # diagnose() reports a single ESS check; mirror it into bulk/tail so
            # the downstream schema matches the manual fallback path exactly.
            "ess_bulk": {"ok": len(ess_bad) == 0, "problematic_params": ess_bad},
            "ess_tail": {"ok": len(ess_bad) == 0, "problematic_params": ess_bad},
            "divergences": {
                "count": n_div,
                "pct": round(float(div.get("pct", 0.0)), 2),
                "ok": n_div == 0,
            },
            "treedepth": {
                "ok": n_td == 0,
                "n_max": n_td,
                "pct": round(float(td.get("pct", 0.0)), 2),
            },
            "bfmi": {"ok": len(failed_chains) == 0, "failed_chains": failed_chains},
        }

    # Fallback for arviz-stats < 1.0.0
    summary = az.summary(idata)
    num_chains = idata.chains.size

    results = {
        "rhat": {
            "max": float(summary["r_hat"].max()),
            "ok": bool((summary["r_hat"] <= 1.01).all()),
            "problematic_params": list(summary[summary["r_hat"] > 1.01].index),
        },
        "ess_bulk": {
            "min": int(summary["ess_bulk"].min()),
            "ok": bool((summary["ess_bulk"] >= 100 * num_chains).all()),
            "problematic_params": list(
                summary[summary["ess_bulk"] < 100 * num_chains].index
            ),
        },
        "ess_tail": {
            "min": int(summary["ess_tail"].min()),
            "ok": bool((summary["ess_tail"] >= 100 * num_chains).all()),
            "problematic_params": list(
                summary[summary["ess_tail"] < 100 * num_chains].index
            ),
        },
    }

    # Divergences
    if "diverging" in idata.sample_stats:
        n_div = int(idata.sample_stats["diverging"].sum())
        total_samples = int(idata.sample_stats["diverging"].size)
        results["divergences"] = {
            "count": n_div,
            "pct": round(100 * n_div / total_samples, 2),
            "ok": n_div == 0,
        }
    else:
        results["divergences"] = {"count": 0, "pct": 0.0, "ok": True}

    results["all_ok"] = all(
        results[k]["ok"] for k in ["rhat", "ess_bulk", "ess_tail", "divergences"]
    )
    results["method"] = "manual"

    return results


def _group_names(idata):
    """Group names as a set, normalized across InferenceData and DataTree.

    PyMC 5 / arviz 0.23 return an ``InferenceData`` whose ``groups`` is a
    *method* yielding bare names (``"log_likelihood"``). PyMC 6 / arviz 1.x
    return a ``DataTree`` whose ``groups`` is a *property* yielding *paths*
    (``"/log_likelihood"``). Calling ``idata.groups()`` on the latter raises
    ``'tuple' object is not callable`` — which, swallowed by check_loo's except,
    silently drops LOO on PyMC 6 even when log_likelihood is present. Normalize
    both shapes to a set of bare names.
    """
    groups = getattr(idata, "groups", None)
    if groups is None:
        return set()
    raw = groups() if callable(groups) else groups
    return {s for g in raw if (s := str(g).strip("/"))}


def _loo_field(loo, *names):
    """First present attribute among ``names`` on an ELPDData, else None.

    arviz 1.x renamed the classic ELPDData fields: ``elpd_loo`` -> ``elpd`` and
    ``p_loo`` -> ``p`` (``se`` and ``pareto_k`` are unchanged). Trying each name
    in turn makes the extractor work on arviz 0.23 and 1.x alike.
    """
    for n in names:
        if hasattr(loo, n):
            return getattr(loo, n)
    return None


def check_loo(idata, model=None):
    """Run LOO-CV and check Pareto k diagnostics."""
    try:
        if "log_likelihood" not in _group_names(idata):
            if model is not None:
                pm.compute_log_likelihood(idata, model=model)
            else:
                return {"computed": False, "error": "No log_likelihood group and no model provided. Pass --model or call pm.compute_log_likelihood(idata, model=model) before saving."}
        loo = az.loo(idata, pointwise=True)
        pareto_k = np.asarray(_loo_field(loo, "pareto_k"), dtype=float)
        elpd = _loo_field(loo, "elpd_loo", "elpd")
        se = _loo_field(loo, "se")
        p_loo = _loo_field(loo, "p_loo", "p")

        # A non-finite Pareto-k means the importance-sampling LOO for that point
        # could not be estimated — arviz 1.x returns NaN there, where arviz 0.23
        # sometimes smoothed it to a finite value. Keep it DISTINCT from a finite
        # k > 0.7 (different diagnosis, different fix), and take the max over the
        # finite values only, so a bare NaN never leaks into the JSON report
        # (json.dumps emits `NaN`, which is invalid JSON).
        finite = pareto_k[np.isfinite(pareto_k)]
        n_nonfinite = int(pareto_k.size - finite.size)

        # Pareto-k cutoff is held at the classic 0.7 (not arviz 1.x's
        # data-dependent good_k) so the same idata yields the same rating on both
        # the PyMC 5 and PyMC 6 stacks — cross-version equivalence over novelty.
        # n_bad counts ONLY finite k > 0.7, so the "k > 0.7" wording downstream is
        # always literally true; non-finite points are reported via n_nonfinite.
        n_high = int(np.sum(finite > 0.7))
        results = {
            "elpd": float(elpd) if elpd is not None else None,
            "se": float(se) if se is not None else None,
            "p_loo": float(p_loo) if p_loo is not None else None,
            "pareto_k": {
                "max": float(finite.max()) if finite.size else None,
                "n_bad": n_high,
                "n_marginal": int(np.sum((finite > 0.5) & (finite <= 0.7))),
                "n_nonfinite": n_nonfinite,
                "ok": n_high == 0 and n_nonfinite == 0,
            },
            "computed": True,
        }
    except Exception as e:
        results = {"computed": False, "error": str(e)}

    return results


def check_posterior_predictive(idata):
    """Check if posterior predictive data exists and basic stats."""
    if not hasattr(idata, "posterior_predictive"):
        return {
            "available": False,
            "message": "No posterior predictive samples found. Run pm.sample_posterior_predictive().",
        }

    pp_vars = list(idata.posterior_predictive.data_vars)
    results = {"available": True, "variables": pp_vars}

    if hasattr(idata, "observed_data"):
        obs_vars = list(idata.observed_data.data_vars)
        results["observed_variables"] = obs_vars

    return results


def generate_report(idata, model=None):
    """Generate complete diagnostics report."""
    report = {
        "convergence": check_convergence(idata),
        "loo": check_loo(idata, model=model),
        "posterior_predictive": check_posterior_predictive(idata),
    }

    # Overall assessment
    issues = []
    if not report["convergence"]["all_ok"]:
        issues.append("Convergence issues detected — results may be unreliable")
    loo_pk = report["loo"].get("pareto_k", {}) if report["loo"].get("computed") else {}
    if loo_pk and not loo_pk.get("ok", True):
        n_high = loo_pk.get("n_bad", 0)
        n_nf = loo_pk.get("n_nonfinite", 0)
        parts = []
        if n_high:
            parts.append(f"{n_high} observation(s) with Pareto k > 0.7")
        if n_nf:
            parts.append(
                f"{n_nf} observation(s) with non-finite Pareto k (LOO could not be estimated)"
            )
        issues.append("; ".join(parts) if parts else "influential observations in LOO")
    if not report["posterior_predictive"]["available"]:
        issues.append("No posterior predictive checks available")

    report["overall"] = {
        "ok": len(issues) == 0,
        "issues": issues,
        "recommendation": "Model is ready for interpretation."
        if len(issues) == 0
        else "Address the following issues before interpreting results: "
        + "; ".join(issues),
    }

    return report


def main():
    parser = argparse.ArgumentParser(description="Bayesian model diagnostics")
    parser.add_argument(
        "--idata", required=True, help="Path to InferenceData (.nc file)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save JSON report (default: print to stdout)",
    )
    args = parser.parse_args()

    try:
        idata = az.from_netcdf(args.idata)
    except Exception as e:
        print(json.dumps({"error": f"Could not load InferenceData: {e}"}))
        sys.exit(1)

    report = generate_report(idata)

    output = json.dumps(report, indent=2, default=_json_default)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Report saved to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
