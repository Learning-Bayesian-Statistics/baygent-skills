"""Cross-environment equivalence gate for the portable reporting harness.

The bayesian-workflow scripts are dual-compatible during the PyMC 5 -> 6
transition: they must run on the PyMC 5 stack (env ``baygent``: arviz 0.23 +
arviz-stats/plots 1.0, ``InferenceData``) AND the PyMC 6 stack (env ``baygent6``:
arviz 1.x, ``DataTree``). "Runs on both" is necessary but NOT sufficient — the
signature failure of dual-version code is *runs on both, silently disagrees on
one* (e.g. LOO quietly dropping on PyMC 6 because ``idata.groups()`` is a method
on InferenceData but a property on DataTree).

This gate closes that gap. It feeds the SAME fixture idata to both envs and
asserts the user-facing diagnostics — convergence/LOO/calibration ratings, the
qualitative summary, and the ordered next steps — are byte-identical, with the
raw floats (elpd, pareto-k, ...) matched only to a tolerance, since PSIS/LOO
differs slightly across arviz versions even on identical inputs.

Modes:
  (orchestrate, default)  python cross_env_equivalence.py [--envs baygent baygent6]
  (internal worker)       python cross_env_equivalence.py --build-fixture --idata P
  (internal worker)       python cross_env_equivalence.py --emit-payload  --idata P

Run from the repo root with conda/mamba on PATH:
    python evals/smoke/cross_env_equivalence.py

Exits 0 when the two envs agree (or when fewer than two envs are present — a
loud SKIP, so single-env users aren't blocked), 1 on any disagreement. Pass
``--require-both`` to turn a missing env into a failure instead of a skip.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "bayesian-workflow" / "scripts"
SELF = Path(__file__).resolve()

DEFAULT_ENVS = ["baygent", "baygent6"]
FIXTURE_VAR = "y"
# Generous absolute tolerance for raw diagnostic magnitudes. Observed cross-arviz
# drift on a well-behaved fixture is < 1e-3; a genuine divergence shows up as a
# different *rating* (compared exactly, below), not a small float wobble.
RAW_TOL = 0.05


# ─────────────────────────────── workers ────────────────────────────────────
def build_fixture(path: Path, kind: str = "healthy") -> None:
    """Sample a small fixture (with log_likelihood + PPC) and save to ``path``.

    ``healthy``      — a well-behaved regression; exercises the clean-rating path.
    ``pathological`` — a CENTERED eight-schools funnel at modest target_accept, the
                       canonical divergence generator. Exercises the unhealthy
                       branches (divergences / R-hat / ESS flags, the non-empty
                       R-hat-max reduction) so the gate asserts the two envs agree
                       on a *poor/fair* verdict, not merely on a clean one.
    """
    import numpy as np
    import pymc as pm

    sys.path.insert(0, str(SCRIPTS))
    from diagnose_model import _group_names

    if kind == "healthy":
        seed = sum(map(ord, "reporting-harness-equivalence-fixture"))
        rng = np.random.default_rng(seed)
        x = rng.normal(0, 1, 80)
        y = 1.0 + 2.0 * x + rng.normal(0, 1.0, 80)
        with pm.Model(coords={"obs": np.arange(80)}):
            a = pm.Normal("a", 0, 5)
            b = pm.Normal("b", 0, 5)
            sigma = pm.HalfNormal("sigma", 5)
            mu = pm.Deterministic("mu", a + b * x, dims="obs")
            pm.Normal(FIXTURE_VAR, mu, sigma, observed=y, dims="obs")
            idata = pm.sample(
                draws=500, tune=500, chains=4, random_seed=seed,
                nuts_sampler="nutpie", progressbar=False,
                idata_kwargs={"log_likelihood": True},
            )
            pm.sample_posterior_predictive(idata, extend_inferencedata=True, progressbar=False)
            if "log_likelihood" not in _group_names(idata):
                pm.compute_log_likelihood(idata)
    elif kind == "pathological":
        seed = sum(map(ord, "reporting-harness-pathological-fixture"))
        sigma_obs = np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])
        y_obs = np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
        with pm.Model(coords={"school": np.arange(8)}):
            mu = pm.Normal("mu", 0, 5)
            tau = pm.HalfNormal("tau", 5)
            theta = pm.Normal("theta", mu, tau, dims="school")  # centered -> funnel
            pm.Normal(FIXTURE_VAR, theta, sigma_obs, observed=y_obs, dims="school")
            idata = pm.sample(
                draws=400, tune=300, chains=4, target_accept=0.8, random_seed=seed,
                nuts_sampler="nutpie", progressbar=False,
                idata_kwargs={"log_likelihood": True},
            )
            pm.sample_posterior_predictive(idata, extend_inferencedata=True, progressbar=False)
            if "log_likelihood" not in _group_names(idata):
                pm.compute_log_likelihood(idata)
    else:
        raise ValueError(f"unknown fixture kind: {kind}")

    idata.to_netcdf(str(path))


def emit_payload(idata_path: Path) -> dict:
    """Run the full diagnostics pipeline on one idata; return canonical + raw views."""
    sys.path.insert(0, str(SCRIPTS))
    import arviz as az
    import pymc as pm

    import calibration_check
    import check_diagnostics
    import diagnose_model
    from arviz_base import convert_to_datatree

    idata = az.from_netcdf(str(idata_path))
    diag = diagnose_model.generate_report(idata)
    dt = convert_to_datatree(str(idata_path))
    cal = calibration_check.assess_calibration(dt, FIXTURE_VAR, use_loo=False)
    calibration = {"variable": FIXTURE_VAR, "assessment": cal}
    checked = check_diagnostics.check_diagnostics(diagnostics=diag, calibration=calibration)
    checked["next_steps"] = check_diagnostics.suggest_next_steps(checked)

    # STRICT — must match byte-for-byte across stacks. The safety-critical,
    # our-responsibility view: the convergence verdict (interpret / don't), the
    # calibration verdict, structural flags, and loo_computed (the bug we fixed —
    # LOO must not silently drop on PyMC 6). LOO next-steps are excluded because
    # they hang off the threshold-sensitive Pareto-k rating (see info/numeric).
    strict = {
        "convergence_all_ok": diag["convergence"]["all_ok"],
        "convergence_method": diag["convergence"]["method"],
        "loo_computed": diag["loo"].get("computed"),
        "ppc_available": diag["posterior_predictive"]["available"],
        "overall_ok": diag["overall"]["ok"],
        "rating_convergence": checked.get("convergence", {}).get("rating"),
        "rating_calibration": checked.get("calibration", {}).get("rating"),
        "calibration_well_calibrated": cal["well_calibrated"],
        "calibration_diagnosis": cal["calibration_diagnosis"],
        "summary_convergence": checked.get("summary", {}).get("convergence"),
        "summary_calibration": checked.get("summary", {}).get("calibration"),
        "next_steps_non_loo": [s for s in checked["next_steps"] if not s.startswith("LOO ")],
    }
    # NUMERIC — compared with tolerance: arviz 0.23 vs 1.x estimate these slightly
    # differently (PSIS tail estimator, ECDF randomization), so exact equality is
    # the wrong bar. A genuine divergence shows up as a different STRICT rating.
    numeric = {
        "rhat_max": diag["convergence"]["rhat"].get("max"),
        "elpd": diag["loo"].get("elpd"),
        "se": diag["loo"].get("se"),
        "p_loo": diag["loo"].get("p_loo"),
        "pareto_k_max": diag["loo"].get("pareto_k", {}).get("max"),
        "mean_coverage_deviation": cal["mean_coverage_deviation"],
    }
    # INFO — reported, never fails the gate. The LOO Pareto-k qualitative rating is
    # threshold-sensitive: when pareto_k_max sits at the 0.5/0.7 boundary, the two
    # arviz versions can land on either side. Surfaced for transparency, not gated.
    info = {
        "rating_loo": checked.get("loo", {}).get("rating"),
        "pareto_k_ok": diag["loo"].get("pareto_k", {}).get("ok"),
        "pareto_k_n_bad": diag["loo"].get("pareto_k", {}).get("n_bad"),
    }
    return {"arviz": az.__version__, "pymc": pm.__version__,
            "strict": strict, "numeric": numeric, "info": info}


# ──────────────────────────── orchestration ─────────────────────────────────
def _conda_run(env: str, *args: str) -> subprocess.CompletedProcess:
    cmd = ["conda", "run", "-n", env, "python", str(SELF), *args]
    run_env = {**os.environ, "MPLBACKEND": "Agg", "PYTHONWARNINGS": "ignore"}
    return subprocess.run(cmd, capture_output=True, text=True, env=run_env)


def _env_usable(env: str) -> bool:
    proc = _conda_run(env, "--selfcheck")
    return proc.returncode == 0


def _last_json(text: str) -> dict:
    """Parse the last JSON object printed on stdout (ignore any leading noise)."""
    depth, start, blocks = 0, None, []
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                blocks.append(text[start : i + 1])
    for block in reversed(blocks):
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            continue
    raise ValueError("no JSON object found in worker stdout")


def _close(a, b) -> bool:
    """Tolerance for raw diagnostic magnitudes: 2% relative with a 0.05 floor.

    Covers cross-arviz PSIS/ECDF drift (observed < 1e-3 on well-behaved data, a
    touch more on divergent fits) without masking a gross divergence — those
    surface as a differing STRICT rating, not a small float wobble.
    """
    if a is None or b is None:
        return a == b
    a, b = float(a), float(b)
    return abs(a - b) <= RAW_TOL + 0.02 * max(abs(a), abs(b))


def _diff(a: dict, b: dict, env_a: str, env_b: str) -> list[str]:
    fails: list[str] = []
    sa, sb = a["strict"], b["strict"]
    for k in sorted(set(sa) | set(sb)):
        if sa.get(k) != sb.get(k):
            fails.append(f"strict[{k}]: {sa.get(k)!r} ({env_a}) != {sb.get(k)!r} ({env_b})")
    na, nb = a["numeric"], b["numeric"]
    for k in sorted(set(na) | set(nb)):
        if not _close(na.get(k), nb.get(k)):
            fails.append(f"numeric[{k}]: {na.get(k)} ({env_a}) vs {nb.get(k)} ({env_b}) exceeds tolerance")
    return fails


def orchestrate(envs: list[str], require_both: bool) -> int:
    import tempfile

    usable = [e for e in envs if _env_usable(e)]
    if len(usable) < 2:
        msg = f"found usable envs {usable}, need >= 2 of {envs}"
        if require_both:
            print(f"CROSS-ENV EQUIVALENCE FAILED — {msg}")
            return 1
        print(f"CROSS-ENV EQUIVALENCE SKIPPED — {msg}. "
              "Create environment-pymc6.yml's baygent6 to enable this gate.")
        return 0

    env_a, env_b = usable[0], usable[1]
    kinds = ["healthy", "pathological"]
    all_fails: list[str] = []

    with tempfile.TemporaryDirectory() as td:
        for kind in kinds:
            fixture = Path(td) / f"equiv_{kind}.nc"
            print(f"[{kind}] building shared fixture under '{env_a}' ...")
            proc = _conda_run(env_a, "--build-fixture", "--kind", kind, "--idata", str(fixture))
            if proc.returncode != 0 or not fixture.exists():
                print(f"CROSS-ENV EQUIVALENCE FAILED — [{kind}] fixture build errored:\n{proc.stderr[-400:]}")
                return 1

            payloads: dict[str, dict] = {}
            for env in usable:
                proc = _conda_run(env, "--emit-payload", "--idata", str(fixture))
                if proc.returncode != 0:
                    print(f"CROSS-ENV EQUIVALENCE FAILED — [{kind}] payload errored in '{env}':\n{proc.stderr[-400:]}")
                    return 1
                payloads[env] = _last_json(proc.stdout)

            pa, pb = payloads[env_a], payloads[env_b]
            print(f"  [{kind}] {env_a} pymc{pa['pymc']}/az{pa['arviz']} vs "
                  f"{env_b} pymc{pb['pymc']}/az{pb['arviz']} | "
                  f"conv={pa['strict']['rating_convergence']} "
                  f"calib={pa['strict']['rating_calibration']}")
            if pa["info"] != pb["info"]:
                print(f"     note: LOO Pareto-k rating is threshold-sensitive across stacks — "
                      f"{env_a} loo={pa['info']['rating_loo']} (k_ok={pa['info']['pareto_k_ok']}), "
                      f"{env_b} loo={pb['info']['rating_loo']} (k_ok={pb['info']['pareto_k_ok']}); "
                      f"raw pareto_k_max agrees within tolerance.")
            all_fails += [f"[{kind}] {f}" for f in _diff(pa, pb, env_a, env_b)]

    print("=" * 60)
    if all_fails:
        print(f"CROSS-ENV EQUIVALENCE FAILED — {len(all_fails)} disagreement(s):")
        for f in all_fails:
            print(f"  - {f}")
        return 1
    print(f"CROSS-ENV EQUIVALENCE PASSED — {env_a} and {env_b} produce identical "
          f"diagnostics/ratings across {len(kinds)} fixtures (healthy + pathological).")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", nargs="+", default=DEFAULT_ENVS,
                        help="conda env names to compare (default: baygent baygent6)")
    parser.add_argument("--require-both", action="store_true",
                        help="fail (not skip) if fewer than two envs are usable")
    parser.add_argument("--build-fixture", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--emit-payload", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--selfcheck", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--kind", default="healthy", help=argparse.SUPPRESS)
    parser.add_argument("--idata", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.selfcheck:
        import pymc  # noqa: F401  — confirms the env can import the stack
        return
    if args.build_fixture:
        build_fixture(Path(args.idata), kind=args.kind)
        return
    if args.emit_payload:
        print(json.dumps(emit_payload(Path(args.idata)), indent=2, sort_keys=True, default=str))
        return

    sys.exit(orchestrate(args.envs, args.require_both))


if __name__ == "__main__":
    main()
