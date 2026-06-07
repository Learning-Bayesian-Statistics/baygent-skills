"""
Smoke test for the portable reporting harness.

Runs the bayesian-workflow diagnostics pipeline end-to-end on a tiny model and
exercises the causal-inference refutation harness on fixtures. This is an
INTEGRATION test: it catches contract breaks between scripts that no
single-script unit test would surface.

Regression guards (each maps to a real bug this test was written to lock down):

  1. diagnose_model.py must emit JSON-serializable output — no numpy int64 or
     xarray Dataset leaking into json.dumps. (Was: `TypeError: Object of type
     int64 is not JSON serializable`, which killed step 1 of the pipeline.)
  2. check_diagnostics.py must produce real, named next steps on the
     arviz_stats.diagnose path — not the placeholder string
     "see diagnose_model.py output for details".
  3. check_refutation.py must NOT rate a *significant* parallel-trends pre-test
     as PASS, and must not award "causal" language when it fails. (Was: the
     generic p-value rater treated low p as good, inverting the verdict.)

Run:
    conda run -n baygent python evals/smoke/test_reporting_harness.py
Exits 0 on success, 1 on any failed assertion.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BAYES_SCRIPTS = REPO / "bayesian-workflow" / "scripts"
CAUSAL_SCRIPTS = REPO / "causal-inference" / "scripts"

FAILURES: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
    if not cond:
        FAILURES.append(msg)


def run(cmd: list) -> tuple[int, str, str]:
    """Run a script under the current interpreter; force a headless MPL backend."""
    env = {**os.environ, "MPLBACKEND": "Agg"}
    proc = subprocess.run(
        [sys.executable, *map(str, cmd)], capture_output=True, text=True, env=env
    )
    return proc.returncode, proc.stdout, proc.stderr


def load_or_fail(path: Path, label: str):
    if not path.exists():
        check(False, f"{label} written")
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as e:
        check(False, f"{label} is valid JSON ({e})")
        return None


def build_idata(path: Path) -> None:
    import numpy as np
    import pymc as pm

    rng = np.random.default_rng(sum(map(ord, "smoke-test")))
    y = rng.normal(3.0, 1.0, size=80)
    with pm.Model():
        mu = pm.Normal("mu", 0, 5)
        sigma = pm.HalfNormal("sigma", 5)
        pm.Normal("obs", mu, sigma, observed=y)
        idata = pm.sample(
            400, tune=400, chains=2, progressbar=False,
            idata_kwargs={"log_likelihood": True}, random_seed=42,
        )
        pm.sample_posterior_predictive(
            idata, extend_inferencedata=True, progressbar=False
        )
    idata.to_netcdf(str(path))


def test_bayesian_pipeline(workdir: Path) -> None:
    print("\n== bayesian-workflow diagnostics pipeline ==")
    idata_path = workdir / "inference_data.nc"
    build_idata(idata_path)

    # Step 1 — diagnose (regression guard #1: must not crash on json.dumps)
    diag_json = workdir / "diagnostics.json"
    rc, _, err = run([BAYES_SCRIPTS / "diagnose_model.py",
                      "--idata", idata_path, "--output", diag_json])
    check(rc == 0, f"diagnose_model.py exits 0 (rc={rc}); stderr: {err.strip()[-200:]}")
    d = load_or_fail(diag_json, "diagnostics.json")
    if d:
        conv = d.get("convergence", {})
        check("all_ok" in conv, "diagnostics.json has convergence.all_ok")
        check("rhat" in conv and "problematic_params" in conv["rhat"],
              "convergence carries the unified rhat schema")
        check("diagnostics" not in conv,
              "raw (non-serializable) diagnose() blob not embedded")

    # Step 2 — calibration
    cal_json = workdir / "calibration.json"
    rc, _, err = run([BAYES_SCRIPTS / "calibration_check.py",
                      "--idata", idata_path, "--output", cal_json,
                      "--save-plots", "--plot-dir", workdir])
    check(rc == 0, f"calibration_check.py exits 0 (rc={rc}); stderr: {err.strip()[-200:]}")
    check((workdir / "pit_ecdf.png").exists(),
          "pit_ecdf.png saved with the name the report template references")

    # Step 3 — interpret (regression guard #2: no placeholder in next steps)
    check_json = workdir / "check_report.json"
    rc, _, err = run([BAYES_SCRIPTS / "check_diagnostics.py",
                      "--diagnostics", diag_json, "--calibration", cal_json,
                      "--output", check_json])
    check(rc == 0, f"check_diagnostics.py exits 0 (rc={rc}); stderr: {err.strip()[-200:]}")
    r = load_or_fail(check_json, "check_report.json")
    if r:
        check(bool(r.get("next_steps")), "check_report has a non-empty next_steps list")
        check("see diagnose_model.py output for details" not in json.dumps(r),
              "no placeholder parameter name leaks into the report")
        check("convergence" in r.get("summary", {}),
              "report has a convergence Assessment line")


def test_refutation_direction(workdir: Path) -> None:
    print("\n== causal-inference refutation harness ==")
    script = CAUSAL_SCRIPTS / "check_refutation.py"

    # Fixture A — parallel-trends pre-test is SIGNIFICANTLY violated (low p).
    # Regression guard #3: must rate FAIL and must not award "causal".
    bad = {
        "design": "DiD",
        "tests": {"parallel_trends_pretest": {"result": 0.01, "notes": "pre-trends differ"}},
        "sensitivity": {"tipping_point": 0.4, "strongest_observed_confounder": 0.1},
    }
    p = workdir / "refutation_bad.json"
    p.write_text(json.dumps(bad))
    out = workdir / "check_bad.json"
    rc, _, err = run([script, "--refutation", p, "--output", out])
    check(rc == 0, f"check_refutation.py exits 0 on violated-trends fixture (rc={rc})")
    r = load_or_fail(out, "check_bad.json")
    if r:
        rating = r["ratings"]["tests"]["parallel_trends_pretest"]["rating"]
        check(rating == "FAIL", f"significant parallel-trends pre-test rated FAIL (got {rating})")
        check(r["language"]["level"] != "causal",
              f"language is not 'causal' when parallel trends fails (got {r['language']['level']})")

    # Fixture B — clean refutation: causal language is earned.
    good = {
        "design": "DiD",
        "tests": {
            "parallel_trends_pretest": {"result": 0.85},
            "placebo_treatment_time": {"hdi": [-0.2, 0.3], "result": 0.05, "passes_zero": True},
        },
        "sensitivity": {"tipping_point": 0.9, "strongest_observed_confounder": 0.1},
    }
    p = workdir / "refutation_good.json"
    p.write_text(json.dumps(good))
    out = workdir / "check_good.json"
    rc, _, err = run([script, "--refutation", p, "--output", out])
    check(rc == 0, f"check_refutation.py exits 0 on clean fixture (rc={rc})")
    r = load_or_fail(out, "check_good.json")
    if r:
        rating = r["ratings"]["tests"]["parallel_trends_pretest"]["rating"]
        check(rating == "PASS", f"high-p parallel-trends pre-test rated PASS (got {rating})")
        check(r["language"]["level"] == "causal",
              f"clean refutation earns 'causal' (got {r['language']['level']})")


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        workdir = Path(td)
        test_bayesian_pipeline(workdir)
        test_refutation_direction(workdir)

    print("\n" + "=" * 56)
    if FAILURES:
        print(f"SMOKE TEST FAILED — {len(FAILURES)} assertion(s):")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("SMOKE TEST PASSED — reporting harness runs end-to-end.")


if __name__ == "__main__":
    main()
