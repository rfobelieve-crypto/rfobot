# -*- coding: utf-8 -*-
"""Stage 1 gate, second half — reverse proof that the synthetic tests bite.

A green suite proves nothing until it has been watched going red for the right
reason (mistake.md 2026-09-03: a guard that can never change colour is as
useless as a broken one; 2026-08-26: if the reverse proof does NOT go red,
suspect the injection before suspecting the guard).

Each mutation breaks ONE frozen definition in profile.py, runs the suite
against the mutant in an isolated directory, and requires evidence that the
suite noticed.  Two kinds of evidence count:

  · the named check printed FAIL                        (wrong value)
  · the suite crashed while running the named test      (assert fired)

Both are catches.  Requiring only the first is what made the 2026-09-06 run
report "guard is blind" for a guard that had in fact asserted correctly.

Run:  python research/poc/tests/test_mutations.py
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "profile.py"
SUITE = HERE / "test_profile.py"

LOOKAHEAD_SLICE = r'hi = int\(np\.searchsorted\(ts, t_ref - MIN_MS, side="right"\)\)'
LOOKAHEAD_ASSERT = (r'    assert used_ts\.max\(\) \+ MIN_MS <= t_ref, \(\n'
                    r'        "look-ahead: a bar closing at/after t_ref entered the profile"\)')

# name, [(pattern, replacement), ...], test-function, check-label
MUTATIONS = [
    ("half-open bin range -> inclusive",
     [(r"b1 = np\.ceil\(high / bin_size\)\.astype\(np\.int64\)",
       "b1 = np.floor(high / bin_size).astype(np.int64) + 2")],
     "test_single_bar_uniform", "uniform: 10 bins"),

    ("look-ahead: window slides forward, assert still there (should crash)",
     [(LOOKAHEAD_SLICE, 'hi = int(np.searchsorted(ts, t_ref, side="right"))')],
     "test_lookahead_excludes_the_reference_bar", "t_ref excludes the un-closed bar"),

    ("look-ahead SILENT: window slides forward AND the assert is removed",
     [(LOOKAHEAD_SLICE, 'hi = int(np.searchsorted(ts, t_ref, side="right"))'),
      (LOOKAHEAD_ASSERT, "    pass")],
     "test_lookahead_excludes_the_reference_bar", "t_ref excludes the un-closed bar"),

    ("POC tie rule -> first tied bin instead of median price",
     [(r"poc = float\(np\.median\(\(tied \+ 0\.5\) \* bin_size\)\)",
       "poc = float(((tied + 0.5) * bin_size)[0])")],
     "test_poc_tie_takes_median_price", "poc tie -> median price"),

    ("degenerate bar -> zero bins (the max(...,1) guard dropped)",
     [(r"n = np\.maximum\(b1 - b0, 1\)", "n = b1 - b0")],
     "test_degenerate_bar", "degenerate bar -> 1 bin"),

    ("time window left edge off by one bar",
     [(r'lo = int\(np\.searchsorted\(ts, t_ref - int\(arg \* 3600_000\), side="left"\)\)',
       'lo = int(np.searchsorted(ts, t_ref - int(arg * 3600_000), side="left")) + 1')],
     "test_time_window_left_edge", "time window = 60 bars"),

    ("volume window stops one bar early",
     [(r"lo = hi - \(k \+ 1\)", "lo = hi - k")],
     "test_volume_window_hits_target", "volume window bar count"),
]


def run_suite(dirpath):
    r = subprocess.run([sys.executable, str(Path(dirpath) / "tests" / "test_profile.py")],
                       capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def evidence(out, rc, func, label):
    """Did the suite notice, and where?

    Returns (caught: bool, where: str).  A mutation is CAUGHT whenever the
    suite fails -- that is the whole question mutation testing asks.  Whether
    it failed at the *expected* test is a second, separate fact, reported but
    not required: a mutation with a wide blast radius legitimately trips an
    earlier test first.

    Getting this wrong is what made the 2026-09-06 run print "guard is blind"
    for a mutation that had in fact killed the suite in its first test.
    """
    if rc == 0:
        return False, "SUITE STAYED GREEN -- the guard is blind"
    if f"FAIL  {label}" in out:
        return True, f"caught at the expected check: {label}"
    started = [ln.strip() for ln in out.splitlines() if ln.strip().startswith("test_")]
    excs = [ln.strip() for ln in out.splitlines() if "Error:" in ln]
    site = started[-1] if started else "?"
    if site == func:
        return True, f"caught at the expected test, by assert: {excs[-1][:60] if excs else 'exception'}"
    first_fail = next((ln.strip() for ln in out.splitlines() if ln.strip().startswith("FAIL")), "")
    detail = first_fail or (excs[-1][:60] if excs else "non-zero exit")
    return True, (f"caught EARLIER than expected (at {site}, not {func}) -- "
                  f"wide blast radius: {detail}")


def main():
    src = SRC.read_text(encoding="utf-8")
    ok = True

    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "tests").mkdir()
        shutil.copy(SRC, Path(d) / "profile.py")
        shutil.copy(SUITE, Path(d) / "tests" / "test_profile.py")
        rc, out = run_suite(d)
        if rc != 0:
            print("baseline suite is NOT green -- fix that first")
            print(out[-2000:])
            sys.exit(1)
        print("baseline: green\n")

    for name, subs, func, label in MUTATIONS:
        mutant = src
        bad_anchor = False
        for pat, rep in subs:
            mutant, n = re.subn(pat, rep, mutant, count=1)
            if n != 1:
                print(f"  SKIP  {name}")
                print(f"        anchor matched {n} times, not 1 -- the INJECTION "
                      f"is broken, not the guard (mistake.md 2026-08-26)")
                bad_anchor = True
                ok = False
                break
        if bad_anchor:
            continue
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "tests").mkdir()
            (Path(d) / "profile.py").write_text(mutant, encoding="utf-8")
            shutil.copy(SUITE, Path(d) / "tests" / "test_profile.py")
            rc, out = run_suite(d)
        caught, where = evidence(out, rc, func, label)
        print(f"  {'PASS' if caught else 'FAIL'}  {name}")
        print(f"        {where}")
        if not caught:
            ok = False

    print()
    print("Stage 1 reverse-proof gate:", "ALL PASS" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
