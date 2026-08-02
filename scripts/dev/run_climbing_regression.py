#!/usr/bin/env python3
"""Run the General-Climbing-G1 regression suite (Stage 10).

Covers all staged climbing tests plus checks that General-Tracking-G1 remains
registered and loadable.

Usage:
    python scripts/dev/run_climbing_regression.py
    python scripts/dev/run_climbing_regression.py --quick
    python scripts/dev/run_climbing_regression.py -v
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

CLIMBING_STAGE_TESTS = sorted(REPO_ROOT.glob("tests/test_climbing_stage*.py"))

REGRESSION_META_TEST = REPO_ROOT / "tests/test_climbing_regression.py"

TRACKING_GUARD_TESTS = [
    REPO_ROOT / "tests/test_task_registry.py",
    REPO_ROOT / "tests/test_train_script.py",
]

# Long-running GPU/PPO harnesses; skipped with --quick.
QUICK_EXCLUDE = {
    "test_ppo_smoke_completes",
    "test_learnability_experiment_completes",
    "test_gpu_batch_64_capacity",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run General-Climbing-G1 regression tests.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip long PPO/GPU smoke tests (CI-friendly).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Pass -v to pytest.",
    )
    parser.add_argument(
        "--no-tracking-guard",
        action="store_true",
        help="Run climbing tests only (omit tracking registry guard).",
    )
    return parser.parse_args(argv)


def build_pytest_cmd(args: argparse.Namespace) -> list[str]:
    test_paths = [str(p) for p in CLIMBING_STAGE_TESTS]
    if REGRESSION_META_TEST.is_file():
        test_paths.append(str(REGRESSION_META_TEST))
    if not args.no_tracking_guard:
        test_paths.extend(str(p) for p in TRACKING_GUARD_TESTS if p.is_file())

    cmd = [sys.executable, "-m", "pytest", *test_paths]
    if args.verbose:
        cmd.append("-v")
    else:
        cmd.extend(["-q", "--tb=short"])

    if args.quick:
        expr = " and ".join(f"not {name}" for name in sorted(QUICK_EXCLUDE))
        cmd.extend(["-k", expr])

    return cmd


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    missing = [p for p in CLIMBING_STAGE_TESTS if not p.is_file()]
    if missing:
        print("[regression] missing climbing stage test files:", file=sys.stderr)
        for path in missing:
            print(f"  - {path.relative_to(REPO_ROOT)}", file=sys.stderr)
        return 1

    cmd = build_pytest_cmd(args)
    print("[regression] running:", " ".join(cmd))
    print(
        f"[regression] climbing stage modules: "
        f"{len(CLIMBING_STAGE_TESTS)} files"
        + (" (quick mode)" if args.quick else "")
    )
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    if result.returncode == 0:
        print("[regression] PASS")
    else:
        print("[regression] FAIL", file=sys.stderr)
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
