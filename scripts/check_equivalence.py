"""Pass/fail check: does a candidate benchmark run reproduce the reference losses?

    python scripts/check_equivalence.py reference.json candidate.json [--rtol 1e-4]
        [--final-rtol 0.02] [--log decisions.jsonl] [--note "why this change"]

Exits 0 when the candidate passes, 1 when it fails. Prints the largest per-step
relative difference for each loss and the speedup (median step time ratio) with
its bootstrap 95% CI. ``--log`` appends the verdict as one JSON line, so every
accepted or rejected change keeps its reason.

Tolerances agreed for the I2 plan:
  numerics-neutral changes   --rtol 1e-4
  lower precision            --rtol 1e-2 --final-rtol 0.02
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

from wrf_pinn.training.benchmark import compare_runs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("reference", type=Path)
    ap.add_argument("candidate", type=Path)
    ap.add_argument("--rtol", type=float, default=1e-4, help="per-step relative tolerance")
    ap.add_argument("--final-rtol", type=float, default=None,
                    help="tolerance on the final total loss (default: --rtol)")
    ap.add_argument("--log", type=Path, default=None, help="append the verdict to this JSONL")
    ap.add_argument("--note", default="", help="what the candidate changes, for the log")
    args = ap.parse_args()

    ref, cand = (json.loads(p.read_text()) for p in (args.reference, args.candidate))
    result = compare_runs(ref, cand, rtol=args.rtol, final_rtol=args.final_rtol)

    print(f"reference: {ref['meta'].get('label') or args.reference}  "
          f"({ref['meta']['git']['commit'][:8]})")
    print(f"candidate: {cand['meta'].get('label') or args.candidate}  "
          f"({cand['meta']['git']['commit'][:8]})")
    print(f"steps compared: {result.steps_compared}")
    for key, value in result.max_rel_diff.items():
        print(f"  {key:14s} max rel diff {value:.2e} (step {result.worst_step[key]})")
    print(f"  final total    rel diff {result.final_rel_diff:.2e}")
    s, lo, hi = result.speedup
    print(f"speedup (median step time): {s:.3f}x  95% CI {lo:.3f}-{hi:.3f}")
    for problem in result.problems:
        print(f"FAIL: {problem}")
    print("PASS" if result.passed else "FAIL")

    if args.log is not None:
        entry = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "note": args.note,
                 "reference": str(args.reference), "candidate": str(args.candidate),
                 "reference_commit": ref["meta"]["git"]["commit"],
                 "candidate_commit": cand["meta"]["git"]["commit"],
                 **dataclasses.asdict(result)}
        with args.log.open("a") as f:
            f.write(json.dumps(entry) + "\n")
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
