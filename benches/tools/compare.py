"""Compare two samply profiles of the same bench, before and after a change.

Usage: python benches/tools/compare.py BEFORE.json.gz AFTER.json.gz [TOP]

Shares are of busy samples (idle threads excluded), so a change that only
shifts how long threads wait does not show up as a gain or regression.
Inlining can move a function's time between ``f`` and ``_f`` from one build to
the next, so both are counted under ``f``.

A profile shows shares, not speed: confirm a gain with criterion
(``--save-baseline`` before, ``--baseline`` after, back to back).
"""

import sys
from collections import Counter

from _profile import leaf_kind, stacks, tsfx_function


def summarise(path: str) -> tuple[int, Counter[str]]:
    """Busy sample count and per-category counts for one profile."""
    counts: Counter[str] = Counter()
    busy = 0
    for _, chain in stacks(path):
        kind = leaf_kind(chain)
        if kind == "idle":
            continue
        busy += 1
        owner = tsfx_function(chain)
        if kind:
            counts[f"[{kind}] all"] += 1
            if owner:
                counts[f"[{kind}] inside tsfx functions"] += 1
        if owner:
            counts["[tsfx] all feature code"] += 1
            counts[owner.rsplit("::", 1)[-1].lstrip("_")] += 1
    return busy, counts


def main(before: str, after: str, top: int) -> None:
    """Print the before/after shares of busy time."""
    busy_a, a = summarise(before)
    busy_b, b = summarise(after)
    totals = sorted(k for k in a | b if k.startswith("["))
    functions = sorted(
        (k for k in a | b if not k.startswith("[")),
        key=lambda k: -(a[k] / busy_a + b[k] / busy_b),
    )[:top]
    print(f"{'share of busy samples':48s} {'before':>7s} {'after':>7s}")
    for k in totals + functions:
        pa, pb = 100 * a[k] / busy_a, 100 * b[k] / busy_b
        print(f"{k[:48]:48s} {pa:6.1f}% {pb:6.1f}%  ({pb - pa:+.1f})")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 15)
