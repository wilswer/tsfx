r"""Where a samply profile spends its time.

Usage: python benches/tools/hot.py PROFILE.json.gz [TOP]

Prints the time attributed to the innermost tsfx feature function on the
stack (inclusive), the busiest leaf functions (self time), and how much of
the profile is idle threads, copying and allocation.

Record a profile of a criterion bench with, for example:

    cargo bench --no-run   # prints the bench binary path
    samply record --save-only --unstable-presymbolicate -r 500 \
        -o efficient.json.gz -- BENCH_BINARY --bench --profile-time 20 \
        stocks_efficient
"""

import sys
from collections import Counter

from _profile import leaf_kind, stacks, tsfx_function


def main(path: str, top: int) -> None:
    """Print the hot spots of the profile at ``path``."""
    owner: Counter[str] = Counter()
    leaf: Counter[str] = Counter()
    kind: Counter[str] = Counter()
    total = 0
    for _, chain in stacks(path):
        total += 1
        owner[tsfx_function(chain) or "(outside tsfx feature code)"] += 1
        leaf[chain[0]] += 1
        kind[leaf_kind(chain) or "other"] += 1

    def show(title: str, counts: Counter[str]) -> None:
        print(f"\n== {title}")
        for name, n in counts.most_common(top):
            print(f"{100 * n / total:5.1f}%  {name[:110]}")

    print(f"samples: {total}")
    show("leaf kind", kind)
    show("innermost tsfx function (inclusive)", owner)
    show("self time (leaf function)", leaf)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 20)
