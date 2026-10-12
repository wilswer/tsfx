# Profiling tools

Helpers for the performance work on the stocks benches
(`benches/benchmarks/extract_stocks.rs`). Run them from the repository root.

| Script | What it does |
|---|---|
| `snapshot.py OUT.json` | Writes every feature value for all tiers and three configs. `cmp` a snapshot from before a change with one from after (extension rebuilt) to show the change keeps every value bit-identical. |
| `hot.py PROFILE.json.gz [TOP]` | Hot spots of one samply profile: idle/copy/alloc share, time per innermost tsfx function, self time. |
| `compare.py BEFORE.json.gz AFTER.json.gz [TOP]` | Before/after shares of busy time (idle threads excluded). |

## Workflow

```bash
cargo bench --no-run                       # prints the bench binary path
cargo bench -- --save-baseline before stocks_
samply record --save-only --unstable-presymbolicate -r 2000 -o before.json.gz \
    -- BENCH_BINARY --bench --profile-time 20 stocks_efficient

# ... make the change, maturin develop --release, cargo bench --no-run ...

cargo bench -- --baseline before stocks_   # the speed verdict
samply record ... -o after.json.gz ...
python benches/tools/compare.py before.json.gz after.json.gz
```

A profile shows where time goes, not whether a change is faster: decide that
with criterion, running the baseline and the change back to back on the same
machine. On Linux, samply needs `kernel.perf_event_paranoid` at 1 or lower.
