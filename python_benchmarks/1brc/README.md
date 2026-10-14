# 1 Billion Row Benchmark

To perform the benchmark you must first generate data (if you are in the root
directory):

```bash
python python_benchmarks/1brc/generate_data.py N
```

where `N` is a number (1_000_000_000 for the proper 1 billion row challenge).

Then compare TSFX against TSFresh like so:

```bash
python python_benchmarks/tsfresh_compare.py --dataset 1brc --rows 10_000_000
```

`--rows` limits how many measurements are read (default 10 million, `0` for
all). For scale, on an 80-core machine with the `efficient` tier (8,859
stations, best of two timed runs):

| rows | TSFX   | TSFresh |
| ---- | ------ | ------- |
| 10M  | 1.3 s  | 52.3 s  |
| 50M  | 7.1 s  | 192.7 s |

For the full billion rows and a comparison with functime, see the
[main README](../../README.md#performance) and `library_compare.py`. See
the docstring of `tsfresh_compare.py` for how the comparison is kept fair.

Both libraries use as many threads (TSFresh: processes) as Polars does. To
limit them, for example if you run out of memory, set:

```bash
export POLARS_MAX_THREADS=16
```
