# 1 Billion Row Benchmark

To do perform the benchmark you must first generate data (if you are in the root
directory):

```bash
python python_benchmarks/1brc/generate_data.py N
```

where `N` is a number (1_000_000_000 for the proper 1 billion row challenge).

Then run benchmark like so:

```bash
python python_benchmarks/1brc/benchmark.py
```

If `tsfx` runs out of memory you can specify how many threads polars should use,
by setting the environment variable:

```bash
export POLARS_MAX_THREADS=16
```
