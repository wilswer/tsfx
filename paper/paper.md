---
title: "TSFX: A Python package for time series feature extraction"
tags:
  - Python
  - Polars
  - Rust
  - time series
  - feature extraction
authors:
  - name: Wilhelm Söderkvist Vermelin
    orcid: 0000-0002-1262-9143
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
affiliations:
  - name: RISE Research Institutes of Sweden
    index: 1
  - name: Mälardalen University
    index: 2
date: 2025-10-16
bibliography: paper.bib
---

# Summary

TSFX is a Python [@python] library for extracting features from time series
data. It is inspired by the tsfresh [@tsfresh] Python package with a special
focus on performance. In order to achieve good performance, it utilizes Polars
[@polars] which is a high performance DataFrame library written in Rust
[@rustlang] with Python bindings created through PyO3 [@pyo3]. The feature
extraction functions are also implemented in Rust for performance. Compared to
tsfresh, TSFX offers a conservative estimate of 10x performance, using the same
set of time series features.

TSFX can be installed using `pip`:

```bash
pip install tsfx
```

TSFX can also be configured using a TOML [@toml] configuration file (default name
`.tsfx-config.toml`).

# Statement of need

Time series is a ubiquitous data modality, present in many domains such as finance,
industry, meteorology, and medicine, to mention a few. As hardware to collect
and store time series data is becoming increasingly affordable, the amount of
available time series data is increasing in many domains. A common preprocessing
step when dealing with time series is feature extraction where useful features,
such as mean, variance, skewness, etc. are extracted from time series to be used
in downstream tasks such as classification, regression or clustering.
For large time series datasets, performance is important for enabling timely
data preprocessing.
TSFX is made for this purpose: extracting features from large time series
datasets.

# Acknowledgements

The TSFX package was developed within the [Vinnova](https://www.vinnova.se) projects
[DFusion](https://www.vinnova.se/en/p/dfusion---disturbance-data-fusion/),
[TolkAI](https://www.vinnova.se/en/p/intepretable-ai-from-start-to-finish/), and
[SIFT](https://www.vinnova.se/en/p/similarity-search-of-time-series-data-evaluation-of-search-engine-in-industrial-process-datasift-/).

# References
