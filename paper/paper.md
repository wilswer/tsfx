---
title: "TSFX: A Python package for time series feature extraction"
tags:
  - Python
  - Polars
  - Rust
  - time series
  - feature extraction
authors:
  - name: Wilhelm Wermelin
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
data. Time series feature extraction means deriving meaningful measures from a
time series that are typically then used for downstream tasks such as
visualization, classification, regression or similarity search. Mathematically,
it is an operation that decouples the time series from the time index. Given a
univariate time series of length $T$, $X_t$, where
$t \in \{1,\,2,\,\dots,\, T-1, T\}$ a feature extractor $f$ is a mapping with
the following characteristic:

$$f: X_t \mapsto \mathbb{R}$$

where $\mathbb{R}$ denotes the real numbers. An example of a time series feature
is the mean (in which case the feature extractor is the mathematical operation
of calculating the mean):

$$ f_\mathrm{mean}(X_t) = \bar{X} = \frac{1}{T}\sum_{t=0}^T X_t. $$

TSFX is a software library for calculating a multitude of such time series
features with a focus on runtime performance.

# Statement of need

Time series is a ubiquitous data modality, present in many domains such as
finance, industry, meteorology, and medicine, to mention a few. As hardware to
collect and store time series data is becoming increasingly affordable, the
amount of available time series data is increasing in many domains.

A common preprocessing step when dealing with time series is feature extraction.
This involves calculating representative features such as mean, variance,
skewness, etc. from the time series to be used in downstream tasks such as
classification, regression or clustering. For large time series datasets,
performance is important for enabling timely data preprocessing. TSFX is made
for this purpose: extracting features from large time series datasets.

Additionally, other more mature alternatives are based on Pandas [@pandas]
DataFrames whereas TSFX is based on Polars [@polars] DataFrames. TSFX makes time
series feature extraction readily available to developers/researchers using
Polars in their projects.

# State of the field

TSFX is inspired by the tsfresh library [@tsfresh] and is to the author's
knowledge the leading time series feature extraction library for Python (at
least measured by GitHub stars). Another popular time series feature extraction
library is tsfel [@tsfel] which offers similar functionality to the tsfresh.

As mentioned above, these libraries are based on Pandas DataFrames and while it
is relatively easy to convert Pandas DataFrames to and from Polars equivalents,
TSFX has access to Polars lazy execution, which can be a significant performance
boost, being written specifically for Polars DataFrames (or LazyFrames).

# Software design

TSFX puts an emphasis on performance on large time series datasets. To this end,
it utilizes Polars [@polars] which is a fast DataFrame library written in Rust
[@rustlang] with Python bindings facilitated through PyO3 [@pyo3]. The feature
extraction functions are implemented in Rust for fast execution. To benchmark,
the "1 billion row challenge" [@1brc] was used. In this benchmark, TSFX took
approx. 250 seconds (using 64 CPU cores) to extract features vs tsfresh which
took approx. 1500 seconds (using 80 CPU cores). Running repeated benchmarks, it
has been observed that compared to tsfresh, TSFX offers about 5 to 10 times
faster feature extraction, using the same set of time series features.

TSFX can be installed using `pip`:

```bash
pip install tsfx
```

TSFX can also be configured using a TOML [@toml] configuration file (default
name `.tsfx-config.toml`).

Below is a simple example of extracting features from a time series dataset:

```python
import polars as pl
from tsfx import (
    ExtractionSettings,
    FeatureSetting,
    extract_features,
)

df = pl.DataFrame(
    {
        "id": ["a", "a", "a", "b", "b", "b", "c", "c", "c"],
        "val": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        "value": [4.0, 5.0, 6.0, 6.0, 5.0, 4.0, 4.0, 5.0, 6.0],
    },
).lazy()
settings = ExtractionSettings(
    grouping_cols=["id"],
    feature_setting=FeatureSetting.Efficient,
    value_cols=["val", "value"],
)
gdf = extract_features(df, settings)
gdf = gdf.sort(by="id")
with pl.Config(set_tbl_width_chars=80):
    print(gdf)
```

Running the code above generates a new DataFrame with the extracted features:

```bash
shape: (3, 314)
┌─────┬────────┬─────────┬─────────┬───┬─────────┬─────────┬─────────┬─────────┐
│ id  ┆ length ┆ val__su ┆ val__me ┆ … ┆ value__ ┆ value__ ┆ value__ ┆ value__ │
│ --- ┆ ---    ┆ m_value ┆ an      ┆   ┆ number_ ┆ number_ ┆ number_ ┆ number_ │
│ str ┆ u32    ┆ s       ┆ ---     ┆   ┆ peaks__ ┆ peaks__ ┆ peaks__ ┆ peaks__ │
│     ┆        ┆ ---     ┆ f64     ┆   ┆ n_3     ┆ n_5     ┆ n_10    ┆ n_50    │
│     ┆        ┆ f64     ┆         ┆   ┆ ---     ┆ ---     ┆ ---     ┆ ---     │
│     ┆        ┆         ┆         ┆   ┆ f64     ┆ f64     ┆ f64     ┆ f64     │
╞═════╪════════╪═════════╪═════════╪═══╪═════════╪═════════╪═════════╪═════════╡
│ a   ┆ 3      ┆ 6.0     ┆ 2.0     ┆ … ┆ 0.0     ┆ 0.0     ┆ 0.0     ┆ 0.0     │
│ b   ┆ 3      ┆ 6.0     ┆ 2.0     ┆ … ┆ 0.0     ┆ 0.0     ┆ 0.0     ┆ 0.0     │
│ c   ┆ 3      ┆ 6.0     ┆ 2.0     ┆ … ┆ 0.0     ┆ 0.0     ┆ 0.0     ┆ 0.0     │
└─────┴────────┴─────────┴─────────┴───┴─────────┴─────────┴─────────┴─────────┘
```

If the DataFrame has a time column, it is also possible to extract over a time
window by passing `DynamicGroupBySettings` into the feature extraction settings,
like so:
`ExtractionSettings(..., dynamic_settings=DynamicGroupBySettings(...))`.

# AI usage disclosure

Generative AI tools were used in the development of this software.

Google Gemini 3.1 Pro was used to help debugging and answer questions about the
code. Selected snippets generated by the tool have been incorporated into the
software. No AI tool added code independently to the codebase, it only provided
snippets and suggestions.

The author have reviewed, edited, validated all AI-assisted outputs and made the
core design decisions. The authors take full responsibility for the accuracy,
originality, licensing, and ethical/legal compliance of all submitted materials.

# Acknowledgements

The TSFX package was developed within the [Vinnova](https://www.vinnova.se)
projects
[DFusion](https://www.vinnova.se/en/p/dfusion---disturbance-data-fusion/),
[TolkAI](https://www.vinnova.se/en/p/intepretable-ai-from-start-to-finish/), and
[SIFT](https://www.vinnova.se/en/p/similarity-search-of-time-series-data-evaluation-of-search-engine-in-industrial-process-datasift-/).
This research work has been funded by the Knowledge Foundation within the
framework of the INDTECH (Grant Number 20200132) and INDTECH+ Research School
project (Grant Number 20220132), participating companies and Mälardalen
University.

# References
