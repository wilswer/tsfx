pub mod error;
pub mod extract;
pub mod feature_extractors;
pub mod reduction;
pub mod scaling;
pub mod utils;

use std::ffi::CString;

use error::ExtractionError;
use extract::{DynamicGroupBySettings, ExtractionSettings, FeatureSetting, collect_in_memory};
use polars::prelude::IntoLazy;
use pyo3::exceptions::{PyUserWarning, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyType;
use pyo3_polars::{PyDataFrame, PyLazyFrame};
use reduction::{Cap, Reduction};
use scaling::Scaling;

/// Defines the complexity level of the feature extraction process.
///
/// Attributes:
///     Minimal: Extracts a small, basic set of features.
///     Efficient: Extracts a balanced set of features optimized for performance.
///     Comprehensive: Extracts an exhaustive set of features.
#[pyclass(from_py_object, name = "FeatureSetting", eq, eq_int)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum PyFeatureSetting {
    Minimal,
    Efficient,
    Comprehensive,
}

/// How to scale the extracted features, column-wise across groups.
///
/// Matches scikit-learn's `StandardScaler` and `MinMaxScaler`: NaN and null
/// are ignored when fitting and kept in the output, as are infinities (which
/// scikit-learn rejects), and (near-)constant features map to 0 (`Standard`)
/// or the lower end of the range (`MinMax`). Grouping and time columns are
/// not scaled.
///
/// Attributes:
///     Standard: Zero mean and unit (population) variance.
///     MinMax: Maps each feature's [min, max] onto `feature_range`, (0, 1) by default.
#[pyclass(from_py_object, name = "ScalerSetting", eq)]
#[derive(Clone, Debug, PartialEq)]
enum PyScalerSetting {
    Standard {},
    #[pyo3(constructor = (feature_range = (0.0, 1.0)))]
    MinMax {
        feature_range: (f64, f64),
    },
}

/// How to drop redundant and uninformative features after extraction, before
/// scaling: features that are mostly missing, almost always one value,
/// near-duplicates of an earlier feature (Spearman correlation), or noise (no
/// better than shuffled shadow features at respecting which rows are
/// neighbors). Each step is turned off with `None`. The reduction is fitted
/// across the extracted rows (groups or windows), so it needs several of
/// them; it cannot be fitted on a single series.
///
/// Attributes:
///     max_missing: Drop a feature whose share of missing values (null, NaN,
///         ±inf) is above this, in [0, 1).
///     max_top_share: Drop a feature whose most frequent value covers more
///         than this share of its non-missing values, in (0, 1).
///     max_correlation: Drop a feature whose |Spearman correlation| with an
///         earlier kept feature is above this, in (0, 1).
///     noise_sensitivity: About this share of pure-noise features is dropped
///         (0.99 lets about 1 in 100 through), in [0.5, 0.999]. Higher is
///         stricter and slower and, with few features left, drops more real
///         features by mistake (a warning says so below 50 features or rows).
///         `None` turns the step off.
///     noise_neighbors: Nearest rows per row in the noise step, at least 1.
///     max_samples: Row cap for the correlation and noise steps and the cap
///         (a seeded sample above it), larger than `noise_neighbors`; the cap
///         then keeps at most `max_samples` - 1 features.
///     seed: Seed of the row sample and the noise step.
///     cap: A `CapSetting` to keep only the most informative features after
///         the noise step. Defaults to None: no cap.
#[pyclass(from_py_object, name = "ReductionSetting", eq, get_all)]
#[derive(Clone, Debug, PartialEq)]
struct PyReductionSetting {
    max_missing: Option<f64>,
    max_top_share: Option<f64>,
    max_correlation: Option<f64>,
    noise_sensitivity: Option<f64>,
    noise_neighbors: usize,
    max_samples: usize,
    seed: u64,
    cap: Option<PyCapSetting>,
}

#[pymethods]
impl PyReductionSetting {
    /// Initialize the reduction settings; invalid values raise `ValueError`.
    #[new]
    #[pyo3(signature = (
        max_missing=Some(0.05),
        max_top_share=Some(0.95),
        max_correlation=Some(0.95),
        noise_sensitivity=Some(0.99),
        noise_neighbors=10,
        max_samples=5000,
        seed=0,
        cap=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        max_missing: Option<f64>,
        max_top_share: Option<f64>,
        max_correlation: Option<f64>,
        noise_sensitivity: Option<f64>,
        noise_neighbors: usize,
        max_samples: usize,
        seed: u64,
        cap: Option<PyCapSetting>,
    ) -> PyResult<Self> {
        let setting = PyReductionSetting {
            max_missing,
            max_top_share,
            max_correlation,
            noise_sensitivity,
            noise_neighbors,
            max_samples,
            seed,
            cap,
        };
        Reduction::from(setting.clone()).validate()?;
        Ok(setting)
    }

    fn __repr__(&self) -> String {
        let opt = |v: Option<f64>| v.map_or("None".to_string(), |v| v.to_string());
        format!(
            "ReductionSetting(max_missing={}, max_top_share={}, max_correlation={}, \
             noise_sensitivity={}, noise_neighbors={}, max_samples={}, seed={}, cap={})",
            opt(self.max_missing),
            opt(self.max_top_share),
            opt(self.max_correlation),
            opt(self.noise_sensitivity),
            self.noise_neighbors,
            self.max_samples,
            self.seed,
            self.cap
                .as_ref()
                .map_or("None".to_string(), PyCapSetting::__repr__)
        )
    }
}

/// Caps the number of feature columns kept; never changes values.
///
/// The cap is the reduction's last step: after the noise step it keeps the
/// most informative of the remaining features, by greedy column subset
/// selection. Each pick is the feature that explains the most of the
/// remaining features' spread (Spearman, on ranks) not yet explained by the
/// ones picked before it; its gain, in features' worth, is the information it
/// adds per dimension (`cap_gain` in the reduction report, with the pick
/// order in `cap_rank` and the share explained so far in `cap_explained`).
///
/// It never keeps more features than the rows used minus one (all rows, or
/// a sample of `max_samples` of them): past that, the kept features explain
/// the others fully, and those are dropped as `explained`. `CapSetting()`
/// keeps picking until nothing is left to explain, so it caps the features
/// at the number of rows; `ReductionSetting(cap=None)` turns the cap off. Run
/// it with the noise step on: it cannot tell noise from a feature unrelated
/// to all the others.
///
/// Attributes:
///     max_features: Keep at most this many features, at least 1. Defaults
///         to None: no count limit.
///     min_explained: Stop once the kept features explain this share of the
///         spread of the features reaching the cap, in (0, 1]. Defaults to
///         1.0: until nothing is left to explain.
#[pyclass(from_py_object, name = "CapSetting", eq, get_all)]
#[derive(Clone, Debug, PartialEq)]
struct PyCapSetting {
    max_features: Option<usize>,
    min_explained: f64,
}

#[pymethods]
impl PyCapSetting {
    /// Initialize the cap; invalid values raise `ValueError`.
    #[new]
    #[pyo3(signature = (max_features=None, min_explained=1.0))]
    fn new(max_features: Option<i64>, min_explained: f64) -> PyResult<Self> {
        // A negative count is a ValueError, as 0 is, not usize's OverflowError
        let max_features = max_features
            .map(|max| {
                usize::try_from(max).map_err(|_| {
                    PyValueError::new_err(format!(
                        "CapSetting max_features must be at least 1, got {max}"
                    ))
                })
            })
            .transpose()?;
        let setting = PyCapSetting {
            max_features,
            min_explained,
        };
        Cap::from(setting.clone()).validate()?;
        Ok(setting)
    }

    fn __repr__(&self) -> String {
        format!(
            "CapSetting(max_features={}, min_explained={:?})",
            self.max_features
                .map_or("None".to_string(), |max| max.to_string()),
            self.min_explained
        )
    }
}

/// Configuration settings for the feature extraction process.
#[pyclass(from_py_object, name = "ExtractionSettings")]
#[derive(Clone)]
struct PyExtractionSettings {
    grouping_cols: Vec<String>,
    value_cols: Vec<String>,
    feature_setting: PyFeatureSetting,
    config_path: Option<String>,
    dynamic_settings: Option<PyDynamicGroupBySettings>,
    scaling: Option<PyScalerSetting>,
    reduction: Option<PyReductionSetting>,
}

/// Settings for performing dynamic, time-based group-by operations.
#[pyclass(from_py_object, name = "DynamicGroupBySettings")]
#[derive(Clone)]
struct PyDynamicGroupBySettings {
    time_col: String,
    every: String,
    period: String,
    offset: String,
    datetime_format: Option<String>,
}

#[pymethods]
impl PyExtractionSettings {
    /// Initialize the extraction settings.
    ///
    /// Args:
    ///     grouping_cols (list[str]): The columns used to group the data (e.g., IDs).
    ///     value_cols (list[str]): The columns containing the numerical values to extract features from.
    ///     feature_setting (FeatureSetting): The complexity/depth of features to calculate.
    ///     config_path (str | None, optional): Path to a TOML feature config file, which must exist. Defaults to None: `.tsfx-config.toml` in the working directory if present, else the default config.
    ///     dynamic_settings (DynamicGroupBySettings | None, optional): Settings for rolling/dynamic time windows. Defaults to None.
    ///     scaling (ScalerSetting | None, optional): Scale the features across groups after extraction. Defaults to None: no scaling.
    ///     reduction (ReductionSetting | None, optional): Drop redundant and uninformative features after extraction, before scaling. Defaults to None: no reduction.
    #[new]
    #[pyo3(signature = (grouping_cols, value_cols, feature_setting, config_path=None, dynamic_settings=None, scaling=None, reduction=None))]
    fn new(
        grouping_cols: Vec<String>,
        value_cols: Vec<String>,
        feature_setting: PyFeatureSetting,
        config_path: Option<String>,
        dynamic_settings: Option<PyDynamicGroupBySettings>,
        scaling: Option<&Bound<PyAny>>,
        reduction: Option<PyReductionSetting>,
    ) -> PyResult<Self> {
        let scaling = match scaling {
            // A variant class without arguments, e.g. `ScalerSetting.Standard`
            Some(s)
                if s.cast::<PyType>()
                    .is_ok_and(|t| t.is_subclass_of::<PyScalerSetting>().unwrap_or(false)) =>
            {
                Some(s.call0()?.extract()?)
            }
            Some(s) => Some(s.extract::<PyScalerSetting>()?),
            None => None,
        };
        if let Some(s) = &scaling {
            Scaling::from(s.clone()).validate()?;
        }
        Ok(PyExtractionSettings {
            grouping_cols,
            value_cols,
            feature_setting,
            config_path,
            dynamic_settings,
            scaling,
            reduction,
        })
    }

    /// Columns the extraction reads: grouping, value and (if set) time
    /// column. Internal, used to collect only these before calling Rust.
    #[getter]
    fn _input_columns(&self) -> Vec<String> {
        let mut cols = self.grouping_cols.clone();
        cols.extend(self.value_cols.iter().cloned());
        if let Some(dynamic) = &self.dynamic_settings {
            cols.push(dynamic.time_col.clone());
        }
        cols
    }
}

#[pymethods]
impl PyDynamicGroupBySettings {
    /// Initialize dynamic time-based group-by settings.
    ///
    /// Args:
    ///     time_col (str): The name of the column containing timestamp data.
    ///     every (str): The interval of the windows (e.g., "1d", "1h").
    ///     period (str): The duration of the windows (e.g., "1d").
    ///     offset (str): The offset of the windows (e.g., "0h").
    ///     datetime_format (str | None, optional): An optional format string for parsing datetimes. Defaults to None.
    #[new]
    #[pyo3(signature = (time_col, every, period, offset, datetime_format=None))]
    fn new(
        time_col: String,
        every: String,
        period: String,
        offset: String,
        datetime_format: Option<String>,
    ) -> Self {
        PyDynamicGroupBySettings {
            time_col,
            every,
            period,
            offset,
            datetime_format,
        }
    }
}

// ... [Trait implementations remain unchanged] ...
impl From<PyFeatureSetting> for FeatureSetting {
    fn from(setting: PyFeatureSetting) -> Self {
        match setting {
            PyFeatureSetting::Minimal => FeatureSetting::Minimal,
            PyFeatureSetting::Efficient => FeatureSetting::Efficient,
            PyFeatureSetting::Comprehensive => FeatureSetting::Comprehensive,
        }
    }
}

impl From<PyScalerSetting> for Scaling {
    fn from(setting: PyScalerSetting) -> Self {
        match setting {
            PyScalerSetting::Standard {} => Scaling::Standard,
            PyScalerSetting::MinMax { feature_range } => Scaling::MinMax { feature_range },
        }
    }
}

impl From<PyReductionSetting> for Reduction {
    fn from(setting: PyReductionSetting) -> Self {
        Reduction {
            max_missing: setting.max_missing,
            max_top_share: setting.max_top_share,
            max_correlation: setting.max_correlation,
            noise_sensitivity: setting.noise_sensitivity,
            noise_neighbors: setting.noise_neighbors,
            max_samples: setting.max_samples,
            seed: setting.seed,
            cap: setting.cap.map(Cap::from),
        }
    }
}

impl From<PyCapSetting> for Cap {
    fn from(setting: PyCapSetting) -> Self {
        Cap {
            max_features: setting.max_features,
            min_explained: setting.min_explained,
        }
    }
}

impl From<PyDynamicGroupBySettings> for DynamicGroupBySettings {
    fn from(opts: PyDynamicGroupBySettings) -> Self {
        DynamicGroupBySettings {
            time_col: opts.time_col,
            every: opts.every,
            period: opts.period,
            offset: opts.offset,
            datetime_format: opts.datetime_format,
        }
    }
}

impl From<PyExtractionSettings> for ExtractionSettings {
    fn from(opts: PyExtractionSettings) -> Self {
        ExtractionSettings {
            grouping_cols: opts.grouping_cols,
            value_cols: opts.value_cols,
            feature_setting: opts.feature_setting.into(),
            config_path: opts.config_path,
            dynamic_settings: opts
                .dynamic_settings
                .map(|dyn_settings| dyn_settings.into()),
            scaling: opts.scaling.map(|scaling| scaling.into()),
            reduction: opts.reduction.map(|reduction| reduction.into()),
        }
    }
}

/// Extract time-series features from a Polars LazyFrame.
///
/// This function computes features based on the provided settings. It evaluates
/// the lazy computation graph and returns an in-memory DataFrame.
///
/// Args:
///     lf (polars.LazyFrame): The input data to extract features from.
///     settings (ExtractionSettings): The configuration controlling grouping and feature complexity.
///     streaming (bool, optional): If True, executes the query using Polars' streaming engine
///         for out-of-core processing. Defaults to False.
///
/// Returns:
///     polars.DataFrame: A new DataFrame containing the grouped IDs and their extracted features.
///
/// Raises:
///     ValueError: If the settings are invalid, or the reduction cannot be fitted
///         (fewer groups or windows than it needs).
///     Exception: If an underlying Polars error occurs during collection.
///
/// Warns:
///     UserWarning: If the reduction's noise step runs on fewer than 50
///         features or rows: it may then drop real features as noise. If
///         the cap runs without the noise step, or the rows (or the
///         `max_samples` sample) stop it before `max_features`.
#[pyfunction]
#[pyo3(signature = (lf, settings, streaming=false))]
fn extract_features(
    py: Python<'_>,
    lf: PyLazyFrame,
    settings: PyExtractionSettings,
    streaming: bool,
) -> PyResult<PyDataFrame> {
    let extraction = extract::extract(lf.into(), settings.into())?;
    warn_about(py, &extraction)?;
    let df = if !streaming {
        collect_in_memory(extraction.features).map_err(ExtractionError::PolarsError)?
    } else {
        extraction
            .features
            .with_streaming(true)
            .collect()
            .map_err(ExtractionError::PolarsError)?
    };
    Ok(PyDataFrame(df))
}

/// Extract features from an in-memory DataFrame.
///
/// Internal counterpart of `extract_features` for the non-streaming path: a
/// DataFrame crosses from Python via the Arrow C interface without copying,
/// while a LazyFrame's plan (and any in-memory data in it) is serialised.
#[pyfunction]
#[pyo3(signature = (df, settings))]
fn extract_features_df(
    py: Python<'_>,
    df: PyDataFrame,
    settings: PyExtractionSettings,
) -> PyResult<PyDataFrame> {
    let extraction = extract::extract(df.0.lazy(), settings.into())?;
    warn_about(py, &extraction)?;
    let df = collect_in_memory(extraction.features).map_err(ExtractionError::PolarsError)?;
    Ok(PyDataFrame(df))
}

/// The reduction's report for an in-memory DataFrame.
///
/// Internal counterpart of `reduction_report`: extracts the features (without
/// scaling, which the reduction does not depend on), fits the reduction and
/// returns its report, one row per feature. Raises `ValueError` when the
/// settings have no reduction.
#[pyfunction]
#[pyo3(signature = (df, settings))]
fn reduction_report_df(
    py: Python<'_>,
    df: PyDataFrame,
    settings: PyExtractionSettings,
) -> PyResult<PyDataFrame> {
    let mut settings: ExtractionSettings = settings.into();
    if settings.reduction.is_none() {
        return Err(PyValueError::new_err(
            "reduction_report needs settings with a reduction: pass \
             reduction=ReductionSetting(...) to ExtractionSettings",
        ));
    }
    settings.scaling = None;
    let extraction = extract::extract(df.0.lazy(), settings)?;
    warn_about(py, &extraction)?;
    let fit = extraction
        .reduction
        .ok_or_else(|| PyValueError::new_err("the reduction was not fitted"))?;
    Ok(PyDataFrame(fit.report))
}

/// Emits the warnings of the extraction's reduction fit as Python
/// `UserWarning`s, attributed to the caller of the Python wrapper.
fn warn_about(py: Python<'_>, extraction: &extract::Extraction) -> PyResult<()> {
    let Some(fit) = &extraction.reduction else {
        return Ok(());
    };
    let category = py.get_type::<PyUserWarning>();
    for warning in &fit.warnings {
        let message =
            CString::new(warning.as_str()).map_err(|e| PyValueError::new_err(e.to_string()))?;
        // 1 is the Python wrapper's line, 2 the user's call
        PyErr::warn(py, category.as_any(), &message, 2)?;
    }
    Ok(())
}

/// Time Series Feature Extraction module.
///
/// This module provides high-performance feature extraction capabilities for
/// time series data, leveraging a Rust core and Polars dataframes.
#[pymodule]
fn tsfx(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyFeatureSetting>()?;
    m.add_class::<PyExtractionSettings>()?;
    m.add_class::<PyDynamicGroupBySettings>()?;
    m.add_class::<PyScalerSetting>()?;
    m.add_class::<PyReductionSetting>()?;
    m.add_class::<PyCapSetting>()?;
    m.add_function(wrap_pyfunction!(extract_features, m)?)?;
    m.add_function(wrap_pyfunction!(extract_features_df, m)?)?;
    m.add_function(wrap_pyfunction!(reduction_report_df, m)?)?;
    Ok(())
}
