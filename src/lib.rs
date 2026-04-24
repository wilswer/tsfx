pub mod error;
pub mod extract;
pub mod feature_extractors;
pub mod utils;

use error::ExtractionError;
use extract::{DynamicGroupBySettings, ExtractionSettings, FeatureSetting, lazy_feature_df};
use pyo3::prelude::*;
use pyo3_polars::{PyDataFrame, PyLazyFrame};

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

/// Configuration settings for the feature extraction process.
#[pyclass(from_py_object, name = "ExtractionSettings")]
#[derive(Clone)]
struct PyExtractionSettings {
    grouping_cols: Vec<String>,
    value_cols: Vec<String>,
    feature_setting: PyFeatureSetting,
    config_path: Option<String>,
    dynamic_settings: Option<PyDynamicGroupBySettings>,
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
    ///     config_path (str | None, optional): Path to a custom configuration JSON/YAML file. Defaults to None.
    ///     dynamic_settings (DynamicGroupBySettings | None, optional): Settings for rolling/dynamic time windows. Defaults to None.
    #[new]
    #[pyo3(signature = (grouping_cols, value_cols, feature_setting, config_path=None, dynamic_settings=None))]
    fn new(
        grouping_cols: Vec<String>,
        value_cols: Vec<String>,
        feature_setting: PyFeatureSetting,
        config_path: Option<String>,
        dynamic_settings: Option<PyDynamicGroupBySettings>,
    ) -> Self {
        PyExtractionSettings {
            grouping_cols,
            value_cols,
            feature_setting,
            config_path,
            dynamic_settings,
        }
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
///     Exception: If an underlying Polars error occurs during collection.
#[pyfunction]
#[pyo3(signature = (lf, settings, streaming=false))]
fn extract_features(
    lf: PyLazyFrame,
    settings: PyExtractionSettings,
    streaming: bool,
) -> PyResult<PyDataFrame> {
    let lf = lf.into();
    let settings = settings.into();
    let lf = if !streaming {
        lazy_feature_df(lf, settings)?
            .collect()
            .map_err(ExtractionError::PolarsError)?
    } else {
        lazy_feature_df(lf, settings)?
            .with_new_streaming(true)
            .collect()
            .map_err(ExtractionError::PolarsError)?
    };
    Ok(PyDataFrame(lf))
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
    m.add_function(wrap_pyfunction!(extract_features, m)?)?;
    Ok(())
}
