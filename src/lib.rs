pub mod extract;
pub mod feature_extractors;

use crate::extract::{lazy_feature_df, DynamicGroupBySettings, ExtractionSettings, FeatureSetting};
use pyo3::prelude::*;
use pyo3_polars::{PyDataFrame, PyLazyFrame};

#[pyclass(name = "FeatureSetting")]
#[derive(Clone)]
enum PyFeatureSetting {
    Minimal,
    Efficient,
    Comprehensive,
}

#[pyclass(name = "ExtractionSettings")]
#[derive(Clone)]
struct PyExtractionSettings {
    grouping_col: String,
    value_cols: Vec<String>,
    feature_setting: PyFeatureSetting,
    dynamic_settings: Option<PyDynamicGroupBySettings>,
}

#[pyclass(name = "DynamicGroupBySettings")]
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
    #[new]
    fn new(
        grouping_col: String,
        value_cols: Vec<String>,
        feature_setting: PyFeatureSetting,
        dynamic_settings: Option<PyDynamicGroupBySettings>,
    ) -> Self {
        PyExtractionSettings {
            grouping_col,
            value_cols,
            feature_setting,
            dynamic_settings,
        }
    }
}

#[pymethods]
impl PyDynamicGroupBySettings {
    #[new]
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
        if opts.dynamic_settings.is_none() {
            ExtractionSettings {
                grouping_col: opts.grouping_col,
                value_cols: opts.value_cols,
                feature_setting: opts.feature_setting.into(),
                dynamic_settings: None,
            }
        } else {
            ExtractionSettings {
                grouping_col: opts.grouping_col,
                value_cols: opts.value_cols,
                feature_setting: opts.feature_setting.into(),
                dynamic_settings: Some(opts.dynamic_settings.unwrap().into()),
            }
        }
    }
}

#[pyfunction]
fn extract_features(df: PyLazyFrame, opts: PyExtractionSettings) -> PyResult<PyDataFrame> {
    let df = df.into();
    let opts = opts.into();
    Ok(PyDataFrame(lazy_feature_df(df, opts).collect().unwrap()))
}

/// A Python module implemented in Rust.
#[pymodule]
fn tsfx(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyFeatureSetting>()?;
    m.add_class::<PyExtractionSettings>()?;
    m.add_class::<PyDynamicGroupBySettings>()?;
    m.add_function(wrap_pyfunction!(extract_features, m)?)?;
    Ok(())
}
