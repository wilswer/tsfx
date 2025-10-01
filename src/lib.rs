pub mod error;
pub mod extract;
pub mod feature_extractors;
pub mod utils;

use error::ExtractionError;
use extract::{lazy_feature_df, DynamicGroupBySettings, ExtractionSettings, FeatureSetting};
use pyo3::prelude::*;
use pyo3_polars::{PyDataFrame, PyLazyFrame};

#[pyclass(name = "FeatureSetting", eq, eq_int)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
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
    config_path: Option<String>,
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
    #[pyo3(signature = (grouping_col, value_cols, feature_setting, config_path=None, dynamic_settings=None))]
    fn new(
        grouping_col: String,
        value_cols: Vec<String>,
        feature_setting: PyFeatureSetting,
        config_path: Option<String>,
        dynamic_settings: Option<PyDynamicGroupBySettings>,
    ) -> Self {
        PyExtractionSettings {
            grouping_col,
            value_cols,
            feature_setting,
            config_path,
            dynamic_settings,
        }
    }
}

#[pymethods]
impl PyDynamicGroupBySettings {
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
            grouping_col: opts.grouping_col,
            value_cols: opts.value_cols,
            feature_setting: opts.feature_setting.into(),
            config_path: opts.config_path,
            dynamic_settings: opts
                .dynamic_settings
                .map(|dyn_settings| dyn_settings.into()),
        }
    }
}

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

/// A Python module implemented in Rust.
#[pymodule]
fn tsfx(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyFeatureSetting>()?;
    m.add_class::<PyExtractionSettings>()?;
    m.add_class::<PyDynamicGroupBySettings>()?;
    m.add_function(wrap_pyfunction!(extract_features, m)?)?;
    Ok(())
}
