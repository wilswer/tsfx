pub mod extract;
pub mod feature_extractors;

use crate::extract::{lazy_feature_df, DynamicGroupBySettings, ExtractionSettings};
use pyo3::prelude::*;
use pyo3_polars::{PyDataFrame, PyLazyFrame};

#[pyclass(name = "ExtractionSettings")]
#[derive(Clone)]
struct PyExtractionSettings {
    grouping_col: String,
    value_cols: Vec<String>,
    dynamic_opts: Option<PyDynamicGroupBySettings>,
}

#[pyclass(name = "DynamicGroupBySettings")]
#[derive(Clone)]
struct PyDynamicGroupBySettings {
    time_col: String,
    every: String,
    period: String,
    offset: String,
}

#[pymethods]
impl PyExtractionSettings {
    #[new]
    fn new(
        grouping_col: String,
        value_cols: Vec<String>,
        dynamic_opts: Option<PyDynamicGroupBySettings>,
    ) -> Self {
        PyExtractionSettings {
            grouping_col,
            value_cols,
            dynamic_opts,
        }
    }
}

#[pymethods]
impl PyDynamicGroupBySettings {
    #[new]
    fn new(time_col: String, every: String, period: String, offset: String) -> Self {
        PyDynamicGroupBySettings {
            time_col,
            every,
            period,
            offset,
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
        }
    }
}

impl From<PyExtractionSettings> for ExtractionSettings {
    fn from(opts: PyExtractionSettings) -> Self {
        if opts.dynamic_opts.is_none() {
            ExtractionSettings {
                grouping_col: opts.grouping_col,
                value_cols: opts.value_cols,
                dynamic_opts: None,
            }
        } else {
            ExtractionSettings {
                grouping_col: opts.grouping_col,
                value_cols: opts.value_cols,
                dynamic_opts: Some(opts.dynamic_opts.unwrap().into()),
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
    m.add_class::<PyExtractionSettings>()?;
    m.add_class::<PyDynamicGroupBySettings>()?;
    m.add_function(wrap_pyfunction!(extract_features, m)?)?;
    Ok(())
}
