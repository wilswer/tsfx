use polars::prelude::PolarsError;
use pyo3::{
    PyErr,
    exceptions::{PyBaseException, PyValueError},
};
use thiserror::Error;

use crate::utils::toml_reader::ConfigError;

#[derive(Error, Debug)]
pub enum ExtractionError {
    #[error("Error extracting feature")]
    FeatureError,
    #[error("Python error: {0}")]
    PythonError(#[source] PyErr),
    #[error("Polars error: {0}")]
    PolarsError(#[from] PolarsError),
    #[error("tsfx config error: {0}")]
    Config(#[from] ConfigError),
}

impl From<ExtractionError> for PyErr {
    fn from(value: ExtractionError) -> Self {
        match value {
            ExtractionError::Config(_) => PyErr::new::<PyValueError, _>(value.to_string()),
            _ => PyErr::new::<PyBaseException, _>(value.to_string()),
        }
    }
}
