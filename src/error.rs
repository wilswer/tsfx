use polars::prelude::PolarsError;
use pyo3::{exceptions::PyBaseException, PyErr};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ExtractionError {
    #[error("Error extracting feature")]
    FeatureError,
    #[error("Python error: {0}")]
    PythonError(#[source] PyErr),
    #[error("Polars error: {0}")]
    PolarsError(#[from] PolarsError),
}

impl From<ExtractionError> for PyErr {
    fn from(value: ExtractionError) -> Self {
        PyErr::new::<PyBaseException, _>(value.to_string())
    }
}
