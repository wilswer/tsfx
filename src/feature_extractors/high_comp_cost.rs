use itertools::Itertools;
use ndarray::{Array1, Axis, Ix1};
use polars::prelude::*;

pub fn high_comp_cost_aggregators(value_cols: &[String]) -> Vec<Expr> {
    let mut aggregators = Vec::new();
    for col in value_cols {
        aggregators.push(sample_entropy(col));
    }
    aggregators
}

fn _into_subchunks(x: &Array1<f64>, chunk_size: usize) -> Vec<Array1<f64>> {
    let mut subchunks = Vec::with_capacity(x.len());
    for chunk in x.axis_windows(Axis(0), chunk_size) {
        subchunks.push(chunk.to_owned());
    }
    subchunks
}

fn _get_matches(templates: Vec<Array1<f64>>, r: f64) -> usize {
    let mut matches = 0;
    for combo in templates.into_iter().combinations(2) {
        let a = combo[0].to_owned();
        let b = combo[1].to_owned();
        let diff = a - b;
        let dist_check = diff.mapv(|x| if x.abs() < r { 1 } else { 0 }).sum();
        if dist_check == diff.len() {
            matches += 1;
        }
    }
    matches
}

fn _out(_: &Schema, _: &Field) -> Result<Field, PolarsError> {
    Ok(Field::new("".into(), DataType::Float64))
}

fn _sample_entropy(s: Column) -> Result<Option<Column>, PolarsError> {
    if s.is_empty() {
        return Ok(Some(Column::new("".into(), &[f64::NAN])));
    }
    let arr = s
        .into_frame()
        .to_ndarray::<Float64Type>(IndexOrder::C)
        .unwrap();
    let arr = arr
        .remove_axis(Axis(1))
        .into_dimensionality::<Ix1>()
        .unwrap();
    let m = 2;
    let r = 0.2 * arr.std(1.0);
    let templates_m = _into_subchunks(&arr, m);
    let matches_m = _get_matches(templates_m, r);
    let templates_m_plus_1 = _into_subchunks(&arr, m + 1);
    let matches_m_plus_1 = _get_matches(templates_m_plus_1, r);
    let out = ((matches_m as f64) / (matches_m_plus_1 as f64)).ln();
    let s = Column::new("".into(), &[out as f64]);
    Ok(Some(s))
}

pub fn sample_entropy(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float32);
    col(name)
        .apply(_sample_entropy, o)
        .get(0)
        .alias(format!("{}__sample_entropy", name))
}
