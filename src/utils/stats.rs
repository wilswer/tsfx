//! Shared statistical conventions for the feature extractors.

use ndarray::ArrayView1;

/// Population variance (`ddof = 0`) of a series.
///
/// Matches `np.var` as used by tsfresh. Returns the raw value; how a zero
/// variance is handled is up to each feature.
pub fn population_var(arr: &ArrayView1<f64>) -> f64 {
    arr.var(0.0)
}

/// Population standard deviation (`ddof = 0`) of a series.
///
/// Matches `np.std` as used by tsfresh. Returns the raw value; how a zero
/// standard deviation is handled is up to each feature.
pub fn population_std(arr: &ArrayView1<f64>) -> f64 {
    arr.std(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_population_var_and_std() {
        let arr = array![1.0, 2.0, 3.0, 4.0];
        assert_eq!(population_var(&arr.view()), 1.25);
        assert_eq!(population_std(&arr.view()), 1.25_f64.sqrt());
    }

    #[test]
    fn test_single_value_is_zero() {
        let arr = array![5.0];
        assert_eq!(population_var(&arr.view()), 0.0);
        assert_eq!(population_std(&arr.view()), 0.0);
    }
}
