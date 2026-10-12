//! All features of a value column computed in one pass over each group.
//!
//! Polars collects a group's values into a list once per `apply`, and with
//! dozens of features that dominated extraction time. Instead, one `apply`
//! per value column builds a [`GroupContext`] (the group's values plus
//! statistics that several features share, each computed at most once) and
//! runs every [`Kernel`] on it, returning all of the group's features as one
//! list. After the group-by, [`KernelPlan::unpack`] turns list position `i`
//! into the named feature column `i`.

use std::cell::OnceCell;
use std::sync::Arc;

use ndarray::{Array1, ArrayView1, s};
use polars::prelude::*;

use super::f64_values;
use crate::utils::stats::population_var;

/// One group's values (nulls dropped) and the statistics features share.
///
/// Shared statistics are computed on first use, with exactly the code the
/// features used before, so sharing does not change any value.
pub struct GroupContext<'a> {
    values: &'a [f64],
    has_nulls: bool,
    x: ArrayView1<'a, f64>,
    mean: OnceCell<f64>,
    var: OnceCell<f64>,
    extremes: OnceCell<(f64, f64)>,
    diffs: OnceCell<Array1<f64>>,
    sorted: OnceCell<Vec<f64>>,
}

impl<'a> GroupContext<'a> {
    pub fn new(values: &'a [f64], has_nulls: bool) -> Self {
        Self {
            values,
            has_nulls,
            x: ArrayView1::from(values),
            mean: OnceCell::new(),
            var: OnceCell::new(),
            extremes: OnceCell::new(),
            diffs: OnceCell::new(),
            sorted: OnceCell::new(),
        }
    }

    /// The group's values, nulls dropped. Never empty.
    pub fn x(&self) -> &ArrayView1<'a, f64> {
        &self.x
    }

    /// The same values as a slice.
    pub fn values(&self) -> &'a [f64] {
        self.values
    }

    /// Whether the group had nulls before they were dropped.
    pub fn has_nulls(&self) -> bool {
        self.has_nulls
    }

    pub fn len(&self) -> usize {
        self.x.len()
    }

    pub fn is_empty(&self) -> bool {
        self.x.is_empty()
    }

    /// Arithmetic mean; NaN if any value is NaN.
    pub fn mean(&self) -> f64 {
        *self.mean.get_or_init(|| self.x.mean().unwrap_or(f64::NAN))
    }

    /// Population variance (`ddof = 0`), see [`population_var`].
    pub fn var(&self) -> f64 {
        *self.var.get_or_init(|| population_var(&self.x))
    }

    /// Population standard deviation (`ddof = 0`): the square root of
    /// [`Self::var`], exactly as `ndarray`'s `std` computes it.
    pub fn std(&self) -> f64 {
        self.var().sqrt()
    }

    /// Smallest value; NaN if any value is NaN (like `np.min`).
    pub fn min(&self) -> f64 {
        self.extremes().0
    }

    /// Largest value; NaN if any value is NaN (like `np.max`).
    pub fn max(&self) -> f64 {
        self.extremes().1
    }

    /// Minimum and maximum in one pass, with exactly the semantics of
    /// `ndarray_stats`' `min`/`max` (both NaN if any value is NaN; on ties
    /// the first value wins, which matters only for `-0.0` versus `0.0`).
    fn extremes(&self) -> (f64, f64) {
        *self
            .extremes
            .get_or_init(|| extremes(self.values.iter().copied()))
    }

    /// The values sorted ascending (`total_cmp`, so NaN sorts last).
    pub fn sorted(&self) -> &[f64] {
        self.sorted.get_or_init(|| {
            let mut sorted = self.values.to_vec();
            sorted.sort_unstable_by(f64::total_cmp);
            sorted
        })
    }

    /// Consecutive differences $x_{i+1} - x_i$ (empty for a single value).
    pub fn diffs(&self) -> &Array1<f64> {
        self.diffs
            .get_or_init(|| &self.x.slice(s![1..]) - &self.x.slice(s![..-1]))
    }
}

/// Minimum and maximum of a non-empty sequence in one pass: both NaN if any
/// value is NaN; on ties the first value wins.
pub(crate) fn extremes(values: impl IntoIterator<Item = f64>) -> (f64, f64) {
    let mut values = values.into_iter();
    let Some(first) = values.next() else {
        return (f64::NAN, f64::NAN);
    };
    if first.is_nan() {
        return (f64::NAN, f64::NAN);
    }
    let (mut min, mut max) = (first, first);
    for v in values {
        if v.is_nan() {
            return (f64::NAN, f64::NAN);
        }
        if v < min {
            min = v;
        } else if v > max {
            max = v;
        }
    }
    (min, max)
}

/// A feature (or a family of features) computed from a [`GroupContext`].
pub trait Kernel: Send + Sync {
    /// Output names without the `{column}__` prefix, in the order
    /// [`Kernel::compute`] pushes the values.
    fn names(&self) -> Vec<String>;

    /// Push one value per name onto `out`.
    fn compute(&self, ctx: &GroupContext, out: &mut Vec<f64>);
}

/// Define a [`Kernel`] with a single output: a unit struct (carrying the
/// doc comment), its output name and how to compute it from the context.
macro_rules! scalar_kernel {
    ($(#[$meta:meta])* $kernel:ident, $name:literal, |$ctx:ident| $body:expr) => {
        $(#[$meta])*
        pub struct $kernel;

        impl $crate::feature_extractors::kernel::Kernel for $kernel {
            fn names(&self) -> Vec<String> {
                vec![$name.to_string()]
            }

            fn compute(
                &self,
                $ctx: &$crate::feature_extractors::kernel::GroupContext,
                out: &mut Vec<f64>,
            ) {
                out.push($body);
            }
        }
    };
}
pub(crate) use scalar_kernel;

/// The kernels of one value column, run by a single `apply` per group.
#[derive(Default)]
pub struct KernelPlan {
    kernels: Vec<Box<dyn Kernel>>,
    names: Vec<String>,
}

impl KernelPlan {
    pub fn push(&mut self, kernel: impl Kernel + 'static) {
        self.names.extend(kernel.names());
        self.kernels.push(Box::new(kernel));
    }

    pub fn is_empty(&self) -> bool {
        self.kernels.is_empty()
    }

    /// Every feature of one group; NaN for all of them if the group has no
    /// non-null values.
    fn run(&self, s: Column) -> PolarsResult<Column> {
        let has_nulls = s.null_count() > 0;
        let s = s.drop_nulls();
        let mut out = Vec::with_capacity(self.names.len());
        if s.is_empty() {
            out.resize(self.names.len(), f64::NAN);
        } else {
            let ca = f64_values(&s)?;
            let ctx = GroupContext::new(ca.cont_slice()?, has_nulls);
            for kernel in &self.kernels {
                kernel.compute(&ctx, &mut out);
            }
        }
        polars_ensure!(
            out.len() == self.names.len(),
            ComputeError: "feature kernels produced {} values for {} names",
            out.len(),
            self.names.len()
        );
        Ok(Column::new(PlSmallStr::EMPTY, out))
    }

    /// The name of the list column that holds `column`'s features until
    /// [`Self::unpack`].
    pub fn packed_name(column: &str) -> String {
        format!("__tsfx_features__{column}")
    }

    /// The aggregation: one list of all features per group.
    pub fn aggregation(self: &Arc<Self>, column: &str) -> Expr {
        let plan = Arc::clone(self);
        col(column)
            .apply(
                move |s| plan.run(s),
                |_, _| Ok(Field::new(PlSmallStr::EMPTY, DataType::Float64)),
            )
            .alias(Self::packed_name(column))
    }

    /// Replace `column`'s packed list column in the aggregated `df` with one
    /// named column per feature (`{column}__{name}`). Done eagerly on the
    /// small aggregated frame: a lazy `list.get` per feature would plan a
    /// second query on every call.
    pub fn unpack(&self, df: &mut DataFrame, column: &str) -> PolarsResult<()> {
        let packed = df.drop_in_place(&Self::packed_name(column))?;
        let list = packed.list()?.rechunk();
        let inner = list.get_inner();
        let inner = inner.f64()?.rechunk();
        let flat = inner.cont_slice()?;
        let n = self.names.len();
        let mut columns: Vec<Vec<f64>> = vec![Vec::with_capacity(list.len()); n];
        for arr in list.downcast_iter() {
            for row in arr.offsets().buffer().windows(2) {
                let values = &flat[row[0] as usize..row[1] as usize];
                polars_ensure!(
                    values.len() == n,
                    ComputeError: "a group produced {} feature values for {} names",
                    values.len(),
                    n
                );
                for (column, &value) in columns.iter_mut().zip(values) {
                    column.push(value);
                }
            }
        }
        let columns = self
            .names
            .iter()
            .zip(columns)
            .map(|(name, values)| Column::new(format!("{column}__{name}").into(), values))
            .collect::<Vec<_>>();
        df.hstack_mut(&columns)?;
        Ok(())
    }
}
