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

use ndarray::ArrayView1;
use ndarray_stats::QuantileExt;
use polars::prelude::*;

use super::f64_values;
use crate::utils::stats::population_var;

/// One group's values (nulls dropped) and the statistics features share.
///
/// Shared statistics are computed on first use, with exactly the code the
/// features used before, so sharing does not change any value.
pub struct GroupContext<'a> {
    values: &'a [f64],
    x: ArrayView1<'a, f64>,
    mean: OnceCell<f64>,
    var: OnceCell<f64>,
    min: OnceCell<f64>,
    max: OnceCell<f64>,
}

impl<'a> GroupContext<'a> {
    pub fn new(values: &'a [f64]) -> Self {
        Self {
            values,
            x: ArrayView1::from(values),
            mean: OnceCell::new(),
            var: OnceCell::new(),
            min: OnceCell::new(),
            max: OnceCell::new(),
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
        *self.min.get_or_init(|| *self.x.min().unwrap_or(&f64::NAN))
    }

    /// Largest value; NaN if any value is NaN (like `np.max`).
    pub fn max(&self) -> f64 {
        *self.max.get_or_init(|| *self.x.max().unwrap_or(&f64::NAN))
    }
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
        let s = s.drop_nulls();
        let mut out = Vec::with_capacity(self.names.len());
        if s.is_empty() {
            out.resize(self.names.len(), f64::NAN);
        } else {
            let ca = f64_values(&s)?;
            let ctx = GroupContext::new(ca.cont_slice()?);
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
