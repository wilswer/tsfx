mod extract;
mod feature_extractors;
mod load;

//use std::fs::File;

use polars::prelude::*;

use tsfx::extract::{lazy_feature_df, ExtractionSettings, FeatureSetting};
use tsfx::load::vessel_lazyframe;

fn main() -> PolarsResult<()> {
    println!("Reading files...");
    let cdf = vessel_lazyframe()?;
    println!("Extracting features...");
    let fdf = lazy_feature_df(
        cdf,
        ExtractionSettings {
            grouping_col: "IMO".to_string(),
            value_cols: vec!["SOG".to_string()],
            feature_setting: FeatureSetting::Efficient,
            dynamic_settings: None,
        },
    );
    println!("Collecting...");
    let df = fdf.collect()?;
    println!("{}", df);
    // let mut file = File::create("../kongsberg/tsfx_features.csv")?;
    // CsvWriter::new(&mut file).finish(&mut df).unwrap();
    Ok(())
}
