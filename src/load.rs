use polars::prelude::*;
use std::{env, fs, path::PathBuf};

fn read_imo(imo: impl Into<String>) -> PolarsResult<LazyFrame> {
    let mut schema = Schema::new();
    schema.with_column("HEADING".into(), DataType::Float32);
    schema.with_column("TIMESTAMP".into(), DataType::Utf8);
    schema.with_column("SOG".into(), DataType::Float32);
    schema.with_column("DRAUGHT".into(), DataType::Float32);
    schema.with_column("DESTINATION".into(), DataType::Utf8);
    schema.with_column("ETA".into(), DataType::Date);
    schema.with_column("LATITUDE".into(), DataType::Float32);
    schema.with_column("LONGITUDE".into(), DataType::Float32);
    LazyCsvReader::new(format!(
        "tolkai_stuff/ais_clean/{imo}.csv",
        imo = imo.into()
    ))
    .has_header(true)
    .with_separator(b';')
    .with_dtype_overwrite(Some(&schema))
    .finish()
}

pub fn vessel_lazyframe() -> PolarsResult<LazyFrame> {
    println!(
        "Current working directory: {:?}",
        get_current_working_dir()?
    );
    let mut paths: Vec<_> = fs::read_dir("tolkai_stuff/ais_clean/")
        .unwrap()
        .map(|r| r.unwrap())
        .collect();
    paths.sort_by_key(|dir| dir.path());
    let mut dfs = Vec::new();

    for path in paths {
        let imo = path
            .path()
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .to_string()
            .split(".")
            .collect::<Vec<&str>>()[0]
            .to_string();

        let df = read_imo(&imo).unwrap();
        let imo_col = lit(imo);
        let options = StrptimeOptions {
            format: Some("%Y-%m-%d %H:%S:%M".into()),
            ..Default::default()
        };
        let fdf = df
            .select([
                col("TIMESTAMP").str().to_date(options),
                col("SOG").cast(DataType::Float32),
            ])
            .filter(col("SOG").gt(2.0))
            .filter(col("SOG").lt(40.0))
            .sort(
                "TIMESTAMP",
                SortOptions {
                    descending: false,
                    multithreaded: true,
                    ..Default::default()
                },
            )
            .with_column(imo_col.alias("IMO"));
        dfs.push(fdf);
        break;
    }
    println!("Concatenating...");
    let cdf = concat(
        dfs,
        UnionArgs {
            parallel: true,
            ..Default::default()
        },
    )?;
    Ok(cdf)
}

pub fn vessel_test_lazyframe() -> PolarsResult<LazyFrame> {
    let mut schema = Schema::new();
    schema.with_column("HEADING".into(), DataType::Float32);
    schema.with_column("TIMESTAMP".into(), DataType::Utf8);
    schema.with_column("SOG".into(), DataType::Float32);
    schema.with_column("DRAUGHT".into(), DataType::Float32);
    schema.with_column("DESTINATION".into(), DataType::Utf8);
    schema.with_column("ETA".into(), DataType::Date);
    schema.with_column("LATITUDE".into(), DataType::Float32);
    schema.with_column("LONGITUDE".into(), DataType::Float32);
    schema.with_column("VESSELID".into(), DataType::Utf8);

    let df = LazyCsvReader::new("../kongsberg/test_vessel_op_data.csv")
        .with_dtype_overwrite(Some(&schema))
        .finish()
        .unwrap();
    let options = StrptimeOptions {
        format: Some("%Y-%m-%d %H:%S:%M".into()),
        ..Default::default()
    };
    Ok(df.select([
        col("TIMESTAMP").str().to_date(options),
        col("SOG").cast(DataType::Float32),
        col("VESSELID"),
    ]))
}
fn get_current_working_dir() -> std::io::Result<PathBuf> {
    env::current_dir()
}

#[cfg(test)]
mod tests {
    use super::vessel_test_lazyframe;

    #[test]
    fn load_test_data() {
        let _ = vessel_test_lazyframe().unwrap().collect().unwrap();
    }
}
