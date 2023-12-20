import polars as pl
from tsfx import (
    DynamicGroupBySettings,
    ExtractionSettings,
    FeatureSetting,
    extract_features,
)

lf = pl.scan_csv("test_data/all_stocks_5yr.csv")
lf = lf.drop_nulls()

dyn_opts = DynamicGroupBySettings(
    time_col="date",
    every="1y",
    period="1y",
    offset="0",
    datetime_format="%Y-%m-%d",
)

opts = ExtractionSettings(
    grouping_col="Name",
    value_cols=["open", "high", "low", "close", "volume"],
    feature_setting=FeatureSetting.Efficient,
    dynamic_settings=dyn_opts,
)
gdf = extract_features(lf, opts)
print(gdf.sort(by=pl.col("Name")))

opts = ExtractionSettings(
    grouping_col="Name",
    feature_setting=FeatureSetting.Efficient,
    value_cols=["open", "high", "low", "close", "volume"],
)
gdf = extract_features(lf, opts)
print(gdf.sort(by=pl.col("Name")))
