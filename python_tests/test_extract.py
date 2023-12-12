import polars as pl
from tsfx import DynamicGroupBySettings, ExtractionSettings, extract_features

df = pl.DataFrame(
    {
        "id": ["a", "a", "a", "b", "b", "b", "c", "c", "c"],
        "val": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        "value": [4.0, 5.0, 6.0, 6.0, 5.0, 4.0, 4.0, 5.0, 6.0],
    }
).lazy()
opts = ExtractionSettings(
    grouping_col="id",
    value_cols=["val", "value"],
)
gdf = extract_features(df, opts)
print(gdf)

tdf = pl.DataFrame(
    {
        "id": ["a", "a", "a", "b", "b", "b", "c", "c", "c"],
        "time": [
            "2001-01-01",
            "2002-01-01",
            "2003-01-01",
            "2001-01-01",
            "2002-01-01",
            "2003-01-01",
            "2001-01-01",
            "2002-01-01",
            "2003-01-01",
        ],
        "val": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        "value": [4.0, 5.0, 6.0, 6.0, 5.0, 4.0, 4.0, 5.0, 6.0],
    }
).lazy()

dyn_opts = DynamicGroupBySettings(
    time_col="time",
    every="3y",
    period="3y",
    offset="0",
    datetime_format="%Y-%m-%d",
)
opts = ExtractionSettings(
    grouping_col="id",
    value_cols=["val", "value"],
    dynamic_settings=dyn_opts,
)
gdf = extract_features(tdf, opts)
print(gdf)
