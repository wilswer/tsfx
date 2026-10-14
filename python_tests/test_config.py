import polars as pl
import pytest
from tsfx import (
    ExtractionSettings,
    FeatureSetting,
    extract_features,
)


def test_empty_config():
    df = pl.DataFrame(
        {
            "id": ["a", "b", "b", "c", "c", "c", "d", "d", "d", "d"],
            "val": [1.0, 1.0, 2.0, 1.0, 2.0, 3.0, 1.0, 1.0, 1.0, 1.0],
        },
    ).lazy()
    opts = ExtractionSettings(
        grouping_cols=["id"],
        feature_setting=FeatureSetting.Efficient,
        value_cols=["val"],
        config_path="./python_tests/data/.tsfx-config-empty.toml",
    )
    gdf = extract_features(df, opts)
    print(gdf.head())
    assert gdf.shape == (4, 1)


def test_invalid_aggregator_raises_value_error():
    df = pl.DataFrame({"id": ["a"] * 6, "val": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    opts = ExtractionSettings(
        grouping_cols=["id"],
        feature_setting=FeatureSetting.Efficient,
        value_cols=["val"],
        config_path="./python_tests/data/.tsfx-config-invalid-aggregator.toml",
    )
    with pytest.raises(ValueError, match="mena"):
        extract_features(df.lazy(), opts)


def test_missing_config_path_raises_value_error():
    df = pl.DataFrame({"id": ["a"] * 3, "val": [1.0, 2.0, 3.0]})
    opts = ExtractionSettings(
        grouping_cols=["id"],
        feature_setting=FeatureSetting.Minimal,
        value_cols=["val"],
        config_path="./python_tests/data/no-such-config.toml",
    )
    with pytest.raises(ValueError, match="no-such-config.toml"):
        extract_features(df.lazy(), opts)


def test_default_config_is_silent(tmp_path, monkeypatch, capfd):
    # No config_path and no .tsfx-config.toml in the working directory: the
    # default config, without printing anything
    monkeypatch.chdir(tmp_path)
    df = pl.DataFrame({"id": ["a"] * 3, "val": [1.0, 2.0, 3.0]})
    opts = ExtractionSettings(
        grouping_cols=["id"],
        feature_setting=FeatureSetting.Minimal,
        value_cols=["val"],
    )
    gdf = extract_features(df.lazy(), opts)
    assert "val__mean" in gdf.columns
    assert capfd.readouterr() == ("", "")
