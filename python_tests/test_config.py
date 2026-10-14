import random
import re
from pathlib import Path

import polars as pl
import pytest
from tsfx import (
    ExtractionSettings,
    FeatureSetting,
    extract_features,
)

REPO_CONFIG = Path(__file__).resolve().parent.parent / ".tsfx-config.toml"


@pytest.mark.parametrize(
    "tier",
    [FeatureSetting.Minimal, FeatureSetting.Efficient, FeatureSetting.Comprehensive],
)
def test_empty_config(tier):
    # Every feature has a config section, so an empty config computes nothing
    # in any tier
    df = pl.DataFrame(
        {
            "id": ["a", "b", "b", "c", "c", "c", "d", "d", "d", "d"],
            "val": [1.0, 1.0, 2.0, 1.0, 2.0, 3.0, 1.0, 1.0, 1.0, 1.0],
        },
    ).lazy()
    opts = ExtractionSettings(
        grouping_cols=["id"],
        feature_setting=tier,
        value_cols=["val"],
        config_path="./python_tests/data/.tsfx-config-empty.toml",
    )
    gdf = extract_features(df, opts)
    print(gdf.head())
    assert gdf.shape == (4, 1)


@pytest.mark.parametrize("enabled", [True, False])
def test_sample_entropy_follows_config_section(tmp_path, enabled):
    # Comprehensive-only and O(n^2): computed only with its config section
    config = tmp_path / "config.toml"
    config.write_text("[mean]\n" + ("[sample_entropy]\n" if enabled else ""))
    df = pl.DataFrame({"id": ["a"] * 6, "val": [1.0, 2.0, 1.0, 3.0, 1.0, 2.0]})
    opts = ExtractionSettings(
        grouping_cols=["id"],
        feature_setting=FeatureSetting.Comprehensive,
        value_cols=["val"],
        config_path=str(config),
    )
    columns = extract_features(df.lazy(), opts).columns
    assert ("val__sample_entropy" in columns) == enabled
    assert "val__mean" in columns


def test_every_feature_comes_from_its_config_section(tmp_path):
    # Each section of the repo config, on its own, adds features, and together
    # they give exactly the full config's features: none is computed without
    # its section
    text = REPO_CONFIG.read_text()
    starts = [m.start() for m in re.finditer(r"^\[\w+\]$", text, re.MULTILINE)]
    sections = [
        text[a:b] for a, b in zip(starts, [*starts[1:], len(text)], strict=True)
    ]
    rng = random.Random(0)
    values = [rng.gauss(0.0, 1.0) for _ in range(60)]
    df = pl.DataFrame({"id": ["a"] * 60, "val": values}).lazy()

    def columns(config_text: str) -> set[str]:
        config = tmp_path / "config.toml"
        config.write_text(config_text)
        opts = ExtractionSettings(
            grouping_cols=["id"],
            feature_setting=FeatureSetting.Comprehensive,
            value_cols=["val"],
            config_path=str(config),
        )
        return set(extract_features(df, opts).columns) - {"id"}

    union = set()
    for section in sections:
        cols = columns(section)
        assert cols, f"no features from {section.splitlines()[0]}"
        union |= cols
    assert union == columns(text)


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
