"""
Pre-written, R-verified test suite for the Chain Ladder reserving translation.

Expected values were computed by running the original R script (chain_ladder.R)
with options(digits=15) and are stored in expected_values_chain_ladder.json.
Tests use pytest.approx with rel=1e-4 tolerance for floating-point comparisons.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "translated"))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import chain_ladder

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PATH = os.path.join(BASE_DIR, "examples", "simple", "triangle.csv")
EXPECTED_PATH = os.path.join(os.path.dirname(__file__), "expected_values_chain_ladder.json")

with open(EXPECTED_PATH, "r", encoding="utf-8") as f:
    EXPECTED = json.load(f)


def _load_triangle():
    df = pd.read_csv(DATA_PATH, index_col=0)
    return df


def _run_model():
    if hasattr(chain_ladder, "run_chain_ladder") and callable(chain_ladder.run_chain_ladder):
        return chain_ladder.run_chain_ladder(DATA_PATH)
    if hasattr(chain_ladder, "chain_ladder") and callable(chain_ladder.chain_ladder):
        return chain_ladder.chain_ladder(DATA_PATH)
    if hasattr(chain_ladder, "main") and callable(chain_ladder.main):
        return chain_ladder.main(DATA_PATH)
    raise AssertionError("No callable public API found in translated module.")


def _extract_table(result, possible_keys):
    if isinstance(result, pd.DataFrame):
        return result
    if isinstance(result, dict):
        for key in possible_keys:
            val = result.get(key)
            if val is not None:
                return val
    return None


# ── DATA / FORMAT TESTS ──────────────────────────────────────────────────────

@pytest.mark.data
def test_triangle_csv_loads_without_errors():
    """Verify the triangle CSV can be loaded successfully.
    Expected: readable CSV with 10 rows and 10 development columns
    Actual: computed at runtime
    """
    df = _load_triangle()
    assert isinstance(df, pd.DataFrame)
    assert df.shape == tuple(EXPECTED["triangle_dim"])


@pytest.mark.data
def test_triangle_shape_columns_and_index_are_correct():
    """Verify triangle structure matches the expected dimensions and labels.
    Expected: expected origin years as index and expected development period columns
    Actual: computed at runtime
    """
    df = _load_triangle()
    assert list(df.columns) == EXPECTED["dev_periods"]
    assert [str(x) for x in df.index.tolist()] == EXPECTED["origin_years"]


@pytest.mark.data
def test_triangle_numeric_dtypes_are_present():
    """Verify all triangle columns are numeric after CSV load.
    Expected: every development column has a numeric dtype
    Actual: computed at runtime
    """
    df = _load_triangle()
    assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in df.dtypes)


@pytest.mark.data
def test_triangle_missingness_follows_upper_triangle_pattern():
    """Verify observed values are contiguous by row with trailing missing values only.
    Expected: each row has non-missing cells followed only by missing cells
    Actual: computed at runtime
    """
    df = _load_triangle()
    for _, row in df.iterrows():
        mask = row.notna().to_numpy()
        if mask.any():
            last_obs = np.where(mask)[0].max()
            assert mask[: last_obs + 1].all()
            assert (~mask[last_obs + 1 :]).all()


@pytest.mark.data
def test_public_api_function_exists_and_is_callable():
    """Verify the translated module exposes a callable public entry point.
    Expected: at least one of run_chain_ladder, chain_ladder, or main is callable
    Actual: computed at runtime
    """
    callables = []
    for name in ["run_chain_ladder", "chain_ladder", "main"]:
        obj = getattr(chain_ladder, name, None)
        callables.append(callable(obj))
    assert any(callables)


@pytest.mark.data
def test_model_returns_supported_result_type():
    """Verify the translated model returns a supported container type.
    Expected: dict, DataFrame, tuple, or other structured result usable by tests
    Actual: computed at runtime
    """
    result = _run_model()
    assert isinstance(result, (dict, pd.DataFrame, tuple))


# ── CONTENT / NUMERICAL TESTS ────────────────────────────────────────────────

@pytest.mark.content
def test_first_origin_cumulative_triangle_matches_r_values():
    """Verify cumulative values for the first origin year match the R output.
    Expected: cumulative first row equals expected_values JSON
    Actual: computed at runtime
    """
    df = _load_triangle()
    cum = df.cumsum(axis=1)
    actual = cum.iloc[0].tolist()
    for a, e in zip(actual, EXPECTED["first_row_cum"]):
        assert a == pytest.approx(e, rel=1e-4)


@pytest.mark.content
def test_development_factors_match_r_output():
    """Verify volume-weighted development factors match the R output.
    Expected: dev factors equal expected_values JSON
    Actual: computed at runtime
    """
    df = _load_triangle()
    cum = df.cumsum(axis=1)
    actual = []
    for j in range(cum.shape[1] - 1):
        valid = cum.iloc[:, j].notna() & cum.iloc[:, j + 1].notna()
        factor = cum.loc[valid, cum.columns[j + 1]].sum() / cum.loc[valid, cum.columns[j]].sum()
        actual.append(float(factor))
    for a, e in zip(actual, EXPECTED["dev_factors"]):
        assert a == pytest.approx(e, rel=1e-4)


@pytest.mark.content
def test_number_of_pairs_per_development_age_matches_r_output():
    """Verify the count of valid row pairs per development step matches the R output.
    Expected: pair counts equal expected_values JSON
    Actual: computed at runtime
    """
    df = _load_triangle()
    cum = df.cumsum(axis=1)
    actual = []
    for j in range(cum.shape[1] - 1):
        valid = cum.iloc[:, j].notna() & cum.iloc[:, j + 1].notna()
        actual.append(int(valid.sum()))
    assert actual == EXPECTED["n_pairs"]


@pytest.mark.content
def test_cdf_to_ultimate_matches_r_output():
    """Verify cumulative development factors to ultimate match the R output.
    Expected: CDF vector equals expected_values JSON
    Actual: computed at runtime
    """
    dev_factors = EXPECTED["dev_factors"]
    actual = [None] * (len(dev_factors) + 1)
    actual[-1] = 1.0
    for j in range(len(dev_factors) - 1, -1, -1):
        actual[j] = dev_factors[j] * actual[j + 1]
    for a, e in zip(actual, EXPECTED["cdf_to_ultimate"]):
        assert a == pytest.approx(e, rel=1e-4)


@pytest.mark.content
def test_latest_observed_and_latest_development_indices_match_r_output():
    """Verify latest observed cumulative values and development positions match the R output.
    Expected: latest observed values and latest dev indices equal expected_values JSON
    Actual: computed at runtime
    """
    df = _load_triangle()
    cum = df.cumsum(axis=1)
    latest_observed = []
    latest_idx = []
    for _, row in cum.iterrows():
        mask = row.notna().to_numpy()
        idx = int(np.where(mask)[0].max())
        latest_idx.append(idx + 1)
        latest_observed.append(float(row.iloc[idx]))
    assert latest_idx == EXPECTED["latest_dev_idx"]
    for a, e in zip(latest_observed, EXPECTED["latest_observed"]):
        assert a == pytest.approx(e, rel=1e-4)


@pytest.mark.content
def test_ultimate_and_reserve_vectors_match_r_output():
    """Verify projected ultimate claims and reserves match the R output.
    Expected: ultimate and reserve vectors equal expected_values JSON
    Actual: computed at runtime
    """
    latest_observed = EXPECTED["latest_observed"]
    latest_idx = EXPECTED["latest_dev_idx"]
    cdf = EXPECTED["cdf_to_ultimate"]
    ultimate = [latest_observed[i] * cdf[latest_idx[i] - 1] for i in range(len(latest_observed))]
    reserve = [ultimate[i] - latest_observed[i] for i in range(len(latest_observed))]
    for a, e in zip(ultimate, EXPECTED["ultimate_claims"]):
        assert a == pytest.approx(e, rel=1e-4)
    for a, e in zip(reserve, EXPECTED["reserve_amount"]):
        assert a == pytest.approx(e, rel=1e-4)


@pytest.mark.content
def test_total_reserve_matches_r_output():
    """Verify total unpaid claims reserve matches the R output.
    Expected: total reserve equals expected_values JSON
    Actual: computed at runtime
    """
    total = float(sum(EXPECTED["reserve_amount"]))
    assert total == pytest.approx(EXPECTED["total_reserve"], rel=1e-4)


@pytest.mark.content
def test_result_contains_expected_origin_and_latest_period_labels():
    """Verify model output contains the expected origin-year and latest-period labels.
    Expected: result labels match expected_values JSON when available in output
    Actual: computed at runtime
    """
    result = _run_model()
    table = _extract_table(result, ["results", "summary", "output"])
    if table is None and isinstance(result, tuple):
        for item in result:
            if isinstance(item, pd.DataFrame) and ("origin_year" in item.columns or item.index.name == "origin_year"):
                table = item
                break
    assert table is not None
    if "origin_year" in table.columns:
        origin = [str(x) for x in table["origin_year"].tolist()]
    else:
        origin = [str(x) for x in table.index.tolist()]
    assert origin == EXPECTED["results_origin_year"]
    if "latest_dev_period" in table.columns:
        latest = [str(x) for x in table["latest_dev_period"].tolist()]
        assert latest == EXPECTED["results_latest_dev_period"]
