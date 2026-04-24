"""
Pre-written, R-verified test suite for the GLM-based claims reserving translation.

Expected values were computed by running the original R script (reserving_glm.R)
with options(digits=15) and are stored in expected_values_reserving_glm.json.
Tests use pytest.approx with rel=1e-4 tolerance for floating-point comparisons.
Stochastic outputs (bootstrap) are tested for shape and sanity only, not exact values.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "translated"))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import reserving_glm as rg

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(BASE_DIR, "examples", "difficult")
CLAIMS_CSV = os.path.join(DATA_DIR, "claims_triangle.csv")
DB_PATH = os.path.join(DATA_DIR, "policies.db")
EXPECTED_PATH = os.path.join(os.path.dirname(__file__), "expected_values_reserving_glm.json")

with open(EXPECTED_PATH, "r", encoding="utf-8") as f:
    EXPECTED = json.load(f)


@pytest.fixture(scope="module")
def triangle():
    return rg.load_claims_triangle(CLAIMS_CSV)


@pytest.fixture(scope="module")
def policy_data():
    return rg.load_policy_data(DB_PATH)


@pytest.fixture(scope="module")
def glm_data(triangle):
    return rg.prepare_glm_data(triangle)


@pytest.fixture(scope="module")
def glm_result(glm_data):
    return rg.fit_odp_glm(glm_data)


@pytest.fixture(scope="module")
def predictions(triangle, glm_result):
    model = glm_result["model"] if isinstance(glm_result, dict) else glm_result.model
    return rg.predict_reserves(model, triangle.shape[0], triangle.shape[1])


@pytest.fixture(scope="module")
def analysis_result():
    return rg.run_analysis(CLAIMS_CSV, DB_PATH, n_boot=1000, seed=42)


# ── DATA / FORMAT TESTS ──────────────────────────────────────────────────────

@pytest.mark.data
def test_module_public_functions_exist():
    """Verify required public API functions are present and callable.
    Expected: all named functions exist and are callable
    Actual: computed at runtime
    """
    for name in EXPECTED["required_module_functions"]:
        assert hasattr(rg, name), f"Missing function: {name}"
        assert callable(getattr(rg, name)), f"Function not callable: {name}"


@pytest.mark.data
def test_claims_triangle_loads_with_expected_shape(triangle):
    """Verify claims triangle loads and has expected dimensions.
    Expected: shape matches expected_values JSON
    Actual: computed at runtime
    """
    assert isinstance(triangle, pd.DataFrame)
    assert tuple(triangle.shape) == tuple(EXPECTED["triangle_dim"])


@pytest.mark.data
def test_claims_triangle_columns_and_index(triangle):
    """Verify triangle column names and origin-year index labels are correct.
    Expected: dev_1..dev_15 columns and 2005..2019 index range
    Actual: computed at runtime
    """
    assert list(triangle.columns) == EXPECTED["triangle_columns"]
    assert int(triangle.index[0]) == EXPECTED["first_origin_year"]
    assert int(triangle.index[-1]) == EXPECTED["last_origin_year"]


@pytest.mark.data
def test_claims_triangle_missing_pattern_is_lower_triangle(triangle):
    """Verify triangle has no unexpected NaN values in observed cells.
    Expected: upper triangle observed and lower triangle missing
    Actual: computed at runtime
    """
    n = triangle.shape[0]
    for i in range(n):
        observed = triangle.iloc[i, : n - i]
        missing = triangle.iloc[i, n - i :]
        assert observed.notna().all()
        if len(missing) > 0:
            assert missing.isna().all()


@pytest.mark.data
def test_policy_data_loads_and_contains_expected_tables(policy_data):
    """Verify SQLite policy data loads into expected table objects.
    Expected: dict with premium_by_year, policies, and claims tables
    Actual: computed at runtime
    """
    assert isinstance(policy_data, dict)
    assert "premium_by_year" in policy_data
    assert "policies" in policy_data
    assert "claims" in policy_data
    assert isinstance(policy_data["premium_by_year"], pd.DataFrame)
    assert isinstance(policy_data["policies"], pd.DataFrame)
    assert isinstance(policy_data["claims"], pd.DataFrame)
    assert len(policy_data["policies"]) == EXPECTED["n_policies"]
    assert len(policy_data["claims"]) == EXPECTED["n_claims"]


@pytest.mark.data
def test_premium_summary_shape_and_columns(policy_data):
    """Verify premium summary has expected shape and columns.
    Expected: 15 rows, 4 columns with premium summary fields
    Actual: computed at runtime
    """
    premium = policy_data["premium_by_year"]
    assert tuple(premium.shape) == (
        EXPECTED["premium_by_year_rows"],
        EXPECTED["premium_by_year_cols"],
    )
    assert list(premium.columns) == [
        "origin_year",
        "total_earned_premium",
        "total_written_premium",
        "n_policies",
    ]


@pytest.mark.data
def test_prepare_glm_data_returns_expected_structure(glm_data):
    """Verify GLM preparation returns expected rows, columns, and dtypes.
    Expected: 120 rows and factor/category-like origin/dev fields present
    Actual: computed at runtime
    """
    assert isinstance(glm_data, pd.DataFrame)
    assert glm_data.shape[0] == EXPECTED["glm_rows"]
    assert glm_data.shape[1] == EXPECTED["glm_cols"]
    for col in ["origin", "dev", "incremental", "origin_f", "dev_f"]:
        assert col in glm_data.columns
    assert np.issubdtype(glm_data["origin"].dtype, np.integer)
    assert np.issubdtype(glm_data["dev"].dtype, np.integer)
    assert np.issubdtype(glm_data["incremental"].dtype, np.number)


@pytest.mark.data
def test_fit_and_prediction_return_types(glm_result, predictions):
    """Verify GLM fit result and reserve predictions return usable objects.
    Expected: fit contains model and phi; predictions are DataFrame with predicted column
    Actual: computed at runtime
    """
    assert isinstance(glm_result, dict)
    assert "model" in glm_result
    assert "phi" in glm_result
    assert isinstance(glm_result["phi"], (float, np.floating))
    assert isinstance(predictions, pd.DataFrame)
    assert "predicted" in predictions.columns
    assert predictions.shape[0] == EXPECTED["predictions_rows"]


# ── CONTENT / NUMERICAL TESTS ────────────────────────────────────────────────

@pytest.mark.content
def test_dispersion_parameter_matches_r(glm_result):
    """Verify fitted dispersion parameter matches R ground truth.
    Expected: phi approximately equals expected_values JSON
    Actual: computed at runtime
    """
    assert glm_result["phi"] == pytest.approx(EXPECTED["phi"], rel=1e-4)


@pytest.mark.content
def test_predictions_sum_to_expected_total_reserve(predictions):
    """Verify predicted lower-triangle amounts sum to total reserve from R.
    Expected: total reserve approximately equals expected_values JSON
    Actual: computed at runtime
    """
    total_reserve = float(predictions["predicted"].sum())
    assert total_reserve == pytest.approx(EXPECTED["total_reserve"], rel=1e-4)


@pytest.mark.content
def test_reserves_by_origin_match_r(predictions):
    """Verify reserve totals by origin period match R ground truth.
    Expected: grouped reserves approximately equal expected_values JSON
    Actual: computed at runtime
    """
    grouped = predictions.groupby("origin", as_index=True)["predicted"].sum()
    for origin, expected_value in EXPECTED["reserves_by_origin"].items():
        assert float(grouped.loc[int(origin)]) == pytest.approx(expected_value, rel=1e-4)


@pytest.mark.content
def test_run_analysis_total_reserve_matches_r(analysis_result):
    """Verify end-to-end analysis returns total reserve consistent with R.
    Expected: total reserve approximately equals expected_values JSON
    Actual: computed at runtime
    """
    assert isinstance(analysis_result, dict)
    assert float(analysis_result["total_reserve"]) == pytest.approx(
        EXPECTED["total_reserve"], rel=1e-4
    )


@pytest.mark.content
def test_first_year_earned_premium_matches_r(policy_data):
    """Verify first origin year earned premium matches R ground truth.
    Expected: first year earned premium approximately equals expected_values JSON
    Actual: computed at runtime
    """
    premium = policy_data["premium_by_year"]
    first_earned = float(premium.iloc[0]["total_earned_premium"])
    assert first_earned == pytest.approx(EXPECTED["first_year_earned"], rel=1e-4)


@pytest.mark.content
def test_bootstrap_output_has_expected_shape_and_sanity(analysis_result):
    """Verify bootstrap output length and basic sanity for stochastic results.
    Expected: 1000 finite nonnegative simulations
    Actual: computed at runtime
    """
    boot = analysis_result["boot_results"]
    assert isinstance(boot, (np.ndarray, list, pd.Series))
    boot_arr = np.asarray(boot, dtype=float)
    assert len(boot_arr) == EXPECTED["boot_length"]
    assert np.isfinite(boot_arr).all()
    assert (boot_arr >= 0).all()


@pytest.mark.content
def test_bootstrap_reproducibility_with_seed(glm_result, glm_data, triangle):
    """Verify bootstrap is reproducible within Python when using the same seed.
    Expected: repeated calls with same seed produce identical arrays
    Actual: computed at runtime
    """
    model = glm_result["model"]
    phi = glm_result["phi"]
    boot1 = np.asarray(
        rg.bootstrap_reserves(model, glm_data, triangle, n_boot=25, phi=phi, seed=42),
        dtype=float,
    )
    boot2 = np.asarray(
        rg.bootstrap_reserves(model, glm_data, triangle, n_boot=25, phi=phi, seed=42),
        dtype=float,
    )
    assert boot1.shape == boot2.shape
    assert np.allclose(boot1, boot2)
