import numpy as np
import pandas as pd
import pytest

from bike_price_prediction import build_features, group_feature, load_and_clean, run

REF_YEAR = 2024


def make_bike_csv(path, n=300, seed=0):
    rng = np.random.default_rng(seed)
    year = rng.integers(2005, 2025, n)
    km = rng.integers(500, 90000, n)
    brand = rng.choice(["Honda", "Bajaj", "Hero", "TVS", "Yamaha"], n)
    seller = rng.choice(["Individual", "Dealer"], n)
    owner = rng.choice(["1st owner", "2nd owner", "3rd owner"], n)
    age = REF_YEAR - year
    base = 120000 * np.exp(-0.08 * age) * np.exp(-km / 150000)
    price = (base * np.exp(rng.normal(0, 0.05, n))).round()
    showroom = np.where(rng.uniform(size=n) < 0.1, np.nan, base * 1.3)
    df = pd.DataFrame({
        "name": [f"{b} Splendor {i}" for i, b in enumerate(brand)],
        "selling_price": price,
        "year": year,
        "seller_type": seller,
        "owner": owner,
        "km_driven": km,
        "ex_showroom_price": showroom,
    })
    df.to_csv(path, index=False)
    return path


def test_load_and_clean_drops_future_and_nonpositive(tmp_path):
    path = make_bike_csv(tmp_path / "bikes.csv")
    raw = pd.read_csv(path)
    raw.loc[0, "selling_price"] = 0
    raw.loc[1, "year"] = REF_YEAR + 5
    raw.to_csv(path, index=False)

    df, price_col, n_raw = load_and_clean(path, REF_YEAR)
    assert price_col == "selling_price"
    assert n_raw == 300
    assert (df["selling_price"] > 0).all()
    assert (df["bike_age"] >= 0).all()
    assert df["ex_showroom_price"].notna().all()


def test_missing_columns_raise(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"name": ["x"], "foo": [1]}).to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_and_clean(path, REF_YEAR)


def test_brand_grouping_limits_categories(tmp_path):
    path = make_bike_csv(tmp_path / "bikes.csv")
    df, _, _ = load_and_clean(path, REF_YEAR)
    assert df["brand"].nunique() <= 8


def test_build_features_log_target(tmp_path):
    path = make_bike_csv(tmp_path / "bikes.csv")
    df, price_col, _ = load_and_clean(path, REF_YEAR)
    X, y = build_features(df, price_col)
    assert price_col not in X.columns and "name" not in X.columns and "year" not in X.columns
    assert np.allclose(y.values, np.log1p(df[price_col].values))


@pytest.mark.parametrize("col,expected", [
    ("brand_Honda", "brand"),
    ("seller_type_Dealer", "seller_type"),
    ("owner_2nd owner", "owner"),
    ("km_driven", "km_driven"),
])
def test_group_feature(col, expected):
    assert group_feature(col) == expected


def test_end_to_end_model_fits_well(tmp_path):
    path = make_bike_csv(tmp_path / "bikes.csv")
    s = run(path, reference_year=REF_YEAR)
    assert s["best"] in s["cv"]
    assert s["test"]["r2"] > 0.5
    assert s["n_test"] == 60
    assert s["n_train"] == 240
    assert "bike_age" in s["drivers"].index or "km_driven" in s["drivers"].index
