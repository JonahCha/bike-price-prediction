"""Used motorcycle resale price prediction (CarDekho / BikeDekho motorcycle data).

Model selection uses 5-fold cross-validation on the training split only.
The held-out test set is scored once, for the selected model.

Usage:
    python bike_price_prediction.py --data data/BIKE_DETAILS_synthetic.csv --outdir docs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split

from plots import save_console_image, save_figures

DEFAULT_DATA = Path("data/BIKE_DETAILS_synthetic.csv")
DEFAULT_OUTDIR = Path("docs")
DEFAULT_REFERENCE_YEAR = 2024  # matches the synthetic data; pass --reference-year to change
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
TOP_BRANDS = 8
DRIVER_GROUPS = ("brand", "seller_type", "owner")


def resolve_price_column(df: pd.DataFrame) -> str:
    if "selling_price" in df.columns:
        return "selling_price"
    candidates = [c for c in df.columns if "price" in c.lower() and "showroom" not in c.lower()]
    if not candidates:
        raise ValueError("No resale price column found (expected 'selling_price').")
    return candidates[0]


def load_and_clean(path: Path, reference_year: int) -> tuple[pd.DataFrame, str, int]:
    """Return (cleaned_frame, price_column, raw_row_count)."""
    raw = pd.read_csv(path)
    price_col = resolve_price_column(raw)
    missing = {price_col, "year", "km_driven"} - set(raw.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    n_raw = len(raw)
    df = raw.drop_duplicates().dropna(subset=[price_col, "year", "km_driven"]).copy()
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df[df[price_col] > 0]

    df["bike_age"] = reference_year - df["year"]
    df = df[df["bike_age"] >= 0].copy()

    if "name" in df.columns:
        brand = df["name"].astype(str).str.split().str[0]
        top = brand.value_counts().nlargest(TOP_BRANDS).index
        df["brand"] = brand.where(brand.isin(top), "Other")

    if "ex_showroom_price" in df.columns:
        df["ex_showroom_price"] = pd.to_numeric(df["ex_showroom_price"], errors="coerce")
        df["ex_showroom_missing"] = df["ex_showroom_price"].isna().astype(int)
        df["ex_showroom_price"] = df["ex_showroom_price"].fillna(df["ex_showroom_price"].median())

    return df, price_col, n_raw


def build_features(df: pd.DataFrame, price_col: str) -> tuple[pd.DataFrame, pd.Series]:
    """Return one-hot encoded features and a log-transformed price target."""
    y = np.log1p(df[price_col])  # log target handles the skew in resale prices
    drop = [c for c in (price_col, "name", "year") if c in df.columns]
    X = pd.get_dummies(df.drop(columns=drop), drop_first=True)
    return X, y


def build_models() -> dict:
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }


def cross_validate_models(X: pd.DataFrame, y: pd.Series) -> dict[str, dict[str, float]]:
    """Mean and std of R2 across folds on the log-price target, training data only."""
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    for name, model in build_models().items():
        scores = cross_validate(model, X, y, cv=cv, scoring="r2")["test_score"]
        results[name] = {"r2": scores.mean(), "r2_std": scores.std()}
    return results


def group_feature(col: str) -> str:
    for base in DRIVER_GROUPS:
        if col.startswith(base + "_"):
            return base
    return col


def top_drivers(rf_model, X_columns) -> pd.Series:
    imp = pd.Series(rf_model.feature_importances_, index=X_columns)
    return imp.groupby(lambda c: group_feature(c)).sum().sort_values(ascending=False)


def run(path: Path, reference_year: int = DEFAULT_REFERENCE_YEAR) -> dict:
    df, price_col, n_raw = load_and_clean(path, reference_year)
    X, y = build_features(df, price_col)

    X_trval, X_te, y_trval, y_te = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # 1. choose the model using training data only
    cv_results = cross_validate_models(X_trval, y_trval)
    best = max(cv_results, key=lambda k: cv_results[k]["r2"])

    # 2. fit the winner on training data and score the test set once, in rupees
    final = build_models()[best].fit(X_trval, y_trval)
    actual = np.expm1(y_te).to_numpy()
    pred = np.expm1(final.predict(X_te))
    test = {
        "r2": r2_score(actual, pred),
        "rmse": float(np.sqrt(mean_squared_error(actual, pred))),
    }

    # 3. feature drivers from a Random Forest fitted on the same training data
    rf = build_models()["Random Forest"].fit(X_trval, y_trval)
    drivers = top_drivers(rf, X.columns)

    return {
        "df": df,
        "price_col": price_col,
        "n_raw": n_raw,
        "n_clean": len(df),
        "n_train": len(y_trval),
        "n_test": len(y_te),
        "cv": cv_results,
        "best": best,
        "test": test,
        "actual": actual,
        "best_pred": pred,
        "drivers": drivers,
    }


def format_report(s: dict) -> str:
    lines = [
        f"Raw rows: {s['n_raw']:,}  after cleaning: {s['n_clean']:,}  "
        f"train: {s['n_train']:,}  test: {s['n_test']:,}",
        "",
        f"Model selection ({CV_FOLDS}-fold CV on training data, R2 on log price):",
    ]
    for name, r in s["cv"].items():
        lines.append(f"  {name:20s} R2={r['r2']:.3f} (+/-{r['r2_std']:.3f})")
    t = s["test"]
    top5 = ", ".join(f"{k} ({v:.3f})" for k, v in s["drivers"].head(5).items())
    lines += [
        "",
        f"Selected: {s['best']}",
        f"Test set (scored once, rupees): R2={t['r2']:.3f}  RMSE=Rs {t['rmse']:,.0f}",
        f"Top drivers: {top5}",
    ]
    return "\n".join(lines)


def resume_bullets(s: dict) -> list[str]:
    t = s["test"]
    top3 = list(s["drivers"].index[:3])
    share = s["drivers"].head(3).sum() * 100
    return [
        f"Built a regression model to predict used-bike resale prices, reaching R-squared of {t['r2']:.2f} "
        f"(RMSE Rs {t['rmse']:,.0f}) on a held-out test set of {s['n_test']:,} listings, after cleaning the data, "
        f"engineering features (bike age, brand, km driven, owner type) and selecting the model with "
        f"5-fold cross-validation across {', '.join(s['cv'])}.",
        f"Identified {', '.join(top3)} as the strongest drivers of resale price, measured by Random Forest "
        f"feature importance ({share:.0f}% of total importance).",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Path to the motorcycle CSV")
    parser.add_argument("--reference-year", type=int, default=DEFAULT_REFERENCE_YEAR,
                        help=f"Year used to compute bike age (default: {DEFAULT_REFERENCE_YEAR})")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR,
                        help="Where to write images/ and results.txt")
    args = parser.parse_args()

    summary = run(args.data, args.reference_year)
    report = format_report(summary)
    bullets = resume_bullets(summary)
    console = report + "\n\n===== RESUME BULLETS =====\n" + "\n".join(f"- {b}" for b in bullets)

    print(console)

    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "results.txt").write_text(console + "\n", encoding="utf-8")
    save_figures(summary, args.outdir / "images")
    save_console_image(console, args.outdir / "images" / "console_output.png")


if __name__ == "__main__":
    main()
