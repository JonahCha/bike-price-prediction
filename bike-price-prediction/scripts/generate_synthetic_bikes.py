"""Generate a SYNTHETIC used-motorcycle dataset with the same schema as the CarDekho
'BIKE DETAILS.csv' file. Use it to run the pipeline end to end without the real file.

For real results, replace data/BIKE_DETAILS_synthetic.csv with the original
BIKE DETAILS.csv (Kaggle: nehalbirla motorcycle dataset) and rerun the pipeline.

Usage:
    python scripts/generate_synthetic_bikes.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("data/BIKE_DETAILS_synthetic.csv")
N_ROWS = 1061
SEED = 42
REF_YEAR = 2024

# (model name, typical new ex-showroom price in Rs)
MODELS = [
    ("Honda Activa 125", 80000), ("Honda Dio", 75000), ("Honda CB Shine", 80000),
    ("Honda Unicorn", 105000), ("Bajaj Pulsar 150", 110000), ("Bajaj Pulsar 220F", 130000),
    ("Bajaj Avenger 220", 125000), ("Hero Splendor Plus", 75000), ("Hero Passion Pro", 78000),
    ("Hero HF Deluxe", 65000), ("TVS Jupiter", 75000), ("TVS Apache RTR 160", 110000),
    ("Yamaha FZ-S", 120000), ("Yamaha R15 V3", 175000), ("Suzuki Access 125", 80000),
    ("Royal Enfield Classic 350", 195000), ("Royal Enfield Bullet 350", 180000),
    ("KTM Duke 200", 190000), ("Kawasaki Ninja 300", 320000),
]
OWNER_FACTOR = {"1st owner": 1.0, "2nd owner": 0.86, "3rd owner": 0.75, "4th owner": 0.66}
SELLER_FACTOR = {"Individual": 1.0, "Dealer": 1.06}


def generate(n: int = N_ROWS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(MODELS), n)
    names = [MODELS[i][0] for i in idx]
    new_price = np.array([MODELS[i][1] for i in idx], dtype=float)

    year = rng.integers(2000, REF_YEAR + 1, n)
    age = REF_YEAR - year
    km = np.clip(rng.gamma(2.0, 9000, n) + age * 1200, 300, 120000).astype(int)
    owner = rng.choice(list(OWNER_FACTOR), n, p=[0.6, 0.27, 0.10, 0.03])
    seller = rng.choice(list(SELLER_FACTOR), n, p=[0.85, 0.15])

    value = (
        new_price
        * np.exp(-0.085 * age)
        * np.exp(-km / 110000)
        * np.array([OWNER_FACTOR[o] for o in owner])
        * np.array([SELLER_FACTOR[s] for s in seller])
        * np.exp(rng.normal(0, 0.08, n))
    )
    price = np.maximum(1000, np.round(value / 500) * 500)

    showroom = np.round(new_price * np.exp(rng.normal(0, 0.03, n)) / 100) * 100
    showroom[rng.uniform(size=n) < 0.35] = np.nan  # the real file has many missing values

    df = pd.DataFrame({
        "name": names,
        "selling_price": price.astype(int),
        "year": year,
        "seller_type": seller,
        "owner": owner,
        "km_driven": km,
        "ex_showroom_price": showroom,
    })
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    generate().to_csv(OUT, index=False)
    print(f"Wrote {OUT}")
