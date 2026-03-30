import pandas as pd
import numpy as np
from pathlib import Path

# -----------------------------
# Configuration
# -----------------------------
DATA_DIR = Path(__file__).resolve().parent

V1_PATH = DATA_DIR / "v1" / "churn_raw.csv"
V2_DIR = DATA_DIR / "v2"

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# -----------------------------
# Utility
# -----------------------------
def ensure_dirs():
    V2_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# v2: More data & Drifted distributions
# -----------------------------
def create_v2(df: pd.DataFrame):
    print("Generating v2...")

    # CHANGE 0: Increase volume by 40%
    print("Increased volume...")
    additional_rows = df.sample(
        frac=0.4,
        replace=False,
        random_state=RANDOM_SEED
    )
    increased_volume_df = pd.concat([df, additional_rows], ignore_index=True)
    
    print("Adding distribution drift...")
    v2_df = increased_volume_df.copy()

    # CHANGE 1: MonthlyCharges drift
    print("Applying MonthlyCharges drift...")
    long_tenure_mask = v2_df["tenure"] > 12
    v2_df.loc[long_tenure_mask, "MonthlyCharges"] *= 1.15
    v2_df["MonthlyCharges"] = v2_df["MonthlyCharges"].clip(upper=120)

    # CHANGE 2: Tenure distribution drift
    print("Applying Tenure distribution drift...")
    short_tenure = v2_df[v2_df["tenure"] < 6]
    long_tenure = v2_df[v2_df["tenure"] > 24]

    short_tenure_kept = short_tenure.sample(
        frac=0.5,
        random_state=RANDOM_SEED
    )

    long_tenure_duplicated = long_tenure.sample(
        frac=0.5,
        replace=True,
        random_state=RANDOM_SEED
    )

    v2_df = pd.concat(
        [
            v2_df[v2_df["tenure"] >= 6],
            short_tenure_kept,
            long_tenure_duplicated,
        ],
        ignore_index=True,
    )

    # CHANGE 3: Contract distribution drift
    print("Applying Contract distribution drift...")
    contract_probs = {
        "Month-to-month": 0.30,
        "One year": 0.35,
        "Two year": 0.35,
    }

    v2_df["Contract"] = np.random.choice(
        list(contract_probs.keys()),
        size=len(v2_df),
        p=list(contract_probs.values()),
    )

    # CHANGE 4: InternetService drift
    print("Applying InternetService distribution drift...")
    def shift_internet_service(value):
        if value == "DSL":
            return np.random.choice(
                ["DSL", "Fiber optic"],
                p=[0.3, 0.7]
            )
        if value == "Fiber optic":
            return "Fiber optic"
        return value  # "No"

    v2_df["InternetService"] = v2_df["InternetService"].apply(
        shift_internet_service
    )

    v2_df.to_csv(V2_DIR / "churn.csv", index=False)

# -----------------------------
# Main
# -----------------------------
def main():
    ensure_dirs()

    print("Loading v1 dataset...")
    df_v1 = pd.read_csv(V1_PATH)
    
    create_v2(df_v1)

    print("\nData generation completed:")
    print(" - data/v2/churn.csv")

if __name__ == "__main__":
    main()
