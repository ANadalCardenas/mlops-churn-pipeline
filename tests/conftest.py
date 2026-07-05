import matplotlib

matplotlib.use("Agg")  # must run before any test imports pyplot, so plotting works headless in CI

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def churn_dataframe():
    """A small synthetic dataset shaped like the real churn CSV (post load_dataset mapping)."""
    rng = np.random.RandomState(42)
    n = 40

    churn = np.array([0, 1] * (n // 2))
    rng.shuffle(churn)

    return pd.DataFrame({
        "customerID": [f"cust-{i}" for i in range(n)],
        "tenure": rng.uniform(1, 72, size=n),
        "MonthlyCharges": rng.uniform(20, 120, size=n),
        "Contract": rng.choice(["Month-to-month", "One year", "Two year"], size=n),
        "PaymentMethod": rng.choice(["Electronic check", "Mailed check", "Bank transfer"], size=n),
        "Churn": churn,
    })
