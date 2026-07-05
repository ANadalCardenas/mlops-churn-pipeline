import pandas as pd
import pytest

from src.data import load_data


def test_load_dataset_maps_churn_to_binary(tmp_path, monkeypatch):
    csv_path = tmp_path / "churn.csv"
    pd.DataFrame({
        "customerID": ["1", "2", "3"],
        "tenure": [1, 2, 3],
        "Churn": ["Yes", "No", "Yes"],
    }).to_csv(csv_path, index=False)

    monkeypatch.setitem(load_data.DATA_PATHS, "test", str(csv_path))

    df = load_data.load_dataset("test")

    assert df["Churn"].tolist() == [1, 0, 1]
    assert df["Churn"].dtype.kind in "iu"  # integer dtype, not the original Yes/No strings


def test_load_dataset_unknown_version_raises_key_error():
    with pytest.raises(KeyError):
        load_data.load_dataset("does-not-exist")
