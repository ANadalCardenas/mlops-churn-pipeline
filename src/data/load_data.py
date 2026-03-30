"""
Responsible for reading the versioned CSV dataset from disk and returning a
clean pandas DataFrame. Maps data version strings (e.g. "v1", "v2") to their
corresponding file paths, reads the CSV with pandas, and converts the target
column `Churn` from "Yes"/"No" strings to 1/0 integers.

Public API:
    load_dataset(data_version: str) -> pd.DataFrame
"""
