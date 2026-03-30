# 📁 Data Versions Overview

### v1. Baseline Dataset

* Original [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) dataset
* Represents initial system deployment
* Used to train the first production model

### v2. Business Growth and Distribution Drift

* Derived from v1
* Simulates:
  * Increased customer base
  * Higher monthly charges
  * Longer customer tenure
  * Shift toward long-term contracts
* Represents real-world data evolution after deployment
* Used to demonstrate monitoring, drift detection, and retraining


## Data generation

v1/churn_raw.csv is the original Telco Customer Churn dataset downloaded from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). v2/churn.csv is generated following the rules above using the python script `generate_v2.py` located in this directory.

To re-generate the data:

1. Remove the contents of v2 (if any).
2. Execute the script:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 generate_v2.py
```