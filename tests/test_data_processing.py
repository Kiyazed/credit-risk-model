import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

import pandas as pd

from src.data_processing import create_features


def test_transaction_hour_created():

    data = pd.DataFrame({
        "CustomerId": [1],
        "Amount": [100],
        "TransactionStartTime": [
            "2025-01-01 10:30:00"
        ]
    })

    result = create_features(data)

    assert "TransactionHour" in result.columns


def test_transaction_month_created():

    data = pd.DataFrame({
        "CustomerId": [1],
        "Amount": [100],
        "TransactionStartTime": [
            "2025-01-01 10:30:00"
        ]
    })

    result = create_features(data)

    assert result["TransactionMonth"].iloc[0] == 1

def test_transaction_count():

     data = pd.DataFrame({
        "CustomerId": [1, 1, 1],
        "Amount": [100, 200, 300],
        "TransactionStartTime": [
            "2025-01-01",
            "2025-01-02",
            "2025-01-03"
        ]
     })

     result = create_features(data)

     assert result["Transaction_Count"].iloc[0] == 3