import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


def create_features(df):

    # Aggregate customer features
    customer_features = df.groupby("CustomerId").agg(
        Total_Transaction_Amount=("Amount", "sum"),
        Avg_Transaction_Amount=("Amount", "mean"),
        Transaction_Count=("Amount", "count"),
        Std_Transaction_Amount=("Amount", "std")
    ).reset_index()

    df = df.merge(customer_features, on="CustomerId")

    # Date features
    df["TransactionStartTime"] = pd.to_datetime(
        df["TransactionStartTime"]
    )

    df["TransactionHour"] = df["TransactionStartTime"].dt.hour
    df["TransactionDay"] = df["TransactionStartTime"].dt.day
    df["TransactionMonth"] = df["TransactionStartTime"].dt.month
    df["TransactionYear"] = df["TransactionStartTime"].dt.year

    return df


def build_pipeline(df):

    categorical = df.select_dtypes(
        include="object"
    ).columns.tolist()

    numerical = df.select_dtypes(
        exclude="object"
    ).columns.tolist()

    preprocessor = ColumnTransformer([
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            numerical
        ),
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]),
            categorical
        )
    ])

    return preprocessor