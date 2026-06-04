import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

mlflow.set_tracking_uri("http://127.0.0.1:5000")
# 1. Initialize MLflow tracking to your local server
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Credit Risk Model")

# 2. Load the data
processed_df = pd.read_csv("data/processed/processed_data.csv") 

# 3. Separate features and target, keeping ONLY numeric columns
X = processed_df.drop(columns=["is_high_risk"])
y = processed_df["is_high_risk"]
X = X.select_dtypes(include=['number'])

# --- SANITY CHECK PRINT ---
print("Training features being used:")
print(list(X.columns))
print("-" * 35)

# 4. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================================
# MODEL 1: LOGISTIC REGRESSION (Baseline)
# ==========================================
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

lr_preds = lr_model.predict(X_test)
lr_probs = lr_model.predict_proba(X_test)[:, 1]

print("--- Logistic Regression Metrics ---")
print("Accuracy: ", accuracy_score(y_test, lr_preds))
print("Precision:", precision_score(y_test, lr_preds))
print("Recall:   ", recall_score(y_test, lr_preds))
print("F1:       ", f1_score(y_test, lr_preds))
print("ROC AUC:  ", roc_auc_score(y_test, lr_probs))
print("-" * 35)


# ==========================================
# MODEL 2: RANDOM FOREST (With MLflow Tracking)
# ==========================================
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

rf_preds = rf.predict(X_test)
rf_probs = rf.predict_proba(X_test)[:, 1]

rf_accuracy = accuracy_score(y_test, rf_preds)
rf_precision = precision_score(y_test, rf_preds)
rf_recall = recall_score(y_test, rf_preds)
rf_f1 = f1_score(y_test, rf_preds)
rf_auc = roc_auc_score(y_test, rf_probs)

print("--- Random Forest Metrics ---")
print(f"Accuracy:  {rf_accuracy}")
print(f"Precision: {rf_precision}")
print(f"Recall:    {rf_recall}")
print(f"F1:        {rf_f1}")
print(f"ROC AUC:   {rf_auc}")

# Send to MLflow
with mlflow.start_run(run_name="Random_Forest_Base"):
    mlflow.log_param("model", "RandomForest")
    mlflow.log_metric("accuracy", rf_accuracy)
    mlflow.log_metric("precision", rf_precision)
    mlflow.log_metric("recall", rf_recall)
    mlflow.log_metric("f1", rf_f1)
    mlflow.log_metric("roc_auc", rf_auc)
    mlflow.sklearn.log_model(rf, "random_forest_model")

print("\nAll done! Refresh MLflow at http://127.0.0.1:5000 to view results.")