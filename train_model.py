# Leonard Dixon - Final Project

# import modules and libraries
import joblib
import pandas as pd
import wandb
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# -----------------------------
# Start W&B experiment
# -----------------------------

run = wandb.init(
    project="flight-delay-mlops",
    name="linear-regression-v2",
    config={
        "model": "LinearRegression",
        "data_version": "flightdata_MLReady_2026_v1",
        "train_period": "Jan-May 2026",
        "test_period": "June 2026",
        "target": "ARR_DELAY",
        "features": [
            "DAY_OF_MONTH",
            "DAY_OF_WEEK",
            "OP_UNIQUE_CARRIER",
            "ORIGIN_AIRPORT_ID",
            "DEST_AIRPORT_ID",
            "CRS_DEP_TIME",
            "CRS_ARR_TIME",
            "CRS_ELAPSED_TIME"
        ],
    }
)

# -----------------------------
# Load data
# -----------------------------

rawData = pd.read_csv("Data/flightdata_MLReady_2026.csv")

# -----------------------------
# Fix data types
# -----------------------------

categorical_cols = [
    "OP_UNIQUE_CARRIER",
    "ORIGIN_AIRPORT_ID",
    "DEST_AIRPORT_ID"
]

for col in categorical_cols:
    rawData[col] = rawData[col].astype("category")

rawData["FL_DATE"] = pd.to_datetime(rawData["FL_DATE"])

# -----------------------------
# Train/Test split
# Train = Jan-May
# Test = June
# -----------------------------

trainData = rawData[rawData["FL_DATE"].dt.month <= 5].copy()
testData = rawData[rawData["FL_DATE"].dt.month == 6].copy()

X_train = trainData.drop(columns=["ARR_DELAY", "FL_DATE","TAIL_NUM","OP_CARRIER_FL_NUM"])
y_train = trainData["ARR_DELAY"]

X_test = testData.drop(columns=["ARR_DELAY", "FL_DATE","TAIL_NUM","OP_CARRIER_FL_NUM"])
y_test = testData["ARR_DELAY"]

# -----------------------------
# Configure preprocessor
# -----------------------------

numeric_cols = [
    "DAY_OF_MONTH",
    "DAY_OF_WEEK",
    "CRS_DEP_TIME",
    "CRS_ARR_TIME",
    "CRS_ELAPSED_TIME"
]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ("num", StandardScaler(), numeric_cols)
])

# -----------------------------
# Configure model pipeline
# -----------------------------

linear_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

# -----------------------------
# Train model
# -----------------------------

linear_model.fit(X_train, y_train)

# -----------------------------
# Test model
# -----------------------------

y_pred_linear = linear_model.predict(X_test)

# -----------------------------
# Regression performance metrics
# -----------------------------

mae = mean_absolute_error(y_test, y_pred_linear)
rmse = mean_squared_error(y_test, y_pred_linear) ** 0.5
r2 = r2_score(y_test, y_pred_linear)

print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)

# -----------------------------
# >15-minute evaluation
# -----------------------------

y_test_binary = (y_test > 15).astype(int)
y_pred_binary = (y_pred_linear > 15).astype(int)

precision = precision_score(y_test_binary, y_pred_binary)
recall = recall_score(y_test_binary, y_pred_binary)
f1 = f1_score(y_test_binary, y_pred_binary)

print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)

# -----------------------------
# Log metrics to W&B
# -----------------------------

run.log({
    "MAE": mae,
    "RMSE": rmse,
    "R2": r2,
    "Precision_15min": precision,
    "Recall_15min": recall,
    "F1_15min": f1
})

# -----------------------------
# Save model locally
# -----------------------------

model_filename = "flight_delay_model.pkl"

joblib.dump(linear_model, model_filename)

# -----------------------------
# Upload model to W&B
# -----------------------------

model_artifact = wandb.Artifact(
    name="flight-delay-model",
    type="model"
)

model_artifact.add_file(model_filename)

run.log_artifact(model_artifact)

# -----------------------------
# Finish W&B run
# -----------------------------

run.finish()