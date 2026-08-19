import os

import pandas as pd
import psycopg2
import streamlit as st

# -----------------------------------
# Database settings
# -----------------------------------

DB_HOST = os.getenv("DB_HOST", "34.199.241.113")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD")


# -----------------------------------
# Streamlit setup
# -----------------------------------

st.set_page_config(
    page_title="Flight Delay Model Monitoring",
    layout="wide"
)

st.title("Flight Delay Model Monitoring Dashboard")


# -----------------------------------
# Database connection
# -----------------------------------

def get_connection():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )


# -----------------------------------
# Load prediction logs
# -----------------------------------

connection = get_connection()

query = """
SELECT *
FROM predictions
ORDER BY prediction_timestamp DESC;
"""

data = pd.read_sql(query, connection)

connection.close()


# =================================================
# SECTION 1 - MONITORING METRICS
# =================================================

st.header("1. Monitoring Metrics")

if len(data) > 0:

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Predictions",
            len(data)
        )

    with col2:
        st.metric(
            "Average Prediction Latency",
            f"{data['prediction_latency_ms'].mean():.2f} ms"
        )

    with col3:
        st.metric(
            "Predictions > 15 Minutes",
            int(data["delay_over_15"].sum())
        )

else:
    st.info("No prediction data available yet.")


# =================================================
# SECTION 2 - PREDICTION LATENCY
# =================================================

st.header("2. Prediction Latency Over Time")

if len(data) > 0:

    latency_data = data[
        ["prediction_timestamp", "prediction_latency_ms"]
    ].copy()

    latency_data["prediction_timestamp"] = pd.to_datetime(
        latency_data["prediction_timestamp"]
    )

    latency_data = latency_data.sort_values(
        "prediction_timestamp"
    )

    st.line_chart(
        latency_data,
        x="prediction_timestamp",
        y="prediction_latency_ms"
    )


# =================================================
# SECTION 3 - PREDICTED CLASS DISTRIBUTION
# =================================================

st.header("3. Predicted Class Distribution")

if len(data) > 0:

    class_counts = (
        data["delay_over_15"]
        .value_counts()
        .rename({
            False: "15 Minutes or Less",
            True: "More Than 15 Minutes"
        })
    )

    st.bar_chart(class_counts)

    st.caption(
        "Distribution of predictions classified as more than "
        "15 minutes late versus 15 minutes or less."
    )


# =================================================
# SECTION 4 - TARGET DRIFT
# =================================================

st.header("4. Target Drift Over Time")

if len(data) > 0:

    drift_data = data.copy()

    drift_data["prediction_timestamp"] = pd.to_datetime(
        drift_data["prediction_timestamp"]
    )

    drift_data["prediction_date"] = (
        drift_data["prediction_timestamp"].dt.date
    )

    drift_data = (
        drift_data
        .groupby("prediction_date")["delay_over_15"]
        .mean()
        .reset_index()
    )

    drift_data["percent_predicted_over_15"] = (
        drift_data["delay_over_15"] * 100
    )

    st.line_chart(
        drift_data,
        x="prediction_date",
        y="percent_predicted_over_15"
    )

    st.caption(
        "Tracks changes in the percentage of predictions "
        "classified as more than 15 minutes late."
    )


# =================================================
# SECTION 5 - RECENT PREDICTIONS
# =================================================

st.header("5. Recent Prediction Logs")

if len(data) > 0:

    display_columns = [
        "prediction_id",
        "prediction_timestamp",
        "airline",
        "flight_num",
        "origin_airport_id",
        "dest_airport_id",
        "predicted_delay",
        "delay_over_15",
        "prediction_latency_ms",
        "actual_delay",
        "feedback_correct"
    ]

    st.dataframe(
        data[display_columns],
        use_container_width=True
    )


# =================================================
# SECTION 6 - USER FEEDBACK
# =================================================

st.header("6. Prediction Feedback")

st.write(
    "Enter the actual arrival delay after the flight is complete."
)

prediction_id = st.number_input(
    "Prediction ID",
    min_value=1,
    step=1
)

actual_delay = st.number_input(
    "Actual Arrival Delay (minutes)",
    step=1.0
)

if st.button("Submit Feedback"):

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        UPDATE predictions
        SET actual_delay = %s,
            feedback_correct = (
                delay_over_15 = (%s > 15)
            ),
            feedback_timestamp = CURRENT_TIMESTAMP
        WHERE prediction_id = %s;
        """,
        (
            actual_delay,
            actual_delay,
            prediction_id
        )
    )

    connection.commit()

    rows_updated = cursor.rowcount

    cursor.close()
    connection.close()

    if rows_updated > 0:
        st.success(
            f"Feedback saved for prediction {prediction_id}."
        )
    else:
        st.error(
            "Prediction ID not found."
        )


# =================================================
# SECTION 7 - LIVE MODEL ACCURACY
# =================================================

st.header("7. Live Accuracy From Feedback")

feedback_data = data[
    data["feedback_correct"].notna()
]

if len(feedback_data) > 0:

    live_accuracy = (
        feedback_data["feedback_correct"].mean() * 100
    )

    st.metric(
        "Live Classification Accuracy",
        f"{live_accuracy:.2f}%"
    )

    st.write(
        f"Feedback records available: {len(feedback_data)}"
    )

else:
    st.info(
        "No feedback has been submitted yet."
    )