# Leonard Dixon - Final Project
# FastAPI Backend

import os
import time
import joblib
import pandas as pd
import wandb
import psycopg2

from fastapi import FastAPI
from pydantic import BaseModel


# -----------------------------------
# Create FastAPI app
# -----------------------------------

app = FastAPI(
    title="Flight Delay Prediction API",
    version="1.0"
)


# -----------------------------------
# Database settings
# -----------------------------------

DB_HOST = os.getenv("DB_HOST", "34.199.241.113")

DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD")


# -----------------------------------
# Load model from W&B Model Registry
# -----------------------------------

run = wandb.init(
    project="flight-delay-mlops",
    job_type="inference"
)

artifact = run.use_artifact(
    "wandb-registry-Model Registry/production:v0"
)

model_dir = artifact.download()

model_path = os.path.join(
    model_dir,
    "flight_delay_model.pkl"
)

model = joblib.load(model_path)

run.finish()


# -----------------------------------
# Prediction input
# -----------------------------------

class FlightInput(BaseModel):
    DAY_OF_MONTH: int
    DAY_OF_WEEK: int
    OP_UNIQUE_CARRIER: str
    TAIL_NUM: str
    OP_CARRIER_FL_NUM: int
    ORIGIN_AIRPORT_ID: int
    DEST_AIRPORT_ID: int
    CRS_DEP_TIME: int
    CRS_ARR_TIME: int
    CRS_ELAPSED_TIME: float


# -----------------------------------
# Health check
# -----------------------------------

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True
    }


# -----------------------------------
# Prediction
# -----------------------------------

@app.post("/predict")
def predict(flight: FlightInput):

    start_time = time.time()

    input_data = pd.DataFrame([{
        "DAY_OF_MONTH": flight.DAY_OF_MONTH,
        "DAY_OF_WEEK": flight.DAY_OF_WEEK,
        "OP_UNIQUE_CARRIER": flight.OP_UNIQUE_CARRIER,
        "TAIL_NUM": flight.TAIL_NUM,
        "OP_CARRIER_FL_NUM": flight.OP_CARRIER_FL_NUM,
        "ORIGIN_AIRPORT_ID": flight.ORIGIN_AIRPORT_ID,
        "DEST_AIRPORT_ID": flight.DEST_AIRPORT_ID,
        "CRS_DEP_TIME": flight.CRS_DEP_TIME,
        "CRS_ARR_TIME": flight.CRS_ARR_TIME,
        "CRS_ELAPSED_TIME": flight.CRS_ELAPSED_TIME
    }])

    # Run prediction
    predicted_delay = float(model.predict(input_data)[0])

    delay_over_15 = predicted_delay > 15

    # Calculate prediction latency
    latency_ms = (time.time() - start_time) * 1000

    # -----------------------------------
    # Save prediction to PostgreSQL
    # -----------------------------------

    connection = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )

    cursor = connection.cursor()

    insert_query = """
        INSERT INTO predictions (
            day_of_month,
            day_of_week,
            airline,
            tail_num,
            flight_num,
            origin_airport_id,
            dest_airport_id,
            crs_dep_time,
            crs_arr_time,
            crs_elapsed_time,
            predicted_delay,
            delay_over_15,
            prediction_latency_ms
        )
        VALUES (
            %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s,
            %s, %s, %s
        )
        RETURNING prediction_id;
    """

    cursor.execute(
        insert_query,
        (
            flight.DAY_OF_MONTH,
            flight.DAY_OF_WEEK,
            flight.OP_UNIQUE_CARRIER,
            flight.TAIL_NUM,
            flight.OP_CARRIER_FL_NUM,
            flight.ORIGIN_AIRPORT_ID,
            flight.DEST_AIRPORT_ID,
            flight.CRS_DEP_TIME,
            flight.CRS_ARR_TIME,
            flight.CRS_ELAPSED_TIME,
            predicted_delay,
            delay_over_15,
            latency_ms
        )
    )

    prediction_id = cursor.fetchone()[0]

    connection.commit()

    cursor.close()
    connection.close()

    # -----------------------------------
    # Return prediction
    # -----------------------------------

    return {
        "prediction_id": prediction_id,
        "predicted_arrival_delay_minutes": round(predicted_delay, 2),
        "delay_over_15_minutes": delay_over_15,
        "prediction_latency_ms": round(latency_ms, 2)
    }