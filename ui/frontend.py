import streamlit as st
import requests

API_URL = "http://3.228.10.94:8000/predict"

st.title("Flight Delay Prediction")

day_of_month = st.number_input("Day of Month", 1, 31, 1)
day_of_week = st.number_input("Day of Week", 1, 7, 4)

airline = st.text_input("Airline", "AA")
tail_num = st.text_input("Tail Number", "N101NN")
flight_num = st.number_input("Flight Number", value=1480)

origin_airport_id = st.number_input("Origin Airport ID", value=12892)
dest_airport_id = st.number_input("Destination Airport ID", value=10721)

crs_dep_time = st.number_input("Scheduled Departure Time", value=1358)
crs_arr_time = st.number_input("Scheduled Arrival Time", value=2229)
crs_elapsed_time = st.number_input("Scheduled Elapsed Time", value=331)

if st.button("Predict"):

    payload = {
        "DAY_OF_MONTH": day_of_month,
        "DAY_OF_WEEK": day_of_week,
        "OP_UNIQUE_CARRIER": airline,
        "TAIL_NUM": tail_num,
        "OP_CARRIER_FL_NUM": flight_num,
        "ORIGIN_AIRPORT_ID": origin_airport_id,
        "DEST_AIRPORT_ID": dest_airport_id,
        "CRS_DEP_TIME": crs_dep_time,
        "CRS_ARR_TIME": crs_arr_time,
        "CRS_ELAPSED_TIME": crs_elapsed_time
    }

    response = requests.post(API_URL, json=payload)

    if response.status_code == 200:

        result = response.json()

        st.success(
            f"Predicted Arrival Delay: "
            f"{result['predicted_arrival_delay_minutes']} minutes"
        )

        st.write(
            "More than 15 minutes late:",
            result["delay_over_15_minutes"]
        )

        st.write(
            "Prediction ID:",
            result["prediction_id"]
        )

    else:
        st.error("Prediction failed.")