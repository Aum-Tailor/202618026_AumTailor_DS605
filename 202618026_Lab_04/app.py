import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="NYC Airbnb Price Predictor", page_icon="🏠", layout="centered")

@st.cache_resource
def load_artifacts():
    model = joblib.load("models/airbnb_price_model.pkl")
    options = joblib.load("models/category_options.pkl")
    return model, options

model, options = load_artifacts()

st.title("🏠 NYC Airbnb Nightly Price Predictor")
st.write("Enter listing details to estimate the nightly price.")

col1, col2 = st.columns(2)

with col1:
    neighbourhood_group = st.selectbox("Neighbourhood Group", options["neighbourhood_group"])
    neighbourhood = st.selectbox("Neighbourhood", sorted(options["neighbourhood"]))
    room_type = st.selectbox("Room Type", options["room_type"])
    minimum_nights = st.number_input("Minimum Nights", min_value=1, max_value=30, value=2)

with col2:
    latitude = st.number_input("Latitude", value=40.7128, format="%.6f")
    longitude = st.number_input("Longitude", value=-73.9560, format="%.6f")
    number_of_reviews = st.number_input("Number of Reviews", min_value=0, value=10)
    reviews_per_month = st.number_input("Reviews per Month", min_value=0.0, value=1.0, format="%.2f")

availability_365 = st.slider("Availability (days/year)", 0, 365, 180)
calculated_host_listings_count = st.number_input("Host's Total Listings", min_value=1, value=1)
days_since_last_review = st.number_input("Days Since Last Review", min_value=0, value=30)

if st.button("Predict Price", type="primary"):
    input_df = pd.DataFrame([{
        "neighbourhood_group": neighbourhood_group,
        "neighbourhood": neighbourhood,
        "room_type": room_type,
        "latitude": latitude,
        "longitude": longitude,
        "minimum_nights": minimum_nights,
        "number_of_reviews": number_of_reviews,
        "reviews_per_month": reviews_per_month,
        "calculated_host_listings_count": calculated_host_listings_count,
        "availability_365": availability_365,
        "days_since_last_review": days_since_last_review
    }])

    log_pred = model.predict(input_df)[0]
    price_pred = np.expm1(log_pred)

    st.success(f"### Estimated Price: ${price_pred:.2f} / night")
    st.caption("Estimate based on historical NYC Airbnb data (2019). May not reflect current market conditions.")