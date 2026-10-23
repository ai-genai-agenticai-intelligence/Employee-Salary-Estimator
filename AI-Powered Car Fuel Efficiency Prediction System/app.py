import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="FuelWise | MPG Predictor",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="collapsed",
)

PROJECT_DIR = Path(__file__).resolve().parent
model = joblib.load(PROJECT_DIR / "Ridge_Lasso_model.pkl")


@st.cache_data
def load_scaling_parameters():
    data = pd.read_csv(PROJECT_DIR / "car-mpg.csv").drop(columns=["car_name"])
    data["origin"] = data["origin"].replace(
        {1: "america", 2: "europe", 3: "asia"}
    )
    data = pd.get_dummies(data, columns=["origin"], dtype=int)
    data = data.replace("?", np.nan).apply(pd.to_numeric, errors="coerce")
    data = data.fillna(data.median())

    features = data.drop(columns=["mpg"])
    return (
        features.mean(),
        features.std(ddof=0),
        data["mpg"].mean(),
        data["mpg"].std(ddof=0),
    )


feature_means, feature_stds, mpg_mean, mpg_std = load_scaling_parameters()
feature_names = list(model.feature_names_in_)

if set(feature_names) != set(feature_means.index):
    raise ValueError("The saved model features do not match car-mpg.csv.")

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(ellipse at 75% 0%, rgba(21, 94, 117, 0.18), transparent 35%),
            #0b1120;
    }
    [data-testid="stHeader"] {
        background: rgba(11, 17, 32, 0.8);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #101b2d 0%, #0d1524 100%);
        border-right: 1px solid rgba(148, 163, 184, 0.12);
    }
    .block-container {
        max-width: 1180px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }
    .hero {
        padding: 2.1rem 2.2rem;
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 22px;
        background: linear-gradient(120deg, rgba(18, 35, 55, 0.96), rgba(13, 24, 39, 0.86));
        box-shadow: 0 18px 50px rgba(0, 0, 0, 0.18);
        margin-bottom: 1.5rem;
    }
    .eyebrow {
        color: #5eead4;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        margin-bottom: 0.65rem;
    }
    .hero h1 {
        color: #f8fafc;
        font-size: clamp(2rem, 4vw, 3rem);
        line-height: 1.1;
        margin: 0;
        letter-spacing: -0.04em;
    }
    .hero p {
        color: #a8b7ca;
        font-size: 1.05rem;
        margin: 0.8rem 0 0;
    }
    .section-label {
        color: #5eead4;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }
    div[data-testid="stForm"] {
        padding: 1.35rem 1.5rem 1.5rem;
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 18px;
        background: rgba(15, 25, 41, 0.78);
    }
    div[data-testid="stFormSubmitButton"] button {
        width: 100%;
        min-height: 3rem;
        color: #06201f;
        background: #5eead4;
        border: 0;
        border-radius: 10px;
        font-weight: 750;
        transition: background 0.2s ease, transform 0.2s ease;
    }
    div[data-testid="stFormSubmitButton"] button:hover {
        color: #06201f;
        background: #99f6e4;
        transform: translateY(-1px);
    }
    div[data-testid="stMetric"] {
        padding: 1rem 1.1rem;
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 14px;
        background: rgba(15, 25, 41, 0.78);
    }
    .info-card {
        height: 100%;
        padding: 1.35rem;
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 18px;
        background: rgba(15, 25, 41, 0.72);
    }
    .info-card h3 {
        color: #f1f5f9;
        font-size: 1.05rem;
        margin: 0.35rem 0 0.65rem;
    }
    .info-card p, .info-card li {
        color: #a8b7ca;
        font-size: 0.92rem;
        line-height: 1.55;
    }
    .info-card ul {
        padding-left: 1.2rem;
        margin-bottom: 0;
    }
    .sidebar-brand {
        color: #f8fafc;
        font-size: 1.25rem;
        font-weight: 750;
        margin: 0.35rem 0 0.2rem;
    }
    .sidebar-copy {
        color: #9aabc0;
        font-size: 0.9rem;
        line-height: 1.5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown('<div class="section-label">Vehicle insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-brand">FuelWise</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-copy">A machine-learning estimate of fuel economy, '
        'built from historical automobile data.</div>',
        unsafe_allow_html=True,
    )
    st.divider()
    st.metric("Dataset average", f"{mpg_mean:.1f} MPG")
    st.caption("Reference average across the vehicles used to prepare the model.")
    st.divider()
    st.markdown("**How to use**")
    st.markdown(
        '<div class="sidebar-copy">Enter the vehicle specifications, choose its '
        'origin, then select <b>Estimate fuel economy</b>.</div>',
        unsafe_allow_html=True,
    )
    st.caption("Model: standardized linear regression")

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">AI-powered vehicle analytics</div>
      <h1>Fuel economy, at a glance.</h1>
      <p>Explore how a car’s specifications translate into an estimated MPG.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

form_column, details_column = st.columns([1.55, 1], gap="large")

with form_column:
    st.markdown('<div class="section-label">Vehicle configuration</div>', unsafe_allow_html=True)
    st.subheader("Tell us about the car")

    with st.form("vehicle_details"):
        first_row = st.columns(2)
        with first_row[0]:
            cyl = st.number_input("Cylinders", min_value=3, max_value=12, value=4)
        with first_row[1]:
            yr = st.number_input(
                "Model year code (70–82)",
                min_value=70,
                max_value=82,
                value=80,
                help="The dataset represents model years as 70 through 82.",
            )

        second_row = st.columns(2)
        with second_row[0]:
            disp = st.number_input("Displacement (cu in)", value=120.0, step=1.0)
        with second_row[1]:
            hp = st.number_input("Horsepower (hp)", value=90.0, step=1.0)

        third_row = st.columns(2)
        with third_row[0]:
            wt = st.number_input("Weight (lb)", value=2500.0, step=50.0)
        with third_row[1]:
            acc = st.number_input("Acceleration (sec)", value=15.0, step=0.5)

        fourth_row = st.columns(2)
        with fourth_row[0]:
            car_type = st.selectbox(
                "Car type code",
                [0, 1],
                help="Encoded as 0 or 1 in the training dataset.",
            )
        with fourth_row[1]:
            origin = st.selectbox("Origin", ["America", "Europe", "Asia"])

        submitted = st.form_submit_button("Estimate fuel economy")

with details_column:
    st.markdown('<div class="section-label">About the estimate</div>', unsafe_allow_html=True)
    st.subheader("A clear, quick forecast")
    st.markdown(
        """
        <div class="info-card">
          <div class="section-label">Your inputs matter</div>
          <h3>Vehicle specifications shape MPG</h3>
          <p>The model considers engine size, power, weight, acceleration, model year,
          body type code, and region of origin.</p>
          <ul>
            <li>Enter measurements using the units shown.</li>
            <li>Choose the car’s manufacturing region.</li>
            <li>Results are estimates, not guaranteed real-world mileage.</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    st.markdown(
        """
        <div class="info-card">
          <div class="section-label">Model details</div>
          <h3>Trained on historical car data</h3>
          <p>Inputs are standardized using the training dataset before prediction.
          The output is converted back to MPG for easier interpretation.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

if submitted:
    origin_features = {
        "origin_america": int(origin == "America"),
        "origin_asia": int(origin == "Asia"),
        "origin_europe": int(origin == "Europe"),
    }
    input_values = {
        "cyl": cyl,
        "disp": disp,
        "hp": hp,
        "wt": wt,
        "acc": acc,
        "yr": yr,
        "car_type": car_type,
        **origin_features,
    }
    input_data = pd.DataFrame(
        [[input_values[name] for name in feature_names]],
        columns=feature_names,
    )
    scaled_input = (input_data - feature_means) / feature_stds
    scaled_prediction = model.predict(scaled_input)[0][0]
    prediction = scaled_prediction * mpg_std + mpg_mean

    st.divider()
    st.markdown('<div class="section-label">Prediction result</div>', unsafe_allow_html=True)
    result_column, conversion_column, average_column = st.columns(3)
    result_column.metric("Estimated fuel economy", f"{prediction:.2f} MPG")
    conversion_column.metric("Equivalent consumption", f"{235.215 / prediction:.2f} L/100 km")
    average_column.metric("Dataset average", f"{mpg_mean:.2f} MPG")
    st.caption("Estimates are based on historical data and may differ from actual driving results.")

st.divider()
footer_left, footer_right = st.columns([2, 1])
with footer_left:
    st.caption("FuelWise · AI-powered car fuel efficiency prediction")
with footer_right:
    st.caption("Developed by Abhishek Sahoo")
