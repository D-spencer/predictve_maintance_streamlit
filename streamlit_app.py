# import streamlit as st
# import pandas as pd

# st.title('🤖 Predictive Maintenance APP')

# st.info('This is a machine Learning App')

# df= pd.read_csv('https://github.com/D-spencer/Predictive_Maintenance_ML_Project/blob/main/Predictive_Maintance_Project/data/predictive_maintenance.csv')
# df

# app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np

@st.cache_resource(show_spinner=False)
def load_model():
    # model.joblib must be in repo root or a path inside the repo
    return joblib.load("model.pk1")

model = load_model()

st.title("Predictive Maintenance — Demo")

# Example: user uploads a CSV with feature columns:
uploaded = st.file_uploader("Upload sensor data CSV", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.write("Preview", df.head())
    # make predictions
    preds = model.predict_proba(df)[:, 1]  # probability of positive class
    df["pred_prob"] = preds
    st.write("Predictions", df)
    st.download_button("Download results", df.to_csv(index=False), "preds.csv")
else:
    st.info("Upload a CSV with features matching model inputs.")
