import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.title('🤖 Predictive Maintenance APP')
st.info('This is a machine Learning App')

with st.expander('Data'):
  st.write('**Raw Data**')
  df = pd.read_csv("https://raw.githubusercontent.com/D-spencer/Predictive_Maintenance_ML_Project/refs/heads/main/Predictive_Maintance_Project/data/preprocessed_predictive_maintenance.csv")
  df



