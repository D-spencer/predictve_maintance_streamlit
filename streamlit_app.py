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

  st.write('***X***')
  X_raw = df.drop('target' , axis =1)
  X_raw

  st.write('***y***')
  y_raw =  df.target
  y_raw 

with st.expander('Data Visulization'):
  st.bar_chart(df['target'].value_counts()  , x_label='Distribution of Target column')

with st.sidebar:
  with st.form():
  st.header('Input Features')
  MachineType = st.selectbox('type' , ('L', 'M' , 'H'))
  Torque =  st.number_input('torque_[nm]',3.8 , 76.6)
  rotational_speed =  st.number_input('rotational_speed_[rpm]',1168, 2886)
  tool_wear =  st.number_input('tool_wear_[min]',0, 253)
  Air_Temperature =  st.number_input('air_temp',295.3, 304.2)
  Process_Temperature = st.number_input('process_temperatuew',308.0, 313.8)
  submit = st.form_submit_button('Predict')
  
  data = {'type': MachineType,
          'torque_[nm]': Torque,
          'rotational_speed_[rpm]': rotational_speed,
          'tool_wear_[min]': tool_wear,
          'air_temperature': Air_Temperature,
          'process_temp': Process_Temperature}
  input_df = pd.DataFrame(data, index=[0])
  input_pred = pd.concat([input_df, X_raw], axis=0)

with st.expander('Input features'):
  st.write('**Input penguin**')
  input_df
  st.write('**Combined penguins data**')
  input_pred

model = joblib.load('predictive_maintance.Pk1')
st.success(f"The Predicted result is :  {model.predict(input_df)[0]}")

proba = model.predict_proba(input_df)
failure_prob = float(proba[0][1])
no_failure_prob = float(proba[0][0])

col1, col2 = st.columns(2)
with col1: 
  st.metric('Probability of No failure' ,  f"{no_failure_prob:.2%}")
  st.progress(no_failure_prob)

with col2: 
  st.metric('Probability of failure',  f"{failure_prob:.2%}")
  st.progress(failure_prob)


             

  
                                     
