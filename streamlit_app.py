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
  st.header('Input Features')
  MachineType = st.selectbox('type' , ('L', 'M' , 'H'))
  Torque =  st.slider('torque_[nm]',3.8 , 42.8, 18.4)
  rotational_speed =  st.slider('rotational_speed_[rpm]',1168, 2886, 2000)
  tool_wear =  st.slider('tool_wear_[min]',0, 253, 124)
  Air_Temperature =  st.slider('air_temp',295.3, 304.2, 299.71)
  Process_Temperature = st.slider('process_temperatuew',308.0, 313.8, 310.76)
  
  data = {'type': MachineType,
          'torque_[nm]': Torque,
          'rotational_speed_[rpm]': rotational_speed,
          'tool_wear_[min]': tool_wear,
          'air_temperature': Air_Temperature,
          'process_temp': Process_Temperature}
  input_df = pd.DataFrame(data, index=[0])
  input_pred = pd.concat([input_df, X_raw], axis=0)


input_pred



  
                                     
