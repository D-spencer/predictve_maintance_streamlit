import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.title('🤖 Predictive Maintenance APP')
st.info('This is a machine Learning App that predict if a machine is likely to fail or not')

with st.expander('Data'):
  st.write('**Raw Data**')
  @st.cache_data
  def load_data():
    return pd.read_csv("https://raw.githubusercontent.com/D-spencer/Predictive_Maintenance_ML_Project/refs/heads/main/Predictive_Maintance_Project/data/preprocessed_predictive_maintenance.csv")
  df = load_data()
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
  with st.form("input value"):
    st.header('Input Features')
    MachineType = st.selectbox('type' , ('L', 'M' , 'H'))
    Air_Temperature =  st.number_input('air_temp',295.3, 304.2)
    Process_Temperature = st.number_input('process_temperature',308.0, 313.8)
    rotational_speed =  st.number_input('rotational_speed_[rpm]',1168, 2886)
    Torque =  st.number_input('torque_[nm]',3.8 , 76.6)
    tool_wear =  st.number_input('tool_wear_[min]',0, 253)
    
    submit = st.form_submit_button('Predict')
    
    input_df = pd.DataFrame([{
       'type': MachineType,
        'air_temperature': Air_Temperature,
        'process_temp': Process_Temperature,
        'rotational_speed_[rpm]': rotational_speed,
        'torque_[nm]': Torque,
        'tool_wear_[min]': tool_wear
   }])
    # input_df = pd.DataFrame(data, index =[0])
    # input_pred = pd.concat([input_df, X_raw], axis=0)

with st.expander('Input features'):
  st.write('**Inputed Values**')
  input_df
  # st.write('**Combined penguins data**')
  # input_pred
  

def load_model():
  return joblib.load('predictive_maintance.Pk1')

model = load_model()
model_pred = model.predict(input_df)[0]
def fail(pred):
  if pred == 1:
    return ' this machine is likely to fail'
  else : 
    return 'This Machine is Okay'
msg = fail(model_pred)
st.success(f"The Predicted result is :  {model_pred} , {msg}" )

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

             

  
                                     
