# -*- coding: utf-8 -*-
"""
Created on Fri May  9 23:00:07 2025

@author: vijay
"""

import numpy as np
import pickle
import streamlit as st

#loading the saved data

def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
print("✅ Model loaded successfully")


#creating a function for prediction

def churn_prediction(input_data):
    
    input_data_ar_numpy_array=np.asarray(input_data)
    input_data = np.array([[1, 0, 1, 0, 10, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 2, 50.0, 500.0]])
    input_data_reshaped = input_data.reshape(1, -1)
    # Make prediction
    prediction = model.predict(input_data_reshaped)
    print("Prediction:", prediction)
    

    if prediction[0] == 1:
        return 'The customer is likely to churn'
    else:
        return 'The customer is not likely to churn'
    
    
def main():
    
    
    #giving a title
    st.title("Churn Prediction Web App")
    
    
    #getting the input data from the user
   
    
    gender = st.text_input('Gender')
    SeniorCitizen = st.text_input("SeniorCitizen")
    Partner = st.text_input('Do you have Partner')
    Dependents = st.text_input('Dependent(yes or no)')
    tenure = st.text_input('Tenure value')
    PhoneService = st.text_input('Phone Service')
    MultipleLines = st.text_input('Multiplelines')
    InternetService = st.text_input('Internet Service')
    OnlineSecurity = st.text_input('Online Security')
    OnlineBackup = st.text_input('Online Bachup')
    DeviceProtection = st.text_input('Device Protection')
    TechSupport = st.text_input('Tech Support')
    StreamingTV = st.text_input('Streaming TV')
    StreamingMovies = st.text_input('Streaming Movies')
    Contract = st.text_input('Contract')
    PaperlessBilling = st.text_input('Paperless Billing')
    PaymentMethod = st.text_input('Payment Method')
    MonthlyCharges = st.text_input('Monthly Charge')
    TotalCharges = st.text_input('Total charge')
    
    
    #code for Prediction
    churn_predict = ''
    
    #creating a button for prediction
    
    if st.button('Customer churn Test Result'):
        churn_predict = churn_prediction([gender, SeniorCitizen, Partner, Dependents, tenure,
        PhoneService, MultipleLines, InternetService, OnlineSecurity,
        OnlineBackup, DeviceProtection, TechSupport, StreamingTV,
        StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
        MonthlyCharges, TotalCharges])
    
    st.success(churn_predict)
    
    
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    