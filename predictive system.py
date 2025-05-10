# -*- coding: utf-8 -*-
"""
Created on Fri May  9 22:29:29 2025

@author: vijay
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Load the correct CSV file
df = pd.read_csv("C:/Users/vijay/OneDrive/Documents/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Preprocessing
df = df.drop("customerID", axis=1)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()

for column in df.select_dtypes(include="object").columns:
    df[column] = LabelEncoder().fit_transform(df[column])

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the trained model as a proper .pkl
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ New model.pkl file created successfully")


import pickle

def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
print("✅ Model loaded successfully")



model = load_model()  # Call the function to get the model object
input_data = np.array([[1, 0, 1, 0, 10, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 2, 50.0, 500.0]])
input_data_reshaped = input_data.reshape(1, -1)

prediction = model.predict(input_data_reshaped)


# Make prediction
prediction = model.predict(input_data_reshaped)
print("Prediction:", prediction)


# Column names must match exactly what the model was trained with
column_names = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges', 'TotalCharges'
]

# Example input values (must match the number and order of columns above)
input_values = [[1, 0, 1, 0, 10, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 2, 50.0, 500.0]]

# Create DataFrame with proper columns
input_df = pd.DataFrame(input_values, columns=column_names)

# Make prediction
prediction = model.predict(input_df)
print("Prediction:", prediction)

if prediction[0] == 1:
    print("The customer is likely to churn")
else:
    print("The customer is not likely to churn")




