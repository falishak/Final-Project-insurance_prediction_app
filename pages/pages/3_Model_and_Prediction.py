import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Title of the page
st.title("🔮 Model and Prediction")

# Load the dataset
df = pd.read_csv("insurance.csv")

# Preparing the dataset for prediction
df = pd.get_dummies(df, drop_first=True)  # One-hot encode categorical variables

# Features (independent variables) and target (dependent variable)
X = df.drop("charges", axis=1)  # All columns except 'charges'
y = df["charges"]  # Target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Model evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display the model performance metrics
st.subheader("Model Evaluation")
st.write(f"Mean Squared Error: {mse}")
st.write(f"R-squared: {r2}")

# Input fields for user to predict charges
st.subheader("Make a Prediction")
age = st.number_input("Age", min_value=18, max_value=100, value=25)
sex = st.selectbox("Sex", options=["male", "female"])
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=22.0)
children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
smoker = st.selectbox("Smoker", options=["yes", "no"])
region = st.selectbox("Region", options=["southwest", "southeast", "northwest", "northeast"])

# Prepare the input data for prediction
input_data = {
    "age": age,
    "sex_male": 1 if sex == "male" else 0,
    "bmi": bmi,
    "children": children,
    "smoker_yes": 1 if smoker == "yes" else 0,
    "region_southeast": 1 if region == "southeast" else 0,
    "region_southwest": 1 if region == "southwest" else 0,
    "region_northeast": 1 if region == "northeast" else 0,
}

# Convert input data into DataFrame
input_df = pd.DataFrame([input_data])

# Make the prediction
prediction = model.predict(input_df)

# Display the prediction result
st.write(f"Predicted Insurance Charges: ${prediction[0]:.2f}")
