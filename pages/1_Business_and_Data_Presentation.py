import streamlit as st
import pandas as pd

# Title of the page
st.title("📊 Business and Data Presentation")

# Brief description of the business case
st.header("Business Case")
st.write("""
    The goal of this app is to predict the insurance cost for individuals based on several factors like age, sex, BMI, 
    children, smoking status, and region. We aim to use linear regression to understand how these factors affect the 
    insurance cost and predict future costs for individuals.
""")

# Brief description of the dataset
st.header("Dataset")
st.write("""
    The dataset used for this project contains information about individuals' age, sex, BMI, number of children, 
    smoking status, and region, along with their corresponding medical insurance costs. The data is sourced from a 
    real-world scenario involving medical insurance pricing.
""")

# Load the dataset
df = pd.read_csv("insurance.csv")

# Show basic information about the dataset
st.subheader("Dataset Preview")
st.write(df.head())

# Show some statistics
st.subheader("Dataset Summary Statistics")
st.write(df.describe())
