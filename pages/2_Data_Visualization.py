import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Title of the page
st.title("📈 Data Visualization")

# Load the dataset
df = pd.read_csv("insurance.csv")

# Visualize the relationship between different features
st.subheader("Distribution of Age")
sns.histplot(df['age'], kde=True)
st.pyplot()

st.subheader("Insurance Charges by Age and Sex")
sns.boxplot(x='sex', y='charges', data=df)
st.pyplot()

st.subheader("Insurance Charges by Smoking Status")
sns.boxplot(x='smoker', y='charges', data=df)
st.pyplot()

st.subheader("Insurance Charges by Region")
sns.boxplot(x='region', y='charges', data=df)
st.pyplot()

st.subheader("Correlation Heatmap")
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
st.pyplot()
