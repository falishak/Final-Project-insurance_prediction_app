import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Title of the page
st.title("📈 Data Visualization")

# Load the dataset
df = pd.read_csv("insurance.csv")

# Distribution of Insurance Charges by Age
st.subheader("Distribution of Insurance Charges by Age")
fig, ax = plt.subplots()
sns.kdeplot(x=df['age'], y=df['charges'], fill=True, cmap="Blues", shade=True, ax=ax)
st.pyplot(fig)

# Boxplot of Insurance Charges by Sex
st.subheader("Insurance Charges by Sex")
fig, ax = plt.subplots()
sns.boxplot(x='sex', y='charges', data=df, palette="Set2", ax=ax)
st.pyplot(fig)

# Boxplot of Insurance Charges by Smoking Status
st.subheader("Insurance Charges by Smoking Status")
fig, ax = plt.subplots()
sns.boxplot(x='smoker', y='charges', data=df, palette="Set2", ax=ax)
st.pyplot(fig)

# Boxplot of Insurance Charges by Region
st.subheader("Insurance Charges by Region")
fig, ax = plt.subplots()
sns.boxplot(x='region', y='charges', data=df, palette="Set2", ax=ax)
st.pyplot(fig)

# Pairplot of Features
st.subheader("Pairplot of Features")
fig, ax = plt.subplots(figsize=(8, 6))  # Resize the figure to fit
sns.pairplot(df[['age', 'bmi', 'children', 'charges']], hue='children')
st.pyplot(fig)

# Correlation Heatmap of Numerical Features
st.subheader("Correlation Heatmap of Numerical Features")
fig, ax = plt.subplots(figsize=(8, 6))
correlation_matrix = df[['age', 'bmi', 'children', 'charges']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5, ax=ax)
st.pyplot(fig)

# Scatter plot of Insurance Charges vs BMI
st.subheader("Insurance Charges vs BMI")
fig, ax = plt.subplots()
sns.scatterplot(x='bmi', y='charges', data=df, hue='smoker', palette="coolwarm", ax=ax)
st.pyplot(fig)

# Histogram of Charges by Region
st.subheader("Histogram of Charges by Region")
fig, ax = plt.subplots()
sns.histplot(data=df, x='charges', hue='region', multiple="stack", palette="Set3", ax=ax)
st.pyplot(fig)

# Regression plot of Insurance Charges vs Age
st.subheader("Insurance Charges vs Age (With Regression Line)")
fig, ax = plt.subplots()
sns.regplot(x='age', y='charges', data=df, scatter_kws={'color':'blue'}, line_kws={'color':'red'}, ax=ax)
st.pyplot(fig)

# Heatmap of Charges by Age and Smoking Status
st.subheader("Heatmap of Charges by Age and Smoking Status")
fig, ax = plt.subplots(figsize=(8, 6))
df_smoker = df[df['smoker'] == 'yes']
df_non_smoker = df[df['smoker'] == 'no']
heat_data = df.groupby(['age', 'smoker'])['charges'].mean().unstack()
sns.heatmap(heat_data, annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# Violin plot of Charges by Smoking Status
st.subheader("Violin Plot of Charges by Smoking Status")
fig, ax = plt.subplots()
sns.violinplot(x='smoker', y='charges', data=df, palette="Set2", ax=ax)
st.pyplot(fig)

# Feature Importance using Random Forest
from sklearn.ensemble import RandomForestRegressor
X = df.drop('charges', axis=1)
y = df['charges']
X = pd.get_dummies(X, drop_first=True)
model = RandomForestRegressor(n_estimators=100)
model.fit(X, y)
importances = model.feature_importances_
features = X.columns
feature_importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values('Importance', ascending=False)

# Bar plot for feature importance
fig, ax = plt.subplots()
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, ax=ax)
st.pyplot(fig)
