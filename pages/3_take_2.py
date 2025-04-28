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
sns.kdeplot(x=df['age'], y=df['charges'], fill=True, cmap="Blues", shade=True)
st.pyplot()

# Boxplot of Insurance Charges by Sex
st.subheader("Insurance Charges by Sex")
sns.boxplot(x='sex', y='charges', data=df, palette="Set2")
st.pyplot()

# Boxplot of Insurance Charges by Smoking Status
st.subheader("Insurance Charges by Smoking Status")
sns.boxplot(x='smoker', y='charges', data=df, palette="Set2")
st.pyplot()

# Boxplot of Insurance Charges by Region
st.subheader("Insurance Charges by Region")
sns.boxplot(x='region', y='charges', data=df, palette="Set2")
st.pyplot()

# Pairplot of Features
st.subheader("Pairplot of Features")
sns.pairplot(df[['age', 'bmi', 'children', 'charges']], hue='children')
st.pyplot()

# Correlation Heatmap of Numerical Features
st.subheader("Correlation Heatmap of Numerical Features")
correlation_matrix = df[['age', 'bmi', 'children', 'charges']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
st.pyplot()

# Scatter plot of Insurance Charges vs BMI
st.subheader("Insurance Charges vs BMI")
sns.scatterplot(x='bmi', y='charges', data=df, hue='smoker', palette="coolwarm")
st.pyplot()

# Histogram of Charges by Region
st.subheader("Histogram of Charges by Region")
sns.histplot(data=df, x='charges', hue='region', multiple="stack", palette="Set3")
st.pyplot()

# Regression plot of Insurance Charges vs Age
st.subheader("Insurance Charges vs Age (With Regression Line)")
sns.regplot(x='age', y='charges', data=df, scatter_kws={'color':'blue'}, line_kws={'color':'red'})
st.pyplot()

# Heatmap for Age vs Charges by Smoker Status
st.subheader("Heatmap of Charges by Age and Smoking Status")
df_smoker = df[df['smoker'] == 'yes']
df_non_smoker = df[df['smoker'] == 'no']
heat_data = df.groupby(['age', 'smoker'])['charges'].mean().unstack()
sns.heatmap(heat_data, annot=True, cmap="coolwarm")
st.pyplot()

# Violin plot of Charges by Smoking Status
st.subheader("Violin Plot of Charges by Smoking Status")
sns.violinplot(x='smoker', y='charges', data=df, palette="Set2")
st.pyplot()

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
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
st.pyplot()
