import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

# Title
st.title("Logistic Regression Prediction App")

# Load dataset directly
df = pd.read_csv("student_mental_health_burnout.csv")
# Remove missing values
df = df.dropna()

# Encode categorical columns
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col].astype(str))

# Target column
target = st.selectbox("Select Target Column", df.columns)

# Features and target
X = df.drop(target, axis=1)
y = df[target]

# Scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
