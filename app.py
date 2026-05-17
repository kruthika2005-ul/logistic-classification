import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

# Title
st.title("Logistic Classification App")

# Upload Dataset
uploaded_file = st.file_uploader("student_mental_health_burnout.csv", type=["csv"])

if uploaded_file is not None:

    # Read dataset
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset")
    st.write(df.head())

    # Remove missing values
    df = df.dropna()

    # Encode categorical columns
    le = LabelEncoder()

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col])

    # Select target column
    target = st.selectbox("Select Target Column", df.columns)

    # Features and target
    X = df.drop(target, axis=1)
    y = df[target]

    # Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42
    )

    # Train Model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Accuracy
    accuracy = model.score(X_test, y_test)

    st.subheader("Model Accuracy")
    st.write(f"Accuracy: {accuracy:.2f}")

    st.success("Logistic Classification Completed")