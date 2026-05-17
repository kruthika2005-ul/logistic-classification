import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

# Title
st.title("Logistic Regression Prediction App")

# Load dataset directly
df = pd.read_csv("student_mental_health_burnout.csv")

# Display dataset
#st.subheader("Dataset")
#st.write(df.head())

# Encode categorical columns
le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

# Target column
target = st.selectbox("Select Target Column", df.columns) # Change target column name if needed

# Features and target
X = df.drop(target, axis=1)
y = df[target]

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)

st.subheader("Model Accuracy")
st.write(f"Accuracy: {accuracy:.2f}")

# Prediction section
st.subheader("Prediction")

inputs = []

for col in X.columns:
    value = st.number_input(f"Enter {col}")
    inputs.append(value)

if st.button("Predict"):

    input_data = scaler.transform([inputs])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("Prediction: Positive")
    else:
        st.error("Prediction: Negative")