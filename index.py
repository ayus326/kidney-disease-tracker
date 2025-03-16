import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("kidney_disease_balanced.csv")


X = df.drop(columns=["classification"])
y = df["classification"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train_scaled, y_train)


y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)


st.title("Kidney Disease Prediction App")


st.sidebar.header("Enter Patient Details")
user_data = {}
for col in X.columns:
    user_data[col] = st.sidebar.number_input(col, float(X[col].min()), float(X[col].max()), float(X[col].mean()))


input_df = pd.DataFrame([user_data])
input_scaled = scaler.transform(input_df)


if st.sidebar.button("Predict"):
    prediction = model.predict(input_scaled)[0]
    result = "Kidney Disease" if prediction == 1 else "No Kidney Disease"
    st.subheader(f"### Prediction: {result}")
    st.write(f"### Model Accuracy: {accuracy:.2%}")


st.sidebar.write(f"**Model Accuracy:** {accuracy:.2%}")