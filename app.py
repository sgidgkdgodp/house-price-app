import streamlit as st
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression

st.title("🏠 Boston House Price Prediction")

df = pd.read_csv("Boston.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

X = df.drop("MEDV", axis=1)
y = df["MEDV"]

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, "model.pkl")
model = joblib.load("model.pkl")

st.subheader("Enter House Details")

inputs = []
for col in X.columns:
    val = st.number_input(col, float(X[col].mean()))
    inputs.append(val)

if st.button("Predict Price"):
    price = model.predict([inputs])
    st.success(f"Predicted House Price: {price[0]:.2f}")
