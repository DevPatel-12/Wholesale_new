import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# ---------------------------------
# Page Config
# ---------------------------------
st.set_page_config(
    page_title="Wholesale Customer Segmentation",
    layout="centered"
)

# ---------------------------------
# Load Pickle File
# ---------------------------------
with open("model_and_scaler.pkl", "rb") as file:
    data = pickle.load(file)

model = data.get("model")
scaler = data.get("scaler")

# ---------------------------------
# App UI
# ---------------------------------
st.title("🛒 Wholesale Customer Segmentation App")
st.write("Predict customer segment based on annual spending behavior.")

st.subheader("Enter Annual Spending Amounts")

fresh = st.number_input("Fresh Products", min_value=0.0)
milk = st.number_input("Milk Products", min_value=0.0)
grocery = st.number_input("Grocery Products", min_value=0.0)
frozen = st.number_input("Frozen Products", min_value=0.0)
detergents = st.number_input("Detergents & Paper", min_value=0.0)
delicassen = st.number_input("Delicatessen Products", min_value=0.0)

# ---------------------------------
# Prediction Logic
# ---------------------------------
if st.button("Predict Customer Segment"):

    input_data = np.array([[fresh, milk, grocery, frozen, detergents, delicassen]])

    # Log t
