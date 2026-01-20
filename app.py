import streamlit as st
import pickle
import numpy as np

# -----------------------------
# Load the trained pickle file
# -----------------------------
with open("processed_wholesale_customers.pkl", "rb") as file:
    data = pickle.load(file)

# If model and scaler are stored together (recommended)
if isinstance(data, dict):
    model = data.get("model")
    scaler = data.get("scaler")
else:
    # If only model is stored
    model = data
    scaler = None

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Wholesale Customer Segmentation", layout="centered")

st.title("🛒 Wholesale Customer Segmentation App")
st.write("Enter annual spending amounts to predict the customer segment.")

# -----------------------------
# User Inputs
# -----------------------------
fresh = st.number_input("Fresh Products Spend", min_value=0.0)
milk = st.number_input("Milk Products Spend", min_value=0.0)
grocery = st.number_input("Grocery Products Spend", min_value=0.0)
frozen = st.number_input("Frozen Products Spend", min_value=0.0)
detergents = st.number_input("Detergents & Paper Spend", min_value=0.0)
delicassen = st.number_input("Delicatessen Spend", min_value=0.0)

# -----------------------------
# Prediction Logic
# -----------------------------
if st.button("Predict Customer Segment"):

    input_data = np.array([[fresh, milk, grocery, frozen, detergents, delicassen]])

    # Apply log transformation (same as training)
    input_log = np.log1p(input_data)

    # Apply scaling if scaler exists
    if scaler is not None:
        input_scaled = scaler.transform(input_log)
    else:
        input_scaled = input_log

    # Predict cluster
    cluster = model.predict(input_scaled)[0]

    # -----------------------------
    # Output
    # -----------------------------
    st.success(f"✅ This customer belongs to **Cluster {cluster}**")

    # Optional business interpretation
    if cluster == 0:
        st.info("🟢 High-value customers with strong overall spending.")
    elif cluster == 1:
        st.info("🟡 Retail-focused customers with grocery and essentials dominance.")
    elif cluster == 2:
        st.info("🔵 HORECA-focused customers with high fresh and frozen spending.")
    else:
        st.info("⚪ Low-spending or occasional buyers.")