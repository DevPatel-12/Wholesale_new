import streamlit as st
import pickle
import numpy as np

# ---------------------------------
# Page Configuration
# ---------------------------------
st.set_page_config(
    page_title="Wholesale Customer Segmentation",
    layout="centered"
)

# ---------------------------------
# Load Model & Scaler
# ---------------------------------
with open("model_and_scaler.pkl", "rb") as file:
    data = pickle.load(file)

model = data["model"]
scaler = data["scaler"]

# ---------------------------------
# App Title
# ---------------------------------
st.title("🛒 Wholesale Customer Segmentation App")
st.write("Predict customer segment based on annual spending behavior.")

# ---------------------------------
# User Inputs
# ---------------------------------
st.subheader("Enter Annual Spending Amounts")

fresh = st.number_input("Fresh Products", min_value=0.0)
milk = st.number_input("Milk Products", min_value=0.0)
grocery = st.number_input("Grocery Products", min_value=0.0)
frozen = st.number_input("Frozen Products", min_value=0.0)
detergents = st.number_input("Detergents & Paper", min_value=0.0)
delicassen = st.number_input("Delicatessen Products", min_value=0.0)

# ---------------------------------
# Prediction
# ---------------------------------
if st.button("Predict Customer Segment"):

    # Combine inputs
    input_data = np.array([[fresh, milk, grocery, frozen, detergents, delicassen]])

    # Apply same preprocessing as training
    input_log = np.log1p(input_data)
    input_scaled = scaler.transform(input_log)

    # Predict cluster
    cluster = model.predict(input_scaled)[0]

    # ---------------------------------
    # Output
    # ---------------------------------
    st.success(f"✅ Customer belongs to **Cluster {cluster}**")

    if cluster == 0:
        st.info("🟢 High-value customers with strong spending across categories.")
    elif cluster == 1:
        st.info("🟡 Retail-focused customers with grocery and essentials dominance.")
    elif cluster == 2:
        st.info("🔵 HORECA-focused customers with high fresh and frozen spending.")
    else:
        st.info("⚪ Low-spending or occasional customers.")
