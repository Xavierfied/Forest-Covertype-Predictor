import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ______________________
import streamlit as st

# --- Forest theme custom CSS ---
st.markdown("""
    <style>
    /* Buttons */
    div.stButton > button:first-child {
        background-color: #2e7d32;
        color:white;
        border-radius: 12px;
        padding:0.6em 1.2em;
        font-weight:bold;
        transition: 0.3s;
    }
    div.stButton > button:first-child:hover {
        background-color: #1b5e20;
        color: #f1f8e9;
    }

    /* Headers */
    h1, h2, h3 {
        color: #1b5e20 !important;
        font-family: 'Trebuchet MS', sans-serif;
    }

    /* Sidebar header */
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2 {
        color: #2e7d32 !important;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------
# ----------------------
# Load pipeline (model + preprocessor)
# ----------------------
pipe = joblib.load("Forest_Cover_Type_predictor.joblib")  # Your saved pipeline

# ----------------------
# Cover type mapping
# ----------------------
cover_type_map = {
    1: "Spruce/Fir",
    2: "Lodgepole Pine",
    3: "Ponderosa Pine",
    4: "Cottonwood/Willow",
    5: "Aspen",
    6: "Douglas-fir",
    7: "Krummholz"
}

# ----------------------
# Streamlit App
# ----------------------
st.sidebar.header("🍃 About")
st.sidebar.write("This app predicts the **Forest CoverType** based on environmental features. \
It uses a trained ML pipeline with preprocessing + model stored in Joblib.")

# --------------------

st.set_page_config(page_title="Forest CoverType Predictor", page_icon="🌲", layout="centered")

st.title("🌲 Forest CoverType Prediction App")
st.markdown("""
This app predicts the **forest cover type** based on cartographic variables. 
Upload your data or input values manually.
""")

# Sidebar
st.sidebar.header("⚙️ Input Options")
input_mode = st.sidebar.radio("Choose input method:", ["Manual Input", "Upload CSV"])

# Feature groups
numeric_features = [
    'Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
    'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
    'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
    'Horizontal_Distance_To_Fire_Points'
]

wilderness_features = [
    'Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4'
]

soil_features = [
    'Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4', 'Soil_Type5',
    'Soil_Type6', 'Soil_Type7', 'Soil_Type8', 'Soil_Type9', 'Soil_Type10',
    'Soil_Type11', 'Soil_Type12', 'Soil_Type13', 'Soil_Type14',
    'Soil_Type15', 'Soil_Type16', 'Soil_Type17', 'Soil_Type18',
    'Soil_Type19', 'Soil_Type20', 'Soil_Type21', 'Soil_Type22',
    'Soil_Type23', 'Soil_Type24', 'Soil_Type25', 'Soil_Type26',
    'Soil_Type27', 'Soil_Type28', 'Soil_Type29', 'Soil_Type30',
    'Soil_Type31', 'Soil_Type32', 'Soil_Type33', 'Soil_Type34',
    'Soil_Type35', 'Soil_Type36', 'Soil_Type37', 'Soil_Type38',
    'Soil_Type39', 'Soil_Type40'
]

# ----------------------
# Manual Input Mode
# ----------------------
def user_input_features():
    st.subheader("Enter Features Manually")
    data = {}

    with st.expander("📏 Numeric Features", expanded=True):
        for feature in numeric_features:
            data[feature] = st.number_input(f"{feature}", min_value=0, step=1)

    with st.expander("🌲 Wilderness Areas", expanded=False):
        for feature in wilderness_features:
            data[feature] = st.selectbox(f"{feature}", [0, 1], index=0)

    with st.expander("🪨 Soil Types", expanded=False):
        for feature in soil_features:
            data[feature] = st.selectbox(f"{feature}", [0, 1], index=0)

    return pd.DataFrame([data])

# ----------------------
# Upload CSV Mode
# ----------------------
def upload_input():
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        return df
    return None

# ----------------------
# Collect input
# ----------------------
if input_mode == "Manual Input":
    input_df = user_input_features()
else:
    input_df = upload_input()

# ----------------------
# Prediction
# ----------------------
if input_df is not None and not input_df.empty:
    st.write("### Input Data Preview:")
    st.dataframe(input_df, use_container_width=True)

    # Predict
    prediction = pipe.predict(input_df)
    prediction_labels = [cover_type_map.get(int(p), "Unknown") for p in prediction]

    st.success(f"🌲 Predicted Cover Type: **{prediction[0]} - {prediction_labels[0]}**")

    # If multiple rows uploaded
    if len(prediction) > 1:
        st.write("### Predictions for Uploaded Data:")
        result_df = input_df.copy()
        result_df["Predicted_Cover_Type"] = prediction
        result_df["Cover_Type_Label"] = prediction_labels
        st.dataframe(result_df, use_container_width=True)

        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Predictions as CSV",
            data=csv,
            file_name="forest_cover_predictions.csv",
            mime="text/csv"
        )