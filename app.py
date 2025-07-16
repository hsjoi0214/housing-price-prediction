import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "scripts"))

from preprocess import add_engineered_features
from feature_selector import FeatureSelector

# Load artifacts
@st.cache_resource
def load_pipeline():
    model = joblib.load("outputs/models/final_regression_model.pkl")
    preprocessor = joblib.load("outputs/models/preprocessor.pkl")
    selected_features = joblib.load("outputs/models/selected_features.pkl")
    return model, preprocessor, selected_features

model, preprocessor, selected_features = load_pipeline()

# --- Streamlit UI ---
st.set_page_config(page_title="Housing Price Prediction", layout="wide")

# Sidebar
tabs = st.sidebar.radio("", ["🏡 Predict Price", "📊 Model Info"])

if tabs == "🏡 Predict Price":
    st.title("Housing Price Prediction")
    st.markdown("Predict the sale price of a house using an advanced regression model.")

    st.subheader("Enter House Details")
    col1, col2 = st.columns(2)

    with col1:
        overall_qual = st.slider("Overall Quality (1–10)", 1, 10, 6)
        year_built = st.number_input("Year Built", value=2005)
        total_sf = st.number_input("Total SF (Above + Basement)", value=2000)
        full_bath = st.number_input("Full Bathrooms", value=2)
        garage_area = st.number_input("Garage Area (sq ft)", value=400)
        has_garage = st.selectbox("Has Garage?", ["Yes", "No"])

    with col2:
        overall_cond = st.slider("Overall Condition (1–10)", 1, 10, 5)
        house_age = st.number_input("House Age", value=18)
        garage_cars = st.number_input("Garage Capacity (cars)", value=2)
        lot_area = st.number_input("Lot Area (sq ft)", value=7000)
        gr_liv_area = st.number_input("Ground Living Area", value=1800)

    if st.button("Predict Sale Price"):
        # Create dataframe
        input_dict = {
            "OverallQual": [overall_qual],
            "OverallCond": [overall_cond],
            "YearBuilt": [year_built],
            "HouseAge": [house_age],
            "TotalSF": [total_sf],
            "FullBath": [full_bath],
            "GarageArea": [garage_area],
            "GarageCars": [garage_cars],
            "LotArea": [lot_area],
            "GrLivArea": [gr_liv_area],
            "HasGarage": [1 if has_garage == "Yes" else 0],
        }

        df_input = pd.DataFrame(input_dict)
        X_proc = model.named_steps['preprocessing'].transform(df_input)
        X_sel = model.named_steps['feature_selection'].transform(X_proc)
        log_price = model.named_steps['regressor'].predict(X_sel)[0]
        final_price = np.expm1(log_price)

        st.success(f"Estimated Sale Price: ${final_price:,.0f}")

# --- Model Info Tab ---
elif tabs == "📊 Model Info":
    st.header("Model Information")

    st.markdown("""
    This model was trained on the Kaggle Housing dataset using:
    
    - XGBoost Regressor
    - 2-stage feature filtering (variance & correlation + model-based)
    - 5-fold cross-validation
    """)

    # Load CV predictions
    df_preds = pd.read_csv("outputs/predictions/submission.csv") if os.path.exists("outputs/predictions/submission.csv") else None

    if df_preds is not None and {'Actual', 'Predicted'}.issubset(df_preds.columns):
        st.subheader("Actual vs Predicted Sale Price")
        fig, ax = plt.subplots()
        ax.scatter(df_preds['Actual'], df_preds['Predicted'], alpha=0.6)
        ax.plot([df_preds['Actual'].min(), df_preds['Actual'].max()],
                [df_preds['Actual'].min(), df_preds['Actual'].max()],
                'r--')
        ax.set_xlabel("Actual Sale Price")
        ax.set_ylabel("Predicted Sale Price")
        ax.set_title("Model Fit")
        st.pyplot(fig)

        rmse = mean_squared_error(df_preds['Actual'], df_preds['Predicted'], squared=False)
        r2 = r2_score(df_preds['Actual'], df_preds['Predicted'])
        st.markdown(f"**RMSE:** ${rmse:,.0f}")
        st.markdown(f"**R² Score:** {r2:.3f}")
    else:
        st.info("Run model evaluation and save predictions to view actual vs predicted chart.")
