import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from sklearn.metrics import mean_squared_error, r2_score

sys.path.append(os.path.join(os.path.dirname(__file__), "scripts"))
from feature_selector import FeatureSelector

# --- Load Artifacts ---
@st.cache_resource
def load_pipeline():
    model = joblib.load("outputs/models/final_regression_model.pkl")
    preprocessor = joblib.load("outputs/models/preprocessor.pkl")
    selected_features = joblib.load("outputs/models/selected_features.pkl")
    return model, preprocessor, selected_features

model, preprocessor, selected_features = load_pipeline()

# --- App Config ---
st.set_page_config(page_title="Housing Price Prediction", layout="wide")
st.title("🏠 Housing Price Explorer & Predictor")

# --- Sidebar Navigation ---
tab = st.sidebar.radio("🔍 Navigate", ["📈 EDA", "🏡 Predict Price", "📊 Model Info"])

# --- Load full dataset for EDA ---
try:
    df_full = pd.read_csv("data/raw/housing_iteration_6_regression.csv")
except FileNotFoundError:
    st.error("⚠️ Could not find 'data/raw/housing_iteration_6_regression.csv'. Please check the path.")
    st.stop()

df_full["LogSalePrice"] = np.log1p(df_full["SalePrice"])

# --- Tab: EDA ---
if tab == "📈 EDA":
    st.header("📈 Exploratory Data Analysis")

    # Dataset summary
    st.subheader("📊 Dataset Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{df_full.shape[0]:,}")
    col2.metric("Features", f"{df_full.shape[1]:,}")
    col3.metric("Median Price", f"${df_full['SalePrice'].median():,.0f}")

    st.subheader("🧪 Feature Distribution")
    col = st.selectbox("Select feature", df_full.columns.sort_values())
    chart_type = st.radio("Chart Type", ["Histogram", "Boxplot"], horizontal=True)

    fig1, ax1 = plt.subplots(figsize=(4, 3))
    if chart_type == "Histogram":
        df_full[col].hist(ax=ax1, bins=20, color='skyblue')
    else:
        ax1.boxplot(df_full[col].dropna(), vert=False)
    ax1.set_title(f"{chart_type} of {col}", fontsize=11)
    ax1.tick_params(labelsize=8)
    fig1.tight_layout()
    st.pyplot(fig1, use_container_width=False)

    if df_full[col].dtype in [np.int64, np.float64]:
        st.subheader("💡 Relationship with Sale Price")
        fig2, ax2 = plt.subplots(figsize=(4, 3))
        ax2.scatter(df_full[col], df_full['SalePrice'], alpha=0.5)
        ax2.set_xlabel(col)
        ax2.set_ylabel("Sale Price")
        fig2.tight_layout()
        st.pyplot(fig2, use_container_width=False)

    with st.expander("🔍 See data sample"):
        st.dataframe(df_full[[col, "SalePrice"]].head(10))

    csv = df_full.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Full Dataset", csv, "housing_data.csv", "text/csv")

# --- Tab: Predict Price ---
elif tab == "🏡 Predict Price":
    st.header("🔮 Predict House Sale Price")
    st.markdown("Fill in the house details to get a price estimate.")
    col1, col2 = st.columns(2)

    with col1:
        overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 6)
        year_built = st.number_input("Year Built", value=2005)
        total_sf = st.number_input("Total SF (Above + Basement)", value=2000)
        full_bath = st.number_input("Full Bathrooms", value=2)
        garage_area = st.number_input("Garage Area (sq ft)", value=400)
        has_garage = st.selectbox("Has Garage?", ["Yes", "No"])

    with col2:
        overall_cond = st.slider("Overall Condition (1-10)", 1, 10, 5)
        house_age = st.number_input("House Age", value=18)
        garage_cars = st.number_input("Garage Capacity (cars)", value=2)
        lot_area = st.number_input("Lot Area (sq ft)", value=7000)
        gr_liv_area = st.number_input("Ground Living Area", value=1800)

    if st.button("Predict Sale Price"):
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
        try:
            X_proc = model.named_steps['preprocessing'].transform(df_input)
            X_sel = model.named_steps['feature_selection'].transform(X_proc)
            log_price = model.named_steps['regressor'].predict(X_sel)[0]
            final_price = np.expm1(log_price)
            st.success(f"Estimated Sale Price: ${final_price:,.0f}")
        except Exception as e:
            st.error(f"🚫 Prediction failed: {e}")

# --- Tab: Model Info ---
elif tab == "📊 Model Info":
    st.header("🧠 Model Details & Performance")
    st.markdown("""
    This model uses:
    - **XGBoost Regressor**
    - **3-stage feature selection**
    - **Log-transformed target**
    - **5-fold cross-validation**
    """)

    pred_path = "outputs/predictions/submission.csv"
    if os.path.exists(pred_path):
        df_preds = pd.read_csv(pred_path)
        if {'Actual', 'Predicted'}.issubset(df_preds.columns):
            st.subheader("Actual vs Predicted Plot")
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.scatter(df_preds['Actual'], df_preds['Predicted'], alpha=0.5)
            ax.plot([df_preds['Actual'].min(), df_preds['Actual'].max()],
                    [df_preds['Actual'].min(), df_preds['Actual'].max()], 'r--')
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=False)

            rmse = mean_squared_error(df_preds['Actual'], df_preds['Predicted'], squared=False)
            r2 = r2_score(df_preds['Actual'], df_preds['Predicted'])
            st.metric("RMSE", f"${rmse:,.0f}")
            st.metric("R² Score", f"{r2:.3f}")

    st.markdown("---")
    st.markdown("Developed for data scientists by a data scientist 🤓")
