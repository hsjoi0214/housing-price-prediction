# styled_app.py

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import base64
from sklearn.metrics import mean_squared_error, r2_score

sys.path.append(os.path.join(os.path.dirname(__file__), "scripts"))
from feature_selector import FeatureSelector
from preprocess import add_engineered_features, custom_fillna

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
with open("assets/icons/golden_house.png", "rb") as img_file:
    encoded_img = base64.b64encode(img_file.read()).decode()

st.markdown(f"""
    <div style="
        display: flex;
        align-items: center;
        gap: 1.5rem;
        padding: 1rem 2rem;
        margin-bottom: 1rem;
        background-color: #141414;
        border-radius: 16px;
        width: fit-content;
        box-shadow: 0 0 20px rgba(240, 196, 32, 0.2);
    ">
        <div style="
            background-color: black;
            padding: 1.2rem;
            border-radius: 20px;
            box-shadow: 0 0 25px #f0c420;
            display: flex;
            align-items: center;
            justify-content: center;
        ">
            <img src="data:image/png;base64,{encoded_img}" alt="App Icon" style="height: 50px;">
        </div>
        <span style="
            font-size: 2rem;
            font-weight: 800;
            color: white;
            letter-spacing: 0.5px;
        ">
            Housing Price Predictor
        </span>
    </div>
""", unsafe_allow_html=True)



# --- Sidebar Navigation (Reordered) ---
tab = st.sidebar.radio("\U0001F50D Navigate", ["\U0001F3E1 Predict Price", "\U0001F4CA Model Info", "\U0001F4C8 EDA"])

# --- Load full dataset for EDA and Default Row ---
try:
    df_full = pd.read_csv("data/raw/housing_iteration_6_regression.csv")
except FileNotFoundError:
    st.error("\u26A0\ufe0f Could not find 'data/raw/housing_iteration_6_regression.csv'. Please check the path.")
    st.stop()

@st.cache_data
def load_default_row():
    df = df_full.copy()
    df = df.drop(columns=["SalePrice"], errors='ignore')
    for col in df.columns:
        if df[col].dtype in ["int64", "float64"]:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])
    row = df.iloc[[0]].copy()
    row["Id"] = 999999  # dummy ID to avoid missing errors
    return row

# --- Tab: Predict Price ---
if tab == "\U0001F3E1 Predict Price":
    st.header("Predict House Sale Price")
    st.markdown("Fill in the house details to get a price estimate.")

    # Side-by-side layout for image + slider on left, inputs on right
    left_col, right_col = st.columns([1, 2], gap="large")

    with left_col:
        st.markdown("### Overall Quality")
        st.markdown('<div style="margin-top: 1rem;"></div>', unsafe_allow_html=True)  # Space under subheading

        overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 6, label_visibility="collapsed")

        st.markdown('<div style="margin-top: 1.5rem;"></div>', unsafe_allow_html=True)  # Space before image
        st.image("assets/banners/house_visual.png", use_container_width=True)

        st.markdown('<div style="margin-top: 1.5rem;"></div>', unsafe_allow_html=True)  # Space before button
        predict_clicked = st.button("Predict Sale Price")

    with right_col:
        year_built = st.number_input("Year Built", value=2005)
        full_bath = st.number_input("Full Bathrooms", value=2)
        garage_area = st.number_input("Garage Area (sq ft)", value=400)
        garage_cars = st.number_input("Garage Capacity (cars)", value=2)
        lot_area = st.number_input("Lot Area (sq ft)", value=7000)
        gr_liv_area = st.number_input("Ground Living Area", value=1800)
        bedroom_abv_gr = st.number_input("Bedrooms (Excludes Basement)", value=3)

    # Prediction logic
    if predict_clicked:  # Prevents double triggering
        input_row = load_default_row()
        input_row["GrLivArea"] = gr_liv_area
        input_row["BedroomAbvGr"] = bedroom_abv_gr
        input_row["FullBath"] = full_bath
        input_row["GarageCars"] = garage_cars
        input_row["GarageArea"] = garage_area
        input_row["YearBuilt"] = year_built
        input_row["LotArea"] = lot_area
        input_row["OverallQual"] = overall_qual

        try:
            input_row = add_engineered_features(input_row)
            input_row = custom_fillna(input_row)
            X_proc = model.named_steps['preprocessing'].transform(input_row)
            X_sel = model.named_steps['feature_selection'].transform(X_proc)
            log_price = model.named_steps['regressor'].predict(X_sel)[0]
            final_price = np.expm1(log_price)
            st.success(f"\U0001F3F7️ Estimated Sale Price: **${final_price:,.0f}**")
        except Exception as e:
            st.error(f"\u274C Prediction failed: {e}")


# --- Tab: Model Info ---
elif tab == "\U0001F4CA Model Info":
    st.header("Model Details & Performance")

    col1, col2, col3 = st.columns([1, 1, 1], gap="large")
    box_style = """
        background-color: #111;
        border: 1.5px solid #d4af37;
        border-radius: 12px;
        padding: 1rem;
        min-height: 270px;
        box-shadow: 0 0 8px rgba(212, 175, 55, 0.1);
    """

    with col1:
        st.markdown(f"""
            <div style="{box_style}">
                <h4 style="color:#d4af37;">📄 Task</h4>
                <p style="color:white;">A supervised learning regression task for predicting house prices.</p> 
                <p style="color:white;">The training dataset includes ~80 raw features for each house.</p>
            </div>
        """, unsafe_allow_html=True)


    with col2:
        st.markdown(f"""
            <div style="{box_style}">
                <h4 style="color:#d4af37;">⚙️ Strategy</h4>
                <ul style="color:white;">
                    <li>Feature engineering to create new variables (e.g., TotalSF, GarageScore).</li>
                    <li>Two-filter selection:
                        <ul>
                            <li>First: Variance + Correlation to remove low-signal and redundant features.</li>
                            <li>Second: <code>SelectFromModel</code> with XGBoost.</li>
                        </ul>
                    </li>
                </ul>
            </div>
        """, unsafe_allow_html=True)


    with col3:
        st.markdown(f"""
            <div style="{box_style}">
                <h4 style="color:#d4af37;">🔧 Model Details</h4>
                <ul style="color:white;">
                    <li>Log-transformed SalePrice</li>
                    <li>5-fold cross-validation</li>
                    <li>XGBoost:
                        <ul>
                            <li><code>max_depth=5</code>, <code>learning_rate=0.05</code></li>
                            <li><code>n_estimators=500</code>, <code>subsample=1.0</code></li>
                            <li><code>reg_alpha=0.5</code>, <code>reg_lambda=0.5</code></li>
                        </ul>
                    </li>
                </ul>
            </div>
        """, unsafe_allow_html=True)



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

# --- Tab: EDA ---
elif tab == "\U0001F4C8 EDA":
    st.header("Exploratory Data Analysis")

    df_full["LogSalePrice"] = np.log1p(df_full["SalePrice"])

    st.subheader("Dataset Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{df_full.shape[0]:,}")
    col2.metric("Features", f"{df_full.shape[1]:,}")
    col3.metric("Median Price", f"${df_full['SalePrice'].median():,.0f}")

    st.subheader("\U0001F58D Feature Distribution")

    # Friendly names for better UX
    feature_display_map = {
        "GrLivArea": "Ground Living Area (sq ft)",
        "GarageArea": "Garage Area (sq ft)",
        "TotalBsmtSF": "Basement Area (sq ft)",
        "1stFlrSF": "First Floor (sq ft)",
        "2ndFlrSF": "Second Floor (sq ft)",
        "LotArea": "Lot Size (sq ft)",
        "BedroomAbvGr": "Bedrooms (Above Ground)",
        "FullBath": "Full Bathrooms",
        "HalfBath": "Half Bathrooms",
        "OverallQual": "Overall Quality (1-10)",
        "OverallCond": "Overall Condition (1-10)",
        "YearBuilt": "Year Built",
        "YearRemodAdd": "Remodel Year",
        "GarageCars": "Garage Capacity (cars)",
        "TotRmsAbvGrd": "Total Rooms (Above Ground)",
        "Fireplaces": "Number of Fireplaces",
        "SalePrice": "Sale Price",
        "3SsnPorch": "Three-Season Porch Area (sq ft)",
        "OpenPorchSF": "Open Porch Area (sq ft)",
        "EnclosedPorch": "Enclosed Porch Area (sq ft)",
        "ScreenPorch": "Screened Porch Area (sq ft)",
        "MSSubClass": "Building Class (Encoded)",
        "MasVnrArea": "Masonry Veneer Area (sq ft)",
        "LotFrontage": "Street Frontage (linear feet)",
        "MiscVal": "Miscellaneous Feature Value ($)",
        "LowQualFinSF": "Low Quality Finished Area (sq ft)",
        "MoSold": "Month Sold (1-12)",
        "BsmtFinSF1": "Finished Basement Area 1 (sq ft)",
        "BsmtFinSF2": "Finished Basement Area 2 (sq ft)",
        "BsmtUnfSF": "Unfinished Basement Area (sq ft)",
        "BsmtFullBath": "Basement Full Bathrooms",
        "BsmtHalfBath": "Basement Half Bathrooms",
        "YrSold": "Year Sold",
        "WoodDeckSF": "Wood Deck Area (sq ft)",
        "GarageYrBlt": "Year Garage Built",
        "KitchenAbvGr": "Kitchens Above Ground"
}


    # Filter numeric columns and apply readable labels
    excluded = {"Id", "LogSalePrice"}
    numeric_cols = [col for col in df_full.columns if df_full[col].dtype in [np.int64, np.float64] and col not in excluded]
    
    feature_options = {col: feature_display_map.get(col, col) for col in numeric_cols}
    inv_feature_map = {v: k for k, v in feature_options.items()}

    # Dropdown
    display_name = st.selectbox("Select feature", sorted(inv_feature_map.keys()))
    raw_feature = inv_feature_map[display_name]

    chart_type = st.radio("Chart Type", ["Histogram"], horizontal=True)

    st.subheader("\U0001F4A1 Feature Distribution")
    fig1, ax1 = plt.subplots(figsize=(4, 3))
    if chart_type == "Histogram":
        df_full[raw_feature].hist(ax=ax1, bins=20, color='skyblue')
    else:
        ax1.boxplot(df_full[raw_feature].dropna(), vert=False)
    ax1.set_title(f"{chart_type} of {display_name}", fontsize=11)
    ax1.tick_params(labelsize=8)
    fig1.tight_layout()
    st.pyplot(fig1, use_container_width=False)

    if df_full[raw_feature].dtype in [np.int64, np.float64]:
        st.subheader("\U0001F4A1 Relationship with Sale Price")
        fig2, ax2 = plt.subplots(figsize=(4, 3))
        ax2.scatter(df_full[raw_feature], df_full['SalePrice'], alpha=0.5)
        ax2.set_xlabel(display_name)
        ax2.set_ylabel("Sale Price")
        fig2.tight_layout()
        st.pyplot(fig2, use_container_width=False)

    with st.expander("\U0001F50D See data sample"):
        st.dataframe(df_full[[raw_feature, "SalePrice"]].head(10))

    csv = df_full.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Full Dataset", csv, "housing_data.csv", "text/csv")
