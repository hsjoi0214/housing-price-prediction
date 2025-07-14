# train_ensemble.py

import os
import joblib
import numpy as np

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold

from preprocess import preprocess_data
from three_stage_filter import run_three_stage_filter

def train_ensemble_models(
    data_path="data/raw/housing_iteration_6_regression.csv",
    output_dir="outputs/models",
    random_state=42
):
    print("📦 Preprocessing...")
    X, y_raw, preprocessor, _, _, _, _ = preprocess_data(data_path)
    y = np.log1p(y_raw)
    feature_names = preprocessor.get_feature_names_out()

    print("🧹 Feature selection...")
    X_filtered, selected_features = run_three_stage_filter(
        X, y, feature_names,
        k_best=100,
        model_thresh=0.001
    )

    os.makedirs(output_dir, exist_ok=True)

    # ✅ Save preprocessing artifacts
    joblib.dump(preprocessor, f"{output_dir}/preprocessor_ensemble.pkl")
    joblib.dump(selected_features, f"{output_dir}/selected_features_ensemble.pkl")

    # ✅ Train XGB
    print("🌲 Training XGBoost...")
    xgb = XGBRegressor(n_estimators=200, learning_rate=0.03, max_depth=5,
                       subsample=0.8, colsample_bytree=0.8,
                       random_state=random_state, verbosity=0)
    xgb.fit(X_filtered, y)
    joblib.dump(xgb, f"{output_dir}/xgb_model.pkl")

    # ✅ Train Random Forest
    print("🌳 Training RandomForest...")
    rf = RandomForestRegressor(n_estimators=200, max_depth=None,
                                min_samples_split=2, min_samples_leaf=1,
                                random_state=random_state)
    rf.fit(X_filtered, y)
    joblib.dump(rf, f"{output_dir}/rf_model.pkl")

    # ✅ Train Ridge Regression
    print("📏 Training Ridge Regression...")
    ridge = Ridge(alpha=10.0)
    ridge.fit(X_filtered, y)
    joblib.dump(ridge, f"{output_dir}/ridge_model.pkl")

    print("✅ All models saved to:", output_dir)


if __name__ == "__main__":
    train_ensemble_models()
