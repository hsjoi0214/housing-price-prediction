"""
Train Model Script - House Price Prediction

Refactored to avoid data leakage using a pipeline-based feature selector.
"""

import os
import joblib
import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline

from preprocess import preprocess_data
from feature_selector import FeatureSelector  # ✅ New transformer replaces three_stage_filter

# -----------------------------------------------------
# Main Training Function
# -----------------------------------------------------

def train_and_save_model(
    data_path="data/raw/housing_iteration_6_regression.csv",
    output_dir="outputs/models",
    random_state=42
):
    print("🔄 Loading and preprocessing data...")

    # Load raw data and preprocessor
    X_raw, y_raw, preprocessor, df_raw, num, cat, ords = preprocess_data(data_path)

    # Log-transform target
    y = np.log1p(y_raw)

    # -----------------------------------------------------
    # Build Full Pipeline: Preprocessing → Feature Selection → Model
    # -----------------------------------------------------
    print("🔍 Building pipeline...")

    full_pipeline = Pipeline([
        ('preprocessing', preprocessor),
        ('feature_selection', FeatureSelector(
            var_thresh=0.01,
            corr_thresh=0.95,
            k_best=100,
            model_thresh=0.001,
            random_state=random_state,
            verbose=True
        )),
        ('regressor', XGBRegressor(objective='reg:squarederror', random_state=random_state, verbosity=0))
    ])

    # -----------------------------------------------------
    # Grid Search CV
    # -----------------------------------------------------
    print("🚀 Performing GridSearchCV on full pipeline...")

    param_grid = {
        'regressor__n_estimators': [100, 200],
        'regressor__max_depth': [3, 5, 7],
        'regressor__learning_rate': [0.03, 0.1],
        'regressor__subsample': [0.8, 1],
        'regressor__colsample_bytree': [0.8, 1]
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)

    grid = GridSearchCV(
        full_pipeline,
        param_grid=param_grid,
        cv=cv,
        n_jobs=-1,
        scoring='neg_root_mean_squared_error',
        verbose=1
    )

    grid.fit(X_raw, y)
    best_model = grid.best_estimator_

    # -----------------------------------------------------
    # Evaluation Summary
    # -----------------------------------------------------
    print(f"\n✅ Best Parameters: {grid.best_params_}")
    print(f"📉 Best CV log-RMSE: {abs(grid.best_score_):.4f}\n")

    # -----------------------------------------------------
    # Save Model and Artifacts
    # -----------------------------------------------------
    print("💾 Saving model and artifacts...")
    os.makedirs(output_dir, exist_ok=True)

    # Save the full pipeline (includes preprocessor + feature selector)
    joblib.dump(best_model, f"{output_dir}/final_regression_model.pkl")

    # Optionally, extract and save preprocessor + selected features separately
    # Only if needed in predict.py
    joblib.dump(preprocessor, f"{output_dir}/preprocessor.pkl")

    # Extract selected feature names
    selector = best_model.named_steps['feature_selection']
    selected_features = selector.get_feature_names_out()
    joblib.dump(selected_features, f"{output_dir}/selected_features.pkl")

    print(f"📁 Model and artifacts saved in `{output_dir}/`")


# -----------------------------------------------------
# Script Entry Point
# -----------------------------------------------------

if __name__ == "__main__":
    train_and_save_model()
