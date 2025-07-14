# train_model.py

import os
import joblib
import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, KFold

from preprocess import preprocess_data
from three_stage_filter import run_three_stage_filter


def train_and_save_model(
    data_path="data/raw/housing_iteration_6_regression.csv",
    output_dir="outputs/models",
    random_state=42
):
    print("📦 Loading and preprocessing data...")
    X, y_raw, preprocessor, df_raw, num, cat, ords = preprocess_data(data_path)
    feature_names = preprocessor.get_feature_names_out()

    # ✅ Transform target to log scale
    y = np.log1p(y_raw)

    print("🧹 Running 3-stage feature selection...")
    X_filtered, selected_features = run_three_stage_filter(
        X, y, feature_names,
        k_best=100,
        model_thresh=0.001 # Use a very low threshold to keep more features
    )

    print(f"🔎 Final feature count after selection: {len(selected_features)}")

    print("🔍 Performing GridSearchCV on XGBRegressor (full K-Fold)...")
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.03, 0.1],
        'subsample': [0.8, 1],
        'colsample_bytree': [0.8, 1]
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)

    grid = GridSearchCV(
        XGBRegressor(random_state=random_state, verbosity=0),
        param_grid,
        cv=kf,
        n_jobs=-1,
        scoring='neg_root_mean_squared_error',
        verbose=1
    )

    grid.fit(X_filtered, y)
    best_model = grid.best_estimator_

    print(f"✅ Best Parameters: {grid.best_params_}")
    print(f"📉 Best CV log-RMSE: {abs(grid.best_score_):.4f}")

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(best_model, f"{output_dir}/final_regression_model.pkl")
    joblib.dump(preprocessor, f"{output_dir}/preprocessor.pkl")
    joblib.dump(selected_features, f"{output_dir}/selected_features.pkl")

    print(f"📁 Model and artifacts saved in `{output_dir}/`")


if __name__ == "__main__":
    train_and_save_model()
