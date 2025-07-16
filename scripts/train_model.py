import os
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

from preprocess import preprocess_data
from feature_selector import FeatureSelector


def train_and_save_model(
    data_path="data/raw/housing_iteration_6_regression.csv",
    output_dir="outputs/models",
    random_state=42
):
    print("Loading and preprocessing data...")

    # Load raw data and preprocessor
    X_raw, y_raw, preprocessor, df_raw, num, cat, ords = preprocess_data(data_path)
    y = np.log1p(y_raw)

    print("Building pipeline...")

    param_grid = [
        {
            'max_depth': 3,
            'learning_rate': 0.05,
            'n_estimators': 500,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'gamma': 0.1,
        },
        {
            'max_depth': 5,
            'learning_rate': 0.05,
            'n_estimators': 500,
            'subsample': 1.0,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.5,
            'reg_lambda': 0.5,
            'gamma': 0.0,
        },
    ]

    best_score = float('inf')
    best_model = None
    best_params = None

    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)

    for params in param_grid:
        print(f"\nTesting params: {params}")
        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_raw)):
            print(f" Fold {fold + 1}/5")

            X_train, X_val = X_raw.iloc[train_idx], X_raw.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Preprocess
            X_train_proc = preprocessor.fit_transform(X_train)
            X_val_proc = preprocessor.transform(X_val)

            # Feature selection
            feature_selector = FeatureSelector(
                var_thresh=0.01,
                corr_thresh=0.95,
                model_thresh=0.001,
                random_state=random_state,
                verbose=False
            )
            X_train_sel = feature_selector.fit_transform(X_train_proc, y_train)
            X_val_sel = feature_selector.transform(X_val_proc)

            # Train XGBoostRegressor
            xgb_model = XGBRegressor(
                objective='reg:squarederror',
                verbosity=0,
                **params
            )
            xgb_model.fit(X_train_sel, y_train)

            preds_val = xgb_model.predict(X_val_sel)
            rmse = np.sqrt(mean_squared_error(y_val, preds_val))
            fold_scores.append(rmse)

        avg_score = np.mean(fold_scores)
        print(f"  Avg CV log-RMSE: {avg_score:.4f}")

        if avg_score < best_score:
            best_score = avg_score
            best_params = params

            # Save full pipeline with XGBRegressor
            best_model = Pipeline([
                ('preprocessing', preprocessor),
                ('feature_selection', feature_selector),
                ('regressor', xgb_model)
            ])

    print(f"\nBest Parameters: {best_params}")
    print(f"Best CV log-RMSE: {best_score:.4f}\n")

    print("Saving model and artifacts...")
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(best_model, f"{output_dir}/final_regression_model.pkl")
    joblib.dump(preprocessor, f"{output_dir}/preprocessor.pkl")

    selector = best_model.named_steps['feature_selection']
    selected_features = selector.get_feature_names_out()
    joblib.dump(selected_features, f"{output_dir}/selected_features.pkl")

    print(f"Model and artifacts saved in `{output_dir}/`")


if __name__ == "__main__":
    train_and_save_model()
