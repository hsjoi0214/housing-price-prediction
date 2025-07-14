"""
Predict Script - House Price Prediction (Pipeline-Aware)

Uses the full saved pipeline to ensure preprocessing + feature selection are applied consistently.
"""

import os
import joblib
import numpy as np
import pandas as pd

from preprocess import add_engineered_features

# -----------------------------------------------------
# Main Prediction Function
# -----------------------------------------------------

def predict_and_save(test_path="data/raw/test.csv",
                     model_path="outputs/models/final_regression_model.pkl",
                     output_path="outputs/predictions/submission.csv"):
    """
    Load the full trained pipeline, preprocess the test set,
    generate predictions, and export a submission CSV.
    """
    print("📦 Loading trained pipeline...")
    pipeline = joblib.load(model_path)

    print("📄 Reading and feature-engineering test data...")
    df = pd.read_csv(test_path)
    df = add_engineered_features(df)

    if 'SalePrice' in df.columns:
        df = df.drop(columns=['SalePrice'])

    print("🔮 Predicting with full pipeline...")
    preds_log = pipeline.predict(df)
    preds = np.expm1(preds_log)

    submission = pd.DataFrame({
        'Id': df['Id'],
        'SalePrice': preds
    })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"✅ Predictions saved to: {output_path}")


# -----------------------------------------------------
# Script Entry Point
# -----------------------------------------------------

if __name__ == "__main__":
    predict_and_save()
