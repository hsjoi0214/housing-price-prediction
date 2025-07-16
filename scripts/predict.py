"""
Predict Script - House Price Prediction (Pipeline-Aware)

Loads pipeline with BoosterWrapper and generates predictions for Kaggle submission.
"""

import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from preprocess import add_engineered_features

# Redefine BoosterWrapper so joblib can unpickle it
class BoosterWrapper:
    """Sklearn-compatible wrapper for xgboost.Booster used at inference time."""
    def __init__(self, booster):
        self.booster = booster

    def predict(self, X):
        return self.booster.predict(xgb.DMatrix(X))

    def fit(self, X, y=None):
        # Dummy fit method so sklearn treats it as a fitted estimator
        return self

    def __sklearn_is_fitted__(self):
        # Mark this object as fitted so sklearn won't raise error
        return True



def predict_and_save(test_path="data/raw/test.csv",
                     model_path="outputs/models/final_regression_model.pkl",
                     output_path="outputs/predictions/submission.csv"):
    print(" Loading trained pipeline...")
    pipeline = joblib.load(model_path)

    print(" Reading and feature-engineering test data...")
    df = pd.read_csv(test_path)
    df = add_engineered_features(df)

    if 'SalePrice' in df.columns:
        df = df.drop(columns=['SalePrice'])

    print(" Predicting with full pipeline...")
    preds_log = pipeline.predict(df)
    preds = np.expm1(preds_log)  # inverse of log1p

    submission = pd.DataFrame({
        'Id': df['Id'],
        'SalePrice': preds
    })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f" Predictions saved to: {output_path}")


if __name__ == "__main__":
    predict_and_save()
