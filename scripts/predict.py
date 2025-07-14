"""
Predict Script - House Price Prediction

This script loads the saved model and preprocessing artifacts, transforms the test dataset,
generates predictions on the log scale, converts them back to the original scale,
and writes the results to a submission CSV file.
"""

import os
import joblib
import numpy as np
import pandas as pd

from preprocess import add_engineered_features, classify_features, build_preprocessor

# -----------------------------------------------------
# Test Data Preprocessing
# -----------------------------------------------------

def preprocess_test_data(test_path, preprocessor, selected_features):
    """
    Load and preprocess the test dataset using the saved preprocessor pipeline.
    Only retain the final selected features.
    """
    df = pd.read_csv(test_path)
    df = add_engineered_features(df)

    if 'SalePrice' in df.columns:
        df = df.drop(columns=['SalePrice'])  # Ensure it's not leaked in test phase

    numerical, categorical, ordinal = classify_features(df)

    # Apply saved preprocessing pipeline
    X_processed = preprocessor.transform(df)
    feature_names = preprocessor.get_feature_names_out()

    # Retain only selected features used during model training
    df_features = pd.DataFrame(X_processed, columns=feature_names)
    df_filtered = df_features[selected_features]

    return df_filtered, df


# -----------------------------------------------------
# Main Prediction Function
# -----------------------------------------------------

def predict_and_save(test_path="data/raw/test.csv",
                     model_dir="outputs/models",
                     output_path="outputs/predictions/submission.csv"):
    """
    Load the model and preprocessing artifacts, predict on test set,
    inverse log-transform the results, and export submission CSV.
    """
    print(" Loading model, preprocessor, and features...")
    model = joblib.load(os.path.join(model_dir, "final_regression_model.pkl"))
    preprocessor = joblib.load(os.path.join(model_dir, "preprocessor.pkl"))
    selected_features = joblib.load(os.path.join(model_dir, "selected_features.pkl"))

    print(" Preprocessing test data...")
    X_test, original_test_df = preprocess_test_data(test_path, preprocessor, selected_features)

    print(" Making predictions...")
    preds_log = model.predict(X_test)
    preds = np.expm1(preds_log)  #  Revert log1p transformation

    # Prepare submission file
    submission = pd.DataFrame({
        'Id': original_test_df['Id'],
        'SalePrice': preds
    })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission.to_csv(output_path, index=False)

    print(f" Predictions saved to: {output_path}")


# -----------------------------------------------------
# Script Entry Point
# -----------------------------------------------------

if __name__ == "__main__":
    predict_and_save()
