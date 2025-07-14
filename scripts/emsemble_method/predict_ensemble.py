# predict_ensemble.py

import pandas as pd
import numpy as np
import os
import joblib

from preprocess import add_engineered_features, classify_features, build_preprocessor

def preprocess_test_data(test_path, preprocessor, selected_features):
    df = pd.read_csv(test_path)
    df = add_engineered_features(df)
    if 'SalePrice' in df.columns:
        df = df.drop(columns=['SalePrice'])

    numerical, categorical, ordinal = classify_features(df)
    X_processed = preprocessor.transform(df)
    feature_names = preprocessor.get_feature_names_out()
    df_features = pd.DataFrame(X_processed, columns=feature_names)
    df_filtered = df_features[selected_features]
    return df_filtered, df

def predict_and_ensemble(test_path="data/raw/test.csv",
                         model_dir="outputs/models",
                         output_path="outputs/predictions/submission.csv"):

    print("📦 Loading models...")
    xgb = joblib.load(os.path.join(model_dir, "xgb_model.pkl"))
    rf = joblib.load(os.path.join(model_dir, "rf_model.pkl"))
    ridge = joblib.load(os.path.join(model_dir, "ridge_model.pkl"))
    preprocessor = joblib.load(os.path.join(model_dir, "preprocessor_ensemble.pkl"))
    selected_features = joblib.load(os.path.join(model_dir, "selected_features_ensemble.pkl"))

    print("🧹 Preprocessing test data...")
    X_test, df_original = preprocess_test_data(test_path, preprocessor, selected_features)

    print("🔮 Predicting from all models...")
    preds_xgb = xgb.predict(X_test)
    preds_rf = rf.predict(X_test)
    preds_ridge = ridge.predict(X_test)

    # 🧠 Weighted ensemble (you can tune these)
    final_preds_log = (
        0.8 * preds_xgb +
        0.15 * preds_rf +
        0.05 * preds_ridge
    )
    final_preds = np.expm1(final_preds_log)

    submission = pd.DataFrame({
        'Id': df_original['Id'],
        'SalePrice': final_preds
    })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"✅ Submission saved to: {output_path}")


if __name__ == "__main__":
    predict_and_ensemble()
