# predict.py

import pandas as pd
import joblib
import os
import numpy as np

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


def predict_and_save(test_path="data/raw/test.csv",
                     model_dir="outputs/models",
                     output_path="outputs/predictions/submission.csv"):

    print("📦 Loading model, preprocessor, and features...")
    model = joblib.load(os.path.join(model_dir, "final_regression_model.pkl"))
    preprocessor = joblib.load(os.path.join(model_dir, "preprocessor.pkl"))
    selected_features = joblib.load(os.path.join(model_dir, "selected_features.pkl"))

    print("🧹 Preprocessing test data...")
    X_test, original_test_df = preprocess_test_data(test_path, preprocessor, selected_features)

    print("🔮 Making predictions...")
    preds_log = model.predict(X_test)
    preds = np.expm1(preds_log)  # ✅ Convert back from log scale

    submission = pd.DataFrame({
        'Id': original_test_df['Id'],
        'SalePrice': preds
    })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"✅ Predictions saved to: {output_path}")


if __name__ == "__main__":
    predict_and_save()
