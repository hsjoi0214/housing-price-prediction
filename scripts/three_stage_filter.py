# three_stage_filter.py

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression, SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


def apply_variance_threshold(X, threshold=0.01):
    selector = VarianceThreshold(threshold=threshold)
    X_reduced = selector.fit_transform(X)
    return X_reduced, selector


def apply_correlation_filter(X, feature_names, threshold=0.95):
    X_df = pd.DataFrame(X, columns=feature_names)
    corr_matrix = X_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    retained = [col for col in feature_names if col not in to_drop]
    X_filtered = X_df[retained].values



    return X_filtered, retained


def apply_select_k_best(X, y, k=50):
    selector = SelectKBest(score_func=f_regression, k=min(k, X.shape[1]))
    X_selected = selector.fit_transform(X, y)
    return X_selected, selector


def apply_select_from_model(X, y, threshold='mean'):
    #model = RandomForestRegressor(n_estimators=100, random_state=42)
    model = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    selector = SelectFromModel(model, threshold=threshold)
    X_selected = selector.fit_transform(X, y)
    return X_selected, selector


def run_three_stage_filter(X, y, feature_names, var_thresh=0.01, corr_thresh=0.95, k_best=100, model_thresh='median'):
    print("Stage 1: Variance Thresholding...")
    X_var, var_selector = apply_variance_threshold(X, threshold=var_thresh)
    retained_1 = var_selector.get_support(indices=True)
    names_1 = [feature_names[i] for i in retained_1]
    print(f"🔍 Features after variance threshold ({var_thresh}): {len(names_1)}")

    print("Stage 2: Correlation Filtering...")
    X_corr, names_2 = apply_correlation_filter(X_var, names_1, threshold=corr_thresh)
    print(f"🔍 Features after correlation filter (>|{corr_thresh}|): {len(names_2)}")

    print("Stage 3a: SelectKBest (f_regression)...")
    X_kbest, kbest_selector = apply_select_k_best(X_corr, y, k=k_best)
    retained_3a = kbest_selector.get_support(indices=True)
    names_3a = [names_2[i] for i in retained_3a]
    print(f"🔍 Features after SelectKBest (top {k_best}): {len(names_3a)}")

    print("Stage 3b: SelectFromModel (RandomForest)...")
    X_model, model_selector = apply_select_from_model(X_kbest, y, threshold=model_thresh)
    retained_3b = model_selector.get_support(indices=True)
    final_names = [names_3a[i] for i in retained_3b]
    print(f"🔍 Features after SelectFromModel (threshold={model_thresh}): {len(final_names)}")

    print(f"\n✅ Final selected features: {len(final_names)}")
    return X_model, final_names

