from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from xgboost import XGBRegressor
import pandas as pd
import numpy as np


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Custom transformer that applies a 3-stage feature selection pipeline:
        1. Variance Thresholding
        2. Correlation Filtering
        3. SelectFromModel (XGBoost)

    Compatible with sklearn Pipelines and GridSearchCV.
    """

    def __init__(self,
                 var_thresh=0.01,
                 corr_thresh=0.95,
                 model_thresh=0.001,
                 random_state=42,
                 verbose=True):
        self.var_thresh = var_thresh
        self.corr_thresh = corr_thresh
        self.model_thresh = model_thresh
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        self.feature_names_ = (
            X.columns.tolist() if isinstance(X, pd.DataFrame)
            else [f'f{i}' for i in range(X.shape[1])]
        )
        X_df = pd.DataFrame(X, columns=self.feature_names_)

        if self.verbose:
            print(" [FeatureSelector] Starting feature selection...")
            print(f" Initial feature count: {X_df.shape[1]}")

        # -------------------------------
        # Stage 1: Variance Thresholding
        # -------------------------------
        self.var_selector_ = VarianceThreshold(threshold=self.var_thresh)
        X_var = self.var_selector_.fit_transform(X_df)
        var_names = [self.feature_names_[i] for i in self.var_selector_.get_support(indices=True)]

        if self.verbose:
            print(f" After VarianceThreshold (>{self.var_thresh}): {len(var_names)} features")

        # -------------------------------
        # Stage 2: Correlation Filtering
        # -------------------------------
        X_var_df = pd.DataFrame(X_var, columns=var_names)
        corr_matrix = X_var_df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > self.corr_thresh)]
        self.corr_names_ = [col for col in var_names if col not in to_drop]

        if self.verbose:
            print(f" After Correlation Filter (>|{self.corr_thresh}|): {len(self.corr_names_)} features")

        # -------------------------------
        # Stage 3: SelectFromModel (XGBoost)
        # -------------------------------
        X_corr = X_var_df[self.corr_names_].values
        model = XGBRegressor(n_estimators=100, random_state=self.random_state, verbosity=0)
        self.model_selector_ = SelectFromModel(model, threshold=self.model_thresh)
        self.model_selector_.fit(X_corr, y)
        self.final_names_ = [self.corr_names_[i] for i in self.model_selector_.get_support(indices=True)]

        if self.verbose:
            print(f" After SelectFromModel (threshold={self.model_thresh}): {len(self.final_names_)} features")
            print(f" Final selected features: {len(self.final_names_)}\n")

        return self

    def transform(self, X):
        X_df = pd.DataFrame(X, columns=self.feature_names_)
        X_var = self.var_selector_.transform(X_df)
        var_names = [self.feature_names_[i] for i in self.var_selector_.get_support(indices=True)]
        X_var_df = pd.DataFrame(X_var, columns=var_names)

        X_corr_df = X_var_df[self.corr_names_]
        X_model = self.model_selector_.transform(X_corr_df)

        return X_model

    def get_feature_names_out(self, input_features=None):
        return self.final_names_
