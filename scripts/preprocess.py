import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer

# -------------------------------
# Define Ordinal Features & Order
# -------------------------------

ORDINAL_FEATURES = [
    'OverallQual', 'OverallCond',
    'ExterQual', 'ExterCond',
    'BsmtQual', 'BsmtCond', 'BsmtExposure',
    'BsmtFinType1', 'BsmtFinType2',
    'HeatingQC', 'KitchenQual', 'FireplaceQu',
    'GarageFinish', 'GarageQual', 'GarageCond',
    'Functional', 'PavedDrive', 'PoolQC', 'Fence', 'Utilities',
    'LandSlope', 'LotShape'
]

ORDINAL_CATEGORIES = [
    list(range(1, 11)), list(range(1, 11)),  # OverallQual, OverallCond
    ['Po', 'Fa', 'TA', 'Gd', 'Ex'], ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
    ['Po', 'Fa', 'TA', 'Gd', 'Ex', 'NA'], ['Po', 'Fa', 'TA', 'Gd', 'Ex', 'NA'],
    ['No', 'Mn', 'Av', 'Gd', 'NA'],
    ['Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ', 'NA'],
    ['Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ', 'NA'],
    ['Po', 'Fa', 'TA', 'Gd', 'Ex'], ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
    ['Po', 'Fa', 'TA', 'Gd', 'Ex', 'NA'],
    ['Unf', 'RFn', 'Fin', 'NA'], ['Po', 'Fa', 'TA', 'Gd', 'Ex', 'NA'],
    ['Po', 'Fa', 'TA', 'Gd', 'Ex', 'NA'],
    ['Sal', 'Sev', 'Maj2', 'Maj1', 'Mod', 'Min2', 'Min1', 'Typ'],
    ['N', 'P', 'Y'],
    ['Fa', 'TA', 'Gd', 'Ex', 'NA'],
    ['MnWw', 'GdWo', 'MnPrv', 'GdPrv', 'NA'],
    ['ELO', 'NoSeWa', 'NoSewr', 'AllPub'],
    ['Gtl', 'Mod', 'Sev'],
    ['Reg', 'IR1', 'IR2', 'IR3']
]

# ----------------------------
# Feature Engineering Function
# ----------------------------

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df['TotalSF'] = df['1stFlrSF'] + df['2ndFlrSF'] + df['TotalBsmtSF']
    df['TotalBathrooms'] = (
        df['FullBath'] + 0.5 * df['HalfBath'] +
        df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath']
    )
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']
    df['GarageScore'] = df['GarageArea'] * df['GarageCars']

    df['HasPool'] = df['PoolQC'].notna().astype(int)
    df['HasFireplace'] = df['FireplaceQu'].notna().astype(int)
    df['HasGarage'] = df['GarageType'].notna().astype(int)
    df['HasBasement'] = df['BsmtQual'].notna().astype(int)
    df['HasPorch'] = (
        (df['OpenPorchSF'] + df['EnclosedPorch'] +
         df['3SsnPorch'] + df['ScreenPorch']) > 0
    ).astype(int)

    df['QualSF'] = df['TotalSF'] * df['OverallQual']
    df['AgeScore'] = df['HouseAge'] * df['OverallCond']
    df['LivLotRatio'] = df['GrLivArea'] / (df['LotArea'] + 1)
    df['GarageScorePerCar'] = df['GarageScore'] / (df['GarageCars'] + 1)
    df['BathsPerRoom'] = df['TotalBathrooms'] / (df['TotRmsAbvGrd'] + 1)
    df['OverallQualCond'] = df['OverallQual'] * df['OverallCond']
    df['SFperRoom'] = df['GrLivArea'] / (df['TotRmsAbvGrd'] + 1)
    df['BasementScore'] = df['TotalBsmtSF'] * df['BsmtFullBath']
    df['AgePerGarage'] = df['HouseAge'] / (df['GarageCars'] + 1)
    df['RemodAgePerSF'] = df['RemodAge'] / (df['TotalSF'] + 1)

    df['IsRemodeled'] = (df['YearBuilt'] != df['YearRemodAdd']).astype(int)
    df['IsOldHouse'] = (df['HouseAge'] > 50).astype(int)

    return df

# ----------------------------
# Custom Imputation Handling
# ----------------------------

def custom_fillna(df: pd.DataFrame) -> pd.DataFrame:
    # LotFrontage → median by Neighborhood
    if 'Neighborhood' in df.columns and 'LotFrontage' in df.columns:
        df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
            lambda x: x.fillna(x.median())
        )

    # GarageYrBlt → impute with YearBuilt
    if 'GarageYrBlt' in df.columns and 'YearBuilt' in df.columns:
        df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['YearBuilt'])

    # MasVnrArea → fill with 0
    if 'MasVnrArea' in df.columns:
        df['MasVnrArea'] = df['MasVnrArea'].fillna(0)

    return df

# ----------------------------
# Feature Classification
# ----------------------------

def classify_features(df: pd.DataFrame):
    all_columns = set(df.columns) - {'SalePrice', 'Id'}
    numerics = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categoricals = df.select_dtypes(include='object').columns.tolist()

    numerical_features = sorted(list(set(numerics) - set(ORDINAL_FEATURES)))
    categorical_features = list(set(categoricals) - set(ORDINAL_FEATURES))
    categorical_features.append('MSSubClass')

    all_classified = set(numerical_features + categorical_features + ORDINAL_FEATURES)
    unclassified = sorted(list(all_columns - all_classified))

    print(f"Total Features: {len(all_columns)}")
    print(f"Numerical Features: {len(numerical_features)}")
    print(f"Categorical Features: {len(categorical_features)}")
    print(f"Ordinal Features: {len(ORDINAL_FEATURES)}")
    print(f"Unclassified Features: {unclassified}")

    return numerical_features, categorical_features, ORDINAL_FEATURES

# ----------------------------
# Preprocessing Pipeline Builder
# ----------------------------

def build_preprocessor(numerical, categorical, ordinal):
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    ordinal_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='NA')),
        ('encoder', OrdinalEncoder(categories=ORDINAL_CATEGORIES,
                                   handle_unknown='use_encoded_value',
                                   unknown_value=-1))
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='NA')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, numerical),
        ('ord', ordinal_pipeline, ordinal),
        ('cat', categorical_pipeline, categorical)
    ])

    return preprocessor

# ----------------------------
# Full Preprocessing Entry Point
# ----------------------------

def preprocess_data(csv_path: str):
    df = pd.read_csv(csv_path)
    df = add_engineered_features(df)
    df = custom_fillna(df)

    X = df.drop(columns=['SalePrice'])
    y = df['SalePrice']

    numerical, categorical, ordinal = classify_features(X)
    preprocessor = build_preprocessor(numerical, categorical, ordinal)

    return X, y.values, preprocessor, df, numerical, categorical, ordinal
