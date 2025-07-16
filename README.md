# Housing Price Prediction — Advanced Regression Pipeline

## Objective

The goal of this project is to accurately predict house sale prices using a combination of domain-driven feature engineering and layered feature filtering — ultimately building a high-performing regression model submitted to the Kaggle competition “House Prices: Advanced Regression Techniques”.

---

## Project Strategy

To build a reliable and scalable pipeline, I implemented a **2-stage filtering strategy** for feature selection combined with a modular preprocessing and modeling approach using XGBoost.

###  Stage 1: Manual / Statistical Filtering
- **Low variance removal**: Drop features with variance below a threshold (0.01).
- **Collinearity check**: Remove features with Pearson correlation > 0.95.
- **Rationale**: This reduces redundancy, simplifies the dataset, and eliminates clearly irrelevant/noisy signals.

###  Stage 2: Model-Based Feature Selection (SelectFromModel)
- Trained an XGBoost regressor and selected features with importance above a threshold (e.g. `0.001`).
- This ensures selected features are not only statistically relevant, but actually useful to the model’s predictive performance.

---

## Why This 2-Stage Approach Works

| Stage | Purpose | Benefit |
|-------|---------|---------|
| **1. Variance/Correlation** | Reduces noise and multicollinearity | Simplifies feature space, avoids overfitting |
| **2. SelectFromModel** | Uses model-driven importance | Aligns feature selection with true signal strength |

This design avoids premature assumptions about linear relationships (as with SelectKBest), and instead relies on the model itself to discover complex patterns.

---

## Tools & Libraries

- **Languages**: Python
- **Libraries**: scikit-learn, XGBoost, pandas, numpy, joblib
- **Other**: KFold CV, Pipelines, FeatureUnion, ColumnTransformer, Jupyter, VSCode

---

## Experiments & Results

All experiments followed a unified preprocessing and feature filtering pipeline, with multiple XGBoost hyperparameter configurations evaluated via cross-validation.

---

### **Current Pipeline (XGBoost + 2-Stage Feature Filtering)**

**Parameters:**
- Variance threshold: 0.01  
- Correlation threshold: 0.95  
- Model-based feature importance: threshold = `0.001`  
- Validation: 5-Fold Cross Validation  
- Model: XGBoost (various parameter sets tested)

**Best Result:**
- Final features: **~50–60**
- Best CV log-RMSE: **0.1349**

---

## Lessons Learned

- **Removing SelectKBest** simplified the flow and actually improved performance.
- **Model-driven feature selection** was more aligned with true predictive power.
- **XGBoost outperformed all other algorithms**, especially when paired with clean feature engineering.
- **Cross-validation (KFold)** gave a more reliable evaluation than single holdout splits.

---

## Feature Engineering

Feature engineering played a crucial role in improving model accuracy. Beyond the raw features provided, several domain-informed transformations were added to enhance predictive signal and reduce redundancy.

### Constructed Features

| Feature | Description |
|--------|-------------|
| `TotalSF` | Combined total living area: 1stFlr + 2ndFlr + TotalBsmtSF |
| `TotalBathrooms` | Combined count: FullBath + 0.5×HalfBath + Basement baths |
| `HouseAge`, `RemodAge` | Years since construction and remodeling respectively |
| `GarageScore` | Interaction: GarageArea × GarageCars |
| `HasPool`, `HasFireplace`, `HasGarage`, `HasBasement`, `HasPorch` | Binary flags for amenity presence |
| `QualSF` | Quality-adjusted living area: TotalSF × OverallQual |
| `AgeScore` | Durability proxy: HouseAge × OverallCond |
| `LivLotRatio` | Density metric: GrLivArea / LotArea |
| `GarageScorePerCar` | Garage efficiency per vehicle |
| `BathsPerRoom` | TotalBathrooms / TotalRooms |
| `OverallQualCond` | Interaction: OverallQual × OverallCond |
| `SFperRoom` | Living area per room |
| `BasementScore` | Composite score: TotalBsmtSF × BsmtFullBath |
| `AgePerGarage` | Aging impact per garage space |
| `RemodAgePerSF` | Years since remodel per square foot |
| `IsRemodeled` | Binary: whether YearBuilt ≠ YearRemodAdd |
| `IsOldHouse` | Binary: whether house is older than 50 years |

These features allowed the model to capture nonlinear patterns and higher-order interactions while remaining interpretable. Many were selected as part of the final feature set by the model-based filter.

---

## Final Solution Pipeline

- Data cleaned, missing values imputed
- Domain-specific feature engineering
- 2-stage feature filtering (variance + correlation → model-based selection)
- Target log-transformed
- XGBoost Regressor evaluated via KFold CV
- Predictions inverse-transformed with `np.expm1`
- Submission prepared for Kaggle

---

## Final Kaggle Result

| Submission | Strategy | Kaggle RMSE Score |
|------------|----------|-------------------|
| Best Model | XGBoost-only + refined feature filtering | **0.12033** |
| Others | Ensemble / Boosted / LightGBM / Extra Features | 0.13–0.15 range |
Submission File: [`submission.csv`](./outputs/predictions/submission.csv)

---

## Project Structure

```text
housing_price_prediction/
│
├── data/
│   └── raw/                    # Contains train.csv, test.csv
├── outputs/
│   └── models/                 # Stores trained model + artifacts
│   └── predictions/            # Final Kaggle submission file
├── scripts/
│   ├── preprocess.py           # Base preprocessing module
│   ├── feature_selector.py     # 2-stage feature filtering (Variance + Corr + XGB)
│   ├── train_model.py          # Train using XGBoost with CV
│   ├── predict.py              # Generates predictions from test.csv
│   └── run_all.sh              # Shell script to run all steps

```
---

## How to Run

### Install dependencies (if needed)
```bash
pip install -r requirements.txt
```

### Train Model
```bash
python scripts/train_model.py
```

### Generate Predictions
```bash
python scripts/predict.py
```

### Run Full Pipeline
```bash
bash scripts/run_all.sh
```

---

## Deployment & Contribution

### Clone this Repository
```bash
git clone https://github.com/your-username/housing-price-prediction.git
cd housing-price-prediction
```

### Contributing
Pull requests are welcome. If you have suggestions or improvements (feature engineering ideas, model tuning), feel free to open an issue.

---

## Languages and Libraries Used
- **Python**
- **XGBoost, scikit-learn, pandas, numpy, joblib**

---

## Data Sources
- [Kaggle House Prices Competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/)

---

## 👤 Author

**Prakash Joshi**  
Data Science Bootcamp

---

## Acknowledgements

Special thanks to my mentor **Sabine Joseph** for continuous feedback and guidance throughout this project.