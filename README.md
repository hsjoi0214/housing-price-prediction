# Housing Price Prediction — Advanced Regression Pipeline

## Objective

The goal of this project is to accurately predict house sale prices using a combination of advanced preprocessing, feature engineering, and multi-stage feature selection techniques — ultimately building a high-performing regression model submitted to the Kaggle competition “House Prices: Advanced Regression Techniques”.

---

## Project Strategy

To build a reliable and scalable pipeline, we followed a **3-stage filtering strategy** for feature selection combined with modular model training and evaluation.

###  Stage 1: Manual / Statistical Filtering
- **Low variance removal**: Drop features with variance below a threshold (0.01).
- **Collinearity check**: Remove features with Pearson correlation > 0.95.
- **Rationale**: This reduces redundancy, simplifies the dataset, and removes clearly irrelevant/noisy features.

###  Stage 2: Statistical Feature Selection (SelectKBest)
- Applied `SelectKBest` with `f_regression` to retain only the top **K = 100** features based on statistical relationship with the target.
- Helps eliminate statistically weak predictors before modeling begins.

###  Stage 3: Model-Based Feature Selection (SelectFromModel)
- Trained a model (RandomForest or XGBoost) and selected features with importance above a threshold (e.g. `mean`, `median`, or `0.001`).
- This filter aligns feature importance with actual predictive performance.

---

## Why This 3-Stage Approach Works

| Stage | Purpose | Benefit |
|-------|---------|---------|
| **1. Variance/Correlation** | Reduces feature space noise and multicollinearity | Makes model simpler and faster |
| **2. SelectKBest** | Identifies statistically significant features | Ensures features are correlated with the target |
| **3. SelectFromModel** | Picks features based on model’s actual usage | Captures non-linear and interaction effects |

---

## Tools & Libraries

- **Languages**: Python
- **Libraries**: scikit-learn, XGBoost, pandas, numpy, joblib
- **Other**: GridSearchCV, Pipelines, FeatureUnion, ColumnTransformer, Jupyter, VSCode

---

## Experiments & Results

All experiments followed the same base preprocessing pipeline, with modifications in model strategy or feature filter configuration.

---

### **Experiment 1: Basic Feature Filtering + Random Forest**

**Parameters:**
- Variance threshold: 0.01  
- Correlation threshold: 0.95  
- KBest = 50  
- SelectFromModel: RandomForestRegressor (threshold = `'mean'`)

**Result:**
- Final features: **2**
- log-RMSE: **0.1843**
- R²: **0.8179**
- RMSE ($): **$35,505.64**
- MAE ($): **$22,197.58**

---

### **Experiment 2: Expanded K + Random Forest**

**Parameters:**
- KBest = 100  
- SelectFromModel: threshold = `'median'`

**Result:**
- Final features: **50**
- log-RMSE: **0.1496**
- R²: **0.8801**
- RMSE ($): **$29,945.89**
- MAE ($): **$17,298.94**

---

### **Experiment 3: XGBoost Replaced RF + KFold + CV Evaluation**

**Changes:**
- Model: XGBoost  
- Validation: 5-Fold CV instead of train/test split  
- Evaluation based on CV best_score

**Result:**
- Final features: **50**
- Best CV log-RMSE: **0.1366**
- 🥈 **First major improvement in leaderboard**

---

### **Experiment 4: Lower Threshold in SelectFromModel**

**Changes:**
- SelectFromModel threshold set to **0.001** (instead of median)
- Model: XGBoost  
- CV evaluation only

**Result:**
- Final features: **57**
- Best CV log-RMSE: **0.1349**

---

### **Experiment 5: Ensembling (XGBoost + RF + Ridge)**

**Changes:**
- Combined predictions from 3 base models  
- Averaged predictions for final output

**Result:**
- Final features: **57**
- Best CV log-RMSE: **0.1349**
- Kaggle score: Slightly worse than XGBoost-only

---

## Lessons Learned

-  **More features isn’t always better** — the quality of selection and modeling matter more.
-  **XGBoost outperformed all other algorithms** consistently in this tabular regression task.
-  **Using KFold CV** provided more stable and generalizable results than a fixed train/test split.
-  Ensembling didn’t improve much due to overlap in what models were learning.
-  Adding interaction features, smoothing, and outlier removal helped *slightly* or even hurt Kaggle score due to leakage risk or overfitting.

---

## Final Solution Pipeline

- Data cleaned, missing values imputed
- Manual + statistical + model-based feature filtering (a 3-filter approach)
- Log-transformed target
- XGBoost Regressor + GridSearchCV
- Prediction output transformed back using `np.expm1`
- Submission prepared and uploaded to Kaggle

---

## Final Kaggle Result

| Submission | Strategy | Kaggle RMSE Score |
|------------|----------|-------------------|
| Best Model | XGBoost-only + refined feature filtering | **0.12033** |
| Others | Ensemble / Boosted / LightGBM / Extra Features | 0.13–0.15 range |

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
│   ├── three_stage_filter.py   # Feature filtering pipeline
│   ├── train_model.py          # Train using XGB (original)
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