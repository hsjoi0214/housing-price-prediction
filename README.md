# Housing Price Prediction — Advanced Regression Pipeline

## Objective

The goal of this project is to accurately predict house sale prices using a combination of domain-driven feature engineering and layered feature filtering — ultimately building a high-performing regression model submitted to the Kaggle competition “House Prices: Advanced Regression Techniques”.

---

## Project Strategy

To build a reliable and scalable pipeline, I implemented a **2-stage filtering strategy** for feature selection combined with a modular preprocessing and modeling approach using XGBoost.

### Stage 1: Manual / Statistical Filtering
- **Low variance removal**: Drop features with variance below a threshold (0.01).
- **Collinearity check**: Remove features with Pearson correlation > 0.95.
- **Rationale**: This reduces redundancy, simplifies the dataset, and eliminates clearly irrelevant/noisy signals.

### Stage 2: Model-Based Feature Selection (SelectFromModel)
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
- **Web Interface**: Streamlit (for model deployment)
- **Visualization**: matplotlib, seaborn
- **Other**: KFold CV, Pipelines, FeatureUnion, ColumnTransformer, Jupyter, shell scripting

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

## Streamlit App Deployment

An interactive web app was created using **Streamlit** to allow users to simulate house characteristics and get instant sale price predictions from the trained model.

### Key Features

- **Two tabs**:
  - **Predict Price** — Set inputs for house attributes (quality, area, bathrooms, etc.) and get predicted price.
  - **Model Info** — Displays how the model was trained, and visualizes top features using bar charts.
  
- **Preprocessing + Feature Selection + Trained Model** runs under the hood.
- **Minimal input requirements** to simulate a house — remaining features are automatically imputed via preprocessing.

### Launch Locally

Ensure the model is trained and artifacts are saved under `outputs/models/`.

```bash
streamlit run app.py
```

This will open the web interface at: http://localhost:8501

---

## Project Structure

```text
housing_price_prediction/
│
├── data/
│   └── raw/                     # Original train/test CSV files
│
├── notebooks/                   # Jupyter notebooks for exploration (optional)
│
├── outputs/
│   ├── models/                  # Trained model and preprocessing artifacts
│   └── predictions/            # Final Kaggle submission file
│
├── scripts/
│   ├── preprocess.py            # Preprocessing logic
│   ├── feature_selector.py      # Variance/correlation/model-based filtering
│   ├── train_model.py           # Train XGBoost model
│   ├── predict.py               # Run prediction on test.csv
│   └── run_all.sh               # End-to-end shell script
│
├── app.py                       # Streamlit app interface
├── README.md
├── requirements.txt
├── .gitignore

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