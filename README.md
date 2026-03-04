# CW1: Diamond Outcome Prediction

**King's College London — Machine Learning Coursework 1**

A regression pipeline to predict a continuous `outcome` variable from diamond attributes, achieving a validation $R^2$ of **0.4715** using XGBoost with feature engineering.

---

## Project Structure

```
CW1-ML/
├── CW1_model1_LinearRegression.py      # Baseline linear model
├── CW1_model2_RandomForest.py          # Random Forest model
├── CW1_model3_GradientBoosting.py      # Gradient Boosting model
├── CW1_model4_XGBooster.py             # Final XGBoost model (best)
├── generate_figures.py                 # Generates all report figures
├── requirements.txt                    # Python dependencies
├── figA_price.png                      # Price skew figure
├── figB_models.png                     # Model comparison figure
├── figC_eval.png                       # Learning curve + predictions
└── figD_heatmap.png                    # Correlation heatmap
```

---

## Setup

1. Clone the repository and navigate into it:
```bash
git clone https://github.com/Armitaeslami/CW1-ML.git
cd CW1-ML
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Place `CW1_train.csv` and `CW1_test.csv` in the project folder.

---

## Running the Models

Each model can be run independently:

```bash
python CW1_model1_LinearRegression.py    # R² = 0.2891
python CW1_model2_RandomForest.py        # R² = 0.4523
python CW1_model3_GradientBoosting.py    # R² = 0.4590
python CW1_model4_XGBooster.py           # R² = 0.4715 ✅ best
```

Each script saves a submission CSV with predicted values.

To regenerate all report figures:
```bash
python generate_figures.py
```

---

## Results

| Model | Validation R² |
|-------|:-------------:|
| Linear / Ridge / Lasso | 0.2891 |
| Random Forest | 0.4523 |
| Gradient Boosting | 0.4590 |
| **XGBoost (final)** | **0.4715** |

> Target threshold: R² ≥ 0.47 ✅

---

## Approach

### Feature Engineering
Three new features were created to expose non-linear structure in the data:

| Feature | Formula | Reason |
|---------|---------|--------|
| `log_price` | log(1 + price) | Corrects heavy right skew (skewness 1.63 → 0.10) |
| `carat2` | carat² | Captures non-linear size effect on outcome |
| `price_per_carat` | price / (carat + ε) | Encodes value density of each diamond |

Categorical columns (`cut`, `color`, `clarity`) were one-hot encoded with `drop_first=True`, expanding the feature matrix to 39 columns.

### Model Selection
Linear models scored ~0.29, confirming the target relationship is non-linear. Tree-based ensemble methods progressively improved performance, with XGBoost achieving the best result through built-in L1/L2 regularisation, column subsampling, and early stopping.

### XGBoost Hyperparameters

| Parameter | Value | Reason |
|-----------|-------|--------|
| `learning_rate` | 0.01 | Small steps for stable learning |
| `max_depth` | 6 | Balances complexity and generalisation |
| `subsample` | 0.8 | Row sampling reduces variance |
| `colsample_bytree` | 0.8 | Feature sampling adds tree diversity |
| `min_child_weight` | 3 | Prevents splits on tiny groups |
| `early_stopping_rounds` | 50 | Auto-halts when validation stops improving |

Training stopped at round **652** out of a possible 2000, indicating clean convergence without overfitting.

---

## Reproducibility

All scripts use `random_state=123` throughout, ensuring fully reproducible results across every run.