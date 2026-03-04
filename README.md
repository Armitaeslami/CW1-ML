# CW1: Diamond Outcome Prediction
**KCL — Machine Learning Coursework 1**

Predicting a continuous `outcome` variable from diamond attributes. Best result: **R² = 0.4715** with XGBoost.

---

## Setup

```bash
git clone https://github.com/Armitaeslami/CW1-ML.git
cd CW1-ML
pip install -r requirements.txt
```

Drop `CW1_train.csv` and `CW1_test.csv` into the project folder, then run whichever model you want:

```bash
python CW1_model1_LinearRegression.py    # R² = 0.2891
python CW1_model2_RandomForest.py        # R² = 0.4523
python CW1_model3_GradientBoosting.py    # R² = 0.4590
python CW1_model4_XGBooster.py           # R² = 0.4715 ✅
```

Each script saves a submission CSV. To regenerate the report figures: `python generate_figures.py`

---

## Results

| Model | R² |
|---|:---:|
| Linear / Ridge / Lasso | 0.2891 |
| Random Forest | 0.4523 |
| Gradient Boosting | 0.4590 |
| **XGBoost** | **0.4715** |

Linear models confirmed the relationship is non-linear (R² ≈ 0.29). From there, tree-based models progressively improved things. XGBoost won out with engineered features (`log_price`, `carat²`, `price_per_carat`), L1/L2 regularisation, and early stopping at round 652.

All scripts use `random_state=123` for reproducibility.