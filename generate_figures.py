import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

# load and prepare data
train_df = pd.read_csv('CW1_train.csv')
test_df  = pd.read_csv('CW1_test.csv')

def add_features(df):
    df = df.copy()
    df['log_price']       = np.log1p(df['price'])
    df['carat2']          = df['carat'] ** 2
    df['price_per_carat'] = df['price'] / (df['carat'] + 1e-9)
    return df

train_df = add_features(train_df)
test_df  = add_features(test_df)

categorical_cols = ['cut', 'color', 'clarity']
train_encoded = pd.get_dummies(train_df, columns=categorical_cols, drop_first=True)
test_encoded  = pd.get_dummies(test_df,  columns=categorical_cols, drop_first=True)
X = train_encoded.drop(columns=['outcome'])
y = train_encoded['outcome']
test_encoded = test_encoded.reindex(columns=X.columns, fill_value=0)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=123)

# train model (needed for figB and figC)
model = XGBRegressor(n_estimators=2000, learning_rate=0.01, max_depth=6,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
    random_state=123, verbosity=0, early_stopping_rounds=50)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
y_pred = model.predict(X_val)

plt.rcParams.update({'font.size': 9, 'font.family': 'serif'})

# figA - price skew before and after log transform
fig, axes = plt.subplots(1, 2, figsize=(7, 2.4))
axes[0].hist(train_df['price'], bins=45, color='#e07b54', edgecolor='white', linewidth=0.3)
axes[0].set_title('Raw price  (skewness = 1.63)', fontsize=9)
axes[0].set_xlabel('price'); axes[0].set_ylabel('count')
axes[1].hist(np.log1p(train_df['price']), bins=45, color='#4a9e6b', edgecolor='white', linewidth=0.3)
axes[1].set_title('log(1+price)  (skewness = 0.10)', fontsize=9)
axes[1].set_xlabel('log(1+price)'); axes[1].set_ylabel('count')
plt.tight_layout(pad=0.8)
plt.savefig('figA_price.png', dpi=160, bbox_inches='tight')
plt.close()

# figB - model comparison bar chart
models = ['Linear/Ridge/Lasso', 'Random Forest', 'Gradient Boosting', 'XGBoost']
scores = [0.2891, 0.4523, 0.4590, 0.4715]
colors = ['#c0392b', '#e67e22', '#2980b9', '#27ae60']
fig, ax = plt.subplots(figsize=(7, 2.2))
bars = ax.barh(models, scores, color=colors, height=0.5, edgecolor='white')
for bar, score in zip(bars, scores):
    ax.text(score+0.004, bar.get_y()+bar.get_height()/2,
            f'{score:.4f}', va='center', fontsize=9, fontweight='bold')
ax.axvline(0.47, color='black', linestyle='--', linewidth=1.3, label='Target (0.47)')
ax.set_xlabel('Validation $R^2$'); ax.set_xlim(0, 0.53)
ax.set_title('Model Comparison', fontsize=9, fontweight='bold')
ax.legend(fontsize=8)
plt.tight_layout(pad=0.8)
plt.savefig('figB_models.png', dpi=160, bbox_inches='tight')
plt.close()

# figC - learning curve and predicted vs actual
val_rmse = model.evals_result()['validation_0']['rmse']
fig, axes = plt.subplots(1, 2, figsize=(7, 2.6))
axes[0].plot(val_rmse, color='#2980b9', linewidth=1.0)
axes[0].axvline(model.best_iteration, color='tomato', linestyle='--',
                linewidth=1.3, label=f'Stop: round {model.best_iteration}')
axes[0].set_xlabel('Boosting Round'); axes[0].set_ylabel('Val RMSE')
axes[0].set_title('Learning Curve', fontsize=9, fontweight='bold')
axes[0].legend(fontsize=8)
axes[1].scatter(y_val, y_pred, alpha=0.2, s=5, color='#8e44ad')
lims = [min(y_val.min(), y_pred.min()), max(y_val.max(), y_pred.max())]
axes[1].plot(lims, lims, 'r--', linewidth=1.3)
axes[1].set_xlabel('Actual'); axes[1].set_ylabel('Predicted')
axes[1].set_title(r'Predicted vs Actual  ($R^2=0.4706$)', fontsize=9, fontweight='bold')
plt.tight_layout(pad=0.8)
plt.savefig('figC_eval.png', dpi=160, bbox_inches='tight')
plt.close()

# figD - full correlation heatmap all 31 attributes
numeric_df = train_df.select_dtypes(include=[np.number])
corr = numeric_df.corr()
fig, ax = plt.subplots(figsize=(18, 15))
mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr, mask=mask, annot=True, fmt='.1f', cmap='coolwarm',
            center=0, linewidths=0.3, ax=ax, annot_kws={'size': 6},
            cbar_kws={'shrink': 0.6})
ax.set_title('Feature Correlation Heatmap - All Attributes',
             fontsize=13, fontweight='bold', pad=12)
ax.tick_params(axis='both', labelsize=8)
plt.tight_layout()
plt.savefig('figD_heatmap.png', dpi=160, bbox_inches='tight')
plt.close()

print("All 4 figures saved.")