import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

train_df = pd.read_csv('CW1_train.csv')
test_df = pd.read_csv('CW1_test.csv')

def add_features(df):
    df = df.copy()
    df['log_price'] = np.log1p(df['price'])
    df['carat2'] = df['carat'] ** 2
    df['price_per_carat'] = df['price'] / (df['carat'] + 1e-9)
    return df

train_df = add_features(train_df)
test_df = add_features(test_df)

categorical_cols = ['cut', 'color', 'clarity']
train_encoded = pd.get_dummies(train_df, columns=categorical_cols, drop_first=True)
test_encoded = pd.get_dummies(test_df, columns=categorical_cols, drop_first=True)

X = train_encoded.drop(columns=['outcome'])
y = train_encoded['outcome']
test_encoded = test_encoded.reindex(columns=X.columns, fill_value=0)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=123
)

model = XGBRegressor(
    n_estimators=2000,
    learning_rate=0.01,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    random_state=123,
    verbosity=0,
    early_stopping_rounds=50
)

model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

y_pred = model.predict(X_val)
u = ((y_val - y_pred) ** 2).sum()
v = ((y_val - y_val.mean()) ** 2).sum()
accuracy = 1 - u / v

print(f"R2 Score: {accuracy:.4f}")

final_predictions = model.predict(test_encoded)
pd.DataFrame({'yhat': final_predictions}).to_csv(
    'CW1_submission_K23170694_XGBooster.csv', index=False
)