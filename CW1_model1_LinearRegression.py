import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 1. Load data
print("Loading data...")
train_df = pd.read_csv('CW1_train.csv')
test_df = pd.read_csv('CW1_test.csv')

# 2. Preprocessing
categorical_cols = ['cut', 'color', 'clarity']
train_encoded = pd.get_dummies(train_df, columns=categorical_cols, drop_first=True)
test_encoded = pd.get_dummies(test_df, columns=categorical_cols, drop_first=True)

# 3. Align Columns
X = train_encoded.drop(columns=['outcome'])
y = train_encoded['outcome']
test_encoded = test_encoded.reindex(columns=X.columns, fill_value=0)

# 4. Split for internal validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=123)

# 5. Train Model
print("Training Model: Linear Regression...")
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Check Accuracy
y_pred = model.predict(X_val)
u = ((y_val - y_pred) ** 2).sum()
v = ((y_val - y_val.mean()) ** 2).sum()
accuracy = 1 - u/v

print(f"\n--- Result ---")
print(f"Model: Linear Regression")
print(f"Current Accuracy (R2 Score): {accuracy:.4f}")

# 7. Generate Submission
final_predictions = model.predict(test_encoded)
pd.DataFrame({'yhat': final_predictions}).to_csv('CW1_submission_K23170694_LinearRegression.csv', index=False)
print("Submission file saved.")
