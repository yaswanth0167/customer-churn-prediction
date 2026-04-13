import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
import pickle

print("Starting Data Load...")
df = pd.read_csv("churn.csv")
df.dropna(inplace=True)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

df['OnlineSecurity'] = df['OnlineSecurity'].replace('No internet service', 'No')
df['TechSupport'] = df['TechSupport'].replace('No internet service', 'No')

df = df[['tenure', 'MonthlyCharges', 'TotalCharges',
         'Contract', 'InternetService', 'OnlineSecurity',
         'TechSupport', 'Churn']]

le = LabelEncoder()
df['Churn'] = le.fit_transform(df['Churn'])

df = pd.get_dummies(df, columns=['Contract', 'InternetService', 'OnlineSecurity', 'TechSupport'], drop_first=True)

X = df.drop('Churn', axis=1)
y = df['Churn']

# Imbalance check
neg_count = sum(y == 0)
pos_count = sum(y == 1)
scale_pos_weight = neg_count / pos_count
print(f"Data Loaded! Negatives: {neg_count}, Positives: {pos_count}, Weight={scale_pos_weight:.2f}")

xgb = XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    objective='binary:logistic',
    random_state=42
)

params = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1, 0.15],
    'max_depth': [3, 4, 5, 6],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.25, 0.5],
    'min_child_weight': [1, 3, 5]
}

print("Initiating Search Space (F1 Optimized)...")
search = RandomizedSearchCV(
    xgb, params, n_iter=25, scoring='f1', 
    cv=5, random_state=42, n_jobs=-1, verbose=1
)
search.fit(X, y)

print("Optimization Complete!")
print(f"Best Parameters: {search.best_params_}")
print(f"Best F1 Score (Cross-Validated): {search.best_score_:.4f}")

# Overwrite the app model
best_model = search.best_estimator_

with open('model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
    
print("Saved optimized model to model.pkl successfully.")
