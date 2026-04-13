import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import pickle

# 1. Train and save the correct model
df = pd.read_csv("churn.csv")
df.dropna(inplace=True)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

# Align columns with app.py
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

xgb = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb.fit(X, y)

with open('model.pkl', 'wb') as f:
    pickle.dump(xgb, f)

# 2. Fix the notebook
with open('churn (1).ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        # Fix the df selection
        if "'PaymentMethod'" in source and "df = df[['tenure'" in source:
            cell['source'] = [
                "df['OnlineSecurity'] = df['OnlineSecurity'].replace('No internet service', 'No')\n",
                "df['TechSupport'] = df['TechSupport'].replace('No internet service', 'No')\n",
                "df = df[['tenure', 'MonthlyCharges', 'TotalCharges',\n",
                "         'Contract', 'InternetService', 'OnlineSecurity',\n",
                "         'TechSupport', 'Churn']]"
            ]
        # Fix the get_dummies call
        if "df = pd.get_dummies(df, drop_first=True)" in source:
            cell['source'] = ["df = pd.get_dummies(df, columns=['Contract', 'InternetService', 'OnlineSecurity', 'TechSupport'], drop_first=True)"]
            
        # Fix the save pickle part
        if "pickle.dump(rf_model," in source:
            cell['source'] = [src.replace('rf_model', 'xgb') for src in cell['source']]

with open('churn (1).ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f)

print("done")
