#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd


# In[ ]:





# In[7]:


import numpy as np


# In[8]:


import matplotlib.pyplot as plt


# In[9]:


import seaborn as sns


# In[10]:


from sklearn.model_selection import train_test_split


# In[11]:


from sklearn.preprocessing import LabelEncoder


# In[12]:


from sklearn.linear_model import LogisticRegression


# In[13]:


from sklearn.metrics import accuracy_score, confusion_matrix


# In[14]:


df=pd.read_csv("churn.csv")
print(df.head())


# In[15]:


import pandas as pd


# In[16]:


import seaborn as sns


# In[17]:


df.isnull().sum()
df.dropna(inplace=True)


# In[18]:


df = df.drop(columns=['customerID'])


# In[53]:


df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)


# In[54]:


df = df[['tenure', 'MonthlyCharges', 'TotalCharges',
         'Contract', 'InternetService', 'OnlineSecurity',
         'TechSupport', 'PaymentMethod', 'Churn']]


# In[89]:


df = pd.get_dummies(df, drop_first=True)


# In[58]:


X = df.drop('Churn', axis=1)
y = df['Churn']


# In[57]:


print(df.columns)


# In[59]:


le = LabelEncoder()
df['Churn'] = le.fit_transform(df['Churn']) 


# In[ ]:





# In[61]:


sns.countplot(x='Churn', data=df)
plt.show()


# In[62]:


X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[63]:


print(df.dtypes)


# In[64]:


X = df.drop(columns=['Churn'])
y = df['Churn']


# In[65]:


categorical_cols = X.select_dtypes(include=['object', 'string']).columns

numeric_cols = X.select_dtypes(exclude=['object']).columns

print("Categorical:", categorical_cols)
print("Numeric:", numeric_cols)


# In[66]:


X_train.head()


# In[67]:


import pandas as pd

X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)

# Align columns in case train/test differ
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)


# In[74]:


model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)


# In[69]:


print("Train accuracy:", model.score(X_train, y_train))
print("Test accuracy:", model.score(X_test, y_test))


# In[70]:


from sklearn.ensemble import RandomForestClassifier


# In[93]:


rf_model = RandomForestClassifier(
    n_estimators=150,
    max_depth=8,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42
)


# In[94]:


rf_model.fit(X_train, y_train)


# In[95]:


print("Train Accuracy:", rf_model.score(X_train, y_train))
print("Test Accuracy:", rf_model.score(X_test, y_test))


# In[79]:


y_pred_rf = rf_model.predict(X_test)


# In[103]:


from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))


# In[97]:


get_ipython().system('pip install xgboost')


# In[98]:


from xgboost import XGBClassifier


# In[ ]:


xgb = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

xgb.fit(X_train, y_train)


# In[100]:


print("Train Accuracy:", xgb.score(X_train, y_train))
print("Test Accuracy:", xgb.score(X_test, y_test))


# In[101]:


import pickle

# Save your trained model
pickle.dump(rf_model, open("model.pkl", "wb"))


# In[104]:


model = pickle.load(open("model.pkl", "rb"))


# In[52]:


pip install streamlit


# In[ ]:





# In[ ]:




