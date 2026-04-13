# Customer Churn Prediction 📊

This repository contains a Machine Learning-powered Customer Churn Prediction web application. It predicts whether a customer is likely to churn (leave the service) or stay based on their demographic, contract, and internet service data. 

The project includes two different user interfaces for the predictions:
1. A **Streamlit** dashboard for a quick, interactive pure-Python frontend.
2. A **Flask** backend API serving a custom high-end HTML/CSS/JS web application.

## 🚀 Features

* **Machine Learning Model:** Uses a pre-trained **XGBoost** model (`model.pkl`) to accurately forecast churn probabilities.
* **Dual Interfaces:** 
  * Run `app.py` for the Streamlit dashboard app.
  * Run `server.py` for the Flask REST API and custom CSS/HTML frontend.
* **Real-time Predictions:** Accepts inputs such as Tenure, Monthly Charges, Total Charges, Contract type, Internet service, Online security, and Tech support.
* **Probabilistic Output:** The Flask app provides exact percentual probabilities for churning vs. staying alongside the binary classification.

## 🛠️ Tech Stack

* **Machine Learning:** Pandas, NumPy, XGBoost, Pickle
* **Web Frameworks:** Flask (Backend/API), Streamlit (Frontend/Dashboard)
* **Frontend:** HTML5, Vanilla CSS, JavaScript (for the Flask custom frontend)

## 📂 Project Structure

```text
├── app.py                      # Streamlit application
├── server.py                   # Flask server application
├── model.pkl                   # Pre-trained XGBoost model
├── churn.ipynb / churn.py      # Jupyter notebooks and scripts used for exploratory data analysis (EDA) and model training
├── archive/ / static/ / templates/ # Frontend assets for Flask app
└── README.md                   # Project description
```

## ⚙️ Installation & Usage

1. **Clone the repository** (if pushed to GitHub):
   ```bash
   git clone <your-repository-url>
   cd "customer churn"
   ```

2. **Install dependencies**:
   Ensure you have Python installed, then install the necessary packages. (You can create a virtual environment first if preferred):
   ```bash
   pip install pandas numpy xgboost streamlit flask flask-cors
   ```

3. **Run the Application:**

   **To start the Streamlit App:**
   ```bash
   streamlit run app.py
   ```
   
   **To start the Flask Web App:**
   ```bash
   python server.py
   ```
   Then open your browser and navigate to `http://127.0.0.1:5000/`.

## 📊 Input Features
The prediction model expects the following features to be provided by the user:
* **Numerical:** Tenure (months), Monthly Charges, Total Charges (input in raw financial amounts contextually, like Dollars).
* **Categorical:** 
  * Contract Type (Month-to-month, One year, Two year)
  * Internet Service (DSL, Fiber optic, No)
  * Online Security (Yes, No)
  * Tech Support (Yes, No)

---
*Developed for accurately predicting customer retention.*
