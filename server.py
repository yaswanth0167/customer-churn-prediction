from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load the trained XGBoost model
try:
    model = pickle.load(open("model.pkl", "rb"))
except Exception as e:
    print("Error loading model:", e)
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.json
    
    # Extract data from request
    tenure = float(data.get('tenure', 0))
    monthly_charges = float(data.get('monthly_charges', 0.0))
    total_charges = float(data.get('total_charges', 0.0))
    contract = data.get('contract', 'Month-to-month')
    internet = data.get('internet', 'DSL')
    security = data.get('security', 'No')
    support = data.get('support', 'No')

    # Construct feature array exactly as expected
    features = [tenure, monthly_charges, total_charges]
    
    features += [1 if contract == "One year" else 0, 1 if contract == "Two year" else 0]
    features += [1 if internet == "Fiber optic" else 0, 1 if internet == "No" else 0]
    features.append(1 if security == "Yes" else 0)
    features.append(1 if support == "Yes" else 0)

    feature_names = [
        'tenure', 'MonthlyCharges', 'TotalCharges', 
        'Contract_One year', 'Contract_Two year', 
        'InternetService_Fiber optic', 'InternetService_No', 
        'OnlineSecurity_Yes', 'TechSupport_Yes'
    ]
    
    df = pd.DataFrame([features], columns=feature_names)
    
    probabilities = model.predict_proba(df)[0]
    churn_prob = float(probabilities[1] * 100)
    stay_prob = float(probabilities[0] * 100)
    is_churn = bool(model.predict(df)[0] == 1)
    
    return jsonify({
        "churn": is_churn,
        "churn_prob": round(churn_prob, 1),
        "stay_prob": round(stay_prob, 1)
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
