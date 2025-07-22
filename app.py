from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the model
try:
    model = pickle.load(open('death-predictive.pkl', 'rb'))
    model_loaded = True
except Exception as e:
    print(f"Error loading model: {e}")
    model_loaded = False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model_loaded:
        return jsonify({'error': 'Model not loaded'})
    
    try:
        # Get form data
        data = {
            'age': float(request.form.get('age')),
            'anaemia': int(request.form.get('anaemia')),
            'creatinine_phosphokinase': float(request.form.get('creatinine_phosphokinase')),
            'diabetes': int(request.form.get('diabetes')),
            'ejection_fraction': float(request.form.get('ejection_fraction')),
            'high_blood_pressure': int(request.form.get('high_blood_pressure')),
            'platelets': float(request.form.get('platelets')),
            'serum_creatinine': float(request.form.get('serum_creatinine')),
            'serum_sodium': float(request.form.get('serum_sodium')),
            'sex': int(request.form.get('sex')),
            'smoking': int(request.form.get('smoking')),
            'time': int(request.form.get('time'))
        }
        
        # Prepare data for prediction
        features = np.array([[
            data['age'], 
            data['anaemia'], 
            data['creatinine_phosphokinase'], 
            data['diabetes'], 
            data['ejection_fraction'], 
            data['high_blood_pressure'], 
            data['platelets'],
            data['serum_creatinine'],
            data['serum_sodium'],
            data['sex'],
            data['smoking'],
            data['time']
        ]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        
        # Format results
        result = "High risk of heart failure" if prediction == 1 else "Low risk of heart failure"
        risk_percentage = round(probability[1] * 100, 1) if prediction == 1 else round(probability[0] * 100, 1)
        
        return jsonify({
            'result': result,
            'risk_percentage': risk_percentage,
            'prediction': int(prediction),
            'data': data
        })
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': 'Invalid input data'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=True)
