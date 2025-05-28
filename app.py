import numpy as np
import pickle
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__, static_folder='static')
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict-form')
def predict_form():
    return render_template('predict_form.html')

@app.route('/predict', methods=["POST"])
def predict():
    try:
        data = [float(request.form[field]) for field in [
            "Gender", "Age", "Patient", "Severity", "BreathShortness",
            "VisualChanges", "NoseBleeding", "Whendiagnoused", 
            "Systolic", "Diastolic", "ControlledDiet"
        ]]
        df = pd.DataFrame([data], columns=[
            'Gender', 'Age', 'Patient', 'Severity', 'BreathShortness',
            'VisualChanges', 'NoseBleeding', 'Whendiagnoused',
            'Systolic', 'Diastolic', 'ControlledDiet'
        ])

        prediction = model.predict(df)[0]

        result_map = {
            0: "NORMAL",
            1: "HYPERTENSION (Stage-1)",
            2: "HYPERTENSION (Stage-2)",
            3: "HYPERTENSIVE CRISIS"
        }
        result = result_map.get(prediction, "Unknown")

        return render_template("result.html", prediction_text=result)

    except Exception as e:
        return render_template("result.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
