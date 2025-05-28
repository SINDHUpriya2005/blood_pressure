from flask import Flask, request, render_template
import pandas as pd
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

# Mapping for prediction results
result_map = {
    0: "NORMAL",
    1: "HYPERTENSION (Stage-1)",
    2: "HYPERTENSION (Stage-2)",
    3: "HYPERTENSIVE CRISIS"
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict-form')
def predict_form():
    return render_template('predict_form.html')

@app.route('/predict', methods=["POST"])
def predict():
    try:
        # Get form data and convert to float
        data = [float(request.form[field]) for field in [
            "Gender", "Age", "Patient", "Severity", "BreathShortness",
            "VisualChanges", "NoseBleeding", "Whendiagnoused", 
            "Systolic", "Diastolic", "ControlledDiet"
        ]]
        
        # Create DataFrame with correct column order
        # Note: This must match the order used during training
        features = ['Gender', 'Age', 'Patient', 'Severity', 'BreathShortness',
                   'VisualChanges', 'NoseBleeding', 'Whendiagnoused',
                   'Systolic', 'Diastolic', 'ControlledDiet']
        
        df = pd.DataFrame([data], columns=features)
        
        # Make prediction
        prediction = model.predict(df)[0]
        result = result_map.get(prediction, "Unknown")
        
        return render_template("result.html", prediction_text=result)

    except Exception as e:
        return render_template("result.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
