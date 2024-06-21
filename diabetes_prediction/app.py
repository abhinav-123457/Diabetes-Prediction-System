from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the model and scaler
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        data = request.form.to_dict()
        
        # Validate inputs
        try:
            input_data = [float(data[field]) for field in ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
        except ValueError:
            return render_template('index.html', result='Please fill out all fields with valid numbers.')
        
        input_data = scaler.transform([input_data])
        prediction = model.predict(input_data)
        result = 'Diabetic' if prediction[0] == 1 else 'Non-Diabetic'
        return render_template('index.html', result=result)
    
    return render_template('index.html', result='')

if __name__ == '__main__':
    app.run(debug=True)
