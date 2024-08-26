from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle

# Load the trained model and the scaler
model = pickle.load(open('logistics.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve the input data from the form
    int_features = [float(x) for x in request.form.values()]
    features = np.array(int_features)
    
    # Scale the input data using the loaded scaler
    scaled_features = scaler.transform([features])
    
    # Make prediction using the loaded model
    prediction = model.predict(scaled_features)
    
    # Prepare the prediction text
    output = 'Heart Disease' if prediction[0] == 1 else 'No Heart Disease'
    
    return render_template('index.html', prediction_text=f'Prediction: {output}')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)
