import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model= pickle.load(open('CRandomForest (2).pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    N = int(request.form['Nitrogen'])
    P = int(request.form['Phosphorus'])
    K = int(request.form['Potassium'])
    Temperature = float(request.form['TEMPERATURE'])
    Humidity = float(request.form['HUMIDITY'])
    pH = float(request.form['PH'])
    Rainfall = float(request.form['RAINFALL'])

    final_features = np.array([[N,P,K,Temperature,Humidity,pH,Rainfall]])
    prediction = model.predict(final_features)

    output = prediction[0]

    return render_template('index.html', prediction_text=output)

if __name__ == "__main__":
    app.run(debug=True)

# Flask>=1.1.1
# gunicorn>=19.9.0
# itsdangerous>=1.1.0
# numpy
# scipy
# pandas>=0.19
# matplotlib>=1.4.3
# scikit-learn>=0.18
# Werkzeug>=0.15.5
# Jinja2>=2.10.1
# MarkupSafe>=1.1.1