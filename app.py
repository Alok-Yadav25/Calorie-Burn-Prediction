import flask as fk
import pandas as pd
import numpy as np
import pickle
import os

app = fk.Flask(__name__)

# Load the trained model
rfr = pickle.load(open('calories/rfr.pkl', 'rb'))  # Random Forest Regressor model

# Prediction function
def pred(Gender, Age, Height, Weight, Duration, Heart_Rate, Body_Temp):
    features = np.array([[Gender, Age, Height, Weight, Duration, Heart_Rate, Body_Temp]])
    prediction = rfr.predict(features)
    return prediction[0]

# Home route to render the HTML form
@app.route('/')
def home():
    return fk.render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        Gender = int(fk.request.form['Gender'])
        Age = int(fk.request.form['Age'])
        Height = float(fk.request.form['Height'])
        Weight = float(fk.request.form['Weight'])
        Duration = float(fk.request.form['Duration'])
        Heart_Rate = float(fk.request.form['Heart_Rate'])
        Body_Temp = float(fk.request.form['Body_Temp'])

        prediction = pred(Gender, Age, Height, Weight, Duration, Heart_Rate, Body_Temp)
        return fk.render_template('index.html', prediction=round(prediction, 2))

    except Exception as e:
        return fk.render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)