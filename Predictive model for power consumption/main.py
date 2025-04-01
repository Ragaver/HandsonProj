#Libraries
from flask import Flask, render_template,request
import joblib
import pandas as pd

#Loading model
preprocessor = joblib.load('preprocessor.pkl')
model = joblib.load('model.pkl')

#Initializing flask
app = Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict", methods = ["POST"])
def predict():
    try:
        Temperature = float(request.form['Temperature'])
        Humidity = float(request.form['Humidity'])
        Wind_Speed = float(request.form['Wind_Speed'])
        general_diffuse_flows = float(request.form['general_diffuse_flows'])
        diffuse_flows = float(request.form['diffuse_flows'])
        Air_Quality_Index_PM = float(request.form['Air_Quality_Index_PM'])
        Cloudiness = int(request.form['Cloudiness'])
        
        
        input_data = pd.DataFrame([[Temperature,Humidity,Wind_Speed, general_diffuse_flows, diffuse_flows,Air_Quality_Index_PM, Cloudiness ]],
                                columns = ['Temperature','Humidity','Wind_Speed', 'general_diffuse_flows', 'diffuse_flows','Air_Quality_Index_PM', 'Cloudiness' ])
        pre = preprocessor.transform(input_data)
        pred = model.predict(pre)[0]
        return render_template('index.html',prediction = f'Power Consumption: {pred}')
    except Exception as e:
        return render_template('index.html',prediction = f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug = True)


