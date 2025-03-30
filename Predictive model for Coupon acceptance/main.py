#Libraries

from flask import Flask, render_template,request
import pandas as pd
import joblib

preprocessor = joblib.load('preprocessor.pkl')
model = joblib.load('model.pkl')

# Initialize Flask App
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict",methods = ['POST'])
def predict():
    try:
        temperature = int(request.form['temperature'])
        age = request.form['age']
        income = request.form['income']
        Bar = request.form['Bar']
        CoffeeHouse = request.form['CoffeeHouse']
        CarryAway = request.form['CarryAway']
        RestaurantLessThan20 = request.form['RestaurantLessThan20']
        Restaurant20To50 = request.form['Restaurant20To50']
        destination = request.form['destination']
        passanger = request.form['passanger']
        weather = request.form['weather']
        coupon = request.form['coupon']
        gender = request.form['gender']
        maritalStatus = request.form['maritalStatus']
        has_children = request.form['has_children']
        education = request.form['education']
        occupation = request.form['occupation']
        toCoupon_GEQ15min = request.form['toCoupon_GEQ15min']
        toCoupon_GEQ25min = request.form['toCoupon_GEQ25min']
        direction_same = request.form['direction_same']
        direction_opp = request.form['direction_opp']
        expiration = request.form['expiration']
        
        input_data = pd.DataFrame([[temperature, age, income, Bar, CoffeeHouse, CarryAway, RestaurantLessThan20, Restaurant20To50, destination, passanger, weather, coupon, gender,  maritalStatus, has_children, education, occupation, toCoupon_GEQ15min, toCoupon_GEQ25min, direction_same, direction_opp, expiration]],columns = ['temperature', 'age', 'income', 'Bar', 'CoffeeHouse', 'CarryAway', 'RestaurantLessThan20', 'Restaurant20To50', 'destination', 'passanger', 'weather', 'coupon', 'gender',  'maritalStatus', 'has_children', 'education', 'occupation', 'toCoupon_GEQ15min', 'toCoupon_GEQ25min', 'direction_same', 'direction_opp', 'expiration'])
        
        transformed_data = preprocessor.transform(input_data)
        pred = model.predict(transformed_data)[0]
        result = "Coupon Accepted" if pred == 1 else "Coupon Not Accepted"
        return render_template('index.html',prediction = f"Prediction is {result}")
    except Exception as e:
        return render_template('index.html',prediction = f'Error : {str(e)}')

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)