#Libraries
from flask import Flask,render_template,request
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

@app.route("/predict",methods = ['POST'])
def predict():
    try:
        age = int(request.form['age'])
        income = int(request.form['income'])
        loan_amount = int(request.form['loan_amount'])
        credit_score = int(request.form['credit_score'])
        employment_status = request.form['employment_status']
        loan_term = int(request.form['loan_term'])
        interest_rate = float(request.form['interest_rate'])
        debt_to_income_ratio = float(request.form['debt_to_income_ratio'])
        num_of_dependents = int(request.form['num_of_dependents'])
        education_level = request.form['education_level']
        home_ownership = request.form['home_ownership']
        marital_status = request.form['marital_status']
        credit_history_length = int(request.form['credit_history_length'])
        num_credit_lines = int(request.form['num_credit_lines'])
        late_payments = int(request.form['late_payments'])
        bankruptcies = int(request.form['bankruptcies'])
        annual_savings = int(request.form['annual_savings'])
        retirement_savings = int(request.form['retirement_savings'])
        monthly_expenses = int(request.form['monthly_expenses'])
        input_data = pd.DataFrame([[age,income,loan_amount,credit_score,employment_status,loan_term,interest_rate,debt_to_income_ratio,num_of_dependents,education_level,home_ownership,marital_status,credit_history_length,num_credit_lines,late_payments,bankruptcies,annual_savings,retirement_savings,monthly_expenses]],columns = ['age','income','loan_amount','credit_score','employment_status','loan_term','interest_rate','debt_to_income_ratio','num_of_dependents','education_level','home_ownership','marital_status','credit_history_length','num_credit_lines','late_payments','bankruptcies','annual_savings','retirement_savings','monthly_expenses'])
        transform_data = preprocessor.transform(input_data)
        pred = model.predict(transform_data)[0]
        result = 'Fradulent detected' if pred == 1 else 'Default loan'
        return render_template('index.html',prediction = f'Prediction: {result}')
    except Exception as e:
        return render_template('index.html',prediction = f'Error: {str(e)}')
if __name__ == '__main__':
    app.run(debug = True)
        
        
        
        