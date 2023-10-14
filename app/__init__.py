from flask import Flask, render_template, request
import pandas as pd
import joblib
#import app

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_placement', methods=['POST'])
def predict_placement():
    # Load the machine learning model for placement prediction
    placement_model = joblib.load('models/placement_model.pkl')
    salary_model = joblib.load('models/salary_model.pkl')

    # Process the form data
    input_data = {
        'ssc_p': float(request.form['ssc_p']),
        'hsc_p': float(request.form['hsc_p']),
        'degree_p': float(request.form['degree_p']),
        'etest_p': float(request.form['etest_p']),
        'mba_p': float(request.form['mba_p']),
        'degree_t': request.form['degree_t'],
        'workex': request.form['workex'],
        'specialisation': request.form['specialisation']
    }

    # Convert categorical variables to one-hot encoding
    #input_data['degree_t'] = 1 if input_data['degree_t'] == 'Others' else 0
    #input_data['degree_t'] = 2 if input_data['degree_t'] == 'Sci&Tech' else 0

    if input_data['degree_t'] == 'Sci&Tech':
        input_data['degree_t'] = 2
    elif input_data['degree_t'] == 'Comm&Mgmt':
        input_data['degree_t'] = 0
    else:
        input_data['degree_t'] = 1
    input_data['workex'] = 1 if input_data['workex'] == 'Yes' else 0
    #input_data['workex_No'] = 1 if input_data['workex'] == 'No' else 0
    input_data['specialisation'] = 1 if input_data['specialisation'] == 'Mkt&HR' else 0

    # Create a DataFrame from the processed input data
    df = pd.DataFrame([input_data])
    #input_data1 = ['ssc_p','hsc_p','degree_p','workex','etest_p','mba_p']
    #df = pd.DataFrame({key: [input_data[key]] if isinstance(input_data[key], (int, float)) else input_data[key] for key in input_data1})

    # Make predictions
    placement_prediction = placement_model.predict(df[['ssc_p','hsc_p','degree_p','workex','etest_p','mba_p','degree_t','specialisation']])[0]
    salary_prediction = salary_model.predict(df[['ssc_p','hsc_p','degree_p','workex','etest_p','mba_p','degree_t','specialisation']])[0]
    print(placement_prediction)
    print(salary_prediction)
    if placement_prediction == 1:
        placement_prediction1 = "Placed"
    else:
        placement_prediction1 = "Not Placed"

    return render_template('result.html', placement_result=placement_prediction1, predicted_salary = salary_prediction)

if __name__ == '__main__':
    app.run(debug=True)
