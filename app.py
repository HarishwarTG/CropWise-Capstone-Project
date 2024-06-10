from pipeline.predict import predict_result
import pandas as pd
from flask import Flask, request, render_template
import numpy as np

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        # Retrieve form data
        data_request = {
            'N': float(request.form.get('N')),
            'P': float(request.form.get('P')),
            'K': float(request.form.get('K')),
            'temp': float(request.form.get('temp')),
            'humidity': float(request.form.get('humidity')),
            'ph': float(request.form.get('ph')),
            'rainfall': float(request.form.get('rainfall')),
            'date': request.form.get('date')
        }

        # Create the data list for prediction
        data = [[data_request['N'], data_request['P'], data_request['K'], data_request['temp'],
                 data_request['humidity'], data_request['ph'], data_request['rainfall']]]

        # Convert date string to Timestamp
        inp_date = pd.Timestamp(data_request['date'])

        # Get the prediction result
        result = predict_result(data, inp_date)

        # Pass the result to the template
        return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
