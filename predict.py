import pickle

from flask import Flask
from flask import request
from flask import jsonify
import  pandas as pd

from waitress import serve

model_file = 'model_dt.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load((f_in))

def predict_single(applicant, dv, model):
  X = dv.transform([applicant])  ## apply the one-hot encoding feature to the applicant data
  y_pred = model.predict(X)
  return y_pred


app = Flask('predict')

@app.route('/predict', methods=['POST'])  ## in order to send the applicant information we need to post its data.
def predict():
    applicant = request.get_json()
## web services work best with json frame,
# So after the user post its data in json format we need to access the body of json.

    prediction = predict_single(applicant,  model)
    


    result = {
## we need to cast numpy float type to python native float type
    'loan_approval_probability': float(prediction)
## same as the line above, casting the value using bool method


        }
## send back the data in json format to the user
    return jsonify(result)

if __name__ == "__main__":
    serve(app, host='0.0.0.0', port=9696)


