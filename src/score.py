import json
import numpy as np
import os
import pickle
import joblib

def init():
    global model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It's the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION).
    # For multiple models, it points to the folder containing all deployed models (./azureml-models).
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'wine_model.pkl')
    model = joblib.load(model_path)

def run(raw_data):
    data = np.array(json.loads(raw_data)['data'])
    # Make prediction.
    y_hat = model.predict(data)

    classes = ['class_0', 'class_1', 'class_2']
    results = [classes[int(y)] for y in y_hat]
    # You can return any data type as long as it's JSON-serializable.
    # return results
    return json.dumps({"predicted_class": results})

#     """
#     testing in postman
#     {
#     "data":[
#         [5.1, 3.5, 1.4, 0.2, 5.1, 3.5, 1.4, 0.2, 5.1, 3.5, 1.4, 0.2, 5.1],
#         [3.1, 2.5, 5.4, 1.2, 3.5, 1.4, 0.2, 5.1, 3.5, 1.4, 0.2, 5.1, 2.3]
#         ]
#     } 
#     """