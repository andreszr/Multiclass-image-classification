import json
import numpy as np
import os
import pickle
import joblib
import base64
from keras.models import load_model
import tensorflow as tf

def init():
    global model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It's the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION).
    # For multiple models, it points to the folder containing all deployed models (./azureml-models).
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'mic_model.h5')
    model = load_model(model_path)
    # model = tf.keras.models.load_model(model_path)
    

def run(raw_data):
    # data = np.array(json.loads(raw_data)['data'])
    data = base64.decodestring(raw_data)
    # Make prediction.
    y_hat = model.predict(data)
    # You can return any data type as long as it's JSON-serializable.
    # setosa_clases = ['Setosa', 'Versicolor', 'Virginica']
    # return the result back
    return json.dumps({"predicted_class": y_hat})