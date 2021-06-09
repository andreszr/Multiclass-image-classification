import json
import numpy as np
import os
import pickle
import joblib
import base64
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
from PIL import Image


def init():
    global model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It's the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION).
    # For multiple models, it points to the folder containing all deployed models (./azureml-models).
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'mic_model.h5')
    model = tf.keras.models.load_model(model_path,custom_objects={'KerasLayer':hub.KerasLayer})
    print(model.summary())

def run(raw_data):
    # data = np.array(json.loads(raw_data)['data'])
    img = base64.b64decode(raw_data)
    # Make prediction.
    y_hat = model.predict(image)
    return json.dumps({"predicted_class": y_hat})