import requests
import pickle
from google.cloud import storage
import numpy as np
from flask import Flask, request
from sklearn.linear_model import LogisticRegression
from markupsafe import escape
import os
from google.cloud import storage

app = Flask(__name__)

def open_pickle_file(bucket_name, file_name):
    # Create a client and specify the bucket name
    client = storage.Client()

    # Get the bucket
    bucket = client.bucket(bucket_name)

    # Get the blob (file) from the bucket
    blob = bucket.blob(file_name)

    # Download the blob to a temporary file
    temp_file_path = "/tmp/temp_file.pickle"
    blob.download_to_filename(temp_file_path)

    # Read the contents of the temporary file
    with open(temp_file_path, "rb") as file:
        data = file.read()

    return data

#bucket_name = 'model-bucket-iris3'
#file_name = 'flower-v1.pkl'
bucket_name = os.environ.get('GCS_BUCKET_NAME')
file_name = os.environ.get('GCS_FILE_NAME')
model_pk = open_pickle_file(bucket_name, file_name)
model = pickle.loads(model_pk)


@app.route('/api_predict', methods = ['POST','GET'])
def api_predict():
    if request.method == 'GET':
        return "Please Send POST Request"
    elif request.method == 'POST':
        
        print("Hello" + str(request.get_json()))
        
        data = request.get_json()
        
        sepal_length = data["sepal_length"]
        sepal_width = data["sepal_width"]
        petal_length = data["petal_length"]
        petal_width = data["petal_width"]
    
        data = np.array([[sepal_length, sepal_width, 
                          petal_length, petal_width]])
        
        prediction = model.predict(data)

        return str(prediction)        

if __name__ == "__main__":
    app.run()

