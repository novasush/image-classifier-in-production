# Image Classifier in Production
This repository contains code for deploying keras model in production on model server and inference using GRPC or HTTP calls.

## Installation:

* Install model server
```bash
echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | sudo tee /etc/apt/sources.list.d/tensorflow-serving.list && \
curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | sudo apt-key add -

sudo apt-get update

sudo apt-get install tensorflow-model-server
```
* setup model dir structure

Make sure your model dir structure is as follows:

```
cat_dog_classifier/
    1/
     saved_model.pb
     variables/
        variables.index
        ...
```

* Setting absolute model path for model server

Assuming your terminal location is in the same folder where model folder is present. 
```bash
export MODEL_DIR=$(pwd)/cat_dog_classifier
```

* Running tensorflow model server
```bash
tensorflow_model_server --port=8500 --rest_api_port=8501 --model_name=cat_dog_classifier --model_base_path="${MODEL_DIR}"
```

* Installing requirements 

(Open New Terminal in same directory)

```bash
pip install -r requirements.txt
```
 `Note: create a virtual environment for preventing dependency issues while installing libraries`

 # Inferencing with GRPC
 ```python
import grpc
import numpy as np
import nsvision as nv
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

label = ['cat', 'dog']
GRPC_MAX_RECEIVE_MESSAGE_LENGTH = 4096 * 4096 * 3
channel = grpc.insecure_channel('localhost:8500', options=[('grpc.max_receive_message_length', GRPC_MAX_RECEIVE_MESSAGE_LENGTH)])
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
grpc_request = predict_pb2.PredictRequest()
grpc_request.model_spec.name = 'cat_dog_classifier'
grpc_request.model_spec.signature_name = 'serving_default'

image = nv.imread('golden-retriever-royalty-free-image-506756303-1560962726.jpg',resize=(150,150),normalize=True)
image = nv.expand_dims(image,axis=0)
grpc_request.inputs['conv2d_input'].CopyFrom(tf.make_tensor_proto(image, shape=image.shape))
result = stub.Predict(grpc_request,10)
result = int(result.outputs['dense_1'].float_val[0])
print(label[result])
#This printed 'dog' on my console
 ```

 # Inferencing with REST API
 ```python
import json
import requests
import nsvision as nv

label = ['cat','dog']
image = nv.imread('cat.2033.jpg',resize=(150,150),normalize=True)
image = nv.expand_dims(image,axis=0)
data = json.dumps({ 
    "instances": image.tolist()
})
headers = {"content-type": "application/json"}

response = requests.post('http://localhost:8501/v1/models/cat_dog_classifier:predict', data=data, headers=headers)
result = int(response.json()['predictions'][0][0])
print(label[result])
#This printed 'cat' on my console
 ```