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
 
 ```