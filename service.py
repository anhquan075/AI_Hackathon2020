from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np 
import time
import logging
import traceback
import os
import io
import requests
import random
import time
import json
from time import gmtime, strftime
import tensorflow as tf
import keras 
from keras.models import model_from_json

from utils import get_config, load_class_names
from src import predict_efficientNetB8, full_flow, predict_efficientNetB7

# setup config
cfg = get_config()
cfg.merge_from_file('configs/service.yaml')
cfg.merge_from_file('configs/rcode.yaml')

#Load model
with open(EfficientNetB7_recognition.json, 'r') as json_file:
    model_json = json_file.read()

# Load weights
B7_WEIGHTS = "model/B7_100epoch_new.h5"
B7_MODEL = model_from_json(model_json)
B7_MODEL.load_weights(B7_WEIGHTS)

# Load weights
MODEL_PATH = cfg.SERVICE.MODEL_PATH
NUMBER_CLASS = cfg.SERVICE.NUMBER_CLASS
BATCH_SIZE = cfg.SERVICE.BATCH_SIZE

# config service
HOST = cfg.SERVICE.SERVICE_IP
PORT = cfg.SERVICE.SERVICE_PORT
LOG_PATH = cfg.SERVICE.LOG_PATH
RCODE = cfg.RCODE
DATA_CK = cfg.SERVICE.DATA_CK_DIR
PATH_DATA_CK = cfg.SERVICE.PATH_DATA_CK

# create labels
LABELS = load_class_names(cfg.SERVICE.CLASS_PATH)

# create logging
if not os.path.exists(LOG_PATH):
    os.mkdir(LOG_PATH)
logging.basicConfig(filename=os.path.join(LOG_PATH, str(time.time())+".log"), filemode="w", level=logging.DEBUG, format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
console = logging.StreamHandler()
console.setLevel(logging.ERROR)
logging.getLogger("").addHandler(console)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder="templates", static_folder="statics")

@app.route('/home')
@app.route('/')
def view_home():
    return render_template('home.html')

@app.route('/hackathon')
def hello_world():
    return render_template('index.html')

@app.route('/predict_test', methods=['POST'])
def predict_test():
    try:
        input_dir = "hackathon_test"
        # if not os.path.exists(input_dir):
        #     os.mkdir(input_dir)
        file = request.files['file']
        image_file = file.read()
        image = cv2.imdecode(np.frombuffer(image_file, dtype=np.uint8), -1)
        # image_path = os.path.join(input_dir, 'test.jpg')
        # cv2.imwrite(image_path, image)

        # get result
        result = predict_efficientNetB7(image, B7_MODEL, LABELS)

        return result
    except Exception as e:
        logger.error(str(e))
        print(e)
        logger.error(str(traceback.print_exc()))
        
        result = {"code": "1001"}
        return result


@app.route('/predict', methods=['GET'])
def predict_image():
    try:
        input_dir = request.args.get('subdir')
        fold = request.args.get('id_fold')
        # image_file = file.read()
        # image = cv2.imdecode(np.frombuffer(image_file, dtype=np.uint8), -1)

        # get result
        t = time.time()
        out_sub_path, out_sub_path_vid = full_flow(input_dir, fold)
        my_time = time.time() - t
        result = {"code": "1000", "out_sub_path": out_sub_path, "out_sub_path_vid": out_sub_path_vid, "my_time": my_time}
        return result
    except Exception as e:
        logger.error(str(e))
        logger.error(str(traceback.print_exc()))
        
        result = {"code": "1001"}
        return result

@app.route('/save_zip', methods=['GET'])
def savezip():
    try:
        mode = request.args.get('id_file')
        os.system("gdown --id {}".format(mode))
        result = {'code': '1000', 'status': "down file done!"}

        return result
    except Exception as e:
        print(str(e))
        print(str(traceback.print_exc()))
        result = {'code': '609', 'status': RCODE.code_609}
        return result

@app.route('/unzip', methods=['GET'])
def unzip_file():
    try:
        filename = request.args.get('filename')
        id_fold = request.args.get('id_fold')
        with open("log.txt", "a+") as f:
            f.write(id_fold)
        os.system("unzip {}.zip".format(filename))
        subdir = os.path.join(DATA_CK, str(id_fold))
        os.system("mv {} {}".format(filename, subdir))
        os.system("rm -rf {}.zip".format(filename))

        result = {'code': '1000', 'subdir': subdir}

        return result
    except Exception as e:
        print(str(e))
        print(str(traceback.print_exc()))
        result = {'code': '609', 'status': RCODE.code_609, 'subdir': 'None' }
        return result
   
if __name__ == "__main__":
    input_dir = "hackathon_test"
    # if not os.path.exists(input_dir):
    #     os.mkdir(input_dir)

    if not os.path.exists(LOG_PATH):
        os.mkdir(LOG_PATH)

    # if not os.path.exists(DATA_CK):
    #     os.mkdir(DATA_CK)
    
    # if not os.path.exists(PATH_DATA_CK):
    #     os.mkdir(PATH_DATA_CK)

    app.run(debug=False, host=HOST, port=PORT)