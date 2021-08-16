import timeit
import traceback
import time
import os
import logging
import requests
import rcode
import random
import string
import cv2
import csv
import json
import base64
import threading
#import uwsgi
import numpy as np
from datetime import date
from datetime import datetime
from flask import Flask, request, jsonify,render_template, Response, send_from_directory, send_file
from flask_restful import Resource, Api
from flask_cors import CORS
from configparser import SafeConfigParser
from PIL import Image, ImageDraw, ImageFont
import cusutil
sem = threading.Semaphore()

app = Flask(__name__, static_url_path='')

CORS(app)
api = Api(app)
#######################################
#####LOAD CONFIG####
config = SafeConfigParser()
config.read("config/web.cfg")
LOG_PATH = str(config.get('main', 'LOG_PATH'))
print("LOG_PATH", LOG_PATH)
SERVER_IP = str(config.get('main', 'SERVER_IP'))
print("SERVER_IP", SERVER_IP)
SERVER_PORT = int(config.get('main', 'SERVER_PORT'))
print("SERVER_PORT", SERVER_PORT)
#######################################
#####CREATE LOGGER#####
logging.basicConfig(filename=os.path.join(LOG_PATH, str(time.time())+".log"), filemode="w", level=logging.DEBUG, format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
console = logging.StreamHandler()
console.setLevel(logging.ERROR)
logging.getLogger("").addHandler(console)
logger = logging.getLogger(__name__)
#######################################
print("service ready")  
######################################
def process_data(path):
    with open("data/matching.txt") as f:
        c = f.readlines()
    matchinglist = {}
    for line in c:
        ipath, lpath = {}
#####WEBSITE#####
@app.route('/')
def index():
    print("load_index")
    data = {'num_page': 0, 'pagefile': []}
    return render_template('index.html', data=data)
@app.route('/thumbnailimg')
def thumbnailimg():
    print("load_iddoc")
    pagefile = []
    index = int(request.args.get('index'))
    if index == None:
        index = 0
    imgperindex = 100
    
    imgpath = request.args.get('imgpath').replace("/home/mmlab/", "/host/")
    labelpath = request.args.get('labelpath').replace("/home/mmlab/", "/host/")
    print("imgpath", imgpath)
    print("labelpath", labelpath)
    pagefile = []
    filelist = os.listdir(imgpath)
    if len(filelist)-1 > index+imgperindex:
        page_filelist = filelist[index*imgperindex:index*imgperindex+imgperindex]
    else:
        page_filelist = filelist[index*imgperindex:len(filelist)]
    for fname in page_filelist:
        fname_ext = ''.join(fname.split(".")[:-1])
        if not os.path.exists(os.path.join(labelpath, fname_ext+".txt")):
            pagefile.append({'imgpath': imgpath, 'labelpath': labelpath, 'fname': fname, 'fname_ext': fname_ext, 'boxes': ""})
            continue
        with open(os.path.join(labelpath, fname_ext+".txt"), encoding="utf-8") as f:
            c = f.readlines()
        boxes = []
        boxes_str = ""
        for line in c:
            print(line.rstrip().split(",")[:4])
            if "," in line:
                x1,y1,x2,y2 = [int(float(x)) for x in line.rstrip().split(",")[:4]]   
                cls = str(line.rstrip().split(",")[4])
                boxes.append((x1,y1,x2,y2,cls))
                boxes_str += str(x1)+","+str(y1)+","+str(x2)+","+str(y2)+","+str(cls)+";"
            else:
                x1,y1,x2,y2 = [int(float(x)) for x in line.rstrip().split(" ")[:4]]    
                cls = str(line.rstrip().split(" ")[4])
                boxes.append((x1,y1,x2,y2,cls))
                boxes_str += str(x1)+","+str(y1)+","+str(x2)+","+str(y2)+","+str(cls)+";"
        pagefile.append({'imgpath': imgpath, 'labelpath': labelpath, 'fname': fname, 'fname_ext': fname_ext, 'boxes': boxes_str})
    data = {'num_page': int(len(filelist)/imgperindex)+1, 'pagefile': pagefile}
    return render_template('index.html', data=data)
@app.route('/singleimg')
def singleimg():
    fname = request.args.get('fname')
    imgpath = request.args.get('imgpath')
    labelpath = request.args.get('labelpath')
    labelpath = labelpath.replace("/home/mmlab/", "/host/")
    fname_ext = ''.join(fname.split(".")[:-1])
    with open(os.path.join(labelpath, fname_ext+".txt"), encoding="utf-8") as f:
        c = f.readlines()
    boxes = []
    boxes_str = ""
    for line in c:
        if "," in line:
            x1,y1,x2,y2 = [int(float(x)) for x in line.rstrip().split(",")[:4]]  
            cls = str(line.rstrip().split(",")[4])
            boxes.append((x1,y1,x2,y2,cls))
            boxes_str += str(x1)+","+str(y1)+","+str(x2)+","+str(y2)+","+str(cls)+";"  
        else:
            x1,y1,x2,y2 = [int(float(x)) for x in line.rstrip().split(" ")[:4]]  
            cls = str(line.rstrip().split(" ")[4])
            boxes.append((x1,y1,x2,y2,cls))
            boxes_str += str(x1)+","+str(y1)+","+str(x2)+","+str(y2)+","+str(cls)+";"
    
    data = {'imgpath': imgpath, 'labelpath': labelpath, 'fname': fname, 'fname_ext': fname_ext, 'boxes': boxes_str}
    return render_template('single.html', data=data)
######################################
@app.route('/get_img_aic')
def get_img_aic():
#    print("get_img")
    fpath = request.args.get('fpath')
    fpath = "/home/mmlab/aic/img/"+fpath
    if os.path.exists(fpath):
        img = cv2.imread(fpath)
    else:
        img = cv2.imread("./static/images/404.jpg")
#    y,x,_ = img.shape
#    ratio = 750/x
#    img = cv2.resize(img, (750, int(y*ratio)))
    ret, jpeg = cv2.imencode('.jpg', img)
    return  Response((b'--frame\r\n'
                     b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tostring() + b'\r\n\r\n'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
#######################################
@app.route('/get_img')
def get_img():
#    print("get_img")
    fpath = request.args.get('fpath')
    boxes_str = request.args.get('boxes')
    if os.path.exists(fpath):
        img = cv2.imread(fpath)
    else:
        img = cv2.imread("./static/images/404.jpg")
    colors = {}
    font_style = ImageFont.truetype("./TimesNewRoman400.ttf", 20)
    if not boxes_str is None:
        boxes = boxes_str.split(";")[:-1]
#        colors = cusutil.random_list_color(10)
        for box in boxes:
            x1,y1,x2,y2 = [int(float(x)) for x in box.split(",")[:4]]    
            cls = str(box.split(",")[4])
            if not cls in colors.keys():
                colors = cusutil.gen_list_color(colors, cls)
            img = cv2.rectangle(img, (x1,y1), (x2,y2), colors[cls], 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
#            cv2.putText(img, str(cls), (x1,y1-10), font, 1, colors[cls], 2, cv2.LINE_AA)
            im_pil = Image.fromarray(img)
            draw = ImageDraw.Draw(im_pil)
            draw.text((x1, y1-25), str(cls), font=font_style, fill=tuple(colors[cls]))
            img = np.asarray(im_pil)
            
    y,x,_ = img.shape
#    ratio = 750/x
#    img = cv2.resize(img, (750, int(y*ratio)))
    ret, jpeg = cv2.imencode('.jpg', img)
    return  Response((b'--frame\r\n'
                     b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tostring() + b'\r\n\r\n'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
#######################################
#######################################
#######################################
@app.route('/aic')
def aic():
    try:
        print("load_iddoc")
        pagefile = []
        index = request.args.get('index')
        show_cls = request.args.get('cls')
        if index == None:
            index = 0
        else:
            index = int(index)
        imgperindex = 100
        
        imgpath = request.args.get('imgpath')
        labelpath = request.args.get('labelpath')
        
        if imgpath == None:
            imgpath = "/host/aic/img/"+imgpath
        else:
            imgpath = imgpath
        if labelpath == None:
            labelpath = "/host/aic/label/"
        else:
            labelpath = labelpath
        if show_cls == None:
            show_cls = -1
        else:
            show_cls = int(show_cls)
        pagefile = []
        print(imgpath, os.path.exists(os.path.join(labelpath, "visual.txt")))
        with open(os.path.join(labelpath, "visual.txt")) as f:
            c = f.readlines()
        
        array = [[] for x in range(8)]
        for line in c:
            line = line.rstrip()
            fname, cls, score = line.split()[0:3]
            array[int(cls)].append((fname,cls,score))
        filelist = None
        if int(show_cls) in [0,1,2,3,4,5,6,7]:
            array[show_cls].sort(key=lambda x:x[2], reverse=False)
            filelist = array[show_cls]
        elif int(show_cls) == -2:
            filelist = []
            for data in array[0]:
#                print(data[0])
                if not os.path.exists(os.path.join(imgpath, data[0])):
                    print(data[0])
                    continue
                try:
                    im = Image.open(os.path.join(imgpath, data[0]))
                    width, height = im.size
                    ratio = [100/677, 1024/683, 960/720, 1080/720, 1000/750, 1912/1080, 512/288, 426/240, 500/300, 459/306, 500/313, 640/360, 660/371, 600/400, 660/410, 640/427, 660/440, 665/449, 500/450, 800/450, 656/480, 653/490, 660/495]
                    if width/height in ratio:
                        filelist.append(data)
                except:
                    True
            filelist.sort(key=lambda x:x[2])
        else:
            filelist = []
            for i in range(8):
                array[i].sort(key=lambda x:x[2], reverse=False)
                filelist.extend(array[i])
        
        if len(filelist)-1 > index+imgperindex:
            page_filelist = filelist[index*imgperindex:index*imgperindex+imgperindex]
        else:
            page_filelist = filelist[index*imgperindex:len(filelist)]

        for info in page_filelist:
            fname = info[0]
            fname_ext = ''.join(fname.split(".")[:-1])
            pagefile.append({'imgpath': imgpath, 'labelpath': labelpath, 'fname': fname, 'fname_ext': fname_ext, 'boxes': "", 'cls': info[1], 'score': round(float(info[2]),3)})
        print("len(filelist)", len(filelist))
        if len(filelist) % imgperindex == 0:
            data = {'num_page': int(len(filelist)/imgperindex), 'pagefile': pagefile}
        else:
            data = {'num_page': int(len(filelist)/imgperindex)+1, 'pagefile': pagefile}
        return render_template('aic.html', data=data)
    except Exception as e:
        print(e)
        return render_template('aic.html', data={'num_page': 0, 'pagefile': []})
@app.route('/aic_getlabel')
def aic_getlabel():
    try:
        fname = request.args.get('fname')
        labelpath = request.args.get('labelpath')
        labelpath = labelpath
        open(os.path.join(labelpath, "visual.txt"), 'a').close()
        with open(os.path.join(labelpath, "visual.txt"), "r", encoding="utf-8") as f:
            c = f.readlines()
        label_dict = {}
        for line in c:
            line = line.rstrip()
            fn, label, score = line.split()[0:3]
            label_dict[fn] = [fn, label, score]
        cls = "-1"
        score = "-1.5"
        fn = fname
        if fname in label_dict.keys():
            cls = label_dict[fname][1]
            fn = fname
            score = label_dict[fname][2]
        else:
            print(fname)
        return jsonify({"cls": cls, "fname": fn, "score": score})
    except Exception as e:
        print(e)
        return jsonify({"cls": "", "fname": "", "score": ""})
@app.route('/aic_setlabel', methods=["POST"])
def aic_setlabel():
    try:
        if request.method != 'POST':
            return {};
        fname = request.json['fname']
        label = request.json['label']
        labelpath = request.json['labelpath']
        labelpath = labelpath
        print("labelpath", labelpath)
        myfile = None
        while (myfile is None):
            try:
                myfile = open(os.path.join(labelpath, "submit.txt"), "r+")
            except IOError:
                print ("Could not open file! Please close Excel!")
        
        open(os.path.join(labelpath, "submit.txt"), 'a').close()
        with open(os.path.join(labelpath, "submit.txt"), "r", encoding="utf-8") as f:
            c = f.readlines()
        with open(os.path.join(labelpath, "submit.txt"), "w", encoding="utf-8") as f:
            flag = False
            for line in c:
                print(line)
                if not fname in line:
                    f.write(line)
                else:
                    f.write(fname+"\t"+label+"\n")
                    flag = True
            if flag == False:
                f.write(fname+"\t"+label+"\n")
        ########
        myfile = None
        while (myfile is None):
            try:
                myfile = open(os.path.join(labelpath, "visual.txt"), "r+")
            except IOError:
                print ("Could not open file! Please close Excel!")
        
        open(os.path.join(labelpath, "visual.txt"), 'a').close()
        with open(os.path.join(labelpath, "visual.txt"), "r", encoding="utf-8") as f:
            c = f.readlines()
        with open(os.path.join(labelpath, "visual.txt"), "w", encoding="utf-8") as f:
            flag = False
            for line in c:
                if not fname in line:
                    f.write(line)
                else:
                    ofname, ocls, oscore = line.split()[0:3]
                    f.write(fname+" "+label+" 1"+"\n")
                    flag = True
            if flag == False:
                f.write(fname+" "+label+" 1"+"\n")
        ########
        return {"cls": label}
    except Exception as e:
        print(e)
        return {"cls": ""}

@app.route('/dowload_submit_file', methods=['GET'])
def dowload_submit_file():
    print("dowload_submit_file")
    # = request.args.get('labelpath')
    imgpath = request.args.get('imgpath')
    fpath = os.path.join(imgpath, "submit.txt")
    print("fpath", fpath)
    return send_file(fpath, as_attachment=True)
#######################################
if __name__ == '__main__':
     app.run(host=SERVER_IP, port=SERVER_PORT, debug=True, threaded=True)
 
























