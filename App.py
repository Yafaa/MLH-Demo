from __future__ import division, print_function
import cv2
import numpy as np
import scipy.ndimage
import csv
import time
from datetime import date
import argparse
import csv
from datetime import date
import sys
import os 
import glob
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer 
import cv2
import numpy as np
from tensorflow import keras
from keras.models import Sequential
from tensorflow.keras.models import load_model
#from keras.models import load_model
from keras.utils import to_categorical
from keras.applications import imagenet_utils
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorflow.keras import layers, models
from PIL import Image
import imutils
from utils import detect_lp
from os.path import splitext,basename
from keras.models import model_from_json
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.mobilenet_v2 import preprocess_input
import glob
import imutils
import h5py
# remove warning message
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
app = Flask(__name__)

def load_modell(path):
    try:
        path = splitext(path)[0]
        with open('%s.json' % path, 'r') as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json, custom_objects={})
        model.load_weights('%s.h5' % path)
        return model
    except Exception as e:
        print(e)


def preprocess_image(image_path,resize=False):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    if resize:
        img = cv2.resize(img, (224,224))
    return img


def get_plate(image_path):
    Dmax = 608
    Dmin = 288
    # Loading model for plate detection
    wpod_net_path = "wpod-net.json"
    wpod_net = load_modell(wpod_net_path)
    
    vehicle = preprocess_image(image_path)
    ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)
    _ , LpImg, _, cor = detect_lp(wpod_net, vehicle, bound_dim, lp_threshold=0.5)
    return vehicle, LpImg, cor

# Create sort_contours() function to grab the contour of each digit from left to right
def sort_contours(cnts,reverse = False):
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                      key=lambda b: b[1][i], reverse=reverse))
    return cnts


def org(pl):
    if pl[3] == 10:
        pl[3] = ' TUN '
    elif pl[2] == 10:
        pl[2] = ' TUN '
    elif len(pl) == 6:
        pl.insert(2, ' TUN ')
    elif len(pl) > 6:
        pl.insert(3, ' TUN ')


def model_predict(img_path):  
    # Loading model for digits regognition
    model = load_model(r'C:\Users\Lenovo\Desktop\AI Session\Intern DataEra\App GUI v2\Intern\fullpy\dig_rec_3.h5')

    # Input : Car Image
    test_image_path = img_path
    vehicle, LpImg,cor = get_plate(test_image_path)
    if (len(LpImg)): #check if there is at least one license image
        plate_image = cv2.convertScaleAbs(LpImg[0], alpha=(255.0))

    # Segmentation
    (h, w, d) = plate_image.shape
    ratio = w/h
    img = cv2.resize(plate_image,(int(100*ratio),100))
    (h, w, d) = plate_image.shape
    img = cv2.cvtColor(plate_image,cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (3,3), 0)
    BLACK = [0, 0, 0]
    img = cv2.copyMakeBorder(img, 7, 7, 7, 7, cv2.BORDER_CONSTANT, value=BLACK) 
    img2 = img.copy()
    img2 = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)[1]
    img2 = cv2.erode(img2, None, iterations=1)
    cnts = cv2.findContours(img2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    img3 = img2.copy()
    img3 = cv2.cvtColor(img3, cv2.COLOR_GRAY2BGR)
    rec = []



    for c in sort_contours(cnts):
        (cx, cy, cw, ch) = cv2.boundingRect(c)
        #if ((cw >= w*0.01) and (cw <= w*0.5) and (ch >= h*0.2)):
        if ((cw <= w*0.5) and (ch >= h*0.37) and (cw < ch) and (ch < cw*5)):
            ROI = img3[cy:cy+ch, cx:cx+cw]
            ROI=ROI[:,:,0]
            # Resizing that digit to (18, 18)
            resized_digit = cv2.resize(ROI, (18,18))
            # Padding the digit with 5 pixels of black color (zeros) in each side to finally          produce the image of (28, 28)
            padded_digit = np.pad(resized_digit, ((5,5),(5,5)), "constant", constant_values=0)
            #cv2.imwrite('diiiig_{}.png'.format(i), ROI)
            cv2.rectangle(img3, (cx,cy), (cx+cw,cy+ch), (0,255,0), 2)
            #normalize image
            normalizedImg = cv2.normalize(padded_digit, padded_digit, 0, 255, cv2.NORM_MINMAX)
            rec.append(normalizedImg)

    inpp = np.array(rec)


    pl = []
    for digit in rec:
        prediction = model.predict(digit.reshape(1, 28, 28, 1))
        pl.append(np.argmax(prediction))

    org(pl)
            
    plt = ''.join(str(x) for x in pl)
    
    return plt


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
      
        # Make prediction
        try :
            preds = model_predict(file_path)
        except :
            Message = "Invalid Image"   
            return Message
        else:
            today = date.today()
            t = time.localtime()
            current_time = time.strftime("%H:%M:%S", t)
            with open('plates.csv','a',newline='') as f:
                write1 = csv.writer(f)
                write1.writerow([today,current_time,os.path.basename(file_path),preds])
            return preds
     
    return None


if __name__ == '__main__':
    #app.run(debug=True,threaded=False)
    app.run()