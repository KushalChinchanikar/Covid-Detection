# Importing required libraries
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from tensorflow.python import keras

import cv2
import os

app = Flask(__name__) # Flask instance with current file i.e app.py

app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024 # max size of file made as 10 MB

ALLOWED_EXTENSIONS=['png','jpg','jpeg']
def allowed_file(filename):
    return '.' in filename and \
    filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

def init():
    global graph
    #graph= tf.get_default_graph()
    

def read_image(filename):
    img= load_img(filename, target_size=(180,180)) # Load the image
    img= img_to_array(img)
    img= img.reshape(1,180,180,3)
    img= img.astype('float32')
    img= img / 181.0
    return img

@app.route("/", methods=['GET','POST'])
def home():
    return render_template('home.html')


@app.route("/predict", methods= ['GET','POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        
        if file and allowed_file(file.filename):
            filename = file.filename
            file_path= os.path.join('Static/Images', filename)
            file.save(file_path)
            # img = read_image(file_path)
            img=cv2.imread(file_path)
            img_height,img_width=180,180
            img_resized=cv2.resize(img,(img_height,img_width))
            img=np.expand_dims(img_resized,axis=0)
            #with graph.as_default:
            
            # model1= load_model('resnetCovidClassifier\saved_model.pb')
            # model1= keras.models.load_model("resnetCovidClassifier")
            # model1= tf.keras.models.load_model('resnetCovidClassifier')
            model1= load_model('resnetCovidClassifier.h5')
            # class_prediction = model1.predict_classes(img)
            predict_x=model1.predict(img) 
            #class_prediction =np.argmax(predict_x,axis=1)
            
            #print(class_prediction)

            if predict_x<0.5:
                product = "Covid"
            elif predict_x>0.5 and predict_x<1:
                product = "Normal"
            else: 
                product="Error"

            #if class_prediction == 0:
            #    product = "Covid"
            #elif class_prediction[0] == 1:
            #    product = "Normal"
            #else: 
            #    product="Error"
            #return "Maybe Working" working for now
            return render_template("predict.html", title="predict", product= product, user_img= file_path)
        #except Exception as e:
        #    return(e)
        # except Exception as e:
        #     return "Unable to read the file. Check file extension."
    #return "Not going in try "
    return render_template("predict.html", title="predict")

if __name__=="__main__":
    init()
    app.debug = True
    app.run()
    




