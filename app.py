from flask import Flask, render_template, request

import jsonify
import requests
import pickle
import numpy as np
import sklearn
import os
import cv2
import io
from PIL import Image
import matplotlib.pyplot as plt
from main import *
from sklearn.preprocessing import StandardScaler

from flask_caching import Cache


cache = Cache()

app = Flask(__name__)
app.config["CACHE_TYPE"] = "null"
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
cache.init_app(app)
#model = pickle.load(open('random_forest_regression_model.pkl', 'rb'))

app.config["IMAGE_UPLOADS"] = "/home/p302793/Dropbox/P302793/Private/Code/lndsat/static"
@app.route('/')
def Home():
    app.config["CACHE_TYPE"] = "null"
    cache.init_app(app)

    with app.app_context():
        cache.clear()

    return render_template('base.html', content="Testing")
    #return "Main"

#@app.route('/general')
#def home2():
#    return render_template('base.html', content='testing')

@app.route('/handle-form',methods=['GET','POST'])
def handle_form():


  app.config["CACHE_TYPE"] = "null"
  cache.init_app(app)
  #cache.init_app(app, config=your_cache_config)

  #with app.app_context():
  #    cache.clear()  
  if request.method=='POST':
      
      if request.files:
          image=request.files["image"]
          #print("**********************************************************")
          #print(image.filename)
          #image_r=image.read()
          #display_image(image_r)
          #image.filename="user_iploaded_image.jpg"
          image.save(os.path.join(app.config["IMAGE_UPLOADS"], "user_uploaded_image.jpg"))
          display_image(image.filename)
          print(" success")
    #file = request.files.get('image-file','')
    #afile= request.files['img-file']
    #afile.save(afile.filename)
    #img=file.read()
  return render_template('base2.html')
#standard_to = StandardScaler()



def display_image(img):
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print(img)
    my_main(img)
    #byteImgIO = io.BytesIO()
    #byteImg = Image.open(img)
    #byteImgIO.seek(0)
    #byteImg=byteImgIO.read()
    #print(byteImg)
    print("!!!!!!!!!!Success")
    #im=cv2.imread(img.strip().decode('utf-8'))
    #print("########################################")
    #print(im)
    #plt.imshow(im/255.0)
    
if __name__=="__main__":
    app.run(debug=True)
