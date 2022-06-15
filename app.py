# from __future__ import division, print_function
import csv
from flask import Flask, render_template,request,redirect,url_for, flash
import diseaseprediction
import joblib
from PIL import Image as pil_image
import numpy as np
import tensorflow as tf
import random
import os
import re
from flask import send_from_directory
from keras.preprocessing import image
from keras.models import model_from_json
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.compat.v1 import ConfigProto
from keras.models import load_model
import cv2
import json
import sys
from glob import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
from skimage import io
from gevent.pywsgi import WSGIServer
from werkzeug.utils import secure_filename
from forms import RegistrationForm,LoginForm,ContactForm
app=Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

########################################################Skin Cancer#########################################################################
 
########################################################covid model###########################################################################################
model222=load_model("my_model.h5")
def api1(full_path):
    data = image.load_img(full_path, target_size=(64, 64, 3))
    data = np.expand_dims(data, axis=0)
    data = data * 1.0 / 255

    #with graph.as_default():
    predict = model222.predict(data)
    return predict

@app.route('/upload11', methods=['POST','GET'])
def upload11_file():
    table=["Pneumonia", "Covid-19", "Normal"]
    generator=random.choice(table)
    if request.method == 'GET':
        return render_template('covid.html')
    else:
        try:
            file = request.files['image']
            full_name = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(full_name)
            indices = {0: 'Normal', 1: 'Pneumonia'}
            result = api1(full_name)
            if(result>50):
                label= indices[1]
                accuracy= result
            else:
                label= indices[0]
                accuracy= 100-result
            return render_template('covid_predict.html', image_file_name = file.filename, label = label, accuracy = accuracy, generator=generator)
        except:
            flash("Please select the image first !!", "danger")      
            return redirect(url_for("Pneumonia"))


@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)
#####################################################FOR THE BRAIN TUMOR MODEL###############################################################
# Classification model
classification_model = load_model('model_classification.h5')

# Segmentation model 
def dice_coef(y_true, y_pred):
    y_truef=K.flatten(y_true)
    y_predf=K.flatten(y_pred)
    And=K.sum(y_truef* y_predf)
    return((2* And + smooth) / (K.sum(y_truef) + K.sum(y_predf) + smooth))

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def iou(y_true, y_pred):
    intersection = K.sum(y_true * y_pred)
    sum_ = K.sum(y_true + y_pred)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac

def jac_distance(y_true, y_pred):
    y_truef=K.flatten(y_true)
    y_predf=K.flatten(y_pred)

    return - iou(y_true, y_pred)

segmentation_model = load_model('model_segmentation.h5',custom_objects={'dice_coef':dice_coef,'jac_distance':jac_distance,'dice_coef_loss': dice_coef_loss,"iou":iou})

classification_model.make_predict_function()

def predict_label(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img ,(256,256))
    img = img.reshape(1,256,256,3)
    img = np.array(img)
    pred1 = classification_model.predict(img)
    pred1 = np.argmax(pred1,axis=1)
#https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
    
    if pred1 == 0:
        return 'Glioma'
    elif pred1 == 1:
        return "Meningioma"
    elif pred1 == 2:
        return "No Tumour"
    
    return "Pituitary"

def predict_segmentation_mask(image_path):
    # reads an brain MRI image
    img = io.imread(image_path)
    img = cv2.resize(img,(256,256))
    img = np.array(img, dtype=np.float64)
    img -= img.mean()
    img /= img.std()
    #img = np.reshape(img, (1,256,256,3) # this is the shape our model expects
    X = np.empty((1,256,256,3))
    X[0,] = img
    predict = segmentation_model.predict(X)

    return predict.reshape(256,256)


@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']
        img_path = "Brain Dataset/" + img.filename 
        #img.save(img_path)

        p = predict_label(img_path)

        predicted_mask = predict_segmentation_mask(img_path)
        original_img = cv2.imread(img_path)
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(7, 5))
        plt.axis('off')
        axes[0].imshow(original_img)
        axes[0].get_xaxis().set_visible(False)
        axes[0].get_yaxis().set_visible(False)
        axes[1].imshow(predicted_mask)
        axes[1].get_xaxis().set_visible(False)
        axes[1].get_yaxis().set_visible(False)

        fig.tight_layout()
        

        seg_path = "static/seg_images/" + img.filename 
        plt.savefig(seg_path)

    return render_template("brain.html", prediction = p,seg_path=seg_path)

#####################################################FOR THE MALERIA MODEL###############################################################
dir_path = os.path.dirname(os.path.realpath(__file__))
app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
model = load_model('model111.h5') #malaria model
# call model to predict an image
def api(full_path):
    data = image.load_img(full_path, target_size=(50, 50, 3))
    data = np.expand_dims(data, axis=0)
    data = data * 1.0 / 255

    #with graph.as_default():
    predicted = model.predict(data)
    return predicted
# procesing uploaded file and predict it
@app.route('/upload', methods=['POST','GET'])
def upload_file():

    if request.method == 'GET':
        return render_template('maleria.html')
    else:
        try:
            file = request.files['image']
            full_name = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(full_name)

            indices = {0: 'PARASITIC', 1: 'Uninfected', 2: 'Invasive carcinomar', 3: 'Normal'}
            result = api(full_name)
            print(result)

            predicted_class = np.asscalar(np.argmax(result, axis=1))
            accuracy = round(result[0][predicted_class] * 100, 2)
            label = indices[predicted_class]
            return render_template('maleria_predict.html', image_file_name = file.filename, label = label, accuracy = accuracy)
        except:
            flash("Please select the image first !!", "danger")      
            return redirect(url_for("Malaria"))    
@app.route("/maleria")
def maleria():
    return render_template('maleria.html')      

######################routes for default home page#########################################################################################
@app.route('/', methods=['GET'])
def home():
        return render_template('index.html', symptoms=symptoms)
################################routes for common disease prediction#########################################################################
with open('dataset/Testing.csv', newline='') as f:
        reader = csv.reader(f)
        symptoms = next(reader)
        symptoms = symptoms[:len(symptoms)-1]
@app.route('/disease', methods=['POST', 'GET'])
def disease_predict():
    selected_symptoms = []
    if(request.form['Symptom1']!="") and (request.form['Symptom1'] not in selected_symptoms):
        selected_symptoms.append(request.form['Symptom1'])
    if(request.form['Symptom2']!="") and (request.form['Symptom2'] not in selected_symptoms):
        selected_symptoms.append(request.form['Symptom2'])
    if(request.form['Symptom3']!="") and (request.form['Symptom3'] not in selected_symptoms):
        selected_symptoms.append(request.form['Symptom3'])
    if(request.form['Symptom4']!="") and (request.form['Symptom4'] not in selected_symptoms):
        selected_symptoms.append(request.form['Symptom4'])
    if(request.form['Symptom5']!="") and (request.form['Symptom5'] not in selected_symptoms):
        selected_symptoms.append(request.form['Symptom5'])
    disease = diseaseprediction.dosomething(selected_symptoms)
    return render_template('disease.html',disease=disease,symptoms=symptoms)

#####################################################routes for diabetes###########################################
@app.route("/skin")
def skin():
    return render_template('skin.html')

#routes for all the front end page features 
@app.route("/about")
def about_page():
    return render_template('about.html')

@app.route("/models")
def service_page():
    return render_template('models.html')

@app.route("/contact", methods=['GET', 'POST'])
def contact():
    form = ContactForm()
    if form.validate_on_submit():
        flash(f'Message Sent {form.name.data}!', 'success')
        return redirect(url_for('home'))
    return render_template('contact.html',form=form)

@app.route("/brain")
def brain_page():
    return render_template('brain.html')

@app.route("/hospitals")
def hospitals_page():
    return render_template('hospitals.html')

@app.route("/covid")
def review_page():
    return render_template('covid.html')

@app.route("/404")
def error_page():
    return render_template('404.html')

@app.route("/register", methods=['GET', 'POST'])
def register():
    form= RegistrationForm()
    if form.validate_on_submit():
        flash(f'Account created for {form.username.data}!', 'success')
        return redirect(url_for('home'))
    return render_template('register.html',title='Register', form=form)

@app.route("/login", methods=['GET', 'POST'])
def login():
    form= LoginForm()
    if form.validate_on_submit():
        if form.email.data=='admin@blog.com' and form.password.data=='password':
            flash('You have been logged in!','success')
            return redirect(url_for('home'))
        else:
            flash('Login Unsuccessfull','danger')  
    return render_template('login.html',title='Login', form=form)      


if __name__ == '__main__':
    app.run(debug=True)
