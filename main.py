import os
from app import app
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from model import create_model
import numpy as np
import operator
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
        
@app.route('/')
def upload_form():
        return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_image():
        if 'file' not in request.files:
                flash('No file part')
                return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
                flash('No image selected for uploading')
                return redirect(request.url)
        if file and allowed_file(file.filename):

                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

                image_file = app.config["UPLOAD_FOLDER"] + filename

                class_labels = ['safe_driving', 'texting_right', 'talking_on_phone_right', 'texting_left', 'talking_on_phone_left','operating_radio', 'drinking', 'reaching_behind', 'doing_hair_makeup', 'talking_to_passanger']
            
                model = create_model()
                model.load_weights("_weights.h5")
                model.compile(loss='categorical_crossentropy',
                              optimizer='rmsprop',
                              metrics=['accuracy'])

                target_size=(150,150)

                image = load_img(image_file, target_size=target_size)

                image_arr = img_to_array(image)

                image_arr = np.expand_dims(image_arr, axis=0)
                image_arr /= 255

                predictions = model.predict(image_arr)

                decoded_predictions = dict(zip(class_labels, predictions[0]))

                decoded_predictions = sorted(decoded_predictions.items(), key=operator.itemgetter(1), reverse=True)

                print()
                count = 1
                for key, value in decoded_predictions[:1]:
                    word_split = key.split("_")
                    if len(word_split) == 3:
                        prediction = "The person in the photo is " + word_split[0] + " " + word_split[1] + " " + word_split[2]
                    elif len(word_split) == 4:
                        prediction = "The person in the photo is " + word_split[0] + " " + word_split[1] + " " + word_split[2] + " " + word_split[3]
                    elif len(word_split) == 2:
                        prediction = "The person in the photo is " + word_split[0] + " " + word_split[1]
                    else:
                        prediction = "The prediction cannot be made for this image, please upload a new one"
                
                return render_template('upload.html', filename=filename, prediction=prediction)
        else:
                flash('Allowed image types are -> png, jpg, jpeg, gif')
                return redirect(request.url)

@app.route("/uploads/<filename>")
def display_image(filename=''):
        
        from flask import send_from_directory

        return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

if __name__ == "__main__":
    app.run()
