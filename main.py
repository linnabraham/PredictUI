from flask import Flask, request, render_template, send_from_directory, url_for
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField
from flask_uploads import UploadSet, IMAGES, configure_uploads
from flask_wtf import FlaskForm
import pandas as pd

app = Flask(__name__)
app.config['SECRET_KEY'] = 'aslskjf'
app.config['UPLOADED_PHOTOS_DEST'] = 'uploads'

photos = UploadSet('photos',IMAGES)
configure_uploads(app, photos)

class UploadForm(FlaskForm):
    photo = FileField(
        validators=[
            FileAllowed(photos, 'Only images are allowed'),
            FileRequired('File field should not be empty') 
        ]
    )
    submit = SubmitField('Upload')

@app.route('/uploads/<filename>')
def get_file(filename):
    return send_from_directory(app.config['UPLOADED_PHOTOS_DEST'], filename)

@app.route('/', methods=['GET','POST'])
def upload_image():
    form = UploadForm()
    message = "Place holder text"
    if form.validate_on_submit():
        filename = photos.save(form.photo.data)
        global file_url
        file_url = url_for('get_file', filename=filename)
        print(file_url)
        print(filename)
        # #Loading CNN model
        # saved_model = 'saved_models/bestmodel.h5'
        # model = load_model(saved_model)
        # try:
        #     #Get image URL as input
        #     #image_url = request.form['image_url']
        #     x = load_img(file_url)
        #     #image = io.imread(image_url)
            
        #     #Apply same preprocessing used while training CNN model
        #     # image_small = st.resize(image, (32,32,3))
        #     # x = np.expand_dims(image_small.transpose(2, 0, 1), axis=0)
            
        #     #Call classify function to predict the image class using the loaded CNN model
        #     # final,pred_class = classify(x, model)
        #     # print(pred_class)
        #     # print(final)
        #     #preds = model.predict(x)
        #     preds = "Hey"
        #     #Store model prediction results to pass to the web page
        #     message = "Model prediction: {}".format(preds)
        #     print('Python module executed successfully')
            
        # except Exception as e:
        #     #Store error to pass to the web page
        #     message = "Error encountered. Try another image. ErrorClass: {}, Argument: {} and Traceback details are: {}".format(e.__class__,e.args,e.__doc__)
        #     final = pd.DataFrame({'A': ['Error'], 'B': [0]})
        

    else:
        file_url=None


    # return render_template('index.html', message=message, form=form, file_url=file_url)
    return render_template('index.html', form=form, file_url=file_url)

import glob
from PIL import Image
import matplotlib.pyplot as plt
from keras.models import load_model, model_from_json
import numpy as np
import os

def load_img(imgpath):
    img = Image.open(imgpath)
    npimg = np.array(img)
    x = npimg
    plt.imshow(x)
    x = np.expand_dims(npimg, axis=0)
    x = x* (1./255)
    return x

@app.route('/predict_object/', methods=['GET', 'POST'])
def render_message():
    #Loading CNN model
    saved_model = 'saved_models/bestmodel.h5'
    model = load_model(saved_model)
    print(file_url)
    try:
        image_url = request.form['image_url']

        x = load_img(image_url)

        preds = model.predict(x)
        final = preds[0]
        #Store model prediction results to pass to the web page
        message = "Model prediction: {}".format(preds)
        print('Python module executed successfully')
        
    except Exception as e:
        #Store error to pass to the web page
        message = "Error encountered. Try another image. ErrorClass: {}, Argument: {} and Traceback details are: {}".format(e.__class__,e.args,e.__doc__)
        final = pd.DataFrame({'A': ['Error'], 'B': [0]})
        
    #Return the model results to the web page
    return render_template('index.html',
                            message=message,
                            data=final.round(decimals=2),
                            image_url=file_url)


# def render_message():
#     #Loading CNN model
#     saved_model = 'saved_models/bestmodel.h5'
#     model = load_model(saved_model)
    
#     try:
#         #Get image URL as input
#         image_url = request.form['image_url']
#         x = load_img(image_url)
#         #image = io.imread(image_url)
        
#         #Apply same preprocessing used while training CNN model
#         # image_small = st.resize(image, (32,32,3))
#         # x = np.expand_dims(image_small.transpose(2, 0, 1), axis=0)
        
#         #Call classify function to predict the image class using the loaded CNN model
#         # final,pred_class = classify(x, model)
#         # print(pred_class)
#         # print(final)
#         preds = model.predict(x)
#         #Store model prediction results to pass to the web page
#         message = "Model prediction: {}".format(preds)
#         print('Python module executed successfully')
        
#     except Exception as e:
#         #Store error to pass to the web page
#         message = "Error encountered. Try another image. ErrorClass: {}, Argument: {} and Traceback details are: {}".format(e.__class__,e.args,e.__doc__)
#         final = pd.DataFrame({'A': ['Error'], 'B': [0]})
        
#     #Return the model results to the web page
#     return render_template('index.html',
#                             message=message,
#                             data=final.round(decimals=2),
#                             image_url=image_url)

if __name__ == "__main__":
    app.run(debug=True)