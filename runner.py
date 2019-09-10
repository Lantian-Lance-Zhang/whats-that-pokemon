from flask import Flask, render_template, request, redirect
import os
from time import sleep
import tensorflow as tf
from PIL import Image
import numpy as np
from datetime import datetime
import shelve
from glob import glob
from random import randrange
import warnings
warnings.simplefilter(action='ignore', category=Warning)

app = Flask(__name__)
# stats = shelve.open("/static/databases/stat")
# if stats["runs"]:
#     stats["runs"] += 1
# else:
#     stats["runs"] = 1
file_path = "default"

@app.route("/")
@app.route("/home")
def home():
	return render_template("home.html")

@app.route("/before_upload")
def before_upload():
    return render_template("before_upload.html")

@app.route("/options", methods=["GET","POST"])
def options():
    global file_path

    file = request.files["inputFile"]
    filename = file.filename
    print("Received file --> " + filename)

    if filename.split(".")[-1] in ["png","jpg","jpeg"]:
        new_fn = "input" + str(datetime.now()) + ".png"
        new_path = os.path.join("static/uploads", new_fn)
        print("Assigned path --> " + new_path)
        file.save(new_path) #can use some improvements here
        file_path = new_path
    else:
        print("False data type --> " + filename.split(".")[-1])
        return render_template("upload_file.html", warning=True)
    return render_template("options.html", filename=file_path)

@app.route("/upload_file", methods=["GET","POST"])
def upload_file():
    return render_template("upload_file.html")

@app.route("/upload", methods=["GET","POST"])
def upload():
    #requesting filepath from page (Solution to threading issue)
    file_path = request.form["filename"]
    print("Filepath --> " + file_path)

    #requesting model_id from options.html
    model_id = request.form["model_id"]
    if len(model_id) > 0:
        if model_id.lower() not in [i.split("/")[-1][:-3].lower() for i in glob("static/models/*.h5")]:
            if model_id.lower() != "default":
                print(model_id)
                print([i.split("/")[-1][:-3].lower() for i in glob("static/models/*.h5")])
                return render_template("options.html", warning=True)

    #creating model_name according to model_id
    if len(model_id) > 0:
        if len(model_id) > 1:
            model_id = model_id.lower()
        else:
            model_id = model_id.upper()
        print("Received model_id --> " + model_id + "\nLength = " + str(len(model_id)))
        model_name = model_id + ".h5"
    else:
        print("Received model_id --> DEFAULT")
        model_id = "default"

    #loading and resizing a image
    print("Loading image from " + file_path)
    image = Image.open(file_path)
    image = image.resize((128,128))
    image_array = np.array([np.asarray(image, dtype="float64")[:,:,:3] / 255.]).reshape((-1,128,128,3))
    print("Processing complete\nLoading model --> " + model_id)

    if model_id != "default":
        #loading a model and making a prediction
        model = tf.keras.models.load_model(os.path.join("static/models",model_name))
        prediction = list(model.predict(image_array)[0])

        #interpreting the prediction based on model_id
        model_keys = shelve.open("static/databases/dictionaries")
        prediction = prediction.index(max(prediction))
        if len(model_id) > 1:
            prediction = model_keys[model_id.lower()][prediction]
        else:
            prediction = model_keys[model_id.upper()][prediction]         
    else:
        prediction = [0,0] #prediction index, prediction strength
        for i in range(1,4):
            model = tf.keras.models.load_model(os.path.join("static/models","default" + str(i) + ".h5"))
            subPrediction = list(model.predict(image_array)[0])
            preditionStrength = max(subPrediction)
            if preditionStrength >= prediction[1]:
                prediction[0] = prediction.index(max(prediction))
                prediction[1] = preditionStrength
                model_id = "default{}".format(str(i))
        prediction = prediction[0]
        model_keys = shelve.open("static/databases/dictionaries")
        prediction = model_keys[model_id.lower()][prediction]
    
    print("Prediction --> " + prediction)

    #choosing a sample image of the pokemon & processing it
    target_dir = os.path.join("static/train", prediction.title())
    sample_dir = os.listdir(target_dir)
    sample_dir = os.path.join(target_dir, sample_dir[randrange(0,10)])
    print("Sample image --> " + sample_dir)

    #saving data for future analysis
    result_path = os.path.join("static/results","{} {} {}.png".format(prediction, model_id, str(datetime.now())))
    print("Saving image to --> " + result_path)
    image.save(result_path)

    return render_template("result.html", result=prediction, file_path=result_path, example=sample_dir)

if __name__ == "__main__":
    print("Initializing...")
    app.run(debug=True, host='0.0.0.0', threaded=True)
    #gunicorn -b 0.0.0.0:5000 -w 5 runner:app
