from flask import Flask, render_template, request, redirect
import os, shelve
from time import sleep
import tensorflow as tf
from PIL import Image
import numpy as np
from datetime import datetime
from glob import glob
from random import randrange
from glob import glob
app = Flask(__name__)
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
    if filename.split(".")[-1] in ["png","jpg"]:
        new_fn = "input" + str(datetime.now()) + ".png"
        new_path = os.path.join("static/uploads", new_fn)
        file.save(new_path) #can use some improvements here
        file_path = new_path
    else:
        return render_template("upload_file.html", warning=True)
    return render_template("options.html")
@app.route("/upload_file", methods=["GET","POST"])
def upload_file():
    return render_template("upload_file.html")
@app.route("/upload", methods=["GET","POST"])
def upload():
    global file_path
    model_id = request.form["model_id"]
    if model_id not in [i.split("/")[-1][:-3] for i in glob("static/models/*.h5")]:
            return render_template("options.html", warning=True)
    if len(model_id) > 0:
        model_name = model_id + ".h5"
    else:
        model_id = "default"
    image = Image.open(file_path)
    image = image.resize((128,128))
    image_array = np.array([np.asarray(image, dtype="float64")[:,:,:3] / 255.]).reshape((-1,128,128,3))
    model = tf.keras.models.load_model(os.path.join("static/models",model_name))
    prediction = list(model.predict(image_array)[0])
    model_keys = shelve.open("static/databases/dictionaries")
    prediction = prediction.index(max(prediction))
    if model_id == "default":
        prediction = "default"
    elif len(model_id) > 1:
        prediction = model_keys[model_id.lower()][prediction]
    else:
        prediction = model_keys[model_id.upper()][prediction]
    target_dir = os.path.join("static/train", prediction.title())
    sample_dir = glob(os.path.join(target_dir,"*.jpg"))[randrange(0, 25)]
    sample = Image.open(sample_dir)
    sample = sample.resize((500,500))
    sample.save(sample_dir)
    result_path = os.path.join("static/results","{} {} {}.png".format(prediction, model_id, str(datetime.now())))
    image.save(result_path)
    return render_template("result.html", result=prediction, file_path=result_path, example=sample_dir)
if __name__ == "__main__":
    print("Initializing...")
    app.run(debug=True)