from flask import Flask, render_template, request
import os
from time import sleep
TENSOR = True
if TENSOR:
    import tensorflow as tf
from PIL import Image
import numpy as np
from datetime import datetime

app = Flask(__name__)

@app.route("/")
@app.route("/home")
def home():
	return render_template("home.html")

@app.route("/before_upload")
def before_upload():
	return render_template("before_upload.html")

@app.route("/upload_file/", methods=["GET", "POST"])
def upload_file():
    return render_template("upload_file.html")

@app.route("/upload/<model_id>", methods=["GET", "POST"])
def upload(model_id):
    upload = request.files["inputFile"]
    filename = "input" + str(datetime.now()) + ".png"
    model_name = model_id + ".h5"
    image_path = os.path.join("static/uploads", filename)
    base = 128

    if upload.filename.split(".")[-1] in ["png", "jpg"]:
        upload.save(image_path)
        print("Processing Image: " + filename)
        image = Image.open(image_path)
        image = image.resize((base,base))
        image_array = np.array([np.asarray(image, dtype="float64")[:,:,:3] / 255.]).reshape((-1,base,base,3))
        print("Image Processing Complete")

        print("Loading model: " + model_name)
        model = tf.keras.models.load_model(os.path.join("static/models", model_name))
        print("Loading Complete")

        prediction = model.predict(image_array)
        if model_id == "fairy":
            prediction = "Clefable" if prediction > 0.5 else "Clefairy"
        print("Prediction: " + prediction)

        return render_template("result.html", result=prediction, filename=filename)

    else:
        return render_template("upload_file.html", warning=True)

if __name__ == "__main__":
    app.run(debug=True)