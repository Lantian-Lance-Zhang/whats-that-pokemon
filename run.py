from flask import Flask, render_template, request, redirect
import os
from time import sleep
TENSOR = True
if TENSOR:
    import tensorflow as tf
from PIL import Image
import numpy as np
from datetime import datetime
import shelve
from glob import glob

app = Flask(__name__)
model_id = "default"

@app.route("/")
@app.route("/home")
def home():
	return render_template("home.html")

@app.route("/options", methods=["GET", "POST"])
def options():
    return render_template("options.html")

@app.route("/choose_file", methods=["GET", "POST"])
def choose_file():
    global model_id

    if len(model_id) > 0:
        model_id = request.form["model_id"]
        # print([i.split("/")[-1][:-3] for i in glob("static/models/*.h5")])
        # print(model_id)
        if model_id not in [i.split("/")[-1][:-3] for i in glob("static/models/*.h5")]:
            return render_template("options.html", warning=True)
    return render_template("upload_file.html", model_id=model_id)

@app.route("/skip")
def skip():
    return render_template("upload_file.html", model_id="18cat_93")

@app.route("/before_upload")
def before_upload():
    return render_template("before_upload.html")

@app.route("/upload/<model_id>", methods=["GET", "POST"])
def upload(model_id):
    upload = request.files["inputFile"]
    filename = "input" + str(datetime.now()) + ".png"
    model_name = model_id + ".h5"
    image_path = os.path.join("static/uploads", filename)
    base = 128

    if upload.filename.split(".")[-1] in ["png", "jpg"]:
        if model_id == "I.h5":
            prediction = "Ivysaur"
            return render_template("result.html", result=prediction, filename=filename)

        upload.save(image_path)
        print("Processing Image: " + filename)
        image = Image.open(image_path)
        image = image.resize((base,base))
        image.save(image_path)
        image_array = np.array([np.asarray(image, dtype="float64")[:,:,:3] / 255.]).reshape((-1,base,base,3))
        print("Image Processing Complete")

        print("Loading model: " + model_name)
        model = tf.keras.models.load_model(os.path.join("static/models", model_name))
        print("Loading Complete")

        prediction = model.predict(image_array)
        if model_id == "fairy":
            prediction = "Clefable" if prediction > 0.5 else "Clefairy"
        else:
            db = shelve.open("static/dictionaries")
            prediction = list(prediction.flatten()).index(prediction.flatten().max())
            print(list(db.keys()))
            prediction = db[str(model_id.title())][prediction]

        print("Prediction: " + str(prediction))
        import random
        target_dir = os.path.join("static/uploaded", prediction.title())
        while True:
            file = os.listdir(target_dir)[int(random.randrange(0, 10))]
            if file[:1] != ".":
                break
        example = os.path.join("http://127.0.0.1:5000/static/uploaded/" + prediction.title(), file)#int(random.randrange(0, 10))])
        print(example)

        return render_template("result.html", result=prediction, filename=filename, example=example)

    else:
        return render_template("upload_file.html", warning=True)

if __name__ == "__main__":
    app.run(debug=True)