from flask import Flask, render_template, request
import os
from time import sleep
from threading import Thread
TENSOR = True
if TENSOR:
    import tensorflow as tf
from PIL import Image
import numpy as np
from datetime import datetime

app = Flask(__name__)
PATH = os.path.dirname(os.path.abspath(__file__))
changed = False
result = "Ditto" #default result
filename = "Ditto.png" #default filename of image displayed on result page
custom_model = False
model_id = ''

@app.route("/")
@app.route("/home")
def home():
	return render_template("home.html")

@app.route("/result")
def show_result():
    return render_template("result.html", result="Ditto")

@app.route("/upload_file/", methods=["POST", "GET"])
def upload_file():
    return render_template("upload_file.html")

@app.route("/before_upload")
def before_upload():
	return render_template("before_upload_test.html")

@app.route("/upload", methods=["GET", "POST"])
def upload_image():
    global changed, result, PATH, filename

    uploaded_file = request.files["inputFile"]
    filename = "input " + str(datetime.now()) + ".png"

    if uploaded_file.filename.split(".")[-1] in ["png", "jpg"]:
        uploads_folder = os.path.join(PATH, "static/uploads")
        uploaded_file.save(os.path.join(uploads_folder, filename))
        changed = True
        while changed:
            sleep(0.5) #wait for CNN to predict
        print("Rendering file: " + filename + "\n")
        return render_template("result.html", result=result, filename=filename)
    else:
        return render_template("upload_file.html", warning=True)

def wait_upload(worker_id):
    global changed, PATH, result, filename

    base = 128
    if TENSOR:
        model_path = "static/models/18cat_99.h5"
        model = tf.keras.models.load_model(os.path.join(PATH, model_path))
    else:
        model = None
    y_dictionary = ['Venusaur', 'Bulbasaur', 'Vileplume', 'Wartortle', 'Weepinbell', 'Tentacruel', 'Vaporeon', 'Victreebel', 'Tentacool', 'Tangela', 'Slowbro', 'Shellder', 'Starmie', 'Seel', 'Staryu', 'Slowpoke', 'Seaking', 'Seadra']

    while True:
        if changed:
            path = os.path.join(os.path.join(PATH, "static/uploads/"), filename)
            print("\nPath = " + path)
            image = Image.open(path)
            image = image.resize((base, base))
            if image:
                print("Image loaded")
            image_array = np.asarray(image, dtype="float64")[:,:,:3]
            image_array = image_array / 255
            image_array = np.array([image_array]).reshape((-1,base,base,3))
            print("Dimensions adjusted")

            if custom_model:
                model = None
                model_path = os.path.join("static/models", model_id)
                model = tf.keras.models.load_model(os.path.join(PATH, model_path))
                if model_id == "fairy.h5":
                    prediction1 = "Clefable" if model.predict(image_array) > 0.5 else "Clefairy"
                    prediction2 = "Clefable" if prediction1 == "Clefairy" else "Clefairy"
            else:
                prediction = model.predict(image_array)
                max_index = list(np.where(prediction == prediction.max())[-1])[0]
                prediction1 = y_dictionary[max_index] #"Eevee" if prediction < 0.5 else "Zapdos"
                prediction = np.delete(prediction, max_index)
                second_largest_index = list(np.where(prediction == prediction.max())[0])[0]
                prediction2 = y_dictionary[second_largest_index]
            print("Prediction = " + str(prediction1))
            print("Second Guess = " + prediction2)
            result = str(prediction1)
            changed = False
        sleep(0.5)

if __name__ == "__main__":
    thread1 = Thread(target=wait_upload, args=(0,))
    thread1.start()
    app.run(debug=True)