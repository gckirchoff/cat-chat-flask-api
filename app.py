
from flask import Flask, flash, request, redirect, url_for, send_from_directory, jsonify
import urllib.request
import sys
import datetime
import re
import os
from werkzeug.utils import secure_filename
from flask_cors import CORS, cross_origin

from math import expm1
import tensorflow_hub as hub
import tensorflow as tf

# import importlib
# tf = importlib.import_module("tensorflow-cpu")

import boto3, botocore


app = Flask(__name__)
CORS(app)

S3_KEY_ID = "----"
S3_SECRET_KEY = "----"
S3_BUCKET = "----"
S3_LOCATION = "https://{}.s3.us-east-2.amazonaws.com/".format(S3_BUCKET)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

UPLOAD_FOLDER = './static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.secret_key = "secret key"
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

s3 = boto3.client(
    "s3",
    aws_access_key_id=S3_KEY_ID,
    aws_secret_access_key=S3_SECRET_KEY
)


def upload_file_to_s3(file, filename, bucket_name, acl="public-read"):
    try:

        s3.upload_fileobj(
            file,
            bucket_name,
            filename,
            ExtraArgs={
                "ACL": acl,
                "ContentType": file.content_type
            }
        )

    except Exception as e:
        print("Something bad happened: ", e)
        return e

    return "{}{}".format(S3_LOCATION, filename)


def load_model(model_path):
    """
    Loads a saved, trained model from a specified path.
    """

    print(f"Loading saved model from: {model_path}...")
    model = tf.keras.models.load_model(model_path, custom_objects={"KerasLayer": hub.KerasLayer})

    return model


# Define image size
IMG_SIZE = 224


# Function for preprocessing images
def process_image(image_path, img_size=IMG_SIZE):
    """
    Takes an image file path and turns that image into a Tensor.
    """

    # Read image file
    image = tf.io.read_file(image_path)

    # Turn the jpg image into numeric Tensor with 3 color channels
    image = tf.image.decode_jpeg(image, channels=3)

    # Convert the color channel values from 0-255 to 0-1 values.
    image = tf.image.convert_image_dtype(image, tf.float32)

    # Resize the image to (244, 244)
    image = tf.image.resize(image, size=[img_size, img_size])

    return image


BATCH_SIZE = 32


def create_data_batches(X, y=None, batch_size=BATCH_SIZE, valid_data=False, test_data=False):
    """
    Creates batches of data out of image (X) and label (y) pairs.

    Shuffles training data, but not validation data

    Also accepts test data as input (no labels).
    """

    print("Creating test data batches...")
    data = tf.data.Dataset.from_tensor_slices(
        (tf.constant(X)))  # This turns all of X into a dataset. Need to make a batch size of 32
    data_batch = data.map(process_image).batch(batch_size)
    return data_batch


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


cat_classification_model = load_model('20210721-122146-full-image-set-mobilenetv3-Adam.h5')
# data = create_data_batches(test_image)
# preds = cat_classification_model.predict(data)
# print(preds[0][0])


@app.route('/wakeup', methods=['GET'])
@cross_origin()
def wakeup():
    return "wake up, heroku server!"



@app.route('/', methods=['GET', 'POST', 'DELETE'])
@cross_origin()
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            extension = re.search("\.(jpeg|jpg|png)$", filename).group()
            filename = filename.replace(extension, datetime.datetime.now().strftime("%Y%m%d%H:%M:%S") + extension)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            file.stream.seek(0)
            file_path = f"static/uploads/{filename}"
            data = create_data_batches([file_path])
            prediction = cat_classification_model.predict(data)[0][0]
            if prediction > 0.8:
                s3_url = upload_file_to_s3(file, filename, S3_BUCKET)
                file.close()
                return jsonify(
                    status="success",
                    # url=url_for('download_file', name=filename),
                    url=s3_url,
                    goodness=str(prediction)
                )
            else:
                os.remove((os.path.join(app.config['UPLOAD_FOLDER'], filename)))
                return jsonify(
                    status="error",
                    goodness=str(prediction)
                )
    if request.method == 'DELETE':
        url = request.json['imageUrl']
        filename = url.split("/")[-1]
        os.remove((os.path.join(app.config['UPLOAD_FOLDER'], filename)))
        s3.delete_object(Bucket=S3_BUCKET, Key=filename)
            # file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # return redirect(url_for('download_file', name=filename))
            # return url_for('download_file', name=filename)
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''


@app.route('/uploads/<name>')
def download_file(name):
    return send_from_directory(app.config["UPLOAD_FOLDER"], name)


if __name__ == "__main__":
    app.run()
