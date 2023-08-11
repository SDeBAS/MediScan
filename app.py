
from flask import Flask, render_template, request
import pickle
import jinja2
import model
import keras_ocr

app = Flask(__name__)


@app.route('/')
def index():
    return render_template("index.html")

@app.route('/team')
def team():
    return render_template("team.html")

@app.route('/upload')
def upload():
    return render_template("upload.html")

def process_file(file_path):
    # Load the model paths
    craft_model_path = 'C:/Users/basud/.keras-ocr/craft_mlt_25k.h5'
    crnn_model_path = 'C:/Users/basud/.keras-ocr/crnn_kurapan.h5'

    # Load the image
    image = keras_ocr.tools.read(file_path)

    # Create the OCR pipeline
    detector = keras_ocr.detection.Detector()
    recognizer = keras_ocr.recognition.Recognizer()
    pipeline = keras_ocr.pipeline.Pipeline(detector=detector, recognizer=recognizer)

    # Perform OCR on the image
    prediction_groups = pipeline.recognize(images=[image])

    # Process the OCR results
    result = []
    for predictions in prediction_groups:
        for word_info in predictions:
            word = word_info[0]
            result.append(word)

    return result


@app.route('/upload_file', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']
    print(file)
    if file.filename == '':
        return "No selected file"

    file_path = f"uploads/{file.filename}"
    file.save(file_path)

    result = process_file(file_path)  # Call your model function

    return render_template('results.html', result=result)
    if 'image' not in request.files:
        return "No file part"

    image = request.files['image']

    if image.filename == '':
        return "No selected file"

    # Here, you can process the uploaded image, save it, or perform any desired action

    return "Image uploaded successfully"




if __name__ == "__main__":
    app.run(debug=True)