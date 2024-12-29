from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io
from PIL import Image

app = Flask(__name__)

# Load your pre-trained model
model = load_model('cancer_detection_model.h5')

def prepare_image(img_bytes):
    # Convert the uploaded image bytes into a PIL image
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    # Resize the image to match the input size expected by your model
    img = img.resize((150, 150))  # Adjust size based on your model input size
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Cast the image array to float32 before dividing
    img_array = img_array.astype(np.float32)
    # Normalize pixel values to be between 0 and 1
    img_array /= 255.0  # Adjust based on your model preprocessing
    # Expand dimensions to match the input shape expected by the model
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image was uploaded
    if 'image' not in request.files:
        return redirect(url_for('home'))

    file = request.files['image']
    if file.filename == '':
        return redirect(url_for('home'))

    if file:
        # Read the image file directly into memory (no saving to disk)
        img_bytes = file.read()

        # Prepare the image for prediction
        img_array = prepare_image(img_bytes)

        # Use the model to predict the class
        prediction = model.predict(img_array)
        cancer_probability = prediction[0][0] * 100  # Convert to percentage

        # Corrected threshold for Cancer/No Cancer
        if cancer_probability >= 50:
            result = " No Cancer"
        else:
            result = "Cancer"

        # Render the results on a new page
        return render_template('results.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
