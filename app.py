from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

#Initialize Flask app
app = Flask(__name__)

#Load the trained model
MODEL_PATH = 'model save/banana_classifier_vgg16.h5'  # Path to saved model
model = load_model(MODEL_PATH)

#Define class labels (must match the class order during training)
#class_labels = ['ambul_kesel_artificial', 'ambul_kesel_natural', 'anamal_natural', 'anamalu_artificial']
class_labels=['anamal_natural','anamalu_artificial','ambul_kesel_artificial','ambul_kesel_natural']

#Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image file is provided
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    # Check if the file is an image
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Save the uploaded file temporarily
        file_path = os.path.join('uploads', file.filename)
        os.makedirs('uploads', exist_ok=True)
        file.save(file_path)

        # Preprocess the image
        img = load_img(file_path, target_size=(224, 224))  # Resize image to match model input
        img_array = img_to_array(img) / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        #  Make prediction
        predictions = model.predict(img_array)
        predicted_class = class_labels[np.argmax(predictions)]  # Get the class with the highest score
        confidence = np.max(predictions)  # Get the confidence score

        #Remove the saved file after prediction
        os.remove(file_path)

        #Return the result as JSON
        return jsonify({
            'predicted_class': predicted_class,
            'confidence': float(confidence)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


#Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)