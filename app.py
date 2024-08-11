from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import io
import librosa

app = Flask(__name__)

# Load your model
cough_model = load_model('cough_model.keras')
breath_model = load_model('breath_model.keras')

duration = 5
sr = 22050

def normalize_audio(input_file, n_mels=128):
    y, orig_sr = librosa.load(input_file)
    target_length = int(sr * duration)
    if len(y) > target_length:
        y = y[:target_length]
    else:
        y = np.pad(y, (0, max(0, target_length - len(y))), "constant")
    
    # Normalize the audio
    y = librosa.util.normalize(y)
    
    # Extract features (e.g., MFCCs)
    mfccs = librosa.feature.mfcc(y=y, sr = sr, n_mfcc=13)
    return mfccs

@app.route('/cough_predict', methods=['POST'])
def cough_predict():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    # If the user does not select a file, the browser submits an empty file without a filename
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Read the file content into a Pandas DataFrame (assuming it's a CSV file)
    try:
        file_content = io.BytesIO(file.read())
        normalized_data = normalize_audio(file_content)
        normalized_data = normalized_data[np.newaxis, ..., np.newaxis]
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

    # Make predictions
    predictions = cough_model.predict(normalized_data)
    
    return jsonify({'predictions': predictions.tolist()})

@app.route('/breath_predict', methods=['POST'])
def breath_predict():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    # If the user does not select a file, the browser submits an empty file without a filename
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Read the file content into a Pandas DataFrame (assuming it's a CSV file)
    try:
        file_content = io.BytesIO(file.read())
        normalized_data = normalize_audio(file_content)
        normalized_data = normalized_data[np.newaxis, ..., np.newaxis]
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

    # Make predictions
    predictions = cough_model.predict(normalized_data)
    
    return jsonify({'predictions': predictions.tolist()})

if __name__ == '__main__':
    app.run(debug=True)