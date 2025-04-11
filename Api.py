from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import os
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from werkzeug.utils import secure_filename
import json

app = Flask(__name__)
CORS(app)

# Load Model
model = load_model('best_skin_problem_model.h5')

# Load the dataset
data = pd.read_csv('Face_Packs.csv', encoding='ISO-8859-1') 

# Preprocessing function
def load_and_preprocess_image(img_path, img_width=224, img_height=224):
    img = load_img(img_path, target_size=(img_width, img_height))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# Get treatment recommendations
def get_treatment_info(problem, exclude_ingredient=None):
    result = data[data['Problem'].str.contains(problem, case=False, na=False)]


    if not result.empty:
        treatments_to_display = [] 
        for _, row in result.iterrows():
            ingredients_list = [ingredient.strip().lower() for ingredient in row['Ingredients'].split(',')]

            # Exclude treatments
            if exclude_ingredient and any(exclude_ingredient.lower() in ingredient for ingredient in ingredients_list):
                continue  


            treatments_to_display.append({
                'Treatment Pack': row['Treatment pack'],
                'Ingredients': row['Ingredients'],
                'How to do': row['How_to']
            })

        return treatments_to_display
    else:
        return []

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join('uploads', filename)
            file.save(file_path)

            # Preprocess 
            preprocessed_image = load_and_preprocess_image(file_path)

            # Predict 
            prediction = model.predict(preprocessed_image)
            predicted_class_index = np.argmax(prediction, axis=1)[0]

            # Fetch class names
            dataset_path = "Dataset"  
            class_labels = sorted(os.listdir(dataset_path))  
            predicted_label = class_labels[predicted_class_index] if predicted_class_index < len(class_labels) else 'Unknown'
            recommendations = get_treatment_info(predicted_label)
            
            print(recommendations)

            response = {
                'predicted_class': predicted_label,
                'product_photo': f'uploads/{filename}',
                'recommendations': recommendations
            }

            return jsonify(response), 200

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': 'An error occurred while processing your request.'}), 500

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
