import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# trained model
model = load_model('best_skin_problem_model.h5')

# labels
dataset_path = "Dataset" 
class_labels = sorted(os.listdir(dataset_path)) 

print(f"Detected Class Labels: {class_labels}")

# preprocess the image
def preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size) 
    img_array = image.img_to_array(img) 
    img_array = np.expand_dims(img_array, axis=0) 
    img_array /= 255.0 
    return img_array

# Function to make predictions
def predict_image(img_path):
    img_array = preprocess_image(img_path)

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class = class_labels[predicted_class_index]
    confidence = np.max(predictions)

    return predicted_class, confidence


if __name__ == "__main__":

    img_path = 'Samples/abc.jpg' 

    if not os.path.exists(img_path):
        print(f"Error: Image file '{img_path}' not found.")
    else:
        predicted_class, confidence = predict_image(img_path)
        print(f"Predicted Class: {predicted_class}")
        print(f"Confidence: {confidence:.2f}")
