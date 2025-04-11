import pandas as pd
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

 
data = pd.read_csv('Face_Packs.csv', encoding='ISO-8859-1')  


 
def get_treatment_info(problem, exclude_ingredient=None):
   
    result = data[data['Problem'].str.contains(problem, case=False, na=False)]

   
    if not result.empty:
        treatments_to_display = []  
        for _, row in result.iterrows():
            ingredients_list = [ingredient.strip().lower() for ingredient in row['Ingredients'].split(',')]

 
            if exclude_ingredient and any(exclude_ingredient.lower() in ingredient for ingredient in ingredients_list):
                continue  

 
            treatments_to_display.append({
                'Treatment Pack': row['Treatment pack'],
                'Ingredients': row['Ingredients'],
                'How to do': row['How_to']
            })

 
        if treatments_to_display:
            for treatment in treatments_to_display:
                print(f"Treatment Pack: {treatment['Treatment Pack']}")
                print(f"Ingredients: {treatment['Ingredients']}")
                print(f"How to do: {treatment['How to do']}")
                print('-' * 50)
        else:
            print(f"No treatments found for '{problem}' excluding '{exclude_ingredient}'.")
    else:
        print("No treatment found for the provided problem.")


# Load the trained model
model = load_model('best_skin_problem_model.h5')

 
dataset_path = "Dataset"  
class_labels = sorted(os.listdir(dataset_path))  

print(f"Detected Class Labels: {class_labels}")


 
def preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)   
    img_array = image.img_to_array(img)  
    img_array = np.expand_dims(img_array, axis=0)  
    img_array /= 255.0   
    return img_array


 
def predict_image(img_path):
 
    img_array = preprocess_image(img_path)

 
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class = class_labels[predicted_class_index]
    confidence = np.max(predictions)

    return predicted_class, confidence


 
if __name__ == "__main__":
    
    img_path = 'Samples/abc.jpg' 

 
    if not os.path.exists(img_path):
        print(f"Error: Image file '{img_path}' not found.")
    elif not img_path.lower().endswith(('.png', '.jpg', '.jpeg')):  
        print("Error: The file is not a valid image format (PNG, JPG, JPEG).")
    else:
        
        predicted_class, confidence = predict_image(img_path)
        print(f"Predicted Class: {predicted_class}")

 
        problem_input = predicted_class
        get_treatment_info(problem_input)  

 
        print("\nAlternative treatments without 'yogurt':")
        get_treatment_info(problem_input, exclude_ingredient="yogurt")
