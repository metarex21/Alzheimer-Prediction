import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the saved model
model = tf.keras.models.load_model('model.h5')

# Define the class names
class_names = {
    1: 'Mild Demented',
    2: 'Moderate Demented',
    3: 'Non Demented',
    4: 'Very Mild Demented'
}

img_height = 220
img_width = 220

# Function to preprocess the image
def preprocess_image(image):
    # Resize the image to match the input size of the model
    image = image.resize((img_width, img_height))
    # Convert the image to numpy array
    image_array = np.array(image)
    # Preprocess the image (normalize, etc.)
    image_array = image_array / 255.0  # Assuming the model expects input in the range [0, 1]
    # Expand dimensions to create a batch of size 1
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Streamlit app header
st.title("Alzheimer's Prediction")

# Add a space to fill name
name = st.text_input("Enter your name:")

# Add a space to upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])



# Predict button
if st.button("Predict"):
    if uploaded_file is None:
        st.warning("Please upload an image.")
    else:
        # Read the uploaded image
        image = Image.open(uploaded_file)
        
        # Preprocess the image
        preprocessed_image = preprocess_image(image)
        
        # Predict the class probabilities
        predictions = model.predict(preprocessed_image)
        
        # Get the predicted class index
        predicted_class_index = np.argmax(predictions[0])
        
        # Get the predicted class label
        predicted_class_label = class_names[predicted_class_index]
        
        # Display the predicted class label
        st.success(f"Predicted Class: {predicted_class_label}")
