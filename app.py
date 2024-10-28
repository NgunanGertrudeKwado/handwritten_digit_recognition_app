import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image

# Load the pre-trained model
model = load_model('C:\\Users\\HP\\Documents\\handwritten_digit_recognition_app\\handwritten_digit_recognition_model.h5')

# Function to preprocess the image for the model
def preprocess_image(image):
    image = image.convert('L')  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28
    image_array = np.array(image)  # Convert to numpy array
    image_array = image_array.astype('float32') / 255.0  # Normalize
    image_array = image_array.reshape(1, 28, 28, 1)  # Reshape for model input
    return image_array

# Streamlit app layout
st.title("Handwritten Digit Recognition")
st.write("Upload a handwritten digit image to see the model's prediction.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image and make a prediction
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_digit = np.argmax(prediction)  # Get the digit with the highest probability
    
    # Display the prediction
    st.write(f"Predicted Digit: {predicted_digit}")
