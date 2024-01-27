import streamlit as st
import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

st.title("Brain Tumor Classification App")

# Load the model
model = load_model("C:\\Users\\trilo\\OneDrive\\Desktop\\Trishala\\Phishing\\BrainTumor Classification DL\\BrainTumor10EpochsCategorical.h5")

# Function to predict the class
def predict_class(image):
    img = Image.open(image).convert('RGB')
    img = img.resize((64, 64))
    img = np.array(img)
    input_img = np.expand_dims(img, axis=0)

    # Use the predict method instead of predict_classes
    predictions = model.predict(input_img)

    # Extract the class with the highest probability
    class_index = np.argmax(predictions)

    # Map the class index to the corresponding label
    if class_index == 0:
        return "No Tumor"
    elif class_index == 1:
        return "Positive"

# Streamlit UI
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Predict the class
    result = predict_class(uploaded_file)

    # Display the result
    st.write(f"Prediction: {result}")
