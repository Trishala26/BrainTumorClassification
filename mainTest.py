import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model = load_model('BrainTumor10EpochsCategorical.h5')

image = cv2.imread("C:\\Users\\trilo\\OneDrive\\Desktop\\Trishala\\Phishing\\BrainTumor Classification DL\\pred\\pred2.jpg")

img = Image.fromarray(image)
img = img.resize((64, 64))
img = np.array(img)

input_img = np.expand_dims(img, axis=0)

# Use the predict method instead of predict_classes
predictions = model.predict(input_img)

# Extract the class with the highest probability
class_index = np.argmax(predictions)

print("Predicted class index:", class_index)
