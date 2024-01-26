import streamlit as st
import numpy as np
import cv
import pickle

# Load the trained model using pickle
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define a function to make predictions on images
def predict(image):
    img = cv.resize(image, (128, 128))
    img = np.array(img).reshape(-1, 128, 128, 3) / 255.0
    pred = model.predict(img)
    return "With Mask" if pred > 0.5 else "Without Mask"

# Create a Streamlit app
st.set_page_config(page_title="Face Mask Detection", page_icon=":mask:", layout="wide")
st.title('Face Mask Detection')
st.write('Upload an image to detect whether a person is wearing a face mask or not')

# Allow the user to upload an image
uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

# Make a prediction if an image is uploaded
if uploaded_file:
    image = cv.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    prediction = predict(image)
    st.write('Prediction:', prediction)