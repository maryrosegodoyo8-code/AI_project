import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# 1. Page Configuration
st.set_page_config(page_title="AI Image Classifier", layout="centered")

# 2. Load the Model
with st.spinner('Loading model...'):
    # Note: Ensure your saved file 'best_model.h5' is actually a MobileNetV2 based model
    model = load_model('best_model.h5')

# 3. UI Header
st.title("🐱 vs 🐶 Image Classifier")
# CHANGED: Updated text to mention MobileNetV2
st.write("Upload an image to classify it using MobileNetV2.")

# 4. File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess
    # MobileNetV2 typically uses 224x224 input size
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Note: If your MobileNetV2 model was trained with tf.keras.applications.mobilenet_v2.preprocess_input,
    # you might need to change the normalization logic below to match that specific preprocessing.
    # Keeping /255.0 here as per your original code.
    img_array /= 255.0 
    
    # Predict
    prediction = model.predict(img_array)
    confidence = prediction[0][0]
    
    # Interpret Result
    # Assuming 0 = Cat, 1 = Dog
    if confidence > 0.5:
        label = "Dog"
        score = confidence * 100
    else:
        label = "Cat"
        score = (1 - confidence) * 100
        
    st.success(f"Prediction: **{label}**")
    st.write(f"Confidence: {score:.2f}%")
