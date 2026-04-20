import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# 1. Load the model
@st.cache_resource
def load_model():
    # Make sure 'your_model_name.keras' matches your filename
    return tf.keras.models.load_model('Character_Recognization.keras')

model = load_model()

# 2. Setup Labels (Make sure these are in alphabetical order!)
# Based on your code, this should be 0-9 then A-Z
class_names = ['0','1','2','3','4','5','6','7','8','9',
               'A','B','C','D','E','F','G','H','I','J','K','L','M',
               'N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

st.set_page_config(page_title="AI OCR Predictor", layout="centered")
st.title("🔠 Character Recognition AI")
st.write("Upload an image of a single letter or number.")

# 3. File Upload
uploaded_file = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # Open and show image
    raw_img = Image.open(uploaded_file)
    st.image(raw_img, caption="Target Image", width=200)

    # 4. Preprocessing
    # Convert PIL to OpenCV (RGB)
    img = np.array(raw_img.convert('RGB'))
    
    # Resize to 32x32 to match your model's input
    resized = cv2.resize(img, (32, 32))
    
    # Normalize (Match your 1/255.0 training logic)
    normalized = resized.astype('float32') / 255.0
    
    # Add batch dimension (1, 32, 32, 3)
    input_data = np.expand_dims(normalized, axis=0)

    # 5. Predict
    if st.button('Predict Character'):
        prediction = model.predict(input_data)
        result_index = np.argmax(prediction)
        confidence = np.max(prediction)

        # 6. Display Result
        st.divider()
        st.header(f"Result: {class_names[result_index]}")
        st.write(f"Confidence Score: {confidence*100:.2f}%")
        
        # Show probability bar chart
        st.bar_chart(prediction[0])