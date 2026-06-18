import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load trained model
model = tf.keras.models.load_model(
    "google_street_view_model.keras"
)

st.title("Google Street View Recognition System")

uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    img = np.array(image)

    img = cv2.resize(
        img,
        (128,128)
    )

    img = img / 255.0

    img = np.expand_dims(
        img,
        axis=0
    )

    prediction = model.predict(img)

    st.subheader("Prediction Result")

    st.write(prediction)
