import streamlit as st
import numpy as np
import cv2
from PIL import Image

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
        (128, 128)
    )

    img = img / 255.0

    img = np.expand_dims(
        img,
        axis=0
    )

    st.subheader("Prediction Result")

    st.success("Image processed successfully!")

    st.write("Shape:", img.shape)
