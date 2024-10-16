import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load your pre-trained model (only once, not every time the script reloads)
model = load_model(r'C:\Users\dhani\OneDrive\Documents\background_remove\background_remove\background_removal_model1.h5')

def remove_background(input_image):
    # Convert image read by PIL to numpy array and then to color (RGB)
    input_image_rgb = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)
    input_image_resized = cv2.resize(input_image_rgb, (256, 256))
    input_image_normalized = input_image_resized / 255.0

    # Add batch dimension
    input_image_normalized = np.expand_dims(input_image_normalized, axis=0)

    # Predict the mask
    predicted_mask = model.predict(input_image_normalized)[0]
    predicted_mask_resized = cv2.resize(predicted_mask, (input_image_rgb.shape[1], input_image_rgb.shape[0]))
    binary_mask = (predicted_mask_resized > 0.5).astype(np.uint8) * 255

    # Create transparent background
    _, alpha_channel = cv2.threshold(binary_mask, 127, 255, cv2.THRESH_BINARY)
    rgba_image = np.dstack((input_image_rgb, alpha_channel))

    return Image.fromarray(rgba_image)

def main():
    st.title('Background Removal App')

    image_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

    if image_file is not None:
        input_image = Image.open(image_file)
        st.image(input_image, caption='Uploaded Image', use_column_width=True)
        st.write("Processing...")

        result_image = remove_background(input_image)
        st.image(result_image, caption='Result Image with Background Removed', use_column_width=True)

if __name__ == "__main__":
    main()
