import streamlit as st
import cv2
import numpy as np

st.set_page_config(
    page_title="Morphologi Operations",
    page_icon="🏡",
    layout="wide",
)

# Fungsi untuk melakukan Dilation
def perform_dilation(image):
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(image, kernel, iterations=1)
    return dilated

# Fungsi untuk melakukan Erosion
def perform_erosion(image):
    kernel = np.ones((5, 5), np.uint8)
    eroded = cv2.erode(image, kernel, iterations=1)
    return eroded

# Fungsi untuk melakukan Opening
def perform_opening(image):
    kernel = np.ones((5, 5), np.uint8)
    opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return opened

# Fungsi untuk melakukan Closing
def perform_closing(image):
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return closed

# Fungsi untuk melakukan Morph Gradient
def perform_morph_gradient(image):
    kernel = np.ones((5, 5), np.uint8)
    gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
    return gradient

# Streamlit UI
st.title("Morphological Operations")
st.markdown("""Morphological operations are image processing operations based on the shape of the object in the image. Morphological operations are usually used to process binary images, but can also be used to process grayscale images.""")
st.write("## How it works?")
st.markdown("""Morphological operations work by using a structuring element to process the image. The structuring element is a matrix that is used to process the image. The structuring element is placed on top of the image, and the structuring element is moved to every pixel in the image. The structuring element is used to determine the value of the pixel in the image. The structuring element is usually a matrix with a size of 3x3 or 5x5.""")

st.write("## Try it out!")
# Upload gambar
image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if image is not None:
    image = cv2.imdecode(np.fromstring(image.read(), np.uint8), 1)

    # Pilihan untuk fitur morfologi
    selected_operation = st.selectbox("Pilih Operasi Morfologi", ["Dilation", "Erosion", "Opening", "Closing", "Morph Gradient"])

    if selected_operation == "Dilation":
        result = perform_dilation(image)
        
    elif selected_operation == "Erosion":
        result = perform_erosion(image)
        
    elif selected_operation == "Opening":
        result = perform_opening(image)
        
    elif selected_operation == "Closing":
        result = perform_closing(image)
        
    elif selected_operation == "Morph Gradient":
        result = perform_morph_gradient(image)
    
    col1, col2 = st.columns(2)
    col1.image(image, caption="Gambar Asli", use_column_width=True, channels="BGR")
    col2.image(result, caption=f"Hasil {selected_operation}", use_column_width=True, channels="BGR")