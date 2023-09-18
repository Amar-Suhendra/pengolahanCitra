import streamlit as st
import cv2
import numpy as np

st.set_page_config(
    page_title="Edge Detection",
    page_icon="🏡",
    layout="wide",
)

st.write("# Edge Detection")

st.markdown(
    """
    Edge detection is a fundamental technique in image processing used to identify boundaries within an image. It plays a crucial role in various computer vision tasks, including object detection, image segmentation, and more. The primary goal of edge detection is to find the regions in an image where pixel intensities change rapidly, ypically indicating the presence of object boundaries or edges.
    """
)

st.write("## How it works?")

st.markdown(
    """
    Edge detection algorithms, such as the Canny edge detector, Sobel, and Laplacian, analyze the gradient or the rate of change in pixel intensities across an image. These algorithms apply convolution operations to highlight areas with significant changes in intensity, making it easier to identify edges."""
)

st.write("## Try it out!")

# Upload gambar
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    
    # default threshold
    threshold1 = 100
    threshold2 = 200

    # slider threshold
    threshold1 = st.slider("Threshold 1", 0, 255, threshold1)
    threshold2 = st.slider("Threshold 2", 0, 255, threshold2)

    # Show the original image and its histogram
    col1, col2 = st.columns(2)
    col1.write("## Original Image")
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    col1.image(image, channels="BGR", width=530)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    col2.write("## Edge Detection Image")

    # Apply edge detection using the Canny method
    edges = cv2.Canny(gray_image, threshold1, threshold2)

    # Display the edge-detected image and the binary histogram
    col2.image([edges], width=530)
