import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Blurred Image",
    page_icon="🏡",
    layout="wide",
)

st.write("# Blurred Image")
st.write(
     "Blurred image is a visual representation of a scene or object in which the details and sharpness have been intentionally reduced or lost. Blurring in images is achieved by applying a blur filter or algorithm that alters the appearance of the pixels in the image. The result is a smoothing of fine details and a reduction in the contrast between adjacent pixels"
)

st.write("## How it works?")
st.write(
     "Blurred image is created by applying a mathematical operation or filter to each pixel in an original image, which alters the pixel's color or brightness based on its surrounding pixels. The specific method used to apply the blur filter determines the type and extent of blurring."
)

st.write("## Try it Out!")

# Function for make blur level 0-100
def blur_image(image, blur_level):
    kernel_size = int(blur_level * 0.01 * 20)

    # make kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    return blurred_image

st.title("Image Blur and Histograms")

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_image is not None:

    # Read and show original image
    image_data = uploaded_image.read()
    image = np.array(bytearray(image_data), dtype=np.uint8)
    original_image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    st.image(original_image, caption="Original Image", use_column_width=True, channels="BGR")

    # add slider to adjust blur level
    blur_level = st.slider("Blur Level (%)", 0, 100, 50, 1)

    # Blur the image
    blurred_image = blur_image(original_image, blur_level)

    # Display the blurred image
    st.image(blurred_image, caption=f"Blurred Image ({blur_level}%)", use_column_width=True, channels="BGR")

    # Calculate histogram for original and blurred image
    original_hist = cv2.calcHist([original_image], [0], None, [256], [0, 256])
    blurred_hist = cv2.calcHist([blurred_image], [0], None, [256], [0, 256])

    combined_hist = np.hstack((original_hist, blurred_hist))

    st.title("Combined Histogram")

    plt.figure(figsize=(12, 5))
    plt.plot(combined_hist[:, 0], color="blue", label="Original Image")
    plt.plot(combined_hist[:, 1], color="red", label="Blurred Image")
    plt.title("Combined Histogram")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.legend()
    st.pyplot(plt)