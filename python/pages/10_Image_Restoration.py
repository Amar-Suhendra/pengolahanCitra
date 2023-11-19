import streamlit as st
import numpy as np
import cv2
import random

def add_salt_and_pepper_noise(img, amount):
    noisy_img = img.copy()
    row, col, _ = img.shape

    # Add white pixels
    num_white_pixels = int(amount * row * col)
    for _ in range(num_white_pixels):
        y_coord = random.randint(0, row - 1)
        x_coord = random.randint(0, col - 1)
        noisy_img[y_coord, x_coord, :] = 255

    # Add black pixels
    num_black_pixels = int(amount * row * col)
    for _ in range(num_black_pixels):
        y_coord = random.randint(0, row - 1)
        x_coord = random.randint(0, col - 1)
        noisy_img[y_coord, x_coord, :] = 0

    return noisy_img

def add_gaussian_noise(img, sigma):
    row, col, _ = img.shape
    gaussian_noise = np.random.normal(0, sigma, (row, col, 3))
    noisy_img = img + gaussian_noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8)

def apply_median_filter(img, kernel_size):
    # Ensure the kernel size is odd
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    return cv2.medianBlur(img, kernel_size)

def apply_outlier_method(img, threshold):
    outlier_img = img.copy()
    low_threshold = np.percentile(outlier_img, threshold / 2)
    high_threshold = np.percentile(outlier_img, 100 - threshold / 2)
    
    outlier_img[outlier_img < low_threshold] = low_threshold
    outlier_img[outlier_img > high_threshold] = high_threshold
    
    return outlier_img

st.title("Image Restoration App")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)  # Read as RGB
    st.image(img, caption="Uploaded Image",channels="BGR", use_column_width=True)

    st.sidebar.title("Noise Options")
    noise_type = st.sidebar.radio("Select noise type:", ("None", "Salt-and-Pepper", "Gaussian"))

    if noise_type != "None":
        if noise_type == "Salt-and-Pepper":
            noise_level = st.sidebar.slider("Noise Level", 0.0, 1.0, 0.5)
            noisy_img = add_salt_and_pepper_noise(img, noise_level)
        elif noise_type == "Gaussian":
            sigma = st.sidebar.slider("Gaussian Noise Sigma", 0.1, 50.0, 10.0)
            noisy_img = add_gaussian_noise(img, sigma)

        st.image(noisy_img, caption=f"Noisy Image ({noise_type} Noise)", channels="BGR", use_column_width=True)

        st.sidebar.title("Restoration Options")
        restoration_type = st.sidebar.radio("Select restoration type:", ("None", "Median Filtering", "Outlier Method"))

        if restoration_type != "None":
            if restoration_type == "Median Filtering":
                kernel_size = st.sidebar.slider("Kernel Size", 1, 15, 3)
                restored_img = apply_median_filter(noisy_img, kernel_size)
            elif restoration_type == "Outlier Method":
                threshold = st.sidebar.slider("Outlier Removal Threshold", 0.0, 100.0, 5.0)
                restored_img = apply_outlier_method(noisy_img, threshold)

            st.image(restored_img, caption=f"Restored Image ({restoration_type})",channels="BGR", use_column_width=True)
