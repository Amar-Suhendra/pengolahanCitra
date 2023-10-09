import streamlit as st
import cv2
import numpy as np
from sklearn.cluster import KMeans
import tempfile


def cartoonize_image(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize image untuk mempercepat processing
    scale_factor = 0.5
    image_rgb = cv2.resize(image_rgb, None, fx=scale_factor, fy=scale_factor)

    # K-means clustering untuk color quantization
    num_clusters = 8
    pixel_data = image_rgb.reshape((-1, 3))
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(pixel_data)
    quantized_colors = kmeans.cluster_centers_.astype(int)
    quantized_image = quantized_colors[kmeans.labels_].reshape(image_rgb.shape)

    grayscale_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # edge detection threshold menggunakan Canny
    edge_threshold = 100
    edges = cv2.Canny(grayscale_image, threshold1=edge_threshold,
                      threshold2=2 * edge_threshold)

    # Invert edges untuk masking
    edges = 255 - edges

    # menggabungkan cartoonized image dan edge mask
    cartoonized = cv2.bitwise_and(quantized_image, quantized_image, mask=edges)

    return cartoonized


def main():
    st.title("Cartoonize Filter with Streamlit")

    uploaded_image = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        col1, col2 = st.columns(2)
        col1.header("Original Image")
        col1.image(uploaded_image, width=400)

        if st.button("Cartoonize"):
            st.spinner("Cartoonizing...")

            # membuat temporary file untuk menyimpan uploaded image
            with tempfile.NamedTemporaryFile(delete=False) as temp_image:
                temp_image.write(uploaded_image.read())

            cartoonized = cartoonize_image(temp_image.name)
            col2.header("Cartoonized Image")
            col2.image(cartoonized, caption="Cartoonized Image", width=400)


if __name__ == "__main__":
    main()
