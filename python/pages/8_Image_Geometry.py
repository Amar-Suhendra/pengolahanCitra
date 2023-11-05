import streamlit as st
import cv2
import numpy as np

def apply_rotation(image, angle):
    # Menghitung pusat gambar
    center = tuple(np.array(image.shape[1::-1]) / 2)
    # Membuat matriks rotasi
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    # Melakukan rotasi pada gambar
    rotated_image = cv2.warpAffine(image, rotation_matrix, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return rotated_image

def apply_scaling(image, scale_factor):
    # Melakukan scaling pada gambar
    scaled_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    return scaled_image

def apply_translation(image, tx, ty):
    # Membuat matriks translasi
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    # Melakukan translasi pada gambar
    translated_image = cv2.warpAffine(image, translation_matrix, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return translated_image

def main():
    st.title("Image Geometry Features with Streamlit")

    # Upload image through Streamlit
    image = st.file_uploader("Pilih gambar", type=["jpg", "jpeg", "png"])

    if image is not None:
        image = cv2.imdecode(np.fromstring(image.read(), np.uint8), 1)


        # Get image dimensions
        height, width, _ = image.shape
        

        # User input for rotation, scaling, and translation
        rotation_angle = st.slider("Rotation Angle (degrees):", -180, 180, 0)
        scaling_factor = st.slider("Scaling Factor:", 0.1, 2.0, 1.0)
        translation_x = st.slider("Translation X:", -width, width, 0)
        translation_y = st.slider("Translation Y:", -height, height, 0)

        # Apply transformations based on user input
        rotated_image = apply_rotation(image, rotation_angle)
        scaled_image = apply_scaling(rotated_image, scaling_factor)
        translated_image = apply_translation(scaled_image, translation_x, translation_y)

        # Display the transformed image
        st.image(translated_image, caption="Transformed Image", use_column_width=True, channels="BGR")

    

if __name__ == "__main__":
    main()
