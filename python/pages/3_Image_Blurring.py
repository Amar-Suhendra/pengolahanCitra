import streamlit as st
import cv2 as cv
import numpy as np

st.set_page_config(
    page_title="Image Blurring",
    page_icon="🏡",
    layout="wide",
)

st.write("# Image Blurring")
st.markdown(
  """
  Blurred image is a visual representation of a scene or object in which the details and sharpness have been intentionally reduced or lost. Blurring in images is achieved by applying a blur filter or algorithm that alters the appearance of the pixels in the image. The result is a smoothing of fine details and a reduction in the contrast between adjacent pixels
""")

st.write("## How it works?")
st.markdown(
  """
  Blurred image is created by applying a mathematical operation or filter to each pixel in an original image, which alters the pixel's color or brightness based on its surrounding pixels. The specific method used to apply the blur filter determines the type and extent of blurring.
""")

# Function for make blur level 0-100
def blur_image(image, blur_level):
    kernel_size = int(blur_level * 0.01 * 20)

    # make kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Apply Gaussian blur
    blurred_image = cv.GaussianBlur(image, (kernel_size, kernel_size), 0)

    return blurred_image

st.write("## Try it Out!")

option = st.selectbox("What do you want to do?", ("Face Blurring", "Image Blurring"))
st.write("You selected:", option)

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
  if option == "Face Blurring":
    #slider to adjust blur level
    blur_level = st.slider("Blur Level (%)", 0, 100, 50, 1)

    # Read and show original image
    image_data = uploaded_image.read()
    image = np.array(bytearray(image_data), dtype=np.uint8)
    original_image = cv.imdecode(image, cv.IMREAD_COLOR)

    # Load the pre-trained face detector
    face_detector = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")  # type: ignore

    # Detect faces in the image
    faces = face_detector.detectMultiScale(original_image, 1.1, 5)

    # Blur all faces found in the image
    for (x, y, w, h) in faces:
        # Extract the region of interest (ROI) of the image
        face = original_image[y:y + h, x:x + w]

        # Blur the ROI
        blurred_face = blur_image(face, blur_level)

        # Merge this blurry ROI back into the original image
        original_image[y:y + blurred_face.shape[0], x:x + blurred_face.shape[1]] = blurred_face

    # Display the blurred image
    col1, col2 = st.columns(2)
    col1.image(original_image, caption="Blurred Image ({blur_level}%)", use_column_width=True, channels="BGR")
    col2.image(uploaded_image, caption="Original Image", use_column_width=True, channels="BGR")
  else:
    #slider to adjust blur level
    blur_level = st.slider("Blur Level (%)", 0, 100, 50, 1)

    # Read and show original image
    image_data = uploaded_image.read()
    image = np.array(bytearray(image_data), dtype=np.uint8)
    original_image = cv.imdecode(image, cv.IMREAD_COLOR)

    # Blur the image
    blurred_image = blur_image(original_image, blur_level)

    # Display the blurred image
    col1, col2 = st.columns(2)

    col1.image(blurred_image, caption=f"Blurred Image ({blur_level}%)", use_column_width=True, channels="BGR")
    col2.image(uploaded_image, caption="Original Image", use_column_width=True, channels="BGR")
