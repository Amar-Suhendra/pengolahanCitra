import streamlit as st
import cv2 as cv
import numpy as np

st.set_page_config(
    page_title="Histogram Equalization",
    page_icon="🏡",
    layout="wide",
)

st.write("# Histogram Equalization")
st.markdown(
  """
  Histogram Equalization is a computer image processing technique used to improve contrast in images. It accomplishes this by effectively spreading out the most frequent intensity values, i.e. stretching out the intensity range of the image.
"""
)

st.write("## How it works?")
st.markdown(
  """
  Histogram equalization enhances image contrast by adjusting pixel intensities to create a more balanced histogram, improving detail visibility, particularly in images with poor initial contrast. It's a valuable technique in image processing, though it may require careful application, especially in color images where channels are processed independently.
"""
)

st.write("## Try it out!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:

  # show original image
  st.write("## Original Image")
  col1, col2 = st.columns(2)
  
  image = cv.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
  hist = cv.calcHist([image], [0], None, [256], [0, 256])
  col1.image(image, channels="BGR", width=400)
  col2.bar_chart(hist)

  # show equalized image
  st.write("## Equalized Image")
  col1, col2 = st.columns(2)

  image2 = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

  equalized_image = cv.equalizeHist(image2)
  equalized_hist = cv.calcHist([equalized_image], [0], None, [256], [0, 256])
  col1.image(equalized_image, channels = "RGB", width=400)
  col2.bar_chart(equalized_hist)
