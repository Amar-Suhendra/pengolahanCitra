import streamlit as st
import cv2
import numpy as np

st.set_page_config(
    page_title="Face Filter",
    page_icon="🏡",
    layout="wide",
)

st.title("Face Filter")
st.markdown("""Face filter is a filter that allows you to add image filters to your face.""")

st.write("## How it works?")
st.markdown("""This filter works by using Haar Cascade Classifier for face detection and adding a filter to the detected face.""")

st.write("## Try it out!")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml") # type: ignore
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml") # type: ignore

glasses = cv2.imread("images/glasses.png", -1)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
  image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  face = face_cascade.detectMultiScale(image, 1.3, 5)

  for (x, y, w, h) in face:
    resized_glasses = cv2.resize(glasses, (w, h), interpolation=cv2.INTER_AREA)
    for i in range(resized_glasses.shape[0]):
      for j in range(resized_glasses.shape[1]):
        if resized_glasses[i, j, 3] != 0:
          image[y + i, x + j, :] = resized_glasses[i, j, :3]
    
  col1, col2 = st.columns(2)
  col1.header("Original")
  col1.image(uploaded_file, channels="BGR", use_column_width=True)
  col2.header("Result")
  col2.image(image, channels="BGR", use_column_width=True)
  