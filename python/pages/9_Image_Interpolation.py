import streamlit as st
import cv2
import numpy as np

st.set_page_config(
    page_title="Image Interpolation",
    page_icon="🏡",
    layout="wide",
)

st.title("Image Interpolation")
st.markdown("""Image interpolation is a technique to resize an image. Image interpolation can be used to resize an image to a larger or smaller size.""")

st.write("## How it works?")
st.markdown("""Image interpolation works by using a transformation matrix. The transformation matrix is a matrix that is used to transform an image. The transformation matrix is used to resize an image to a larger or smaller size.""")

st.write("## Try it out!")
# Upload image through Streamlit
image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if image is not None:
    image = cv2.imdecode(np.fromstring(image.read(), np.uint8), 1)

    # Select interpolation method
    selected_interpolation = st.selectbox("Select Interpolation Method", ["Nearest Neighbor", "Bilinear", "Bicubic"])

    # Slider to adjust
    # User input for scaling
    scaling_factor = st.slider("Scaling Factor:", 0.1, 5.0, 1.0,)

    # Resize image
    if selected_interpolation == "Nearest Neighbor":
        result = cv2.resize(image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_NEAREST)
    elif selected_interpolation == "Bilinear":
        result = cv2.resize(image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_LINEAR)
    elif selected_interpolation == "Bicubic":
        result = cv2.resize(image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_CUBIC)
    
    # Show result
    col1, col2 = st.columns(2)
    col1.header("Original")
    col1.image(image, channels="BGR", use_column_width=True)
    col2.header("Result")
    col2.image(result, channels="BGR", use_column_width=True)

