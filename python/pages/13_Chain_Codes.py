import streamlit as st
import cv2
import numpy as np

st.set_page_config(
    page_title="Chain Codes",
    page_icon="🏡",
    layout="wide",
)

st.write("# Chain Codes")
st.markdown(
    """
  Chain codes are a lossless compression technique used for representing contour shapes in an image. It is a simple technique that can be easily implemented in any programming language. It is also known as the Freeman Chain Code, named after its inventor, Kenneth Freeman, who developed the technique in 1961.
"""
)

st.write("## How it works?")
st.markdown("""
Chain codes have 2 types of codes, 4-directional and 8-directional. 4-directional chain codes are used to represent contours with 4-connected pixels, while 8-directional chain codes are used to represent contours with 8-connected pixels.
""")

st.write("## Try it out!")

chain_code = []

def find_starting_direction(thresh_img, option, directions, height):
    width = thresh_img.shape[1]
    
    for i, direction in enumerate(directions):
        next_point = (0 + direction[0], 0 + direction[1])
        if 0 <= next_point[0] < height and 0 <= next_point[1] < width and thresh_img[next_point] == 255:
            return i
    return -1

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:

    img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)

    # Convert the image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to get a binary image
    _, thresh_img = cv2.threshold(gray_img, 128, 255, cv2.THRESH_BINARY)

    # Display the binary image
    st.write("### Binary Image")
    st.image(thresh_img, use_column_width=300)

    # Calculate chain code directly from the binary image
    option = st.selectbox("Select chain code type", [
                          "4-directional", "8-directional"])
    
    # Define directions for 4-directional and 8-directional chain codes
    directions_4 = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    directions_8 = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
    
    directions = directions_4 if option == "4-directional" else directions_8
    height, width = thresh_img.shape

    # Find the starting direction based on the first non-zero pixel
    initial_direction = find_starting_direction(thresh_img, option, directions, height)

    if initial_direction != -1:
        current_point = (0, 0)
        for _ in range(1000):  # A safety limit to avoid infinite loop
            for i, direction in enumerate(directions):
                next_point = (
                    current_point[0] + direction[0], current_point[1] + direction[1])
                if 0 <= next_point[0] < height and 0 <= next_point[1] < width and thresh_img[next_point] == 255:
                    chain_code.append(i)
                    current_point = next_point
                    break

        # Convert chain code to string
        chain_code_str = "".join([str(code) for code in chain_code])

        # Display the chain code
        st.write("### Chain Code")
        st.write(chain_code_str)
