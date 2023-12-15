import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

def generate_freeman_plot(freeman_code):
    img_size = 20
    img = np.zeros((img_size, img_size))

    x, y = 4, 4
    img[y][x] = 1

    for direction in freeman_code:
        if direction in [1, 2, 3]:
            y -= 1
        if direction in [5, 6, 7]:
            y += 1
        if direction in [3, 4, 5]:
            x -= 1
        if direction in [0, 1, 7]:
            x += 1

        img[y][x] = 1

    return img

def streamlit_freeman_code_app():
    st.title("Chain Code Generator")

    freeman_code_input = st.text_input("Enter Freeman Code (comma-separated):", "3, 3, 3, 6, 6, 4, 6, 7, 7, 0, 0, 6")

    try:
        freeman_code = [int(code) for code in freeman_code_input.split(",")]
    except ValueError:
        st.error("Invalid input. Please enter a valid list of integers separated by commas.")
        return

    img = generate_freeman_plot(freeman_code)

    fig, ax = plt.subplots()
    ax.imshow(img, cmap='binary')
    ax.axis('off')  # Hide axes
    st.pyplot(fig)

if __name__ == '__main__':
    streamlit_freeman_code_app()