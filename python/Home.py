import streamlit as st

st.set_page_config(
    page_title="Home",
    page_icon="🏡",
   layout="wide",
)

st.write("# Welcome to Image Processing Web App")

st.markdown(
    """
    Image Processing Web App is a web application that allows you to upload an image and apply various image processing techniques to it. this web app is built using [Streamlit](https://streamlit.io/).

    ## How to use?
    1. Click on the menu on the left to select the image processing technique you want to apply.
    2. Upload an image.
    3. Done and dusted! You can now see the result of the image processing technique you selected.

    ## Image Processing Techniques we support
    - [Histogram Equalization Python](https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_histograms/py_histogram_equalization/py_histogram_equalization.html)

    ## About us
    We are a group of students from [Universitas Syiah Kuala](https://usk.ac.id) who are taking a course called Image Processing in the semester of 2023/2024. This web app is our project for the course.
    ### Our team members
    - [Amar Suhendra](https://www.instagram.com/suhendra.amar/)
    - [Teuku Muhammad Roy Adrian](https://www.instagram.com/adr.roy_/)
    - [Hadafee Mudo](https://www.instagram.com/_dfee__/)

"""
)
