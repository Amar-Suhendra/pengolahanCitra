import streamlit as st
from PIL import Image
import io

# Fungsi untuk kompresi kehilangan (lossy)
def compress_lossy(image, quality):
    img = Image.open(image)
    img_io = io.BytesIO()
    
    # Menangani kompresi PNG dengan kehilangan
    if img.format == 'PNG':
        img = img.convert("RGB")
    
    img.save(img_io, format='JPEG', quality=quality)
    return img_io, img_io.tell()  # Mengembalikan juga ukuran file

# Fungsi untuk kompresi tanpa kehilangan (lossless)
def compress_lossless(image):
    img = Image.open(image)
    img_io = io.BytesIO()
    img.save(img_io, format='PNG')
    return img_io, img_io.tell()  # Mengembalikan juga ukuran file

# Tampilan aplikasi web menggunakan Streamlit
st.title("Image Compression App")

# Upload gambar
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    original_img_size = uploaded_image.size  # Ukuran file asli

    # Pilihan metode kompresi
    compression_method = st.radio("Select Compression Method", ["Lossy", "Lossless"])
    

    if compression_method == "Lossy":
        # Pengaturan kualitas kompresi (0-100, 100=tanpa kehilangan)
        quality = st.slider("Select Compression Quality (0 - 100)", 0, 100, 75)
        st.write(f"Compression Quality: {quality}")

        col1, col2 = st.columns(2)
        col1.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        col1.write(f"Original Image Size: {original_img_size} bytes")
        
        # Kompresi gambar kehilangan
        compressed_image, compressed_size = compress_lossy(uploaded_image, quality)
        col2.image(compressed_image, caption=f"Lossy Compressed Image (Quality: {quality})", use_column_width=True)
        col2.write(f"Compressed Image Size: {compressed_size} bytes")

    elif compression_method == "Lossless":
        # Kompresi gambar tanpa kehilangan

        col1, col2 = st.columns(2)
        col1.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        col1.write(f"Original Image Size: {original_img_size} bytes")
        compressed_image, compressed_size = compress_lossless(uploaded_image)
        col2.image(compressed_image, caption="Lossless Compressed Image", use_column_width=True)
        col2.write(f"Compressed Image Size: {compressed_size} bytes")