import streamlit as st
from utils import process_receipt_image 
# App title
st.title("Receipt Data Extractor")

# Dropdown menu for pipeline selection
pipeline_option = st.selectbox(
    "Select a processing pipeline:",
    ["Pipeline 1: K-means + Edge Detection", "Pipeline 2: K-means Only", "Pytesseract"],
)

# File uploader
uploaded_file = st.file_uploader("Upload a receipt image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Display uploaded image
    print("uploaded_file", uploaded_file)
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # Process the image
    with st.spinner("Processing..."): 
        extracted_data = process_receipt_image(uploaded_file, pipeline_option)
       

    # Display results
    if extracted_data:
        st.success("Data extracted successfully!")
        for key, value in extracted_data.items():
            st.write(f"**{key}:** {value}")
    else:
        st.error("No relevant data could be extracted. Please try again.")
