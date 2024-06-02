import streamlit as st
import google.generativeai as genai
from PIL import Image
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure the Google Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load the Gemini Pro Vision Model
model = genai.GenerativeModel('gemini-pro-vision')


# Function to get response from Gemini model
def get_gemini_response(input_prompt, image, user_input_prompt):
    response = model.generate_content([input_prompt, image[0], user_input_prompt])
    return response.text


# Function to convert uploaded file to image bytes
def input_image_bytes(uploaded_file):
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        image_parts = [
            {
                "mime_type": uploaded_file.type,
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No File Uploaded")


# Initialize the Streamlit app
st.set_page_config(page_title="Multi-Image Data Extractor")

st.title("Multi-Image Data Extractor")
st.write("Upload images of similar type documents to extract and store information in a CSV file.")

input_prompt = """
You are an expert in understanding images. Please try to answer the question using the information from the uploaded
image.
"""

# User input prompt
user_input_prompt = st.text_input("User Input Prompt", key="input")

# File uploader for multiple images
uploaded_files = st.file_uploader("Upload Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# Initialize an empty list to hold data
data = []

# Process each uploaded image
if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        st.image(image, caption=uploaded_file.name, use_column_width=True)

    if st.button("Process Images"):
        for uploaded_file in uploaded_files:
            input_image_data = input_image_bytes(uploaded_file)
            response_text = get_gemini_response(input_prompt, input_image_data, user_input_prompt)
            text_data = {"filename": uploaded_file.name, "text": response_text}
            data.append(text_data)

        # Convert extracted data to a DataFrame
        df = pd.DataFrame(data)

        # Display DataFrame
        st.write(df)

        # Download button for CSV
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download CSV", data=csv, file_name='extracted_data.csv', mime='text/csv')
