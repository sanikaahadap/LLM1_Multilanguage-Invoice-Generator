import streamlit as st
import google.generativeai as genai
from PIL import Image
import os
from dotenv import load_dotenv
import fitz  # PyMuPDF
import io

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load the Gemini Pro Vision Model
model = genai.GenerativeModel('gemini-pro-vision')


def get_gemini_response(input_prompt, documents, user_input_prompt):
    response = model.generate_content([input_prompt] + documents + [user_input_prompt])
    return response.text


def input_document_bytes(uploaded_file):
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        if uploaded_file.type == "application/pdf":
            # Convert PDF to images
            pdf_document = fitz.open(stream=bytes_data, filetype="pdf")
            images = []
            for page_number in range(len(pdf_document)):
                page = pdf_document.load_page(page_number)
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                with io.BytesIO() as output:
                    img.save(output, format="PNG")
                    img_byte_array = output.getvalue()
                    images.append({
                        "mime_type": "image/png",
                        "data": img_byte_array
                    })
            return images
        else:
            # For image files (jpeg, jpg, png)
            image_parts = [
                {
                    "mime_type": uploaded_file.type,
                    "data": bytes_data
                }
            ]
            return image_parts
    else:
        raise FileNotFoundError("No File Uploaded")


# Initialize the Streamlit App
st.set_page_config(page_title="Document Content Query App")
input_prompt = """
You are an expert in understanding various document formats. Please try to answer the question using the information 
from the uploaded documents.
"""
user_input_prompt = st.text_input("User Input Prompt", key="input")
upload_files = st.file_uploader("Upload Documents", type=["pdf", "jpg", "jpeg", "png"], accept_multiple_files=True)

if upload_files:
    for file in upload_files:
        if file.type in ["image/jpeg", "image/jpg", "image/png"]:
            image = Image.open(file)
            st.image(image, caption=f"Uploaded Image: {file.name}", use_column_width=True)
        elif file.type == "application/pdf":
            st.write(f"Uploaded PDF: {file.name}")

submit = st.button("Find the Answer from the Documents")
if submit and upload_files:
    all_documents_data = []
    for file in upload_files:
        try:
            all_documents_data.extend(input_document_bytes(file))
        except Exception as e:
            st.error(f"Error processing file {file.name}: {str(e)}")
            continue
    try:
        response = get_gemini_response(input_prompt, all_documents_data, user_input_prompt)
        st.subheader("Response")
        st.write(response)
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
