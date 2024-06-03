import streamlit as st
import google.generativeai as genai
from PIL import Image
import os
from dotenv import load_dotenv
import fitz  # PyMuPDF
import io
import pandas as pd

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load the Gemini Pro Vision Model
model = genai.GenerativeModel('gemini-pro-vision')


def get_gemini_response(input_prompt, document):
    response = model.generate_content([input_prompt, document])
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


def extract_details_from_document(document_data):
    input_prompt = "Extract relevant details from this document."
    try:
        response = get_gemini_response(input_prompt, document_data)
        return response
    except Exception as e:
        st.error(f"Error extracting details: {str(e)}")
        return None


def parse_details(response_text):
    details = {}
    lines = response_text.split('\n')
    for line in lines:
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            if key in details:
                if isinstance(details[key], list):
                    details[key].append(value)
                else:
                    details[key] = [details[key], value]
            else:
                details[key] = value
    return details


# Initialize the Streamlit App
st.set_page_config(page_title="Document Details Extraction App")
st.title("Document Details Extraction")

st.write("""
Upload documents of the same type (e.g., ID cards, birth certificates) and extract common details into a CSV file.
""")

upload_files = st.file_uploader("Upload Documents", type=["pdf", "jpg", "jpeg", "png"], accept_multiple_files=True)

if upload_files:
    extracted_data = []

    for file in upload_files:
        if file.type in ["image/jpeg", "image/jpg", "image/png"]:
            image = Image.open(file)
            st.image(image, caption=f"Uploaded Image: {file.name}", use_column_width=True)
        elif file.type == "application/pdf":
            st.write(f"Uploaded PDF: {file.name}")

        document_data = input_document_bytes(file)
        for document in document_data:
            details = extract_details_from_document(document)
            if details:
                parsed_details = parse_details(details)
                extracted_data.append(parsed_details)

    if extracted_data:
        st.subheader("Extracted Details")
        df = pd.DataFrame(extracted_data)

        # Display the DataFrame
        st.write(df)

        # Provide a download button for the CSV
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name='extracted_details.csv',
            mime='text/csv',
        )
