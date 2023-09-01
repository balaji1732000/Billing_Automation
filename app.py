from dotenv import load_dotenv
from pytesseract import image_to_string
from PIL import Image
from io import BytesIO
import pypdfium2 as pdfium
import streamlit as st
import multiprocessing
from tempfile import NamedTemporaryFile
import pandas as pd
import json
import requests
import os

load_dotenv()

azure_api_key = os.getenv("AZURE_OPENAPI_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
# 1. Convert PDF file into images via pypdfium2

# def convert_pdf_to_images(file_path, scale=300/72):

#     pdf_file = pdfium.PdfDocument(file_path)

#     page_indices = [i for i in range(len(pdf_file))]

#     renderer = pdf_file.render(
#         pdfium.PdfBitmap.to_pil,
#         page_indices=page_indices,
#         scale=scale,
#     )

#     final_images = []

#     for i, image in zip(page_indices, renderer):

#         image_byte_array = BytesIO()
#         image.save(image_byte_array, format='jpeg', optimize=True)
#         image_byte_array = image_byte_array.getvalue()
#         final_images.append(dict({i: image_byte_array}))

#     return final_images


def convert_pdf_to_images(file_path, scale=300 / 72):
    try:
        pdf_file = pdfium.PdfDocument(file_path)
        page_indices = [i for i in range(len(pdf_file))]

        renderer = pdf_file.render(
            pdfium.PdfBitmap.to_pil,
            page_indices=page_indices,
            scale=scale,
        )

        final_images = []

        for i, image in zip(page_indices, renderer):
            image_byte_array = BytesIO()
            image.save(image_byte_array, format="jpeg", optimize=True)
            image_byte_array = image_byte_array.getvalue()
            final_images.append({i: image_byte_array})

        return final_images
    except pdfium.PdfiumError as e:
        print(f"Error while converting PDF: {e}")
        return []


# 2. Extract text from images via pytesseract


def extract_text_from_img(list_dict_final_images):
    image_list = [list(data.values())[0] for data in list_dict_final_images]
    image_content = []

    for index, image_bytes in enumerate(image_list):
        image = Image.open(BytesIO(image_bytes))
        raw_text = str(image_to_string(image))
        image_content.append(raw_text)

    return "\n".join(image_content)


def extract_content_from_url(url: str):
    images_list = convert_pdf_to_images(url)
    text_with_pytesseract = extract_text_from_img(images_list)

    return text_with_pytesseract


# 3. Extract structured info from text via LLM

# def extract_structured_data(content: str, data_points):
#     llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613", openai_api_key = os.getenv("OPENAI_API_KEY"))
#     template = """
#     You are an expert admin people who will extract core information from documents

#     {content}

#     Above is the content; please try to extract all data points from the content above
#     and export in a JSON array format:
#     {data_points}

#     Now please extract details from the content  and export in a JSON array format,
#     return ONLY the JSON array:
#     """

#     prompt = PromptTemplate(
#         input_variables=["content", "data_points"],
#         template=template,
#     )

#     chain = LLMChain(llm=llm, prompt=prompt)

#     results = chain.run(content=content, data_points=data_points)

#     return results

import os
import requests

# def extract_structured_data_with_azure(content: str, data_points, azure_api_key):
#     template = """
#     You are an expert admin people who will extract core information from documents

#     {content}

#     Above is the content; please try to extract all data points from the content above
#     and export in a JSON array format:
#     {data_points}

#     Now please extract details from the content  and export in a JSON array format,
#     return ONLY the JSON array:
#     """

#     prompt = template.format(content=content, data_points=data_points)

#     url = "https://dwspoc.openai.azure.com/openai/deployments/GPTDavinci/completions?api-version=2022-12-01"

#     headers = {"Content-Type": "application/json", "api-key": azure_api_key}

#     data = {
#         "prompt": prompt,
#         "max_tokens": 400,
#         "temperature": 0.5,
#         "top_p": 1,
#         "stop": None,
#     }

#     response = requests.post(url, headers=headers, json=data)
#     response_data = response.json()

#     return response_data


import openai


def extract_structured_data_with_openai(content: str, data_points, openai_api_key):
    template = """
    You are an expert admin people who will extract core information from documents

    {content}

    Above is the content; please try to extract all data points from the content above 
    and export in a JSON array format:
    {data_points}

    Now please extract details from the content  and export in a JSON array format, 
    return ONLY the JSON array:

    """

    prompt = template.format(content=content, data_points=data_points)

    # Set your OpenAI API key
    openai.api_key = openai_api_key

    # Construct the message with the prompt
    message = [{"role": "system", "content": prompt}]

    # Define the function call and description
    function_description = "Extract structured data from content using data_points."
    function_call = "auto"

    # Create the chat completion using OpenAI API
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=message,
    )

    # Extract the text from the response
    if response.choices and response.choices[0].message:
        return response.choices[0].message.get("content", "").strip()
    else:
        return "Failed to extract structured data"


# 4. Send Data to make.com via webhook
def send_to_make(data):
    # Replace with your own link
    webhook_url = "https://hook.eu2.make.com/sp3yp63pai7gdm45x3klh22bekvy8yc5"

    json = {"data": data}

    try:
        response = requests.post(webhook_url, json=json)
        response.raise_for_status()  # Check for any HTTP errors
        print("Data sent successfully!")
    except requests.exceptions.RequestException as e:
        print(f"Failed to send data: {e}")


# 5. Streamlit app
def main():
    default_data_points = """{
        "invoice_item": "what is the item that charged",
        "Amount": "how much does the invoice item cost in total",
        "Company_name": "company that issued the invoice",
        "invoice_date": "when was the invoice issued",
    }"""

    st.set_page_config(page_title="Doc extraction", page_icon=":bird:")

    st.header("Doc extraction :bird:")

    data_points = st.text_area("Data points", value=default_data_points, height=170)

    uploaded_files = st.file_uploader("upload PDFs", accept_multiple_files=True)

    if uploaded_files is not None and data_points is not None:
        results = []
        for file in uploaded_files:
            with NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                f.write(file.getbuffer())
                absolute_path = os.path.abspath(f.name)
                content = extract_content_from_url(absolute_path)
                print(content)
                data = extract_structured_data_with_openai(
                    content, data_points, openai_api_key
                )
                json_data = json.loads(data)
                if isinstance(json_data, list):
                    results.extend(json_data)  # Use extend() for lists
                else:
                    results.append(json_data)  # Wrap the dict in a list

        if len(results) > 0:
            try:
                df = pd.DataFrame(results)
                st.subheader("Results")
                st.data_editor(df)
                if st.button("Sync to Make"):
                    send_to_make(results)
                    st.write("Synced to Make!")
            except Exception as e:
                st.error(f"An error occurred while creating the DataFrame: {e}")
                st.write(results)  # Print the data to see its content

    # st.set_page_config(page_title="PDF File Permissions Checker")

    # st.header("PDF File Permissions Checker")

    # uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    # if uploaded_file is not None:
    #     # Display absolute file path
    #     file_path = uploaded_file.name
    #     abs_file_path = os.path.abspath(file_path)
    #     st.write("Uploaded File Path:", abs_file_path)

    #     # Display file contents
    #     file_contents = uploaded_file.read()
    #     st.write("File Contents:")
    #     st.write(file_contents)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
