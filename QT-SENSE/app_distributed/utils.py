# utils.py

import re
import json
import logging
import pandas as pd
from io import BytesIO
import tiktoken
import streamlit as st

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def count_tokens(text, encoding_name='cl100k_base'):
    """
    Counts the number of tokens in a given text using the specified encoding.
    """
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))

def clean_dataframe(df):
    """
    Converts list-type columns into JSON strings for better display and download.
    """
    for column in df.columns:
        if df[column].apply(lambda x: isinstance(x, list)).any():
            df[column] = df[column].apply(lambda x: json.dumps(x) if isinstance(x, list) else x)
    return df

def display_and_download(df, button_label, filename, height=400):
    """
    Displays a DataFrame and provides options to download it as CSV or Excel.
    """
    df = clean_dataframe(df)
    st.dataframe(df, height=height)

    csv = df.to_csv(index=False).encode('utf-8')

    excel_buffer = BytesIO()
    try:
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            df.to_excel(writer, index=False)
        excel_data = excel_buffer.getvalue()
    except Exception as e:
        logger.error(f"Error converting DataFrame to Excel: {e}")
        excel_data = None

    st.download_button(
        label=f"Download {button_label} as CSV",
        data=csv,
        file_name=f"{filename}.csv",
        mime='text/csv'
    )

    if excel_data:
        st.download_button(
            label=f"Download {button_label} as Excel",
            data=excel_data,
            file_name=f"{filename}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

def clean_response(response_text):
    """
    Cleans the GPT response to extract valid JSON.
    """
    response_text = response_text.strip()
    response_text = re.sub(r"^'''json", "", response_text, flags=re.IGNORECASE)
    response_text = re.sub(r"'''$", "", response_text, flags=re.IGNORECASE)
    if response_text.lower().startswith('json'):
        response_text = response_text[4:].strip()
    match = re.search(r'(\[.*\]|\{.*\})', response_text, re.DOTALL)
    return match.group(1) if match else response_text

def clean_text(text):
    """
    Cleans and normalizes text data.
    """
    return re.sub(r'\s+', ' ', text).strip()

def download_leads(leads_df, excel_filename):
    """
    Facilitates the download of DataFrames as Excel files.
    """
    try:
        leads_df = leads_df.fillna('').astype(str)
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            leads_df.to_excel(writer, index=False)
        excel_buffer.seek(0)
        unique_key = f"download_final_{excel_filename}"
        st.download_button(
            label="Download Excel File",
            data=excel_buffer,
            file_name=f"{excel_filename}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=unique_key
        )
    except Exception as e:
        logger.error(f"Error while downloading Excel file: {e}")
