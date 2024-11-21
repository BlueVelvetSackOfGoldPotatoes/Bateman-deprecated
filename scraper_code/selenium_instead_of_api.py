import streamlit as st
import pandas as pd
import json
import uuid
import re
import time
import os
import random
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import ElementNotInteractableException, TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
# from webdriver_manager.utils import get_browser_version_from_os # DEPRECATED
from fake_useragent import UserAgent
from bs4 import BeautifulSoup
import openai
from openai import OpenAI
from streamlit_tags import st_tags
from selenium.webdriver.chrome.service import Service
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import tiktoken
from io import BytesIO
import logging

# Add these imports:
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# ==========================
# Configure Logging
# ==========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "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")

# Set SerpAPI Key
# SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY", "65a4ab6977ea5aa74cb95a5bf2a01df6c811d1a45784b2ba77a242341f0456e5")

# Set Serper API key
# SERPER_API_KEY = os.getenv("SERPER_API", "8737c516a0b54a948a09f868cf1a9c38dd12991e")

client = OpenAI(api_key=OPENAI_API_KEY)

# ==========================
# Define Utility Functions
# ==========================

def count_tokens(text, encoding_name='cl100k_base'):
    """
    Counts the number of tokens in a given text using the specified encoding.
    
    Parameters:
    - text (str): The text to count tokens for.
    - encoding_name (str): The name of the encoding to use.
    
    Returns:
    - int: The number of tokens.
    """
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))

def display_and_download(df, button_label, filename, height=400):
    """
    Displays the DataFrame in the UI with a fixed height and provides download buttons for CSV and Excel formats.

    Parameters:
    - df (pd.DataFrame): The DataFrame to display.
    - button_label (str): The label for the download buttons.
    - filename (str): The base name of the file to download.
    - height (int): The height of the dataframe display.
    """
    st.dataframe(df, height=height)  # Display the DataFrame with a fixed height

    # Convert DataFrame to CSV
    csv = df.to_csv(index=False).encode('utf-8')

    # Convert DataFrame to Excel using BytesIO
    excel_buffer = BytesIO()
    try:
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            df.to_excel(writer, index=False)
        excel_data = excel_buffer.getvalue()
    except Exception as e:
        st.error(f"Error converting DataFrame to Excel: {e}")
        logger.error(f"Error converting DataFrame to Excel: {e}")
        excel_data = None

    # Download buttons without keys
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
    Cleans the GPT response by removing any prefixes like 'json' or triple quotes.
    Extracts the JSON part of the response.
    """
    response_text = response_text.strip()
    response_text = re.sub(r"^'''json", "", response_text, flags=re.IGNORECASE)
    response_text = re.sub(r"'''$", "", response_text, flags=re.IGNORECASE)
    if response_text.lower().startswith('json'):
        response_text = response_text[4:].strip()
    match = re.search(r'(\[.*\]|\{.*\})', response_text, re.DOTALL)
    return match.group(1) if match else response_text

def scrape_landing_page_with_selenium(url):
    """
    Scrapes the landing page of a given URL and returns the cleaned text.

    Parameters:
    - url (str): The URL to scrape.

    Returns:
    - str: The cleaned text extracted from the page.
    """
    try:
        options = Options()
        # Rotate User-Agent
        ua = UserAgent()
        user_agent = ua.random
        options.add_argument(f'user-agent={user_agent}')

        # Comment out headless mode for testing
        # options.add_argument('--headless')

        options.add_argument('--disable-gpu')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-blink-features=AutomationControlled')  # Prevent detection
        options.add_argument("--window-size=1920,1080")  # Ensure consistent window size

        # Optional: Disable images for faster loading
        prefs = {"profile.managed_default_content_settings.images": 2}
        options.add_experimental_option("prefs", prefs)

        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        driver.get(url)

        # Random Delay
        time.sleep(random.uniform(2, 5))  # Wait for the page to load

        html = driver.page_source
        driver.quit()

        soup = BeautifulSoup(html, 'html.parser')
        for script_or_style in soup(["script", "style"]):
            script_or_style.extract()
        return soup.get_text(separator=' ')
    except Exception as e:
        st.error(f"Error scraping {url}: {e}")
        logger.error(f"Error scraping {url}: {e}")
        return ""

def perform_google_search(lead_name, num_results=3):
    """
    Performs a Google search for the given lead name and retrieves the specified number of result URLs.
    """
    try:
        options = Options()
        # Rotate User-Agent
        ua = UserAgent()
        user_agent = ua.random
        options.add_argument(f'user-agent={user_agent}')

        # Uncomment the line below to run in headless mode
        # options.add_argument('--headless')

        options.add_argument('--disable-gpu')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_argument("--window-size=1920,1080")

        # Optional: Disable images for faster loading
        prefs = {"profile.managed_default_content_settings.images": 2}
        options.add_experimental_option("prefs", prefs)

        # Initialize ChromeDriver
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        wait = WebDriverWait(driver, 30)  # Increased timeout

        driver.get("https://www.google.com")

        # Ensure screenshots directory exists
        if not os.path.exists('screenshots'):
            os.makedirs('screenshots')

        # Random Delay
        time.sleep(random.uniform(2, 4))

        # Handle Consent Dialog
        consent_handled = False
        try:
            # First, check if consent dialog is present without switching frames
            consent_button_selectors = [
                # German consent buttons
                "//button[.//div[contains(text(), 'Alle ablehnen')]]",
                "//button[.//div[contains(text(), 'Ich stimme zu')]]",
                "//button[.//div[contains(text(), 'Alles akzeptieren')]]",
                # English consent buttons
                "//button[.//div[contains(text(), 'I agree')]]",
                "//button[.//div[contains(text(), 'Accept all')]]",
                # By button ID
                "//button[@id='W0wltc']",
                # By class name
                "//button[contains(@class, 'tHlp8d')]",
                # Any button within the consent form
                "//button"
            ]

            # Attempt to click the consent button without switching frames
            for selector in consent_button_selectors:
                try:
                    consent_button = wait.until(EC.element_to_be_clickable((By.XPATH, selector)))
                    consent_button.click()
                    logger.info(f"Clicked consent button using selector: {selector}")
                    consent_handled = True
                    time.sleep(random.uniform(1, 2))
                    break
                except Exception as e:
                    logger.debug(f"Consent button not found with selector {selector}: {e}")
                    continue

            # If not found, try switching to the consent iframe
            if not consent_handled:
                try:
                    wait.until(EC.frame_to_be_available_and_switch_to_it((By.XPATH, "//iframe[contains(@src, 'consent')]")))
                    logger.info("Switched to consent iframe.")

                    for selector in consent_button_selectors:
                        try:
                            consent_button = wait.until(EC.element_to_be_clickable((By.XPATH, selector)))
                            consent_button.click()
                            logger.info(f"Clicked consent button within iframe using selector: {selector}")
                            consent_handled = True
                            time.sleep(random.uniform(1, 2))
                            break
                        except Exception as e:
                            logger.debug(f"Consent button not found within iframe with selector {selector}: {e}")
                            continue
                    driver.switch_to.default_content()
                except Exception as e:
                    logger.debug(f"Consent iframe not found: {e}")
                    driver.switch_to.default_content()

            if not consent_handled:
                logger.error("Consent button not found with any of the selectors.")
                # Save page source and screenshot for debugging
                page_source = driver.page_source
                with open(f"screenshots/consent_page_{uuid.uuid4()}.html", "w", encoding="utf-8") as f:
                    f.write(page_source)
                screenshot_path = f"screenshots/consent_error_{uuid.uuid4()}.png"
                driver.save_screenshot(screenshot_path)
                logger.info("Saved the consent page source and screenshot for debugging.")
                # Optionally, raise an exception or handle as needed

        except Exception as e:
            logger.error(f"Failed to handle consent dialog: {e}", exc_info=True)
            st.error(f"Failed to handle consent dialog: {e}")
            # Save page source and screenshot for debugging
            page_source = driver.page_source
            with open(f"screenshots/consent_page_{uuid.uuid4()}.html", "w", encoding="utf-8") as f:
                f.write(page_source)
            screenshot_path = f"screenshots/consent_error_{uuid.uuid4()}.png"
            driver.save_screenshot(screenshot_path)
            logger.info("Saved the consent page source and screenshot for debugging.")
            driver.switch_to.default_content()

        # Proceed to interact with the search box
        # Wait for any overlays to disappear
        try:
            wait.until(EC.invisibility_of_element((By.ID, "uMousc")))
            logger.info("Overlay 'uMousc' is no longer visible.")
        except Exception as e:
            logger.info(f"Overlay 'uMousc' not found or still visible: {e}")
            # Optionally, remove the overlay using JavaScript
            try:
                driver.execute_script("document.getElementById('uMousc').style.display = 'none';")
                logger.info("Removed overlay 'uMousc' using JavaScript.")
            except Exception as e:
                logger.error(f"Failed to remove overlay 'uMousc' using JavaScript: {e}")

        # Wait for Search Box
        search_box = wait.until(EC.visibility_of_element_located((By.NAME, "q")))
        # Scroll to the search box
        driver.execute_script("arguments[0].scrollIntoView();", search_box)
        time.sleep(random.uniform(0.5, 1.0))
        # Move to the search box and click
        actions = ActionChains(driver)
        actions.move_to_element(search_box).click().perform()
        search_box.clear()
        search_box.send_keys(lead_name)
        search_box.submit()

        # Random Delay
        time.sleep(random.uniform(2, 4))

        # Wait for Results
        wait.until(EC.presence_of_element_located((By.ID, "search")))

        # Retrieve Results
        results = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'div.yuRUbf a')))
        urls = [result.get_attribute('href') for result in results if result.get_attribute('href')]

        driver.quit()

        if not urls:
            st.warning(f"No URLs found for {lead_name}.")
            logger.warning(f"No URLs found for {lead_name}.")
        else:
            logger.info(f"Found {len(urls)} URLs for {lead_name}.")

        return urls[:num_results]

    except Exception as e:
        logger.error(f"Error during Google search for {lead_name}: {e}", exc_info=True)
        st.error(f"Error during Google search for {lead_name}: {e}")
        try:
            # Capture screenshot if driver is defined
            if 'driver' in locals():
                screenshot_path = f"screenshots/error_{uuid.uuid4()}.png"
                driver.save_screenshot(screenshot_path)
                logger.info(f"Screenshot saved to {screenshot_path}")
        except Exception as screenshot_error:
            logger.error(f"Failed to capture screenshot: {screenshot_error}")
        finally:
            driver.quit()
        return []

def extract_persons(leads_info_df):
    """
    Extracts unique persons associated with the leads from the leads_info_df DataFrame.
    
    Parameters:
    - leads_info_df (pd.DataFrame): DataFrame containing lead information with 'Contacts' field.
    
    Returns:
    - list of tuples: Each tuple contains the person's name and the associated lead.
    """
    persons = []
    for _, row in leads_info_df.iterrows():
        contacts = row.get('Contacts', [])
        if isinstance(contacts, str):
            try:
                contacts = json.loads(contacts)
            except json.JSONDecodeError:
                logger.warning(f"Failed to decode Contacts for lead: {row.get('Company/Group Name', '')}")
                continue
        for contact in contacts:
            name = contact.get('Name')
            if name:
                persons.append((name, row.get('Company/Group Name', '')))
    # Remove duplicates
    unique_persons = list(set(persons))
    return unique_persons

def extract_lead_info_with_gpt(text, columns_to_retrieve, lead_name, lead_category, source_urls):
    """
    Extracts specified information from the scraped text using GPT.

    Parameters:
    - text (str): The scraped text from the lead's website.
    - columns_to_retrieve (list): The information fields to extract.
    - lead_name (str): The name of the lead.
    - lead_category (str): The category/type of the lead.
    - source_urls (str): The URLs from which the information was scraped.

    Returns:
    - dict: Extracted information about the lead, including source URLs.
    """
    # Truncate text based on token count
    encoding = 'cl100k_base'
    model_max_tokens = 8192  # Update based on your model's token limit

    # Initial prompt without text
    initial_prompt = f"""
You are an AI assistant that extracts the following information from the given text:

[{', '.join([f'"{col}"' for col in columns_to_retrieve])}, "Contacts"]

Lead Name: {lead_name}
Lead Category: {lead_category}

Text:
"""

    # Estimate tokens for the prompt
    tokens_initial = count_tokens(initial_prompt, encoding)
    tokens_text = count_tokens(text, encoding)

    # Calculate available tokens for text
    available_tokens = model_max_tokens - tokens_initial - 1000  # Reserve tokens for response

    # Estimate words to fit into available tokens
    avg_words_per_token = 0.75
    max_words = int(available_tokens * avg_words_per_token)

    words = text.split()
    if len(words) > max_words:
        text = ' '.join(words[:max_words])
        st.warning(f"Text for {lead_name} was truncated to fit token limits.")
        logger.info(f"Truncated text for {lead_name} to {max_words} words.")

    prompt = f"""
You are an AI assistant that extracts the following information from the given text:

[{', '.join([f'"{col}"' for col in columns_to_retrieve])}, "Contacts"]

Lead Name: {lead_name}
Lead Category: {lead_category}

Text:
{text}

Please provide the extracted information in JSON format only. Ensure that all list fields contain only strings and that contacts are connected to specific individuals. For example, the "Contacts" field should be a list of objects with "Name", "Email", and "Phone Number" where applicable.

Output:
{{
    "Company/Group Name": "",
    "CEO/PI": "",
    "Researchers": ["Researcher 1", "Researcher 2"],
    "Grant Received": ["Amount: $500K, Date: July 1, 2020, Source: Phase III Start Grant", "Amount: $100K, Date: July 1, 2018, Source: MassVentures START Program"],
    "Date": "",
    "Country": "",
    "University": ["University 1", "University 2"],
    "Contacts": [
        {{
            "Name": "Person 1",
            "Email": "person1@example.com",
            "Phone Number": "123-456-7890"
        }},
        {{
            "Name": "Person 2",
            "Email": "person2@example.com",
            "Phone Number": "098-765-4321"
        }}
    ],
    "Summary": ""
}}
"""

    # Recalculate tokens for the final prompt
    total_tokens = count_tokens(prompt, encoding)
    if total_tokens > model_max_tokens:
        st.error(f"Lead info for {lead_name} exceeds the maximum token limit even after truncation. Skipping this lead.")
        logger.error(f"Lead info for {lead_name} exceeds the maximum token limit. Skipping.")
        return {
            "Company/Group Name": lead_name,
            "Category": lead_category,
            "Contacts": [],
            "Source URLs": source_urls
        }

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )

        lead_info_text = clean_response(response.choices[0].message.content)

        # Log the raw response
        logger.info(f"Raw gpt-4o Response for Lead Information Extraction ({lead_name}):")
        logger.info(lead_info_text)

        lead_info = json.loads(lead_info_text)
        lead_info['Source URLs'] = source_urls  # Add the source URLs to the lead info
    except json.JSONDecodeError:
        st.error("Failed to parse the response from GPT while extracting lead info.")
        logger.error("Failed to parse the response from GPT while extracting lead info.")
        logger.debug(f"Received Response: {lead_info_text}")
        lead_info = {
            "Company/Group Name": lead_name,
            "Category": lead_category,
            "Contacts": [],
            "Source URLs": source_urls
        }
    except Exception as e:
        st.error(f"An error occurred while extracting lead info for {lead_name}: {e}")
        logger.error(f"An error occurred while extracting lead info for {lead_name}: {e}")
        lead_info = {
            "Company/Group Name": lead_name,
            "Category": lead_category,
            "Contacts": [],
            "Source URLs": source_urls
        }

    return lead_info

def extract_person_info_with_gpt(text, person_name, source_urls):
    """
    Extracts specified information about a person from the scraped text using GPT.

    Parameters:
    - text (str): The scraped text from the person's webpage.
    - person_name (str): The name of the person.
    - source_urls (str): The URLs from which the information was scraped.

    Returns:
    - dict: Extracted information about the person, including source URLs.
    """
    # Truncate text based on token count
    encoding = 'cl100k_base'
    model_max_tokens = 8192  # Update based on your model's token limit

    # Define columns to retrieve
    columns_to_retrieve = ["Position", "Email", "Phone Number", "Affiliation", "Summary"]

    # Initial prompt without text
    initial_prompt = f"""
You are an AI assistant that extracts the following information from the given text:

[{', '.join([f'"{col}"' for col in columns_to_retrieve])}, "Contacts"]

Person Name: {person_name}

Text:
"""

    # Estimate tokens for the prompt
    tokens_initial = count_tokens(initial_prompt, encoding)
    tokens_text = count_tokens(text, encoding)

    # Calculate available tokens for text
    available_tokens = model_max_tokens - tokens_initial - 1000  # Reserve tokens for response

    # Estimate words to fit into available tokens
    avg_words_per_token = 0.75
    max_words = int(available_tokens * avg_words_per_token)

    words = text.split()
    if len(words) > max_words:
        text = ' '.join(words[:max_words])
        st.warning(f"Text for {person_name} was truncated to fit token limits.")
        logger.info(f"Truncated text for {person_name} to {max_words} words.")

    prompt = f"""
You are an AI assistant that extracts the following information from the given text:

[{', '.join([f'"{col}"' for col in columns_to_retrieve])}, "Contacts"]

Person Name: {person_name}

Text:
{text}

Please provide the extracted information in JSON format only. Ensure that all list fields contain only strings and that contacts are connected to specific individuals. For example, the "Contacts" field should be a list of objects with "Name", "Email", and "Phone Number" where applicable.

Output:
{{
    "Position": "",
    "Email": "",
    "Phone Number": "",
    "Affiliation": "",
    "Summary": ""
}}
"""

    # Recalculate tokens for the final prompt
    total_tokens = count_tokens(prompt, encoding)
    if total_tokens > model_max_tokens:
        st.error(f"Person info for {person_name} exceeds the maximum token limit even after truncation. Skipping this person.")
        logger.error(f"Person info for {person_name} exceeds the maximum token limit. Skipping.")
        return {
            "Position": "",
            "Email": "",
            "Phone Number": "",
            "Affiliation": "",
            "Summary": "",
            "Source URLs": source_urls
        }

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )

        person_info_text = clean_response(response.choices[0].message.content)

        # Log the raw response
        logger.info(f"Raw gpt-4o Response for Person Information Extraction ({person_name}):")
        logger.info(person_info_text)

        person_info = json.loads(person_info_text)
        person_info['Source URLs'] = source_urls  # Add the source URLs to the person info
    except json.JSONDecodeError:
        st.error("Failed to parse the response from GPT while extracting person info.")
        logger.error("Failed to parse the response from GPT while extracting person info.")
        logger.debug(f"Received Response: {person_info_text}")
        person_info = {
            "Position": "",
            "Email": "",
            "Phone Number": "",
            "Affiliation": "",
            "Summary": "",
            "Source URLs": source_urls
        }
    except Exception as e:
        st.error(f"An error occurred while extracting person info for {person_name}: {e}")
        logger.error(f"An error occurred while extracting person info for {person_name}: {e}")
        person_info = {
            "Position": "",
            "Email": "",
            "Phone Number": "",
            "Affiliation": "",
            "Summary": "",
            "Source URLs": source_urls
        }

    return person_info

def generate_leads_with_gpt(context, num_leads, lead_types):
    """
    Generates a list of leads using OpenAI's GPT model based on the provided context.

    Parameters:
    - context (str): The context used for search and filtering.
    - num_leads (int): The number of leads to generate per lead type.
    - lead_types (list): The types of leads to generate.

    Returns:
    - list of tuples: Each tuple contains the lead's name and type.
    """
    # Generate leads for each type
    leads = []
    for lead_type in lead_types:
        prompt = f"""
You are an AI assistant tasked with generating a list of {num_leads} unique and realistic {lead_type.lower()}s that are relevant to the following context:

{context}

Please provide the names along with their type in a JSON array. Ensure that the names are real-world {lead_type.lower()}s and avoid generic terms or scientific concepts. Each {lead_type.lower()} should be distinct and plausible.

Output:
[
    {{"name": "Name 1", "type": "{lead_type}"}},
    {{"name": "Name 2", "type": "{lead_type}"}},
    ...
]
"""
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5
            )

            names_text = clean_response(response.choices[0].message.content)

            # Log the raw response
            logger.info(f"Raw gpt-4o Response for Lead Generation ({lead_type}):")
            logger.info(names_text)

            lead_names = json.loads(names_text)
            # Append to the overall leads list
            leads.extend([(lead['name'], lead['type']) for lead in lead_names[:num_leads]])
        except json.JSONDecodeError:
            st.error(f"Failed to parse the response from GPT for {lead_type}. Please try again.")
            logger.error(f"Failed to parse the response from GPT for {lead_type}.")
            logger.debug(f"Received Response: {names_text}")
        except Exception as e:
            st.error(f"An error occurred while generating leads for {lead_type}: {e}")
            logger.error(f"An error occurred while generating leads for {lead_type}: {e}")

    return leads

def rank_leads(leads_df, weights):
    """
    Ranks the leads based on the presence of contact information, grants, and recency of the date.

    Parameters:
    - leads_df (pd.DataFrame): The DataFrame containing the leads information.
    - weights (dict): A dictionary containing weights for each criterion.

    Returns:
    - pd.DataFrame: The DataFrame with an additional 'Ranking' column.
    """
    current_year = time.localtime().tm_year

    def calculate_score(row):
        score = 0
        # Email
        if isinstance(row.get('Contacts'), str):
            try:
                contacts = json.loads(row['Contacts'])
                for contact in contacts:
                    if contact.get('Email') and isinstance(contact.get('Email'), str) and contact['Email'].strip().lower() != 'not provided':
                        score += weights['email']
            except:
                logger.warning(f"Failed to parse Contacts for ranking: {row.get('Company/Group Name', '')}")

        # Phone Number
        if isinstance(row.get('Contacts'), str):
            try:
                contacts = json.loads(row['Contacts'])
                for contact in contacts:
                    if contact.get('Phone Number') and isinstance(contact.get('Phone Number'), str) and contact['Phone Number'].strip().lower() != 'not provided':
                        score += weights['phone']
            except:
                logger.warning(f"Failed to parse Contacts for ranking: {row.get('Company/Group Name', '')}")

        # Grants Received
        if row.get('Grant Received'):
            # Count number of grants if it's a list, else check if string is not empty
            if isinstance(row['Grant Received'], list):
                grant_count = len(row['Grant Received'])
                score += weights['grants'] * grant_count
            elif isinstance(row['Grant Received'], str) and row['Grant Received'].strip():
                score += weights['grants']

        # Date
        if row.get('Date'):
            try:
                year = int(row['Date'])
                # Normalize the year score between 0 and 1
                year_score = (year - 2000) / (current_year - 2000)  # Assuming data from year 2000 onwards
                year_score = max(0, min(year_score, 1))  # Clamp between 0 and 1
                score += weights['date'] * year_score
            except:
                logger.warning(f"Invalid date format for ranking: {row.get('Date', '')}")
        return score

    leads_df['Ranking'] = leads_df.apply(calculate_score, axis=1)
    return leads_df

def clean_text(text):
    """
    Cleans the scraped text by removing extra whitespace.

    Parameters:
    - text (str): The scraped text.

    Returns:
    - str: The cleaned text.
    """
    return re.sub(r'\s+', ' ', text).strip()

def clean_dataframe(df):
    """
    Cleans the DataFrame by ensuring all columns have consistent types.
    Specifically, converts list-type entries to JSON strings.

    Parameters:
    - df (pd.DataFrame): The DataFrame to clean.

    Returns:
    - pd.DataFrame: The cleaned DataFrame.
    """
    for column in df.columns:
        # Check if any entry in the column is a list
        if df[column].apply(lambda x: isinstance(x, list)).any():
            # Convert all list entries to JSON strings for better preservation
            df[column] = df[column].apply(lambda x: json.dumps(x) if isinstance(x, list) else x)
    return df

def download_leads(leads_df, excel_filename):
    """
    Downloads the DataFrame as an Excel file.

    Parameters:
    - leads_df (pd.DataFrame): The DataFrame containing the leads data.
    - excel_filename (str): The desired filename for the Excel file.
    """
    try:
        # Ensure all data is string or numeric to prevent Excel issues
        leads_df = leads_df.fillna('').astype(str)
        # Export to Excel bytes using BytesIO
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            leads_df.to_excel(writer, index=False)
        excel_buffer.seek(0)
        # Generate unique key for download button
        unique_key = f"download_final_{excel_filename}"
        st.download_button(
            label="Download Final Excel File",
            data=excel_buffer,
            file_name=f"{excel_filename}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=unique_key
        )
    except Exception as e:
        st.error(f"Error while downloading Excel file: {e}")
        logger.error(f"Error while downloading Excel file: {e}")

def extract_person_info_with_gpt(text, person_name, source_urls):
    """
    Extracts specified information about a person from the scraped text using GPT.

    Parameters:
    - text (str): The scraped text from the person's webpage.
    - person_name (str): The name of the person.
    - source_urls (str): The URLs from which the information was scraped.

    Returns:
    - dict: Extracted information about the person, including source URLs.
    """
    # Truncate text based on token count
    encoding = 'cl100k_base'
    model_max_tokens = 8192  # Adjust based on your model's limit

    prompt = f"""
You are an AI assistant that extracts detailed information about a person from the given text.

Person Name: {person_name}

Text:
{text}

Please provide the extracted information in JSON format only. Include the following fields:
- "Name"
- "Education"
- "Current Position"
- "Work Area"
- "Hobbies"
- "Email"
- "Phone Number"
- "Active Grants"

Ensure that all fields are filled accurately based on the provided text. If certain information is not available, use "Not Available".

Output:
{{
    "Name": "",
    "Education": "",
    "Current Position": "",
    "Work Area": "",
    "Hobbies": "",
    "Email": "",
    "Phone Number": "",
    "Active Grants": ""
}}
"""

    total_tokens = count_tokens(prompt, encoding)
    if total_tokens > model_max_tokens:
        st.error(f"Person info for {person_name} exceeds the maximum token limit. Skipping this person.")
        logger.error(f"Person info for {person_name} exceeds the maximum token limit. Skipping.")
        return {
            "Name": person_name,
            "Education": "Not Available",
            "Current Position": "Not Available",
            "Work Area": "Not Available",
            "Hobbies": "Not Available",
            "Email": "Not Available",
            "Phone Number": "Not Available",
            "Active Grants": "Not Available",
            "Source URLs": source_urls
        }

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )

        person_info_text = clean_response(response.choices[0].message.content)

        # Log the raw response
        logger.info(f"Raw gpt-4o Response for Person Information Extraction ({person_name}):")
        logger.info(person_info_text)

        person_info = json.loads(person_info_text)
        person_info['Source URLs'] = source_urls  # Add the source URLs to the person info
    except json.JSONDecodeError:
        st.error("Failed to parse the response from GPT while extracting person info.")
        logger.error("Failed to parse the response from GPT while extracting person info.")
        logger.debug(f"Received Response: {person_info_text}")
        person_info = {
            "Name": person_name,
            "Education": "Not Available",
            "Current Position": "Not Available",
            "Work Area": "Not Available",
            "Hobbies": "Not Available",
            "Email": "Not Available",
            "Phone Number": "Not Available",
            "Active Grants": "Not Available",
            "Source URLs": source_urls
        }
    except Exception as e:
        st.error(f"An error occurred while extracting person info for {person_name}: {e}")
        logger.error(f"An error occurred while extracting person info for {person_name}: {e}")
        person_info = {
            "Name": person_name,
            "Education": "Not Available",
            "Current Position": "Not Available",
            "Work Area": "Not Available",
            "Hobbies": "Not Available",
            "Email": "Not Available",
            "Phone Number": "Not Available",
            "Active Grants": "Not Available",
            "Source URLs": source_urls
        }

    return person_info

def extract_persons(leads_info_df):
    """
    Extracts a list of unique persons from the leads information DataFrame.

    Parameters:
    - leads_info_df (pd.DataFrame): The DataFrame containing the leads information.

    Returns:
    - list of tuples: Each tuple contains the person's name and the associated lead.
    """
    persons = []
    for _, row in leads_info_df.iterrows():
        if 'Contacts' in row and isinstance(row['Contacts'], str):
            try:
                contacts = json.loads(row['Contacts'])
                for contact in contacts:
                    name = contact.get('Name', '').strip()
                    if name and name.lower() != 'not provided':
                        persons.append((name, row['Name']))
            except:
                continue
    # Remove duplicates
    unique_persons = list(set(persons))
    return unique_persons

def generate_person_leads(persons, num_person_leads=1):
    """
    Generates detailed leads for each person using GPT.

    Parameters:
    - persons (list of tuples): Each tuple contains the person's name and the associated lead.
    - num_person_leads (int): Number of detailed leads to generate per person.

    Returns:
    - list of dicts: Each dict contains the person's detailed information.
    """
    person_leads = []
    for person_name, associated_lead in persons:
        prompt = f"""
You are an AI assistant tasked with generating detailed information for the following person:

Person Name: {person_name}
Associated Lead: {associated_lead}

Please provide the detailed information as specified below in a JSON format.

Output:
{{
    "Name": "",
    "Education": "",
    "Current Position": "",
    "Work Area": "",
    "Hobbies": "",
    "Email": "",
    "Phone Number": "",
    "Active Grants": ""
}}
"""
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5
            )

            person_info_text = clean_response(response.choices[0].message.content)

            # Log the raw response
            logger.info(f"Raw gpt-4o Response for Person Lead Generation ({person_name}):")
            logger.info(person_info_text)

            person_info = json.loads(person_info_text)
            person_info['Associated Lead'] = associated_lead
            person_leads.append(person_info)
        except json.JSONDecodeError:
            st.error(f"Failed to parse the response from GPT-4o for {person_name}. Please try again.")
            logger.error(f"Failed to parse the response from GPT-4o for {person_name}.")
            logger.debug(f"Received Response: {person_info_text}")
        except Exception as e:
            st.error(f"An error occurred while generating person lead for {person_name}: {e}")
            logger.error(f"An error occurred while generating person lead for {person_name}: {e}")

    return person_leads

# ==========================
# Initialize Session State
# ==========================

if 'step' not in st.session_state:
    st.session_state['step'] = 1
if 'leads' not in st.session_state:
    st.session_state['leads'] = []
if 'leads_df' not in st.session_state:
    st.session_state['leads_df'] = pd.DataFrame()
if 'leads_info_df' not in st.session_state:
    st.session_state['leads_info_df'] = pd.DataFrame()
if 'ranked_leads_df' not in st.session_state:
    st.session_state['ranked_leads_df'] = pd.DataFrame()
if 'person_leads_df' not in st.session_state:
    st.session_state['person_leads_df'] = pd.DataFrame()
if 'excel_filename' not in st.session_state:
    st.session_state['excel_filename'] = "leads"

# ==========================
# Main Application Function
# ==========================

def main():
    st.set_page_config(layout="wide")  # Optional: Use wide layout for better visibility
    st.title("Sequential Lead Generation Tool")

    st.header("Lead Generation Console")

    # Step 1: Generate Leads
    if st.session_state['step'] >= 1:
        st.subheader("Step 1: Generate Leads")
        context = st.text_area("Context used for search and filtering:", height=150, key='context_input')
        num_leads_total = st.number_input("Total number of leads per type:", min_value=1, max_value=100, value=10, step=1, key='num_leads_total')
        default_lead_types = ["Companies", "Research Groups"]
        lead_types = st_tags(
            label='',
            text='Add or remove lead types:',
            value=default_lead_types,
            suggestions=default_lead_types,
            maxtags=10,
            key='lead_types'
        )
        generate_btn = st.button("Generate Leads", key='generate_leads_btn')
        if generate_btn:
            if not context.strip():
                st.error("Context cannot be empty.")
                logger.error("Lead generation attempted with empty context.")
            elif not lead_types:
                st.error("Please specify at least one lead type.")
                logger.error("Lead generation attempted without specifying lead types.")
            else:
                with st.spinner('Generating leads...'):
                    num_leads_per_type = num_leads_total
                    leads = generate_leads_with_gpt(context, num_leads_per_type, lead_types)
                if leads:
                    st.session_state['leads'] = leads
                    st.session_state['leads_df'] = pd.DataFrame(leads, columns=["Name", "Type"])
                    st.session_state['step'] = 2
                    st.success(f"Leads generated successfully! Total leads: {len(leads)}")
                    logger.info(f"Generated {len(leads)} leads.")
                else:
                    st.error("Failed to generate leads.")
                    logger.error("Lead generation failed.")

        # Display the Leads DataFrame if it exists
        if not st.session_state['leads_df'].empty:
            st.write("### Generated Leads")
            display_and_download(
                df=st.session_state['leads_df'],
                button_label="Leads",
                filename="leads"
            )

    # Step 2: Search and Scrape Lead Information
    # Step 2: Search and Scrape Lead Information
    if st.session_state['step'] >= 2:
        st.subheader("Step 2: Search and Scrape Lead Information")
        columns_to_retrieve = st_tags(
            label='',
            text='Add or remove information fields:',
            value=["Company/Group Name", "CEO/PI", "Researchers", "Grant Received", "Date", "Country", "University", "Email", "Phone Number", "Summary"],
            suggestions=["Company/Group Name", "CEO/PI", "Researchers", "Grant Received", "Date", "Country", "University", "Email", "Phone Number", "Summary"],
            maxtags=20,
            key='columns_to_retrieve'
        )
        search_btn = st.button("Search and Scrape Leads", key='search_leads_btn')
        if search_btn:
            if not st.session_state['leads']:
                st.error("No leads to search. Please generate leads first.")
                logger.error("Lead scraping attempted without any generated leads.")
            elif not columns_to_retrieve:
                st.error("Please specify at least one information field to retrieve.")
                logger.error("Lead scraping attempted without specifying information fields.")
            else:
                with st.spinner('Searching and scraping lead information...'):
                    leads_info = []
                    for idx, (lead_name, lead_category) in enumerate(st.session_state['leads']):
                        urls = perform_google_search(lead_name)
                        if not urls:
                            st.warning(f"No URLs found for {lead_name}.")
                            logger.warning(f"No URLs found for {lead_name}.")
                            continue
                        scraped_text = ""
                        sources = []
                        for url in urls[:3]:
                            text = scrape_landing_page_with_selenium(url)
                            if text:
                                scraped_text += text + " "
                                sources.append(url)
                        if not scraped_text.strip():
                            st.warning(f"No text scraped from URLs for {lead_name}.")
                            logger.warning(f"No text scraped from URLs for {lead_name}.")
                            continue
                        cleaned_text = clean_text(scraped_text)
                        source_urls = ', '.join(sources)
                        lead_info = extract_lead_info_with_gpt(cleaned_text, columns_to_retrieve, lead_name, lead_category, source_urls)
                        if lead_info:
                            leads_info.append(lead_info)
                    if leads_info:
                        st.session_state['leads_info_df'] = pd.DataFrame(leads_info)
                        st.session_state['step'] = 3
                        st.success("Lead information scraped successfully!")
                        logger.info("Lead information scraping completed successfully.")
                    else:
                        st.error("Failed to scrape lead information.")
                        logger.error("Lead information scraping failed.")

        # Display the Leads Information DataFrame if it exists
        if not st.session_state['leads_info_df'].empty:
            st.write("### Leads Information")
            display_and_download(
                df=st.session_state['leads_info_df'],
                button_label="Leads Information",
                filename="leads_info"
            )

    # Step 3: Rank the Leads (Automatic)
    if st.session_state['step'] >= 3:
        st.subheader("Step 3: Rank the Leads")
        st.markdown("**Configure Ranking Weights:**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            weight_email = st.slider("Email Weight", min_value=0.0, max_value=10.0, value=5.0, step=0.1, key='weight_email')
        with col2:
            weight_phone = st.slider("Phone Weight", min_value=0.0, max_value=10.0, value=5.0, step=0.1, key='weight_phone')
        with col3:
            weight_grants = st.slider("Grants Weight", min_value=0.0, max_value=10.0, value=3.0, step=0.1, key='weight_grants')
        with col4:
            weight_date = st.slider("Date Weight", min_value=0.0, max_value=10.0, value=2.0, step=0.1, key='weight_date')

        # Automatically rank leads after scraping
        if 'ranked_leads_df' not in st.session_state or st.session_state['ranked_leads_df'].empty:
            with st.spinner('Ranking leads...'):
                weights = {
                    'email': weight_email,
                    'phone': weight_phone,
                    'grants': weight_grants,
                    'date': weight_date
                }
                ranked_leads_df = rank_leads(st.session_state['leads_info_df'], weights)
                st.session_state['ranked_leads_df'] = ranked_leads_df
                st.session_state['step'] = 4
                st.success("Leads ranked successfully!")
                logger.info("Leads ranked successfully.")

        # Display the Ranked Leads DataFrame if it exists
        if not st.session_state['ranked_leads_df'].empty:
            st.write("### Ranked Leads")
            display_and_download(
                df=st.session_state['ranked_leads_df'],
                button_label="Ranked Leads",
                filename="ranked_leads"
            )

    # Step 4: Download Final Leads Data
    if st.session_state['step'] >= 4:
        st.subheader("Step 4: Download Final Leads Data")
        excel_filename = st.text_input(
            "Enter a name for the Excel file (including .xlsx):",
            value=st.session_state['excel_filename'],
            key='excel_input'
        )
        download_btn = st.button("Download Final Excel File", key='download_excel_btn')
        if download_btn:
            if st.session_state['ranked_leads_df'].empty:
                st.error("No leads to download. Please generate and rank leads first.")
                logger.error("Download attempted without any ranked leads.")
            else:
                download_leads(st.session_state['ranked_leads_df'], excel_filename)
                logger.info(f"Final leads data downloaded as {excel_filename}.xlsx")

    # Step 5: Extract and Scrape Persons Associated with Leads
    if st.session_state['step'] >= 4:
        st.subheader("Step 5: Extract and Scrape Persons Associated with Leads")
        extract_persons_btn = st.button("Extract and Scrape Persons", key='extract_persons_btn')
        if extract_persons_btn:
            with st.spinner('Extracting persons associated with leads...'):
                persons = extract_persons(st.session_state['leads_info_df'])
                if not persons:
                    st.warning("No persons found associated with the leads.")
                    logger.warning("No persons found associated with the leads.")
                else:
                    st.success(f"Found {len(persons)} unique persons associated with the leads.")
                    logger.info(f"Found {len(persons)} unique persons associated with the leads.")
            if persons:
                with st.spinner('Generating detailed person information...'):
                    person_leads = []
                    for person_name, associated_lead in persons:
                        # Perform Google search for each person
                        person_urls = perform_google_search(person_name)
                        if not person_urls:
                            st.warning(f"No URLs found for {person_name}.")
                            logger.warning(f"No URLs found for {person_name}.")
                            continue
                        scraped_text = ""
                        sources = []
                        # Use only the first 3 URLs
                        for url in person_urls[:3]:
                            text = scrape_landing_page_with_selenium(url)
                            if text:
                                scraped_text += text + " "
                                sources.append(url)
                        if not scraped_text.strip():
                            st.warning(f"No text scraped from URLs for {person_name}.")
                            logger.warning(f"No text scraped from URLs for {person_name}.")
                            continue
                        cleaned_text = clean_text(scraped_text)
                        # Combine all sources into a single string separated by commas
                        source_urls = ', '.join(sources)
                        person_info = extract_person_info_with_gpt(cleaned_text, person_name, source_urls)
                        if person_info:
                            person_leads.append(person_info)
                    if person_leads:
                        person_leads_df = pd.DataFrame(person_leads)
                        st.session_state['person_leads_df'] = person_leads_df
                        st.success("Person information scraped successfully!")
                        logger.info("Person information scraped successfully.")
                        
                        # Display the Person Leads DataFrame
                        display_and_download(
                            df=st.session_state['person_leads_df'],
                            button_label="Person Leads",
                            filename="person_leads"
                        )
                    else:
                        st.error("Failed to scrape person information.")
                        logger.error("Person information scraping failed.")

    # Step 6: Download Person Leads Data
    if st.session_state['step'] >= 4 and not st.session_state['person_leads_df'].empty:
        st.subheader("Step 6: Download Person Leads Data")
        person_excel_filename = st.text_input(
            "Enter a name for the Persons Excel file (including .xlsx):",
            value="person_leads.xlsx",
            key='person_excel_input'
        )
        person_download_btn = st.button("Download Person Leads Data", key='download_person_excel_btn')
        if person_download_btn:
            # Update the session state with the new filename
            st.session_state['excel_filename'] = person_excel_filename
            download_leads(st.session_state['person_leads_df'], person_excel_filename)
            logger.info(f"Person leads data downloaded as {person_excel_filename}.xlsx")

    # ==========================
    # Analytics Section
    # ==========================
    if st.session_state['step'] >= 4:
        st.subheader("Analytics")
        analytics_expander = st.expander("View Analytics")
        with analytics_expander:
            # Number of Grants
            if 'Grant Received' in st.session_state['ranked_leads_df'].columns:
                st.markdown("### Number of Grants per Lead")
                def count_grants(grant_entry):
                    if isinstance(grant_entry, list):
                        return len(grant_entry)
                    elif isinstance(grant_entry, str) and grant_entry.strip():
                        return 1
                    else:
                        return 0
                st.session_state['ranked_leads_df']['Grant Count'] = st.session_state['ranked_leads_df']['Grant Received'].apply(count_grants)
                grant_counts = st.session_state['ranked_leads_df']['Grant Count']

                # Plotting with Matplotlib
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(st.session_state['ranked_leads_df']['Name'], grant_counts)
                ax.set_xlabel('Lead Name')
                ax.set_ylabel('Number of Grants')
                ax.set_title('Number of Grants per Lead')
                ax.set_xticklabels(st.session_state['ranked_leads_df']['Name'], rotation=90)
                ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
                plt.tight_layout()
                st.pyplot(fig)

            # Grant Dates
            if 'Grant Received' in st.session_state['ranked_leads_df'].columns:
                st.markdown("### Grant Dates Distribution")
                def extract_years(grant_entry):
                    years = []
                    if isinstance(grant_entry, list):
                        for grant in grant_entry:
                            match = re.search(r'\b(19|20)\d{2}\b', grant)
                            if match:
                                years.append(int(match.group()))
                    elif isinstance(grant_entry, str):
                        match = re.search(r'\b(19|20)\d{2}\b', grant_entry)
                        if match:
                            years.append(int(match.group()))
                    return years

                st.session_state['ranked_leads_df']['Grant Years'] = st.session_state['ranked_leads_df']['Grant Received'].apply(extract_years)
                all_grant_years = [year for sublist in st.session_state['ranked_leads_df']['Grant Years'] for year in sublist]
                if all_grant_years:
                    grant_years_df = pd.DataFrame(all_grant_years, columns=['Year'])
                    grant_years_df_sorted = grant_years_df['Year'].value_counts().sort_index()

                    # Plotting with Matplotlib
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(grant_years_df_sorted.index, grant_years_df_sorted.values, marker='o')
                    ax.set_xlabel('Year')
                    ax.set_ylabel('Number of Grants')
                    ax.set_title('Grant Dates Distribution Over Years')
                    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
                    plt.tight_layout()
                    st.pyplot(fig)

            # Number of Contacts
            st.markdown("### Contact Information Summary")
            contacts_df = st.session_state['ranked_leads_df'][['Contacts']]
            def count_emails(contacts):
                try:
                    contacts_list = json.loads(contacts)
                    return sum(1 for contact in contacts_list if contact.get('Email') and contact['Email'].strip().lower() != 'not provided')
                except:
                    return 0
            def count_phones(contacts):
                try:
                    contacts_list = json.loads(contacts)
                    return sum(1 for contact in contacts_list if contact.get('Phone Number') and contact['Phone Number'].strip().lower() != 'not provided')
                except:
                    return 0
            num_emails = contacts_df['Contacts'].apply(count_emails).sum()
            num_phones = contacts_df['Contacts'].apply(count_phones).sum()
            st.write(f"**Total Emails Provided:** {int(num_emails)}")
            st.write(f"**Total Phone Numbers Provided:** {int(num_phones)}")

            # Number of Researchers
            st.markdown("### Number of Researchers per Lead")
            def count_researchers(researchers):
                if isinstance(researchers, list):
                    return len(researchers)
                elif isinstance(researchers, str) and researchers.strip():
                    return 1
                else:
                    return 0
            st.session_state['ranked_leads_df']['Researcher Count'] = st.session_state['ranked_leads_df']['Researchers'].apply(count_researchers)
            researcher_counts = st.session_state['ranked_leads_df']['Researcher Count']

            # Plotting with Matplotlib
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(st.session_state['ranked_leads_df']['Name'], researcher_counts)
            ax.set_xlabel('Lead Name')
            ax.set_ylabel('Number of Researchers')
            ax.set_title('Number of Researchers per Lead')
            ax.set_xticklabels(st.session_state['ranked_leads_df']['Name'], rotation=90)
            ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
            plt.tight_layout()
            st.pyplot(fig)

            # Word Cloud for Summaries
            st.markdown("### Word Cloud for Summaries")
            summaries = st.session_state['ranked_leads_df']['Summary'].dropna().tolist()
            combined_text = ' '.join(summaries)
            if combined_text:
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(combined_text)
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
            else:
                st.write("No summaries available for word cloud.")

            # Scraped Websites
            st.markdown("### Scraped Websites")
            if 'Source URLs' in st.session_state['ranked_leads_df'].columns:
                scraped_websites = st.session_state['ranked_leads_df']['Source URLs']
                # Display as a list with clickable links
                for idx, urls in enumerate(scraped_websites):
                    url_list = urls.split(', ')
                    st.markdown(f"**Lead {idx+1}:**")
                    for url in url_list:
                        st.markdown(f"- [{url}]({url})")
            else:
                st.write("No scraped websites data available.")

            # Additional Analytics for Person Leads
            if not st.session_state['person_leads_df'].empty:
                st.markdown("### Number of Active Grants per Person")
                def count_active_grants(grant_entry):
                    if isinstance(grant_entry, str) and grant_entry.strip().lower() != 'not available':
                        return 1
                    elif isinstance(grant_entry, list):
                        return len(grant_entry)
                    else:
                        return 0
                st.session_state['person_leads_df']['Active Grants Count'] = st.session_state['person_leads_df']['Active Grants'].apply(count_active_grants)
                active_grants_counts = st.session_state['person_leads_df']['Active Grants Count']

                # Plotting with Matplotlib
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(st.session_state['person_leads_df']['Name'], active_grants_counts)
                ax.set_xlabel('Person Name')
                ax.set_ylabel('Number of Active Grants')
                ax.set_title('Number of Active Grants per Person')
                ax.set_xticklabels(st.session_state['person_leads_df']['Name'], rotation=90)
                ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
                plt.tight_layout()
                st.pyplot(fig)

                # Scraped Websites for Persons
                st.markdown("### Scraped Websites for Persons")
                scraped_person_websites = st.session_state['person_leads_df']['Source URLs']
                # Display as a list with clickable links
                for idx, urls in enumerate(scraped_person_websites):
                    url_list = urls.split(', ')
                    st.markdown(f"**Person {idx+1}:**")
                    for url in url_list:
                        st.markdown(f"- [{url}]({url})")

# ==========================
# Run the Application
# ==========================

# Remove the if __name__ == '__main__': block to ensure Streamlit executes the script correctly
main()