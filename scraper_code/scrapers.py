# scrapers.py

import time
import logging
import traceback
from typing import Tuple, Optional
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.common.exceptions import (
    NoSuchElementException,
    TimeoutException,
    WebDriverException,
)
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from pdf_utils import download_pdf, sanitize_filename, extract_metadata
from bs4 import BeautifulSoup
import re
import os
from tqdm import tqdm

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Clear existing handlers
if logger.hasHandlers():
    logger.handlers.clear()

# Create console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# Create formatter and add to handler
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)

# Add handler to logger
logger.addHandler(ch)

def scrape_page_articles_rsc(
    url: str,
    output_folder: str,
    csv_path: str,
    journal_name: str,
    vpn_index: int,
) -> Tuple[Optional[str], int, int]:
    """Scrape articles from RSC's Themed Collections pages and download PDFs using Selenium."""
    count = 0

    # Set up Selenium WebDriver with headless Chrome using WebDriver Manager
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in headless mode
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/58.0.3029.110 Safari/537.3"
    )

    try:
        # Create a Service object
        service = Service(ChromeDriverManager().install())
        # Initialize WebDriver with service and options
        driver = webdriver.Chrome(service=service, options=chrome_options)
        logger.info("WebDriver initialized successfully.")
    except WebDriverException as e:
        logger.error(f"Error initializing WebDriver: {e}")
        return None, count, vpn_index

    driver.set_page_load_timeout(30)  # Set timeout

    try:
        driver.get(url)
        logger.info(f"Navigated to {url}")
    except TimeoutException:
        logger.error(f"Timeout while loading page: {url}")
        driver.quit()
        return None, count, vpn_index

    # Handle potential pop-ups (e.g., cookie consent)
    try:
        consent_button = driver.find_element(By.ID, "onetrust-accept-btn-handler")
        consent_button.click()
        logger.info("Dismissed cookie consent.")
        time.sleep(2)
    except NoSuchElementException:
        logger.info("No cookie consent button found.")

    # Allow time for JavaScript to render
    time.sleep(5)  # Wait for 5 seconds

    # Extract page source and parse with BeautifulSoup
    page_source = driver.page_source
    soup = BeautifulSoup(page_source, 'lxml')

    # Find all paper entries
    paper_entries = soup.find_all('div', class_='capsule')

    logger.info(f"Found {len(paper_entries)} papers on this page.")

    if not paper_entries:
        logger.info("No papers found on this page.")
        driver.quit()
        return None, count, vpn_index

    # Prepare list of DOIs to process
    dois = []
    for entry in paper_entries:
        # Extract DOI
        doi_tag = entry.find('a', href=re.compile(r'https?://doi\.org/'))
        doi = doi_tag['href'].split('/')[-1] if doi_tag else "N/A"

        if doi == "N/A":
            logger.warning("No DOI found for an article. Skipping.")
            continue

        dois.append(doi)

    if not dois:
        logger.info("No papers found on this page.")
        driver.quit()
        return None, count, vpn_index

    # Remove already downloaded DOIs by checking if JSON file exists
    filtered_dois = []
    for doi_clean in dois:
        json_path = os.path.join("./data/rsc/doi", f"{sanitize_filename(doi_clean)}.json")
        if not os.path.exists(json_path):
            filtered_dois.append(doi_clean)
        else:
            logger.info(f"Already downloaded DOI: {doi_clean}. Skipping.")

    if not filtered_dois:
        logger.info("All papers on this page have already been downloaded.")
        driver.quit()
        return None, count, vpn_index

    # Initialize tqdm progress bar
    with tqdm(total=len(filtered_dois), desc="Downloading PDFs", unit="pdf") as pbar:
        # Process each paper sequentially
        for doi_clean in filtered_dois:
            pdf_url = construct_pdf_url(doi_clean)
            if not pdf_url:
                logger.warning(f"Could not construct PDF URL for DOI: {doi_clean}. Skipping.")
                pbar.update(1)
                continue

            success = download_pdf(pdf_url, output_folder, csv_path, doi_clean)
            if success:
                count += 1
                # Extract metadata and save
                extract_metadata(doi_clean, csv_path)
                logger.info(f"Downloaded and processed PDF for DOI: {doi_clean}")
            pbar.update(1)
            time.sleep(1)  # Optional delay to prevent rate limiting

    driver.quit()
    logger.info(f"Finished downloading articles from {url}. Total PDFs downloaded: {count}")
    return None, count, vpn_index

def scrape_issue_page_rsc(
    url: str,
    output_folder: str,
    csv_path: str,
    journal_name: str,
    vpn_index: int,
) -> Tuple[Optional[str], int, int]:
    """Scrape articles from RSC's Specific Issue pages and navigate through previous issues."""
    count = 0

    # Set up Selenium WebDriver with headless Chrome using WebDriver Manager
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in headless mode
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/58.0.3029.110 Safari/537.3"
    )

    try:
        # Create a Service object
        service = Service(ChromeDriverManager().install())
        # Initialize WebDriver with service and options
        driver = webdriver.Chrome(service=service, options=chrome_options)
        logger.info("WebDriver initialized successfully.")
    except WebDriverException as e:
        logger.error(f"Error initializing WebDriver: {e}")
        return None, count, vpn_index

    driver.set_page_load_timeout(30)  # Set timeout

    try:
        driver.get(url)
        logger.info(f"Navigated to {url}")
    except TimeoutException:
        logger.error(f"Timeout while loading page: {url}")
        driver.quit()
        return None, count, vpn_index

    # Handle potential pop-ups (e.g., cookie consent)
    try:
        consent_button = driver.find_element(By.ID, "onetrust-accept-btn-handler")
        consent_button.click()
        logger.info("Dismissed cookie consent.")
        time.sleep(2)
    except NoSuchElementException:
        logger.info("No cookie consent button found.")

    # Allow time for JavaScript to render
    time.sleep(5)  # Wait for 5 seconds

    # Extract page source and parse with BeautifulSoup
    page_source = driver.page_source
    soup = BeautifulSoup(page_source, 'lxml')

    # Find all paper entries
    paper_entries = soup.find_all('div', class_='capsule')

    logger.info(f"Found {len(paper_entries)} papers on this page.")

    if not paper_entries:
        logger.info("No papers found on this page.")
        driver.quit()
        return None, count, vpn_index

    # Prepare list of DOIs to process
    dois = []
    for entry in paper_entries:
        # Extract DOI
        doi_tag = entry.find('a', href=re.compile(r'https?://doi\.org/'))
        doi = doi_tag['href'].split('/')[-1] if doi_tag else "N/A"

        if doi == "N/A":
            logger.warning("No DOI found for an article. Skipping.")
            continue

        dois.append(doi)

    if not dois:
        logger.info("No papers found on this page.")
        driver.quit()
        return None, count, vpn_index

    # Remove already downloaded DOIs by checking if JSON file exists
    filtered_dois = []
    for doi_clean in dois:
        json_path = os.path.join("./data/rsc/doi", f"{sanitize_filename(doi_clean)}.json")
        if not os.path.exists(json_path):
            filtered_dois.append(doi_clean)
        else:
            logger.info(f"Already downloaded DOI: {doi_clean}. Skipping.")

    if not filtered_dois:
        logger.info("All papers on this page have already been downloaded.")
        driver.quit()
        return None, count, vpn_index

    # Initialize tqdm progress bar
    with tqdm(total=len(filtered_dois), desc="Downloading PDFs", unit="pdf") as pbar:
        # Process each paper sequentially
        for doi_clean in filtered_dois:
            pdf_url = construct_pdf_url(doi_clean)
            if not pdf_url:
                logger.warning(f"Could not construct PDF URL for DOI: {doi_clean}. Skipping.")
                pbar.update(1)
                continue

            success = download_pdf(pdf_url, output_folder, csv_path, doi_clean)
            if success:
                count += 1
                # Extract metadata and save
                extract_metadata(doi_clean, csv_path)
                logger.info(f"Downloaded and processed PDF for DOI: {doi_clean}")
            pbar.update(1)
            time.sleep(1)  # Optional delay to prevent rate limiting

    driver.quit()
    logger.info(f"Finished downloading articles from {url}. Total PDFs downloaded: {count}")
    return None, count, vpn_index

def construct_pdf_url(doi_clean: str) -> Optional[str]:
    """Construct the correct PDF URL based on DOI."""
    # For RSC journals, the PDF URL can be constructed using the DOI
    # Example DOI: D3OB00345A
    # The URL pattern is: https://pubs.rsc.org/en/content/articlepdf/{year}/{journal_code}/{doi}
    # Need to extract the year and journal code from the DOI

    # Extract year and journal code from DOI
    match = re.match(r'D(\d)([A-Z]{2})(\d+)([A-Z])', doi_clean)
    if not match:
        logger.warning(f"DOI format is unexpected: {doi_clean}. Cannot construct PDF URL.")
        return None

    year_digit = match.group(1)  # e.g., '3' for 2023
    journal_code = match.group(2).lower()  # e.g., 'ob' for Organic & Biomolecular Chemistry
    year = f"202{year_digit}"

    pdf_url = f"https://pubs.rsc.org/en/content/articlepdf/{year}/{journal_code}/{doi_clean.lower()}"
    return pdf_url
