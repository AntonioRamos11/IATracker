import hashlib
from pathlib import Path
import requests
import re
import mimetypes
import logging
import time
import random
import ratelimiter as ratelimiter
import undetected_chromedriver as uc
from ratelimiter import RateLimiter
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from fake_useragent import UserAgent
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

def get_random_headers():
    ua = UserAgent()
    return {    

        'User-Agent': ua.random,
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Referer': 'https://arxiv.org/',
        'DNT': '1'  # Do Not Track
    }

limiter = RateLimiter(max_calls=3, period=300)  # 3 requests every 5 minutes


def log_request_details(response, *args, **kwargs):
    logger.info(f"Request URL: {response.request.url}")
    logger.info(f"Request Headers: {response.request.headers}")
    logger.info(f"Request Body: {response.request.body}")
    logger.info(f"Response Status Code: {response.status_code}")
    logger.info(f"Response Headers: {response.headers}")

def solve_captcha_manually(pdf_url):
    """Opens the URL in a real browser to solve the CAPTCHA manually."""
    options = uc.ChromeOptions()
    prefs = {"download.default_directory": str(Path("/home/p0wden/Documents/IAResearchAgregator/papers/arxiv/").resolve())}
    options.add_experimental_option("prefs", prefs)


    driver = uc.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    driver.get(pdf_url)
    print("Solve the CAPTCHA manually, then press Enter in the terminal...")
    input()  # Wait for user confirmation

    final_url = driver.current_url
    driver.quit()

    logger.info(f"Final URL after solving CAPTCHA: {final_url}")
    
    return final_url if final_url.endswith(".pdf") else None

def safe_download(url, max_retries=3):
    for _ in range(max_retries):
        limiter.wait()
    
 
        response = requests.get(url, stream=True, headers=get_random_headers(), hooks={'response': log_request_details})

        if response and response.status_code == 200:
            if "CAPTCHA" in response.text:
                captcha_solution = solve_captcha_manually(url)
            else:
                return response.content
        
       
        
        time.sleep(random.uniform(2, 5))
    
    logger.error(f"Failed to download: {url}")
    return None

def is_html_content(content: bytes) -> bool:
    """Check if the content is HTML based on its initial bytes."""
    return content.strip().startswith(b'<!DOCTYPE html>') or b'<html' in content[:1000].lower()

def store_pdf(pdf_url: str, source: str, title: str) -> str:
    """Store PDF with content-based hashing and title in filename"""
    content = safe_download(pdf_url)
    
    if content is None:
        logger.error(f"Failed to download content from {pdf_url}")
        return None

    # Check if the content is HTML
    if is_html_content(content):
        logger.warning(f"The content from {pdf_url} is HTML, not a PDF.")
        return None

    file_hash = hashlib.sha256(content).hexdigest()[:16]
    
    # Clean the title to create a valid filename
    clean_title = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_').replace('\n', '')
    # Create directory structure
    save_path = Path(f"papers/{source}/{clean_title}_{file_hash}.pdf")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if the file already exists
    if save_path.exists():
        return str(save_path)
    
    with open(save_path, "wb") as f:
        f.write(content)
    
    return str(save_path)