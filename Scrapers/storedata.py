import hashlib
from pathlib import Path
import requests
import re
import mimetypes
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import RateLimiter
import random
limiter = RateLimiter(max_calls=3, period=300)  # 3 solicitudes cada 5 minutos

def safe_download(url, max_retries=3):
    for _ in range(max_retries):
        limiter.wait()
        
        response = requests.get(url, stream=True)

        
        if response and response.status_code == 200:
            return response.content
        
        if "CAPTCHA" in response.text:
            captcha_solution = solve_captcha()
            # Implementar lógica para enviar solución de CAPTCHA
        
        time.sleep(random.uniform(2, 5))
    
    logger.error(f"Failed to download: {url}")
    return None

def is_html_content(content: bytes) -> bool:
    """Check if the content is HTML based on its initial bytes."""
    return content.strip().startswith(b'<!DOCTYPE html>') or b'<html' in content[:1000].lower()

def store_pdf(pdf_url: str, source: str, title: str) -> str:
    """Store PDF with content-based hashing and title in filename"""

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