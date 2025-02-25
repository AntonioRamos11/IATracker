import hashlib
from pathlib import Path
import requests
import re

def store_pdf(pdf_url: str, source: str, title: str) -> str:
    """Store PDF with content-based hashing and title in filename"""
    response = requests.get(pdf_url, stream=True)
    content = response.content
    file_hash = hashlib.sha256(content).hexdigest()[:16]
    
    # Clean the title to create a valid filename
    clean_title = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')
    #year later
    #year = f"20{pdf_url[2:4]}" if len(pdf_url) > 4 else "unknown"
    # Create directory structure
    save_path = Path(f"papers/{source}/{clean_title}_{file_hash}.pdf")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if the file already exists
    if save_path.exists():
        return str(save_path)
    
    with open(save_path, "wb") as f:
        f.write(content)
    
    return str(save_path)