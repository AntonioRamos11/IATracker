import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import random
import time
import logging
import undetected_chromedriver as uc
from pathlib import Path
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArXivDownloader:
    def __init__(self):
        self.session = requests.Session()
        self._setup_session()
        self.captcha_auth = None  # Initialize CAPTCHA authentication mechanism
    
    def _setup_session(self):
        # Configure retries and connection policies
        retry = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("https://", adapter)
        
        # Rotate User-Agents
        self.user_agents = [
            'Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0'
        ]
        
    def _get_headers(self, pdf_url):
        """Generate specific headers for each request"""
        return {
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br, zstd',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'DNT': '1',
            'Referer': pdf_url,  # Critical to avoid detection
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'same-origin',
            'Priority': 'u=0, i'
        }
    
    def _handle_captcha(self, pdf_url):
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
    
    def download_pdf(self, arxiv_id):
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
        
        try:
            # First request to establish cookies
            if not self.captcha_auth:
                self.session.get("https://arxiv.org", headers=self._get_headers(pdf_url))
                
            # Main request with exponential backoff
            response = self.session.get(
                pdf_url,
                headers=self._get_headers(pdf_url),
                cookies={'captchaAuth': self.captcha_auth} if self.captcha_auth else None,
                timeout=10
            )
            
            if "captcha" in response.text.lower():
                self.captcha_auth = self._handle_captcha(pdf_url)
                return self.download_pdf(arxiv_id)
                
            if response.status_code == 304:
                logger.info("PDF not modified, using cache")
                return None  # Implement cache logic
                
            return response.content
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Download error: {str(e)}")
            time.sleep(random.uniform(2, 5))
            return self.download_pdf(arxiv_id)