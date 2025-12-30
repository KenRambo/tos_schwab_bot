"""
Automated Schwab Authentication using Headless Browser

This script automates the OAuth flow using Selenium, eliminating the need
to manually copy/paste the redirect URL.

Requirements:
    pip install selenium webdriver-manager

Usage:
    python auto_auth.py --username YOUR_SCHWAB_USERNAME --password YOUR_SCHWAB_PASSWORD

Or set environment variables:
    export SCHWAB_USERNAME='your_username'
    export SCHWAB_PASSWORD='your_password'
    python auto_auth.py
"""
import os
import sys
import time
import json
import argparse
import logging
from urllib.parse import urlparse, parse_qs
from pathlib import Path

logger = logging.getLogger(__name__)

# Check for selenium
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium.common.exceptions import TimeoutException, NoSuchElementException
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

try:
    from webdriver_manager.chrome import ChromeDriverManager
    WEBDRIVER_MANAGER_AVAILABLE = True
except ImportError:
    WEBDRIVER_MANAGER_AVAILABLE = False

from schwab_auth import SchwabAuth


class AutomatedAuth:
    """Automates Schwab OAuth flow using headless browser"""
    
    AUTH_URL = "https://api.schwabapi.com/v1/oauth/authorize"
    
    def __init__(
        self,
        app_key: str,
        app_secret: str,
        redirect_uri: str = "https://127.0.0.1:8182/callback",
        headless: bool = True,
        timeout: int = 60
    ):
        self.app_key = app_key
        self.app_secret = app_secret
        self.redirect_uri = redirect_uri
        self.headless = headless
        self.timeout = timeout
        self.driver = None
    
    def _setup_driver(self):
        """Setup Chrome WebDriver"""
        options = Options()
        
        if self.headless:
            options.add_argument('--headless=new')
        
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--window-size=1920,1080')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_experimental_option('excludeSwitches', ['enable-automation'])
        options.add_experimental_option('useAutomationExtension', False)
        
        # User agent to appear more like a real browser
        options.add_argument('--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        
        if WEBDRIVER_MANAGER_AVAILABLE:
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=options)
        else:
            # Try system Chrome
            self.driver = webdriver.Chrome(options=options)
        
        # Stealth settings
        self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    
    def _build_auth_url(self) -> str:
        """Build the authorization URL"""
        import urllib.parse
        import secrets
        
        state = secrets.token_urlsafe(32)
        
        params = {
            'client_id': self.app_key,
            'redirect_uri': self.redirect_uri,
            'response_type': 'code',
            'scope': 'api',
            'state': state
        }
        
        return f"{self.AUTH_URL}?{urllib.parse.urlencode(params)}"
    
    def authenticate(self, username: str, password: str) -> str:
        """
        Perform automated authentication.
        
        Returns the authorization code on success.
        Raises exception on failure.
        """
        if not SELENIUM_AVAILABLE:
            raise RuntimeError("Selenium not installed. Run: pip install selenium webdriver-manager")
        
        try:
            print("🚀 Starting automated authentication...")
            self._setup_driver()
            
            # Navigate to auth URL
            auth_url = self._build_auth_url()
            print(f"📍 Navigating to Schwab login...")
            self.driver.get(auth_url)
            
            # Wait for login page to load
            wait = WebDriverWait(self.driver, self.timeout)
            
            # Schwab login page - find username field
            print("🔐 Entering credentials...")
            
            # Try different possible selectors for username
            username_selectors = [
                (By.ID, 'loginIdInput'),
                (By.NAME, 'loginId'),
                (By.CSS_SELECTOR, 'input[type="text"]'),
                (By.CSS_SELECTOR, '#loginIdInput'),
            ]
            
            username_field = None
            for by, selector in username_selectors:
                try:
                    username_field = wait.until(EC.presence_of_element_located((by, selector)))
                    break
                except TimeoutException:
                    continue
            
            if not username_field:
                raise Exception("Could not find username field")
            
            # Enter username
            username_field.clear()
            username_field.send_keys(username)
            time.sleep(0.5)
            
            # Find and fill password
            password_selectors = [
                (By.ID, 'passwordInput'),
                (By.NAME, 'password'),
                (By.CSS_SELECTOR, 'input[type="password"]'),
            ]
            
            password_field = None
            for by, selector in password_selectors:
                try:
                    password_field = self.driver.find_element(by, selector)
                    break
                except NoSuchElementException:
                    continue
            
            if not password_field:
                raise Exception("Could not find password field")
            
            password_field.clear()
            password_field.send_keys(password)
            time.sleep(0.5)
            
            # Find and click login button
            login_selectors = [
                (By.ID, 'btnLogin'),
                (By.CSS_SELECTOR, 'button[type="submit"]'),
                (By.CSS_SELECTOR, '#btnLogin'),
                (By.XPATH, '//button[contains(text(), "Log in")]'),
                (By.XPATH, '//button[contains(text(), "Login")]'),
            ]
            
            login_button = None
            for by, selector in login_selectors:
                try:
                    login_button = self.driver.find_element(by, selector)
                    break
                except NoSuchElementException:
                    continue
            
            if login_button:
                print("🔘 Clicking login...")
                login_button.click()
            else:
                # Try submitting form
                password_field.send_keys(Keys.RETURN)
            
            time.sleep(3)
            
            # Check for 2FA or security questions
            current_url = self.driver.current_url
            if 'security' in current_url.lower() or 'verify' in current_url.lower():
                print("⚠️  2FA or security verification required!")
                print("    Please complete verification manually.")
                
                if self.headless:
                    raise Exception("2FA required - run with --no-headless to complete manually")
                
                # Wait for user to complete 2FA (up to 2 minutes)
                print("    Waiting for verification (2 min timeout)...")
                start = time.time()
                while time.time() - start < 120:
                    if 'callback' in self.driver.current_url or 'code=' in self.driver.current_url:
                        break
                    time.sleep(2)
            
            # Wait for redirect or authorization page
            print("⏳ Waiting for authorization...")
            
            # Look for "Accept" or "Allow" button on OAuth consent page
            time.sleep(2)
            
            accept_selectors = [
                (By.ID, 'accept'),
                (By.CSS_SELECTOR, 'button[type="submit"]'),
                (By.XPATH, '//button[contains(text(), "Accept")]'),
                (By.XPATH, '//button[contains(text(), "Allow")]'),
                (By.XPATH, '//button[contains(text(), "Authorize")]'),
                (By.XPATH, '//input[@value="Accept"]'),
            ]
            
            for by, selector in accept_selectors:
                try:
                    accept_button = self.driver.find_element(by, selector)
                    if accept_button.is_displayed():
                        print("✅ Clicking authorize...")
                        accept_button.click()
                        time.sleep(2)
                        break
                except NoSuchElementException:
                    continue
            
            # Wait for redirect to callback URL
            print("⏳ Waiting for callback redirect...")
            
            start = time.time()
            while time.time() - start < self.timeout:
                current_url = self.driver.current_url
                
                if 'callback' in current_url and 'code=' in current_url:
                    print("✅ Got callback URL!")
                    
                    # Extract authorization code
                    parsed = urlparse(current_url)
                    params = parse_qs(parsed.query)
                    
                    if 'code' in params:
                        code = params['code'][0]
                        print(f"🔑 Authorization code: {code[:30]}...")
                        return code
                
                time.sleep(1)
            
            raise TimeoutException("Timeout waiting for authorization callback")
            
        finally:
            if self.driver:
                self.driver.quit()
    
    def get_tokens(self, username: str, password: str, token_file: str = "schwab_tokens.json") -> bool:
        """
        Complete the full authentication flow and save tokens.
        
        Returns True on success, False on failure.
        """
        try:
            # Get authorization code
            auth_code = self.authenticate(username, password)
            
            # Exchange for tokens
            print("🔄 Exchanging code for tokens...")
            auth = SchwabAuth(
                app_key=self.app_key,
                app_secret=self.app_secret,
                redirect_uri=self.redirect_uri,
                token_file=token_file
            )
            
            success = auth.exchange_code(auth_code)
            
            if success:
                print("✅ Authentication successful! Tokens saved.")
                return True
            else:
                print("❌ Failed to exchange authorization code")
                return False
                
        except Exception as e:
            print(f"❌ Authentication failed: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description='Automated Schwab Authentication')
    parser.add_argument('--username', '-u', help='Schwab username', 
                        default=os.getenv('SCHWAB_USERNAME'))
    parser.add_argument('--password', '-p', help='Schwab password',
                        default=os.getenv('SCHWAB_PASSWORD'))
    parser.add_argument('--no-headless', action='store_true', 
                        help='Run browser visibly (for 2FA)')
    parser.add_argument('--token-file', default='schwab_tokens.json',
                        help='Token file path')
    
    args = parser.parse_args()
    
    # Check for credentials
    if not args.username or not args.password:
        print("❌ Missing credentials!")
        print("\nProvide credentials via:")
        print("  1. Command line: python auto_auth.py -u USERNAME -p PASSWORD")
        print("  2. Environment variables: SCHWAB_USERNAME and SCHWAB_PASSWORD")
        print("  3. .env file with SCHWAB_USERNAME and SCHWAB_PASSWORD")
        sys.exit(1)
    
    # Check for required packages
    if not SELENIUM_AVAILABLE:
        print("❌ Selenium not installed!")
        print("\nInstall with: pip install selenium webdriver-manager")
        sys.exit(1)
    
    # Load API credentials
    app_key = os.getenv('SCHWAB_APP_KEY')
    app_secret = os.getenv('SCHWAB_APP_SECRET')
    redirect_uri = os.getenv('SCHWAB_REDIRECT_URI', 'https://127.0.0.1:8182/callback')
    
    if not app_key or not app_secret:
        print("❌ Missing API credentials!")
        print("Set SCHWAB_APP_KEY and SCHWAB_APP_SECRET environment variables")
        sys.exit(1)
    
    # Run authentication
    auto_auth = AutomatedAuth(
        app_key=app_key,
        app_secret=app_secret,
        redirect_uri=redirect_uri,
        headless=not args.no_headless
    )
    
    success = auto_auth.get_tokens(
        username=args.username,
        password=args.password,
        token_file=args.token_file
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
