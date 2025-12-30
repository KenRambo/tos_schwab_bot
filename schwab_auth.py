"""
Schwab API Authentication Handler

Handles OAuth2 flow for Schwab API access.
Tokens are stored locally and refreshed automatically.
"""
import json
import time
import base64
import hashlib
import secrets
import webbrowser
import urllib.parse
import ssl
import os
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import requests
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import logging

logger = logging.getLogger(__name__)


class CallbackHandler(BaseHTTPRequestHandler):
    """HTTP handler to capture OAuth callback"""
    
    authorization_code: Optional[str] = None
    error_message: Optional[str] = None
    
    def do_GET(self):
        """Handle GET request from OAuth callback"""
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)
        
        if 'code' in params:
            CallbackHandler.authorization_code = params['code'][0]
            CallbackHandler.error_message = None
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b"""
                <html><body>
                <h1>Authorization Successful!</h1>
                <p>You can close this window and return to the bot.</p>
                <script>window.close();</script>
                </body></html>
            """)
            logger.info("Authorization code received successfully")
        else:
            CallbackHandler.error_message = params.get('error', ['Unknown error'])[0]
            self.send_response(400)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            error = CallbackHandler.error_message
            self.wfile.write(f"<html><body><h1>Error: {error}</h1></body></html>".encode())
            logger.error(f"OAuth error: {error}")
    
    def log_message(self, format, *args):
        """Suppress HTTP server logs"""
        pass


def create_self_signed_cert():
    """Create a temporary self-signed certificate for HTTPS callback"""
    try:
        from OpenSSL import crypto
        
        # Create key pair
        key = crypto.PKey()
        key.generate_key(crypto.TYPE_RSA, 2048)
        
        # Create certificate
        cert = crypto.X509()
        cert.get_subject().CN = "localhost"
        cert.set_serial_number(1000)
        cert.gmtime_adj_notBefore(0)
        cert.gmtime_adj_notAfter(60 * 60)  # Valid for 1 hour
        cert.set_issuer(cert.get_subject())
        cert.set_pubkey(key)
        cert.sign(key, 'sha256')
        
        # Write to temp files
        cert_dir = tempfile.mkdtemp()
        cert_path = os.path.join(cert_dir, "cert.pem")
        key_path = os.path.join(cert_dir, "key.pem")
        
        with open(cert_path, "wb") as f:
            f.write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert))
        with open(key_path, "wb") as f:
            f.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, key))
        
        return cert_path, key_path
    except ImportError:
        return None, None


class SchwabAuth:
    """Schwab OAuth2 Authentication Manager"""
    
    AUTH_URL = "https://api.schwabapi.com/v1/oauth/authorize"
    TOKEN_URL = "https://api.schwabapi.com/v1/oauth/token"
    
    def __init__(self, app_key: str, app_secret: str, redirect_uri: str, token_file: str = "schwab_tokens.json"):
        self.app_key = app_key
        self.app_secret = app_secret
        self.redirect_uri = redirect_uri
        self.token_file = Path(token_file)
        
        self._access_token: Optional[str] = None
        self._refresh_token: Optional[str] = None
        self._token_expiry: Optional[datetime] = None
        
        # Load existing tokens if available
        self._load_tokens()
    
    @property
    def access_token(self) -> Optional[str]:
        """Get current access token, refreshing if needed"""
        if self._access_token and self._token_expiry:
            # Refresh 5 minutes before expiry
            if datetime.now() >= self._token_expiry - timedelta(minutes=5):
                logger.info("Access token expiring soon, refreshing...")
                self.refresh_access_token()
        return self._access_token
    
    @property
    def is_authenticated(self) -> bool:
        """Check if we have valid tokens"""
        return self._access_token is not None and self._refresh_token is not None
    
    def _get_auth_header(self) -> str:
        """Generate Basic auth header for token requests"""
        credentials = f"{self.app_key}:{self.app_secret}"
        encoded = base64.b64encode(credentials.encode()).decode()
        return f"Basic {encoded}"
    
    def _load_tokens(self) -> bool:
        """Load tokens from file"""
        if not self.token_file.exists():
            return False
        
        try:
            with open(self.token_file, 'r') as f:
                data = json.load(f)
            
            self._access_token = data.get('access_token')
            self._refresh_token = data.get('refresh_token')
            
            if 'expires_at' in data:
                self._token_expiry = datetime.fromisoformat(data['expires_at'])
            
            logger.info("Loaded existing tokens from file")
            return True
            
        except Exception as e:
            logger.error(f"Error loading tokens: {e}")
            return False
    
    def _save_tokens(self) -> None:
        """Save tokens to file"""
        data = {
            'access_token': self._access_token,
            'refresh_token': self._refresh_token,
            'expires_at': self._token_expiry.isoformat() if self._token_expiry else None,
            'saved_at': datetime.now().isoformat()
        }
        
        with open(self.token_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info("Saved tokens to file")
    
    def start_auth_flow(self, auto_open_browser: bool = True) -> str:
        """
        Start OAuth2 authorization flow.
        Returns the authorization URL.
        """
        # Generate state for CSRF protection
        state = secrets.token_urlsafe(32)
        
        params = {
            'client_id': self.app_key,
            'redirect_uri': self.redirect_uri,
            'response_type': 'code',
            'scope': 'api',  # Request API access
            'state': state
        }
        
        auth_url = f"{self.AUTH_URL}?{urllib.parse.urlencode(params)}"
        
        if auto_open_browser:
            logger.info("Opening browser for authorization...")
            webbrowser.open(auth_url)
        
        return auth_url
    
    def authorize_interactive(self, port: int = 8182) -> bool:
        """
        Perform interactive authorization flow.
        Manual URL paste method - most reliable for Schwab OAuth.
        """
        # Reset callback handler
        CallbackHandler.authorization_code = None
        CallbackHandler.error_message = None
        
        # Generate authorization URL
        auth_url = self.start_auth_flow(auto_open_browser=False)
        
        print("\n" + "=" * 60)
        print("SCHWAB AUTHORIZATION")
        print("=" * 60)
        print("\n1. Opening browser to Schwab login...")
        
        # Try to open browser
        try:
            webbrowser.open(auth_url)
        except:
            print("\n   Could not open browser. Open this URL manually:")
            print(f"   {auth_url}")
        
        print("\n2. Log in and authorize the application")
        print("\n3. You'll be redirected to a page that WON'T LOAD - that's OK!")
        print("\n4. QUICKLY copy the URL from your browser's address bar")
        print("   (It starts with https://127.0.0.1:8182/callback?code=...)")
        print("\n5. Paste it below and press Enter:")
        print("=" * 60)
        
        # Get URL from user - be fast!
        print("\nPaste redirect URL here: ", end="", flush=True)
        
        try:
            redirect_url = input().strip()
        except KeyboardInterrupt:
            print("\nCancelled.")
            return False
        
        if not redirect_url:
            print("No URL provided.")
            return False
        
        # Extract and exchange code immediately
        auth_code = self._extract_code_from_url(redirect_url)
        
        if auth_code:
            logger.info(f"Authorization code extracted: {auth_code[:30]}...")
            return self.exchange_code(auth_code)
        else:
            logger.error("Could not extract authorization code from URL")
            print("\n❌ Could not find authorization code in that URL.")
            print("   Make sure you copied the ENTIRE URL from the address bar.")
            return False
    
    def _extract_code_from_url(self, url: str) -> Optional[str]:
        """Extract authorization code from redirect URL"""
        try:
            # URL decode the entire URL first
            url = urllib.parse.unquote(url)
            
            parsed = urllib.parse.urlparse(url)
            params = urllib.parse.parse_qs(parsed.query)
            
            if 'code' in params:
                code = params['code'][0]
                # Decode again in case of double-encoding
                code = urllib.parse.unquote(code)
                return code
            
            # Fallback: Try to find code parameter manually
            import re
            match = re.search(r'code=([^&\s]+)', url)
            if match:
                return urllib.parse.unquote(match.group(1))
            
            return None
        except Exception as e:
            logger.error(f"Error parsing redirect URL: {e}")
            return None
    
    def exchange_code(self, authorization_code: str) -> bool:
        """Exchange authorization code for tokens"""
        headers = {
            'Authorization': self._get_auth_header(),
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
        data = {
            'grant_type': 'authorization_code',
            'code': authorization_code,
            'redirect_uri': self.redirect_uri
        }
        
        try:
            response = requests.post(self.TOKEN_URL, headers=headers, data=data)
            response.raise_for_status()
            
            token_data = response.json()
            self._access_token = token_data['access_token']
            self._refresh_token = token_data['refresh_token']
            self._token_expiry = datetime.now() + timedelta(seconds=token_data.get('expires_in', 1800))
            
            self._save_tokens()
            logger.info("Successfully obtained access tokens")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Token exchange failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response: {e.response.text}")
            return False
    
    def refresh_access_token(self) -> bool:
        """Refresh the access token using refresh token"""
        if not self._refresh_token:
            logger.error("No refresh token available")
            return False
        
        headers = {
            'Authorization': self._get_auth_header(),
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
        data = {
            'grant_type': 'refresh_token',
            'refresh_token': self._refresh_token
        }
        
        try:
            response = requests.post(self.TOKEN_URL, headers=headers, data=data)
            response.raise_for_status()
            
            token_data = response.json()
            self._access_token = token_data['access_token']
            
            # Schwab may return a new refresh token
            if 'refresh_token' in token_data:
                self._refresh_token = token_data['refresh_token']
            
            self._token_expiry = datetime.now() + timedelta(seconds=token_data.get('expires_in', 1800))
            
            self._save_tokens()
            logger.info("Successfully refreshed access token")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Token refresh failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response: {e.response.text}")
            return False
    
    def get_headers(self, include_content_type: bool = False) -> Dict[str, str]:
        """Get headers for API requests"""
        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Accept': 'application/json'
        }
        # Only include Content-Type for POST/PUT requests, not GET
        if include_content_type:
            headers['Content-Type'] = 'application/json'
        return headers
