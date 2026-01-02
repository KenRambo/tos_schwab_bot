"""
Schwab API Authentication Handler (Process-Safe)

Handles OAuth2 flow for Schwab API access.
Tokens are stored locally and refreshed automatically.

Fixes included:
- Cross-process lock with PID + timestamp + stale-lock cleanup
- No deadlocks (no nested lock acquisition)
- Atomic token file writes
- readonly mode for optimizers (prevents refresh/write races)

Interactive "paste redirect URL" flow preserved.
"""
import json
import time
import base64
import secrets
import webbrowser
import urllib.parse
import os
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import requests
from http.server import BaseHTTPRequestHandler
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)


# ============================
# Lock helpers (PID + stale cleanup)
# ============================

def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _read_lockfile(lock_path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(lock_path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _write_lockfile(lock_path: Path, payload: Dict[str, Any]) -> None:
    tmp = lock_path.with_suffix(lock_path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, lock_path)


@contextmanager
def token_file_lock(lock_path: Path, timeout_s: float = 10.0, stale_s: float = 120.0, poll_s: float = 0.05):
    """
    Cross-process lock implemented via lock file:
      - lock contains {"pid": <pid>, "created_at": <epoch>}
      - if lock is stale (older than stale_s) OR pid is not alive -> auto removed
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    pid = os.getpid()

    while True:
        # If lock exists, check stale / owner
        if lock_path.exists():
            info = _read_lockfile(lock_path) or {}
            owner_pid = int(info.get("pid", -1)) if str(info.get("pid", "")).isdigit() else -1
            created_at = float(info.get("created_at", 0.0)) if isinstance(info.get("created_at", 0.0), (int, float)) else 0.0

            age = time.time() - created_at if created_at > 0 else float("inf")
            owner_alive = _pid_is_alive(owner_pid)

            if (age > stale_s) or (owner_pid > 0 and not owner_alive):
                # stale lock; remove
                try:
                    lock_path.unlink(missing_ok=True)
                except Exception:
                    pass
                continue

        # Try to create lock atomically
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            _write_lockfile(lock_path, {"pid": pid, "created_at": time.time()})
            break
        except FileExistsError:
            if (time.time() - start) >= timeout_s:
                raise TimeoutError(f"Timeout acquiring lock: {lock_path}")
            time.sleep(poll_s)

    try:
        yield
    finally:
        try:
            lock_path.unlink(missing_ok=True)
        except Exception:
            pass


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


# ============================
# OAuth callback handler (kept from your file)
# ============================

class CallbackHandler(BaseHTTPRequestHandler):
    authorization_code: Optional[str] = None
    error_message: Optional[str] = None

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)

        if "code" in params:
            CallbackHandler.authorization_code = params["code"][0]
            CallbackHandler.error_message = None
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(
                b"""
                <html><body>
                <h1>Authorization Successful!</h1>
                <p>You can close this window and return to the bot.</p>
                <script>window.close();</script>
                </body></html>
                """
            )
        else:
            CallbackHandler.error_message = params.get("error", ["Unknown error"])[0]
            self.send_response(400)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            error = CallbackHandler.error_message
            self.wfile.write(f"<html><body><h1>Error: {error}</h1></body></html>".encode())

    def log_message(self, format, *args):
        pass


def create_self_signed_cert():
    """Kept for compatibility; not required for your paste-URL flow."""
    try:
        from OpenSSL import crypto  # type: ignore

        key = crypto.PKey()
        key.generate_key(crypto.TYPE_RSA, 2048)

        cert = crypto.X509()
        cert.get_subject().CN = "localhost"
        cert.set_serial_number(1000)
        cert.gmtime_adj_notBefore(0)
        cert.gmtime_adj_notAfter(60 * 60)
        cert.set_issuer(cert.get_subject())
        cert.set_pubkey(key)
        cert.sign(key, "sha256")

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


# ============================
# SchwabAuth
# ============================

class SchwabAuth:
    AUTH_URL = "https://api.schwabapi.com/v1/oauth/authorize"
    TOKEN_URL = "https://api.schwabapi.com/v1/oauth/token"

    def __init__(
        self,
        app_key: str,
        app_secret: str,
        redirect_uri: str,
        token_file: str = "schwab_tokens.json",
        readonly: bool = False,
        refresh_if_needed: bool = True,
        lock_timeout_s: float = 10.0,
        lock_stale_s: float = 120.0,
    ):
        self.app_key = app_key
        self.app_secret = app_secret
        self.redirect_uri = redirect_uri

        self.token_file = Path(token_file)
        self.lock_file = self.token_file.with_suffix(self.token_file.suffix + ".lock")

        self.readonly = bool(readonly)
        self.refresh_if_needed = bool(refresh_if_needed) and (not self.readonly)

        self.lock_timeout_s = float(lock_timeout_s)
        self.lock_stale_s = float(lock_stale_s)

        self._access_token: Optional[str] = None
        self._refresh_token: Optional[str] = None
        self._token_expiry: Optional[datetime] = None

        # load tokens (lock-protected)
        self._load_tokens_locked()

    @property
    def is_authenticated(self) -> bool:
        return self._access_token is not None and self._refresh_token is not None

    @property
    def access_token(self) -> Optional[str]:
        """
        Returns current access token.
        If refresh_if_needed=True (and not readonly), refreshes within 5 minutes of expiry.
        """
        if self._access_token and self._token_expiry and self.refresh_if_needed:
            if datetime.now() >= self._token_expiry - timedelta(minutes=5):
                ok = self.refresh_access_token()
                if not ok:
                    # reload in case another process refreshed
                    self._load_tokens_locked()
        return self._access_token

    def _get_auth_header(self) -> str:
        creds = f"{self.app_key}:{self.app_secret}"
        encoded = base64.b64encode(creds.encode()).decode()
        return f"Basic {encoded}"

    # --------- Token file IO (NO nested lock) ---------

    def _load_tokens_unlocked(self) -> bool:
        if not self.token_file.exists():
            return False
        try:
            with open(self.token_file, "r") as f:
                data = json.load(f)
            self._access_token = data.get("access_token")
            self._refresh_token = data.get("refresh_token")
            exp = data.get("expires_at")
            self._token_expiry = datetime.fromisoformat(exp) if exp else None
            return True
        except Exception as e:
            logger.error(f"Error loading tokens: {e}")
            return False

    def _save_tokens_unlocked(self) -> None:
        if self.readonly:
            return
        data = {
            "access_token": self._access_token,
            "refresh_token": self._refresh_token,
            "expires_at": self._token_expiry.isoformat() if self._token_expiry else None,
            "saved_at": datetime.now().isoformat(),
        }
        _atomic_write_json(self.token_file, data)

    def _load_tokens_locked(self) -> bool:
        if not self.token_file.exists():
            return False
        try:
            with token_file_lock(self.lock_file, timeout_s=self.lock_timeout_s, stale_s=self.lock_stale_s):
                return self._load_tokens_unlocked()
        except TimeoutError as e:
            logger.error(f"Error loading tokens: {e}")
            return False

    # --------- Interactive auth (your original flow preserved) ---------

    def start_auth_flow(self, auto_open_browser: bool = True) -> str:
        state = secrets.token_urlsafe(32)
        params = {
            "client_id": self.app_key,
            "redirect_uri": self.redirect_uri,
            "response_type": "code",
            "scope": "api",
            "state": state,
        }
        auth_url = f"{self.AUTH_URL}?{urllib.parse.urlencode(params)}"
        if auto_open_browser:
            webbrowser.open(auth_url)
        return auth_url

    def authorize_interactive(self, port: int = 8182) -> bool:
        CallbackHandler.authorization_code = None
        CallbackHandler.error_message = None

        auth_url = self.start_auth_flow(auto_open_browser=False)

        print("\n" + "=" * 60)
        print("SCHWAB AUTHORIZATION")
        print("=" * 60)
        print("\n1. Opening browser to Schwab login...")
        try:
            webbrowser.open(auth_url)
        except Exception:
            print("\n   Could not open browser. Open this URL manually:")
            print(f"   {auth_url}")

        print("\n2. Log in and authorize the application")
        print("\n3. You'll be redirected to a page that WON'T LOAD - that's OK!")
        print("\n4. QUICKLY copy the URL from your browser's address bar")
        print("   (It starts with https://127.0.0.1:8182/callback?code=...)")
        print("\n5. Paste it below and press Enter:")
        print("=" * 60)
        print("\nPaste redirect URL here: ", end="", flush=True)

        try:
            redirect_url = input().strip()
        except KeyboardInterrupt:
            print("\nCancelled.")
            return False

        if not redirect_url:
            print("No URL provided.")
            return False

        auth_code = self._extract_code_from_url(redirect_url)
        if auth_code:
            return self.exchange_code(auth_code)

        print("\n❌ Could not find authorization code in that URL.")
        print("   Make sure you copied the ENTIRE URL from the address bar.")
        return False

    def _extract_code_from_url(self, url: str) -> Optional[str]:
        try:
            url = urllib.parse.unquote(url)
            parsed = urllib.parse.urlparse(url)
            params = urllib.parse.parse_qs(parsed.query)
            if "code" in params:
                code = urllib.parse.unquote(params["code"][0])
                return code

            import re
            m = re.search(r"code=([^&\s]+)", url)
            if m:
                return urllib.parse.unquote(m.group(1))
            return None
        except Exception as e:
            logger.error(f"Error parsing redirect URL: {e}")
            return None

    # --------- Token exchange / refresh ---------

    def exchange_code(self, authorization_code: str) -> bool:
        headers = {
            "Authorization": self._get_auth_header(),
            "Content-Type": "application/x-www-form-urlencoded",
        }
        data = {
            "grant_type": "authorization_code",
            "code": authorization_code,
            "redirect_uri": self.redirect_uri,
        }

        try:
            response = requests.post(self.TOKEN_URL, headers=headers, data=data, timeout=30)
            response.raise_for_status()

            token_data = response.json()
            self._access_token = token_data["access_token"]
            self._refresh_token = token_data["refresh_token"]
            self._token_expiry = datetime.now() + timedelta(seconds=token_data.get("expires_in", 1800))

            # lock and save once
            if not self.readonly:
                with token_file_lock(self.lock_file, timeout_s=self.lock_timeout_s, stale_s=self.lock_stale_s):
                    self._save_tokens_unlocked()

            return True

        except requests.exceptions.RequestException as e:
            logger.error(f"Token exchange failed: {e}")
            if hasattr(e, "response") and e.response is not None:
                logger.error(f"Response: {e.response.text}")
            return False

    def refresh_access_token(self) -> bool:
        if self.readonly:
            return False
        if not self._refresh_token:
            return False

        try:
            # single refresh writer
            with token_file_lock(self.lock_file, timeout_s=self.lock_timeout_s, stale_s=self.lock_stale_s):
                # reload tokens without re-locking (prevents deadlock)
                self._load_tokens_unlocked()

                # if another process already refreshed, stop
                if self._access_token and self._token_expiry:
                    if datetime.now() < (self._token_expiry - timedelta(minutes=5)):
                        return True

                headers = {
                    "Authorization": self._get_auth_header(),
                    "Content-Type": "application/x-www-form-urlencoded",
                }
                data = {"grant_type": "refresh_token", "refresh_token": self._refresh_token}

                response = requests.post(self.TOKEN_URL, headers=headers, data=data, timeout=30)
                response.raise_for_status()

                token_data = response.json()
                self._access_token = token_data["access_token"]
                if "refresh_token" in token_data:
                    self._refresh_token = token_data["refresh_token"]
                self._token_expiry = datetime.now() + timedelta(seconds=token_data.get("expires_in", 1800))

                self._save_tokens_unlocked()
                return True

        except requests.exceptions.RequestException as e:
            logger.error(f"Token refresh failed: {e}")
            if hasattr(e, "response") and e.response is not None:
                logger.error(f"Response: {e.response.text}")
            return False
        except TimeoutError as e:
            logger.error(str(e))
            return False

    def get_headers(self, include_content_type: bool = False) -> Dict[str, str]:
        headers = {"Authorization": f"Bearer {self.access_token}", "Accept": "application/json"}
        if include_content_type:
            headers["Content-Type"] = "application/json"
        return headers
