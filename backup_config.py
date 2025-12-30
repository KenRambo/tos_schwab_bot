"""
Backup and Restore Configuration & Tokens

Save your .env and tokens to a backup file, or restore them.

Usage:
    python backup_config.py save           # Save to backup
    python backup_config.py save my_backup # Save to custom file
    python backup_config.py restore        # Restore from backup
    python backup_config.py show           # Show current config
"""
import os
import sys
import json
import base64
import getpass
import argparse
from pathlib import Path
from datetime import datetime
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# Check for cryptography
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False


def derive_key(password: str, salt: bytes) -> bytes:
    """Derive encryption key from password"""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=480000,
    )
    return base64.urlsafe_b64encode(kdf.derive(password.encode()))


def encrypt_data(data: dict, password: str) -> dict:
    """Encrypt data with password"""
    salt = os.urandom(16)
    key = derive_key(password, salt)
    fernet = Fernet(key)
    
    encrypted = fernet.encrypt(json.dumps(data).encode())
    
    return {
        'salt': base64.b64encode(salt).decode(),
        'data': base64.b64encode(encrypted).decode(),
        'version': 1
    }


def decrypt_data(encrypted: dict, password: str) -> dict:
    """Decrypt data with password"""
    salt = base64.b64decode(encrypted['salt'])
    data = base64.b64decode(encrypted['data'])
    
    key = derive_key(password, salt)
    fernet = Fernet(key)
    
    decrypted = fernet.decrypt(data)
    return json.loads(decrypted)


def load_env_file(filepath: str = '.env') -> dict:
    """Load .env file into dictionary"""
    env_vars = {}
    
    if not os.path.exists(filepath):
        return env_vars
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                # Remove quotes
                value = value.strip('"\'')
                env_vars[key.strip()] = value
    
    return env_vars


def save_env_file(env_vars: dict, filepath: str = '.env'):
    """Save dictionary to .env file"""
    with open(filepath, 'w') as f:
        f.write(f"# ToS Trading Bot Configuration\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n\n")
        
        for key, value in env_vars.items():
            # Quote values with spaces
            if ' ' in str(value):
                f.write(f'{key}="{value}"\n')
            else:
                f.write(f'{key}={value}\n')


def load_tokens(filepath: str = 'schwab_tokens.json') -> dict:
    """Load tokens from file"""
    if not os.path.exists(filepath):
        return {}
    
    with open(filepath, 'r') as f:
        return json.load(f)


def save_tokens(tokens: dict, filepath: str = 'schwab_tokens.json'):
    """Save tokens to file"""
    with open(filepath, 'w') as f:
        json.dump(tokens, f, indent=2)


def save_backup(backup_file: str = 'bot_backup.json', encrypt: bool = True):
    """Save configuration and tokens to backup file"""
    print("📦 Creating backup...")
    
    # Gather data
    backup_data = {
        'created': datetime.now().isoformat(),
        'env': load_env_file('.env'),
        'tokens': load_tokens('schwab_tokens.json'),
    }
    
    # Also try to get from environment if .env is empty
    if not backup_data['env']:
        env_keys = [
            'SCHWAB_APP_KEY', 'SCHWAB_APP_SECRET', 'SCHWAB_REDIRECT_URI',
            'SCHWAB_USERNAME', 'SCHWAB_PASSWORD',
            'DISCORD_WEBHOOK', 'TELEGRAM_BOT_TOKEN', 'TELEGRAM_CHAT_ID'
        ]
        for key in env_keys:
            val = os.getenv(key)
            if val:
                backup_data['env'][key] = val
    
    if not backup_data['env'] and not backup_data['tokens']:
        print("⚠️  No configuration or tokens found to backup!")
        return False
    
    # Encrypt if requested
    if encrypt and CRYPTO_AVAILABLE:
        password = getpass.getpass("Enter backup password: ")
        password2 = getpass.getpass("Confirm password: ")
        
        if password != password2:
            print("❌ Passwords don't match!")
            return False
        
        if len(password) < 4:
            print("❌ Password too short (min 4 characters)")
            return False
        
        backup_data = encrypt_data(backup_data, password)
        backup_data['encrypted'] = True
    else:
        backup_data['encrypted'] = False
        if encrypt and not CRYPTO_AVAILABLE:
            print("⚠️  cryptography not installed - saving unencrypted")
            print("   Install with: pip install cryptography")
    
    # Save
    with open(backup_file, 'w') as f:
        json.dump(backup_data, f, indent=2)
    
    print(f"✅ Backup saved to: {backup_file}")
    print(f"   Encrypted: {backup_data.get('encrypted', False)}")
    print(f"   Contains .env: {bool(backup_data.get('env') or backup_data.get('data'))}")
    print(f"   Contains tokens: {bool(backup_data.get('tokens') or backup_data.get('data'))}")
    
    return True


def restore_backup(backup_file: str = 'bot_backup.json'):
    """Restore configuration and tokens from backup file"""
    if not os.path.exists(backup_file):
        print(f"❌ Backup file not found: {backup_file}")
        return False
    
    print(f"📂 Restoring from: {backup_file}")
    
    with open(backup_file, 'r') as f:
        backup_data = json.load(f)
    
    # Decrypt if needed
    if backup_data.get('encrypted'):
        if not CRYPTO_AVAILABLE:
            print("❌ Backup is encrypted but cryptography not installed")
            print("   Install with: pip install cryptography")
            return False
        
        password = getpass.getpass("Enter backup password: ")
        
        try:
            backup_data = decrypt_data(backup_data, password)
        except Exception as e:
            print(f"❌ Failed to decrypt: {e}")
            print("   Check your password and try again")
            return False
    
    # Confirm restore
    print(f"\n📋 Backup created: {backup_data.get('created', 'Unknown')}")
    print(f"   .env keys: {list(backup_data.get('env', {}).keys())}")
    print(f"   Has tokens: {bool(backup_data.get('tokens'))}")
    
    confirm = input("\n⚠️  This will overwrite existing config. Continue? (y/N): ")
    if confirm.lower() != 'y':
        print("Cancelled.")
        return False
    
    # Restore .env
    if backup_data.get('env'):
        save_env_file(backup_data['env'], '.env')
        print("✅ Restored .env")
    
    # Restore tokens
    if backup_data.get('tokens'):
        save_tokens(backup_data['tokens'], 'schwab_tokens.json')
        print("✅ Restored schwab_tokens.json")
    
    print("\n✅ Restore complete!")
    print("   Restart the bot to use restored configuration.")
    
    return True


def show_config():
    """Display current configuration (masked)"""
    print("📋 Current Configuration\n")
    
    # .env
    print("═" * 50)
    print(".env file:")
    print("═" * 50)
    
    env_vars = load_env_file('.env')
    if env_vars:
        for key, value in env_vars.items():
            # Mask sensitive values
            if any(s in key.lower() for s in ['secret', 'password', 'token', 'key']):
                masked = value[:4] + '*' * (len(value) - 8) + value[-4:] if len(value) > 8 else '****'
                print(f"  {key}: {masked}")
            else:
                print(f"  {key}: {value}")
    else:
        print("  (not found or empty)")
    
    # Check environment variables
    print("\n" + "═" * 50)
    print("Environment Variables:")
    print("═" * 50)
    
    env_keys = ['SCHWAB_APP_KEY', 'SCHWAB_APP_SECRET', 'SCHWAB_USERNAME']
    for key in env_keys:
        val = os.getenv(key)
        if val:
            masked = val[:4] + '*' * (len(val) - 8) + val[-4:] if len(val) > 8 else '****'
            print(f"  {key}: {masked}")
        else:
            print(f"  {key}: (not set)")
    
    # Tokens
    print("\n" + "═" * 50)
    print("Tokens (schwab_tokens.json):")
    print("═" * 50)
    
    tokens = load_tokens('schwab_tokens.json')
    if tokens:
        if tokens.get('access_token'):
            at = tokens['access_token']
            print(f"  access_token: {at[:20]}...{at[-10:]}")
        if tokens.get('refresh_token'):
            rt = tokens['refresh_token']
            print(f"  refresh_token: {rt[:20]}...{rt[-10:]}")
        if tokens.get('expires_at'):
            print(f"  expires_at: {tokens['expires_at']}")
    else:
        print("  (not found)")
    
    print()


def main():
    parser = argparse.ArgumentParser(description='Backup and restore bot configuration')
    parser.add_argument('action', choices=['save', 'restore', 'show'],
                        help='Action to perform')
    parser.add_argument('file', nargs='?', default='bot_backup.json',
                        help='Backup file path (default: bot_backup.json)')
    parser.add_argument('--no-encrypt', action='store_true',
                        help='Save backup without encryption')
    
    args = parser.parse_args()
    
    if args.action == 'save':
        success = save_backup(args.file, encrypt=not args.no_encrypt)
        sys.exit(0 if success else 1)
    
    elif args.action == 'restore':
        success = restore_backup(args.file)
        sys.exit(0 if success else 1)
    
    elif args.action == 'show':
        show_config()


if __name__ == "__main__":
    main()
