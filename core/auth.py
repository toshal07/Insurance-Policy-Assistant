"""SQLite + PBKDF2-HMAC user authentication for PolicyAI."""
import hashlib
import hmac
import os
from pathlib import Path
import re
import secrets
import sqlite3

_EMAIL_RE = re.compile(r'^[^@\s]+@[^@\s]+\.[^@\s]+$')
_ITERATIONS = 200_000


def _db_path() -> Path:
    configured = os.getenv('POLICYAI_DB_PATH')
    path = Path(configured) if configured else Path(__file__).resolve().parent.parent / '.policyai' / 'users.db'
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path())
    conn.row_factory = sqlite3.Row
    conn.execute(
        '''
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        '''
    )
    conn.commit()
    return conn


def _hash_password(password: str, salt_hex: str) -> str:
    digest = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), bytes.fromhex(salt_hex), _ITERATIONS)
    return digest.hex()


def _public_user(row: sqlite3.Row) -> dict:
    parts = [p for p in row['name'].strip().split() if p]
    initials = ''.join(p[0].upper() for p in parts[:2]) or row['email'][0].upper()
    return {'id': row['id'], 'name': row['name'], 'email': row['email'], 'initials': initials}


def register_user(name: str, email: str, password: str, confirm_password: str | None = None) -> dict:
    """Validate and register a new PolicyAI user account."""
    clean_name = re.sub(r'\s+', ' ', (name or '').strip())
    clean_email = (email or '').strip().lower()
    if len(clean_name) < 2:
        raise ValueError('Enter your full name (at least 2 characters).')
    if not _EMAIL_RE.match(clean_email):
        raise ValueError('Enter a valid email address.')
    if not password or len(password) < 8:
        raise ValueError('Password must be at least 8 characters long.')
    if confirm_password is not None and password != confirm_password:
        raise ValueError('Passwords do not match.')

    salt_hex = secrets.token_hex(16)
    password_hash = _hash_password(password, salt_hex)
    user_id = secrets.token_hex(16)

    with _connect() as conn:
        existing = conn.execute('SELECT id FROM users WHERE email = ?', (clean_email,)).fetchone()
        if existing:
            raise ValueError('An account with this email already exists. Please log in.')
        conn.execute(
            'INSERT INTO users (id, name, email, password_hash, salt) VALUES (?, ?, ?, ?, ?)',
            (user_id, clean_name[:120], clean_email[:180], password_hash, salt_hex),
        )
        conn.commit()
        row = conn.execute('SELECT id, name, email FROM users WHERE id = ?', (user_id,)).fetchone()
    return _public_user(row)


def authenticate_user(email: str, password: str) -> dict:
    """Verify email and password and return public user profile."""
    clean_email = (email or '').strip().lower()
    if not _EMAIL_RE.match(clean_email) or not password:
        raise ValueError('Enter your email and password.')

    with _connect() as conn:
        row = conn.execute(
            'SELECT id, name, email, password_hash, salt FROM users WHERE email = ?',
            (clean_email,),
        ).fetchone()
    if not row:
        raise ValueError('Invalid email or password.')

    candidate_hash = _hash_password(password, row['salt'])
    if not hmac.compare_digest(candidate_hash, row['password_hash']):
        raise ValueError('Invalid email or password.')
    return _public_user(row)
