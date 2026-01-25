"""Database service for user management."""
import sqlite3
import hashlib
from typing import Optional, Dict, Any
from config.settings import Config


class DatabaseService:
    """Handles all database operations."""
    
    def __init__(self, db_path: str = None):
        """Initialize database service."""
        self.db_path = db_path or Config.DATABASE
        self._init_db()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Create and return database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _init_db(self) -> None:
        """Initialize the database with users table."""
        conn = self._get_connection()
        conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                first_name TEXT NOT NULL,
                last_name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                organization TEXT,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password using SHA-256."""
        return hashlib.sha256(password.encode()).hexdigest()
    
    @staticmethod
    def verify_password(password: str, password_hash: str) -> bool:
        """Verify password against hash."""
        return hashlib.sha256(password.encode()).hexdigest() == password_hash
    
    def create_user(self, first_name: str, last_name: str, email: str, 
                   password: str, organization: Optional[str] = None) -> bool:
        """Create a new user."""
        try:
            conn = self._get_connection()
            password_hash = self.hash_password(password)
            conn.execute('''
                INSERT INTO users (first_name, last_name, email, organization, password_hash)
                VALUES (?, ?, ?, ?, ?)
            ''', (first_name, last_name, email.lower(), organization, password_hash))
            conn.commit()
            conn.close()
            return True
        except sqlite3.IntegrityError:
            return False
    
    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Get user by email."""
        conn = self._get_connection()
        user = conn.execute('''
            SELECT id, first_name, last_name, email, password_hash
            FROM users WHERE email = ?
        ''', (email.lower(),)).fetchone()
        conn.close()
        
        if user:
            return dict(user)
        return None
    
    def email_exists(self, email: str) -> bool:
        """Check if email already exists."""
        conn = self._get_connection()
        result = conn.execute(
            'SELECT id FROM users WHERE email = ?', (email.lower(),)
        ).fetchone()
        conn.close()
        return result is not None