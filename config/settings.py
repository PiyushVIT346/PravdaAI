"""Application configuration settings."""
import os
from dataclasses import dataclass


@dataclass
class Config:
    """Application configuration."""
    
    # Flask settings
    SECRET_KEY: str = os.getenv('SECRET_KEY', 'your-super-secret-key-change-this-in-production')
    MAX_CONTENT_LENGTH: int = 16 * 1024 * 1024  # 16MB
    
    # Directory paths
    BASE_DIR: str = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    UPLOAD_FOLDER: str = os.path.join(BASE_DIR, 'uploads')
    LAWS_PDF_FOLDER: str = os.path.join(BASE_DIR, 'laws_pdfs2')
    TEMPLATE_FOLDER: str = os.path.join(BASE_DIR, 'templates')
    STATIC_FOLDER: str = os.path.join(BASE_DIR, 'static')
    
    # Database
    DATABASE: str = os.path.join(BASE_DIR, 'users.db')
    
    # API Keys
    GEMINI_API_KEY: str = os.getenv('GEMINI_API_KEY', '')
    
    # Model settings
    LLM_MODEL: str = "gemini-2.0-flash-lite"
    LLM_TEMPERATURE: float = 0.3
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Text splitter settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    @classmethod
    def validate(cls) -> None:
        """Validate required configuration."""
        if not cls.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY environment variable is required")