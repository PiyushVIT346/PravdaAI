"""Flask application factory."""
import os
import logging
from flask import Flask
from config.settings import Config
from services.legal_assistant import LegalAIAssistant
from routes.auth import auth_bp
from routes.api import api_bp, set_legal_assistant
from routes.dashboard import dashboard_bp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_app(gemini_api_key: str = None) -> Flask:
    """Create and configure the Flask application."""
    
    # Create Flask app
    app = Flask(
        __name__,
        template_folder=Config.TEMPLATE_FOLDER,
        static_folder=Config.STATIC_FOLDER
    )
    
    # Configure app
    app.secret_key = Config.SECRET_KEY
    app.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH
    app.config['UPLOAD_FOLDER'] = Config.UPLOAD_FOLDER
    
    # Create necessary directories
    os.makedirs(Config.TEMPLATE_FOLDER, exist_ok=True)
    os.makedirs(os.path.join(Config.STATIC_FOLDER, 'css'), exist_ok=True)
    os.makedirs(os.path.join(Config.STATIC_FOLDER, 'js'), exist_ok=True)
    os.makedirs(os.path.join(Config.STATIC_FOLDER, 'images'), exist_ok=True)
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(Config.LAWS_PDF_FOLDER, exist_ok=True)
    
    # Initialize Legal AI Assistant
    try:
        api_key = gemini_api_key or Config.GEMINI_API_KEY
        Config.validate()
        
        legal_assistant = LegalAIAssistant(api_key)
        set_legal_assistant(legal_assistant)
        logger.info("Legal AI Assistant initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Legal AI Assistant: {e}")
        raise
    
    # Register blueprints
    app.register_blueprint(auth_bp)
    app.register_blueprint(api_bp)
    app.register_blueprint(dashboard_bp)
    
    return app