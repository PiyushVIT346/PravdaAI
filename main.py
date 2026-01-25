"""Main entry point for the Legal AI Assistant application."""
import os
from app import create_app


if __name__ == '__main__':
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    
    app = create_app(gemini_api_key)
    app.run(debug=True, host='0.0.0.0', port=5000)