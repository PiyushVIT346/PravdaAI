"""API routes for legal assistant functionality."""
import os
import logging
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
from config.settings import Config
from utils.auth import login_required

logger = logging.getLogger(__name__)

api_bp = Blueprint('api', __name__, url_prefix='/api')

# Global assistant instance (will be set by app factory)
legal_assistant = None


def set_legal_assistant(assistant):
    """Set the global legal assistant instance."""
    global legal_assistant
    legal_assistant = assistant


@api_bp.route('/upload', methods=['POST'])
@login_required
def upload_document():
    """Endpoint for uploading documents."""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if file and file.filename.lower().endswith('.pdf'):
        filename = secure_filename(file.filename)
        file_path = os.path.join(Config.UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        success = legal_assistant.process_document(file_path)
        
        if success:
            return jsonify({
                "message": "Document uploaded and processed successfully",
                "filename": filename,
                "success": True
            })
        else:
            return jsonify({"error": "Failed to process document"}), 500
    
    return jsonify({"error": "Only PDF files are supported"}), 400


@api_bp.route('/query', methods=['POST'])
@login_required
def handle_query():
    """Main endpoint for handling user queries."""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({"error": "No query provided"}), 400
        
        user_query = data['query']
        uploaded_file = data.get('uploaded_file')
        
        result = legal_assistant.query(user_query, uploaded_file)
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error in query endpoint: {e}")
        return jsonify({
            "error": "Internal server error",
            "success": False
        }), 500


@api_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "service": "Legal AI Assistant"})