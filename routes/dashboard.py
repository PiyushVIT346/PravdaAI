"""Dashboard and main application routes."""
from flask import Blueprint, render_template
from utils.auth import login_required

dashboard_bp = Blueprint('dashboard', __name__)


@dashboard_bp.route('/dashboard')
@login_required
def dashboard():
    """Dashboard route - requires login."""
    return render_template('index.html')