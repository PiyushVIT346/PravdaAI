"""Authentication routes (login, register, logout)."""
from flask import Blueprint, render_template, request, redirect, url_for, flash, session, jsonify
from services.database import DatabaseService

auth_bp = Blueprint('auth', __name__)
db_service = DatabaseService()


@auth_bp.route('/')
def home():
    """Home page route."""
    return render_template('home.html')


@auth_bp.route('/register', methods=['GET'])
def register_page():
    """Registration page route."""
    return render_template('register.html')


@auth_bp.route('/register', methods=['POST'])
def register_submit():
    """Handle user registration."""
    try:
        first_name = request.form.get('firstName', '').strip()
        last_name = request.form.get('lastName', '').strip()
        email = request.form.get('email', '').strip().lower()
        organization = request.form.get('organization', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirmPassword', '')
        
        # Validation
        if not all([first_name, last_name, email, password]):
            flash('All required fields must be filled.', 'error')
            return render_template('register.html')
        
        if password != confirm_password:
            flash('Passwords do not match.', 'error')
            return render_template('register.html')
        
        if len(password) < 6:
            flash('Password must be at least 6 characters long.', 'error')
            return render_template('register.html')
        
        # Check if email exists
        if db_service.email_exists(email):
            flash('An account with this email already exists.', 'error')
            return render_template('register.html')
        
        # Create user
        success = db_service.create_user(
            first_name, last_name, email, password, organization or None
        )
        
        if success:
            flash('Account created successfully! Please log in.', 'success')
            return redirect(url_for('auth.login_page'))
        else:
            flash('An error occurred during registration. Please try again.', 'error')
            return render_template('register.html')
        
    except Exception as e:
        flash('An error occurred during registration. Please try again.', 'error')
        return render_template('register.html')


@auth_bp.route('/login', methods=['GET'])
def login_page():
    """Login page route."""
    return render_template('login.html')


@auth_bp.route('/login', methods=['POST'])
def login_submit():
    """Handle user login."""
    try:
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        remember = request.form.get('remember')
        
        if not email or not password:
            return jsonify({
                'success': False, 
                'message': 'Email and password are required.'
            }), 400
        
        user = db_service.get_user_by_email(email)
        
        if user and db_service.verify_password(password, user['password_hash']):
            session['user_id'] = user['id']
            session['user_name'] = f"{user['first_name']} {user['last_name']}"
            session['user_email'] = user['email']
            
            if remember:
                session.permanent = True
            
            return jsonify({'success': True, 'message': 'Login successful!'})
        else:
            return jsonify({
                'success': False, 
                'message': 'Invalid email or password.'
            }), 401
            
    except Exception as e:
        return jsonify({
            'success': False, 
            'message': 'An error occurred during login.'
        }), 500


@auth_bp.route('/logout')
def logout():
    """Logout route."""
    session.clear()
    flash('You have been logged out successfully.', 'success')
    return redirect(url_for('auth.home'))