import os
import psycopg2
from flask import Flask, render_template, request, redirect, url_for, flash, session
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'default_dev_key')

# Use the Transaction pooler connection string which is IPv4 compatible
DATABASE_URL = "postgresql://postgres.plaftwkftdfqjlwaeilp:88FRiObNWvKGuPzH@aws-0-sa-east-1.pooler.supabase.com:6543/postgres"

def get_db_connection():
    try:
        # Using the Transaction pooler URL
        conn = psycopg2.connect(DATABASE_URL)
        app.logger.info("Database connection successful using Transaction pooler")
        return conn
    except psycopg2.OperationalError as e:
        app.logger.error(f"Database connection error: {e}")
        return None

# The rest of your routes remain the same
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username')
    password = request.form.get('password')
    role = request.form.get('role')

    if not all([username, password, role]):
        flash("All fields are required")
        return redirect(url_for('index'))

    conn = get_db_connection()
    if not conn:
        flash("Database connection error. Please try again later.")
        return redirect(url_for('index'))

    try:
        with conn.cursor() as cursor:
            db_role = 'admin' if role == 'admin' else 'technician'
            
            # Verify password using PostgreSQL's crypt()
            cursor.execute(
                """SELECT id, username, role FROM users 
                WHERE username = %s AND role = %s 
                AND password = crypt(%s, password)""",
                (username, db_role, password)
            )
            user = cursor.fetchone()

            if user:
                session['user_id'] = user[0]
                session['username'] = user[1]
                session['role'] = user[2]

                if user[2] == 'admin':
                    return render_template('dashboard.html', 
                                        username=username.title(), 
                                        role=user[2])
                else:
                    flash("Access denied: You are not an admin.")
                    return redirect(url_for('index'))
            else:
                flash("Invalid credentials. Please try again.")
                return redirect(url_for('index'))
    except Exception as e:
        app.logger.error(f"Login error: {e}")
        flash("An error occurred. Please try again.")
        return redirect(url_for('index'))
    finally:
        conn.close()

@app.route('/logout')
def logout():
    session.clear()
    flash("You have been logged out")
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)