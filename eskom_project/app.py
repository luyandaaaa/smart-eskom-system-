from flask import Flask, render_template, jsonify, request, redirect, url_for, flash, session
import pandas as pd
from datetime import datetime
import os
import psycopg2

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Add a secret key for sessions and flash messages

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

# Load data from CSV files
def load_data():
    # Load meter data
    soweto_meters = pd.read_csv('soweto_smart_meters.csv')
    tembisa_meters = pd.read_csv('tembisa_smart_meters.csv')
    
    # Load substation data
    substation_data = pd.read_csv('substation_consumption.csv')
    
    # Convert dates to datetime objects - Using format='mixed' to handle different date formats
    soweto_meters['purchase_date'] = pd.to_datetime(soweto_meters['purchase_date'], format='mixed')
    tembisa_meters['purchase_date'] = pd.to_datetime(tembisa_meters['purchase_date'], format='mixed')
    substation_data['date'] = pd.to_datetime(substation_data['date'], format='mixed')
    
    # Add area identifier
    soweto_meters['area'] = 'soweto'
    tembisa_meters['area'] = 'tembisa'
    
    # Combine meter data
    all_meters = pd.concat([soweto_meters, tembisa_meters])
    
    # Detect suspicious meters (simple heuristic for demo)
    all_meters['status'] = 'active'
    all_meters.loc[all_meters['kwh_purchased'] == 0, 'status'] = 'inactive'
    
    # More sophisticated anomaly detection would go here
    avg_kwh = all_meters[all_meters['kwh_purchased'] > 0]['kwh_purchased'].mean()
    std_kwh = all_meters[all_meters['kwh_purchased'] > 0]['kwh_purchased'].std()
    threshold = avg_kwh + 2 * std_kwh
    
    all_meters.loc[
        (all_meters['kwh_purchased'] > threshold) & 
        (all_meters['status'] == 'active'), 
        'status'
    ] = 'suspicious'
    
    return {
        'soweto_meters': soweto_meters.to_dict('records'),
        'tembisa_meters': tembisa_meters.to_dict('records'),
        'all_meters': all_meters.to_dict('records'),
        'substation_data': substation_data.to_dict('records')
    }

# Global data variable
data = load_data()

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

@app.route('/alerts')
def alerts():
    return render_template('alerts.html')

@app.route('/map')
def map_view():
    return render_template('map.html')

@app.route('/reports')
def reports():
    return render_template('reports.html')

@app.route('/technicians')
def technicians():
    return render_template('technicians.html')



@app.route('/logout')
def logout():
    session.clear()
    flash("You have been logged out")
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    # Check if user is logged in and is an admin
    if 'user_id' not in session or session.get('role') != 'admin':
        flash("Please login as an admin to access the dashboard")
        return redirect(url_for('index'))
    
    return render_template('dashboard.html')

@app.route('/api/meter-data')
def get_meter_data():
    # Check if user is logged in
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized access"}), 401
    
    return jsonify(data)

@app.route('/api/summary-stats')
def get_summary_stats():
    # Check if user is logged in
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized access"}), 401
    
    all_meters = data['all_meters']
    soweto_meters = [m for m in all_meters if m['area'] == 'soweto']
    tembisa_meters = [m for m in all_meters if m['area'] == 'tembisa']
    
    def calculate_stats(meters):
        total = len(meters)
        active = len([m for m in meters if m['status'] == 'active'])
        suspicious = len([m for m in meters if m['status'] == 'suspicious'])
        inactive = len([m for m in meters if m['status'] == 'inactive'])
        total_revenue = 0
        for m in meters:
            try:
                if pd.notna(m['amount_paid']):
                    total_revenue += float(m['amount_paid'])
            except (ValueError, TypeError):
                continue
        total_kwh = 0
        for m in meters:
            try:
                if pd.notna(m['kwh_purchased']):
                    total_kwh += float(m['kwh_purchased'])
            except (ValueError, TypeError):
                continue
        avg_kwh = total_kwh / total if total > 0 else 0
        
        # Find peak purchase time (simplified)
        purchase_times = []
        for m in meters:
            try:
                if pd.notna(m['purchase_time']):
                    time_obj = datetime.strptime(m['purchase_time'], '%H:%M:%S').time()
                    purchase_times.append(time_obj.hour)
            except:
                continue
        
        peak_hour = max(set(purchase_times), key=purchase_times.count) if purchase_times else None
        peak_time = f"{peak_hour}:00-{peak_hour+1}:00" if peak_hour is not None else "N/A"
        
        return {
            'total_meters': total,
            'active_meters': active,
            'suspicious_meters': suspicious,
            'inactive_meters': inactive,
            'total_revenue': total_revenue,
            'total_kwh': total_kwh,
            'avg_kwh': avg_kwh,
            'peak_time': peak_time
        }
    
    return jsonify({
        'all': calculate_stats(all_meters),
        'soweto': calculate_stats(soweto_meters),
        'tembisa': calculate_stats(tembisa_meters)
    })

@app.route('/api/consumption-trends')
def get_consumption_trends():
    # Check if user is logged in
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized access"}), 401
    
    substation_data = data['substation_data']
    
    # Group by month for trends
    df = pd.DataFrame(substation_data)
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.strftime('%b')
    
    monthly_trends = df.groupby('month')['total_kwh_supplied'].sum().reset_index()
    
    return jsonify({
        'labels': monthly_trends['month'].tolist(),
        'data': monthly_trends['total_kwh_supplied'].tolist()
    })

@app.route('/api/alerts')
def get_alerts():
    # Check if user is logged in
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized access"}), 401
    
    all_meters = data['all_meters']
    
    # Generate alerts based on suspicious meters
    alerts = []
    for meter in all_meters:
        if meter['status'] == 'suspicious':
            alerts.append({
                'type': 'warning',
                'title': 'High Consumption Detected',
                'location': meter['address'],
                'time': f"{meter['purchase_date'].strftime('%d/%m/%Y')} {meter['purchase_time']}",
                'icon': 'fa-exclamation-triangle',
                'meter_id': meter['meter_number']
            })
        elif meter['status'] == 'inactive':
            alerts.append({
                'type': 'info',
                'title': 'Inactive Meter',
                'location': meter['address'],
                'time': f"{meter['purchase_date'].strftime('%d/%m/%Y')} {meter['purchase_time']}",
                'icon': 'fa-info-circle',
                'meter_id': meter['meter_number']
            })
    
    # Sort by date (newest first)
    alerts.sort(key=lambda x: datetime.strptime(x['time'], '%d/%m/%Y %H:%M:%S'), reverse=True)
    
    return jsonify(alerts[:10])  # Return only the 10 most recent alerts

@app.route('/api/high-risk-areas')
def get_high_risk_areas():
    # Check if user is logged in
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized access"}), 401
    
    all_meters = data['all_meters']
    
    # Group by address blocks to find high risk areas
    address_blocks = {}
    for meter in all_meters:
        # Extract block identifier from address (e.g., "Block A" from "Block A House 1 Soweto")
        block = ' '.join(meter['address'].split()[:2])
        
        if block not in address_blocks:
            address_blocks[block] = {
                'suspicious': 0,
                'inactive': 0,
                'total': 0,
                'area': meter['area']
            }
        
        address_blocks[block]['total'] += 1
        if meter['status'] == 'suspicious':
            address_blocks[block]['suspicious'] += 1
        elif meter['status'] == 'inactive':
            address_blocks[block]['inactive'] += 1
    
    # Calculate risk scores
    risk_areas = []
    for block, stats in address_blocks.items():
        risk_score = (stats['suspicious'] * 2 + stats['inactive']) / stats['total'] if stats['total'] > 0 else 0
        
        if risk_score > 0.5:
            risk_level = 'high'
        elif risk_score > 0.2:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        risk_areas.append({
            'name': f"{block} ({stats['area'].capitalize()})",
            'incidents': stats['suspicious'] + stats['inactive'],
            'risk': risk_level,
            'area': stats['area']
        })
    
    # Sort by risk level (high to low)
    risk_areas.sort(key=lambda x: (
        -float('inf') if x['risk'] == 'high' else 
        -1 if x['risk'] == 'medium' else 0,
        -x['incidents']
    ))
    
    return jsonify(risk_areas)

if __name__ == '__main__':
    app.run(debug=True)