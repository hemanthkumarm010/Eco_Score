from flask import Flask, request, jsonify, send_from_directory
from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
from flask_bcrypt import Bcrypt
import mysql.connector
from sklearn.preprocessing import StandardScaler
from predict import predict_sustainability  # Assuming predict.py contains the prediction function

appy = Flask(__name__)
bcrypt = Bcrypt(appy)

app = Flask(__name__, static_folder='frontend')

# Load your model and the scaler used during training
model = joblib.load(r'C:\Users\Satya Prakash\OneDrive\Desktop\eco-score\models\random_forest_model.joblib')
scaler = joblib.load(r'C:\Users\Satya Prakash\OneDrive\Desktop\eco-score\models\scaler.joblib')  # Load the scaler if saved

EMISSION_FACTORS = {
    "Transportation": 0.2,  # kgCO2/km
    "Electricity": 0.475,   # kgCO2/kWh
    "Diet": 1.5,            # kgCO2/meal
    "Waste": 0.15           # kgCO2/kg
}

db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'Anju@2004',
    'database': 'eco_score'
}

# Database connection
def get_db_connection():
    return mysql.connector.connect(**db_config)

#@app.route('/')
#def serve_frontend():
    #return send_from_directory('frontend', 'main.html')
    
@app.route('/calculate', methods=['POST'])
def calculate():
    # Get user inputs from the form
    distance = float(request.form['distance']) * 365  # Daily to yearly
    electricity = float(request.form['electricity']) * 12  # Monthly to yearly
    waste = float(request.form['waste']) * 52  # Weekly to yearly
    meals = int(request.form['meals']) * 365  # Daily to yearly

    # Calculate carbon emissions
    transportation_emissions = EMISSION_FACTORS["Transportation"] * distance
    electricity_emissions = EMISSION_FACTORS["Electricity"] * electricity
    diet_emissions = EMISSION_FACTORS["Diet"] * meals
    waste_emissions = EMISSION_FACTORS["Waste"] * waste

    # Convert emissions to tonnes and round to 2 decimals
    transportation_emissions = round(transportation_emissions / 1000, 2)
    electricity_emissions = round(electricity_emissions / 1000, 2)
    diet_emissions = round(diet_emissions / 1000, 2)
    waste_emissions = round(waste_emissions / 1000, 2)

    # Calculate total emissions
    total_emissions = round(
        transportation_emissions + electricity_emissions + diet_emissions + waste_emissions, 2
    )

    # Prepare results to send back
    results = {
        "transportation": transportation_emissions,
        "electricity": electricity_emissions,
        "diet": diet_emissions,
        "waste": waste_emissions,
        "total": total_emissions
    }
    return jsonify(results)


@app.route('/index.html')
def serve_index():
    # Serve the prediction page
    return send_from_directory('frontend', 'index.html')

@app.route('/')
def serve_after_login():
    # Serve the prediction page
    return send_from_directory('frontend', 'afterlogin.html')

@app.route('/dashboard.html')
def serve_carbon():
    # Serve the prediction page
    return send_from_directory('frontend', 'dashboard.html')

@app.route('/carbon.html')
def serve_dashboard():
    # Serve the prediction page
    return send_from_directory('frontend', 'carbon.html')
    
@app.route('/<path:path>')
def send_static(path):
    return send_from_directory('frontend', path)

@app.route('/signup', methods=['POST'])
def signup():
    try:
        # Log received data
        data = request.get_json()
        print(f"Received signup data: {data}")

        email = data.get('email')
        username = data.get('username')
        password = data.get('password')

        print(f"Extracted fields - Email: {email}, Username: {username}")

        # Database connection
        conn = get_db_connection()
        print("Database connection established")

        cursor = conn.cursor()

        # Hash the password
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        print("Password hashed successfully")

        # Insert user into the database
        query = "INSERT INTO users (email, username, password) VALUES (%s, %s, %s)"
        cursor.execute(query, (email, username, hashed_password))
        conn.commit()

        print("User registered successfully in the database")
        return jsonify({'message': 'User registered successfully!'}), 201

    except Exception as e:
        print(f"Error during signup: {e}")
        return jsonify({'error': str(e)}), 500

    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        # Fetch user details from the database
        query = "SELECT * FROM users WHERE email = %s"
        cursor.execute(query, (email,))
        user = cursor.fetchone()

        if user and bcrypt.check_password_hash(user['password'], password):
            return jsonify({'message': 'Login successful!', 'redirect': '/afterlogin.html'}), 200
        else:
            return jsonify({'error': 'Invalid credentials!'}), 401
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()  # Logs the full stack trace
        return jsonify({'error': str(e)}), 500
    finally:
        cursor.close()
        conn.close()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the request
        user_input = request.get_json()
        print("Received user input:", user_input)

        # Call the predict function from predict.py
        prediction = predict_sustainability(model, user_input, scaler)

        # Return the prediction as JSON
        print("Prediction:", prediction)
        return jsonify({'prediction': prediction})

    except Exception as e:
        print("Error occurred:", e)  # Print any exceptions
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)