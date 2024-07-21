from flask import Flask, render_template, request
import joblib
import pandas as pd
from math import radians, sin, cos, sqrt, atan2
from sklearn.preprocessing import LabelEncoder


app = Flask(__name__)

# Load the saved model and preprocessing pipeline
model = joblib.load('fnn_model.joblib')
preprocessing_pipeline = joblib.load('pipeline.joblib')

# Define the columns that require one-hot encoding
one_hot_columns = ['Type of Card', 'Entry Mode', 'Type of Transaction', 'Merchant Group',
                   'Country of Transaction', 'Country of Residence', 'Gender', 'Bank', 'Shipping Address']

# Define country coordinates
country_coordinates = {
    'China': (35.8617, 104.1954),
    'India': (20.5937, 78.9629),
    'Russia': (61.5240, 105.3188),
    'USA': (37.0902, -95.7129),
    'United Kingdom': (55.3781, -3.4360)
}

# Function to extract country from one-hot encoded columns
def get_country_from_columns(row, prefix):
    for col in row.index:
        if col.startswith(prefix) and row[col] == 1:
            return col.replace(prefix, '')
    return None

# Haversine function to calculate distance
def haversine(coord1, coord2):
    # Convert latitude and longitude from degrees to radians
    lat1, lon1 = radians(coord1[0]), radians(coord1[1])
    lat2, lon2 = radians(coord2[0]), radians(coord2[1])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    radius = 6371  # Radius of Earth in kilometers
    distance = radius * c
    return distance


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about-us')
def about_us():
    return render_template('about.html')


@app.route('/contact-us')
def contact_us():
    return render_template('contact.html')

@app.route('/vision')
def vision():
    return render_template('vision.html')
# Route to render the form template
@app.route('/fraud-detection', methods=['POST','GET'])
def form():
    name = request.form.get('username')
    password = request.form.get('password')
    if name == "admin" and password=="admin":
        return render_template("index.html")

    return render_template('login.html')

# Assume day_mapping maps day names to numerical values
day_mapping = {
    'monday': 1,
    'tuesday': 2,
    'wednesday': 3,
    'thursday': 4,
    'friday': 5,
    'saturday': 6,
    'sunday': 7
}

# Function to convert day name to numerical value
def map_day_to_number(day_name):
    return day_mapping.get(day_name.lower(), 0)  # Default to 0 if day_name is not found

le = LabelEncoder()
le.fit(['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'])

# Route to handle form submission and make prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get form data and create input DataFrame
    data = {
        'Transaction ID': request.form['transactionNo'],
        'Date': request.form['date'],
        'Day of Week': request.form['day'].lower(),
        'Time': request.form['time'],
        'Type of Card': request.form['cardType'],
        'Entry Mode': request.form['entryMode'],
        'Amount': float(request.form['amount']),
        'Type of Transaction': request.form['transactionType'],
        'Merchant Group': request.form['merchantGroup'],
        'Country of Transaction': request.form['country'],
        'Shipping Address': request.form['shippingAddress'],
        'Country of Residence': request.form['residence'],
        'Gender': request.form['gender'],
        'Age': int(request.form['age']),
        'Bank': request.form['bank']
    }
    input_df = pd.DataFrame([data])

    print("Input DataFrame:\n", input_df)

    # Convert 'Date' column to datetime format
    input_df['Date'] = pd.to_datetime(input_df['Date'])

    # Apply label encoding for 'Day of Week'
    input_df['Day of Week'] = le.transform(input_df['Day of Week'])

    # Apply one-hot encoding
    one_hot_encoded_df = pd.get_dummies(input_df, columns=one_hot_columns, drop_first=True)

    print("One-Hot Encoded DataFrame:\n", one_hot_encoded_df)

    # Ensure all expected columns are present in the input
    expected_columns = preprocessing_pipeline.named_steps['std_scaler'].get_feature_names_out()
    missing_cols = set(expected_columns) - set(one_hot_encoded_df.columns)
    for col in missing_cols:
        one_hot_encoded_df[col] = 0
    one_hot_encoded_df = one_hot_encoded_df[expected_columns]

    print("One-Hot Encoded DataFrame with all expected columns:\n", one_hot_encoded_df)

    # Apply remaining preprocessing
    preprocessed_data = preprocessing_pipeline.transform(one_hot_encoded_df)

    print("Preprocessed Data Shape:\n", preprocessed_data.shape)
    print("Preprocessed Data:\n", preprocessed_data)

    # Make prediction
    prediction = model.predict(preprocessed_data)

    print("Prediction:\n", prediction[0])

    return render_template('result.html', prediction=prediction[0])


if __name__ == "__main__":
    app.run(debug=True)
