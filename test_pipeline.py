import requests
import joblib
import pandas as pd

# 1. Configuration (Replace with your actual API key)
API_KEY = "518fa772db95f5f464b75d46770458b4"
# Testing with a coastal city in Maharashtra
CITY = "Ratnagiri" 
URL = f"http://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={API_KEY}&units=metric"

def get_live_prediction():
    print(f"Fetching live weather data for {CITY}...")
    
    try:
        # 2. Fetch Data from API
        response = requests.get(URL)
        data = response.json()
        
        if response.status_code != 200:
            print(f"Error fetching data: {data['message']}")
            return

        # 3. Extract and Convert Units to match Kaggle Dataset
        # Temp is already in Celsius because of units=metric
        temp_c = data['main']['temp']
        humidity = data['main']['humidity']
        pressure_mb = data['main']['pressure'] # hPa is equal to millibars
        
        # Convert Wind from m/s to km/h
        wind_speed_kmh = data['wind']['speed'] * 3.6 
        
        # Convert Visibility from meters to km (Default to 10km if missing)
        visibility_km = data.get('visibility', 10000) / 1000 

        print("\n--- Current Weather Conditions ---")
        print(f"Temperature: {temp_c}°C")
        print(f"Humidity: {humidity}%")
        print(f"Wind Speed: {wind_speed_kmh:.2f} km/h")
        print(f"Visibility: {visibility_km} km")
        print(f"Pressure: {pressure_mb} mb")

        # 4. Prepare data for the model (MUST match training feature order perfectly)
        features = ['Temperature (C)', 'Humidity', 'Wind Speed (km/h)', 'Visibility (km)', 'Pressure (millibars)']
        live_data = pd.DataFrame([[temp_c, humidity, wind_speed_kmh, visibility_km, pressure_mb]], columns=features)

        # 5. Load Models and Predict
        print("\nLoading ML Pipeline...")
        scaler = joblib.load('coastal_scaler.pkl')
        model = joblib.load('coastal_risk_xgboost.pkl')

        # Scale the live data
        live_data_scaled = scaler.transform(live_data)

        # Predict Risk
        prediction = model.predict(live_data_scaled)[0]
        
        # Map output to human-readable labels
        risk_map = {0: "🟢 LOW RISK (Normal Conditions)", 
                    1: "🟡 MEDIUM RISK (Be Alert)", 
                    2: "🔴 HIGH RISK (Severe Storm/Cyclone Warning)"}
        
        print("\n=====================================")
        print(f"🌊 SYSTEM ALERT: {risk_map[prediction]}")
        print("=====================================\n")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    get_live_prediction()