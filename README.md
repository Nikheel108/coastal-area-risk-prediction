# Coastal Area Risk Prediction

A Streamlit web app for predicting coastal area risks using XGBoost machine learning model with real-time weather data from OpenWeather API.

## Setup & Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Local Development

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Nikheel108/coastal-area-risk-prediction.git
   cd coastal-area-risk-prediction
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Get API Key:**
   - Sign up at [OpenWeather](https://openweathermap.org/api) to get a free API key

4. **Configure API Key (Choose one):**

   **Option A: Using .env file (Recommended for Local Development)**
   ```bash
   cp .env.example .env
   # Edit .env and add your API key
   OPENWEATHER_API_KEY=your_actual_api_key_here
   ```

   **Option B: Using Streamlit Secrets (Local Testing)**
   ```bash
   mkdir -p ~/.streamlit
   cp .streamlit/secrets.toml.example ~/.streamlit/secrets.toml
   # Edit ~/.streamlit/secrets.toml with your API key
   ```

5. **Run the app:**
   ```bash
   streamlit run app.py
   ```

   The app will open at `http://localhost:8501`

## Streamlit Cloud Deployment

### Steps:
1. Push code to GitHub (make sure `.env` is in `.gitignore`)
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Create a new app and select this repository
4. After deployment, go to **App menu** → **Settings** → **Secrets**
5. Add your OpenWeather API key:
   ```
   OPENWEATHER_API_KEY = "your_actual_api_key_here"
   ```
6. Reboot the app

## Features

- 🌊 Real-time coastal weather monitoring for 7 Indian coastal cities
- 🧠 ML-powered risk prediction (Secure, Elevated Risk, Critical Threat)
- 📊 5-day atmospheric trend visualization
- 🔍 Explainable AI (SHAP) analysis
- 🗺️ Geospatial threat map
- ⚡ Fast predictions with XGBoost

## Project Structure

```
.
├── app.py                          # Main Streamlit app
├── coastal_risk_xgboost.pkl       # Trained XGBoost model
├── coastal_scaler.pkl             # Feature scaler
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment variables template
├── .streamlit/
│   ├── config.toml               # Streamlit configuration
│   └── secrets.toml.example      # Secrets template
└── README.md                      # This file
```

## Technologies Used

- **Framework:** Streamlit
- **ML Model:** XGBoost
- **Data Processing:** Pandas, NumPy, scikit-learn
- **Visualization:** Plotly
- **Explainability:** SHAP
- **Weather Data:** OpenWeather API

## Troubleshooting

**"API Key Required" error:**
- Ensure you've set up the API key correctly (see Setup section above)
- For Streamlit Cloud, check that the secret is added in app settings

**"Error loading ML models":**
- Ensure `coastal_scaler.pkl` and `coastal_risk_xgboost.pkl` are in the project root

**App not displaying correctly:**
- Clear browser cache and refresh
- Check that all dependencies are installed: `pip install -r requirements.txt`

## License

This project is open source and available under the MIT License.

## Author

Developed by **Nikheel** | SY B.Tech AIML | Powered by XGBoost