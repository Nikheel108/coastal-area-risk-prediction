import streamlit as st
import requests
import joblib
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import shap
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# ==========================================
# 1. PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(page_title="Coastal Defense Command", page_icon="🌊", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;500;600;700;800;900&display=swap');
    html, body, [class*="css"]  { font-family: 'Sora', 'Segoe UI', sans-serif; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    .title-text { font-size: 3rem; background: -webkit-linear-gradient(#0284C7, #0EA5E9); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 900; text-align: center; margin-bottom: 0px; padding-top: 1rem;}
    .subtitle-text { text-align: center; color: #475569; font-size: 1.1rem; margin-bottom: 2rem; font-weight: 500; letter-spacing: 1px;}
    
    .metric-container { display: flex; justify-content: space-between; gap: 10px; margin-top: 10px; margin-bottom: 20px;}
    .metric-card { background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 15px 10px; width: 100%; display: flex; flex-direction: column; align-items: center; justify-content: center; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.04);}
    .metric-value { font-size: 1.5rem; font-weight: 900; color: #0F172A; margin: 0; line-height: 1.1;}
    .metric-label { font-size: 0.8rem; color: #475569; font-weight: 800; text-transform: uppercase; margin-top: 6px;}

    .alert-low { background: #ECFDF5; border-left: 8px solid #10B981; padding: 15px; border-radius: 8px; margin: 10px 0; color: #064E3B; box-shadow: 0 2px 5px rgba(0,0,0,0.05);}
    .alert-med { background: #FFFBEB; border-left: 8px solid #F59E0B; padding: 15px; border-radius: 8px; margin: 10px 0; color: #78350F; box-shadow: 0 2px 5px rgba(0,0,0,0.05);}
    .alert-high { background: #FEF2F2; border-left: 8px solid #EF4444; padding: 15px; border-radius: 8px; margin: 10px 0; color: #7F1D1D; box-shadow: 0 4px 12px rgba(239,68,68,0.3); animation: pulse-light 2s infinite;}
    
    @keyframes pulse-light {
        0% { box-shadow: 0 0 0 0 rgba(239,68,68,0.4); }
        70% { box-shadow: 0 0 0 15px rgba(239,68,68,0); }
        100% { box-shadow: 0 0 0 0 rgba(239,68,68,0); }
    }
    .alert-title { font-size: 1.3rem; font-weight: 900; margin-bottom: 2px; }
    .alert-desc { font-size: 0.95rem; font-weight: 500; opacity: 0.9; margin: 0;}
    .section-header { font-size: 1.4rem; font-weight: 800; color: #0F172A; margin-top: 15px; border-bottom: 3px solid #CBD5E1; padding-bottom: 5px;}
    
    .stTabs [data-baseweb="tab-list"] { gap: 20px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: transparent; border-radius: 4px 4px 0px 0px; padding-top: 10px; padding-bottom: 10px; font-size: 1.1rem; font-weight: 700; color: #475569;}
    .stTabs [aria-selected="true"] { color: #0284C7 !important; border-bottom: 3px solid #0284C7 !important;}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. CONFIG & ML PIPELINE
# ==========================================
# Load API key from Streamlit Secrets (production) or .env (development)
try:
    API_KEY = st.secrets.get("OPENWEATHER_API_KEY")
except Exception:
    API_KEY = os.getenv('OPENWEATHER_API_KEY', None)

@st.cache_resource
def load_ml_components():
    scaler = joblib.load('coastal_scaler.pkl')
    model = joblib.load('coastal_risk_xgboost.pkl')
    return scaler, model

try:
    scaler, model = load_ml_components()
except Exception as e:
    st.error(f"Error loading ML models: {e}")
    st.stop()

coastal_coords = {
    "Ratnagiri": {"lat": 16.9902, "lon": 73.3000},
    "Mumbai": {"lat": 18.9667, "lon": 72.8333},
    "Chennai": {"lat": 13.0827, "lon": 80.2707},
    "Kochi": {"lat": 9.9312, "lon": 76.2673},
    "Mangaluru": {"lat": 12.9141, "lon": 74.8560},
    "Puri": {"lat": 19.8135, "lon": 85.8312},
    "Visakhapatnam": {"lat": 17.6868, "lon": 83.2185}
}

features_list = ['Temperature (C)', 'Humidity', 'Wind Speed (km/h)', 'Visibility (km)', 'Pressure (millibars)']

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def predict_risk(temp, humidity, wind, visibility, pressure):
    df = pd.DataFrame([[temp, humidity, wind, visibility, pressure]], columns=features_list)
    scaled_data = scaler.transform(df)
    return model.predict(scaled_data)[0], scaled_data

def render_alert(prediction, is_future=False):
    time_context = "in 72 Hours" if is_future else "Currently"
    if prediction == 0:
        return f'<div class="alert-low"><div class="alert-title">🟢 SECURE ({time_context})</div><div class="alert-desc">Atmospheric conditions stable.</div></div>'
    elif prediction == 1:
        return f'<div class="alert-med"><div class="alert-title">🟡 ELEVATED RISK ({time_context})</div><div class="alert-desc">Anomalous metrics detected. Advise monitoring.</div></div>'
    elif prediction == 2:
        return f'<div class="alert-high"><div class="alert-title">🔴 CRITICAL THREAT ({time_context})</div><div class="alert-desc">Severe cyclone/flood conditions predicted.</div></div>'

# ==========================================
# 4. MAIN DASHBOARD UI
# ==========================================
spacer1, main_col, spacer2 = st.columns([1, 4, 1])

with main_col:
    st.markdown('<div class="title-text">🌊 COASTAL DEFENSE COMMAND</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle-text">Dual-Phase Meteorological Radar & Threat Assessment</div>', unsafe_allow_html=True)

    CITY = st.selectbox("TARGET REGION", list(coastal_coords.keys()), label_visibility="collapsed")

    if st.button("INITIALIZE DUAL-PHASE SCAN 🔍", width='stretch', type="primary"):
        if not API_KEY or API_KEY == 'YOUR_OPENWEATHER_API_KEY_HERE':
            st.error("⚠️ **API Key Required** - Please configure your OpenWeather API key:\n\n**For Local Development:** Add `OPENWEATHER_API_KEY=your_key` to `.env`\n\n**For Streamlit Cloud:** Add the secret in app settings → Secrets")
        else:
            with st.spinner('Establishing uplink with weather satellites... 📡'):
                try:
                    # --- API CALLS ---
                    curr_url = f"http://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={API_KEY}&units=metric"
                    curr_res = requests.get(curr_url).json()
                    
                    fore_url = f"http://api.openweathermap.org/data/2.5/forecast?q={CITY}&appid={API_KEY}&units=metric"
                    fore_res = requests.get(fore_url).json()
                    
                    if curr_res.get('cod') != 200 or str(fore_res.get('cod')) != "200":
                        st.error("Signal Lost: Check API Key or City Name.")
                    else:
                        c_temp, c_hum, c_press = curr_res['main']['temp'], curr_res['main']['humidity'], curr_res['main']['pressure']
                        c_wind = curr_res['wind']['speed'] * 3.6 
                        c_vis = curr_res.get('visibility', 10000) / 1000 
                        
                        f_data = fore_res['list'][23] # 72-Hour Data
                        f_temp, f_hum, f_press = f_data['main']['temp'], f_data['main']['humidity'], f_data['main']['pressure']
                        f_wind = f_data['wind']['speed'] * 3.6 
                        f_vis = f_data.get('visibility', 10000) / 1000 

                        curr_pred, curr_scaled = predict_risk(c_temp, c_hum, c_wind, c_vis, c_press)
                        fore_pred, fore_scaled = predict_risk(f_temp, f_hum, f_wind, f_vis, f_press)

                        tab_curr, tab_fore, tab_analytics, tab_map = st.tabs(["🌤️ Live Conditions (T+0)", "🔮 72-Hour Outlook (T+72)", "📊 Advanced Analytics", "🗺️ Geospatial Threat Map"])
                        
                        # --- TAB 1: LIVE ---
                        with tab_curr:
                            st.markdown(f"""
                            <div class="metric-container">
                                <div class="metric-card"><p class="metric-value">{c_wind:.1f}</p><p class="metric-label">💨 Wind (km/h)</p></div>
                                <div class="metric-card"><p class="metric-value">{c_press}</p><p class="metric-label">📉 Press (mb)</p></div>
                                <div class="metric-card"><p class="metric-value">{c_hum}%</p><p class="metric-label">💧 Humidity</p></div>
                                <div class="metric-card"><p class="metric-value">{c_temp:.1f}°</p><p class="metric-label">🌡️ Temp (C)</p></div>
                            </div>
                            """, unsafe_allow_html=True)
                            st.markdown(render_alert(curr_pred, False), unsafe_allow_html=True)

                        # --- TAB 2: FORECAST ---
                        with tab_fore:
                            st.markdown(f"""
                            <div class="metric-container">
                                <div class="metric-card"><p class="metric-value">{f_wind:.1f}</p><p class="metric-label">💨 Wind (km/h)</p></div>
                                <div class="metric-card"><p class="metric-value">{f_press}</p><p class="metric-label">📉 Press (mb)</p></div>
                                <div class="metric-card"><p class="metric-value">{f_hum}%</p><p class="metric-label">💧 Humidity</p></div>
                                <div class="metric-card"><p class="metric-value">{f_temp:.1f}°</p><p class="metric-label">🌡️ Temp (C)</p></div>
                            </div>
                            """, unsafe_allow_html=True)
                            st.markdown(render_alert(fore_pred, True), unsafe_allow_html=True)

                        # --- TAB 3: HIGH-CONTRAST ANALYTICS ---
                        with tab_analytics:
                            st.markdown("### 📈 5-Day Atmospheric Threat Trend")
                            st.caption("Visualizing the interaction between Wind Speed and Atmospheric Pressure. (High Contrast Mode)")
                            
                            dates = [datetime.strptime(item['dt_txt'], '%Y-%m-%d %H:%M:%S') for item in fore_res['list']]
                            winds = [item['wind']['speed'] * 3.6 for item in fore_res['list']]
                            pressures = [item['main']['pressure'] for item in fore_res['list']]
                            df_trend = pd.DataFrame({'Date': dates, 'Wind Speed (km/h)': winds, 'Pressure (mb)': pressures})
                            
                            # HIGH CONTRAST TREND GRAPH
                            fig_trend = go.Figure()
                            fig_trend.add_trace(go.Scatter(x=df_trend['Date'], y=df_trend['Wind Speed (km/h)'], name="Wind Speed", mode='lines+markers', line=dict(color="#DC2626", width=4), marker=dict(size=8, color="#991B1B")))
                            fig_trend.add_trace(go.Scatter(x=df_trend['Date'], y=df_trend['Pressure (mb)'], name="Pressure", yaxis="y2", mode='lines+markers', line=dict(color="#2563EB", width=4, dash='dot'), marker=dict(size=8, color="#1E40AF")))
                            
                            fig_trend.update_layout(
                                plot_bgcolor="#F8FAFC", # Light gray background to make lines pop
                                paper_bgcolor="rgba(0,0,0,0)",
                                font=dict(family="Sora, Segoe UI, sans-serif", color="#0F172A"),
                                xaxis=dict(showgrid=True, gridwidth=1, gridcolor='#E2E8F0', title=dict(text="Forecast Timeline (Next 120 Hours)", font=dict(color="#0F172A", size=14)), tickfont=dict(color="#0F172A", size=12)),
                                yaxis=dict(showgrid=True, gridwidth=1, gridcolor='#E2E8F0', title=dict(text="Wind Speed (km/h)", font=dict(color="#DC2626", size=14)), tickfont=dict(color="#DC2626", size=12)),
                                yaxis2=dict(title=dict(text="Pressure (mb)", font=dict(color="#2563EB", size=14)), tickfont=dict(color="#2563EB", size=12), anchor="x", overlaying="y", side="right"),
                                margin={"r":0,"t":40,"l":0,"b":0}, hovermode="x unified"
                            )
                            st.plotly_chart(fig_trend, width='stretch')

                            st.write("---")

                            st.markdown("### 🧠 Explainable AI (XAI) Engine")
                            st.caption("Mathematical breakdown of why the XGBoost model predicted the current threat level.")
                            
                            explainer = shap.TreeExplainer(model)
                            shap_values = explainer.shap_values(curr_scaled)
                            
                            if isinstance(shap_values, list): impact_values = shap_values[curr_pred][0]
                            elif len(shap_values.shape) == 3: impact_values = shap_values[0, :, curr_pred]
                            else: impact_values = shap_values[0]

                            df_shap = pd.DataFrame({'Feature': features_list, 'Impact on Prediction': impact_values})
                            df_shap = df_shap.sort_values(by='Impact on Prediction', ascending=True)

                            # HIGH CONTRAST SHAP GRAPH
                            fig_shap = px.bar(
                                df_shap, x='Impact on Prediction', y='Feature', orientation='h',
                                color='Impact on Prediction', 
                                color_continuous_scale=['#1E3A8A', '#E2E8F0', '#991B1B'] # Dark Blue to Dark Red
                            )
                            # Adding black borders around the bars for crisp visibility
                            fig_shap.update_traces(marker_line_color='#0F172A', marker_line_width=1.5, opacity=1)
                            fig_shap.update_layout(
                                plot_bgcolor="#F8FAFC", paper_bgcolor="rgba(0,0,0,0)", showlegend=False,
                                font=dict(family="Sora, Segoe UI, sans-serif", color="#0F172A"),
                                xaxis=dict(title=dict(text="Mathematical Impact Weight", font=dict(color="#0F172A", size=14)), tickfont=dict(color="#0F172A", size=12)),
                                yaxis=dict(title="", tickfont=dict(color="#0F172A", size=13)),
                                margin={"r":0,"t":20,"l":0,"b":0}
                            )
                            st.plotly_chart(fig_shap, width='stretch')

                        # --- TAB 4: HIGH CONTRAST RADAR MAP ---
                        with tab_map:
                            st.markdown('<div class="section-header">Geospatial Threat Map</div>', unsafe_allow_html=True)
                            map_data = []
                            for c_name, coords in coastal_coords.items():
                                if c_name == CITY:
                                    if curr_pred == 0: status, color_code, size = "Secure", "#10B981", 20
                                    elif curr_pred == 1: status, color_code, size = "Warning", "#F59E0B", 25
                                    else: status, color_code, size = "Critical", "#EF4444", 30
                                else:
                                    status, color_code, size = "Monitoring", "#64748B", 10
                                
                                map_data.append({"City": c_name, "Latitude": coords["lat"], "Longitude": coords["lon"], "Status": status, "Size": size})

                            df_map = pd.DataFrame(map_data)
                            fig_map = px.scatter_map(
                                df_map, lat="Latitude", lon="Longitude", hover_name="City", color="Status", size="Size",
                                color_discrete_map={"Secure": "#10B981", "Warning": "#F59E0B", "Critical": "#EF4444", "Monitoring": "#64748B"},
                                zoom=4.5, center={"lat": 19.5, "lon": 78.5}, 
                                map_style="open-street-map" # Switched to OSM for incredible land/water contrast
                            )
                            
                            # Scatter map markers - use opacity for better contrast.
                            fig_map.update_traces(marker=dict(opacity=0.95))
                            fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(255,255,255,0.8)"))
                            st.plotly_chart(fig_map, width='stretch')

                except Exception as e:
                    st.error(f"System Failure: {e}")

st.write("---")
st.markdown("<p style='text-align: center; color: #64748B; font-weight: 600;'>Developed by Nikheel | SY B.Tech AIML | Powered by XGBoost</p>", unsafe_allow_html=True)