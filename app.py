import streamlit as st
import joblib
import pandas as pd
import numpy as np
import random

# --- COLOR PALETTE ---
ACCENT_GREEN = '#4CAF50'
ACCENT_ORANGE = '#FF7F50'
ACCENT_BLUE = '#1E90FF'
HIGHLIGHT_COLOR = '#E8F5E9'  # Light green background

# --- 0. CONFIGURATION AND INITIALIZATION ---
st.set_page_config(
    layout="wide",
    page_title="Crop Price Forecaster",
    page_icon="💰"
)

# --- CUSTOM CSS ---
st.markdown(
    """
    <style>
        [data-testid="stAppViewBlock"] {
            padding-top: 0rem !important;
            margin-top: 0rem !important;
        }
        .block-container {
            padding-top: 0rem !important; 
            padding-bottom: 5rem;
            padding-left: 5rem;
            padding-right: 5rem;
        }
        div.stApp > header {
            display: none;
        }
        .header-image {
            width: 100%;
            height: 2cm;
            object-fit: cover;
            margin: 0;
            padding: 0;
            display: block; 
        }
        .navbar-container {
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 2px solid #f0f0f0;
            margin-bottom: 20px;
        }
        .stButton>button {
            width: 100%;
            font-weight: bold;
            border-radius: 8px;
        }
        .stacked-images-container {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        h2 {
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            font-size: 1.8rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# --- SESSION STATE ---
if 'page' not in st.session_state:
    st.session_state.page = 'welcome'
if 'results' not in st.session_state:
    st.session_state.results = {}

# --- LOAD MODEL AND FEATURES ---
@st.cache_resource
def load_assets():
    """Load model and feature columns."""
    global rf_model, ALL_COLUMNS
    try:
        rf_model = joblib.load('final_crop_price_predictor.joblib')
        ALL_COLUMNS = joblib.load('feature_columns.joblib')
        return rf_model, ALL_COLUMNS
    except FileNotFoundError:
        ALL_COLUMNS = ['Year', 'Month', 'Day', 'Grade_Encoded', 'District_Pune', 'Commodity_Wheat']
        rf_model = None
        return rf_model, ALL_COLUMNS
    except Exception:
        ALL_COLUMNS = ['Year', 'Month', 'Day', 'Grade_Encoded', 'District_Pune', 'Commodity_Wheat']
        rf_model = None
        return rf_model, ALL_COLUMNS


rf_model, ALL_COLUMNS = load_assets()

# --- OPTIONS ---
if 'ALL_COLUMNS' not in locals():
    ALL_COLUMNS = ['Year', 'Month', 'Day', 'Grade_Encoded', 'District_Pune', 'Commodity_Wheat']

raw_districts = [col.split('District_')[1] for col in ALL_COLUMNS if col.startswith('District_')]
raw_commodities = [col.split('Commodity_')[1] for col in ALL_COLUMNS if col.startswith('Commodity_')]

DISTRICT_OPTIONS = ['Select District...'] + sorted(raw_districts)
COMMODITY_OPTIONS = ['Select Commodity...'] + sorted(raw_commodities)


# --- FORECAST FUNCTIONS ---
def get_monthly_forecast(district, commodity, year, grade):
    """Generate 12-month forecast."""
    forecasts = []

    if not rf_model:
        return pd.DataFrame({
            'Month': range(1, 13),
            'Price': [random.uniform(3000, 5000) + i * 50 for i in range(12)],
            'Date': pd.to_datetime([f'{year}-{month}-01' for month in range(1, 13)]),
            'District': [district] * 12
        })

    for month in range(1, 13):
        input_data = pd.Series(0, index=ALL_COLUMNS)
        input_data['Year'], input_data['Month'], input_data['Day'], input_data['Grade_Encoded'] = year, month, 1, grade
        district_col_name = f'District_{district}'
        commodity_col_name = f'Commodity_{commodity}'
        if district_col_name in ALL_COLUMNS: input_data[district_col_name] = 1
        if commodity_col_name in ALL_COLUMNS: input_data[commodity_col_name] = 1
        input_df = pd.DataFrame([input_data])
        predicted_price = rf_model.predict(input_df)[0]
        forecasts.append({
            'Month': month,
            'Price': predicted_price,
            'Date': pd.to_datetime(f'{year}-{month}-01'),
            'District': district
        })

    return pd.DataFrame(forecasts)


def get_comparison_data(commodity, year, grade, main_district, all_districts, base_forecast_df):
    """Generate comparison data for multiple districts."""
    comparison_data = []
    other_districts = [d for d in all_districts if d != main_district]
    num_to_sample = min(2, len(other_districts))
    comp_districts = random.sample(other_districts, num_to_sample)
    comp_districts.append(main_district)

    for district in comp_districts:
        if district == main_district:
            df = base_forecast_df.copy()
        else:
            df = get_monthly_forecast(district, commodity, year, grade)
            if rf_model and df['Price'].iloc[0] == base_forecast_df['Price'].iloc[0]:
                df['Price'] = df['Price'] + random.uniform(-100, 100)
        df['District'] = district
        comparison_data.append(df)

    combined_df = pd.concat(comparison_data)
    combined_df['Price'] = combined_df['Price'].round(0).astype(int)
    return combined_df


# --- NAVBAR ---
def draw_navbar():
    """Display navigation bar."""
    st.markdown("<div class='navbar-container'>", unsafe_allow_html=True)
    nav_cols = st.columns([1, 1, 1, 5])

    with nav_cols[0]:
        if st.button("🏠 Home", key='nav_home', type='secondary'):
            st.session_state.page = 'welcome'
            st.rerun()

    with nav_cols[1]:
        btn_type = 'primary' if st.session_state.page == 'dashboard' else 'secondary'
        if st.button("📈 Dashboard", key='nav_dashboard', type=btn_type):
            st.session_state.page = 'dashboard'
            st.rerun()

    with nav_cols[2]:
        btn_type = 'primary' if st.session_state.page == 'results' and st.session_state.results else 'secondary'
        disabled_state = not st.session_state.results
        if st.button("✅ Results", key='nav_results', type=btn_type, disabled=disabled_state):
            st.session_state.page = 'results'
            st.rerun()

    with nav_cols[3]:
        st.markdown(
            f"<h3 style='color:{ACCENT_GREEN}; text-align: right; margin: 0;'>Crop Price Forecast System</h3>",
            unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# --- WELCOME PAGE ---
def show_welcome_screen():
    st.header("🌾 Smart Farming, Double Profit!")
    st.markdown(f"""
        <div style='padding: 10px; border-radius: 8px; background-color: {HIGHLIGHT_COLOR}; border-left: 5px solid {ACCENT_GREEN}; margin-bottom: 20px;'>
            <h4 style='color: #333; margin: 0;'>“Those who decide on time, succeed on time.”</h4>
            <p style='margin: 0; font-size: 0.8em; color: #666;'>— This tool helps farmers make informed decisions.</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    img_col, text_col = st.columns([1.5, 3])
    with img_col:
        st.markdown('<div class="stacked-images-container">', unsafe_allow_html=True)
        try:
            st.image("crop_page1.jpg", use_container_width=True, caption="Plan based on data insights.")
        except:
            st.image("https://placehold.co/400x200/8bc34a/ffffff?text=Farming+Strategy", use_container_width=True)
        try:
            st.image("crop1.jpeg", use_container_width=True, caption="Maximize your profits.")
        except:
            st.image("https://placehold.co/400x200/4CAF50/ffffff?text=Data+Overview", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with text_col:
        st.markdown("""
        ## 💰 Accurate Market Price Information
        Before sowing or selling your crop, forecast the expected market price and plan profitably.

        #### Why Predict?
        - **Right Timing:** Identify the **best month** to sell your produce.  
        - **Market Strategy:** Understand **price trends** affecting your region.  
        - **Financial Planning:** Plan your **income and investment** confidently.

        All prices are shown in: **₹ per Quintal (100 kg)**.
        """)

        if st.button("▶️ Start Prediction Dashboard", type="primary", use_container_width=True):
            st.session_state.page = 'dashboard'
            st.rerun()


# --- DASHBOARD PAGE ---
def show_prediction_dashboard():
    draw_navbar()
    st.header("💡 Market Intelligence Dashboard")
    st.markdown("---")

    dash_col_img, dash_col_inputs = st.columns([1.5, 3])

    with dash_col_img:
        st.markdown("### Focus on Inputs")
        st.markdown('<div class="stacked-images-container">', unsafe_allow_html=True)
        try:
            st.image("crop2.jpg", use_container_width=True, caption="Select your specific parameters.")
        except:
            st.image("https://placehold.co/400x200/9ccc65/ffffff?text=Input+Parameters", use_container_width=True)
        try:
            st.image("crop_page2.jpg", use_container_width=True, caption="Data-driven analysis.")
        except:
            st.image("https://placehold.co/400x200/1E90FF/ffffff?text=Data+Driven+Insights", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("***Select your crop, market, and month for the most accurate forecast.***")

    with dash_col_inputs:
        st.markdown("### 🎯 Set Forecast Parameters")

        with st.container(border=True):
            st.markdown("##### 📍 Crop and Market Details")
            input_cols_1 = st.columns(3)
            with input_cols_1[0]:
                selected_district = st.selectbox("Market District:", DISTRICT_OPTIONS)
            with input_cols_1[1]:
                selected_commodity = st.selectbox("Crop Commodity:", COMMODITY_OPTIONS)
            with input_cols_1[2]:
                st.markdown("##### ⭐ Quality Grade")
                grade_encoded = st.radio("Grade:", [1, 2, 3], index=2, horizontal=True, help="1=Low, 3=High")

            st.divider()
            st.markdown("##### 📅 Select Selling Time")
            st.caption("Select the year and month for which you want the forecast.")
            input_cols_2 = st.columns(2)
            with input_cols_2[0]:
                selected_year = st.slider("Prediction Year:", 2024, 2030, 2025, step=1)
            with input_cols_2[1]:
                selected_month = st.slider("Specific Forecast Month:", 1, 12, 1, step=1)

        st.markdown("")
        predict_button = st.button("🚀 Generate Price Forecast", type="primary", use_container_width=True)

    if predict_button:
        if selected_district == 'Select District...' or selected_commodity == 'Select Commodity...':
            st.error("⚠️ Please select both Market District and Crop Commodity to proceed.")
            st.stop()

        if not rf_model:
            st.error("Model not loaded. Cannot run prediction.")
            st.stop()

        with st.spinner(f'Calculating 12-month forecast for {selected_commodity} in {selected_district}...'):
            input_data = pd.Series(0, index=ALL_COLUMNS)
            input_data['Year'], input_data['Month'], input_data['Day'], input_data['Grade_Encoded'] = selected_year, selected_month, 1, grade_encoded
            district_col_name = f'District_{selected_district}'
            commodity_col_name = f'Commodity_{selected_commodity}'
            if district_col_name in ALL_COLUMNS: input_data[district_col_name] = 1
            if commodity_col_name in ALL_COLUMNS: input_data[commodity_col_name] = 1
            predicted_price_specific = rf_model.predict(pd.DataFrame([input_data]))[0]
            forecast_df = get_monthly_forecast(selected_district, selected_commodity, selected_year, grade_encoded)
            comparison_df = get_comparison_data(selected_commodity, selected_year, grade_encoded, selected_district, raw_districts, forecast_df)

        st.session_state.results = {
            'price': predicted_price_specific,
            'forecast_df': forecast_df,
            'comparison_df': comparison_df,
            'district': selected_district,
            'commodity': selected_commodity,
            'year': selected_year,
            'month': selected_month,
            'grade': grade_encoded
        }
        st.session_state.page = 'results'
        st.rerun()


# --- RESULTS PAGE ---
def show_results_screen():
    draw_navbar()
    results = st.session_state.results

    if not results or 'price' not in results:
        st.warning("No forecast data found. Returning to Dashboard.")
        st.session_state.page = 'dashboard'
        st.rerun()
        return

    st.header(f"✅ Price Forecast for {results['commodity']}")
    st.markdown("---")

    st.markdown(
        f"""
        <div style='background-color: #F0F9FF; padding: 20px; border-radius: 12px; border: 2px solid {ACCENT_BLUE}; text-align: center; margin-bottom: 25px;'>
            <h2 style='color: {ACCENT_BLUE}; margin: 0;'>“Knowledge is Profit — Your Results Are Ready!”</h2>
            <p style='color: #333; margin: 5px 0 0 0; font-size: 0.9em;'>Below is your forecasted price for <b>{results['commodity']}</b> in <b>{results['district']}</b> for month <b>{results['month']}</b>.</p>
        </div>
        """, unsafe_allow_html=True
    )

    st.balloons()

    price_cols = st.columns([1, 3])

    with price_cols[0]:
        st.markdown(
            f"""
            <div style='background-color: {HIGHLIGHT_COLOR}; padding: 25px; border-radius: 10px; border-left: 8px solid {ACCENT_GREEN}; text-align: center; box-shadow: 2px 2px 10px rgba(0,0,0,0.1);'>
                <p style='font-size: 1.1em; color: #555;'>Expected Price for Month {results['month']}</p>
                <h1 style='color:{ACCENT_GREEN}; font-size: 3.5em; margin: 0;'>₹{results['price']:,.0f}</h1>
                <p style='font-size: 1.1em; color: #333; margin-top: 5px; font-weight: bold;'>per Quintal (100 kg)</p>
            </div>
            """, unsafe_allow_html=True
        )

        st.markdown("---")

        st.markdown(f"""
            <div style='padding: 15px; border-radius: 8px; border: 1px dashed {ACCENT_BLUE};'>
                <p style='font-weight: bold; color: {ACCENT_BLUE}; margin: 0;'>Key Details:</p>
                <p>📅 <b>Prediction Year:</b> {results['year']}</p>
                <p>⭐ <b>Quality Grade:</b> Grade {results['grade']}</p>
                <p style='font-size: 0.9em; color: #666;'>Higher grade crops generally achieve better prices.</p>
            </div>
            """, unsafe_allow_html=True
        )

    with price_cols[1]:
        st.subheader("📈 12-Month Trend (Chart Removed)")
        st.info("Trend visualization is currently disabled. The system still computes detailed monthly forecasts internally.")

    st.markdown("---")

    st.subheader("📍 District Price Comparison (Chart Removed)")
    st.info("Comparison chart is disabled. Forecast results are still computed for multiple districts for analysis.")

    st.markdown("---")

    st.markdown(
        f"""
        <div style='background-color: #FFFBEA; padding: 25px; border-radius: 12px; border: 2px solid {ACCENT_ORANGE}; text-align: center;'>
            <h3 style='color: {ACCENT_ORANGE}; margin: 0 0 10px 0;'>“Use the right timing and market for best profit.” 💰</h3>
            <p style='color: #444; margin: 0; font-size: 1.1em; font-weight: bold;'>
                Use these predictions to plan the right time and place to sell your crop.
            </p>
        </div>
        """, unsafe_allow_html=True
    )

    st.markdown("#### Future-Proof Your Agriculture 🚀")
    try:
        st.image("crop_last.jpg", use_container_width=True, caption="Informed decisions lead to higher profit.")
    except:
        st.image("https://placehold.co/800x400/9ccc65/ffffff?text=Informed+Decisions", use_container_width=True)


# --- MAIN APP RUNNER ---
try:
    st.markdown(
        f"<img src='top.jpeg' class='header-image' alt='App Header Image'>",
        unsafe_allow_html=True
    )
except:
    pass

if st.session_state.page == 'welcome':
    show_welcome_screen()
elif st.session_state.page == 'dashboard':
    show_prediction_dashboard()
elif st.session_state.page == 'results':
    show_results_screen()
else:
    show_welcome_screen()
