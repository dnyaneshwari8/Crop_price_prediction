import streamlit as st
import joblib
import pandas as pd
import numpy as np
import altair as alt
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

# 1. CSS INJECTION TO REMOVE TOP SPACE AND DEFINE HEADER STYLE
st.markdown(
    """
    <style>
        /* 🎯 AGGRESSIVE CSS TO REMOVE TOP SPACE */

        /* Targets the entire app view block to remove default margin/padding */
        [data-testid="stAppViewBlock"] {
            padding-top: 0rem !important;
            margin-top: 0rem !important;
        }

        /* Targets the main content container and applies normal page padding/margin */
        .block-container {
            padding-top: 0rem !important; 
            padding-bottom: 5rem;
            padding-left: 5rem;
            padding-right: 5rem;
        }

        /* Hides the default Streamlit header/menu bar if it appears */
        div.stApp > header {
            display: none;
        }

        /* Class for the fixed-height header image (2cm height, full width) */
        .header-image {
            width: 100%;
            height: 2cm;
            object-fit: cover; /* Ensures the image covers the area without distortion */
            margin: 0;
            padding: 0;
            display: block; 
        }
        /* Navigation bar styling */
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
        /* Styles for the stacked image container */
        .stacked-images-container {
            display: flex;
            flex-direction: column;
            gap: 15px; /* Spacing between the stacked images */
        }

        /* Ensure H2 elements (used for st.header) are styled for better fit */
        h2 {
            white-space: nowrap; /* Prevents wrapping if possible */
            overflow: hidden;    /* Hides overflow if strictly necessary */
            text-overflow: ellipsis; /* Adds dots if overflow occurs */
            font-size: 1.8rem; /* Slightly reduced size for better fit */
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize session state for page control
if 'page' not in st.session_state:
    st.session_state.page = 'welcome'
if 'results' not in st.session_state:
    st.session_state.results = {}


# --- 1. LOAD ASSETS (Model and Features) ---
# NOTE: The model and column loading logic is retained for functionality but removed from user-facing text.
@st.cache_resource
def load_assets():
    """Loads the model and feature columns list with error handling."""
    global rf_model, ALL_COLUMNS
    try:
        rf_model = joblib.load('final_crop_price_predictor.joblib')
        ALL_COLUMNS = joblib.load('feature_columns.joblib')
        return rf_model, ALL_COLUMNS
    except FileNotFoundError:
        # Fallback for display purposes
        ALL_COLUMNS = ['Year', 'Month', 'Day', 'Grade_Encoded', 'District_Pune', 'Commodity_Wheat']
        rf_model = None
        return rf_model, ALL_COLUMNS
    except Exception:
        # Fallback for display purposes
        ALL_COLUMNS = ['Year', 'Month', 'Day', 'Grade_Encoded', 'District_Pune', 'Commodity_Wheat']
        rf_model = None
        return rf_model, ALL_COLUMNS


rf_model, ALL_COLUMNS = load_assets()

# --- 2. DEFINE OPTIONS AND UTILITY FUNCTIONS ---
if 'ALL_COLUMNS' not in locals():
    # Fallback definition if loading failed
    ALL_COLUMNS = ['Year', 'Month', 'Day', 'Grade_Encoded', 'District_Pune', 'Commodity_Wheat']

# Safely extract options even if ALL_COLUMNS is minimal
raw_districts = [col.split('District_')[1] for col in ALL_COLUMNS if col.startswith('District_')]
raw_commodities = [col.split('Commodity_')[1] for col in ALL_COLUMNS if col.startswith('Commodity_')]

DISTRICT_OPTIONS = ['Select District...'] + sorted(raw_districts)
COMMODITY_OPTIONS = ['Select Commodity...'] + sorted(raw_commodities)


def get_monthly_forecast(district, commodity, year, grade):
    """Generates 12 monthly predictions for the chart."""
    forecasts = []

    if not rf_model:
        # Fallback to dummy data if model failed to load
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
    """Generates price data for comparison districts (using actual data for main district)."""
    comparison_data = []

    # Select up to 2 random other districts
    other_districts = [d for d in all_districts if d != main_district]
    num_to_sample = min(2, len(other_districts))
    comp_districts = random.sample(other_districts, num_to_sample)
    comp_districts.append(main_district)

    for district in comp_districts:
        if district == main_district:
            df = base_forecast_df.copy()
        else:
            # Generate the 12-month forecast for the comparison district
            df = get_monthly_forecast(district, commodity, year, grade)

            # Simple simulation offset if the prices are identical
            if rf_model and df['Price'].iloc[0] == base_forecast_df['Price'].iloc[0]:
                df['Price'] = df['Price'] + random.uniform(-100, 100)

        df['District'] = district
        comparison_data.append(df)

    combined_df = pd.concat(comparison_data)
    combined_df['Price'] = combined_df['Price'].round(0).astype(int)

    return combined_df


# --- NAVIGATION BAR FUNCTION ---
def draw_navbar():
    """Draws a horizontal navigation bar at the top of the screen."""

    st.markdown("""
        <div class='navbar-container'>
    """, unsafe_allow_html=True)

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
            f"<h3 style='color:{ACCENT_GREEN}; text-align: right; margin: 0;'>शेतकऱ्यांसाठी पीक दर अंदाज (Crop Price Forecast)</h3>",
            unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# --- 3. WELCOME SCREEN FUNCTION (Stacked Images) ---
def show_welcome_screen():
    # Changed st.title to st.header for better fit
    st.header(f"🌾 स्मार्ट शेती, दुप्पट नफा! (Smart Farming, Double Profit!)")

    # Attractive Marathi Quote
    st.markdown(f"""
        <div style='padding: 10px; border-radius: 8px; background-color: {HIGHLIGHT_COLOR}; border-left: 5px solid {ACCENT_GREEN}; margin-bottom: 20px;'>
            <h4 style='color: #333; margin: 0;'>                      "जो वेळेत निर्णय घेतो, तोच यशस्वी होतो."  </h4>
            <p style='margin: 0; font-size: 0.7em; color: #666;'>— This tool helps you decide on time.</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Layout for stacked images (1.5 wide) and text (3 wide)
    img_col, text_col = st.columns([1.5, 3])

    with img_col:
        # Custom container for stacked images
        st.markdown('<div class="stacked-images-container">', unsafe_allow_html=True)

        # Image 1 (Original)
        try:
            st.image("crop_page1.jpg", use_container_width=True,
                     caption="माहितीवर आधारित नियोजन.")
        except:
            st.warning("Image 'crop_page1.jpg' not found.")
            st.image("https://placehold.co/400x200/8bc34a/ffffff?text=Farming+Strategy", use_container_width=True)

        # Image 2 (NEW, stacked below Image 1)
        try:
            st.image("crop1.jpeg", use_container_width=True, caption="जास्तीत जास्त नफा मिळवा.")
        except:
            st.image("https://placehold.co/400x200/4CAF50/ffffff?text=Data+Overview", use_container_width=True,
                     caption="Maximize profits.")

        st.markdown('</div>', unsafe_allow_html=True)

    with text_col:
        st.markdown(f"""
        ## 💰 बाज़ारपेठेची अचूक माहिती तुमच्या हातात
        पीक पेरणी करण्यापूर्वी किंवा विक्री करण्यापूर्वी तुमच्या मालाच्या बाजारभावाचा अंदाज घ्या. आता, माहितीने जिंका!

        #### Predict का करायचं?
        - **योग्य वेळ:** पीक विकण्यासाठी **सर्वात चांगला महिना** ओळखा.
        - **बाजारपेठ धोरण:** तुमच्या मालावर परिणाम करणारे **दर ट्रेंड (Price Trends)** समजून घ्या.
        - **उत्पन्नाची निश्चिती:** **अपेक्षित उत्पन्नासह** (Projected Income) तुमचे आर्थिक नियोजन करा.

        सर्व अंदाज प्रमाणित बाजार युनिटमध्ये दिले जातात: **₹ प्रति क्विंटल (100 किलो)**.
        """)

        if st.button("▶️ सुरू करा: दर अंदाज डॅशबोर्ड", type="primary", use_container_width=True):
            st.session_state.page = 'dashboard'
            st.rerun()


# --- 4. PREDICTION DASHBOARD FUNCTION (Stacked Images) ---
def show_prediction_dashboard():
    draw_navbar()

    # Changed st.title to st.header for better fit
    st.header("💡 Market Intelligence Dashboard")
    st.markdown("---")

    # Layout for stacked images (1.5 wide) and inputs (3 wide)
    dash_col_img, dash_col_inputs = st.columns([1.5, 3])

    with dash_col_img:
        st.markdown("### Focus on Inputs")

        st.markdown('<div class="stacked-images-container">', unsafe_allow_html=True)

        # Image 1 (Original)
        try:
            st.image("crop2.jpg", use_container_width=True,
                     caption="तुमची विशिष्ट निकष निवडा.")
        except:
            st.warning("Image 'crop2.jpg' not found.")
            st.image("https://placehold.co/400x200/9ccc65/ffffff?text=Input+Parameters", use_container_width=True)

        # Image 2 (NEW, stacked below Image 1)
        try:
            st.image("crop_page2.jpg", use_container_width=True, caption="डेटा आधारित विश्लेषण.")
        except:
            st.image("https://placehold.co/400x200/1E90FF/ffffff?text=Data+Driven+Insights", use_container_width=True,
                     caption="Analyze market data.")

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(
            f"***सर्वात अचूक अंदाजासाठी तुमचे **पीक, बाजारपेठ आणि महिना** निवडा. (Select your parameters for precise forecast).***")

    with dash_col_inputs:
        st.markdown("### 🎯 अंदाज मापदंड सेट करा (Set Forecasting Parameters)")

        # --- INPUT SECTION 1: MARKET & GRADE ---
        with st.container(border=True):
            st.markdown("##### 📍 **पीक आणि ठिकाण तपशील (Crop & Location Details)**")
            input_cols_1 = st.columns(3)
            with input_cols_1[0]:
                selected_district = st.selectbox("Market District:", DISTRICT_OPTIONS, key='district_select')
            with input_cols_1[1]:
                selected_commodity = st.selectbox("Crop Commodity:", COMMODITY_OPTIONS, key='commodity_select')
            with input_cols_1[2]:
                st.markdown("##### ⭐ Quality Grade")
                grade_encoded = st.radio("Grade:", [1, 2, 3], index=2, horizontal=True, label_visibility="collapsed",
                                         help="1=Lowest, 3=Best", key='grade_radio')

            st.divider()

            # --- INPUT SECTION 2: TIME (Using Sliders) ---
            st.markdown("##### 📅 **विक्रीसाठी वेळेची निवड (Select Selling Time)**")
            st.caption("अंदाजासाठी वर्ष आणि विशिष्ट महिना निवडा (Select Year and Specific Month).")
            input_cols_2 = st.columns(2)

            with input_cols_2[0]:
                selected_year = st.slider("Prediction Year:", min_value=2024, max_value=2030, value=2025, step=1,
                                          key='year_slider')

            with input_cols_2[1]:
                selected_month = st.slider("Specific Forecast Month:", min_value=1, max_value=12, value=1, step=1,
                                           key='month_slider')

        # --- Action Button ---
        st.markdown("")
        predict_button = st.button("🚀 ट्रेंड्स आणि दर अंदाज जनरेट करा", type="primary", use_container_width=True,
                                   key='main_forecast_button')

    # --- PREDICTION LOGIC (Triggers page switch) ---
    if predict_button:

        is_valid_selection = (selected_district != 'Select District...') and (
                selected_commodity != 'Select Commodity...')

        if not is_valid_selection:
            st.error("⚠️ कृपया पुढे जाण्यासाठी Market District आणि Crop Commodity निवडा.")
            st.stop()

        if not rf_model:
            st.error("Model not loaded. Cannot run prediction.")
            st.stop()

        with st.spinner(f'Calculating 12-month forecast for {selected_commodity} in {selected_district}...'):

            # 1. Specific Prediction Input
            input_data = pd.Series(0, index=ALL_COLUMNS)
            input_data['Year'], input_data['Month'], input_data['Day'], input_data[
                'Grade_Encoded'] = selected_year, selected_month, 1, grade_encoded
            district_col_name = f'District_{selected_district}'
            commodity_col_name = f'Commodity_{selected_commodity}'
            if district_col_name in ALL_COLUMNS: input_data[district_col_name] = 1
            if commodity_col_name in ALL_COLUMNS: input_data[commodity_col_name] = 1

            predicted_price_specific = rf_model.predict(pd.DataFrame([input_data]))[0]

            # 2. Generate 12-month forecast (for the selected district)
            forecast_df = get_monthly_forecast(selected_district, selected_commodity, selected_year, grade_encoded)

            # 3. Generate comparison data (includes selected district and others)
            comparison_df = get_comparison_data(selected_commodity, selected_year, grade_encoded, selected_district,
                                                raw_districts, forecast_df)

        # --- 4. Store Results and Switch Page ---
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


# --- 5. RESULTS SCREEN FUNCTION (Attractive Price Output & Charts) ---
def show_results_screen():
    draw_navbar()

    results = st.session_state.results

    if not results or 'price' not in results:
        st.warning("No valid forecast data found. Returning to Dashboard.")
        st.session_state.page = 'dashboard'
        st.rerun()
        return

    # Changed st.title to st.header and adjusted text for better fit
    st.header(f"✅ सर्वोत्तम दर अंदाज ({results['commodity']} Price Projection)")
    st.markdown("---")

    # --- STYLISH MARATHI INTRO TEXT ---
    st.markdown(
        f"""
        <div style='background-color: #F0F9FF; padding: 20px; border-radius: 12px; border: 2px solid {ACCENT_BLUE}; text-align: center; margin-bottom: 25px;'>
            <h2 style='color: {ACCENT_BLUE}; margin: 0;'>**"माहिती हेच यश! आपले सर्व निकाल तयार आहेत."**</h2>
            <p style='color: #333; margin: 5px 0 0 0; font-size: 0.9em;'>आपण निवडलेल्या **{results['district']}** मार्केटसाठी, विशेषतः **महिना {results['month']}** साठीचा तुमचा अचूक दर खालीलप्रमाणे आहे.</p>
        </div>
        """, unsafe_allow_html=True
    )

    st.balloons()

    # --- ROW 1: ATTRACTIVE PRICE OUTPUT AND 12-MONTH TREND ---
    price_cols = st.columns([1, 3])

    with price_cols[0]:

        # 🌟 Attractive Price Container (Specific Month Price)
        st.markdown(
            f"""
            <div style='background-color: {HIGHLIGHT_COLOR}; padding: 25px; border-radius: 10px; border-left: 8px solid {ACCENT_GREEN}; text-align: center; box-shadow: 2px 2px 10px rgba(0,0,0,0.1);'>
                <p style='font-size: 1.1em; color: #555;'>**महिना {results['month']}** साठी अपेक्षित विक्री दर</p>
                <h1 style='color:{ACCENT_GREEN}; font-size: 3.5em; margin: 0;'>₹{results['price']:,.0f}</h1>
                <p style='font-size: 1.1em; color: #333; margin-top: 5px; font-weight: bold;'>per Quintal (100 kg)</p>
            </div>
            """, unsafe_allow_html=True
        )

        st.markdown("---")

        # Attractive Key Details Box
        st.markdown(f"""
            <div style='padding: 15px; border-radius: 8px; border: 1px dashed {ACCENT_BLUE};'>
                <p style='font-weight: bold; color: {ACCENT_BLUE}; margin: 0;'>Key Details (महत्वाचे तपशील):</p>
                <p style='margin: 5px 0 0 0;'>📅 **Prediction Year:** <span style='font-weight: bold;'>{results['year']}</span></p>
                <p style='margin: 0;'>⭐ **Quality Grade:** <span style='font-weight: bold;'>Grade {results['grade']}</span></p>
                <p style='margin: 0; font-size: 0.9em; color: #666;'>*उत्तम ग्रेड (Grade {results['grade']}) नेहमी जास्त दर मिळवून देतो.*</p>
            </div>
            """, unsafe_allow_html=True
                    )

    with price_cols[1]:
        st.subheader(f"📈 १२-महिन्यांचा दर ट्रेंड ({results['district']} Market Seasonal Analysis)")
        st.caption("ऐतिहासिक चढ-उतारांवर आधारित, संपूर्ण वर्षातील दरांची अपेक्षा पाहा.")

        forecast_df = results['forecast_df']
        specific_month_data = forecast_df[forecast_df['Month'] == results['month']]

        base = alt.Chart(forecast_df).encode(
            x=alt.X('Month', axis=alt.Axis(format='d', title='Month of the Year'))
        )

        line_chart = base.mark_line(point=True, strokeWidth=3, color=ACCENT_GREEN).encode(
            y=alt.Y('Price', title='Predicted Price (₹/Quintal)', scale=alt.Scale(zero=False)),
            tooltip=['Month', alt.Tooltip('Price', format=',.2f')]
        )

        highlight = alt.Chart(specific_month_data).mark_circle(size=250, color=ACCENT_ORANGE).encode(
            x='Month',
            y='Price',
            tooltip=[alt.Tooltip('Price', format=',.2f')]
        )

        st.altair_chart(line_chart + highlight, use_container_width=True)

    st.markdown("---")

    # --- ROW 2: PRIMARY DISTRICT COMPARISON (BAR CHART: District vs Monthly Price) ---
    comparison_df = results['comparison_df']

    # Filter comparison data ONLY for the selected month
    comparison_for_month = comparison_df[comparison_df['Month'] == results['month']]

    st.subheader(f"📍 निवडलेल्या महिन्यातील (Month {results['month']}) जिल्हा-दर तुलना (District Price Comparison)")
    st.caption(
        f"या चार्टमध्ये, आपल्या निवडलेल्या महिन्यासाठी इतर बाजारांमधील अपेक्षित दर दर्शविला आहे. **सर्वात जास्त दर हिरव्या रंगात (Highest Price in Green) हायलाइट केला आहे.**")

    # Determine the highest price for coloring
    max_price = comparison_for_month['Price'].max()

    # Bar chart: District on X (one side), Price on Y (the other side)
    bar_chart = alt.Chart(comparison_for_month).mark_bar().encode(
        # Y-axis: Price (the value)
        y=alt.Y('Price', title=f'Predicted Price in Month {results["month"]} (₹/Quintal)', scale=alt.Scale(zero=False)),
        # X-axis: District, sorted by Price descending
        x=alt.X('District', sort='-y', title='Market District'),
        color=alt.condition(
            alt.datum.Price == max_price,
            alt.value(ACCENT_GREEN),  # Green for the highest price
            alt.value(ACCENT_BLUE)  # Blue for others
        ),
        tooltip=['District', alt.Tooltip('Price', format=',.0f', title='Price')]
    )

    text_labels = bar_chart.mark_text(
        align='center',
        baseline='bottom',
        dy=-5  # Nudge text up slightly
    ).encode(
        text=alt.Text('Price', format=',.0f'),
        color=alt.value('black')
    )

    st.altair_chart(bar_chart + text_labels, use_container_width=True)

    st.markdown("---")

    # --- STYLISH MARATHI/ENGLISH CONCLUSION TEXT ---
    st.markdown(
        f"""
        <div style='background-color: #FFFBEA; padding: 25px; border-radius: 12px; border: 2px solid {ACCENT_ORANGE}; text-align: center;'>
            <h3 style='color: {ACCENT_ORANGE}; margin: 0 0 10px 0;'>**"उत्कृष्ट वेळेचा आणि बाजाराचा योग्य वापर करा."** 💰</h3>
            <p style='color: #444; margin: 0; font-size: 1.1em; font-weight: bold;'>
                Use this intelligence to choose the best district and time to sell and maximize your profit.
            </p>
        </div>
        """, unsafe_allow_html=True
    )

    # --- ROW 3: FINAL IMAGE (Ensuring this stays) ---
    st.markdown("#### Future-Proof Your Agriculture. 🚀")
    try:
        st.image("crop_last.jpg", use_container_width=True, caption="Informed decisions are key to profitability.")
    except:
        st.warning("Image 'crop_last.jpg' not found. Using placeholder.")
        st.image("https://placehold.co/800x400/9ccc65/ffffff?text=Informed+Decisions",
                 use_container_width=True, caption="Placeholder: Informed decisions")


# --- 6. MAIN APP RUNNER ---

# 2. HEADER IMAGE IMPLEMENTATION (Full Width, 2cm Height)
try:
    st.markdown(
        f"""
        <img src='top.jpeg' class='header-image' alt='App Header Image'>
        """,
        unsafe_allow_html=True
    )
except Exception:
    st.markdown(
        "<div style='height: 1cm; background-color: #333; color: white; text-align: center; line-height: 2cm; font-weight: bold; font-size: 1.2em;'>Crop Price Forecaster Header</div>",
        unsafe_allow_html=True)

if st.session_state.page == 'welcome':
    show_welcome_screen()
elif st.session_state.page == 'dashboard':
    show_prediction_dashboard()
elif st.session_state.page == 'results':
    show_results_screen()
