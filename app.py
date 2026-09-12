import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="EV Purchase Predictor", page_icon="🔋", layout="centered")

MODEL_PATH = "F:/Electric_Vehicle_Purchases/final_stacking_model.joblib"

WILL_BUY_GIF = "https://media.giphy.com/media/9OB40VsTjX9NmUMhmf/giphy.gif"
WONT_BUY_GIF = "https://media.giphy.com/media/SreAp5WMZHfhK/giphy.gif"


st.title("🔋 Electric Vehicle Purchase Predictor")
st.caption("Fill in your details to see if you're likely to go electric.")
st.divider()


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

with st.form("ev_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=16, max_value=100, value=30, step=1)
    with col2:
        annual_income = st.number_input("Annual income (USD)", min_value=0, value=50000, step=1000)

    col3, col4 = st.columns(2)
    with col3:
        daily_km = st.number_input("Daily commute (km)", min_value=0.0, value=15.0, step=1.0)
    with col4:
        no_cars = st.number_input("Number of cars owned", min_value=0, max_value=10, value=1, step=1)

    st.subheader("Charging access")
    col5, col6 = st.columns(2)
    with col5:
        charging_station_home = st.number_input("Charging stations near home", min_value=0, value=1, step=1)
    with col6:
        charging_station_work = st.number_input("Charging stations near work", min_value=0, value=1, step=1)

    col7, col8 = st.columns(2)
    with col7:
        home_charging = st.selectbox("Home charging possible?", ["Yes", "No"])
    with col8:
        subsidy = st.selectbox("Subsidy available?", ["Yes", "No"])

    st.subheader("Preferences")
    col9, col10 = st.columns(2)
    with col9:
        enviorment_level = st.slider("Environmental concern level (1-10)", 1, 10, 5)
    with col10:
        range_level = st.selectbox("Range anxiety level", ["Low", "Medium", "High"])

    predict_clicked = st.form_submit_button("Predict", use_container_width=True)

if predict_clicked:
    input_df = pd.DataFrame(
        [
            {
                "Age": age,
                "Annual_Income_USD": annual_income,
                "Daily_Commute_km": daily_km,
                "Number_of_Cars_Owned": no_cars,
                "Charging_Stations_Near_Home": charging_station_home,
                "Charging_Stations_Near_Work": charging_station_work,
                "Environmental_Concern_Level": enviorment_level,
                "Home_Charging_Possible": 1 if home_charging == "Yes" else 0,
                "Subsidy_Available": 1 if subsidy == "Yes" else 0,
                "Range_Anxiety_Level": range_level,
            }
        ]
    )

    prediction = model.predict(input_df)[0]

    st.divider()
    if prediction == 1:
        st.success("✅ Likely to buy an EV!")
        st.image(WILL_BUY_GIF, use_container_width=True)
    else:
        st.error("❌ Not likely to buy an EV.")
        st.image(WONT_BUY_GIF, use_container_width=True)