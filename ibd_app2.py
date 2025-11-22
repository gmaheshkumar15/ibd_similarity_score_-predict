import streamlit as st
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# -----------------------------
# Load trained models safely
# -----------------------------
try:
    log_model = joblib.load("logistic_final.pkl")
    scaler = joblib.load("scaler_final.pkl")
except FileNotFoundError as e:
    st.error(f"Error: Missing model file - {e}")
    st.stop()

# -----------------------------
# Merge Map
# -----------------------------
MERGE_MAP = {
    "WHEAT(CHAPATI,ROTI,NAAN,DALIA,RAWA/SOOJI,SEVIYAAN": [
        "Wheat (Chapati, Roti, Naan, Dalia, Rawa/Sooji, Seviyaan)"
    ],
    "WHEAT FREE CEREALS": [
        "Rice (Rice, Rice Flour, Dosa, Poha, Idli, Murmura)",
        "Maize (Chapati, Chhali, Bhutta, Corn Cob)",
        "Oats (Oats Meal, Rolled)",
        "Barley",
        "Ragi, Bajra, Jowar",
        "Amaranth (Chulai, Rajgira, Seel)",
        "Others (e.g., Besan Roti, etc.)"
    ],
    "FRUITS": [
        "Red, Deep Orange, Yellow Fruits (Mango, Papaya, Peach, etc.)",
        "Citrus Fruits (Lemon, Orange, Grapefruit, etc.)",
        "Berries and Grapes (Raspberry, Cherry, Strawberry, Amla, Grapes)",
        "Others (Apple, Banana, Cheeku, Kiwi, etc.)"
    ],
    "OTHER VEGETABLES": [
        "Green Leafy", "Green (Tinda, Tori, Kaddu, etc.)", "Cruciferous",
        "Bulbs (Garlic, Onion)", "Others (Brinjal, Carrot, Radish, Cucumber, Turnip, Ginger, etc.)"
    ],
    "STARCHY(POTATO,SWEET PATATO,ARBI ETC)": ["Starchy (Potato, Sweet Potato, Arbi, etc.)"],
    "PULSES AND LEGUMES": [
        "Pulses (Lentils, Arhar, Tur, Green Grams, Black Grams, etc.)",
        "Legumes (Green Peas, Chickpea, Rajma, Rongi, etc.)", "Soybeans"
    ],
    "PREDOMINANT SATURATED FATS": ["Desi Ghee, Butter, Malai", "Coconut Oil, Palm Oil"],
    "PREDOMINANT UNSATURATED FATS": [
        "Rice Bran Oil", "Sunflower Oil", "Safflower Oil",
        "Linseed Oil", "Canola Oil", "Mustard Oil", "Olive Oil"
    ],
    "TRANS FATS": ["Dalda", "Vanaspati"],
    "NUTS AND OILSEEDS": [
        "Almonds", "Walnuts", "Groundnuts", "Cashewnuts",
        "Flax Seeds", "Sunflower Seeds"
    ],
    "EGGS,FISH AND POULTRY": ["Eggs", "Chicken/Turkey", "Fish and Seafood"],
    "RED MEAT": ["Red Meat (Mutton, Pork, Beef)"],
    "MILK ": ["Milk"],
    "LOW LACTOSE DAIRY": [
        "Homemade Curd", "Homemade Buttermilk/Lassi/Chaach", "Cottage Cheese (Paneer)"
    ],
    "SWEETEND BEVERAGES": [
        "Carbonated Drinks/Soda", "Bottled/Tetra-Pack/Powdered Juices/Fruit Drinks/Concentrates",
        "Energy Drinks", "Bottled/Packed Dairy Drinks"
    ],
    "ULTRA PROCESSED FOODS": [
        "Packed Breads/Buns/Kulcha/Pav", "Cakes/Muffin/Pastry/Cake Mix",
        "Breakfast Cereal/Breakfast Bars", "Ice Cream", "Puddings and Pies",
        "Jellies and Jam", "Chocolates", "Candies/Gummies",
        "Dressings/Mayonnaise/Spreads/Margarines", "Packed Soups",
        "Instant Noodles", "Packed Meat/Fish/Vegetables", "Processed Cheese",
        "Pre-Prepared Ready to Eat Meals", "Condensed Milk/Milkmaid"
    ],
    "READT TO EAT PACKAGED SNACKS": [
        "Salty (Chips, Kurkure, Cookies, Biscuits)",
        "Sweet (Biscuits, Rusks, Cookies)"
    ],
    "SAVORY SNACKS": [
        "Samosa, Kachori, Pakora, Mathri, etc.",
        "Manchurian, Burger, Hot Dogs, etc.",
        "Bhel Puri, Muri, Pani Puri, Puchka, Bhalla, Dhokla",
        "Pizza, Pasta, Noodles, Patty, Momos, etc."
    ],
    "PROCESSED FOODS": [
        "Frozen Food", "Ketchup/Puree", "Pickles",
        "Chutney", "Canned Vegetables in Vinegar",
        "Canned Fruits in Syrup", "Canned Fish",
        "Smoked Meat/Fish/Sausages", "Soy/Almond/Oat Milk, Tofu"
    ],
    "INDIAN SWEET MEATS": ["Khoya Burfi, Rabri, Ladoo, Kalakand, Gulab Jamun, etc.", "Khoya"],
    "FOOD SUPPLEMENTS": [
        "Calcium Supplements", "Vitamin D", "Zinc", "Iron", "Protein Supplements"
    ],
    "ERGOGENIC SUPPLEMENTS": ["Fat Burners/Body Building Gym Supplements"]
}

# -----------------------------
# Feature names
# -----------------------------
try:
    feature_names = list(log_model.feature_names_in_)
except AttributeError:
    try:
        feature_names = scaler.feature_names_in_.tolist()
    except AttributeError:
        feature_names = list(MERGE_MAP.keys())

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="IBD Risk Prediction", layout="wide")

# -----------------------------
# Custom CSS (thick, dark line)
# -----------------------------
st.markdown(
    """
    <style>
    .stApp { background-color: #ADD8E6; }
    hr { border: none; border-top: 3px solid #000000; margin: 0px 0; }
    .logo-left, .logo-right { width: 120px; display:block; margin:auto; }
    .institute-name { text-align:center; font-weight:bold; font-size:16px; margin-top:5px; }
    .large-score { font-size: 32px !important; font-weight: bold; color: #8B0000; text-align: center; margin-top: 5px; margin-bottom: 5px; }
    .pred-label { font-size: 18px; font-weight: 600; text-align: center; margin-bottom: 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Logos and Title
# -----------------------------
col_logo_left, col_title, col_logo_right = st.columns([1, 5, 1])
with col_logo_left:
    st.markdown(
        '<img src="https://brandlogovector.com/wp-content/uploads/2022/04/IIT-Delhi-Icon-Logo.png" class="logo-left">',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="institute-name">Indian Institute of Technology Delhi</div>', unsafe_allow_html=True)
with col_title:
    st.markdown(
        "<h1 style='text-align:center; font-size:32px:line-height:2; color:black;'>DMCH-IITD Machine Learning Tool for Estimating the Diet Percentage Similarity with Respect to Diets Consumed by Inflammatory Bowel Disease Patients Prior to Diagnosis</h1>",
        unsafe_allow_html=True,
    )



with col_logo_right:
    st.markdown('<img src="https://raw.githubusercontent.com/gmaheshkumar15/ibd_similarity_score_-predictor3/main/dmch.jpeg" class="logo-right">', unsafe_allow_html=True)
    st.markdown('<div class="institute-name">Dayanand Medical College and Hospital Ludhiana</div>', unsafe_allow_html=True)



st.markdown("<hr>", unsafe_allow_html=True)

st.markdown(
    """
    <p style='text-align:left; font-size:20px; color:black; line-height:1.75;'>
    This tool is developed by DMCH Ludhiana and IIT Delhi. It uses machine learning (ML) models to estimate the similarity of a diet with those consumed by patients prior to an Inflammatory Bowel Disease (IBD) diagnosis. 
    The ML model was trained based on data from a dietary survey conducted by DMCH Ludhiana among IBD patients and Controls without IBD. 
    IBD patients were asked to report their dietary habits prior to diagnosis, and Controls were asked to report current food habits.</p>
    """,
    unsafe_allow_html=True,
)

st.markdown("<hr>", unsafe_allow_html=True)

# -----------------------------
# Helper Functions
# -----------------------------
def clean_feature_name(name):
    return name.replace("_", " ").title()

feature_value_limits = {
    "WHEAT(CHAPATI,ROTI,NAAN,DALIA,RAWA/SOOJI,SEVIYAAN": list(range(0, 6)),
    "WHEAT FREE CEREALS": list(range(0, 36)),
    "FRUITS": list(range(0, 21)),
    "OTHER VEGETABLES": list(range(0, 26)),
    "STARCHY(POTATO,SWEET PATATO,ARBI ETC)": list(range(0, 6)),
    "PULSES AND LEGUMES": list(range(0, 16)),
    "PREDOMINANT SATURATED FATS": list(range(0, 11)),
    "PREDOMINANT UNSATURATED FATS": list(range(0, 11)),
    "TRANS FATS": list(range(0, 6)),
    "NUTS AND OILSEEDS": list(range(0, 6)),
    "EGGS,FISH AND POULTRY": list(range(0, 16)),
    "RED MEAT": list(range(0, 6)),
    "MILK ": list(range(0, 6)),
    "LOW LACTOSE DAIRY": list(range(0, 16)),
    "SWEETEND BEVERAGES": list(range(0, 21)),
    "ULTRA PROCESSED FOODS": list(range(0, 76)),
    "READT TO EAT PACKAGED SNACKS": list(range(0, 11)),
    "SAVORY SNACKS": list(range(0, 21)),
    "PROCESSED FOODS": list(range(0, 46)),
    "INDIAN SWEET MEATS": list(range(0, 11)),
    "FOOD SUPPLEMENTS": list(range(0, 26)),
    "ERGOGENIC SUPPLEMENTS": list(range(0, 6)),
}

col_input, col_output = st.columns([5, 1])

# -----------------------------
# Input Form
# -----------------------------
features = {}
with col_input:
    st.header(
        "In the below fields, provide information about your dietary habits. Select the level of consumption for each food item (higher values indicate higher consumption, and vice versa)."
    )
    n = len(feature_names)
    half = n // 2 if n > 1 else 1

    for i in range(half):
        c1, c2 = st.columns(2, gap="medium")

        with c1:
            fname1 = feature_names[i]
            options1 = feature_value_limits.get(fname1.upper(), list(range(38)))
            features[fname1] = st.selectbox(label=clean_feature_name(fname1), options=options1, index=0, key=fname1)
            merged_list_1 = MERGE_MAP.get(fname1.upper(), [])
            if merged_list_1:
                st.markdown(f"<b>Examples:</b> {', '.join(merged_list_1)}", unsafe_allow_html=True)
            st.markdown("<hr>", unsafe_allow_html=True)

        with c2:
            idx2 = i + half
            if idx2 < n:
                fname2 = feature_names[idx2]
                options2 = feature_value_limits.get(fname2.upper(), list(range(38)))
                features[fname2] = st.selectbox(label=clean_feature_name(fname2), options=options2, index=0, key=fname2)
                merged_list_2 = MERGE_MAP.get(fname2.upper(), [])
                if merged_list_2:
                    st.markdown(f"<b>Examples:</b> {', '.join(merged_list_2)}", unsafe_allow_html=True)
                st.markdown("<hr>", unsafe_allow_html=True)

input_df = pd.DataFrame([features], columns=feature_names)

# -----------------------------
# Prediction Output
# -----------------------------
with col_output:
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Results")
    predict_clicked = st.button("Predict")

    if predict_clicked:
        try:
            scaled_input = scaler.transform(input_df)
            logistic_score = log_model.predict_proba(scaled_input)[0][1] * 100
    

            st.markdown(
                "<p style='font-size:30px; font-weight:bold; margin-bottom:5px;'>Similarity Score (0–100):</p>",
                unsafe_allow_html=True,
            )
            pcol1, pcol2, pcol3 = st.columns(3)

            with pcol1:
                st.markdown(f"<div class='large-score'>{logistic_score:.0f}</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction Error: {e}")
