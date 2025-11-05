import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="🏠 House Price Prediction", layout="centered")
st.title("🏠 House Price Prediction (Kaggle)")
st.markdown("Interactive demo using a trained Random Forest model. Edit inputs in the sidebar and click Predict.")

# Load pipeline
MODEL_PATH = os.path.join("model", "model.pkl")
@st.cache_resource
def load_model(path=MODEL_PATH):
    return joblib.load(path)

model = load_model()

st.sidebar.header("Input House Features")
OverallQual = st.sidebar.slider("Overall Material and Finish (OverallQual)", 1, 10, 6)
GrLivArea = st.sidebar.number_input("Above grade (ground) living area (sqft) - GrLivArea", min_value=300, max_value=6000, value=1500, step=10)
GarageCars = st.sidebar.slider("Garage capacity (cars)", 0, 4, 2)
TotalBsmtSF = st.sidebar.number_input("Total basement area (sqft)", min_value=0, max_value=4000, value=800, step=10)
FirstFlrSF = st.sidebar.number_input("First Floor area (sqft) - 1stFlrSF", min_value=300, max_value=4000, value=1000, step=10)
FullBath = st.sidebar.slider("Full bathrooms above grade", 0, 4, 2)
YearBuilt = st.sidebar.slider("Year built", 1870, 2020, 2000)
LotArea = st.sidebar.number_input("Lot area (sqft)", min_value=130, max_value=200000, value=9000, step=10)
Neighborhood = st.sidebar.selectbox("Neighborhood", options=sorted(list(model.named_steps['preprocessor'].transformers_[1][1].named_steps['ohe'].categories_[0])))

input_dict = {
    "OverallQual": OverallQual,
    "GrLivArea": GrLivArea,
    "GarageCars": GarageCars,
    "TotalBsmtSF": TotalBsmtSF,
    "1stFlrSF": FirstFlrSF,
    "FullBath": FullBath,
    "YearBuilt": YearBuilt,
    "LotArea": LotArea,
    "Neighborhood": Neighborhood
}

input_df = pd.DataFrame([input_dict])

if st.button("Predict Price"):
    pred = model.predict(input_df)[0]
    st.success(f"Estimated Sale Price: ${pred:,.0f}")
    st.write("---")
    st.write("Input features:") 
    st.json(input_dict)

# Show quick EDA
if st.checkbox("Show sample of training data (first 5 rows)"):
    df = pd.read_csv(os.path.join('data','train.csv'))
    st.dataframe(df.head())

if st.checkbox("Show model performance on test set"):
    # Basic reported metrics (precomputed in training script are not imported here), so we compute quickly on a subset.
    df = pd.read_csv(os.path.join('data','train.csv'))
    features = ["OverallQual","GrLivArea","GarageCars","TotalBsmtSF","1stFlrSF","FullBath","YearBuilt","LotArea","Neighborhood","SalePrice"]
    st.write(df[features].dropna().head())
    st.info("Model was trained with a RandomForestRegressor. For full evaluation, run train_model.py locally.")
