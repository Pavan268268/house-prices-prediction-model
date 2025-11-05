# 🏠 House Prices Prediction Model (Kaggle Dataset)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)]()
[![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red)]()
[![Scikit-learn](https://img.shields.io/badge/ML-Library-green)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()

An end-to-end **Machine Learning project** that predicts **house prices** using the famous [Kaggle House Prices – Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) dataset.

This project demonstrates data preprocessing, model training, evaluation, and deployment using **Streamlit** as the interactive frontend.

---

## 🚀 Features
✅ Clean and interpretable ML pipeline (Scikit-learn)  
✅ Random Forest Regressor trained on real Kaggle data  
✅ Streamlit web app for live price prediction  
✅ Easy local setup and Streamlit Cloud deployment  
✅ Ready for portfolio and resume display  

---

## 🧠 Tech Stack
- **Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, Joblib  
- **Frontend:** Streamlit  
- **Model:** RandomForestRegressor  
- **Dataset:** Kaggle’s *House Prices – Advanced Regression Techniques*

---

## 📂 Folder Structure
house_price_prediction/
├── data/
│ ├── train.csv
│ ├── test.csv
│ ├── sample_submission.csv
│ └── data_description.txt
├── model/
│ └── model.pkl
├── app.py
├── train_model.py
├── requirements.txt
└── README.md

yaml
Copy code

---

## ⚙️ How to Run Locally

### 1️⃣ Clone the repository
bash
git clone https://github.com/Pavan268268/house-prices-prediction-model.git
cd house-prices-prediction-model
2️⃣ Install dependencies
bash
Copy code
pip install -r requirements.txt
3️⃣ (Optional) Retrain the model
bash
Copy code
python train_model.py
4️⃣ Run the Streamlit app
bash
Copy code
streamlit run app.py
The app will open automatically at http://localhost:8501

📊 Model Performance
Metric	Value
Mean Absolute Error (MAE)	18299.65
R² Score	0.896

Model used: RandomForestRegressor (100 estimators)

🖥️ Streamlit App Preview
Once running, you’ll see:

Sidebar sliders & dropdowns for house features

Real-time predicted house price

Optional data preview & feature exploration

Example:

yaml
Copy code
Overall Quality: 7
Living Area: 1800 sqft
Garage: 2 Cars
Predicted Price: ~$230,000
🌐 Deploying to Streamlit Cloud
Go to Streamlit Cloud

Sign in with your GitHub account

Click New app

Select this repo → app.py

Click Deploy

Get your public shareable app link 🎯

🧩 Future Improvements
Add XGBoost / LightGBM models for comparison

Feature importance & SHAP explanations

Interactive correlation heatmap (EDA page)

Model versioning and performance logging

💡 About This Project

Demonstrate full-cycle ML development (EDA → Model → App)

Build a portfolio-ready ML project for hiring assessments

Showcase real-world data handling and deployment skills

📬 Connect with Me
👤 Pavan Mantena

⭐ If you like this project, please give it a star on GitHub!

yaml
Copy code

---
- Add a **“Live Demo” section** with a sample Streamlit Cloud link placeholder (so you can replace it later),  
- or include a **small image preview badge** (for example a screenshot of the app UI once you deploy it)?
