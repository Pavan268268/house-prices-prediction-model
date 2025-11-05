# 🏠 House Price Prediction (Kaggle Dataset)

An end-to-end Machine Learning project that predicts house prices using the **Kaggle House Prices dataset**.

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn (RandomForest)
- Streamlit (Frontend)

## Folder Structure
```
house_price_prediction/
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── sample_submission.csv
│   └── data_description.txt
├── model/
│   └── model.pkl
├── app.py
├── train_model.py
├── requirements.txt
└── README.md
```

## How to run locally
1. (Optional) Create a virtualenv: `python -m venv venv` and activate it.
2. Install dependencies: `pip install -r requirements.txt`
3. (Optional) Re-train model: `python train_model.py` (this overwrites model/model.pkl)
4. Run Streamlit: `streamlit run app.py`

App will open at `http://localhost:8501` by default.

## Deploying to Streamlit Cloud
1. Push this repository to GitHub.
2. Go to https://streamlit.io/cloud and connect your GitHub account.
3. Create a new app and select `app.py` as the entry point.

