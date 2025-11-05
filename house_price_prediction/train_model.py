import pandas as pd
import joblib, os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

BASE_DIR = os.path.dirname(__file__)
df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'train.csv'))

features = ["OverallQual","GrLivArea","GarageCars","TotalBsmtSF","1stFlrSF","FullBath","YearBuilt","LotArea","Neighborhood"]
target = 'SalePrice'
data = df[features + [target]].copy()
data['Neighborhood'] = data['Neighborhood'].fillna('None')

X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numeric_feats = ["OverallQual","GrLivArea","GarageCars","TotalBsmtSF","1stFlrSF","FullBath","YearBuilt","LotArea"]
categorical_feats = ["Neighborhood"]

numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='None')), ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))])

preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_feats), ('cat', categorical_transformer, categorical_feats)], remainder='drop')

pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1))])

pipeline.fit(X_train, y_train)
preds = pipeline.predict(X_test)
print('MAE:', mean_absolute_error(y_test, preds))
print('R2:', r2_score(y_test, preds))

model_dir = os.path.join(BASE_DIR, 'model')
os.makedirs(model_dir, exist_ok=True)
joblib.dump(pipeline, os.path.join(model_dir, 'model.pkl'))
print('Saved model to', os.path.join(model_dir, 'model.pkl'))
