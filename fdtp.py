# app.py
import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

# Try importing XGBoost, fallback to GradientBoosting if unavailable
try:
    from xgboost import XGBRegressor
    xgb_available = True
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor
    xgb_available = False

# ------------------------------
# Load dataset
# ------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Food_Delivery_Times.csv")
    return df

df = load_data()

st.title("🍔 Food Delivery Time Predictor")
st.write("This app predicts delivery time (in minutes) using Random Forest and XGBoost.")

# ------------------------------
# Features & target
# ------------------------------
FEATURES = [
    'Distance_km',
    'Weather',
    'Traffic_Level',
    'Time_of_Day',
    'Vehicle_Type',
    'Preparation_Time_min',
    'Courier_Experience_yrs'
]
TARGET = 'Delivery_Time_min'

X = df[FEATURES]
y = df[TARGET]

numeric_features = ['Distance_km', 'Preparation_Time_min', 'Courier_Experience_yrs']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_features = ['Weather', 'Traffic_Level', 'Time_of_Day', 'Vehicle_Type']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))

])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# ------------------------------
# Train models (only once, cache them)
# ------------------------------
@st.cache_resource
def train_models():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Random Forest
    rf = RandomForestRegressor(n_estimators=120, max_depth=12, random_state=42, n_jobs=-1)
    rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', rf)])
    rf_pipeline.fit(X_train, y_train)

    # XGBoost (or GradientBoosting if not installed)
    if xgb_available:
        xgb = XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.08,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=1
        )
    else:
        xgb = GradientBoostingRegressor(n_estimators=200, max_depth=6, learning_rate=0.08, random_state=42)

    xgb_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', xgb)])
    xgb_pipeline.fit(X_train, y_train)

    # Evaluate
    def eval_model(model):
        preds = model.predict(X_test)
        rmse = mean_squared_error(y_test, preds, squared=False)
        r2 = r2_score(y_test, preds)
        return rmse, r2

    rf_rmse, rf_r2 = eval_model(rf_pipeline)
    xgb_rmse, xgb_r2 = eval_model(xgb_pipeline)

    return rf_pipeline, xgb_pipeline, (rf_rmse, rf_r2), (xgb_rmse, xgb_r2)

rf_model, xgb_model, rf_metrics, xgb_metrics = train_models()

# ------------------------------
# Sidebar inputs
# ------------------------------
st.sidebar.header("🔧 Input Features")

Distance_km = st.sidebar.number_input("Distance (km)", min_value=0.0, value=3.0, step=0.1)
Preparation_Time_min = st.sidebar.number_input("Preparation Time (min)", min_value=0, value=10, step=1)
Courier_Experience_yrs = st.sidebar.number_input("Courier Experience (yrs)", min_value=0.0, value=1.0, step=0.1)

Weather = st.sidebar.selectbox("Weather", ["Sunny", "Rainy", "Cloudy", "Stormy", "Windy"])
Traffic_Level = st.sidebar.selectbox("Traffic Level", ["Low", "Medium", "High"])
Time_of_Day = st.sidebar.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"])
Vehicle_Type = st.sidebar.selectbox("Vehicle Type", ["Bike", "Bike-2", "Car", "Bike-3"])

model_choice = st.sidebar.radio("Choose Model", ["Random Forest", "XGBoost"])

# ------------------------------
# Prediction
# ------------------------------
input_df = pd.DataFrame([{
    'Distance_km': Distance_km,
    'Preparation_Time_min': Preparation_Time_min,
    'Courier_Experience_yrs': Courier_Experience_yrs,
    'Weather': Weather,
    'Traffic_Level': Traffic_Level,
    'Time_of_Day': Time_of_Day,
    'Vehicle_Type': Vehicle_Type
}])

st.subheader("Input Preview")
st.write(input_df)

if model_choice == "Random Forest":
    model = rf_model
    rmse, r2 = rf_metrics
else:
    model = xgb_model
    rmse, r2 = xgb_metrics

pred = model.predict(input_df)[0]
st.subheader("Predicted Delivery Time")
st.metric(label="Delivery Time (min)", value=f"{pred:.1f} min")

st.write("---")
st.subheader("📊 Model Test Performance")
st.write(f"**Random Forest** — RMSE: {rf_metrics[0]:.2f}, R²: {rf_metrics[1]:.3f}")
st.write(f"**XGBoost** — RMSE: {xgb_metrics[0]:.2f}, R²: {xgb_metrics[1]:.3f}")

