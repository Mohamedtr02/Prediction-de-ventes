from prophet import Prophet
import pandas as pd
from .build_features import load_and_merge_data

def train_prophet_model(product_id="P001", forecast_days=30):
    df = load_and_merge_data()
    df = df[df["product_id"] == product_id].copy()

    # Format Prophet : ds, y
    prophet_df = df[["date", "sales", "temp_avg", "rain_mm", "is_holiday", "promotion"]].rename(
        columns={"date": "ds", "sales": "y"}
    )

    # Initialisation
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative"
    )

    # Ajout des regresseurs
    model.add_regressor("temp_avg")
    model.add_regressor("rain_mm")
    model.add_regressor("is_holiday")
    model.add_regressor("promotion")

    model.fit(prophet_df)

    # Prévision multi-pas
    future = model.make_future_dataframe(periods=forecast_days)
    # Fusionner avec les features externes pour la période future (à estimer ou simuler)
    # Ici, on duplique les dernières valeurs pour simplifier (à améliorer en production)
    last_features = prophet_df[["temp_avg", "rain_mm", "is_holiday", "promotion"]].iloc[-1]
    for col in ["temp_avg", "rain_mm", "is_holiday", "promotion"]:
        future[col] = last_features[col]

    forecast = model.predict(future)

    return model, forecast, prophet_df