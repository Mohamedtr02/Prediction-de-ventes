import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from features.build_features import load_and_merge_data

def create_lagged_features(df, target_col="sales", lags=[1, 7, 14]):
    df = df.copy()
    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df.groupby("product_id")[target_col].shift(lag)
    return df

def add_time_features(df):
    df = df.copy()
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_year"] = df["date"].dt.dayofyear
    df["quarter"] = df["date"].dt.quarter
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df

def train_xgboost_model(product_id="P001", forecast_days=30):
    df = load_and_merge_data()
    df = df[df["product_id"] == product_id].copy()
    df = df.sort_values("date").reset_index(drop=True)

    if len(df) < 30:
        raise ValueError(f"Pas assez de données pour {product_id} (min. 30 jours)")

    df = add_time_features(df)
    df = create_lagged_features(df, lags=[1, 7, 14])

    feature_cols = [
        "temp_avg", "rain_mm", "is_holiday", "promotion",
        "month", "day_of_week", "day_of_year", "quarter", "is_weekend",
        "sales_lag_1", "sales_lag_7", "sales_lag_14"
    ]

    df = df.dropna(subset=feature_cols + ["sales"]).reset_index(drop=True)
    if len(df) < 20:
        raise ValueError(f"Données insuffisantes après nettoyage pour {product_id}")

    X = df[feature_cols]
    y = df["sales"]

    # Entraînement
    model = XGBRegressor(
        n_estimators=400,
        learning_rate=0.06,
        max_depth=5,
        subsample=0.9,
        random_state=42
    )
    model.fit(X, y)

    # Évaluation avec scikit-learn (sur les 14 derniers jours)
    eval_start = max(0, len(X) - 14)
    if eval_start < len(X) - 1:
        X_eval = X.iloc[eval_start:]
        y_eval = y.iloc[eval_start:]
        y_pred_eval = model.predict(X_eval)
        mae = mean_absolute_error(y_eval, y_pred_eval)
        rmse = np.sqrt(mean_squared_error(y_eval, y_pred_eval))
    else:
        mae, rmse = 0, 0  # pas assez de données pour évaluer

    #Prévision multi-pas
    last_date = df["date"].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)
    future_df = pd.DataFrame({"date": future_dates})
    future_df["product_id"] = product_id

    last_row = df.iloc[-1]
    for col in ["temp_avg", "rain_mm", "is_holiday", "promotion"]:
        future_df[col] = last_row[col]

    future_df = add_time_features(future_df)

    sales_history = df["sales"].tolist()
    predictions = []

    for i in range(forecast_days):
        lag1 = sales_history[-1]
        lag7 = sales_history[-7] if len(sales_history) >= 7 else np.mean(sales_history[-min(7, len(sales_history)):])
        lag14 = sales_history[-14] if len(sales_history) >= 14 else np.mean(sales_history[-min(14, len(sales_history)):])

        future_df.loc[i, "sales_lag_1"] = float(lag1)
        future_df.loc[i, "sales_lag_7"] = float(lag7)
        future_df.loc[i, "sales_lag_14"] = float(lag14)

        X_future = future_df[feature_cols].iloc[[i]]
        pred_val = model.predict(X_future)[0]
        pred = max(0, int(round(float(pred_val))))
        predictions.append(pred)
        sales_history.append(pred)

    forecast = pd.DataFrame({"date": future_dates, "yhat": predictions})
    hist = df[["date", "sales"]].rename(columns={"sales": "y"})

    # Renvoyer aussi les métriques
    metrics = {"mae": round(mae, 2), "rmse": round(rmse, 2)}
    return model, forecast, hist, metrics
