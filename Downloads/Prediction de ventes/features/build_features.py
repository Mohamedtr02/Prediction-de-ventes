# features/build_features.py (corrigé)
import pandas as pd

def load_and_merge_data():
    sales = pd.read_csv("data/sales_data.csv", parse_dates=["date"])
    holidays = pd.read_csv("data/holidays.csv", parse_dates=["date"])
    weather = pd.read_csv("data/weather.csv", parse_dates=["date"])

    df = sales.merge(weather, on="date", how="left")
    df["is_holiday"] = df["date"].isin(holidays["date"]).astype(int)
    df["promotion"] = (df["sales"] > df.groupby("product_id")["sales"].transform(lambda x: x.rolling(7, min_periods=1).mean() * 1.3)).astype(int)

    # CORRIGÉ : pas de inplace=True sur une copie
    df["temp_avg"] = df["temp_avg"].fillna(df["temp_avg"].median())
    df["rain_mm"] = df["rain_mm"].fillna(0)

    return df