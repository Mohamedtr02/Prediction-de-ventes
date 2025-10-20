import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Créer le dossier data
os.makedirs("data", exist_ok=True)

# 1. Plage de dates : 2 ans
start_date = datetime(2023, 1, 1)
end_date = datetime(2024, 12, 31)
dates = pd.date_range(start=start_date, end=end_date, freq='D')
date_df = pd.DataFrame({"date": dates})

# 2. Produits
products = [
    {"product_id": "P001", "category": "Électronique", "base_sales": 80, "stock_init": 2000},
    {"product_id": "P002", "category": "Électronique", "base_sales": 60, "stock_init": 1500},
    {"product_id": "P003", "category": "Maison", "base_sales": 100, "stock_init": 2500},
]

# 3. Jours fériés en France (2023–2024)
french_holidays = {
    "2023-01-01", "2023-04-10", "2023-05-01", "2023-05-08", "2023-05-18",
    "2023-05-29", "2023-07-14", "2023-08-15", "2023-11-01", "2023-11-11", "2023-12-25",
    "2024-01-01", "2024-04-01", "2024-05-01", "2024-05-08", "2024-05-09",
    "2024-05-20", "2024-07-14", "2024-08-15", "2024-11-01", "2024-11-11", "2024-12-25"
}
holidays_df = pd.DataFrame({
    "date": pd.to_datetime(list(french_holidays)),
    "holiday_name": "Jour férié"
})

# 4. Génération météo
np.random.seed(42)
weather_data = []
for date in dates:
    # Température moyenne selon la saison
    doy = date.dayofyear
    temp = 10 + 12 * np.cos(2 * np.pi * (doy - 172) / 365.25)  # pic en juillet
    temp += np.random.normal(0, 3)  # bruit
    rain = max(0, np.random.gamma(1.5, 2)) if np.random.rand() < 0.4 else 0.0
    weather_data.append({"date": date, "temp_avg": round(temp, 1), "rain_mm": round(rain, 1)})

weather_df = pd.DataFrame(weather_data)

# 5. Génération des ventes + stock
all_sales = []

for prod in products:
    df = date_df.copy()
    df["product_id"] = prod["product_id"]
    df["category"] = prod["category"]
    
    # Saisonnalité (pic en décembre)
    df["month"] = df["date"].dt.month
    seasonal_factor = np.where(df["month"] == 12, 1.8,
                      np.where(df["month"].isin([6, 7, 8]), 1.2,
                      np.where(df["month"].isin([11]), 1.4, 1.0)))
    
    # Effet météo (ex: moins de ventes s'il pleut beaucoup)
    df = df.merge(weather_df, on="date", how="left")
    rain_effect = np.where(df["rain_mm"] > 10, 0.85, 1.0)
    
    # Jours fériés
    df["is_holiday"] = df["date"].isin(holidays_df["date"])
    holiday_effect = np.where(df["is_holiday"], 1.5, 1.0)
    
    # Promotions aléatoires (5% des jours)
    promo_days = np.random.rand(len(df)) < 0.05
    df["promotion"] = promo_days.astype(int)
    promo_effect = np.where(promo_days, 2.0, 1.0)
    
    # Calcul des ventes
    base = prod["base_sales"]
    sales = base * seasonal_factor * rain_effect * holiday_effect * promo_effect
    sales = np.round(sales + np.random.normal(0, 5, len(sales))).astype(int)
    sales = np.clip(sales, 0, None)  # pas de ventes négatives
    
    df["sales"] = sales
    
    # Stock : on part du stock initial et on décrémente
    stock = [prod["stock_init"]]
    for s in sales[:-1]:
        stock.append(max(0, stock[-1] - s))
    df["stock"] = stock
    
    all_sales.append(df[["date", "product_id", "category", "sales", "stock"]])

# 6. Sauvegarde
sales_full = pd.concat(all_sales, ignore_index=True)
sales_full.to_csv("data/sales_data.csv", index=False)
weather_df.to_csv("data/weather.csv", index=False)
holidays_df.to_csv("data/holidays.csv", index=False)

print("✅ Données synthétiques générées dans le dossier 'data/'")
print("- sales_data.csv")
print("- weather.csv")
print("- holidays.csv")