import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from models.xgboost_model import train_xgboost_model
from features.build_features import load_and_merge_data

# Style seaborn
sns.set_style("whitegrid")

st.set_page_config(page_title="Prévision des ventes (XGBoost)", layout="wide")
st.title("🔮 Prévision des ventes avec XGBoost + Alertes")

try:
    df_full = load_and_merge_data()
    products = sorted(df_full["product_id"].unique())
except Exception as e:
    st.error(f"Erreur de chargement des données : {e}")
    st.stop()

selected_product = st.selectbox("Sélectionner un produit", products)
forecast_horizon = st.slider("Horizon de prévision (jours)", 1, 90, 30)
stock_threshold = st.number_input("Seuil d'alerte de stock", 0, 1000, 10, 5)

try:
    model, forecast, hist, metrics = train_xgboost_model(selected_product, forecast_horizon)

    current_stock = df_full[df_full["product_id"] == selected_product]["stock"].iloc[-1]
    st.metric("📦 Stock actuel", f"{int(current_stock)} unités")

    if current_stock <= stock_threshold:
        st.error(f"⚠️ Stock critique ! ({int(current_stock)} ≤ {stock_threshold})")
    else:
        st.success(f"✅ Stock suffisant (> {stock_threshold})")

    # Projection du stock
    forecast["yhat"] = pd.to_numeric(forecast["yhat"], errors='coerce').fillna(0)
    cum_sales = forecast["yhat"].astype(float).cumsum()
    projected_stock = current_stock - cum_sales
    min_stock = projected_stock.min()

    if min_stock <= stock_threshold:
        st.warning(f"⚠️ Stock projeté pourrait atteindre **{int(min_stock)}** (≤ seuil)")
    else:
        st.success(f"✅ Stock projeté toujours > seuil (min: {int(min_stock)})")

    # --- Afficher les métriques ---
    st.subheader("📈 Performance du modèle (scikit-learn)")
    col1, col2 = st.columns(2)
    col1.metric("MAE (14 derniers jours)", metrics["mae"])
    col2.metric("RMSE", metrics["rmse"])

    # --- Visualisation avec matplotlib + seaborn ---
    st.subheader("📊 Ventes : Historique vs Prévision (matplotlib + seaborn)")

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=hist, x='date', y='y', label='Historique', marker='o', ax=ax, linewidth=2)
    sns.lineplot(data=forecast, x='date', y='yhat', label='Prévision', linestyle='--', marker='x', ax=ax, linewidth=2)

    # Ligne de seuil de stock (optionnel, mais on peut ajouter une annotation)
    ax.set_title(f"Prévision des ventes – Produit {selected_product}", fontsize=14)
    ax.set_ylabel("Nombre de ventes", fontsize=12)
    ax.set_xlabel("Date", fontsize=12)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    st.pyplot(fig)

    # Tableau
    st.subheader("📋 Prévisions détaillées")
    display_df = forecast[["date", "yhat"]].copy()
    display_df.columns = ["Date", "Ventes prévues"]
    display_df["Ventes prévues"] = display_df["Ventes prévues"].astype(int)
    st.dataframe(display_df, use_container_width=True)

except Exception as e:
    st.error("❌ Erreur lors de la prévision")
    st.code(str(e))
