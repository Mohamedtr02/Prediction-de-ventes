from prophet.plot import plot_plotly, plot_components_plotly
import plotly.graph_objs as go

def plot_forecast_and_components(model, forecast, historical):
    fig1 = plot_plotly(model, forecast)
    fig2 = plot_components_plotly(model, forecast)

    # Ajout du stock critique (si disponible)
    if "stock" in historical.columns:
        stock_alert = historical[historical["stock"] < 10]  # seuil = 10
        if not stock_alert.empty:
            fig1.add_trace(go.Scatter(
                x=stock_alert["ds"],
                y=stock_alert["y"],
                mode="markers",
                marker=dict(color="red", size=10),
                name="⚠️ Stock critique"
            ))

    return fig1, fig2