import pandas as pd
import plotly.express as px

def plot_features(model, features):
    importances = model.feature_importances_

    df = pd.DataFrame({
        "feature": features,
        "importance": importances
    }).sort_values(by="importance", ascending=False)

    fig = px.bar(
        df,
        x="importance",
        y="feature",
        orientation='h',
        title="<b>Quais fatores mais influenciam o tempo gasto?</b>",
        labels={
            "importance":"Peso no Modelo",
            "feature":""
        },
        color='importance',
        color_continuous_scale='Viridis',
        template='plotly_white'
    )
    fig.update_layout(
        showlegend=False,
        title_font_size=20,
        margin=dict(l=20, r=20, t=50, b=20),
        height=400
    )
    return fig

def plot_real_vs_predito(y_test, previsoes):
    print("Gerando gráfico real x previsto")
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, previsoes, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    plt.xlabel("Valor Real (Minutos)")
    plt.ylabel("Previsão do Modelo (Minutos)")
    plt.title("Comparação: Real vs. Previsão")
    plt.show()