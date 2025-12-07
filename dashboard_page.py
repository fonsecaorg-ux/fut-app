import streamlit as st
import pandas as pd
import plotly.express as px

def render_dashboard():
    st.title("📊 Dashboard de Performance")
    st.markdown("---")

    # DADOS SIMULADOS (Baseados no seu dia hoje)
    dados_hoje = [
        {"Data": "06/12", "Jogo": "Bilbao x Atl. Madrid", "Liga": "La Liga", "Mercado": "Escanteios/Cartões", "Resultado": "Green", "Lucro": 24.09},
        {"Data": "06/12", "Jogo": "Verona x Atalanta", "Liga": "Série A", "Mercado": "Cartões", "Resultado": "Green (Cashout)", "Lucro": 8.27},
        {"Data": "06/12", "Jogo": "Nantes x Lens", "Liga": "Ligue 1", "Mercado": "Cartões Individuais", "Resultado": "Red", "Lucro": -5.00},
        {"Data": "06/12", "Jogo": "Wolfsburg x Union Berlin", "Liga": "Bundesliga", "Mercado": "Escanteios/Cartões", "Resultado": "Red", "Lucro": -5.00},
        {"Data": "06/12", "Jogo": "Sassuolo x Fiorentina", "Liga": "Série A", "Mercado": "Escanteios Time", "Resultado": "Red", "Lucro": -5.00},
        {"Data": "06/12", "Jogo": "Betis x Barcelona", "Liga": "La Liga", "Mercado": "Escanteios/Cartões", "Resultado": "Green", "Lucro": 10.00},
    ]
    
    df = pd.DataFrame(dados_hoje)

    # KPIs
    total_apostas = len(df)
    total_greens = len(df[df["Resultado"].str.contains("Green")])
    win_rate = (total_greens / total_apostas) * 100
    lucro_total = df["Lucro"].sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Tips", total_apostas)
    col2.metric("Assertividade", f"{win_rate:.1f}%")
    col3.metric("Lucro Hoje", f"R$ {lucro_total:.2f}", delta=f"{lucro_total:.2f}")

    st.markdown("---")

    # GRÁFICOS
    st.subheader("Performance por Liga")
    df_liga = df.groupby(["Liga", "Resultado"]).size().reset_index(name="Contagem")
    fig_liga = px.bar(df_liga, x="Liga", y="Contagem", color="Resultado", 
                      color_discrete_map={"Green": "#00CC96", "Red": "#EF553B", "Green (Cashout)": "#636EFA"})
    st.plotly_chart(fig_liga, use_container_width=True)

    st.subheader("Histórico")
    def color_result(val):
        color = '#d4edda' if 'Green' in val else '#f8d7da' if 'Red' in val else ''
        return f'background-color: {color}'
    st.dataframe(df.style.applymap(color_result, subset=['Resultado']), use_container_width=True)
