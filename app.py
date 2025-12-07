import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import poisson
import json
import hmac
import os
from datetime import datetime

# ==============================================================================
# 0. CONFIGURAÇÃO E LOGIN
# ==============================================================================
st.set_page_config(page_title="FutPrevisão Pro", layout="wide", page_icon="⚽")

def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    def password_entered():
        if "passwords" in st.secrets:
            user = st.session_state["username"]
            password = st.session_state["password"]
            
            if user in st.secrets["passwords"] and \
               hmac.compare_digest(password, st.secrets["passwords"][user]):
                st.session_state["password_correct"] = True
                del st.session_state["password"]
                del st.session_state["username"]
            else:
                st.session_state["password_correct"] = False
                st.error("😕 Usuário ou senha incorretos")
        else:
            st.error("Erro: Senhas não configuradas.")

    if st.session_state["password_correct"]: return True
    st.markdown("### 🔒 Acesso Restrito - FutPrevisão Pro")
    st.text_input("Usuário", key="username")
    st.text_input("Senha", type="password", key="password")
    st.button("Entrar", on_click=password_entered)
    return False

if not check_password(): st.stop()

# ==============================================================================
# 1. FUNÇÕES DO DASHBOARD (CARREGAR E SALVAR)
# ==============================================================================
DATA_FILE = "historico_apostas.json"

def carregar_historico():
    if not os.path.exists(DATA_FILE):
        # Cria um arquivo inicial com os dados de hoje como exemplo
        dados_iniciais = [
            {"Data": "06/12/2025", "Jogo": "Bilbao x Atl. Madrid", "Liga": "La Liga", "Mercado": "Escanteios/Cartões", "Resultado": "Green", "Lucro": 24.09},
            {"Data": "06/12/2025", "Jogo": "Verona x Atalanta", "Liga": "Série A", "Mercado": "Cartões", "Resultado": "Green (Cashout)", "Lucro": 8.27},
            {"Data": "06/12/2025", "Jogo": "Nantes x Lens", "Liga": "Ligue 1", "Mercado": "Cartões Individuais", "Resultado": "Red", "Lucro": -5.00},
            {"Data": "06/12/2025", "Jogo": "Wolfsburg x Union Berlin", "Liga": "Bundesliga", "Mercado": "Escanteios/Cartões", "Resultado": "Red", "Lucro": -5.00},
        ]
        with open(DATA_FILE, "w") as f:
            json.dump(dados_iniciais, f)
        return pd.DataFrame(dados_iniciais)
    else:
        with open(DATA_FILE, "r") as f:
            dados = json.load(f)
        return pd.DataFrame(dados)

def salvar_aposta(nova_aposta):
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            dados = json.load(f)
    else:
        dados = []
    
    dados.append(nova_aposta)
    with open(DATA_FILE, "w") as f:
        json.dump(dados, f)

def render_dashboard():
    st.title("📊 Dashboard de Performance")
    
    # --- FORMULÁRIO DE CADASTRO ---
    with st.expander("➕ Registrar Nova Aposta (Red/Green)", expanded=False):
        st.caption("Adicione o resultado do seu bilhete aqui:")
        c1, c2, c3 = st.columns(3)
        with c1:
            data_input = st.date_input("Data", datetime.now())
            jogo_input = st.text_input("Jogo (ex: Flamengo x Vasco)")
        with c2:
            liga_input = st.selectbox("Liga", ["Brasileirão", "Premier League", "La Liga", "Bundesliga", "Série A", "Ligue 1", "Outra"])
            mercado_input = st.selectbox("Mercado", ["Escanteios", "Cartões", "Gols", "Múltipla (Criar Aposta)", "Outro"])
        with c3:
            resultado_input = st.radio("Resultado", ["Green", "Green (Cashout)", "Red"], horizontal=True)
            valor_input = st.number_input("Lucro/Prejuízo (R$)", min_value=-10000.0, max_value=10000.0, step=0.50, help="Use negativo (-) para Red")

        if st.button("💾 Salvar no Histórico"):
            # Ajuste automático: Se for Red e o usuário colocou valor positivo, converte para negativo
            if resultado_input == "Red" and valor_input > 0:
                valor_input = valor_input * -1
            
            nova_entrada = {
                "Data": data_input.strftime("%d/%m/%Y"),
                "Jogo": jogo_input,
                "Liga": liga_input,
                "Mercado": mercado_input,
                "Resultado": resultado_input,
                "Lucro": valor_input
            }
            salvar_aposta(nova_entrada)
            st.success("Aposta registrada! Atualizando...")
            st.rerun()

    st.markdown("---")

    # --- CARREGAR DADOS ---
    df = carregar_historico()

    if df.empty:
        st.warning("Nenhum dado registrado ainda.")
        return

    # CÁLCULOS (KPIs)
    total_apostas = len(df)
    total_greens = len(df[df["Resultado"].str.contains("Green")])
    win_rate = (total_greens / total_apostas) * 100
    lucro_total = df["Lucro"].sum()

    # VISUALIZAÇÃO - TOPO
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Tips", total_apostas)
    col2.metric("Assertividade", f"{win_rate:.1f}%")
    col3.metric("Lucro Total", f"R$ {lucro_total:.2f}", delta=f"{lucro_total:.2f}")

    st.markdown("---")

    # GRÁFICOS
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Performance por Liga")
        # Gráfico de Barras Agrupado
        fig_liga = px.histogram(df, x="Liga", color="Resultado", 
                                color_discrete_map={"Green": "#00CC96", "Red": "#EF553B", "Green (Cashout)": "#636EFA"},
                                title="Contagem de Resultados por Liga")
        st.plotly_chart(fig_liga, use_container_width=True)
    
    with c2:
        st.subheader("Distribuição")
        fig_pizza = px.pie(df, names="Resultado", title="Greens vs Reds", 
                           color="Resultado",
                           color_discrete_map={"Green": "#00CC96", "Red": "#EF553B", "Green (Cashout)": "#636EFA"})
        st.plotly_chart(fig_pizza, use_container_width=True)

    # TABELA
    st.subheader("📜 Histórico de Entradas")
    
    # Ordenar por data (se possível) ou índice reverso para ver os mais recentes primeiro
    df_show = df.iloc[::-1] 
    
    def color_result(val):
        color = '#d4edda' if 'Green' in val else '#f8d7da' if 'Red' in val else ''
        return f'background-color: {color}'
    st.dataframe(df_show.style.applymap(color_result, subset=['Resultado']), use_container_width=True)

# ==============================================================================
# 2. MENU DE NAVEGAÇÃO
# ==============================================================================
st.sidebar.header("Navegação")
pagina = st.sidebar.radio("Ir para:", ["🏠 Previsões do Jogo", "📊 Dashboard de Performance"])

if pagina == "📊 Dashboard de Performance":
    render_dashboard()
    st.stop() 

# ==============================================================================
# 3. DADOS E PREVISÕES (CÓDIGO ORIGINAL)
# ==============================================================================
BACKUP_TEAMS = {
    "Arsenal": {"corners": 6.82, "cards": 1.00, "fouls": 10.45, "goals_f": 2.3, "goals_a": 0.8},
    "Man City": {"corners": 7.45, "cards": 1.50, "fouls": 9.20, "goals_f": 2.7, "goals_a": 0.8},
}

def safe_float(value):
    try: return float(str(value).replace(',', '.'))
    except: return 0.0

@st.cache_data(ttl=3600)
def load_data():
    try:
        df = pd.read_csv("dados_times.csv")
        teams_dict = {}
        for _, row in df.iterrows():
            teams_dict[row['Time']] = {
                'corners': safe_float(row['Escanteios']),
                'cards': safe_float(row['CartoesAmarelos']), 
                'fouls': safe_float(row['Faltas']),
                'goals_f': safe_float(row['GolsFeitos']),
                'goals_a': safe_float(row['GolsSofridos'])
            }
    except:
        teams_dict = BACKUP_TEAMS
    
    try:
        df_ref = pd.read_csv("arbitros.csv")
        referees = dict(zip(df_ref['Nome'], df_ref['Fator']))
    except:
        referees = {}
        
    referees[' Estilo: Rigoroso (+ Cartões)'] = 1.25
    referees[' Estilo: Normal (Padrão)'] = 1.00
    referees[' Estilo: Conservador (- Cartões)'] = 0.80
        
    return teams_dict, referees

teams_data, referees_data = load_data()

# BARRA LATERAL (CONTINUAÇÃO)
st.sidebar.markdown("---")
st.sidebar.title("FutPrevisão Pro v2.8") # Versão atualizada

def carregar_metadados():
    try:
        with open("metadados.json", "r", encoding='utf-8') as f:
            return json.load(f)
    except: return None

meta = carregar_metadados()
if meta:
    st.sidebar.caption("🤖 **Status do Robô:**")
    st.sidebar.text(f"{meta['ultima_verificacao']}")
    
    if 'log' in meta:
        st.sidebar.caption("📋 Relatório:")
        st.sidebar.info(meta['log'])
    
    if meta['times_alterados'] == 0 and 'log' not in meta:
        st.sidebar.info("✔ Base verificada e estável.")
        
    st.sidebar.caption(f"📡 Fontes: {meta.get('fontes', 'Adamchoi & FBref')}")
else:
    st.sidebar.warning("⚠ Aguardando robô...")

st.sidebar.markdown("---")
st.sidebar.header("Configuração da Partida")

team_list = sorted(list(teams_data.keys()))
home_team = st.sidebar.selectbox("Mandante", team_list, index=0)
away_team = st.sidebar.selectbox("Visitante", team_list, index=1)

st.sidebar.markdown("---")
st.sidebar.caption("🧠 **Contexto**")
context_options = {
    "⚪ Neutro (Meio de Tabela": 1.0,
    "🔥 Must Win (Z4)": 1.15,
    "🏆 Must Win (Título/Libertadores)": 1.15,
    "❄️ Desmobilizado (Rebaixado)": 0.85,
    "💪 Super Favorito": 1.25,
    "🚑 Crise": 0.80
}
ctx_h = st.sidebar.selectbox(f"Momento: {home_team}", list(context_options.keys()), index=0)
ctx_a = st.sidebar.selectbox(f"Momento: {away_team}", list(context_options.keys()), index=0)
f_h = context_options[ctx_h]
f_a = context_options[ctx_a]

st.sidebar.markdown("---")
referee_list = sorted(list(referees_data.keys()))
ref_name = st.sidebar.selectbox("Árbitro", referee_list)
ref_factor = referees_data[ref_name]
st.sidebar.metric("Rigor", ref_factor)

champions_mode = st.sidebar.checkbox("Modo Champions (-15%)", value=False)

# CÁLCULOS
def calculate_metrics(home, away, ref_factor, is_champions, fact_h, fact_a):
    h_data = teams_data[home]
    a_data = teams_data[away]
    
    # Escanteios
    corn_h = (h_data['corners'] * 1.10) * fact_h
    corn_a = (a_data['corners'] * 0.85) * fact_a
    if is_champions: 
        corn_h *= 0.85
        corn_a *= 0.85
    total_corners = corn_h + corn_a
        
    # Cartões
    tension_boost = 1.10 if fact_h > 1.0 or fact_a > 1.0 else 1.0
    tension = ((h_data['fouls'] + a_data['fouls']) / 24.0) * tension_boost
    tension = max(0.85, min(tension, 1.40))
    
    card_h = h_data['cards'] * tension * ref_factor
    card_a = a_data['cards'] * tension * ref_factor
    total_cards = card_h + card_a
    
    # Gols
    avg_l = 1.3
    exp_h = ((h_data['goals_f'] * fact_h)/avg_l) * (a_data['goals_a']/avg_l) * avg_l
    exp_a = ((a_data['goals_f'] * fact_a)/avg_l) * (h_data['goals_a']/avg_l) * avg_l
    
    return {'total_corners': total_corners, 'ind_corn_h': corn_h, 'ind_corn_a': corn_a,
            'total_cards': total_cards, 'ind_card_h': card_h, 'ind_card_a': card_a,
            'goals_h': exp_h, 'goals_a': exp_a, 'tension': tension}

def prob_over(exp, line): return poisson.sf(int(line), exp) * 100

# INTERFACE PRINCIPAL
st.title("⚽ FutPrevisão Pro")

tab_analise, tab_scanner = st.tabs(["📊 Análise do Jogo", "🔍 Scanner de Oportunidades"])

# --- ABA 1: ANÁLISE DO JOGO ---
with tab_analise:
    st.markdown(f"### {home_team} x {away_team}")
    
    if st.sidebar.button("Gerar Previsões 🚀", type="primary"):
        m = calculate_metrics(home_team, away_team, ref_factor, champions_mode, f_h, f_a)
        
        # Cabeçalho
        c1, c2, c3 = st.columns(3)
        c1.metric("Escanteios", f"{m['total_corners']:.2f}")
        c2.metric("Cartões", f"{m['total_cards']:.2f}")
        c3.metric("Tensão", f"{m['tension']:.2f}")
        st.divider()

        # Escanteios
        st.subheader("🚩 Escanteios")
        cols = st.columns(6)
        for i, line in enumerate([7.5, 8.5, 9.5, 10.5, 11.5, 12.5]):
            p = prob_over(m['total_corners'], line)
            c = "green" if p >= 70 else "orange" if p >= 50 else "red"
            cols[i].markdown(f"**+{line}**\n:{c}[**{p:.1f}%**]")

        col_h, col_m, col_a = st.columns([1, 1, 1])
        with col_h:
            st.markdown(f"**🏠 {home_team}**")
            st.write(f"M: **{m['ind_corn_h']:.2f}**")
            for l in [2.5, 3.5, 4.5, 5.5]: st.write(f"+{l}: **{prob_over(m['ind_corn_h'], l):.1f}%**")
        with col_a:
            st.markdown(f"**✈️ {away_team}**")
            st.write(f"M: **{m['ind_corn_a']:.2f}**")
            for l in [2.5, 3.5, 4.5, 5.5]: st.write(f"+{l}: **{prob_over(m['ind_corn_a'], l):.1f}%**")
        with col_m:
            fig = go.Figure(data=[go.Bar(x=[home_team, away_team], y=[m['ind_corn_h'], m['ind_corn_a']], marker_color=['blue', 'red'])])
            fig.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("🟨 Cartões")
        ch, cm, ca = st.columns([1, 1, 1])
        with ch:
            st.markdown(f"**🏠 {home_team}**")
            st.write(f"M: **{m['ind_card_h']:.2f}**")
            for l in [1.5, 2.5, 3.5]: st.markdown(f"+{l}: **{prob_over(m['ind_card_h'], l):.1f}%**")
        with ca:
            st.markdown(f"**✈️ {away_team}**")
            st.write(f"M: **{m['ind_card_a']:.2f}**")
            for l in [1.5, 2.5, 3.5]: st.markdown(f"+{l}: **{prob_over(m['ind_card_a'], l):.1f}%**")
        with cm:
            figc = go.Figure(data=[go.Bar(x=[home_team, away_team], y=[m['ind_card_h'], m['ind_card_a']], marker_color=['gold', 'gold'])])
            figc.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(figc, use_container_width=True)

        st.divider()
        st.subheader("⚽ Gols")
        ph = [poisson.pmf(i, m['goals_h']) for i in range(6)]
        pa = [poisson.pmf(i, m['goals_a']) for i in range(6)]
        prob_h = sum([ph[i]*pa[j] for i in range(6) for j in range(6) if i > j]) * 100
        prob_d = sum([ph[i]*pa[j] for i in range(6) for j in range(6) if i == j]) * 100
        prob_a = sum([ph[i]*pa[j] for i in range(6) for j in range(6) if i < j]) * 100
        c1, c2, c3 = st.columns(3)
        c1.metric(home_team, f"{prob_h:.1f}%")
        c2.metric("Empate", f"{prob_d:.1f}%")
        c3.metric(away_team, f"{prob_a:.1f}%")
        
        with st.expander("Placar Exato"):
            matrix = [[ph[i]*pa[j]*100 for j in range(5)] for i in range(5)]
            fig2 = go.Figure(data=go.Heatmap(z=matrix, x=[f"{away_team} {j}" for j in range(5)], y=[f"{home_team} {i}" for i in range(5)], colorscale='Viridis', texttemplate="%{z:.1f}%"))
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("👈 Configure o jogo na barra lateral e clique em 'Gerar Previsões'.")

# --- ABA 2: SCANNER ---
with tab_scanner:
    st.subheader("🕵️‍♂️ Scanner de Oportunidades")
    st.markdown("Os melhores times da base de dados para cada mercado.")
    
    df_rank = pd.DataFrame.from_dict(teams_data, orient='index')
    
    col_s1, col_s2, col_s3 = st.columns(3)
    
    with col_s1:
        st.markdown("### 🚩 Máquinas de Cantos")
        top_cantos = df_rank.sort_values(by='corners', ascending=False).head(10)
        st.dataframe(
            top_cantos[['corners']],
            column_config={"corners": st.column_config.ProgressColumn("Média", format="%.2f", min_value=0, max_value=12)},
            use_container_width=True
        )

    with col_s2:
        st.markdown("### 🟨 Mais Violentos")
        top_cards = df_rank.sort_values(by='cards', ascending=False).head(10)
        st.dataframe(
            top_cards[['cards', 'fouls']],
            column_config={
                "cards": st.column_config.NumberColumn("Cartões", format="%.2f"),
                "fouls": st.column_config.NumberColumn("Faltas", format="%.1f")
            },
            use_container_width=True
        )

    with col_s3:
        st.markdown("### ⚽ Melhores Ataques")
        top_gols = df_rank.sort_values(by='goals_f', ascending=False).head(10)
        st.dataframe(
            top_gols[['goals_f']],
            column_config={"goals_f": st.column_config.ProgressColumn("Gols Feitos", format="%.2f", min_value=0, max_value=4)},
            use_container_width=True
        )
        
    st.success("💡 Dica: Verifique se algum destes times joga hoje!")