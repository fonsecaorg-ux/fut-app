import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import poisson
import json
import hmac

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
# 1. DADOS
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
        referees = {"Genérico": 1.0}
        
    return teams_dict, referees

teams_data, referees_data = load_data()

# ==============================================================================
# 2. BARRA LATERAL
# ==============================================================================
st.sidebar.title("FutPrevisão Pro v2.6")

# Metadados do Robô
def carregar_metadados():
    try:
        with open("metadados.json", "r", encoding='utf-8') as f:
            return json.load(f)
    except: return None

meta = carregar_metadados()
if meta:
    st.sidebar.caption("🤖 **Status do Robô:**")
    st.sidebar.text(f"Verificado em:\n{meta['ultima_verificacao']}")
    if meta['times_alterados'] > 0:
        st.sidebar.success(f"⚡ Atualizado! {meta['times_alterados']} times mudaram.")
    else:
        st.sidebar.info("✔ Base verificada e estável.")
    st.sidebar.caption(f"📡 Fontes: {meta.get('fontes', 'Adamchoi & FBref')}")
else:
    st.sidebar.warning("⚠ Aguardando robô...")

st.sidebar.markdown("---")
st.sidebar.header("Configuração")

team_list = sorted(list(teams_data.keys()))
home_team = st.sidebar.selectbox("Mandante", team_list, index=0)
away_team = st.sidebar.selectbox("Visitante", team_list, index=1)

referee_list = sorted(list(referees_data.keys())) + ["Outro"]
ref_name = st.sidebar.selectbox("Árbitro", referee_list)
if ref_name == "Outro":
    ref_factor = st.sidebar.slider("Rigor", 0.8, 1.4, 1.0, 0.1)
else:
    ref_factor = referees_data[ref_name]
    st.sidebar.metric("Rigor", ref_factor)

champions_mode = st.sidebar.checkbox("Modo Champions/Clássico (-15%)", value=False)

# ==============================================================================
# 3. CÁLCULOS
# ==============================================================================
def calculate_metrics(home, away, ref_factor, is_champions):
    h_data = teams_data[home]
    a_data = teams_data[away]
    
    # --- ESCANTEIOS ---
    corn_h = h_data['corners'] * 1.10
    corn_a = a_data['corners'] * 0.85
    
    if is_champions:
        corn_h *= 0.85
        corn_a *= 0.85
        
    total_corners = corn_h + corn_a
        
    # --- CARTÕES (INDIVIDUALIZADO) ---
    tension = (h_data['fouls'] + a_data['fouls']) / 24.0
    tension = max(0.85, min(tension, 1.30))
    
    # Calculamos o esperado individualmente aplicando Tensão e Juiz
    card_h = h_data['cards'] * tension * ref_factor
    card_a = a_data['cards'] * tension * ref_factor
    
    total_cards = card_h + card_a
    
    # --- GOLS ---
    avg_l = 1.3
    exp_h = (h_data['goals_f']/avg_l) * (a_data['goals_a']/avg_l) * avg_l
    exp_a = (a_data['goals_f']/avg_l) * (h_data['goals_a']/avg_l) * avg_l
    
    return {
        'total_corners': total_corners,
        'ind_corn_h': corn_h,
        'ind_corn_a': corn_a,
        'total_cards': total_cards,
        'ind_card_h': card_h, # Individual Casa
        'ind_card_a': card_a, # Individual Fora
        'goals_h': exp_h,
        'goals_a': exp_a,
        'tension': tension
    }

# Função auxiliar para probabilidade
def prob_over(expectation, line):
    k = int(line) 
    prob = poisson.sf(k, expectation)
    return prob * 100

# ==============================================================================
# 4. INTERFACE
# ==============================================================================
st.title(f"📊 {home_team} x {away_team}")

if st.sidebar.button("Gerar Previsões 🚀", type="primary"):
    
    m = calculate_metrics(home_team, away_team, ref_factor, champions_mode)
    
    # --- CABEÇALHO RESUMO ---
    c1, c2, c3 = st.columns(3)
    c1.metric("Média Escanteios", f"{m['total_corners']:.2f}")
    c2.metric("Média Cartões", f"{m['total_cards']:.2f}")
    c3.metric("Tensão", f"{m['tension']:.2f}")
    st.divider()

    # --- SEÇÃO 1: ESCANTEIOS ---
    st.subheader("🚩 Escanteios (Corners)")
    
    # Radar Total
    cols = st.columns(6)
    lines = [7.5, 8.5, 9.5, 10.5, 11.5, 12.5]
    for i, col in enumerate(cols):
        line = lines[i]
        prob = prob_over(m['total_corners'], line)
        color = "green" if prob >= 70 else "orange" if prob >= 50 else "red"
        col.markdown(f"**Over {line}**")
        col.markdown(f":{color}[**{prob:.1f}%**]")

    st.markdown("<br>", unsafe_allow_html=True)

    # Individuais Escanteios
    col_h, col_m, col_a = st.columns([1, 1, 1])
    with col_h:
        st.markdown(f"### 🏠 {home_team}")
        st.write(f"Média: **{m['ind_corn_h']:.2f}**")
        for line in [2.5, 3.5, 4.5, 5.5]:
            p = prob_over(m['ind_corn_h'], line)
            st.write(f"Over {line}: **{p:.1f}%**")

    with col_a:
        st.markdown(f"### ✈️ {away_team}")
        st.write(f"Média: **{m['ind_corn_a']:.2f}**")
        for line in [2.5, 3.5, 4.5, 5.5]:
            p = prob_over(m['ind_corn_a'], line)
            st.write(f"Over {line}: **{p:.1f}%**")

    with col_m:
        st.markdown("### 🆚 Força (Cantos)")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=[home_team, away_team], y=[m['ind_corn_h'], m['ind_corn_a']], marker_color=['blue', 'red']))
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # --- SEÇÃO 2: CARTÕES (NOVO) ---
    st.subheader("🟨 Cartões (Cards)")
    
    cc_h, cc_m, cc_a = st.columns([1, 1, 1])
    
    # Individual Cartões Casa
    with cc_h:
        st.markdown(f"### 🏠 {home_team}")
        st.write(f"Média Esperada: **{m['ind_card_h']:.2f}**")
        # Linhas típicas de cartões individuais
        for line in [1.5, 2.5, 3.5]:
            p = prob_over(m['ind_card_h'], line)
            # Destaque visual se for alta probabilidade
            destaque = "green" if p > 60 else "white"
            st.markdown(f"Over {line}: :{destaque}[**{p:.1f}%**]")

    # Individual Cartões Fora
    with cc_a:
        st.markdown(f"### ✈️ {away_team}")
        st.write(f"Média Esperada: **{m['ind_card_a']:.2f}**")
        for line in [1.5, 2.5, 3.5]:
            p = prob_over(m['ind_card_a'], line)
            destaque = "green" if p > 60 else "white"
            st.markdown(f"Over {line}: :{destaque}[**{p:.1f}%**]")

    # Comparativo Cartões
    with cc_m:
        st.markdown("### 🆚 Disciplina")
        fig_c = go.Figure()
        fig_c.add_trace(go.Bar(x=[home_team, away_team], y=[m['ind_card_h'], m['ind_card_a']], marker_color=['gold', 'gold']))
        fig_c.update_layout(height=250, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_c, use_container_width=True)

    st.divider()

    # --- SEÇÃO 3: GOLS ---
    st.subheader("⚽ Probabilidades de Gols")
    
    ph = [poisson.pmf(i, m['goals_h']) for i in range(6)]
    pa = [poisson.pmf(i, m['goals_a']) for i in range(6)]
    prob_home = sum([ph[i]*pa[j] for i in range(6) for j in range(6) if i > j]) * 100
    prob_draw = sum([ph[i]*pa[j] for i in range(6) for j in range(6) if i == j]) * 100
    prob_away = sum([ph[i]*pa[j] for i in range(6) for j in range(6) if i < j]) * 100
    
    c_g1, c_g2, c_g3 = st.columns(3)
    c_g1.metric(f"Vitória {home_team}", f"{prob_home:.1f}%")
    c_g2.metric("Empate", f"{prob_draw:.1f}%")
    c_g3.metric(f"Vitória {away_team}", f"{prob_away:.1f}%")
    
    with st.expander("Ver Mapa de Calor (Placar Exato)"):
        matrix = [[ph[i]*pa[j]*100 for j in range(5)] for i in range(5)]
        fig2 = go.Figure(data=go.Heatmap(
            z=matrix, 
            x=[f"{away_team} {j}" for j in range(5)], 
            y=[f"{home_team} {i}" for i in range(5)], 
            colorscale='Viridis', texttemplate="%{z:.1f}%"
        ))
        st.plotly_chart(fig2, use_container_width=True)
