import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import poisson
import json
import hmac

# Configuração e Login (Igual ao anterior, vou resumir para caber, mantenha o seu check_password)
st.set_page_config(page_title="FutPrevisão Pro", layout="wide", page_icon="⚽")

def check_password():
    if "password_correct" not in st.session_state: st.session_state["password_correct"] = False
    def password_entered():
        if "passwords" in st.secrets:
            if st.session_state["username"] in st.secrets["passwords"] and hmac.compare_digest(st.session_state["password"], st.secrets["passwords"][st.session_state["username"]]):
                st.session_state["password_correct"] = True
                del st.session_state["password"]; del st.session_state["username"]
            else: st.session_state["password_correct"] = False; st.error("😕 Senha incorreta")
        else: st.error("Erro config")
    if st.session_state["password_correct"]: return True
    st.markdown("### 🔒 Login"); st.text_input("User", key="username"); st.text_input("Pass", type="password", key="password"); st.button("Entrar", on_click=password_entered); return False

if not check_password(): st.stop()

# ==============================================================================
# 1. DADOS
# ==============================================================================
BACKUP_TEAMS = {"Arsenal": {"corners": 6.82, "cards": 1.0, "fouls": 10.5, "goals_f": 2.3, "goals_a": 0.8}}

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
                'corners': safe_float(row['Escanteios']), 'cards': safe_float(row['CartoesAmarelos']), 
                'fouls': safe_float(row['Faltas']), 'goals_f': safe_float(row['GolsFeitos']), 'goals_a': safe_float(row['GolsSofridos'])
            }
    except: teams_dict = BACKUP_TEAMS
    
    try:
        df_ref = pd.read_csv("arbitros.csv")
        referees = dict(zip(df_ref['Nome'], df_ref['Fator']))
    except: referees = {}
    
    # --- AQUI: ADICIONANDO OS GENÉRICOS ---
    referees['⚠️ (Genérico) Rigoroso'] = 1.25
    referees['⚠️ (Genérico) Normal'] = 1.00
    referees['⚠️ (Genérico) Leniente/Conservador'] = 0.80
    
    return teams_dict, referees

teams_data, referees_data = load_data()

# ==============================================================================
# 2. BARRA LATERAL
# ==============================================================================
st.sidebar.title("FutPrevisão Pro v2.7")

def carregar_metadados():
    try:
        with open("metadados.json", "r", encoding='utf-8') as f: return json.load(f)
    except: return None

meta = carregar_metadados()
if meta:
    st.sidebar.caption("🤖 **Status do Robô:**")
    st.sidebar.text(f"{meta['ultima_verificacao']}")
    
    # Mensagem de Log do Robô (NOVO)
    if 'log' in meta:
        st.sidebar.caption("📋 Relatório:")
        st.sidebar.info(meta['log'])
    
    if meta['times_alterados'] == 0 and 'log' not in meta:
        st.sidebar.info("✔ Base verificada e estável.")
    st.sidebar.caption(f"📡 Fontes: {meta.get('fontes', 'Adamchoi & FBref')}")
else:
    st.sidebar.warning("⚠ Aguardando primeira execução...")

st.sidebar.markdown("---")
st.sidebar.header("Configuração")

team_list = sorted(list(teams_data.keys()))
home_team = st.sidebar.selectbox("Mandante", team_list, index=0)
away_team = st.sidebar.selectbox("Visitante", team_list, index=1)

st.sidebar.markdown("---")
st.sidebar.caption("🧠 **Contexto**")
context_options = {
    "⚪ Neutro": 1.0, "🔥 Must Win": 1.15, "❄️ Desmobilizado": 0.85,
    "💪 Super Favorito": 1.25, "🚑 Crise": 0.80
}
f_h = context_options[st.sidebar.selectbox(f"Momento: {home_team}", list(context_options.keys()), index=0)]
f_a = context_options[st.sidebar.selectbox(f"Momento: {away_team}", list(context_options.keys()), index=0)]

st.sidebar.markdown("---")
# Lista de árbitros com genéricos no topo
referee_list = sorted(list(referees_data.keys()))
# Força os genéricos para o topo da lista visualmente se necessário, ou deixa ordenado
ref_name = st.sidebar.selectbox("Árbitro", referee_list)
ref_factor = referees_data[ref_name]
st.sidebar.metric("Rigor", ref_factor)

champions_mode = st.sidebar.checkbox("Modo Champions (-15%)", value=False)

# ==============================================================================
# 3. CÁLCULOS
# ==============================================================================
def calculate_metrics(home, away, ref_factor, is_champions, fact_h, fact_a):
    h_data = teams_data[home]; a_data = teams_data[away]
    
    corn_h = (h_data['corners'] * 1.10) * fact_h
    corn_a = (a_data['corners'] * 0.85) * fact_a
    if is_champions: corn_h *= 0.85; corn_a *= 0.85
    total_corners = corn_h + corn_a
        
    tension_boost = 1.10 if fact_h > 1.0 or fact_a > 1.0 else 1.0
    tension = ((h_data['fouls'] + a_data['fouls']) / 24.0) * tension_boost
    tension = max(0.85, min(tension, 1.40))
    
    card_h = h_data['cards'] * tension * ref_factor
    card_a = a_data['cards'] * tension * ref_factor
    total_cards = card_h + card_a
    
    avg_l = 1.3
    exp_h = ((h_data['goals_f'] * fact_h)/avg_l) * (a_data['goals_a']/avg_l) * avg_l
    exp_a = ((a_data['goals_f'] * fact_a)/avg_l) * (h_data['goals_a']/avg_l) * avg_l
    
    return {'total_corners': total_corners, 'ind_corn_h': corn_h, 'ind_corn_a': corn_a,
            'total_cards': total_cards, 'ind_card_h': card_h, 'ind_card_a': card_a,
            'goals_h': exp_h, 'goals_a': exp_a, 'tension': tension}

def prob_over(exp, line): return poisson.sf(int(line), exp) * 100

# ==============================================================================
# 4. INTERFACE
# ==============================================================================
st.title(f"📊 {home_team} x {away_team}")

if st.sidebar.button("Gerar Previsões 🚀", type="primary"):
    m = calculate_metrics(home_team, away_team, ref_factor, champions_mode, f_h, f_a)
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Escanteios", f"{m['total_corners']:.2f}")
    c2.metric("Cartões", f"{m['total_cards']:.2f}")
    c3.metric("Tensão", f"{m['tension']:.2f}")
    st.divider()

    st.subheader("🚩 Escanteios")
    cols = st.columns(6)
    for i, line in enumerate([7.5, 8.5, 9.5, 10.5, 11.5, 12.5]):
        p = prob_over(m['total_corners'], line)
        c = "green" if p >= 70 else "orange" if p >= 50 else "red"
        cols[i].markdown(f"**+{line}**\n:{c}[**{p:.1f}%**]")

    col_h, col_m, col_a = st.columns([1, 1, 1])
    with col_h:
        st.markdown(f"**🏠 {home_team}**"); st.write(f"M: **{m['ind_corn_h']:.2f}**")
        for l in [2.5, 3.5, 4.5, 5.5]: st.write(f"+{l}: **{prob_over(m['ind_corn_h'], l):.1f}%**")
    with col_a:
        st.markdown(f"**✈️ {away_team}**"); st.write(f"M: **{m['ind_corn_a']:.2f}**")
        for l in [2.5, 3.5, 4.5, 5.5]: st.write(f"+{l}: **{prob_over(m['ind_corn_a'], l):.1f}%**")
    with col_m:
        fig = go.Figure(data=[go.Bar(x=[home_team, away_team], y=[m['ind_corn_h'], m['ind_corn_a']], marker_color=['blue', 'red'])])
        fig.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10)); st.plotly_chart(fig, use_container_width=True)

    st.divider(); st.subheader("🟨 Cartões")
    ch, cm, ca = st.columns([1, 1, 1])
    with ch:
        st.markdown(f"**🏠 {home_team}**"); st.write(f"M: **{m['ind_card_h']:.2f}**")
        for l in [1.5, 2.5, 3.5]: st.markdown(f"+{l}: **{prob_over(m['ind_card_h'], l):.1f}%**")
    with ca:
        st.markdown(f"**✈️ {away_team}**"); st.write(f"M: **{m['ind_card_a']:.2f}**")
        for l in [1.5, 2.5, 3.5]: st.markdown(f"+{l}: **{prob_over(m['ind_card_a'], l):.1f}%**")
    with cm:
        figc = go.Figure(data=[go.Bar(x=[home_team, away_team], y=[m['ind_card_h'], m['ind_card_a']], marker_color=['gold', 'gold'])])
        figc.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10)); st.plotly_chart(figc, use_container_width=True)

    st.divider(); st.subheader("⚽ Gols")
    ph = [poisson.pmf(i, m['goals_h']) for i in range(6)]; pa = [poisson.pmf(i, m['goals_a']) for i in range(6)]
    prob_h = sum([ph[i]*pa[j] for i in range(6) for j in range(6) if i > j]) * 100
    prob_d = sum([ph[i]*pa[j] for i in range(6) for j in range(6) if i == j]) * 100
    prob_a = sum([ph[i]*pa[j] for i in range(6) for j in range(6) if i < j]) * 100
    c1, c2, c3 = st.columns(3); c1.metric(home_team, f"{prob_h:.1f}%"); c2.metric("Empate", f"{prob_d:.1f}%"); c3.metric(away_team, f"{prob_a:.1f}%")
    
    with st.expander("Placar Exato"):
        matrix = [[ph[i]*pa[j]*100 for j in range(5)] for i in range(5)]
        fig2 = go.Figure(data=go.Heatmap(z=matrix, x=[f"{away_team} {j}" for j in range(5)], y=[f"{home_team} {i}" for i in range(5)], colorscale='Viridis', texttemplate="%{z:.1f}%"))
        st.plotly_chart(fig2, use_container_width=True)
