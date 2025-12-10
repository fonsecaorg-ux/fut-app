import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import uuid
import math
from datetime import datetime

# ==============================================================================
# 0. CONFIGURAÇÃO INICIAL
# ==============================================================================
st.set_page_config(page_title="FutPrevisão Pro", layout="wide", page_icon="⚽")

# Tenta importar Plotly (Opcional)
try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# ==============================================================================
# 1. FUNÇÕES MATEMÁTICAS (SUBSTITUI SCIPY)
# ==============================================================================
def poisson_pmf(k, mu):
    """Calcula probabilidade exata (Probability Mass Function)"""
    return (math.exp(-mu) * (mu ** k)) / math.factorial(k)

def poisson_sf(k, mu):
    """Calcula probabilidade de ser MAIOR que k (Survival Function)"""
    # P(X > k) = 1 - P(X <= k)
    cdf = 0
    for i in range(int(k) + 1):
        cdf += poisson_pmf(i, mu)
    return 1 - cdf

def prob_over(exp, line):
    """Calcula % de chance de ser maior que a linha"""
    return poisson_sf(int(line), exp) * 100

# ==============================================================================
# 2. SEGURANÇA E LOGIN
# ==============================================================================
USERS = {
    "diego": "@Casa612"
}

def check_login():
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    if st.session_state["logged_in"]: return True

    st.markdown("### 🔒 Acesso FutPrevisão Pro")
    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        username = st.text_input("Usuário")
        password = st.text_input("Senha", type="password")
        if st.button("Entrar"):
            if username in USERS and password == USERS[username]:
                st.session_state["logged_in"] = True
                st.rerun()
            else:
                st.error("❌ Credenciais inválidas")
    return False

if not check_login(): st.stop()

# ==============================================================================
# 3. CARREGAMENTO DE DADOS (BLINDADO)
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
        # Tenta ler com diferentes codificações e separadores
        try:
            df = pd.read_csv("dados_times.csv", encoding='utf-8')
        except:
            df = pd.read_csv("dados_times.csv", encoding='latin1', sep=';') # Tenta formato Excel BR

        teams_dict = {}
        # Normaliza nomes de colunas (remove espaços)
        df.columns = [c.strip() for c in df.columns]
        
        for _, row in df.iterrows():
            # Verifica se as colunas existem antes de ler
            if 'Time' in row and 'Escanteios' in row:
                teams_dict[row['Time']] = {
                    'corners': safe_float(row['Escanteios']),
                    'cards': safe_float(row['CartoesAmarelos']) if 'CartoesAmarelos' in row else 2.0, 
                    'fouls': safe_float(row['Faltas']) if 'Faltas' in row else 12.0,
                    'goals_f': safe_float(row['GolsFeitos']) if 'GolsFeitos' in row else 1.5,
                    'goals_a': safe_float(row['GolsSofridos']) if 'GolsSofridos' in row else 1.0
                }
    except Exception as e:
        teams_dict = BACKUP_TEAMS
    
    return teams_dict

teams_data = load_data()
team_list_raw = sorted(list(teams_data.keys()))
team_list_with_empty = [""] + team_list_raw

# Mercados
MERCADOS_LISTA = ["Selecione..."]
MERCADOS_LISTA.extend([f"Cantos > {i}" for i in [7.5, 8.5, 9.5, 10.5, 11.5, 12.5]])
MERCADOS_LISTA.extend([f"Cartões > {i}" for i in [1.5, 2.5, 3.5, 4.5]])
MERCADOS_LISTA.extend(["Gols > 0.5", "Gols > 1.5", "Gols > 2.5", "Ambas Marcam", "ML Casa", "ML Fora"])

# ==============================================================================
# 4. LÓGICA DE PREVISÃO
# ==============================================================================
def calcular_previsao(home, away, f_h, f_a):
    h_data = teams_data.get(home, BACKUP_TEAMS["Arsenal"])
    a_data = teams_data.get(away, BACKUP_TEAMS["Man City"])
    
    # Escanteios
    corn_h = (h_data['corners'] * 1.10) * f_h
    corn_a = (a_data['corners'] * 0.85) * f_a
    total_corners = corn_h + corn_a
    
    # Cartões (Intensidade)
    avg_fouls = (h_data['fouls'] + a_data['fouls']) / 2
    tension = avg_fouls / 12.0
    card_h = h_data['cards'] * tension
    card_a = a_data['cards'] * tension
    total_cards = card_h + card_a
    
    # Gols
    avg_l = 1.3
    goals_h = ((h_data['goals_f'] * f_h) / avg_l) * (a_data['goals_a'] / avg_l) * avg_l
    goals_a = ((a_data['goals_f'] * f_a) / avg_l) * (h_data['goals_a'] / avg_l) * avg_l
    
    return {
        "corners": {"t": total_corners, "h": corn_h, "a": corn_a},
        "cards": {"t": total_cards, "h": card_h, "a": card_a},
        "goals": {"h": goals_h, "a": goals_a}
    }

def get_color(prob):
    if prob >= 70: return "green"
    if prob >= 50: return "orange"
    return "red"

# ==============================================================================
# 5. GESTÃO DE DADOS (JSON)
# ==============================================================================
DATA_FILE = "historico_bilhetes_v6.json"
CONFIG_FILE = "config_banca.json"

def carregar_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f: return json.load(f)
    return {"banca_inicial": 1000.0, "stop_loss": 50.0}

def salvar_config(cfg):
    with open(CONFIG_FILE, "w") as f: json.dump(cfg, f)

def carregar_tickets():
    if not os.path.exists(DATA_FILE): return []
    try:
        with open(DATA_FILE, "r") as f: return json.load(f)
    except: return []

def salvar_ticket(ticket_data):
    dados = carregar_tickets()
    dados.append(ticket_data)
    with open(DATA_FILE, "w") as f: json.dump(dados, f, indent=2)

def excluir_ticket(id_ticket):
    dados = carregar_tickets()
    novos = [t for t in dados if t.get("id") != id_ticket]
    with open(DATA_FILE, "w") as f: json.dump(novos, f, indent=2)

# ==============================================================================
# 6. DASHBOARD
# ==============================================================================
def render_dashboard():
    st.title("📊 Painel de Controle")
    
    cfg = carregar_config()
    with st.sidebar.expander("⚙️ Configurações"):
        nb = st.number_input("Banca Inicial", value=cfg["banca_inicial"])
        ns = st.number_input("Stop Loss", value=cfg["stop_loss"])
        if st.button("Salvar"):
            salvar_config({"banca_inicial": nb, "stop_loss": ns})
            st.rerun()

    # KPIs
    tickets = carregar_tickets()
    lucro_total = sum(t["Lucro"] for t in tickets)
    banca_atual = cfg["banca_inicial"] + lucro_total
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Banca", f"R$ {banca_atual:,.2f}", f"{lucro_total:,.2f}")
    c2.metric("Bilhetes", len(tickets))
    
    # CSS Custom
    st.markdown("""
    <style>
        .card { padding: 15px; border-radius: 8px; margin-bottom: 10px; border-left: 5px solid #ccc; background: #f9f9f9; }
        .win { border-color: #28a745; background: #d4edda; }
        .loss { border-color: #dc3545; background: #f8d7da; }
        .cash { border-color: #ffc107; background: #fff3cd; }
    </style>
    """, unsafe_allow_html=True)

    tab_sim, tab_new, tab_hist, tab_grf = st.tabs(["🔮 Simulação", "➕ Novo Bilhete", "📜 Histórico", "📈 Gráficos"])

    # --- ABA SIMULAÇÃO ---
    with tab_sim:
        st.markdown("### 🤖 Previsão IA")
        col_sel1, col_sel2 = st.columns(2)
        h_t = col_sel1.selectbox("Mandante", team_list_raw, index=0, key="sim_h")
        a_t = col_sel2.selectbox("Visitante", team_list_raw, index=1, key="sim_a")
        
        with st.expander("Ajustes Avançados"):
            ctx_h = st.slider("Momento Casa", 0.8, 1.2, 1.0, 0.1)
            ctx_a = st.slider("Momento Fora", 0.8, 1.2, 1.0, 0.1)

        if st.button("Calcular Probabilidades", type="primary"):
            res = calcular_previsao(h_t, a_t, ctx_h, ctx_a)
            
            c_res1, c_res2, c_res3 = st.columns(3)
            with c_res1:
                st.info(f"🚩 **Cantos** (Exp: {res['corners']['t']:.2f})")
                for l in [8.5, 9.5, 10.5]:
                    p = prob_over(res['corners']['t'], l)
                    st.write(f"+{l}: :{get_color(p)}[**{p:.1f}%**]")
            
            with c_res2:
                st.success(f"⚽ **Gols**")
                p_btts = (1 - poisson_pmf(0, res['goals']['h'])) * (1 - poisson_pmf(0, res['goals']['a'])) * 100
                st.write(f"Ambas Marcam: :{get_color(p_btts)}[**{p_btts:.1f}%**]")
                for l in [1.5, 2.5]:
                    p = prob_over(res['goals']['h'] + res['goals']['a'], l)
                    st.write(f"+{l}: :{get_color(p)}[**{p:.1f}%**]")

            with c_res3:
                st.warning(f"🟨 **Cartões** (Exp: {res['cards']['t']:.2f})")
                for l in [3.5, 4.5]:
                    p = prob_over(res['cards']['t'], l)
                    st.write(f"+{l}: :{get_color(p)}[**{p:.1f}%**]")

    # --- ABA NOVO BILHETE ---
    with tab_new:
        st.markdown("### Registrar Aposta")
        if 'n_games' not in st.session_state: st.session_state['n_games'] = 1
        
        col_d, col_s, col_o = st.columns(3)
        dt = col_d.date_input("Data")
        sk = col_s.number_input("Stake", min_value=1.0, value=10.0)
        od = col_o.number_input("Odd", min_value=1.01, value=1.5)
        res_opt = st.selectbox("Resultado", ["Green", "Red", "Cashout", "Reembolso"])
        
        profit = (sk * od - sk) if res_opt == "Green" else (-sk if res_opt == "Red" else 0.0)
        if res_opt == "Cashout":
            c_val = st.number_input("Valor Retorno")
            profit = c_val - sk
            
        st.write(f"Lucro: **R$ {profit:.2f}**")
        
        jogos = []
        for i in range(st.session_state['n_games']):
            st.markdown(f"**Jogo {i+1}**")
            c_m, c_v, c_merc = st.columns([2,2,2])
            jm = c_m.selectbox(f"Mandante {i}", team_list_with_empty, key=f"nm_{i}")
            jv = c_v.selectbox(f"Visitante {i}", team_list_with_empty, key=f"nv_{i}")
            jmerc = c_merc.selectbox(f"Mercado {i}", MERCADOS_LISTA, key=f"nmerc_{i}")
            jogos.append({"Nome": f"{jm} x {jv}", "Selecoes": [{"Mercado": jmerc}]})
        
        c_b1, c_b2 = st.columns(2)
        if c_b1.button("➕ Jogo"):
            st.session_state['n_games'] += 1
            st.rerun()
        if c_b2.button("💾 Salvar"):
            tk = {
                "id": str(uuid.uuid4())[:8], "Data": dt.strftime("%d/%m/%Y"),
                "Stake": sk, "Odd": od, "Lucro": profit, "Resultado": res_opt, "Jogos": jogos
            }
            salvar_ticket(tk)
            st.success("Salvo!")
            st.session_state['n_games'] = 1
            st.rerun()

    # --- ABA HISTÓRICO ---
    with tab_hist:
        for t in tickets:
            cls = "win" if "Green" in t["Resultado"] else "loss" if "Red" in t["Resultado"] else "cash"
            st.markdown(f"""
            <div class="card {cls}">
                <b>{t['Data']}</b> | Stake: {t['Stake']} | Lucro: <b>{t['Lucro']:.2f}</b><br>
                <small>{', '.join([j['Nome'] for j in t['Jogos']])}</small>
            </div>
            """, unsafe_allow_html=True)
            if st.button("🗑️", key=t['id']):
                excluir_ticket(t['id'])
                st.rerun()

    # --- ABA GRÁFICOS ---
    with tab_grf:
        if HAS_PLOTLY and tickets:
            df = pd.DataFrame(tickets)
            df['Data_Dt'] = pd.to_datetime(df['Data'], format="%d/%m/%Y")
            df = df.sort_values('Data_Dt')
            df['Acumulado'] = df['Lucro'].cumsum()
            st.plotly_chart(px.line(df, x='Data_Dt', y='Acumulado', markers=True), use_container_width=True)
        else:
            st.info("Sem dados ou Plotly não instalado.")

if __name__ == "__main__":
    render_dashboard()
