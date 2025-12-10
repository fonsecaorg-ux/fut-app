import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import uuid
import math
from pathlib import Path
from datetime import datetime

# ==============================================================================
# 0. CONFIGURAÇÃO INICIAL
# ==============================================================================
st.set_page_config(page_title="FutPrevisão Pro V2.1", layout="wide", page_icon="⚽")

# Tenta importar Plotly
try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# ==============================================================================
# 1. FUNÇÕES MATEMÁTICAS (POISSON)
# ==============================================================================
def poisson_pmf(k, mu):
    return (math.exp(-mu) * (mu ** k)) / math.factorial(k)

def poisson_sf(k, mu):
    cdf = 0
    for i in range(int(k) + 1):
        cdf += poisson_pmf(i, mu)
    return 1 - cdf

def prob_over(exp, line):
    return poisson_sf(int(line), exp) * 100

def get_color(prob):
    if prob >= 70: return "green"
    if prob >= 50: return "orange"
    return "red"

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
# 3. CARREGAMENTO DE DADOS
# ==============================================================================

# --- A. Dados Matemáticos (CSV) ---
BACKUP_TEAMS = {
    "Arsenal": {"corners": 6.82, "cards": 1.00, "fouls": 10.45, "goals_f": 2.3, "goals_a": 0.8},
    "Man City": {"corners": 7.45, "cards": 1.50, "fouls": 9.20, "goals_f": 2.7, "goals_a": 0.8},
}

def safe_float(value):
    try: return float(str(value).replace(',', '.'))
    except: return 0.0

@st.cache_data(ttl=3600)
def load_csv_data():
    try:
        try: df = pd.read_csv("dados_times.csv", encoding='utf-8')
        except: df = pd.read_csv("dados_times.csv", encoding='latin1', sep=';')
        
        teams_dict = {}
        df.columns = [c.strip() for c in df.columns]
        for _, row in df.iterrows():
            if 'Time' in row:
                teams_dict[row['Time']] = {
                    'corners': safe_float(row.get('Escanteios', 0)),
                    'cards': safe_float(row.get('CartoesAmarelos', 2.0)), 
                    'fouls': safe_float(row.get('Faltas', 12.0)),
                    'goals_f': safe_float(row.get('GolsFeitos', 1.5)),
                    'goals_a': safe_float(row.get('GolsSofridos', 1.0))
                }
        return teams_dict
    except: return BACKUP_TEAMS

@st.cache_data(ttl=3600)
def load_referees():
    try:
        # Lê o arquivo de árbitros e cria um dicionário {Nome: Fator}
        df = pd.read_csv("arbitros.csv")
        return dict(zip(df['Nome'], df['Fator']))
    except:
        return {}

# --- B. Dados Históricos (TXT Adam Choi) ---
ARQUIVOS_ADAM = {
    "Premier League": "Escanteios Preimier League - codigo fonte.txt",
    "La Liga": "Escanteios Espanha.txt",
    "Serie A": "Escanteios Italia.txt",
    "Bundesliga": "Escanteios Alemanha.txt",
    "Ligue 1": "Escanteios França.txt",
}

class AdamChoiLoader:
    def __init__(self):
        self.data = {}
        self.load_files()

    def load_files(self):
        pasta = Path(__file__).parent
        for liga, arquivo in ARQUIVOS_ADAM.items():
            caminho = pasta / arquivo
            if caminho.exists():
                try:
                    with open(caminho, 'r', encoding='utf-8') as f:
                        raw = f.read().strip()
                        if '{' in raw: raw = raw[raw.find('{'):]
                        self.data[liga] = json.loads(raw)
                except: pass

    def get_history(self, team, league, key):
        if league not in self.data: return None
        for t in self.data[league].get('teams', []):
            if t['teamName'] == team:
                stats = t.get(key)
                if stats and len(stats) >= 3:
                    return stats[0], stats[1], stats[2] 
        return None

# Inicializa Carregadores
teams_data = load_csv_data()
referees_data = load_referees()
team_list_raw = sorted(list(teams_data.keys()))
team_list_with_empty = [""] + team_list_raw
history_loader = AdamChoiLoader()

# ==============================================================================
# 4. LÓGICA DE PREVISÃO
# ==============================================================================
def calcular_previsao(home, away, f_h, f_a, ref_factor=1.0):
    h_data = teams_data.get(home, BACKUP_TEAMS["Arsenal"])
    a_data = teams_data.get(away, BACKUP_TEAMS["Man City"])
    
    # Escanteios (Contextual)
    corn_h = (h_data['corners'] * 1.10) * f_h
    corn_a = (a_data['corners'] * 0.85) * f_a
    total_corners = corn_h + corn_a
    
    # Cartões (Intensidade + Contexto + Árbitro)
    tension_boost = 1.0
    if f_h > 1.05 or f_a > 1.05: tension_boost = 1.15
    
    avg_fouls = (h_data['fouls'] + a_data['fouls']) / 2
    tension = (avg_fouls / 12.0) * tension_boost
    
    # Aplica o fator do árbitro
    card_h = h_data['cards'] * tension * ref_factor
    card_a = a_data['cards'] * tension * ref_factor
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
    
    # CSS Customizado
    st.markdown("""
    <style>
        .bet-card-green { background-color: #d4edda; border-left: 5px solid #28a745; padding: 15px; border-radius: 5px; margin-bottom: 10px; }
        .bet-card-red { background-color: #f8d7da; border-left: 5px solid #dc3545; padding: 15px; border-radius: 5px; margin-bottom: 10px; }
        .stat-box { background: #f8f9fa; padding: 10px; border-radius: 8px; margin-bottom: 5px; border: 1px solid #dee2e6; }
        .stat-label { font-size: 12px; color: #666; }
        .stat-val { font-weight: bold; font-size: 14px; }
        .real-data { font-size: 11px; color: #007bff; font-weight: bold; }
        .total-header { font-size: 16px; font-weight: bold; color: #333; margin-top: 15px; margin-bottom: 5px; border-bottom: 2px solid #ddd; }
    </style>
    """, unsafe_allow_html=True)
    
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
    
    tab_sim, tab_new, tab_hist, tab_grf = st.tabs(["🔮 Simulação IA", "➕ Novo Bilhete", "📜 Histórico", "📈 Gráficos"])

    # --- ABA 1: SIMULAÇÃO (ATUALIZADA) ---
    with tab_sim:
        st.markdown("### 🏟️ Análise de Jogo")
        
        # 1. Seletores Principais
        col_liga, col_arb = st.columns([2, 2])
        liga_sel = col_liga.selectbox("Liga (para dados históricos)", list(ARQUIVOS_ADAM.keys()))
        
        # --- NOVO: Dropdown de Árbitros (Lê do CSV) ---
        lista_arbitros = sorted(list(referees_data.keys()))
        if not lista_arbitros: lista_arbitros = ["Genérico (Padrão)"]
        
        arbitro_sel = col_arb.selectbox("Árbitro da Partida", lista_arbitros)
        ref_factor = referees_data.get(arbitro_sel, 1.0)
        col_arb.caption(f"Fator de Rigor: {ref_factor}x")

        col_h, col_a = st.columns(2)
        home_team = col_h.selectbox("Mandante", team_list_raw, index=0)
        away_team = col_a.selectbox("Visitante", team_list_raw, index=1 if len(team_list_raw) > 1 else 0)

        # 2. Contexto
        st.write("---")
        st.caption("Contexto da Partida")
        
        ctx_map = {
            "Neutro (Meio Tabela)": 1.0,
            "Briga por Título 🏆": 1.15,
            "Z4 (Rebaixamento) 🔥": 1.10,
            "Rebaixado (Desmotivado) ❄️": 0.85
        }
        
        cc1, cc2 = st.columns(2)
        c_ctx_h = cc1.selectbox(f"Momento {home_team}", list(ctx_map.keys()), index=0)
        c_ctx_a = cc2.selectbox(f"Momento {away_team}", list(ctx_map.keys()), index=0)
        
        factor_h = ctx_map[c_ctx_h]
        factor_a = ctx_map[c_ctx_a]

        if st.button("🔮 Gerar Relatório Completo", type="primary"):
            # Cálculos Matemáticos (Passando o fator do árbitro)
            m = calcular_previsao(home_team, away_team, factor_h, factor_a, ref_factor)
            
            st.divider()
            st.markdown(f"#### 📋 Relatório: {home_team} x {away_team}")
            
            # --- COLUNAS DE ANÁLISE ---
            col_cantos, col_cartoes = st.columns(2)
            
            # === BLOCO ESCANTEIOS ===
            with col_cantos:
                st.info("🚩 **Escanteios**")
                
                # NOVO: TOTAIS
                st.markdown('<p class="total-header">Totais da Partida</p>', unsafe_allow_html=True)
                st.write(f"Expectativa Total: **{m['corners']['t']:.2f}**")
                for line in [7.5, 8.5, 9.5]:
                    prob = prob_over(m['corners']['t'], line)
                    st.markdown(f"Mais de {line}: :{get_color(prob)}[**{prob:.0f}%**]")
                
                st.markdown("---")
                
                # INDIVIDUAIS
                st.markdown(f"**🏠 {home_team}** (Exp: {m['corners']['h']:.2f})")
                for line, key_adam in [(3.5, 'homeTeamOver35'), (4.5, 'homeTeamOver45'), (5.5, 'homeTeamOver55')]:
                    prob_math = prob_over(m['corners']['h'], line)
                    hist_data = history_loader.get_history(home_team, liga_sel, key_adam)
                    txt_real = f"({hist_data[1]}/{hist_data[0]} - {hist_data[2]})" if hist_data else ""
                    st.markdown(f"+{line}: :{get_color(prob_math)}[**{prob_math:.0f}%**] {txt_real}")

                st.markdown("")
                
                st.markdown(f"**✈️ {away_team}** (Exp: {m['corners']['a']:.2f})")
                for line, key_adam in [(3.5, 'awayTeamOver35'), (4.5, 'awayTeamOver45'), (5.5, 'awayTeamOver55')]:
                    prob_math = prob_over(m['corners']['a'], line)
                    hist_data = history_loader.get_history(away_team, liga_sel, key_adam)
                    txt_real = f"({hist_data[1]}/{hist_data[0]} - {hist_data[2]})" if hist_data else ""
                    st.markdown(f"+{line}: :{get_color(prob_math)}[**{prob_math:.0f}%**] {txt_real}")

            # === BLOCO CARTÕES ===
            with col_cartoes:
                st.warning("🟨 **Cartões**")
                
                # NOVO: TOTAIS
                st.markdown('<p class="total-header">Totais da Partida</p>', unsafe_allow_html=True)
                st.write(f"Expectativa Total: **{m['cards']['t']:.2f}**")
                for line in [2.5, 3.5, 4.5]:
                    prob = prob_over(m['cards']['t'], line)
                    st.markdown(f"Mais de {line}: :{get_color(prob)}[**{prob:.0f}%**]")

                st.markdown("---")
                
                # INDIVIDUAIS
                st.markdown(f"**🏠 {home_team}** (Exp: {m['cards']['h']:.2f})")
                for line in [1.5, 2.5]:
                    prob = prob_over(m['cards']['h'], line)
                    st.markdown(f"+{line}: :{get_color(prob)}[**{prob:.0f}%**]")
                
                st.markdown("")
                
                st.markdown(f"**✈️ {away_team}** (Exp: {m['cards']['a']:.2f})")
                for line in [1.5, 2.5]:
                    prob = prob_over(m['cards']['a'], line)
                    st.markdown(f"+{line}: :{get_color(prob)}[**{prob:.0f}%**]")

    # --- ABA NOVO BILHETE (MANTIDA) ---
    with tab_new:
        st.markdown("### Registrar Aposta")
        if 'n_games' not in st.session_state: st.session_state['n_games'] = 1
        
        c_d, c_s, c_o = st.columns(3)
        dt = c_d.date_input("Data")
        sk = c_s.number_input("Stake", min_value=1.0, value=10.0)
        od = c_o.number_input("Odd", min_value=1.01, value=1.5)
        res_opt = st.selectbox("Resultado", ["Green ✅", "Green (Cashout) 💰", "Red ❌", "Reembolso 🔄"])
        
        profit = (sk * od - sk) if "Green" in res_opt else (-sk if "Red" in res_opt else 0.0)
        if "Cashout" in res_opt: profit = st.number_input("Retorno") - sk
        
        st.write(f"Lucro: **R$ {profit:.2f}**")
        
        jogos = []
        for i in range(st.session_state['n_games']):
            st.markdown(f"**Jogo {i+1}**")
            c_m, c_v, c_merc = st.columns([2,2,2])
            jm = c_m.selectbox(f"M {i}", team_list_with_empty, key=f"nm_{i}")
            jv = c_v.selectbox(f"V {i}", team_list_with_empty, key=f"nv_{i}")
            jmerc = c_merc.text_input(f"Mercado {i}", key=f"nmerc_{i}")
            jogos.append({"Nome": f"{jm} x {jv}", "Selecoes": [{"Mercado": jmerc}]})
        
        c_b1, c_b2 = st.columns(2)
        if c_b1.button("➕ Jogo"): st.session_state['n_games'] += 1; st.rerun()
        if c_b2.button("💾 Salvar"):
            tk = {"id": str(uuid.uuid4())[:8], "Data": dt.strftime("%d/%m/%Y"), "Stake": sk, "Odd": od, "Lucro": profit, "Resultado": res_opt, "Jogos": jogos}
            salvar_ticket(tk)
            st.success("Salvo!"); st.session_state['n_games'] = 1; st.rerun()

    with tab_hist:
        for t in tickets:
            cls = "bet-card-green" if "Green" in t["Resultado"] else "bet-card-red" if "Red" in t["Resultado"] else "bet-card-cashout"
            st.markdown(f"""<div class="{cls}"><b>{t['Data']}</b> | R$ {t['Lucro']:.2f} | {t['Resultado']}</div>""", unsafe_allow_html=True)
            if st.button("🗑️", key=t['id']): excluding_ticket(t['id']); st.rerun()

    with tab_grf:
        if HAS_PLOTLY and tickets:
            df = pd.DataFrame(tickets)
            df['Data_Dt'] = pd.to_datetime(df['Data'], format="%d/%m/%Y")
            df = df.sort_values('Data_Dt')
            df['Acumulado'] = df['Lucro'].cumsum()
            st.plotly_chart(px.line(df, x='Data_Dt', y='Acumulado', markers=True), use_container_width=True)

if __name__ == "__main__":
    render_dashboard()
