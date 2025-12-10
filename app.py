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
st.set_page_config(page_title="FutPrevisão Pro V2.3", layout="wide", page_icon="⚽")

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
        df = pd.read_csv("arbitros.csv")
        return dict(zip(df['Nome'], df['Fator']))
    except:
        return {}

# --- B. Dados Históricos (TXT Adam Choi) ---
FILES_CONFIG = {
    "Premier League": {
        "corners": "Escanteios Preimier League - codigo fonte.txt",
        "cards": "Cartoes Premier League - Inglaterra.txt"
    },
    "La Liga": {
        "corners": "Escanteios Espanha.txt",
        "cards": "Cartoes La Liga - Espanha.txt"
    },
    "Serie A": {
        "corners": "Escanteios Italia.txt",
        "cards": "Cartoes Serie A - Italia.txt"
    },
    "Bundesliga": {
        "corners": "Escanteios Alemanha.txt",
        "cards": "Cartoes Bundesliga - Alemanha.txt"
    },
    "Ligue 1": {
        "corners": "Escanteios França.txt",
        "cards": "Cartoes Ligue 1 - França.txt"
    }
}

class AdamChoiLoader:
    def __init__(self):
        self.data_corners = {}
        self.data_cards = {}
        self.load_all_files()

    def load_all_files(self):
        pasta = Path(__file__).parent
        for liga, files in FILES_CONFIG.items():
            try:
                path_c = pasta / files["corners"]
                if path_c.exists():
                    with open(path_c, 'r', encoding='utf-8') as f:
                        raw = f.read().strip()
                        if '{' in raw: raw = raw[raw.find('{'):]
                        self.data_corners[liga] = json.loads(raw)
            except: pass
            try:
                path_k = pasta / files["cards"]
                if path_k.exists():
                    with open(path_k, 'r', encoding='utf-8') as f:
                        raw = f.read().strip()
                        if '{' in raw: raw = raw[raw.find('{'):]
                        self.data_cards[liga] = json.loads(raw)
            except: pass

    def get_history(self, team, league, market_type, key):
        source = self.data_corners if market_type == 'corners' else self.data_cards
        if league not in source: return None
        for t in source[league].get('teams', []):
            if t['teamName'] == team:
                stats = t.get(key)
                if stats and isinstance(stats, list) and len(stats) >= 3:
                    return stats[0], stats[1], stats[2]
        return None

    def get_history_global(self, team, market_type, key):
        """Busca o time em QUALQUER liga carregada (para Copas)"""
        source = self.data_corners if market_type == 'corners' else self.data_cards
        for league in source:
            for t in source[league].get('teams', []):
                if t['teamName'] == team:
                    stats = t.get(key)
                    if stats and isinstance(stats, list) and len(stats) >= 3:
                        return stats[0], stats[1], stats[2], league # Retorna também a liga de origem
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
    
    # Escanteios
    corn_h = (h_data['corners'] * 1.10) * f_h
    corn_a = (a_data['corners'] * 0.85) * f_a
    total_corners = corn_h + corn_a
    
    # Cartões
    tension_boost = 1.0
    if f_h > 1.05 or f_a > 1.05: tension_boost = 1.15
    avg_fouls = (h_data['fouls'] + a_data['fouls']) / 2
    tension = (avg_fouls / 12.0) * tension_boost
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
    
    st.markdown("""
    <style>
        .bet-card-green { background-color: #d4edda; border-left: 5px solid #28a745; padding: 15px; border-radius: 5px; margin-bottom: 10px; }
        .bet-card-red { background-color: #f8d7da; border-left: 5px solid #dc3545; padding: 15px; border-radius: 5px; margin-bottom: 10px; }
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

    tickets = carregar_tickets()
    lucro_total = sum(t["Lucro"] for t in tickets)
    banca_atual = cfg["banca_inicial"] + lucro_total
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Banca", f"R$ {banca_atual:,.2f}", f"{lucro_total:,.2f}")
    c2.metric("Bilhetes", len(tickets))
    
    # NOVAS ABAS: Incluindo Copas
    tab_ligas, tab_copas, tab_new, tab_hist, tab_grf = st.tabs(["🔮 Ligas", "🏆 Copas", "➕ Novo Bilhete", "📜 Histórico", "📈 Gráficos"])

    # --- ABA 1: SIMULAÇÃO LIGAS (ORIGINAL) ---
    with tab_ligas:
        st.markdown("### 🏟️ Análise de Ligas (Nacionais)")
        
        col_liga, col_arb = st.columns([2, 2])
        liga_sel = col_liga.selectbox("Liga", list(FILES_CONFIG.keys()), key="liga_nacional")
        
        lista_arbitros = sorted(list(referees_data.keys()))
        if not lista_arbitros: lista_arbitros = ["Genérico (Padrão)"]
        arbitro_sel = col_arb.selectbox("Árbitro", lista_arbitros, key="arb_nacional")
        ref_factor = referees_data.get(arbitro_sel, 1.0)

        col_h, col_a = st.columns(2)
        # Filtra times da liga se possível, senão mostra todos
        teams_liga = history_loader.data_corners.get(liga_sel, {}).get('teams', [])
        names_liga = sorted([t['teamName'] for t in teams_liga]) if teams_liga else team_list_raw
        
        home_team = col_h.selectbox("Mandante", names_liga, index=0, key="h_nac")
        away_team = col_a.selectbox("Visitante", names_liga, index=1 if len(names_liga)>1 else 0, key="a_nac")

        st.caption("Contexto")
        ctx_map = {"Neutro": 1.0, "Título 🏆": 1.15, "Z4 🔥": 1.10, "Rebaixado ❄️": 0.85}
        cc1, cc2 = st.columns(2)
        f_h = ctx_map[cc1.selectbox(f"Momento {home_team}", list(ctx_map.keys()), key="ctx_h_n")]
        f_a = ctx_map[cc2.selectbox(f"Momento {away_team}", list(ctx_map.keys()), key="ctx_a_n")]

        if st.button("🔮 Simular Liga", type="primary"):
            m = calcular_previsao(home_team, away_team, f_h, f_a, ref_factor)
            
            st.divider()
            c_cant, c_cart = st.columns(2)
            
            with c_cant:
                st.info("🚩 **Escanteios**")
                st.write(f"Exp. Total: **{m['corners']['t']:.2f}**")
                for l in [7.5, 8.5, 9.5]:
                    p = prob_over(m['corners']['t'], l)
                    st.write(f"+{l}: :{get_color(p)}[**{p:.0f}%**]")
                
                st.markdown("---")
                # Individuais
                st.write(f"**🏠 {home_team}**")
                for l, k in [(3.5, 'homeTeamOver35'), (4.5, 'homeTeamOver45'), (5.5, 'homeTeamOver55')]:
                    pm = prob_over(m['corners']['h'], l)
                    hist = history_loader.get_history(home_team, liga_sel, 'corners', k)
                    tr = f"({hist[1]}/{hist[0]} - {hist[2]})" if hist else ""
                    st.write(f"+{l}: :{get_color(pm)}[**{pm:.0f}%**] {tr}")
                
                st.write(f"**✈️ {away_team}**")
                for l, k in [(3.5, 'awayTeamOver35'), (4.5, 'awayTeamOver45'), (5.5, 'awayTeamOver55')]:
                    pm = prob_over(m['corners']['a'], l)
                    hist = history_loader.get_history(away_team, liga_sel, 'corners', k)
                    tr = f"({hist[1]}/{hist[0]} - {hist[2]})" if hist else ""
                    st.write(f"+{l}: :{get_color(pm)}[**{pm:.0f}%**] {tr}")

            with c_cart:
                st.warning("🟨 **Cartões**")
                st.write(f"Exp. Total: **{m['cards']['t']:.2f}**")
                for l in [2.5, 3.5, 4.5]:
                    p = prob_over(m['cards']['t'], l)
                    st.write(f"+{l}: :{get_color(p)}[**{p:.0f}%**]")
                
                st.markdown("---")
                st.write(f"**🏠 {home_team}**")
                for l, k in [(1.5, 'homeCardsOver15'), (2.5, 'homeCardsOver25')]:
                    pm = prob_over(m['cards']['h'], l)
                    hist = history_loader.get_history(home_team, liga_sel, 'cards', k)
                    tr = f"({hist[1]}/{hist[0]} - {hist[2]})" if hist else ""
                    st.write(f"+{l}: :{get_color(pm)}[**{pm:.0f}%**] {tr}")
                
                st.write(f"**✈️ {away_team}**")
                for l, k in [(1.5, 'awayCardsOver15'), (2.5, 'awayCardsOver25')]:
                    pm = prob_over(m['cards']['a'], l)
                    hist = history_loader.get_history(away_team, liga_sel, 'cards', k)
                    tr = f"({hist[1]}/{hist[0]} - {hist[2]})" if hist else ""
                    st.write(f"+{l}: :{get_color(pm)}[**{pm:.0f}%**] {tr}")

    # --- ABA 2: SIMULAÇÃO COPAS (NOVA) ---
    with tab_copas:
        st.markdown("### 🏆 Análise de Copas (Internacionais)")
        st.caption("Cruzamento de dados globais (Champions, Libertadores, etc.)")
        
        # Seleção de Árbitro (Igual)
        col_arb_c, col_vazia = st.columns([2, 2])
        arbitro_sel_c = col_arb_c.selectbox("Árbitro", lista_arbitros, key="arb_copa")
        ref_factor_c = referees_data.get(arbitro_sel_c, 1.0)

        # Seleção de Times (GLOBAL - Lista Completa)
        col_hc, col_ac = st.columns(2)
        home_team_c = col_hc.selectbox("Mandante", team_list_raw, index=0, key="h_copa")
        away_team_c = col_ac.selectbox("Visitante", team_list_raw, index=1, key="a_copa")

        st.caption("Contexto (Mata-mata)")
        cc1c, cc2c = st.columns(2)
        f_hc = ctx_map[cc1c.selectbox(f"Momento {home_team_c}", list(ctx_map.keys()), key="ctx_h_c")]
        f_ac = ctx_map[cc2c.selectbox(f"Momento {away_team_c}", list(ctx_map.keys()), key="ctx_a_c")]

        if st.button("🏆 Simular Copa", type="primary"):
            # Usa a mesma lógica matemática
            mc = calcular_previsao(home_team_c, away_team_c, f_hc, f_ac, ref_factor_c)
            
            st.divider()
            st.markdown(f"#### 🌍 {home_team_c} x {away_team_c}")
            
            col_cantosc, col_cartoesc = st.columns(2)
            
            with col_cantosc:
                st.info("🚩 **Escanteios (Global)**")
                st.write(f"Exp. Total: **{mc['corners']['t']:.2f}**")
                for l in [7.5, 8.5, 9.5]:
                    p = prob_over(mc['corners']['t'], l)
                    st.write(f"+{l}: :{get_color(p)}[**{p:.0f}%**]")
                
                st.markdown("---")
                
                # Busca Global (Varre todas as ligas para achar o time)
                st.write(f"**🏠 {home_team_c}**")
                for l, k in [(3.5, 'homeTeamOver35'), (4.5, 'homeTeamOver45'), (5.5, 'homeTeamOver55')]:
                    pm = prob_over(mc['corners']['h'], l)
                    # Get Global History
                    gh = history_loader.get_history_global(home_team_c, 'corners', k)
                    tr = f"({gh[1]}/{gh[0]} - {gh[2]} na {gh[3]})" if gh else "(Sem dados)"
                    st.write(f"+{l}: :{get_color(pm)}[**{pm:.0f}%**] {tr}")

                st.write(f"**✈️ {away_team_c}**")
                for l, k in [(3.5, 'awayTeamOver35'), (4.5, 'awayTeamOver45'), (5.5, 'awayTeamOver55')]:
                    pm = prob_over(mc['corners']['a'], l)
                    gh = history_loader.get_history_global(away_team_c, 'corners', k)
                    tr = f"({gh[1]}/{gh[0]} - {gh[2]} na {gh[3]})" if gh else "(Sem dados)"
                    st.write(f"+{l}: :{get_color(pm)}[**{pm:.0f}%**] {tr}")

            with col_cartoesc:
                st.warning("🟨 **Cartões (Global)**")
                st.write(f"Exp. Total: **{mc['cards']['t']:.2f}**")
                for l in [2.5, 3.5, 4.5]:
                    p = prob_over(mc['cards']['t'], l)
                    st.write(f"+{l}: :{get_color(p)}[**{p:.0f}%**]")
                
                st.markdown("---")
                st.write(f"**🏠 {home_team_c}**")
                for l, k in [(1.5, 'homeCardsOver15'), (2.5, 'homeCardsOver25')]:
                    pm = prob_over(mc['cards']['h'], l)
                    gh = history_loader.get_history_global(home_team_c, 'cards', k)
                    tr = f"({gh[1]}/{gh[0]} - {gh[2]})" if gh else ""
                    st.write(f"+{l}: :{get_color(pm)}[**{pm:.0f}%**] {tr}")

                st.write(f"**✈️ {away_team_c}**")
                for l, k in [(1.5, 'awayCardsOver15'), (2.5, 'awayCardsOver25')]:
                    pm = prob_over(mc['cards']['a'], l)
                    gh = history_loader.get_history_global(away_team_c, 'cards', k)
                    tr = f"({gh[1]}/{gh[0]} - {gh[2]})" if gh else ""
                    st.write(f"+{l}: :{get_color(pm)}[**{pm:.0f}%**] {tr}")

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
