import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import uuid
import math
import difflib
from pathlib import Path
from datetime import datetime

# ==============================================================================
# 0. CONFIGURAÇÃO INICIAL
# ==============================================================================
st.set_page_config(page_title="FutPrevisão Pro V3.1 (Scanner)", layout="wide", page_icon="⚽")

# Tenta importar Plotly
try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# ==============================================================================
# 1. FUNÇÕES MATEMÁTICAS
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
# 2. SEGURANÇA
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
# 3. CARREGAMENTO DE DADOS E MAPEAMENTO
# ==============================================================================

# --- Mapeamento de Nomes (CSV Calendário -> CSV Dados/TXT Adam Choi) ---
NAME_MAPPING = {
    # INGLATERRA
    "Man City": "Man City", "Manchester City": "Man City",
    "Man Utd": "Man United", "Man United": "Man United", "Manchester United": "Man United",
    "Forest": "Nott'm Forest", "Nott'm Forest": "Nott'm Forest", "Nottingham Forest": "Nott'm Forest",
    "Wolves": "Wolves", "Wolverhampton": "Wolves",
    "Sheffield Utd": "Sheffield Utd", "Sheffield United": "Sheffield Utd",
    
    # ESPANHA
    "Atl. Madrid": "Atl Madrid", "Atl Madrid": "Atl Madrid", "Atlético de Madrid": "Atl Madrid",
    "Athletic Club": "Athletic Club", "Athletic Bilbao": "Athletic Club", # CORREÇÃO AQUI
    "Real Betis": "Betis", "Betis": "Betis",
    "Real Madrid": "Real Madrid", 
    "Real Sociedad": "Real Sociedad",
    "Celta": "Celta", "Celta de Vigo": "Celta",
    "Rayo Vallecano": "Rayo Vallecano",
    "Alaves": "Alaves", "Alavés": "Alaves",
    "Real Oviedo": "Oviedo",
    
    # ITÁLIA
    "Inter": "Inter", "Inter de Milão": "Inter",
    "Milan": "Milan", "AC Milan": "Milan",
    "Roma": "Roma", "AS Roma": "Roma",
    "Verona": "Hellas Verona", "Hellas Verona": "Hellas Verona",
    
    # ALEMANHA
    "Leverkusen": "Bayer 04 Leverkusen", "Bayer Leverkusen": "Bayer 04 Leverkusen", # CORREÇÃO AQUI
    "Bayern": "Bayern Munich", "Bayern Munich": "Bayern Munich", "Bayern de Munique": "Bayern Munich",
    "Dortmund": "Dortmund", "Borussia Dortmund": "Dortmund",
    "M'gladbach": "Gladbach", "Gladbach": "Gladbach", "Borussia M'Gladbach": "Gladbach",
    "Frankfurt": "Eintracht Frankfurt", "Eintracht Frankfurt": "Eintracht Frankfurt",
    "Mainz 05": "Mainz", "Mainz": "Mainz",
    "Bremen": "Werder Bremen", "Werder Bremen": "Werder Bremen",
    "HSV": "Hamburg", "Hamburg": "Hamburg",
    
    # FRANÇA
    "PSG": "Paris SG", "Paris Saint-Germain": "Paris SG", # CORREÇÃO AQUI
    "St Etienne": "St Etienne", "Saint-Etienne": "St Etienne",
    "Le Havre": "Le Havre"
}

# --- Dados Matemáticos ---
BACKUP_TEAMS = {"Arsenal": {"corners": 6.0, "cards": 1.5, "fouls": 10.0, "goals_f": 1.5, "goals_a": 1.0}}

def safe_float(value):
    try: return float(str(value).replace(',', '.'))
    except: return 0.0

@st.cache_data(ttl=3600)
def load_csv_data():
    try:
        # Tenta ler com diferentes codificações
        try: df = pd.read_csv("dados_times.csv", encoding='utf-8')
        except: df = pd.read_csv("dados_times.csv", encoding='latin1', sep=';')
        
        teams_dict = {}
        df.columns = [c.strip() for c in df.columns]
        for _, row in df.iterrows():
            if 'Time' in row:
                t_name = str(row['Time']).strip()
                teams_dict[t_name] = {
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
    except: return {}

# --- Dados Históricos ---
FILES_CONFIG = {
    "Premier League": {"corners": "Escanteios Preimier League - codigo fonte.txt", "cards": "Cartoes Premier League - Inglaterra.txt"},
    "La Liga": {"corners": "Escanteios Espanha.txt", "cards": "Cartoes La Liga - Espanha.txt"},
    "Serie A": {"corners": "Escanteios Italia.txt", "cards": "Cartoes Serie A - Italia.txt"},
    "Bundesliga": {"corners": "Escanteios Alemanha.txt", "cards": "Cartoes Bundesliga - Alemanha.txt"},
    "Ligue 1": {"corners": "Escanteios França.txt", "cards": "Cartoes Ligue 1 - França.txt"}
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
                p = pasta / files["corners"]
                if p.exists():
                    with open(p, 'r', encoding='utf-8') as f:
                        raw = f.read().strip()
                        if '{' in raw: raw = raw[raw.find('{'):]
                        self.data_corners[liga] = json.loads(raw)
            except: pass
            try:
                p = pasta / files["cards"]
                if p.exists():
                    with open(p, 'r', encoding='utf-8') as f:
                        raw = f.read().strip()
                        if '{' in raw: raw = raw[raw.find('{'):]
                        self.data_cards[liga] = json.loads(raw)
            except: pass

    def find_best_match(self, target_name, available_names):
        target = target_name.strip()
        
        # 1. Mapeamento
        if target in NAME_MAPPING:
            target = NAME_MAPPING[target]
            
        target_lower = target.lower()
        
        # 2. Exato
        for name in available_names:
            if name.lower() == target_lower: return name
            
        # 3. Substring
        for name in available_names:
            nl = name.lower()
            if len(target_lower)>3 and len(nl)>3:
                if target_lower in nl or nl in target_lower: return name
                
        # 4. Fuzzy
        matches = difflib.get_close_matches(target, available_names, n=1, cutoff=0.6)
        return matches[0] if matches else None

    def get_history(self, team, league, market_type, key):
        source = self.data_corners if market_type == 'corners' else self.data_cards
        # Se liga não especificada ou não encontrada, busca em todas (fallback)
        leagues_to_search = [league] if league in source else list(source.keys())
        
        for l in leagues_to_search:
            avail = [t['teamName'] for t in source[l].get('teams', [])]
            matched = self.find_best_match(team, avail)
            if matched:
                for t in source[l]['teams']:
                    if t['teamName'] == matched:
                        stats = t.get(key)
                        if stats and isinstance(stats, list) and len(stats) >= 3:
                            return stats[0], stats[1], stats[2]
        return None

# Inicializa
teams_data = load_csv_data()
referees_data = load_referees()
team_list_raw = sorted(list(teams_data.keys()))
team_list_with_empty = [""] + team_list_raw
history_loader = AdamChoiLoader()

# ==============================================================================
# 4. CALENDÁRIO SCANNER
# ==============================================================================
def load_calendar():
    calendar_files = ["premier_league.csv", "la_liga.csv", "serie_a.csv", "bundesliga.csv", "ligue_1.csv"]
    all_games = []
    
    for f in calendar_files:
        try:
            if os.path.exists(f):
                df = pd.read_csv(f)
                # Padroniza colunas
                df.columns = [c.strip() for c in df.columns]
                all_games.append(df)
        except: pass
        
    if all_games:
        full_df = pd.concat(all_games, ignore_index=True)
        # Converte data para datetime para filtrar
        try:
            full_df['Data_Dt'] = pd.to_datetime(full_df['Data'], format="%d/%m/%Y", dayfirst=True)
        except:
            pass # Se falhar, mantém string para debug
        return full_df
    return pd.DataFrame()

# ==============================================================================
# 5. LÓGICA DE PREVISÃO
# ==============================================================================
def normalize_team_name_for_math(name):
    # Tenta achar a chave correta no dados_times.csv
    # Usa a mesma lógica de mapeamento reverso ou fuzzy
    if name in teams_data: return name
    if name in NAME_MAPPING:
        mapped = NAME_MAPPING[name]
        # O mapeamento aponta para o nome do Adam Choi, mas as vezes o CSV tem outro
        # Vamos tentar achar no CSV
        matches = difflib.get_close_matches(mapped, teams_data.keys(), n=1, cutoff=0.6)
        if matches: return matches[0]
    
    # Tenta fuzzy direto
    matches = difflib.get_close_matches(name, teams_data.keys(), n=1, cutoff=0.6)
    return matches[0] if matches else "Arsenal" # Fallback seguro

def calcular_previsao(home, away, f_h=1.0, f_a=1.0, ref_factor=1.0):
    h_key = normalize_team_name_for_math(home)
    a_key = normalize_team_name_for_math(away)
    
    h_data = teams_data.get(h_key, BACKUP_TEAMS["Arsenal"])
    a_data = teams_data.get(a_key, BACKUP_TEAMS["Arsenal"])
    
    # Cantos
    corn_h = (h_data['corners'] * 1.10) * f_h
    corn_a = (a_data['corners'] * 0.85) * f_a
    total_corners = corn_h + corn_a
    
    # Cartões
    avg_fouls = (h_data['fouls'] + a_data['fouls']) / 2
    tension = avg_fouls / 12.0
    # Boost de tensão se contexto alto
    if f_h > 1.05 or f_a > 1.05: tension *= 1.15
        
    card_h = h_data['cards'] * tension * ref_factor
    card_a = a_data['cards'] * tension * ref_factor
    
    return {
        "corners": {"t": total_corners, "h": corn_h, "a": corn_a},
        "cards": {"t": card_h+card_a, "h": card_h, "a": card_a}
    }

# ==============================================================================
# 6. GESTÃO DE BILHETES
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
# 7. DASHBOARD
# ==============================================================================
def render_dashboard():
    st.title("📊 Painel de Controle V3.0")
    
    st.markdown("""
    <style>
        .bet-card-green { background-color: #d4edda; border-left: 5px solid #28a745; padding: 15px; border-radius: 5px; margin-bottom: 10px; }
        .bet-card-red { background-color: #f8d7da; border-left: 5px solid #dc3545; padding: 15px; border-radius: 5px; margin-bottom: 10px; }
        .scan-card { background: #f0f8ff; border: 1px solid #bce8f1; padding: 10px; border-radius: 5px; margin-bottom: 8px; }
        .scan-high { border-left: 5px solid #28a745; } /* Alta Confiança */
        .scan-med { border-left: 5px solid #ffc107; } /* Média Confiança */
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar Config
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
    
    # TABS
    tab_scan, tab_sim, tab_new, tab_hist, tab_grf = st.tabs(["🔍 Scanner (Novo)", "🔮 Simulação Manual", "➕ Bilhete", "📜 Histórico", "📈 Gráficos"])

    # --- ABA SCANNER (A NOVIDADE V3.0) ---
    with tab_scan:
        st.markdown("### 🔍 Scanner de Oportunidades")
        st.caption("Varre os jogos do calendário e encontra valor.")
        
        cal_df = load_calendar()
        if cal_df.empty:
            st.warning("Nenhum arquivo de calendário encontrado (.csv).")
        else:
            # Filtro de Data
            dias_disponiveis = sorted(cal_df['Data'].unique())
            dia_selecionado = st.selectbox("Selecione a Data:", dias_disponiveis, index=0)
            
            jogos_do_dia = cal_df[cal_df['Data'] == dia_selecionado]
            st.info(f"Jogos encontrados para {dia_selecionado}: **{len(jogos_do_dia)}**")
            
            if st.button("🚀 ESCANEAR JOGOS DO DIA", type="primary"):
                results = []
                
                progress_bar = st.progress(0)
                total_j = len(jogos_do_dia)
                
                for idx, row in jogos_do_dia.iterrows():
                    mandante = row['Mandante']
                    visitante = row['Visitante']
                    liga = row.get('Liga', 'Premier League') # Default se vazio
                    
                    # Roda Previsão (Contexto Neutro para Scanner)
                    m = calcular_previsao(mandante, visitante)
                    
                    # --- CHECAGENS ---
                    
                    # 1. Escanteios Casa +3.5
                    prob_math = prob_over(m['corners']['h'], 3.5)
                    hist = history_loader.get_history(mandante, liga, 'corners', 'homeTeamOver35')
                    if hist and prob_math > 65 and float(hist[2]) > 70:
                        results.append({"Jogo": f"{mandante} x {visitante}", "Aposta": f"{mandante} +3.5 Cantos", "Confiança": "Alta", "Math": f"{prob_math:.0f}%", "Real": f"{hist[2]}% ({hist[1]}/{hist[0]})"})
                    
                    # 2. Escanteios Fora +3.5
                    prob_math = prob_over(m['corners']['a'], 3.5)
                    hist = history_loader.get_history(visitante, liga, 'corners', 'awayTeamOver35')
                    if hist and prob_math > 65 and float(hist[2]) > 70:
                        results.append({"Jogo": f"{mandante} x {visitante}", "Aposta": f"{visitante} +3.5 Cantos", "Confiança": "Alta", "Math": f"{prob_math:.0f}%", "Real": f"{hist[2]}% ({hist[1]}/{hist[0]})"})

                    # 3. Cartões Casa +1.5
                    prob_math = prob_over(m['cards']['h'], 1.5)
                    hist = history_loader.get_history(mandante, liga, 'cards', 'homeCardsOver15')
                    if hist and prob_math > 60 and float(hist[2]) > 75:
                        results.append({"Jogo": f"{mandante} x {visitante}", "Aposta": f"{mandante} +1.5 Cartões", "Confiança": "Média", "Math": f"{prob_math:.0f}%", "Real": f"{hist[2]}% ({hist[1]}/{hist[0]})"})

                    # 4. Cartões Fora +1.5
                    prob_math = prob_over(m['cards']['a'], 1.5)
                    hist = history_loader.get_history(visitante, liga, 'cards', 'awayCardsOver15')
                    if hist and prob_math > 60 and float(hist[2]) > 75:
                        results.append({"Jogo": f"{mandante} x {visitante}", "Aposta": f"{visitante} +1.5 Cartões", "Confiança": "Média", "Math": f"{prob_math:.0f}%", "Real": f"{hist[2]}% ({hist[1]}/{hist[0]})"})

                    progress_bar.progress((idx + 1) / total_j)
                
                st.write("---")
                if results:
                    st.success(f"Encontrei {len(results)} oportunidades de valor!")
                    for res in results:
                        css = "scan-high" if res["Confiança"] == "Alta" else "scan-med"
                        st.markdown(f"""
                        <div class="scan-card {css}">
                            <strong>⚽ {res['Jogo']}</strong><br>
                            👉 {res['Aposta']} | 📊 Math: {res['Math']} | 📈 Real: {res['Real']}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.warning("Nenhuma oportunidade de ALTA confiança encontrada hoje. Tente simular manualmente.")

    # --- ABA SIMULAÇÃO MANUAL (ANTIGA ABA LIGAS/COPAS UNIFICADA) ---
    with tab_sim:
        st.markdown("### 🔮 Simulação Manual")
        
        c1, c2 = st.columns(2)
        home = c1.selectbox("Mandante", team_list_raw, index=0, key="man_h")
        away = c2.selectbox("Visitante", team_list_raw, index=1, key="man_a")
        
        c3, c4, c5 = st.columns(3)
        liga_sel = c3.selectbox("Liga (Dados)", list(FILES_CONFIG.keys()), key="man_lig")
        arb_sel = c4.selectbox("Árbitro", sorted(list(referees_data.keys())) or ["Genérico"], key="man_arb")
        
        ctx_map = {"Neutro": 1.0, "Título 🏆": 1.15, "Z4 🔥": 1.10, "Rebaixado ❄️": 0.85}
        f_h = ctx_map[c5.selectbox("Momento Jogo", list(ctx_map.keys()), key="man_ctx")]
        
        if st.button("🔮 Analisar", type="primary"):
            ref_f = referees_data.get(arb_sel, 1.0)
            m = calcular_previsao(home, away, f_h, f_h, ref_f) # Contexto simétrico para simplificar ou criar 2 inputs
            
            st.divider()
            col_cant, col_cart = st.columns(2)
            
            with col_cant:
                st.info("🚩 **Escanteios**")
                st.write(f"Exp Total: **{m['corners']['t']:.2f}**")
                for l in [8.5, 9.5, 10.5]:
                    p = prob_over(m['corners']['t'], l)
                    st.write(f"+{l}: :{get_color(p)}[**{p:.0f}%**]")
                
                st.markdown("---")
                # Individuais com Busca Inteligente
                st.write(f"**🏠 {home}**")
                for l, k in [(3.5, 'homeTeamOver35'), (4.5, 'homeTeamOver45')]:
                    pm = prob_over(m['corners']['h'], l)
                    hist = history_loader.get_history(home, liga_sel, 'corners', k)
                    # Se não achar na liga selecionada, tenta global
                    if not hist: hist = history_loader.get_history_global(home, 'corners', k)
                    
                    tr = f"({hist[1]}/{hist[0]} - {hist[2]})" if hist else "(Sem dados)"
                    st.write(f"+{l}: :{get_color(pm)}[**{pm:.0f}%**] {tr}")
                
                st.write(f"**✈️ {away}**")
                for l, k in [(3.5, 'awayTeamOver35'), (4.5, 'awayTeamOver45')]:
                    pm = prob_over(m['corners']['a'], l)
                    hist = history_loader.get_history(away, liga_sel, 'corners', k)
                    if not hist: hist = history_loader.get_history_global(away, 'corners', k)
                    
                    tr = f"({hist[1]}/{hist[0]} - {hist[2]})" if hist else "(Sem dados)"
                    st.write(f"+{l}: :{get_color(pm)}[**{pm:.0f}%**] {tr}")

            with col_cart:
                st.warning("🟨 **Cartões**")
                st.write(f"Exp Total: **{m['cards']['t']:.2f}**")
                for l in [3.5, 4.5]:
                    p = prob_over(m['cards']['t'], l)
                    st.write(f"+{l}: :{get_color(p)}[**{p:.0f}%**]")
                
                st.markdown("---")
                st.write(f"**🏠 {home}**")
                for l, k in [(1.5, 'homeCardsOver15')]:
                    pm = prob_over(m['cards']['h'], l)
                    hist = history_loader.get_history(home, liga_sel, 'cards', k)
                    if not hist: hist = history_loader.get_history_global(home, 'cards', k)
                    tr = f"({hist[1]}/{hist[0]} - {hist[2]})" if hist else "(Sem dados)"
                    st.write(f"+{l}: :{get_color(pm)}[**{pm:.0f}%**] {tr}")
                
                st.write(f"**✈️ {away}**")
                for l, k in [(1.5, 'awayCardsOver15')]:
                    pm = prob_over(m['cards']['a'], l)
                    hist = history_loader.get_history(away, liga_sel, 'cards', k)
                    if not hist: hist = history_loader.get_history_global(away, 'cards', k)
                    tr = f"({hist[1]}/{hist[0]} - {hist[2]})" if hist else "(Sem dados)"
                    st.write(f"+{l}: :{get_color(pm)}[**{pm:.0f}%**] {tr}")

    # --- ABAS DE GESTÃO (MANTIDAS) ---
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
        if st.button("💾 Salvar Bilhete"):
            tk = {"id": str(uuid.uuid4())[:8], "Data": dt.strftime("%d/%m/%Y"), "Stake": sk, "Odd": od, "Lucro": profit, "Resultado": res_opt, "Jogos": []}
            salvar_ticket(tk)
            st.success("Salvo!")

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