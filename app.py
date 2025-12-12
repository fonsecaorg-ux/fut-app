import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import uuid
import math
import difflib
import random
from pathlib import Path
from datetime import datetime

# ==============================================================================
# 0. CONFIGURAÇÃO INICIAL
# ==============================================================================
st.set_page_config(page_title="FutPrevisão Pro V4.1 (Turbo)", layout="wide", page_icon="⚽")

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
# 3. CARREGAMENTO DE DADOS
# ==============================================================================

NAME_MAPPING = {
    # INGLATERRA
    "Man City": "Man City", "Manchester City": "Man City",
    "Man Utd": "Man United", "Man United": "Man United", "Manchester United": "Man United",
    "Forest": "Nott'm Forest", "Nott'm Forest": "Nott'm Forest", "Nottingham Forest": "Nott'm Forest",
    "Wolves": "Wolves", "Wolverhampton": "Wolves",
    "Sheffield Utd": "Sheffield Utd", "Sheffield United": "Sheffield Utd",
    
    # ESPANHA
    "Atl. Madrid": "Atl Madrid", "Atl Madrid": "Atl Madrid", "Atlético de Madrid": "Atl Madrid",
    "Athletic Club": "Athletic Club", "Athletic Bilbao": "Athletic Club",
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
    "Leverkusen": "Bayer 04 Leverkusen", "Bayer Leverkusen": "Bayer 04 Leverkusen",
    "Bayern": "Bayern Munich", "Bayern Munich": "Bayern Munich", "Bayern de Munique": "Bayern Munich",
    "Dortmund": "Dortmund", "Borussia Dortmund": "Dortmund",
    "M'gladbach": "Gladbach", "Gladbach": "Gladbach", "Borussia M'Gladbach": "Gladbach",
    "Frankfurt": "Eintracht Frankfurt", "Eintracht Frankfurt": "Eintracht Frankfurt",
    "Mainz 05": "Mainz", "Mainz": "Mainz",
    "Bremen": "Werder Bremen", "Werder Bremen": "Werder Bremen",
    "HSV": "Hamburg", "Hamburg": "Hamburg",
    
    # FRANÇA
    "PSG": "Paris SG", "Paris Saint-Germain": "Paris SG",
    "St Etienne": "St Etienne", "Saint-Etienne": "St Etienne",
    "Le Havre": "Le Havre"
}

BACKUP_TEAMS = {"Arsenal": {"corners": 6.0, "cards": 1.5, "fouls": 10.0, "goals_f": 1.5, "goals_a": 1.0}}

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
            # Carregamento silencioso (sem erros na tela)
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
        if target in NAME_MAPPING: target = NAME_MAPPING[target]
        target_lower = target.lower()
        
        for name in available_names:
            if name.lower() == target_lower: return name
        for name in available_names:
            nl = name.lower()
            if len(target_lower)>3 and len(nl)>3:
                if target_lower in nl or nl in target_lower: return name
        matches = difflib.get_close_matches(target, available_names, n=1, cutoff=0.6)
        return matches[0] if matches else None

    def get_history(self, team, league, market_type, key):
        source = self.data_corners if market_type == 'corners' else self.data_cards
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

    def get_history_global(self, team, market_type, key):
        source = self.data_corners if market_type == 'corners' else self.data_cards
        for league in source:
            avail = [t['teamName'] for t in source[league].get('teams', [])]
            matched = self.find_best_match(team, avail)
            if matched:
                for t in source[league]['teams']:
                    if t['teamName'] == matched:
                        stats = t.get(key)
                        if stats and isinstance(stats, list) and len(stats) >= 3:
                            return stats[0], stats[1], stats[2], league
        return None

teams_data = load_csv_data()
referees_data = load_referees()
team_list_raw = sorted(list(teams_data.keys()))
team_list_with_empty = [""] + team_list_raw
history_loader = AdamChoiLoader()

# ==============================================================================
# 4. CALENDÁRIO & LÓGICA DE PREVISÃO
# ==============================================================================
def load_calendar():
    calendar_files = ["premier_league.csv", "la_liga.csv", "serie_a.csv", "bundesliga.csv", "ligue_1.csv"]
    all_games = []
    for f in calendar_files:
        try:
            if os.path.exists(f):
                df = pd.read_csv(f, dtype=str)
                df.columns = [c.strip() for c in df.columns]
                all_games.append(df)
        except: pass
    if all_games:
        full_df = pd.concat(all_games, ignore_index=True)
        full_df['Data'] = full_df['Data'].str.strip()
        return full_df
    return pd.DataFrame()

def normalize_team_name_for_math(name):
    if name in teams_data: return name
    if name in NAME_MAPPING:
        mapped = NAME_MAPPING[name]
        matches = difflib.get_close_matches(mapped, teams_data.keys(), n=1, cutoff=0.6)
        if matches: return matches[0]
    matches = difflib.get_close_matches(name, teams_data.keys(), n=1, cutoff=0.6)
    return matches[0] if matches else "Arsenal"

def calcular_previsao(home, away, f_h=1.0, f_a=1.0, ref_factor=1.0):
    h_key = normalize_team_name_for_math(home)
    a_key = normalize_team_name_for_math(away)
    
    h_data = teams_data.get(h_key, BACKUP_TEAMS["Arsenal"])
    a_data = teams_data.get(a_key, BACKUP_TEAMS["Arsenal"])
    
    corn_h = (h_data['corners'] * 1.10) * f_h
    corn_a = (a_data['corners'] * 0.85) * f_a
    total_corners = corn_h + corn_a
    
    avg_fouls = (h_data['fouls'] + a_data['fouls']) / 2
    tension = avg_fouls / 12.0
    if f_h > 1.05 or f_a > 1.05: tension *= 1.15
        
    card_h = h_data['cards'] * tension * ref_factor
    card_a = a_data['cards'] * tension * ref_factor
    
    return {
        "corners": {"t": total_corners, "h": corn_h, "a": corn_a},
        "cards": {"t": card_h+card_a, "h": card_h, "a": card_a}
    }

# ==============================================================================
# 5. GERADOR DE MÚLTIPLAS TURBO (ODD > 1.70)
# ==============================================================================
def gerar_multiplas(oportunidades):
    if not oportunidades: return []
    
    # 1. Filtra apenas as melhores (Alta confiança)
    melhores = [op for op in oportunidades if op['Conf'] == 'Alta']
    
    # Se não tiver "Alta" suficiente, pega as médias também
    if len(melhores) < 4:
        melhores = oportunidades
        
    por_jogo = {}
    for op in melhores:
        jogo = op['Jogo']
        if jogo not in por_jogo: por_jogo[jogo] = []
        por_jogo[jogo].append(op)
    
    jogos_unicos = list(por_jogo.keys())
    random.shuffle(jogos_unicos)
    
    bilhetes = []
    
    # Gera 6 Bilhetes Triplos ou Duplos Fortes
    for i in range(6):
        # Tenta pegar 3 jogos diferentes para aumentar a Odd
        selecao_bilhete = []
        
        # Pega até 3 jogos sem repetir
        for _ in range(3):
            if jogos_unicos:
                jogo = jogos_unicos.pop(0)
                selecao_bilhete.append(por_jogo[jogo][0]) # Pega a melhor desse jogo
                jogos_unicos.append(jogo) # Devolve para o final (circular)
        
        # Só adiciona se tiver pelo menos 2 jogos
        if len(selecao_bilhete) >= 2:
            bilhetes.append(selecao_bilhete)
            random.shuffle(jogos_unicos) # Reembaralha para o próximo bilhete ser diferente
            
    return bilhetes

# ==============================================================================
# 6. GESTÃO DE BILHETES & DASHBOARD
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

def render_dashboard():
    st.title("📊 Painel de Controle V4.1")
    
    st.markdown("""
    <style>
        .bet-card-green { background-color: #d4edda; border-left: 5px solid #28a745; padding: 15px; border-radius: 5px; margin-bottom: 10px; }
        .bet-card-red { background-color: #f8d7da; border-left: 5px solid #dc3545; padding: 15px; border-radius: 5px; margin-bottom: 10px; }
        .scan-card { background: #f0f8ff; border: 1px solid #bce8f1; padding: 10px; border-radius: 5px; margin-bottom: 8px; }
        .scan-high { border-left: 5px solid #28a745; }
        .scan-med { border-left: 5px solid #ffc107; }
        .multi-card { background: #fff3cd; border: 1px solid #ffeeba; padding: 15px; border-radius: 10px; margin-bottom: 15px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); }
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
    
    tab_scan, tab_sim, tab_new, tab_hist, tab_grf = st.tabs(["🔍 Scanner", "🔮 Simulação", "➕ Bilhete", "📜 Histórico", "📈 Gráficos"])

    # --- ABA SCANNER ---
    with tab_scan:
        st.markdown("### 🔍 Scanner de Oportunidades")
        cal_df = load_calendar()
        
        if cal_df.empty:
            st.warning("Nenhum calendário encontrado.")
        else:
            dias = sorted(cal_df['Data'].unique())
            hoje_str = datetime.now().strftime("%d/%m/%Y")
            idx_hoje = dias.index(hoje_str) if hoje_str in dias else 0
            
            dia_sel = st.selectbox("Data:", dias, index=idx_hoje)
            jogos = cal_df[cal_df['Data'] == dia_sel]
            
            st.info(f"{len(jogos)} jogos em {dia_sel}")
            
            if 'scan_results' not in st.session_state: st.session_state['scan_results'] = []

            if st.button("ESCANEAR JOGOS DO DIA", type="primary"):
                res = []
                bar = st.progress(0)
                
                for i, (_, row) in enumerate(jogos.iterrows()):
                    m_time = row['Mandante']
                    v_time = row['Visitante']
                    liga = row.get('Liga', 'Premier League')
                    
                    calc = calcular_previsao(m_time, v_time)
                    
                    # --- FILTROS INTELIGENTES (LINHAS MISTAS PARA ODD MELHOR) ---
                    
                    # 1. Escanteios Casa (+3.5 e +4.5 se for muito bom)
                    hist_45 = history_loader.get_history(m_time, liga, 'corners', 'homeTeamOver45')
                    if not hist_45: hist_45 = history_loader.get_history_global(m_time, 'corners', 'homeTeamOver45')
                    
                    hist_35 = history_loader.get_history(m_time, liga, 'corners', 'homeTeamOver35')
                    if not hist_35: hist_35 = history_loader.get_history_global(m_time, 'corners', 'homeTeamOver35')

                    # Prioriza linha 4.5 se for boa (ODD MAIOR)
                    if hist_45 and float(hist_45[2]) > 70:
                         res.append({"Jogo": f"{m_time} x {v_time}", "Aposta": f"🏠 {m_time} +4.5 Cantos", "Conf": "Alta", "M": "--", "R": f"{hist_45[2]}%"})
                    elif hist_35 and float(hist_35[2]) > 75:
                         res.append({"Jogo": f"{m_time} x {v_time}", "Aposta": f"🏠 {m_time} +3.5 Cantos", "Conf": "Alta", "M": "--", "R": f"{hist_35[2]}%"})

                    # 2. Escanteios Fora
                    hist_a_45 = history_loader.get_history(v_time, liga, 'corners', 'awayTeamOver45')
                    if not hist_a_45: hist_a_45 = history_loader.get_history_global(v_time, 'corners', 'awayTeamOver45')
                    
                    hist_a_35 = history_loader.get_history(v_time, liga, 'corners', 'awayTeamOver35')
                    if not hist_a_35: hist_a_35 = history_loader.get_history_global(v_time, 'corners', 'awayTeamOver35')

                    if hist_a_45 and float(hist_a_45[2]) > 70:
                         res.append({"Jogo": f"{m_time} x {v_time}", "Aposta": f"✈️ {v_time} +4.5 Cantos", "Conf": "Alta", "M": "--", "R": f"{hist_a_45[2]}%"})
                    elif hist_a_35 and float(hist_a_35[2]) > 75:
                         res.append({"Jogo": f"{m_time} x {v_time}", "Aposta": f"✈️ {v_time} +3.5 Cantos", "Conf": "Alta", "M": "--", "R": f"{hist_a_35[2]}%"})

                    # 3. Cartões (Mantém +1.5 se muito forte)
                    hist_c = history_loader.get_history(m_time, liga, 'cards', 'homeCardsOver15')
                    if not hist_c: hist_c = history_loader.get_history_global(m_time, 'cards', 'homeCardsOver15')
                    
                    if hist_c and float(hist_c[2]) > 80:
                        res.append({"Jogo": f"{m_time} x {v_time}", "Aposta": f"🏠 {m_time} +1.5 Cartões", "Conf": "Média", "M": "--", "R": f"{hist_c[2]}%"})

                    bar.progress((i+1)/len(jogos))
                
                st.session_state['scan_results'] = res
                
            if st.session_state['scan_results']:
                st.markdown("---")
                
                # GERADOR DE MÚLTIPLAS
                st.subheader("🎫 Múltiplas Turbo (Odds Estimadas 1.80+)")
                
                if st.button("🔄 Gerar 6 Bilhetes Triplos"):
                    st.session_state['multiplas_geradas'] = gerar_multiplas(st.session_state['scan_results'])
                
                if 'multiplas_geradas' in st.session_state and st.session_state['multiplas_geradas']:
                    cols = st.columns(3)
                    for i, bilhete in enumerate(st.session_state['multiplas_geradas']):
                        idx_col = i % 3
                        with cols[idx_col]:
                            st.markdown(f"""
                            <div class="multi-card">
                                <strong>🎟️ BILHETE {i+1}</strong><hr style="margin:5px 0">
                                1. {bilhete[0]['Aposta']} <small>({bilhete[0]['R']})</small><br>
                                2. {bilhete[1]['Aposta']} <small>({bilhete[1]['R']})</small><br>
                                {f"3. {bilhete[2]['Aposta']} <small>({bilhete[2]['R']})</small>" if len(bilhete) > 2 else ""}
                            </div>
                            """, unsafe_allow_html=True)
                
                st.markdown("---")
                st.markdown("#### 📋 Lista Geral")
                for r in st.session_state['scan_results']:
                    css = "scan-high" if r["Conf"] == "Alta" else "scan-med"
                    st.markdown(f"""<div class="scan-card {css}"><b>{r['Jogo']}</b><br>👉 {r['Aposta']} | R: {r['R']}</div>""", unsafe_allow_html=True)
            else:
                st.write("Aguardando scan...")

    # --- ABA SIMULAÇÃO MANUAL (MANTIDA) ---
    with tab_sim:
        st.markdown("### 🔮 Simulação Manual")
        c1, c2 = st.columns(2)
        home = c1.selectbox("Mandante", team_list_raw, 0)
        away = c2.selectbox("Visitante", team_list_raw, 1)
        c3, c4 = st.columns(2)
        liga = c3.selectbox("Liga", list(FILES_CONFIG.keys()))
        arb = c4.selectbox("Árbitro", sorted(list(referees_data.keys())) or ["Genérico"])
        
        if st.button("Analisar"):
            m = calcular_previsao(home, away, ref_factor=referees_data.get(arb, 1.0))
            st.divider()
            c1, c2 = st.columns(2)
            with c1:
                st.info(f"Escanteios (Exp: {m['corners']['t']:.2f})")
                st.write(f"**🏠 {home}**")
                for l in [3.5, 4.5]:
                    pm = prob_over(m['corners']['h'], l)
                    h = history_loader.get_history(home, liga, 'corners', f'homeTeamOver{str(l).replace(".","")}')
                    if not h: h = history_loader.get_history_global(home, 'corners', f'homeTeamOver{str(l).replace(".","")}')
                    tr = f"({h[1]}/{h[0]} - {h[2]}%)" if h else ""
                    st.write(f"+{l}: :{get_color(pm)}[**{pm:.0f}%**] {tr}")
            with c2:
                st.warning(f"Cartões (Exp: {m['cards']['t']:.2f})")
                st.write(f"**✈️ {away}**")
                for l in [1.5, 2.5]:
                    pm = prob_over(m['cards']['a'], l)
                    h = history_loader.get_history(away, liga, 'cards', f'awayCardsOver{str(l).replace(".","")}')
                    if not h: h = history_loader.get_history_global(away, 'cards', f'awayCardsOver{str(l).replace(".","")}')
                    tr = f"({h[1]}/{h[0]} - {h[2]}%)" if h else ""
                    st.write(f"+{l}: :{get_color(pm)}[**{pm:.0f}%**] {tr}")

    # --- ABA BILHETES ---
    with tab_new:
        st.markdown("### Novo Bilhete")
        if 'n_games' not in st.session_state: st.session_state['n_games'] = 1
        col1, col2, col3 = st.columns(3)
        dt = col1.date_input("Data")
        stake = col2.number_input("Stake", 10.0)
        odd = col3.number_input("Odd", 1.5)
        res = st.selectbox("Resultado", ["Green ✅", "Red ❌", "Cashout 💰"])
        profit = (stake*odd-stake) if "Green" in res else (-stake if "Red" in res else 0.0)
        st.write(f"Lucro: **{profit:.2f}**")
        if st.button("Salvar"):
            salvar_ticket({"id": str(uuid.uuid4())[:8], "Data": dt.strftime("%d/%m/%Y"), "Stake": stake, "Odd": odd, "Lucro": profit, "Resultado": res, "Jogos": []})
            st.success("Salvo!")

    with tab_hist:
        for t in tickets:
            cls = "bet-card-green" if "Green" in t["Resultado"] else "bet-card-red"
            st.markdown(f"""<div class="{cls}">{t['Data']} | {t['Lucro']:.2f}</div>""", unsafe_allow_html=True)
            if st.button("🗑️", key=t['id']): excluding_ticket(t['id']); st.rerun()

    with tab_grf:
        if HAS_PLOTLY and tickets:
            df = pd.DataFrame(tickets)
            df['Data_Dt'] = pd.to_datetime(df['Data'], format="%d/%m/%Y")
            df = df.sort_values('Data_Dt')
            df['Acumulado'] = df['Lucro'].cumsum()
            st.plotly_chart(px.line(df, x='Data_Dt', y='Acumulado'), use_container_width=True)

if __name__ == "__main__":
    render_dashboard()