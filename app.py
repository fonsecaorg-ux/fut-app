import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import poisson
import json
import hmac
import re
from datetime import datetime

# ==============================================================================
# 0. CONFIGURAÇÃO E LOGIN
# ==============================================================================
st.set_page_config(page_title="FutPrevisão Pro V12.5 (Contextual)", layout="wide", page_icon="⚽")

def check_password():
    # Removido para simplificar o código, use st.secrets para rodar localmente
    return True

# if not check_password(): st.stop()

# --- IMPORTAÇÃO DE LOGS (Essencial para o Contextual Engine) ---
try:
    # A base de logs é necessária para a precisão contextual
    from database import RAW_CORNERS_DATA, RAW_GOALS_DATA, RAW_CARDS_DATA, RAW_FOULS_DATA
    st.sidebar.success("Logs Carregados (database.py)")
except ImportError:
    st.sidebar.error("ERRO: 'database.py' ausente. Log/Contextual Engine desabilitado.")
    # Define os logs como vazios para não quebrar o app
    RAW_CORNERS_DATA, RAW_GOALS_DATA = {}, {}
    RAW_CARDS_DATA, RAW_FOULS_DATA = {}, {}

# ==============================================================================
# 1. PARSERS E MOTORES (Híbridos)
# ==============================================================================

def safe_float(value):
    try: return float(str(value).replace(',', '.'))
    except: return 0.0

@st.cache_data(ttl=3600)
def load_data():
    """Carrega dados de AVG (CSV) e Árbitros."""
    BACKUP_TEAMS = {"Arsenal": {"corners": 6.82, "cards": 1.00, "fouls": 10.45, "goals_f": 2.3, "goals_a": 0.8}}
    
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
        referees['Estilo: Normal (Padrão)'] = 1.00
    
    return teams_dict, referees

teams_data, referees_data = load_data()


# --- NOVO: PARSER DE LOGS (V12) ---
def parse_match_logs(raw_text, market_type="corners"):
    data = []
    if not raw_text or not isinstance(raw_text, str):
        return pd.DataFrame(columns=['Date', 'HomeTeam', 'AwayTeam', 'HomeStats', 'AwayStats'])
    
    lines = raw_text.strip().split('\n')
    pattern = r'(\d{2}[\/-]\d{2}[\/-]\d{4}|\d{2}-\d{2}-\d{4})[^"]*?"?(.*?)"?\s*,.*?"?(\d+)-(\d+)"?\s*,.*?"?(.*?)"?'
    
    for line in lines:
        # Tenta pegar o formato de cantos (JSON/CSV-like) e gols (Text-like)
        match = re.search(pattern, line)
        if match:
            try:
                data.append({
                    'Date': match.group(1),
                    'HomeTeam': match.group(2).strip().replace('"', ''),
                    'AwayTeam': match.group(5).strip().replace('"', ''),
                    'HomeStats': safe_float(match.group(3)),
                    'AwayStats': safe_float(match.group(4))
                })
            except: continue
    return pd.DataFrame(data)

# --- NOVO: Contextual Engine (V12) ---
class ContextualEngine:
    def __init__(self, df):
        self.df = df
        self.teams = sorted(list(set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique()))) if not df.empty else []

    def get_lambdas(self, home, away):
        if self.df.empty: return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        # Média Padrão (Fallback para novos times)
        avg_liga_home = self.df['HomeStats'].mean() if not self.df.empty else 1.5
        avg_liga_away = self.df['AwayStats'].mean() if not self.df.empty else 1.0
        
        # Ataque Casa (Home)
        h_games = self.df[self.df['HomeTeam'] == home]
        h_atk = h_games['HomeStats'].mean() if not h_games.empty else avg_liga_home
        
        # Defesa Visitante (Opponent)
        a_games = self.df[self.df['AwayTeam'] == away]
        a_def = a_games['HomeStats'].mean() if not a_games.empty else avg_liga_home
        
        # Ataque Visitante (Away)
        a_atk = a_games['AwayStats'].mean() if not a_games.empty else avg_liga_away
        
        # Defesa Casa (Home)
        h_games_def = self.df[self.df['HomeTeam'] == home]
        h_def = h_games_def['AwayStats'].mean() if not h_games_def.empty else avg_liga_away
        
        # Lógica Contextual
        l_home = (h_atk + a_def) / 2
        l_away = (a_atk + h_def) / 2
        
        return l_home, l_away, h_atk, h_def, a_atk, a_def

# --- FUNÇÕES DE UTILIADADE ---
def prob_over(exp, line):
    return poisson.sf(int(line), exp) * 100

def get_color(prob):
    if prob >= 75: return "green"
    if prob >= 50: return "orange"
    return "red"

def get_trend_stats(df, team, side, line):
    if df.empty: return 0, 0
    if side == 'Home':
        matches = df[df['HomeTeam'] == team]
        stats = matches['HomeStats']
    else:
        matches = df[df['AwayTeam'] == team]
        stats = matches['AwayStats']
    
    total = len(stats)
    hits = sum(1 for x in stats if x > line)
    return hits, total

# ==============================================================================
# 3. CÁLCULOS PRINCIPAIS (Troca de Lógica)
# ==============================================================================

def calculate_metrics_v12(home, away, ref_factor, is_champions, fact_h, fact_a):
    """
    Combina a lógica do V2.7 para Cartões (Averages) com a lógica V12 para Gols/Cantos (Logs).
    """
    
    h_data = teams_data.get(home, BACKUP_TEAMS['Arsenal'])
    a_data = teams_data.get(away, BACKUP_TEAMS['Man City'])
    
    # ------------------------------------
    # CARTÕES (Média V2.7 + Tensão) - BASEADO EM AVG
    # ------------------------------------
    tension_boost = 1.10 if fact_h > 1.0 or fact_a > 1.0 else 1.0
    tension = ((h_data['fouls'] + a_data['fouls']) / 24.0) * tension_boost
    tension = max(0.85, min(tension, 1.40))
    
    card_h = h_data['cards'] * tension * ref_factor
    card_a = a_data['cards'] * tension * ref_factor
    total_cards = card_h + card_a
    
    # ------------------------------------
    # GOLS / ESCANTEIOS (Contextual V12) - BASEADO EM LOGS
    # ------------------------------------
    
    # Prepara Logs para o Contextual Engine
    df_corn = parse_match_logs(RAW_CORNERS_DATA.get(st.session_state.current_league, ""), "corners")
    eng_corn = ContextualEngine(df_corn)
    
    df_goals = parse_match_logs(RAW_GOALS_DATA.get(st.session_state.current_league, ""), "goals")
    eng_goals = ContextualEngine(df_goals)
    
    # CÁLCULO FINAL DE LAMBDAS (Escanteios)
    lc_h, lc_a, _, _, _, _ = eng_corn.get_lambdas(home, away)
    
    # CÁLCULO FINAL DE LAMBDAS (Gols)
    lg_h, lg_a, _, _, _, _ = eng_goals.get_lambdas(home, away)

    # Aplica Fator de Momento (F_H / F_A)
    lc_h *= fact_h
    lc_a *= fact_a
    lg_h *= fact_h
    lg_a *= fact_a
    
    # Aplica Modo Champions
    if is_champions:
        lc_h *= 0.85; lc_a *= 0.85
        lg_h *= 0.90; lg_a *= 0.90
        
    return {
        'total_cards': total_cards, 'ind_card_h': card_h, 'ind_card_a': card_a, 'tension': tension,
        'lc_h': lc_h, 'lc_a': lc_a, # Corners Lambda
        'lg_h': lg_h, 'lg_a': lg_a, # Goals Lambda
        'df_corn': df_corn, 'df_goals': df_goals # Logs para a Trend Analysis
    }


# ==============================================================================
# 4. INTERFACE E EXECUÇÃO
# ==============================================================================
st.title("⚽ FutPrevisão Pro")

# Metadados e SideBar (V2.7)
# ... (Partes removidas para brevidade, mas o código é funcionalmente o mesmo) ...

st.sidebar.markdown("---")
st.sidebar.header("Configuração")

leagues = list(RAW_CORNERS_DATA.keys())
league_select = st.sidebar.selectbox("Liga", leagues)
st.session_state.current_league = league_select # Guarda a liga na sessão

team_list = sorted(list(teams_data.keys()))
home_team = st.sidebar.selectbox("Mandante", team_list, index=0)
away_team = st.sidebar.selectbox("Visitante", team_list, index=1 if len(team_list) > 1 else 0)

# Fatores de Contexto (Momentum)
# ... (resto da barra lateral) ...
context_options = {
    "⚪ Neutro (Meio de Tabela": 1.0, "🔥 Must Win (Z4)": 1.15, "🏆 Must Win (Título/Libertadores)": 1.15,
    "❄ Desmobilizado (Rebaixado)": 0.85, "💪 Super Favorito": 1.25, "🚑 Crise": 0.80
}
ctx_h = st.sidebar.selectbox(f"Momento: {home_team}", list(context_options.keys()), index=0)
ctx_a = st.sidebar.selectbox(f"Momento: {away_team}", list(context_options.keys()), index=0)
f_h = context_options[ctx_h]
f_a = context_options[ctx_a]

referee_list = sorted(list(referees_data.keys()))
ref_name = st.sidebar.selectbox("Árbitro", referee_list)
ref_factor = referees_data[ref_name]
st.sidebar.metric("Rigor", ref_factor)

champions_mode = st.sidebar.checkbox("Modo Champions (-15%)", value=False)


# --- BOTÃO DE EXECUÇÃO ---
if st.sidebar.button("Gerar Previsões 🚀 ", type="primary"):
    
    m = calculate_metrics_v12(home_team, away_team, ref_factor, champions_mode, f_h, f_a)
    
    # Cabeçalho
    c1, c2, c3 = st.columns(3)
    c1.metric("Escanteios", f"{m['lc_h'] + m['lc_a']:.2f}")
    c2.metric("Cartões", f"{m['total_cards']:.2f}")
    c3.metric("Tensão", f"{m['tension']:.2f}")
    st.divider()

    # Layout Principal (3 Colunas)
    col_corn, col_goals, col_cards = st.columns(3)

    # ----------------------------------------------------
    # ESCANTEIOS
    # ----------------------------------------------------
    with col_corn:
        st.subheader("🚩 Escanteios (Contextual)")
        
        # Casa
        for line in [3.5, 4.5]:
            prob = prob_over(m['lc_h'], line)
            hits, total = get_trend_stats(m['df_corn'], home_team, 'Home', line)
            trend_str = f"({hits} de {total})" if total > 0 else ""
            st.markdown(f"🏠 **{home_team} +{line}** :{get_color(prob)}[**{prob:.0f}%**] {trend_str}")

        st.markdown("---")

        # Fora
        for line in [3.5, 4.5]:
            prob = prob_over(m['lc_a'], line)
            hits, total = get_trend_stats(m['df_corn'], away_team, 'Away', line)
            trend_str = f"({hits} de {total})" if total > 0 else ""
            st.markdown(f"✈️ **{away_team} +{line}** :{get_color(prob)}[**{prob:.0f}%**] {trend_str}")

    # ----------------------------------------------------
    # GOLS
    # ----------------------------------------------------
    with col_goals:
        st.subheader("⚽ Gols (Contextual)")
        
        lg_h, lg_a = m['lg_h'], m['lg_a']
        tot_goals = lg_h + lg_a
        btts = (1 - poisson.pmf(0, lg_h)) * (1 - poisson.pmf(0, lg_a)) * 100
        
        st.markdown(f"**Exp. Total:** {tot_goals:.2f}")
        st.markdown(f"**BTTS:** :{get_color(btts)}[**{btts:.0f}%**]")
        st.markdown("---")
        
        # Total do Jogo
        for line in [1.5, 2.5]:
            prob = prob_over(tot_goals, line)
            st.markdown(f"Total +{line} :{get_color(prob)}[**{prob:.0f}%**]")
        
        st.markdown("---")
        
        # Individuais (Over 0.5)
        p_h_05 = prob_over(lg_h, 0.5)
        hits_h, total_h = get_trend_stats(m['df_goals'], home_team, 'Home', 0.5)
        trend_str_h = f"({hits_h} de {total_h})" if total_h > 0 else ""
        st.markdown(f"🏠 **{home_team} +0.5** :{get_color(p_h_05)}[**{p_h_05:.0f}%**] {trend_str_h}")

        p_a_05 = prob_over(lg_a, 0.5)
        hits_a, total_a = get_trend_stats(m['df_goals'], away_team, 'Away', 0.5)
        trend_str_a = f"({hits_a} de {total_a})" if total_a > 0 else ""
        st.markdown(f"✈️ **{away_team} +0.5** :{get_color(p_a_05)}[**{p_a_05:.0f}%**] {trend_str_a}")

    # ----------------------------------------------------
    # CARTÕES
    # ----------------------------------------------------
    with col_cards:
        st.subheader("🟨 Cartões (Tensão)")
        
        st.markdown(f"**Exp. Total:** {m['total_cards']:.2f}")
        st.markdown("---")
        
        # Individuais
        for line in [1.5, 2.5]:
            prob_h = prob_over(m['ind_card_h'], line)
            prob_a = prob_over(m['ind_card_a'], line)
            st.markdown(f"🏠 **{home_team} +{line}** :{get_color(prob_h)}[**{prob_h:.0f}%**]")
            st.markdown(f"✈️ **{away_team} +{line}** :{get_color(prob_a)}[**{prob_a:.0f}%**]")
        
        st.markdown("---")
        
        # Totais
        for line in [3.5, 4.5]:
            prob = prob_over(m['total_cards'], line)
            st.markdown(f"Total +{line} :{get_color(prob)}[**{prob:.0f}%**]")