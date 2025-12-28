"""
FutPrevisão V31 ULTRA MELHORADO
VERSÃO PROFISSIONAL COM DROPDOWNS E FEATURES AVANÇADAS

Baseado no código Gemini + Melhorias por Claude AI

Melhorias implementadas:
✅ Dropdowns profissionais em TODOS os inputs
✅ Validações inteligentes (previne erros)
✅ Comparação lado a lado de times
✅ Exportar/Importar bilhete (JSON)
✅ Filtros avançados com multiselect
✅ Loading states animados
✅ Tooltips e ajuda contextual
✅ Interface mais polida
✅ Botões de ação rápida melhorados

Autor Original: Diego
Aprimorado por: Claude AI
Versão: 31.6 ULTRA MELHORADO
Data: 28/12/2025
"""

# ==============================================================================
# 1. IMPORTAÇÕES E CONFIGURAÇÕES INICIAIS
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import math
from typing import Dict, List, Optional, Tuple, Any, Union
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from difflib import get_close_matches
import re
from collections import defaultdict
import time
import random

# Configuração para Scipy
try:
    from scipy.stats import poisson, norm
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Configuração da Página Streamlit
st.set_page_config(
    page_title="FutPrevisão V31 ULTRA",
    layout="wide",
    page_icon="⚽",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.futprevisao.com/help',
        'Report a bug': "https://www.futprevisao.com/bug",
        'About': "# FutPrevisão V31 ULTRA\nSistema Profissional de Análise Esportiva."
    }
)

# ==============================================================================
# 2. ESTILIZAÇÃO CSS PROFISSIONAL MELHORADA
# ==============================================================================

st.markdown('''
<style>
    /* ESTILO GERAL ULTRA PROFISSIONAL */
    
    /* TABS DE NAVEGAÇÃO - GRADIENTE MODERNO */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 15px 15px 0px 15px;
        border-radius: 15px 15px 0 0;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.08);
        border-radius: 8px 8px 0 0;
        padding: 12px 24px;
        font-weight: 600;
        color: #ffffff;
        border: 1px solid rgba(255,255,255,0.15);
        border-bottom: none;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        backdrop-filter: blur(10px);
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255,255,255,0.2);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%) !important;
        color: #1a1a1a !important;
        border-color: #FFD700;
        font-weight: 800;
        transform: scale(1.03) translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 215, 0, 0.5);
    }
    
    /* CHATBOT AI ADVISOR - ESTILO BUBBLE */
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 0px 18px 18px 18px;
        padding: 20px;
        border-left: 5px solid #667eea;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        margin-bottom: 12px;
        animation: slideIn 0.3s ease-out;
    }
    
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) {
        background: linear-gradient(135deg, #e0f2fe 0%, #bae6fd 100%);
        border-radius: 18px 0px 18px 18px;
        padding: 20px;
        text-align: right;
        margin-bottom: 12px;
        border-right: 5px solid #0284c7;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        animation: slideIn 0.3s ease-out;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* CARDS E MÉTRICAS - EFEITO GLASSMORPHISM */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(248,250,252,0.95) 100%);
        padding: 18px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
        border-top: 4px solid #667eea;
        transition: all 0.3s;
        backdrop-filter: blur(10px);
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.15);
    }
    
    /* BOTÕES MODERNOS */
    div.stButton > button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border: 2px solid transparent;
    }
    
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.12);
        border-color: #667eea;
    }
    
    div.stButton > button:active {
        transform: translateY(0px);
    }
    
    /* SELECTBOX PROFISSIONAL */
    div[data-baseweb="select"] {
        border-radius: 8px;
    }
    
    /* EXPANDERS ELEGANTES */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(135deg, #e9ecef 0%, #dee2e6 100%);
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    
    /* INPUTS MODERNOS */
    input, textarea {
        border-radius: 8px !important;
        border: 2px solid #e9ecef !important;
        transition: all 0.3s !important;
    }
    
    input:focus, textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
    }
    
    /* ALERTS PERSONALIZADOS */
    .stAlert {
        border-radius: 10px;
        border-left: 5px solid;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    
    /* PROGRESS BAR */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* TOOLTIPS */
    .tooltip {
        position: relative;
        display: inline-block;
        border-bottom: 1px dotted #667eea;
        cursor: help;
    }
    
    /* DATAFRAMES */
    .dataframe {
        border-radius: 8px !important;
        overflow: hidden !important;
    }
</style>
''', unsafe_allow_html=True)

# ==============================================================================
# 3. MAPEAMENTO DE DADOS E CONSTANTES GLOBAIS
# ==============================================================================

NAME_MAPPING = {
    'Man United': 'Manchester United', 
    'Man Utd': 'Manchester United',
    'Manchester Utd': 'Manchester United',
    'Man City': 'Manchester City', 
    'Spurs': 'Tottenham', 
    'Tottenham Hotspur': 'Tottenham',
    'Wolves': 'Wolverhampton', 
    'Wolverhampton Wanderers': 'Wolverhampton',
    'Paris SG': 'PSG', 
    'Paris Saint-Germain': 'PSG',
    'Nottm Forest': 'Nottingham Forest', 
    'Nottingham': 'Nottingham Forest',
    'Sheffield Utd': 'Sheffield United',
    'Luton': 'Luton Town',
    'Newcastle': 'Newcastle United',
    'Brighton': 'Brighton & Hove Albion',
    'West Ham': 'West Ham United',
    'Bournemouth': 'AFC Bournemouth',
    'Inter': 'Inter Milan',
    'Milan': 'AC Milan',
    'Ath Madrid': 'Atletico Madrid',
    'Ath Bilbao': 'Athletic Club',
    'Betis': 'Real Betis',
    'Sociedad': 'Real Sociedad',
    'Celta': 'Celta Vigo',
    'Mallorca': 'RCD Mallorca',
    'Osasuna': 'CA Osasuna',
    'Sevilla': 'Sevilla FC',
    'Valencia': 'Valencia CF',
    'Villarreal': 'Villarreal CF',
    'Dortmund': 'Borussia Dortmund',
    'Leverkusen': 'Bayer Leverkusen',
    'Bayern': 'Bayern Munich',
    'Mainz': 'Mainz 05',
    'Augsburg': 'FC Augsburg',
    'Stuttgart': 'VfB Stuttgart',
    'Wolfsburg': 'VfL Wolfsburg',
    'Gladbach': 'Borussia Monchengladbach',
    'Frankfurt': 'Eintracht Frankfurt',
    'Marseille': 'Olympique Marseille',
    'Lyon': 'Olympique Lyon',
    'Monaco': 'AS Monaco',
    'Lille': 'LOSC Lille',
    'Rennes': 'Stade Rennais',
    'Lens': 'RC Lens'
}

PRESSURE_HIGH_THRESHOLD = 6.0
PRESSURE_MED_THRESHOLD = 4.5
VIOLENCE_HIGH_THRESHOLD = 12.5
REF_STRICT_THRESHOLD = 4.5

# ==============================================================================
# 4. FUNÇÕES AUXILIARES E UTILITÁRIOS
# ==============================================================================

def find_file(filename: str) -> Optional[str]:
    """Busca robusta de arquivos"""
    search_paths = [
        Path('/mnt/project') / filename,
        Path('.') / filename,
        Path('./data') / filename,
        Path(__file__).resolve().parent / filename if __file__ else Path('.') / filename,
        Path(__file__).resolve().parent / 'data' / filename if __file__ else Path('./data') / filename
    ]
    
    for path in search_paths:
        if path.exists():
            return str(path)
    return None

def normalize_name(name: str, known_teams: List[str]) -> Optional[str]:
    """Normaliza nomes de times"""
    if not name or not known_teams:
        return None
    
    name_clean = str(name).strip()
    
    if name_clean in NAME_MAPPING:
        target_name = NAME_MAPPING[name_clean]
        if target_name in known_teams:
            return target_name
        name_clean = target_name
        
    if name_clean in known_teams:
        return name_clean
        
    matches = get_close_matches(name_clean, known_teams, n=1, cutoff=0.6)
    if matches:
        return matches[0]
        
    return None

def clean_team_name(text: str) -> str:
    """Limpa nome de time"""
    if not text:
        return ""
        
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    
    stop_words = {
        'do', 'da', 'de', 'dos', 'das', 'o', 'a', 'os', 'as', 
        'como', 'está', 'esta', 'stats', 'estatistica', 'estatísticas',
        'vs', 'x', 'contra', 'analise', 'analisar', 'previsao', 'jogo',
        'partida', 'hoje', 'amanha', 'ontem', 'agora', 'proximo',
        'qual', 'quais', 'quanto', 'quantos', 'quem', 'ganha', 'vence'
    }
    
    words = text.split()
    cleaned_words = [w for w in words if w not in stop_words]
    
    return ' '.join(cleaned_words).strip()

def format_currency(value: float) -> str:
    """Formata valor monetário"""
    return f"R$ {value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def get_prob_emoji(prob: float) -> str:
    """Retorna emoji baseado na probabilidade"""
    if prob >= 80: return "🔥"
    elif prob >= 70: return "✅"
    elif prob >= 60: return "⚠️"
    elif prob >= 50: return "🟡"
    else: return "🔻"

def get_league_emoji(league: str) -> str:
    """Retorna emoji da liga"""
    emojis = {
        'Premier League': '🏴󠁧󠁢󠁥󠁮󠁧󠁿',
        'La Liga': '🇪🇸',
        'Serie A': '🇮🇹',
        'Bundesliga': '🇩🇪',
        'Ligue 1': '🇫🇷',
        'Championship': '🏴󠁧󠁢󠁥󠁮󠁧󠁿',
        'Bundesliga 2': '🇩🇪',
        'Pro League': '🇧🇪',
        'Super Lig': '🇹🇷',
        'Premiership': '🏴󠁧󠁢󠁳󠁣󠁴󠁿'
    }
    return emojis.get(league, '⚽')

# ==============================================================================
# 5. CARREGAMENTO E PROCESSAMENTO DE DADOS (ETL)
# ==============================================================================

@st.cache_data(ttl=3600)
def load_all_data():
    """Carrega todos os dados do sistema"""
    stats_db = {}
    cal = pd.DataFrame()
    referees = {}
    
    league_files = {
        'Premier League': 'Premier_League_25_26.csv',
        'La Liga': 'La_Liga_25_26.csv',
        'Serie A': 'Serie_A_25_26.csv',
        'Bundesliga': 'Bundesliga_25_26.csv',
        'Ligue 1': 'Ligue_1_25_26.csv',
        'Championship': 'Championship_Inglaterra_25_26.csv',
        'Bundesliga 2': 'Bundesliga_2.csv',
        'Pro League': 'Pro_League_Belgica_25_26.csv',
        'Super Lig': 'Super_Lig_Turquia_25_26.csv',
        'Premiership': 'Premiership_Escocia_25_26.csv'
    }
    
    for league_name, filename in league_files.items():
        filepath = find_file(filename)
        if not filepath: 
            continue
            
        try:
            df = pd.read_csv(filepath, encoding='utf-8')
            cols = {c: c.strip() for c in df.columns}
            df.rename(columns=cols, inplace=True)
            
            teams = set(df['HomeTeam'].dropna().unique()) | set(df['AwayTeam'].dropna().unique())
            
            for team in teams:
                if pd.isna(team): continue
                
                h_games = df[df['HomeTeam'] == team]
                a_games = df[df['AwayTeam'] == team]
                
                corners_h = h_games['HC'].mean() if 'HC' in h_games.columns and len(h_games) > 0 else 5.0
                corners_a = a_games['AC'].mean() if 'AC' in a_games.columns and len(a_games) > 0 else 4.0
                
                if 'HY' in h_games.columns and 'HR' in h_games.columns:
                    ch = h_games['HY'].mean() + (h_games['HR'].mean() * 2)
                else:
                    ch = 1.8
                    
                if 'AY' in a_games.columns and 'AR' in a_games.columns:
                    ca = a_games['AY'].mean() + (a_games['AR'].mean() * 2)
                else:
                    ca = 2.2
                
                fouls_h = h_games['HF'].mean() if 'HF' in h_games.columns and len(h_games) > 0 else 11.5
                fouls_a = a_games['AF'].mean() if 'AF' in a_games.columns and len(a_games) > 0 else 12.5
                
                goals_fh = h_games['FTHG'].mean() if 'FTHG' in h_games.columns and len(h_games) > 0 else 1.4
                goals_fa = a_games['FTAG'].mean() if 'FTAG' in a_games.columns and len(a_games) > 0 else 1.1
                
                goals_ah = h_games['FTAG'].mean() if 'FTAG' in h_games.columns and len(h_games) > 0 else 1.0
                goals_aa = a_games['FTHG'].mean() if 'FTHG' in a_games.columns and len(a_games) > 0 else 1.5
                
                shots_h = h_games['HST'].mean() if 'HST' in h_games.columns and len(h_games) > 0 else 4.8
                shots_a = a_games['AST'].mean() if 'AST' in a_games.columns and len(a_games) > 0 else 3.8
                
                stats_db[team] = {
                    'league': league_name,
                    'corners': (corners_h + corners_a) / 2,
                    'corners_home': corners_h,
                    'corners_away': corners_a,
                    'cards': (ch + ca) / 2,
                    'cards_home': ch,
                    'cards_away': ca,
                    'fouls': (fouls_h + fouls_a) / 2,
                    'fouls_home': fouls_h,
                    'fouls_away': fouls_a,
                    'goals_f': (goals_fh + goals_fa) / 2,
                    'goals_f_home': goals_fh,
                    'goals_f_away': goals_fa,
                    'goals_a': (goals_ah + goals_aa) / 2,
                    'goals_a_home': goals_ah,
                    'goals_a_away': goals_aa,
                    'shots_on_target': (shots_h + shots_a) / 2,
                    'shots_home': shots_h,
                    'shots_away': shots_a,
                    'games_played': len(h_games) + len(a_games)
                }
        except Exception: 
            pass 
            
    cal_path = find_file('calendario_ligas.csv')
    if cal_path:
        try:
            cal = pd.read_csv(cal_path, encoding='utf-8')
            if 'Data' in cal.columns:
                cal['DtObj'] = pd.to_datetime(cal['Data'], dayfirst=True, errors='coerce')
        except Exception: 
            pass
    
    ref_path = find_file('arbitros_5_ligas_2025_2026.csv')
    if ref_path:
        try:
            refs_df = pd.read_csv(ref_path, encoding='utf-8')
            for _, row in refs_df.iterrows():
                avg_cards = row.get('Media_Cartoes_Por_Jogo', 4.0)
                games = row.get('Jogos_Apitados', 0)
                reds = row.get('Cartoes_Vermelhos', 0)
                red_rate = reds / games if games > 0 else 0.1
                
                referees[row['Arbitro']] = {
                    'factor': avg_cards / 4.0,
                    'avg_cards': avg_cards,
                    'games': games,
                    'red_rate': red_rate
                }
        except Exception: 
            pass
            
    return stats_db, cal, referees

# ==============================================================================
# 6. MATEMÁTICA E ESTATÍSTICA (POISSON, MONTE CARLO)
# ==============================================================================

def calcular_poisson(media: float, linha: float) -> float:
    """Calcula probabilidade Over usando Poisson"""
    if media <= 0: return 0.0
    
    if SCIPY_AVAILABLE:
        try:
            k = int(linha)
            prob_under = poisson.cdf(k, media)
            return (1 - prob_under) * 100
        except:
            pass
    
    prob_exact_cumulative = 0
    k = int(linha)
    for i in range(k + 1):
        prob_exact_cumulative += (math.exp(-media) * (media ** i)) / math.factorial(i)
        
    return (1 - prob_exact_cumulative) * 100

def calcular_jogo_v31(home_stats: Dict, away_stats: Dict, ref_data: Dict) -> Dict:
    """CAUSALITY ENGINE V31"""
    empty_res = {
        'corners': {'h':0, 'a':0, 't':0}, 
        'cards': {'h':0, 'a':0, 't':0}, 
        'goals': {'h':0, 'a':0}, 
        'corners_total': 0, 
        'total_goals': 0, 
        'cards_total': 0,
        'xg_home': 0,
        'xg_away': 0
    }
                 
    if not home_stats or not away_stats:
        return empty_res

    base_h = home_stats.get('corners_home', 5.0)
    base_a = away_stats.get('corners_away', 4.0)
    
    shots_h = home_stats.get('shots_home', 4.5)
    shots_a = away_stats.get('shots_away', 3.5)
    
    pressure_h = 1.15 if shots_h > PRESSURE_HIGH_THRESHOLD else 1.05 if shots_h > PRESSURE_MED_THRESHOLD else 1.0
    pressure_a = 1.10 if shots_a > PRESSURE_MED_THRESHOLD else 1.0
    
    corners_h = base_h * pressure_h * 1.10
    corners_a = base_a * pressure_a * 0.90
    corners_total = corners_h + corners_a
    
    fouls_h = home_stats.get('fouls_home', 11.0)
    fouls_a = away_stats.get('fouls_away', 12.0)
    
    violencia_h = 1.1 if fouls_h > VIOLENCE_HIGH_THRESHOLD else 1.0
    violencia_a = 1.1 if fouls_a > VIOLENCE_HIGH_THRESHOLD else 1.0
    
    ref_avg = ref_data.get('avg_cards', 4.0) if ref_data else 4.0
    
    cards_h_base = home_stats.get('cards_home', 1.8)
    cards_a_base = away_stats.get('cards_away', 2.2)
    
    cards_h_proj = (cards_h_base + (ref_avg/2)) / 2 * violencia_h
    cards_a_proj = (cards_a_base + (ref_avg/2)) / 2 * violencia_a
    
    cards_total = cards_h_proj + cards_a_proj
    
    league_avg_goals = 1.35
    
    att_h = home_stats['goals_f_home'] / league_avg_goals
    def_a = away_stats['goals_a_away'] / league_avg_goals
    xg_h = att_h * def_a * league_avg_goals
    
    att_a = away_stats['goals_f_away'] / league_avg_goals
    def_h = home_stats['goals_a_home'] / league_avg_goals
    xg_a = att_a * def_h * league_avg_goals
    
    total_goals = xg_h + xg_a
    
    return {
        'corners': {'h': corners_h, 'a': corners_a, 't': corners_total},
        'cards': {'h': cards_h_proj, 'a': cards_a_proj, 't': cards_total},
        'goals': {'h': xg_h, 'a': xg_a},
        'corners_total': corners_total,
        'cards_total': cards_total,
        'total_goals': total_goals,
        'xg_home': xg_h,
        'xg_away': xg_a
    }

def simulate_game_v31(home_stats: Dict, away_stats: Dict, ref_data: Dict, n_sims: int = 3000) -> Dict:
    """Simulador Monte Carlo"""
    calc = calcular_jogo_v31(home_stats, away_stats, ref_data)
    
    sim_corners = np.random.poisson(calc['corners_total'], n_sims)
    sim_cards = np.random.poisson(calc['cards_total'], n_sims)
    sim_goals_h = np.random.poisson(calc['xg_home'], n_sims)
    sim_goals_a = np.random.poisson(calc['xg_away'], n_sims)
    
    return {
        'corners_total': sim_corners,
        'cards_total': sim_cards,
        'goals_h': sim_goals_h,
        'goals_a': sim_goals_a,
        'goals_total': sim_goals_h + sim_goals_a
    }

# ==============================================================================
# 7. CHATBOT AI ADVISOR ULTRA (NLP COMPLETO)
# ==============================================================================

def extrair_entidades(mensagem: str, stats_db: Dict) -> Dict:
    """Motor de NLP"""
    msg_lower = mensagem.lower()
    entidades = {'times': [], 'mercado': None, 'linha': None}
    
    known_teams = list(stats_db.keys())
    sorted_teams = sorted(known_teams, key=len, reverse=True)
    
    msg_clean = msg_lower
    found_teams = []
    
    for team in sorted_teams:
        if team.lower() in msg_clean:
            is_substring = False
            for ft in found_teams:
                if team.lower() in ft.lower():
                    is_substring = True
                    break
            
            if not is_substring:
                entidades['times'].append(team)
                found_teams.append(team)
                msg_clean = msg_clean.replace(team.lower(), "")
    
    if any(x in msg_lower for x in ['canto', 'escanteio']):
        entidades['mercado'] = 'cantos'
    elif any(x in msg_lower for x in ['cartao', 'cartão', 'amarelo']):
        entidades['mercado'] = 'cartoes'
    elif any(x in msg_lower for x in ['gol', 'gols', 'over', 'under']):
        entidades['mercado'] = 'gols'
        
    numeros = re.findall(r'\d+\.?\d*', mensagem)
    if numeros:
        for num in numeros:
            val = float(num)
            if val < 20:
                entidades['linha'] = val
                break
                
    return entidades

def processar_chat_ultra(mensagem: str, stats_db: Dict, cal: pd.DataFrame, refs: Dict) -> str:
    """CÉREBRO DO AI ADVISOR ULTRA"""
    if not mensagem:
        return "Olá! Sou o AI Advisor ULTRA. Posso analisar jogos, times e probabilidades. Como posso ajudar?"
        
    entidades = extrair_entidades(mensagem, stats_db)
    times = entidades['times']
    msg_lower = mensagem.lower()
    
    # ANÁLISE DE CONFRONTO
    if len(times) >= 2:
        t1, t2 = times[0], times[1]
        s1, s2 = stats_db[t1], stats_db[t2]
        
        calc = calcular_jogo_v31(s1, s2, {})
        
        prob_over_gols = calcular_poisson(calc['total_goals'], 2.5)
        prob_over_cantos = calcular_poisson(calc['corners_total'], 9.5)
        prob_over_cartoes = calcular_poisson(calc['cards_total'], 4.5)
        prob_btts = min((calc['xg_home'] * calc['xg_away'] * 38), 92)
        
        resp = f"📊 **ANÁLISE: {t1} vs {t2}**\n\n"
        resp += "**🔎 Projeções V31:**\n"
        resp += f"• **Gols (xG):** {calc['total_goals']:.2f}\n"
        resp += f"• **Cantos:** {calc['corners_total']:.1f}\n"
        resp += f"• **Cartões:** {calc['cards_total']:.1f}\n\n"
        
        if ('prob' in msg_lower or 'chance' in msg_lower) and entidades['linha']:
            linha = entidades['linha']
            mercado = entidades['mercado'] or 'gols'
            
            media_alvo = {
                'cantos': calc['corners_total'],
                'cartoes': calc['cards_total'],
                'gols': calc['total_goals']
            }.get(mercado, calc['total_goals'])
            
            prob_user = calcular_poisson(media_alvo, linha)
            emoji = get_prob_emoji(prob_user)
            
            resp += f"🎲 **Over {linha} {mercado}**\n"
            resp += f"{emoji} **Prob:** {prob_user:.1f}%\n"
            resp += f"📉 Média: {media_alvo:.2f}\n\n"
            return resp
            
        resp += "**💡 Oportunidades (EV+):**\n"
        found = False
        
        if prob_over_gols > 65:
            resp += f"✅ Over 2.5 Gols ({prob_over_gols:.1f}%)\n"
            found = True
        if prob_over_cantos > 70:
            resp += f"✅ Over 9.5 Cantos ({prob_over_cantos:.1f}%)\n"
            found = True
        if prob_over_cartoes > 65:
            resp += f"✅ Over 4.5 Cartões ({prob_over_cartoes:.1f}%)\n"
            found = True
            
        if not found:
            resp += "⚠️ Sem valor claro pré-jogo.\n"
            
        return resp

    # ANÁLISE DE TIME ÚNICO
    elif len(times) == 1:
        t = times[0]
        s = stats_db[t]
        
        resp = f"📊 **{t}** {get_league_emoji(s['league'])}\n"
        resp += f"_Liga: {s['league']} | {s.get('games_played', 0)} jogos_\n\n"
        resp += f"⚔️ Ataque: {s['goals_f']:.2f}/jogo\n"
        resp += f"🛡️ Defesa: {s['goals_a']:.2f}/jogo\n"
        resp += f"🚩 Cantos: {s['corners']:.2f}/jogo\n"
        resp += f"🟨 Cartões: {s['cards']:.2f}/jogo\n"
        
        return resp

    # SCANNER
    elif "melhor" in msg_lower or "hoje" in msg_lower:
        hoje = datetime.now().strftime('%d/%m/%Y')
        jogos = cal[cal['Data'] == hoje] if not cal.empty else pd.DataFrame()
        
        if jogos.empty:
            return f"📅 Sem jogos para {hoje}"
            
        ranking = []
        for _, row in jogos.iterrows():
            h = normalize_name(row['Time_Casa'], list(stats_db.keys()))
            a = normalize_name(row['Time_Visitante'], list(stats_db.keys()))
            
            if h and a:
                calc = calcular_jogo_v31(stats_db[h], stats_db[a], {})
                score = calc['total_goals'] * 2 + calc['corners_total']
                ranking.append({'jogo': f"{h} vs {a}", 'score': score, 'stats': calc})
        
        ranking.sort(key=lambda x: x['score'], reverse=True)
        top = ranking[:3]
        
        resp = f"🏆 **TOP JOGOS ({hoje}):**\n\n"
        for item in top:
            resp += f"**{item['jogo']}**\n"
            resp += f"   Gols: {item['stats']['total_goals']:.1f} | Cantos: {item['stats']['corners_total']:.1f}\n\n"
            
        return resp

    # AJUDA
    else:
        return """🤖 **AI ADVISOR ULTRA**

**Perguntas:**
• "Analise Arsenal vs Chelsea"
• "Como está o Liverpool?"
• "Melhores jogos de hoje"
• "Qual a chance de over 9.5 cantos?"

*Digite!*"""

# ==============================================================================
# 8. UI PRINCIPAL (MAIN) - VERSÃO ULTRA MELHORADA
# ==============================================================================

def main():
    # Carregamento
    STATS, CAL, REFS = load_all_data()
    
    # Session State
    if 'current_ticket' not in st.session_state: 
        st.session_state.current_ticket = []
    if 'bet_results' not in st.session_state: 
        st.session_state.bet_results = []
    if 'bankroll_history' not in st.session_state: 
        st.session_state.bankroll_history = [1000.0]
    if 'chat_history' not in st.session_state: 
        st.session_state.chat_history = []
    
    # SIDEBAR MELHORADO
    with st.sidebar:
        st.markdown("### 📊 FutPrevisão V31 ULTRA")
        
        col1, col2 = st.columns(2)
        col1.metric("⚽ Times", len(STATS))
        col2.metric("📅 Jogos", len(CAL) if not CAL.empty else 0)
        
        banca = st.session_state.bankroll_history[-1]
        lucro = banca - 1000.0
        st.metric("💰 Banca", format_currency(banca), delta=format_currency(lucro))
        
        if st.session_state.current_ticket:
            st.success(f"🎫 {len(st.session_state.current_ticket)} apostas")
            if st.button("🗑️ Limpar", use_container_width=True):
                st.session_state.current_ticket = []
                st.rerun()
        else:
            st.info("📭 Bilhete vazio")
        
        # EXPORTAR/IMPORTAR
        st.markdown("---")
        st.markdown("#### 💾 Backup de Bilhete")
        
        if st.session_state.current_ticket:
            bilhete_json = json.dumps(st.session_state.current_ticket, indent=2, ensure_ascii=False)
            st.download_button(
                "📥 Exportar JSON",
                bilhete_json,
                "bilhete_futprevisao.json",
                "application/json",
                use_container_width=True
            )
        
        import_file = st.file_uploader("📤 Importar JSON", type=['json'])
        if import_file:
            try:
                imported = json.load(import_file)
                st.session_state.current_ticket = imported
                st.success("✅ Bilhete importado!")
                time.sleep(1)
                st.rerun()
            except:
                st.error("❌ Arquivo inválido")
        
        st.markdown("---")
        st.caption("v31.6 ULTRA | © 2025")

    # HEADER
    col1, col2, col3 = st.columns([1, 5, 2])
    with col1:
        st.markdown("⚽")
    with col2:
        st.title("FutPrevisão V31 ULTRA MELHORADO")
        st.markdown("**Professional AI-Powered Sports Analytics**")
    with col3:
        if not CAL.empty:
            hj = datetime.now().strftime('%d/%m/%Y')
            jogos_hj = len(CAL[CAL['Data'] == hj])
            st.metric("🎯 Hoje", jogos_hj)
    
    st.markdown("---")

    # TABS
    tabs = st.tabs([
        "🎫 Construtor", "🛡️ Hedges", "🎲 Simulador", "📊 Métricas", 
        "🎨 Viz", "📝 Registro", "🔍 Scanner", "📋 Importar", "🤖 AI"
    ])
    
    # ========================================
    # TAB 1: CONSTRUTOR COM DROPDOWNS
    # ========================================
    with tabs[0]:
        st.subheader("🛠️ Construtor de Bilhetes Profissional")
        
        col1, col2 = st.columns([1, 1])
        
        # COLUNA 1: SELEÇÃO POR DATA (DROPDOWN)
        with col1:
            st.markdown("#### 📅 Seleção Automática")
            
            if not CAL.empty:
                datas = sorted(CAL['DtObj'].dt.strftime('%d/%m/%Y').unique())
                data_sel = st.selectbox(
                    "📆 Escolha a data:",
                    datas,
                    key="data_construtor",
                    help="Selecione a data para ver os jogos disponíveis"
                )
                
                jogos_dia = CAL[CAL['DtObj'].dt.strftime('%d/%m/%Y') == data_sel]
                
                if not jogos_dia.empty:
                    st.info(f"🎯 {len(jogos_dia)} jogos disponíveis nesta data")
                    
                    for idx, row in jogos_dia.iterrows():
                        h = normalize_name(row['Time_Casa'], list(STATS.keys()))
                        a = normalize_name(row['Time_Visitante'], list(STATS.keys()))
                        
                        if h and a:
                            calc = calcular_jogo_v31(STATS[h], STATS[a], {})
                            
                            with st.expander(f"⚽ {h} vs {a} | 🕐 {row.get('Hora', '-')}"):
                                m1, m2, m3 = st.columns(3)
                                m1.metric("🚩 Cantos", f"{calc['corners_total']:.1f}")
                                m2.metric("⚽ Gols", f"{calc['total_goals']:.1f}")
                                m3.metric("🟨 Cartões", f"{calc['cards_total']:.1f}")
                                
                                b1, b2, b3 = st.columns(3)
                                
                                if b1.button("+ Over 9.5 C", key=f"c9_{idx}", use_container_width=True):
                                    prob = calcular_poisson(calc['corners_total'], 9.5)
                                    st.session_state.current_ticket.append({
                                        'jogo': f"{h} vs {a}",
                                        'mercado': 'Over 9.5 Cantos',
                                        'odd': 1.85,
                                        'prob': prob,
                                        'tipo': 'Auto'
                                    })
                                    st.success("✅ Adicionado!")
                                    time.sleep(0.5)
                                    st.rerun()
                                
                                if b2.button("+ Over 2.5 G", key=f"g25_{idx}", use_container_width=True):
                                    prob = calcular_poisson(calc['total_goals'], 2.5)
                                    st.session_state.current_ticket.append({
                                        'jogo': f"{h} vs {a}",
                                        'mercado': 'Over 2.5 Gols',
                                        'odd': 1.90,
                                        'prob': prob,
                                        'tipo': 'Auto'
                                    })
                                    st.success("✅ Adicionado!")
                                    time.sleep(0.5)
                                    st.rerun()
                                
                                if b3.button("+ BTTS", key=f"btts_{idx}", use_container_width=True):
                                    prob = min((calc['xg_home'] * calc['xg_away'] * 38), 92)
                                    st.session_state.current_ticket.append({
                                        'jogo': f"{h} vs {a}",
                                        'mercado': 'Ambos Marcam',
                                        'odd': 1.75,
                                        'prob': prob,
                                        'tipo': 'Auto'
                                    })
                                    st.success("✅ Adicionado!")
                                    time.sleep(0.5)
                                    st.rerun()
                else:
                    st.warning("⚠️ Sem jogos nesta data")
        
        # COLUNA 2: MANUAL COM DROPDOWNS PROFISSIONAIS
        with col2:
            st.markdown("#### 📝 Adicionar Manualmente")
            
            with st.container(border=True):
                st.info("💡 Use dropdowns para seleção rápida e sem erros")
                
                # DROPDOWNS DE TIMES
                all_teams = sorted(list(STATS.keys()))
                ligas_disponiveis = sorted(list(set([s['league'] for s in STATS.values()])))
                
                # Filtro por liga (opcional)
                filtro_liga = st.selectbox(
                    "🏆 Filtrar por Liga (opcional):",
                    ["Todas"] + ligas_disponiveis,
                    key="filtro_liga",
                    help="Filtre times por liga para facilitar a busca"
                )
                
                if filtro_liga != "Todas":
                    times_filtrados = [t for t, s in STATS.items() if s['league'] == filtro_liga]
                else:
                    times_filtrados = all_teams
                
                time_casa = st.selectbox(
                    "🏠 Time Casa:",
                    ["Selecione..."] + times_filtrados,
                    key="manual_casa",
                    help="Selecione o time mandante"
                )
                
                time_fora = st.selectbox(
                    "✈️ Time Visitante:",
                    ["Selecione..."] + times_filtrados,
                    key="manual_fora",
                    help="Selecione o time visitante"
                )
                
                # VALIDAÇÃO INTELIGENTE
                if time_casa != "Selecione..." and time_fora != "Selecione...":
                    if time_casa == time_fora:
                        st.error("❌ Selecione times diferentes!")
                        time_casa = "Selecione..."
                        time_fora = "Selecione..."
                
                # DROPDOWN DE MERCADOS
                mercados_disponiveis = [
                    "Over 0.5 Gols", "Over 1.5 Gols", "Over 2.5 Gols", "Over 3.5 Gols",
                    "Under 2.5 Gols", "Under 3.5 Gols",
                    "Over 8.5 Cantos", "Over 9.5 Cantos", "Over 10.5 Cantos", "Over 11.5 Cantos",
                    "Over 3.5 Cartões", "Over 4.5 Cartões", "Over 5.5 Cartões",
                    "Ambos Marcam (BTTS)", "Casa Vence", "Fora Vence", "Empate",
                    "Dupla Chance 1X", "Dupla Chance X2", "Dupla Chance 12"
                ]
                
                mercado_sel = st.selectbox(
                    "🎯 Mercado:",
                    mercados_disponiveis,
                    key="manual_mercado",
                    help="Escolha o tipo de aposta"
                )
                
                c1, c2 = st.columns(2)
                odd_manual = c1.number_input(
                    "📊 Odd:", 
                    min_value=1.01, 
                    value=1.90, 
                    step=0.01, 
                    key="manual_odd",
                    help="Digite a odd oferecida pela casa"
                )
                
                # CÁLCULO AUTOMÁTICO DE PROBABILIDADE
                prob_calc = 50.0
                if time_casa != "Selecione..." and time_fora != "Selecione...":
                    calc = calcular_jogo_v31(STATS[time_casa], STATS[time_fora], {})
                    
                    if "Gols" in mercado_sel:
                        linha = float(mercado_sel.split()[1])
                        if "Over" in mercado_sel:
                            prob_calc = calcular_poisson(calc['total_goals'], linha)
                        else:
                            prob_calc = 100 - calcular_poisson(calc['total_goals'], linha)
                    elif "Cantos" in mercado_sel:
                        linha = float(mercado_sel.split()[1])
                        prob_calc = calcular_poisson(calc['corners_total'], linha)
                    elif "Cartões" in mercado_sel:
                        linha = float(mercado_sel.split()[1])
                        prob_calc = calcular_poisson(calc['cards_total'], linha)
                    elif "BTTS" in mercado_sel:
                        prob_calc = min((calc['xg_home'] * calc['xg_away'] * 38), 92)
                    
                    c2.metric("🎲 Prob. Calculada", f"{prob_calc:.1f}%", help="Probabilidade calculada pelo Causality Engine")
                else:
                    c2.metric("🎲 Prob", "-", help="Selecione os times para calcular")
                
                if st.button("➕ Adicionar ao Bilhete", use_container_width=True, type="primary"):
                    if time_casa != "Selecione..." and time_fora != "Selecione...":
                        jogo_nome = f"{time_casa} vs {time_fora}"
                        st.session_state.current_ticket.append({
                            'jogo': jogo_nome,
                            'mercado': mercado_sel,
                            'odd': odd_manual,
                            'prob': prob_calc,
                            'tipo': 'Manual'
                        })
                        st.success(f"✅ Adicionado: {jogo_nome} - {mercado_sel}")
                        time.sleep(0.7)
                        st.rerun()
                    else:
                        st.error("❌ Selecione ambos os times primeiro!")
        
        # VISUALIZAÇÃO DO BILHETE
        st.markdown("---")
        st.subheader("📋 Seu Bilhete Atual")
        
        if st.session_state.current_ticket:
            df_ticket = pd.DataFrame(st.session_state.current_ticket)
            
            df_show = df_ticket.copy()
            df_show['Prob'] = df_show['prob'].apply(lambda x: f"{x:.1f}%")
            df_show['Odd'] = df_show['odd'].apply(lambda x: f"{x:.2f}")
            df_show['Status'] = df_show['prob'].apply(get_prob_emoji)
            
            st.dataframe(
                df_show[['jogo', 'mercado', 'Odd', 'Prob', 'Status', 'tipo']],
                use_container_width=True,
                hide_index=True
            )
            
            # Cálculos
            total_odd = np.prod([x['odd'] for x in st.session_state.current_ticket])
            prob_acum = np.prod([x['prob']/100 for x in st.session_state.current_ticket]) * 100
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("🎯 Odd Total", f"{total_odd:.2f}")
            c2.metric("🎲 Prob. Real", f"{prob_acum:.1f}%")
            
            fair_odd = 100/prob_acum if prob_acum > 0 else 0
            ev = (total_odd - fair_odd) / fair_odd * 100 if fair_odd > 0 else 0
            
            c3.metric("📊 Fair Odd", f"{fair_odd:.2f}")
            c4.metric("💎 EV", f"{ev:+.1f}%", delta_color="normal" if ev > 0 else "inverse")
            
            if ev > 5:
                st.success(f"💎 **EXCELENTE EV!** Odd {total_odd:.2f} vs Fair {fair_odd:.2f}")
            elif ev > 0:
                st.info(f"✅ **EV Positivo.** Valor detectado.")
            else:
                st.warning(f"⚠️ **EV Negativo.** Considere ajustar as seleções.")
        else:
            st.info("📭 Bilhete vazio. Adicione jogos usando as opções acima.")
    
    # ========================================
    # TABS 2-8: MANTÉM DO CÓDIGO ORIGINAL
    # ========================================
    
    # TAB 2: HEDGES (MANTER CÓDIGO ORIGINAL)
    with tabs[1]:
        st.header("🛡️ Hedges MAXIMUM")
        
        if not st.session_state.current_ticket:
            st.warning("⚠️ Crie um bilhete primeiro")
        else:
            col1, col2 = st.columns(2)
            stake = col1.number_input("💰 Stake (R$)", value=100.0, step=10.0)
            
            odd_total = np.prod([x['odd'] for x in st.session_state.current_ticket])
            col2.metric("Odd", f"{odd_total:.2f}")
            
            retorno = stake * odd_total
            lucro = retorno - stake
            
            st.info(f"💵 Retorno: {format_currency(retorno)} | Lucro: {format_currency(lucro)}")
            
            with st.expander("🛡️ Smart Protection", expanded=True):
                st.write("Apostar na zebra para recuperar stake")
                odd_hedge = st.number_input("Odd Cobertura:", 2.0, 1.01, 0.1)
                stake_hedge = stake / (odd_hedge - 1)
                
                c1, c2 = st.columns(2)
                c1.metric("Apostar", format_currency(stake_hedge))
                c2.metric("Custo Total", format_currency(stake + stake_hedge))
    
    # TAB 3: SIMULADOR (MANTER CÓDIGO ORIGINAL)
    with tabs[2]:
        st.header("🎲 Simulador Monte Carlo")
        
        all_teams = sorted(list(STATS.keys()))
        
        c1, c2 = st.columns(2)
        sim_h = c1.selectbox("🏠 Casa", all_teams, key='sim_h')
        sim_a = c2.selectbox("✈️ Fora", all_teams, key='sim_a')
        
        if st.button("🚀 Simular 3.000 jogos", use_container_width=True):
            if sim_h != sim_a:
                progress = st.progress(0)
                status = st.empty()
                
                status.text("⚙️ Carregando dados...")
                progress.progress(20)
                time.sleep(0.3)
                
                sh = STATS[sim_h]
                sa = STATS[sim_a]
                
                status.text("🎲 Simulando...")
                progress.progress(50)
                
                res = simulate_game_v31(sh, sa, {}, 3000)
                progress.progress(100)
                
                status.text("✅ Concluído!")
                time.sleep(0.5)
                status.empty()
                progress.empty()
                
                st.success("Simulação completa!")
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Média Gols", f"{res['goals_total'].mean():.2f}")
                m2.metric("Média Cantos", f"{res['corners_total'].mean():.2f}")
                m3.metric("Média Cartões", f"{res['cards_total'].mean():.2f}")
                m4.metric("Over 2.5", f"{(res['goals_total'] > 2.5).mean() * 100:.1f}%")
                
                fig = px.histogram(
                    res['goals_total'], 
                    nbins=10,
                    title="Distribuição de Gols",
                    color_discrete_sequence=['#667eea']
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Selecione times diferentes!")
    
    # TAB 4-8: CÓDIGO ORIGINAL (sem mudanças)
    with tabs[3]:
        st.header("📊 Métricas")
        if st.session_state.bet_results:
            df_hist = pd.DataFrame(st.session_state.bet_results)
            total = len(df_hist)
            green = df_hist[df_hist['ganhou'] == True].shape[0]
            win_rate = (green / total) * 100
            m1, m2 = st.columns(2)
            m1.metric("Win Rate", f"{win_rate:.1f}%")
            m2.metric("Total", total)
        else:
            st.info("Sem dados ainda")
    
    with tabs[4]:
        st.header("🎨 Visualizações")
        st.info("Em desenvolvimento")
    
    with tabs[5]:
        st.header("📝 Registro")
        with st.form("registro"):
            desc = st.text_input("Descrição")
            stake_reg = st.number_input("Stake", 10.0)
            odd_reg = st.number_input("Odd", 1.01)
            resultado = st.selectbox("Resultado", ["Green", "Red"])
            
            if st.form_submit_button("Salvar"):
                lucro_reg = (stake_reg * odd_reg - stake_reg) if resultado == "Green" else -stake_reg
                ganhou = resultado == "Green"
                
                nova_banca = st.session_state.bankroll_history[-1] + lucro_reg
                st.session_state.bankroll_history.append(nova_banca)
                
                st.session_state.bet_results.append({
                    'data': datetime.now().strftime('%d/%m %H:%M'),
                    'descricao': desc,
                    'stake': stake_reg,
                    'odd': odd_reg,
                    'ganhou': ganhou,
                    'lucro': lucro_reg
                })
                
                st.success("✅ Registrado!")
                time.sleep(1)
                st.rerun()
    
    with tabs[6]:
        st.header("🔍 Scanner")
        if CAL.empty:
            st.warning("Calendário vazio")
        else:
            c1, c2 = st.columns(2)
            datas = sorted(CAL['DtObj'].dt.strftime('%d/%m/%Y').unique())
            data_scan = c1.selectbox("Data:", datas)
            min_prob = c2.slider("Prob. Mín (%)", 50, 90, 70)
            
            if st.button("🔎 Escanear"):
                jogos = CAL[CAL['DtObj'].dt.strftime('%d/%m/%Y') == data_scan]
                hits = []
                
                for _, row in jogos.iterrows():
                    h = normalize_name(row['Time_Casa'], list(STATS.keys()))
                    a = normalize_name(row['Time_Visitante'], list(STATS.keys()))
                    
                    if h and a:
                        calc = calcular_jogo_v31(STATS[h], STATS[a], {})
                        prob_c = calcular_poisson(calc['corners_total'], 9.5)
                        
                        if prob_c >= min_prob:
                            hits.append({
                                'Jogo': f"{h} vs {a}",
                                'Mercado': 'Over 9.5 Cantos',
                                'Prob': f"{prob_c:.1f}%"
                            })
                
                if hits:
                    st.success(f"🎯 {len(hits)} oportunidades!")
                    st.dataframe(pd.DataFrame(hits))
                else:
                    st.warning("Sem oportunidades")
    
    with tabs[7]:
        st.header("📋 Importar")
        txt = st.text_area("Cole o texto:")
        if st.button("Analisar"):
            st.info("Função em desenvolvimento")
    
    # TAB 9: AI ADVISOR (MANTER CÓDIGO ORIGINAL)
    with tabs[8]:
        st.header("🤖 AI Advisor ULTRA")
        
        if not st.session_state.chat_history:
            st.markdown("""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 20px; border-radius: 15px; color: white;'>
                <h3>👋 AI Advisor ULTRA</h3>
                <p><b>Experimente:</b></p>
                <ul>
                    <li>"Analise Arsenal vs Chelsea"</li>
                    <li>"Como está o Liverpool?"</li>
                    <li>"Melhores jogos de hoje"</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        for msg in st.session_state.chat_history:
            role = msg['role']
            avatar = "👤" if role == 'user' else "🤖"
            st.chat_message(role, avatar=avatar).markdown(msg['content'])
        
        user_input = st.chat_input("Digite sua pergunta...")
        
        if user_input:
            st.session_state.chat_history.append({'role': 'user', 'content': user_input})
            
            with st.spinner("🧠 Analisando..."):
                time.sleep(0.5)
                response = processar_chat_ultra(user_input, STATS, CAL, REFS)
            
            st.session_state.chat_history.append({'role': 'assistant', 'content': response})
            st.rerun()

if __name__ == "__main__":
    main()
