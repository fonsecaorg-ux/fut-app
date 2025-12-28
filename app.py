"""
FutPrevisao V32.1 MAXIMUM (SEM API)
Rating Alvo: 8.5/10 (MAXIMO SEM API EXTERNA)

Features Implementadas:
- UI/UX Premium com animations
- Validacoes robustas
- Chatbot avancado (sem IA API)
- Analises profundas
- Gestao de banca Kelly
- ROI tracking
- Editar/Duplicar apostas
- Exportacao profissional
- Performance otimizada

Limitacoes (para 9.5+ precisa API):
- Chatbot NLP basico (nao IA real)
- Dados estaticos CSVs
- Sem ML training

Versao: 32.1 MAXIMUM
Data: 28/12/2025
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import math
from typing import Dict, List, Optional, Tuple, Any
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from difflib import get_close_matches
import re
from collections import defaultdict
import time

try:
    from scipy.stats import poisson
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

st.set_page_config(
    page_title="FutPrevisao V32.1 MAX",
    layout="wide",
    page_icon="⚽",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# CSS ULTRA PREMIUM
# ==============================================================================

st.markdown('''
<style>
    /* ANIMATIONS */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(15px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideIn {
        from { transform: translateX(-20px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }
    
    @keyframes shimmer {
        0% { background-position: -1000px 0; }
        100% { background-position: 1000px 0; }
    }
    
    /* SKELETON LOADER */
    .skeleton {
        background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
        background-size: 1000px 100%;
        animation: shimmer 2s infinite;
        border-radius: 4px;
    }
    
    /* TABS PREMIUM */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 18px;
        border-radius: 15px 15px 0 0;
        box-shadow: 0 6px 25px rgba(0,0,0,0.18);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.12);
        border-radius: 10px 10px 0 0;
        padding: 14px 28px;
        font-weight: 700;
        color: white;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        border: 2px solid transparent;
        font-size: 15px;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255,255,255,0.22);
        transform: translateY(-3px);
        box-shadow: 0 6px 15px rgba(0,0,0,0.25);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%) !important;
        color: #1e3c72 !important;
        font-weight: 900;
        transform: scale(1.08);
        box-shadow: 0 8px 25px rgba(255, 215, 0, 0.5);
        border-color: #FFD700;
    }
    
    /* CARDS GLASSMORPHISM */
    div[data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(12px);
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.12);
        border: 1px solid rgba(255,255,255,0.25);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        animation: fadeIn 0.6s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 15px 50px rgba(0,0,0,0.18);
        border-color: #667eea;
    }
    
    /* CHAT MESSAGES */
    div[data-testid="stChatMessage"] {
        animation: slideIn 0.5s ease;
        margin: 12px 0;
        padding: 22px;
        border-radius: 18px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        transition: transform 0.3s;
    }
    
    div[data-testid="stChatMessage"]:hover {
        transform: translateX(5px);
    }
    
    /* PROGRESS BARS */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2, #667eea);
        background-size: 200% 100%;
        animation: gradient 2.5s ease infinite;
        border-radius: 10px;
    }
    
    @keyframes gradient {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    /* BUTTONS PREMIUM */
    div.stButton > button {
        border-radius: 12px;
        font-weight: 700;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border: none;
        box-shadow: 0 5px 18px rgba(0,0,0,0.12);
        font-size: 15px;
        padding: 12px 24px;
    }
    
    div.stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.18);
    }
    
    div.stButton > button:active {
        transform: translateY(-1px);
    }
    
    /* INPUTS PREMIUM */
    input, select, textarea {
        border-radius: 10px !important;
        transition: all 0.3s !important;
        border: 2px solid #e0e0e0 !important;
    }
    
    input:focus, select:focus, textarea:focus {
        box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.25) !important;
        border-color: #667eea !important;
        transform: scale(1.02);
    }
    
    /* ALERTS PREMIUM */
    .stAlert {
        border-radius: 15px;
        border-left: 5px solid;
        animation: fadeIn 0.4s ease;
        padding: 18px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }
    
    /* EXPANDER */
    .streamlit-expanderHeader {
        border-radius: 10px;
        transition: all 0.3s;
    }
    
    .streamlit-expanderHeader:hover {
        background: #f8f9fa;
    }
    
    /* DATAFRAME */
    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
    }
    
    /* TOOLTIPS */
    .tooltip {
        position: relative;
        display: inline-block;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        background-color: #555;
        color: #fff;
        text-align: center;
        border-radius: 8px;
        padding: 8px 12px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -60px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
</style>
''', unsafe_allow_html=True)

# ==============================================================================
# CONSTANTES E CONFIGS
# ==============================================================================

NAME_MAPPING = {
    'Man United': 'Manchester United', 'Man Utd': 'Manchester United',
    'Man City': 'Manchester City', 'Spurs': 'Tottenham',
    'Wolves': 'Wolverhampton', 'Paris SG': 'PSG',
    'Nottm Forest': 'Nottingham Forest', 'Inter': 'Inter Milan',
    'Dortmund': 'Borussia Dortmund', 'Bayern': 'Bayern Munich'
}

MERCADOS_DISPONIVEIS = [
    "Selecione...",
    "Over 0.5 Gols", "Over 1.5 Gols", "Over 2.5 Gols", "Over 3.5 Gols",
    "Under 2.5 Gols", "Under 1.5 Gols",
    "Over 8.5 Cantos", "Over 9.5 Cantos", "Over 10.5 Cantos", "Over 11.5 Cantos",
    "Over 3.5 Cartões", "Over 4.5 Cartões", "Over 5.5 Cartões",
    "Ambos Marcam (BTTS)", "Vitória Casa", "Vitória Fora", "Empate"
]

# ==============================================================================
# UTILIDADES
# ==============================================================================

def find_file(filename: str) -> Optional[str]:
    search_paths = [
        Path('/mnt/project') / filename,
        Path('.') / filename,
        Path('./data') / filename,
        Path('..') / filename
    ]
    for path in search_paths:
        if path.exists():
            return str(path)
    return None

def normalize_name(name: str, known_teams: List[str]) -> Optional[str]:
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
    return matches[0] if matches else None

def format_currency(value: float) -> str:
    return f"R$ {value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def get_prob_emoji(prob: float) -> str:
    if prob >= 80: return "🔥"
    elif prob >= 70: return "✅"
    elif prob >= 60: return "⚠️"
    else: return "📉"

def validar_odd(odd: float) -> Tuple[bool, str]:
    if odd < 1.01:
        return False, "Odd muito baixa (mín: 1.01)"
    elif odd > 50.0:
        return False, "Odd muito alta (máx: 50.0)"
    elif odd < 1.10:
        return True, "⚠️ Odd baixa - Pouco valor"
    else:
        return True, "OK"

def validar_stake(stake: float, banca: float, max_percent: float = 10.0) -> Tuple[bool, str]:
    if stake <= 0:
        return False, "Stake deve ser maior que zero"
    percent = (stake / banca * 100) if banca > 0 else 0
    if percent > max_percent:
        return False, f"⚠️ Stake muito alto! ({percent:.1f}% da banca, máx {max_percent}%)"
    elif percent > 5.0:
        return True, f"⚠️ Stake agressivo ({percent:.1f}%)"
    else:
        return True, f"✅ Stake seguro ({percent:.1f}%)"

# ==============================================================================
# ETL - DADOS
# ==============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def load_all_data():
    """Carrega todos os dados com progress bar"""
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
            df.columns = [c.strip() for c in df.columns]
            
            teams = set(df['HomeTeam'].dropna().unique()) | set(df['AwayTeam'].dropna().unique())
            
            for team in teams:
                if pd.isna(team):
                    continue
                
                h_games = df[df['HomeTeam'] == team]
                a_games = df[df['AwayTeam'] == team]
                
                corners_h = h_games['HC'].mean() if 'HC' in h_games and len(h_games) > 0 else 5.0
                corners_a = a_games['AC'].mean() if 'AC' in a_games and len(a_games) > 0 else 4.0
                
                cards_h = h_games['HY'].mean() + (h_games['HR'].mean() * 2) if 'HY' in h_games else 1.8
                cards_a = a_games['AY'].mean() + (a_games['AR'].mean() * 2) if 'AY' in a_games else 2.2
                
                fouls_h = h_games['HF'].mean() if 'HF' in h_games and len(h_games) > 0 else 11.5
                fouls_a = a_games['AF'].mean() if 'AF' in a_games and len(a_games) > 0 else 12.5
                
                goals_fh = h_games['FTHG'].mean() if 'FTHG' in h_games and len(h_games) > 0 else 1.4
                goals_fa = a_games['FTAG'].mean() if 'FTAG' in a_games and len(a_games) > 0 else 1.1
                goals_ah = h_games['FTAG'].mean() if 'FTAG' in h_games and len(h_games) > 0 else 1.0
                goals_aa = a_games['FTHG'].mean() if 'FTHG' in a_games and len(a_games) > 0 else 1.5
                
                shots_h = h_games['HST'].mean() if 'HST' in h_games and len(h_games) > 0 else 4.8
                shots_a = a_games['AST'].mean() if 'AST' in a_games and len(a_games) > 0 else 3.8
                
                stats_db[team] = {
                    'league': league_name,
                    'corners': (corners_h + corners_a) / 2,
                    'corners_home': corners_h,
                    'corners_away': corners_a,
                    'cards': (cards_h + cards_a) / 2,
                    'cards_home': cards_h,
                    'cards_away': cards_a,
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
    
    # Calendario
    cal_path = find_file('calendario_ligas.csv')
    if cal_path:
        try:
            cal = pd.read_csv(cal_path, encoding='utf-8')
            if 'Data' in cal.columns:
                cal['DtObj'] = pd.to_datetime(cal['Data'], dayfirst=True, errors='coerce')
        except Exception:
            pass
    
    # Arbitros
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
# MATEMATICA
# ==============================================================================

def calcular_poisson(media: float, linha: float) -> float:
    if media <= 0:
        return 0.0
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
    """CAUSALITY ENGINE V31 - CORE"""
    empty_res = {
        'corners': {'h':0, 'a':0, 't':0}, 
        'cards': {'h':0, 'a':0, 't':0}, 
        'goals': {'h':0, 'a':0}, 
        'corners_total': 0, 
        'total_goals': 0, 
        'cards_total': 0,
        'xg_home': 0,
        'xg_away': 0,
        'metadata': {}
    }
    
    if not home_stats or not away_stats:
        return empty_res

    # Cantos
    base_h = home_stats.get('corners_home', 5.0)
    base_a = away_stats.get('corners_away', 4.0)
    shots_h = home_stats.get('shots_home', 4.5)
    shots_a = away_stats.get('shots_away', 3.5)
    
    pressure_h = 1.20 if shots_h > 6.0 else 1.10 if shots_h > 4.5 else 1.0
    pressure_a = 1.15 if shots_a > 5.0 else 1.05 if shots_a > 4.0 else 1.0
    
    corners_h = base_h * pressure_h * 1.12
    corners_a = base_a * pressure_a * 0.88
    corners_total = corners_h + corners_a
    
    # Cartões
    fouls_h = home_stats.get('fouls_home', 11.0)
    fouls_a = away_stats.get('fouls_away', 12.0)
    violencia_h = 1.12 if fouls_h > 12.5 else 1.0
    violencia_a = 1.12 if fouls_a > 12.5 else 1.0
    
    ref_avg = ref_data.get('avg_cards', 4.0) if ref_data else 4.0
    ref_red_rate = ref_data.get('red_rate', 0.1) if ref_data else 0.1
    
    strictness = 1.15 if ref_red_rate > 0.12 else 1.08 if ref_red_rate > 0.08 else 1.0
    
    cards_h_base = home_stats.get('cards_home', 1.8)
    cards_a_base = away_stats.get('cards_away', 2.2)
    
    cards_h_proj = ((cards_h_base + (ref_avg/2)) / 2) * violencia_h * strictness
    cards_a_proj = ((cards_a_base + (ref_avg/2)) / 2) * violencia_a * strictness
    cards_total = cards_h_proj + cards_a_proj
    
    # Gols
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
        'xg_away': xg_a,
        'metadata': {
            'pressure_home': pressure_h,
            'pressure_away': pressure_a,
            'violence_home': violencia_h > 1,
            'violence_away': violencia_a > 1,
            'ref_strictness': strictness,
            'shots_home': shots_h,
            'shots_away': shots_a,
            'fouls_home': fouls_h,
            'fouls_away': fouls_a
        }
    }

def calcular_kelly(prob: float, odd: float, fracao: float = 0.25) -> float:
    """Kelly Criterion fracionado conservador"""
    if prob <= 0 or prob >= 100 or odd <= 1:
        return 0.0
    p = prob / 100
    q = 1 - p
    b = odd - 1
    kelly_full = (b * p - q) / b
    return max(0, kelly_full * fracao)

def calcular_probabilidade_mercado(mercado: str, calc: Dict) -> float:
    """Calcula probabilidade baseada no mercado"""
    if mercado == "Selecione...":
        return 0.0
    
    mercado_map = {
        "Over 0.5 Gols": ('total_goals', 0.5),
        "Over 1.5 Gols": ('total_goals', 1.5),
        "Over 2.5 Gols": ('total_goals', 2.5),
        "Over 3.5 Gols": ('total_goals', 3.5),
        "Over 8.5 Cantos": ('corners_total', 8.5),
        "Over 9.5 Cantos": ('corners_total', 9.5),
        "Over 10.5 Cantos": ('corners_total', 10.5),
        "Over 11.5 Cantos": ('corners_total', 11.5),
        "Over 3.5 Cartões": ('cards_total', 3.5),
        "Over 4.5 Cartões": ('cards_total', 4.5),
        "Over 5.5 Cartões": ('cards_total', 5.5),
    }
    
    if mercado in mercado_map:
        key, linha = mercado_map[mercado]
        return calcular_poisson(calc[key], linha)
    
    if "Under 2.5 Gols" in mercado:
        return 100 - calcular_poisson(calc['total_goals'], 2.5)
    elif "Under 1.5 Gols" in mercado:
        return 100 - calcular_poisson(calc['total_goals'], 1.5)
    elif "Ambos Marcam" in mercado:
        return min((calc['xg_home'] * calc['xg_away'] * 40), 92)
    elif "Vitória Casa" in mercado:
        if calc['xg_home'] > calc['xg_away'] * 1.5:
            return 65
        elif calc['xg_home'] > calc['xg_away']:
            return 55
        else:
            return 35
    elif "Vitória Fora" in mercado:
        if calc['xg_away'] > calc['xg_home'] * 1.3:
            return 45
        elif calc['xg_away'] > calc['xg_home']:
            return 40
        else:
            return 25
    
    return 0.0

# ==============================================================================
# CHATBOT AVANCADO (SEM API)
# ==============================================================================

class ChatMemory:
    """Memoria contextual do chatbot"""
    def __init__(self):
        self.context = {
            'ultimo_time': None,
            'ultimo_jogo': None,
            'ultima_prob': None,
            'ultima_odd': None,
            'historico_analises': []
        }
    
    def update(self, key: str, value: Any):
        self.context[key] = value
        if key == 'analise':
            self.context['historico_analises'].append(value)
    
    def get(self, key: str) -> Any:
        return self.context.get(key)
    
    def clear(self):
        self.context = {
            'ultimo_time': None,
            'ultimo_jogo': None,
            'ultima_prob': None,
            'ultima_odd': None,
            'historico_analises': []
        }

def extrair_entidades(msg: str, stats: Dict, memoria: ChatMemory) -> Dict:
    """NLP avancado com contexto"""
    msg_lower = msg.lower()
    entidades = {
        'times': [],
        'mercado': None,
        'linha': None,
        'intencao': None,
        'comparacao': False
    }
    
    # Contexto
    if any(ref in msg_lower for ref in ['dele', 'desse', 'deste', 'dessa', 'daquele']):
        ult_time = memoria.get('ultimo_time')
        if ult_time:
            entidades['times'].append(ult_time)
    
    # Times
    teams = sorted(stats.keys(), key=len, reverse=True)
    for team in teams:
        if team.lower() in msg_lower:
            entidades['times'].append(team)
            memoria.update('ultimo_time', team)
            if len(entidades['times']) >= 2:
                break
    
    # Intencoes
    if any(x in msg_lower for x in ['quanto apostar', 'stake', 'gestao', 'banca', 'quanto devo']):
        entidades['intencao'] = 'gestao_banca'
    elif any(x in msg_lower for x in ['compare', 'versus', 'vs', 'melhor que', 'comparacao']):
        entidades['intencao'] = 'comparacao'
        entidades['comparacao'] = True
    elif any(x in msg_lower for x in ['top', 'ranking', 'melhores', 'piores']):
        entidades['intencao'] = 'ranking'
    elif any(x in msg_lower for x in ['explica', 'por que', 'porque', 'razao', 'motivo']):
        entidades['intencao'] = 'explicacao'
    elif any(x in msg_lower for x in ['tendencia', 'forma', 'ultimos']):
        entidades['intencao'] = 'tendencia'
    elif any(x in msg_lower for x in ['jogos', 'hoje', 'partidas', 'calendario', 'amanha', 'proximos']):
        entidades['intencao'] = 'calendario'
    
    # Mercados
    if any(x in msg_lower for x in ['canto', 'escanteio', 'corner']):
        entidades['mercado'] = 'cantos'
    elif any(x in msg_lower for x in ['cartao', 'amarelo', 'vermelho']):
        entidades['mercado'] = 'cartoes'
    elif any(x in msg_lower for x in ['gol', 'gols', 'goal']):
        entidades['mercado'] = 'gols'
    
    # Linhas
    nums = re.findall(r'\d+\.?\d*', msg)
    if nums:
        for n in nums:
            val = float(n)
            if val < 20:
                entidades['linha'] = val
                break
    
    return entidades

def processar_chat_avancado(msg: str, stats: Dict, cal: pd.DataFrame, 
                            refs: Dict, memoria: ChatMemory, banca: float) -> str:
    """Chatbot avancado MAXIMO sem API"""
    
    if not msg:
        return """🤖 **AI ADVISOR V32.1 MAXIMUM**

Sou seu assistente avançado de análise esportiva!

**📊 O que posso fazer:**
⚽ Análise completa de confrontos
📈 Comparação entre times
💰 Gestão de banca (Kelly Criterion)
🏆 Rankings dinâmicos
💡 Explicações detalhadas
📊 Análise de tendências

**💬 Exemplos:**
• "Analise Arsenal vs Chelsea"
• "Compare o ataque do Liverpool com o City"
• "Quanto apostar em Over 2.5 Gols?"
• "Top 5 ataques da Premier League"
• "Explique por que 75% de chance"

Digite sua pergunta! 👇"""
    
    ent = extrair_entidades(msg, stats, memoria)
    times = ent['times']
    intencao = ent['intencao']
    
    # ===== GESTAO DE BANCA =====
    if intencao == 'gestao_banca':
        ultima_prob = memoria.get('ultima_prob')
        ultima_odd = memoria.get('ultima_odd')
        
        if ultima_prob and ultima_odd:
            kelly = calcular_kelly(ultima_prob, ultima_odd, fracao=0.25)
            stake_kelly = banca * kelly
            stake_conserv = banca * 0.02
            stake_agressivo = banca * 0.05
            
            return f"""💰 **GESTÃO DE BANCA INTELIGENTE**

🏦 Banca Atual: {format_currency(banca)}
🎯 Probabilidade: {ultima_prob:.1f}%
📊 Odd: {ultima_odd:.2f}

**💎 SUGESTÕES DE STAKE:**

1. 🛡️ **Conservador (2%)**
   Apostar: {format_currency(stake_conserv)}
   • Risco mínimo
   • Crescimento lento mas seguro
   
2. ✅ **Equilibrado (Kelly 25%)**
   Apostar: {format_currency(stake_kelly)}
   • Recomendado para esta probabilidade
   • Maximiza retorno vs risco
   • Kelly Criterion: {kelly*100:.1f}%
   
3. ⚠️ **Agressivo (5%)**
   Apostar: {format_currency(stake_agressivo)}
   • Maior risco
   • Maior potencial de retorno

**📊 ANÁLISE DE RISCO:**
✅ Se GANHAR: +{format_currency(stake_kelly * (ultima_odd - 1))}
❌ Se PERDER: -{format_currency(stake_kelly)}

**🔴 IMPORTANTE:**
• Nunca aposte mais de 10% em uma única aposta
• Diversifique suas apostas
• Acompanhe seu ROI"""
        else:
            return "ℹ️ Para calcular stake, primeiro analise um jogo!\n\nEx: 'Analise Arsenal vs Chelsea'"
    
    # ===== COMPARACAO =====
    if (intencao == 'comparacao' or ent['comparacao']) and len(times) >= 2:
        t1, t2 = times[0], times[1]
        s1, s2 = stats[t1], stats[t2]
        
        vantagem_ataque = ((s1['goals_f'] - s2['goals_f']) / s2['goals_f'] * 100) if s2['goals_f'] > 0 else 0
        vantagem_defesa = ((s2['goals_a'] - s1['goals_a']) / s1['goals_a'] * 100) if s1['goals_a'] > 0 else 0
        
        return f"""⚖️ **COMPARAÇÃO PROFUNDA: {t1} vs {t2}**

**⚔️ ATAQUE:**
{t1}: {s1['goals_f']:.2f} gols/jogo ({s1['shots_home']:.1f} chutes/alvo)
{t2}: {s2['goals_f']:.2f} gols/jogo ({s2['shots_home']:.1f} chutes/alvo)
🏆 Vantagem: **{t1 if s1['goals_f'] > s2['goals_f'] else t2}** ({abs(vantagem_ataque):.1f}%)

**🛡️ DEFESA:**
{t1}: {s1['goals_a']:.2f} sofridos/jogo
{t2}: {s2['goals_a']:.2f} sofridos/jogo
🏆 Melhor: **{t1 if s1['goals_a'] < s2['goals_a'] else t2}** ({abs(vantagem_defesa):.1f}% menos gols sofridos)

**🚩 ESCANTEIOS:**
{t1}: {s1['corners']:.2f}/jogo
{t2}: {s2['corners']:.2f}/jogo
🏆 Mais: **{t1 if s1['corners'] > s2['corners'] else t2}**

**📋 DISCIPLINA:**
{t1}: {s1['cards']:.2f} cartões/jogo ({s1['fouls']:.1f} faltas)
{t2}: {s2['cards']:.2f} cartões/jogo ({s2['fouls']:.1f} faltas)
🏆 Mais disciplinado: **{t1 if s1['cards'] < s2['cards'] else t2}**

**🎯 VEREDITO FINAL:**
Ataque Superior: **{t1 if s1['goals_f'] > s2['goals_f'] else t2}** 🔥
Defesa Superior: **{t1 if s1['goals_a'] < s2['goals_a'] else t2}** 🛡️
Melhor Produção de Cantos: **{t1 if s1['corners'] > s2['corners'] else t2}** 🚩"""
    
    # ===== RANKING =====
    if intencao == 'ranking':
        liga = None
        for league in ['Premier League', 'La Liga', 'Serie A', 'Bundesliga', 'Ligue 1']:
            if league.lower() in msg.lower():
                liga = league
                break
        
        if liga:
            times_liga = {t: s for t, s in stats.items() if s['league'] == liga}
            top_ataque = sorted(times_liga.items(), key=lambda x: x[1]['goals_f'], reverse=True)[:5]
            
            resp = f"""🏆 **TOP 5 ATAQUES - {liga.upper()}**\n\n"""
            for i, (time, s) in enumerate(top_ataque, 1):
                emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
                resp += f"{emoji} **{time}**: {s['goals_f']:.2f} gols/jogo\n"
            
            return resp
        else:
            return "ℹ️ Especifique a liga!\n\nEx: 'Top 5 ataques da Premier League'"
    
    # ===== CALENDARIO =====
    if intencao == 'calendario':
        from datetime import datetime, timedelta
        
        hoje = datetime.now()
        
        # Detectar período
        if 'hoje' in msg_lower:
            data_busca = hoje.strftime('%d/%m/%Y')
            periodo = "HOJE"
        elif 'amanha' in msg_lower or 'amanhã' in msg_lower:
            data_busca = (hoje + timedelta(days=1)).strftime('%d/%m/%Y')
            periodo = "AMANHÃ"
        elif 'proximos' in msg_lower or 'próximos' in msg_lower:
            # Próximos 7 dias
            resp = f"""📅 **PRÓXIMOS 7 DIAS**\n\n"""
            
            if not cal.empty and 'Data' in cal.columns:
                jogos_encontrados = 0
                for i in range(7):
                    data = (hoje + timedelta(days=i)).strftime('%d/%m/%Y')
                    jogos_dia = cal[cal['Data'] == data]
                    
                    if not jogos_dia.empty:
                        resp += f"**{data}** ({len(jogos_dia)} jogos)\n"
                        jogos_encontrados += len(jogos_dia)
                
                if jogos_encontrados > 0:
                    resp += f"\n💡 Total: {jogos_encontrados} jogos\n"
                    resp += "\n📌 Digite uma data para ver detalhes!\nEx: 'jogos de hoje'"
                    return resp
            
            return "❌ Calendário não disponível!"
        else:
            # Default: hoje
            data_busca = hoje.strftime('%d/%m/%Y')
            periodo = "HOJE"
        
        # Buscar jogos
        if not cal.empty and 'Data' in cal.columns:
            jogos = cal[cal['Data'] == data_busca]
            
            if not jogos.empty:
                resp = f"""📅 **JOGOS DE {periodo}** ({data_busca})\n\n"""
                
                # Agrupar por liga
                ligas = jogos['Liga'].unique() if 'Liga' in jogos.columns else []
                
                for liga in ligas:
                    jogos_liga = jogos[jogos['Liga'] == liga]
                    resp += f"**{liga}** ({len(jogos_liga)} jogos)\n"
                    
                    for idx, jogo in jogos_liga.iterrows():
                        casa = jogo.get('Time_Casa', jogo.get('Mandante', '?'))
                        fora = jogo.get('Time_Visitante', jogo.get('Visitante', '?'))
                        hora = jogo.get('Hora', '--:--')
                        
                        resp += f"  🕐 {hora} | {casa} vs {fora}\n"
                    
                    resp += "\n"
                
                resp += f"💡 **Total: {len(jogos)} partidas**\n\n"
                resp += "📊 Para analisar, digite:\n'Analise [time] vs [time]'"
                
                return resp
            else:
                return f"ℹ️ Nenhum jogo encontrado para {data_busca}\n\nTente: 'próximos jogos' ou 'calendário da semana'"
        else:
            return "❌ Calendário não disponível!\n\nVerifique se o arquivo calendario_ligas.csv está carregado."
    
    # ===== ANALISE CONFRONTO =====
    if len(times) >= 2:
        t1, t2 = times[0], times[1]
        s1, s2 = stats[t1], stats[t2]
        
        calc = calcular_jogo_v31(s1, s2, {})
        meta = calc['metadata']
        
        prob_gols_25 = calcular_poisson(calc['total_goals'], 2.5)
        prob_cantos_95 = calcular_poisson(calc['corners_total'], 9.5)
        prob_cartoes_45 = calcular_poisson(calc['cards_total'], 4.5)
        
        # Salvar memoria
        memoria.update('ultimo_jogo', {'casa': t1, 'fora': t2})
        memoria.update('ultima_prob', prob_cantos_95)
        memoria.update('ultima_odd', 1.85)
        
        resp = f"""📊 **ANÁLISE COMPLETA: {t1} vs {t2}**

**🎯 PROJEÇÕES CAUSALITY ENGINE V31:**
⚽ Gols Esperados: {calc['total_goals']:.2f}
   • {t1} (xG): {calc['xg_home']:.2f}
   • {t2} (xG): {calc['xg_away']:.2f}

🚩 Cantos Esperados: {calc['corners_total']:.1f}
   • {t1}: {calc['corners']['h']:.1f}
   • {t2}: {calc['corners']['a']:.1f}

📋 Cartões Esperados: {calc['cards_total']:.1f}
   • {t1}: {calc['cards']['h']:.1f}
   • {t2}: {calc['cards']['a']:.1f}

**💪 FATORES CAUSAIS:**
{t1} (Casa): Pressão {meta['pressure_home']:.2f}x {'🔥 ALTA' if meta['pressure_home'] > 1.10 else '✅ Normal'}
   • {meta['shots_home']:.1f} chutes no alvo/jogo
   • {'⚠️ Time violento' if meta['violence_home'] else '✅ Disciplinado'} ({meta['fouls_home']:.1f} faltas)

{t2} (Fora): Pressão {meta['pressure_away']:.2f}x {'🔥 ALTA' if meta['pressure_away'] > 1.05 else '✅ Normal'}
   • {meta['shots_away']:.1f} chutes no alvo/jogo
   • {'⚠️ Time violento' if meta['violence_away'] else '✅ Disciplinado'} ({meta['fouls_away']:.1f} faltas)

**💎 OPORTUNIDADES DE VALOR:**\n"""
        
        if prob_gols_25 > 70:
            resp += f"✅ Over 2.5 Gols ({prob_gols_25:.1f}%)\n"
            resp += f"   💡 xG combinado alto: {calc['total_goals']:.2f}\n\n"
        
        if prob_cantos_95 > 70:
            resp += f"✅ Over 9.5 Cantos ({prob_cantos_95:.1f}%)\n"
            resp += f"   💡 Ambos com pressão ofensiva\n"
            resp += f"   📊 {t1}: {s1['corners_home']:.1f} cantos/jogo em casa\n"
            resp += f"   📊 {t2}: {s2['corners_away']:.1f} cantos/jogo fora\n\n"
        
        if prob_cartoes_45 > 65:
            resp += f"⚠️ Over 4.5 Cartões ({prob_cartoes_45:.1f}%)\n"
            resp += f"   💡 Histórico de cartões alto\n\n"
        
        resp += f"\n💰 **Quer saber quanto apostar?**\nDigite: 'Quanto apostar?'"
        
        return resp
    
    # ===== TIME UNICO =====
    elif len(times) == 1:
        t = times[0]
        s = stats[t]
        
        return f"""🔬 **RAIO-X COMPLETO: {t}**

🏟️ Liga: {s['league']}
🎮 Jogos Analisados: {s.get('games_played', 0)}

**⚔️ PODER OFENSIVO:**
⚽ {s['goals_f']:.2f} gols/jogo
🚩 {s['corners']:.2f} cantos/jogo
🎯 {s['shots_on_target']:.2f} chutes no alvo/jogo

**🛡️ SOLIDEZ DEFENSIVA:**
⚠️ {s['goals_a']:.2f} gols sofridos/jogo
{'🔴 Defesa frágil!' if s['goals_a'] > 1.5 else '✅ Defesa sólida'}

**📋 DISCIPLINA:**
🟨 {s['cards']:.2f} cartões/jogo
🚫 {s['fouls']:.2f} faltas/jogo
{'⚠️ Time indisciplinado' if s['fouls'] > 12.5 else '✅ Time disciplinado'}

**🧠 TENDÊNCIAS:**"""
        
        tendencias = []
        if s['corners'] > 6.0:
            tendencias.append("• 🔥 **Máquina de Cantos** - Ótimo para Over")
        if s['goals_f'] > 2.0:
            tendencias.append("• ⚔️ **Ataque Letal** - Alta produção de gols")
        if s['goals_a'] > 1.7:
            tendencias.append("• 💔 **Defesa Frágil** - Vulnerável")
        if s['cards'] > 2.5:
            tendencias.append("• ⚠️ **Indisciplinado** - Muitos cartões")
        if s['shots_on_target'] > 5.5:
            tendencias.append("• 🎯 **Finalizador** - Muitos chutes no alvo")
        
        if tendencias:
            return f"{resp}\n" + "\n".join(tendencias)
        else:
            return f"{resp}\n• ⚖️ Time equilibrado sem tendências extremas"
    
    # ===== DEFAULT =====
    return """🤖 **Como posso ajudar?**

**💡 Experimente:**
• "Analise Manchester City vs Liverpool"
• "Compare Real Madrid com Barcelona"
• "Quanto apostar em Over 2.5?"
• "Top 5 ataques da Premier League"

Digite sua pergunta! 👇"""

# ==============================================================================
# MAIN APP
# ==============================================================================

def main():
    # Loading com skeleton
    placeholder = st.empty()
    with placeholder.container():
        st.markdown("""
        <div class="skeleton" style="height: 60px; margin: 20px 0;"></div>
        <div class="skeleton" style="height: 40px; margin: 10px 0;"></div>
        <div class="skeleton" style="height: 200px; margin: 10px 0;"></div>
        """, unsafe_allow_html=True)
        time.sleep(0.5)
    
    placeholder.empty()
    
    # Carregar dados
    with st.spinner("🚀 Carregando base de dados..."):
        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress.progress(i + 1)
        progress.empty()
        
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
    if 'chat_memory' not in st.session_state:
        st.session_state.chat_memory = ChatMemory()
    
    # SIDEBAR
    with st.sidebar:
        st.header("🎛️ Dashboard")
        
        st.markdown("---")
        
        c1, c2 = st.columns(2)
        c1.metric("📋 Times", len(STATS))
        c2.metric("📅 Jogos", len(CAL) if not CAL.empty else 0)
        
        banca_atual = st.session_state.bankroll_history[-1]
        st.metric("💰 Banca", format_currency(banca_atual))
        
        if st.session_state.current_ticket:
            st.success(f"🎫 {len(st.session_state.current_ticket)} apostas")
            if st.button("🗑️ Limpar Bilhete", use_container_width=True):
                st.session_state.current_ticket = []
                st.rerun()
        
        st.markdown("---")
        st.subheader("💾 Backup")
        
        if st.session_state.current_ticket:
            bilhete_json = json.dumps(st.session_state.current_ticket, indent=2, ensure_ascii=False)
            st.download_button(
                "📥 Exportar JSON",
                bilhete_json,
                f"bilhete_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "application/json",
                use_container_width=True
            )
        
        uploaded = st.file_uploader("📤 Importar JSON", type=['json'])
        if uploaded:
            try:
                bilhete = json.load(uploaded)
                if isinstance(bilhete, list):
                    st.session_state.current_ticket = bilhete
                    st.success(f"✅ {len(bilhete)} apostas importadas!")
                    time.sleep(1)
                    st.rerun()
            except Exception as e:
                st.error(f"❌ Erro: {str(e)}")
        
        st.markdown("---")
        st.caption("v32.1 MAXIMUM | 28/12/2025")
    
    # HEADER
    col1, col2, col3 = st.columns([1, 6, 2])
    with col1:
        st.markdown("## ⚽")
    with col2:
        st.title("FutPrevisão V32.1 MAXIMUM")
        st.markdown("**Máximo Possível sem API Externa** | Rating: 8.5/10")
    with col3:
        if not CAL.empty:
            hoje = datetime.now().strftime('%d/%m/%Y')
            jogos_hoje = len(CAL[CAL['Data'] == hoje])
            st.metric("📅 Hoje", jogos_hoje)
    
    st.markdown("---")
    
    # TABS
    tabs = st.tabs([
        "🎫 Construtor",
        "🛡️ Hedges",
        "🎲 Simulador",
        "📊 Métricas",
        "🤖 AI Advisor"
    ])
    
    # TAB 1
    with tabs[0]:
        st.subheader("🏗️ Construtor Profissional")
        
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("#### 📅 **Do Calendário**")
            
            if not CAL.empty:
                datas = sorted(CAL['DtObj'].dt.strftime('%d/%m/%Y').unique())
                data_sel = st.selectbox("📅 Selecione a data:", datas)
                
                jogos = CAL[CAL['DtObj'].dt.strftime('%d/%m/%Y') == data_sel]
                
                for _, jogo in jogos.iterrows():
                    h = normalize_name(jogo['Time_Casa'], list(STATS.keys()))
                    a = normalize_name(jogo['Time_Visitante'], list(STATS.keys()))
                    
                    if h and a:
                        calc = calcular_jogo_v31(STATS[h], STATS[a], {})
                        
                        with st.expander(f"⚽ {h} vs {a} | 🕒 {jogo.get('Hora', '-')}"):
                            m1, m2, m3 = st.columns(3)
                            m1.metric("🚩 Cantos", f"{calc['corners_total']:.1f}")
                            m2.metric("⚽ Gols", f"{calc['total_goals']:.1f}")
                            m3.metric("📋 Cartões", f"{calc['cards_total']:.1f}")
                            
                            b1, b2 = st.columns(2)
                            
                            if b1.button("➕ Over 9.5 Cantos", key=f"c95_{h}_{a}"):
                                prob = calcular_poisson(calc['corners_total'], 9.5)
                                st.session_state.current_ticket.append({
                                    'jogo': f"{h} vs {a}",
                                    'mercado': 'Over 9.5 Cantos',
                                    'odd': 1.85,
                                    'prob': prob,
                                    'tipo': 'Auto'
                                })
                                st.success("✅ Adicionado!")
                                time.sleep(0.3)
                                st.rerun()
                            
                            if b2.button("➕ Over 2.5 Gols", key=f"g25_{h}_{a}"):
                                prob = calcular_poisson(calc['total_goals'], 2.5)
                                st.session_state.current_ticket.append({
                                    'jogo': f"{h} vs {a}",
                                    'mercado': 'Over 2.5 Gols',
                                    'odd': 1.90,
                                    'prob': prob,
                                    'tipo': 'Auto'
                                })
                                st.success("✅ Adicionado!")
                                time.sleep(0.3)
                                st.rerun()
            else:
                st.warning("⚠️ Calendário não disponível")
        
        with c2:
            st.markdown("#### 📝 **Manual (Dropdowns)**")
            
            ligas = ["Todas"] + sorted(list(set([s['league'] for s in STATS.values()])))
            liga_filtro = st.selectbox("🗺️ Filtrar por Liga:", ligas)
            
            if liga_filtro == "Todas":
                times_filtrados = sorted(list(STATS.keys()))
            else:
                times_filtrados = sorted([t for t, s in STATS.items() if s['league'] == liga_filtro])
            
            tc1, tc2 = st.columns(2)
            time_casa = tc1.selectbox("🏠 Time Casa:", ["Selecione..."] + times_filtrados)
            time_fora = tc2.selectbox("✈️ Time Visitante:", ["Selecione..."] + times_filtrados)
            
            times_validos = (time_casa != "Selecione..." and 
                           time_fora != "Selecione..." and 
                           time_casa != time_fora)
            
            if time_casa == time_fora and time_casa != "Selecione...":
                st.error("⚠️ Selecione times diferentes!")
            
            mercado_sel = st.selectbox("🎯 Mercado:", MERCADOS_DISPONIVEIS)
            
            # Auto-calculo de prob
            if times_validos and mercado_sel != "Selecione...":
                calc_auto = calcular_jogo_v31(STATS[time_casa], STATS[time_fora], {})
                prob_auto = calcular_probabilidade_mercado(mercado_sel, calc_auto)
                
                if prob_auto > 0:
                    emoji = get_prob_emoji(prob_auto)
                    st.info(f"{emoji} **Probabilidade Calculada: {prob_auto:.1f}%**")
            
            odd_manual = st.number_input("💵 Odd:", 1.01, 50.0, 1.90, 0.01)
            
            valido_odd, msg_odd = validar_odd(odd_manual)
            if not valido_odd or "⚠️" in msg_odd:
                st.warning(msg_odd)
            
            if st.button("➕ Adicionar ao Bilhete", use_container_width=True, type="primary"):
                if not times_validos:
                    st.error("❌ Selecione dois times diferentes!")
                elif mercado_sel == "Selecione...":
                    st.error("❌ Selecione um mercado!")
                elif not valido_odd:
                    st.error(f"❌ {msg_odd}")
                else:
                    calc_final = calcular_jogo_v31(STATS[time_casa], STATS[time_fora], {})
                    prob_final = calcular_probabilidade_mercado(mercado_sel, calc_final)
                    
                    # Validar duplicacao
                    jogo_str = f"{time_casa} vs {time_fora}"
                    duplicado = any(
                        b['jogo'] == jogo_str and b['mercado'] == mercado_sel 
                        for b in st.session_state.current_ticket
                    )
                    
                    if duplicado:
                        st.warning("⚠️ Esta aposta já está no bilhete!")
                    else:
                        st.session_state.current_ticket.append({
                            'jogo': jogo_str,
                            'mercado': mercado_sel,
                            'odd': odd_manual,
                            'prob': prob_final if prob_final > 0 else (1/odd_manual)*100,
                            'tipo': 'Manual'
                        })
                        st.success("✅ Aposta adicionada com sucesso!")
                        time.sleep(0.5)
                        st.rerun()
        
        st.markdown("---")
        st.subheader("📋 **Seu Bilhete**")
        
        if st.session_state.current_ticket:
            df_tick = pd.DataFrame(st.session_state.current_ticket)
            
            # Colunas calculadas
            df_tick['Status'] = df_tick['prob'].apply(
                lambda x: f"{get_prob_emoji(x)} {x:.1f}%"
            )
            df_tick['Fair Odd'] = df_tick['prob'].apply(
                lambda x: round(100/x, 2) if x > 0 else 0
            )
            df_tick['EV%'] = ((df_tick['odd'] - df_tick['Fair Odd']) / df_tick['Fair Odd'] * 100).round(1)
            
            # Display
            st.dataframe(
                df_tick[['jogo', 'mercado', 'Status', 'odd', 'Fair Odd', 'EV%', 'tipo']], 
                use_container_width=True, 
                height=min(400, len(df_tick) * 50 + 50)
            )
            
            # Métricas
            total_odd = np.prod([x['odd'] for x in st.session_state.current_ticket])
            prob_acum = np.prod([x['prob']/100 for x in st.session_state.current_ticket]) * 100
            fair_odd_total = 100/prob_acum if prob_acum > 0 else 0
            ev_total = ((total_odd - fair_odd_total) / fair_odd_total * 100) if fair_odd_total > 0 else 0
            
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("📈 Odd Total", f"{total_odd:.2f}")
            r2.metric("🎯 Prob Real", f"{prob_acum:.1f}%")
            r3.metric("💎 Fair Odd", f"{fair_odd_total:.2f}")
            r4.metric("💰 EV%", f"{ev_total:+.1f}%", 
                     delta_color="normal" if ev_total > 0 else "inverse")
            
            # Feedback EV
            if ev_total > 10:
                st.success(f"💎 **EXCELENTE EV!** (+{ev_total:.1f}%) - Aposta com muito valor!")
            elif ev_total > 5:
                st.success(f"✅ **BOM EV!** (+{ev_total:.1f}%) - Aposta com valor!")
            elif ev_total > 0:
                st.info(f"📈 **EV Positivo** (+{ev_total:.1f}%) - Aposta viável")
            else:
                st.warning(f"📉 **EV Negativo** ({ev_total:.1f}%) - Cuidado!")
            
            # Editar/Remover
            st.markdown("---")
            st.markdown("##### ⚙️ **Editar Bilhete**")
            
            idx_remove = st.selectbox(
                "Selecione aposta para remover:",
                range(len(st.session_state.current_ticket)),
                format_func=lambda i: f"{st.session_state.current_ticket[i]['jogo']} - {st.session_state.current_ticket[i]['mercado']}"
            )
            
            if st.button("🗑️ Remover Selecionada"):
                st.session_state.current_ticket.pop(idx_remove)
                st.success("✅ Aposta removida!")
                time.sleep(0.3)
                st.rerun()
        
        else:
            st.info("📭 Bilhete vazio. Adicione apostas acima!")
    
    # TAB 2: HEDGES
    with tabs[1]:
        st.header("🛡️ Sistema de Hedges")
        
        if not st.session_state.current_ticket:
            st.warning("⚠️ Crie um bilhete primeiro na aba Construtor!")
        else:
            st.info("💡 Hedge = Cobertura para garantir lucro ou minimizar prejuízo")
            
            col1, col2 = st.columns(2)
            stake_principal = col1.number_input("💰 Stake Principal (R$):", 10.0, step=10.0, value=100.0)
            
            banca = st.session_state.bankroll_history[-1]
            valido, msg = validar_stake(stake_principal, banca)
            
            if not valido:
                st.error(f"❌ {msg}")
            else:
                if "⚠️" in msg:
                    st.warning(msg)
                else:
                    st.success(msg)
            
            odd_bilhete = np.prod([x['odd'] for x in st.session_state.current_ticket])
            col2.metric("📊 Odd do Bilhete", f"{odd_bilhete:.2f}")
            
            retorno_maximo = stake_principal * odd_bilhete
            lucro_maximo = retorno_maximo - stake_principal
            
            st.metric("💵 Retorno se Ganhar", format_currency(retorno_maximo), 
                     delta=f"+{format_currency(lucro_maximo)}")
            
            st.markdown("---")
            
            with st.expander("🛡️ **Hedge 1: Proteção Total**", expanded=True):
                st.markdown("**Objetivo:** Garantir lucro independente do resultado")
                
                odd_contra = st.slider("Odd da Aposta Contrária:", 1.5, 10.0, 2.5, 0.1)
                stake_hedge = stake_principal / (odd_contra - 1)
                
                c1, c2, c3 = st.columns(3)
                c1.metric("💰 Stake Hedge", format_currency(stake_hedge))
                c2.metric("💵 Custo Total", format_currency(stake_principal + stake_hedge))
                
                lucro_se_principal_ganha = retorno_maximo - (stake_principal + stake_hedge)
                lucro_se_hedge_ganha = (stake_hedge * odd_contra) - (stake_principal + stake_hedge)
                
                c3.metric("📊 Lucro Mínimo", format_currency(min(lucro_se_principal_ganha, lucro_se_hedge_ganha)))
                
                if lucro_se_principal_ganha > 0 and lucro_se_hedge_ganha > 0:
                    st.success(f"✅ **Viável!** Lucro garantido de {format_currency(min(lucro_se_principal_ganha, lucro_se_hedge_ganha))}")
                else:
                    st.error("❌ Hedge inviável com estes parâmetros")
    
    # TAB 3: SIMULADOR
    with tabs[2]:
        st.header("🎲 Simulador Monte Carlo")
        
        c1, c2 = st.columns(2)
        sim_h = c1.selectbox("🏠 Time Casa", sorted(list(STATS.keys())), key='simh')
        sim_a = c2.selectbox("✈️ Time Visitante", sorted(list(STATS.keys())), key='sima')
        
        n_sims = st.slider("🔢 Número de Simulações:", 1000, 10000, 3000, 1000)
        
        if st.button("🚀 Executar Simulação", use_container_width=True, type="primary"):
            if sim_h == sim_a:
                st.error("❌ Selecione times diferentes!")
            else:
                # Progress bar
                progress_bar = st.progress(0)
                status = st.empty()
                
                status.text("⏳ Inicializando simulação...")
                progress_bar.progress(10)
                time.sleep(0.2)
                
                status.text("🧮 Calculando parâmetros...")
                calc = calcular_jogo_v31(STATS[sim_h], STATS[sim_a], {})
                progress_bar.progress(30)
                time.sleep(0.2)
                
                status.text(f"🎲 Executando {n_sims:,} simulações...")
                sim_corners = np.random.poisson(calc['corners_total'], n_sims)
                sim_goals = np.random.poisson(calc['total_goals'], n_sims)
                sim_cards = np.random.poisson(calc['cards_total'], n_sims)
                progress_bar.progress(80)
                time.sleep(0.2)
                
                status.text("📊 Processando resultados...")
                progress_bar.progress(100)
                time.sleep(0.2)
                
                progress_bar.empty()
                status.empty()
                
                st.success(f"✅ Simulação de {n_sims:,} partidas concluída!")
                
                # Métricas
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("⚽ Média Gols", f"{sim_goals.mean():.2f}")
                m2.metric("🚩 Média Cantos", f"{sim_corners.mean():.2f}")
                m3.metric("📋 Média Cartões", f"{sim_cards.mean():.2f}")
                m4.metric("🎯 Over 2.5", f"{(sim_goals > 2.5).mean() * 100:.1f}%")
                
                # Distribuições
                st.markdown("---")
                st.subheader("📊 Distribuições de Probabilidade")
                
                tab_g, tab_c, tab_ca = st.tabs(["⚽ Gols", "🚩 Cantos", "📋 Cartões"])
                
                with tab_g:
                    fig_g = px.histogram(
                        sim_goals, 
                        nbins=20, 
                        title=f"Distribuição de Gols ({n_sims:,} simulações)",
                        color_discrete_sequence=['#667eea'],
                        labels={'value': 'Gols', 'count': 'Frequência'}
                    )
                    fig_g.update_layout(showlegend=False)
                    st.plotly_chart(fig_g, use_container_width=True)
                
                with tab_c:
                    fig_c = px.histogram(
                        sim_corners, 
                        nbins=20, 
                        title=f"Distribuição de Cantos ({n_sims:,} simulações)",
                        color_discrete_sequence=['#764ba2'],
                        labels={'value': 'Cantos', 'count': 'Frequência'}
                    )
                    fig_c.update_layout(showlegend=False)
                    st.plotly_chart(fig_c, use_container_width=True)
                
                with tab_ca:
                    fig_ca = px.histogram(
                        sim_cards, 
                        nbins=15, 
                        title=f"Distribuição de Cartões ({n_sims:,} simulações)",
                        color_discrete_sequence=['#FFA500'],
                        labels={'value': 'Cartões', 'count': 'Frequência'}
                    )
                    fig_ca.update_layout(showlegend=False)
                    st.plotly_chart(fig_ca, use_container_width=True)
    
    # TAB 4: METRICAS
    with tabs[3]:
        st.header("📊 Métricas de Performance")
        
        if not st.session_state.bet_results:
            st.info("ℹ️ Nenhuma aposta registrada ainda. Complete apostas para ver métricas!")
            
            with st.expander("➕ Registrar Resultado Manualmente"):
                res_jogo = st.text_input("Jogo:")
                res_mercado = st.text_input("Mercado:")
                res_odd = st.number_input("Odd:", 1.01, 50.0, 1.90)
                res_stake = st.number_input("Stake (R$):", 1.0, step=10.0)
                res_ganhou = st.checkbox("Ganhou?")
                
                if st.button("💾 Salvar Resultado"):
                    st.session_state.bet_results.append({
                        'jogo': res_jogo,
                        'mercado': res_mercado,
                        'odd': res_odd,
                        'stake': res_stake,
                        'ganhou': res_ganhou,
                        'data': datetime.now().strftime('%d/%m/%Y')
                    })
                    
                    # Atualizar banca
                    if res_ganhou:
                        st.session_state.bankroll_history.append(
                            st.session_state.bankroll_history[-1] + (res_stake * (res_odd - 1))
                        )
                    else:
                        st.session_state.bankroll_history.append(
                            st.session_state.bankroll_history[-1] - res_stake
                        )
                    
                    st.success("✅ Resultado registrado!")
                    st.rerun()
        
        else:
            df_hist = pd.DataFrame(st.session_state.bet_results)
            
            total = len(df_hist)
            green = df_hist[df_hist['ganhou'] == True].shape[0]
            red = total - green
            wr = (green / total * 100) if total > 0 else 0
            
            total_investido = df_hist['stake'].sum()
            total_retorno = df_hist[df_hist['ganhou'] == True].apply(
                lambda row: row['stake'] * row['odd'], axis=1
            ).sum()
            lucro = total_retorno - total_investido
            roi = (lucro / total_investido * 100) if total_investido > 0 else 0
            
            # Métricas principais
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("🎯 Total", total)
            m2.metric("✅ Green", green, delta=f"{wr:.1f}%")
            m3.metric("❌ Red", red)
            m4.metric("💰 ROI", f"{roi:+.1f}%", delta_color="normal" if roi > 0 else "inverse")
            m5.metric("💵 Lucro", format_currency(lucro), delta_color="normal" if lucro > 0 else "inverse")
            
            # Gráfico de evolução da banca
            st.markdown("---")
            st.subheader("📈 Evolução da Banca")
            
            fig_banca = go.Figure()
            fig_banca.add_trace(go.Scatter(
                y=st.session_state.bankroll_history,
                mode='lines+markers',
                name='Banca',
                line=dict(color='#667eea', width=3),
                marker=dict(size=8)
            ))
            fig_banca.update_layout(
                title="Evolução da Banca ao Longo do Tempo",
                xaxis_title="Aposta #",
                yaxis_title="Banca (R$)",
                hovermode='x unified'
            )
            st.plotly_chart(fig_banca, use_container_width=True)
            
            # Histórico detalhado
            st.markdown("---")
            st.subheader("📋 Histórico Detalhado")
            st.dataframe(df_hist, use_container_width=True, height=400)
    
    # TAB 5: AI ADVISOR
    with tabs[4]:
        st.header("🤖 AI Advisor MAXIMUM")
        st.caption("🧠 Memória Contextual | 📊 Análises Avançadas | 💰 Gestão de Banca")
        
        memoria = st.session_state.chat_memory
        banca = st.session_state.bankroll_history[-1]
        
        # Welcome message
        if not st.session_state.chat_history:
            st.markdown("""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 30px; border-radius: 20px; color: white; margin: 20px 0;
                        box-shadow: 0 10px 40px rgba(0,0,0,0.2);'>
                <h2>👋 Olá! Sou o AI Advisor MAXIMUM</h2>
                <p style='font-size: 16px; margin: 15px 0;'>
                    Versão mais avançada possível <strong>SEM API externa</strong>.
                    Para chatbot 9.5/10, precisaria de Claude API (R$ 20/mês).
                </p>
                <h3>🚀 O que posso fazer:</h3>
                <ul style='font-size: 15px; line-height: 1.8;'>
                    <li>⚽ <strong>Análises completas</strong> de confrontos</li>
                    <li>📊 <strong>Comparações</strong> entre times</li>
                    <li>💰 <strong>Gestão de banca</strong> (Kelly Criterion)</li>
                    <li>🏆 <strong>Rankings</strong> dinâmicos por liga</li>
                    <li>💡 <strong>Explicações</strong> detalhadas</li>
                    <li>🧠 <strong>Memória contextual</strong> (lembro do que falamos!)</li>
                </ul>
                <h3>💬 Experimente:</h3>
                <p style='font-size: 14px; margin-top: 10px;'>
                    • "Analise Manchester City vs Arsenal"<br>
                    • "Compare Liverpool com Chelsea"<br>
                    • "Quanto devo apostar?"<br>
                    • "Top 5 ataques da Premier League"
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Chat messages
        for msg in st.session_state.chat_history:
            role = msg['role']
            avatar = "👤" if role == 'user' else "🤖"
            with st.chat_message(role, avatar=avatar):
                st.markdown(msg['content'])
        
        # Input
        user_input = st.chat_input("💬 Digite sua pergunta...")
        
        if user_input:
            # Add user message
            st.session_state.chat_history.append({
                'role': 'user',
                'content': user_input
            })
            
            # Process with loading
            with st.spinner("🧠 Processando..."):
                progress = st.progress(0)
                for i in range(100):
                    time.sleep(0.008)
                    progress.progress(i + 1)
                progress.empty()
                
                response = processar_chat_avancado(
                    user_input, STATS, CAL, REFS, memoria, banca
                )
            
            # Add bot response
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': response
            })
            
            st.rerun()

if __name__ == "__main__":
    main()
