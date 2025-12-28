"""
FutPrevisão V31 MAXIMUM + AI Advisor ULTRA
CÓDIGO COMPLETO - VERSÃO LIMPA E CORRIGIDA (TAB CONSTRUTOR)
Versão: 31.6 ULTRA FINAL (CORREÇÃO)
Data: 28/12/2025

Este arquivo foi limpo de caracteres Unicode ocultos e teve a aba Construtor corrigida.
"""

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

# Configuração para Scipy (Matemática Avançada)
try:
    from scipy.stats import poisson, norm
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Diretório base do projeto
BASE_DIR = Path(__file__).resolve().parent

# ==============================================================================
# CONFIGURAÇÃO DA PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="FutPrevisão V31 ULTRA",
    layout="wide",
    page_icon="⚽",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.futprevisao.com/help',
        'About': "# FutPrevisão V31 ULTRA\nSistema Profissional de Análise Esportiva."
    }
)

# ==============================================================================
# ESTILIZAÇÃO CSS (GLASSMORPHISM)
# ==============================================================================
st.markdown('''
<style>
    /* ESTILO GERAL */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 15px 15px 0px 15px;
        border-radius: 15px 15px 0 0;
        box-shadow: 0 8px 25px rgba(30, 60, 114, 0.3);
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.1);
        border-radius: 8px 8px 0 0;
        padding: 12px 24px;
        font-weight: 600;
        color: #e0e0e0;
        border: 1px solid rgba(255,255,255,0.1);
        border-bottom: none;
        transition: all 0.3s;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255,255,255,0.25);
        color: #ffffff;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%) !important;
        color: #1a1a1a !important;
        border-color: #FFD700;
        font-weight: 800;
        transform: scale(1.02);
    }
    /* CHATBOT */
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) {
        background: #f8f9fa;
        border-radius: 0px 15px 15px 15px;
        padding: 20px;
        border-left: 5px solid #1e3c72;
        margin-bottom: 10px;
        color: #2c3e50;
    }
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) {
        background: #e0f2fe;
        border-radius: 15px 0px 15px 15px;
        padding: 20px;
        text-align: right;
        margin-bottom: 10px;
        border-right: 5px solid #0284c7;
        color: #0f172a;
    }
    /* CARDS */
    div[data-testid="metric-container"] {
        background: #ffffff;
        padding: 18px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
        border-top: 4px solid #1e3c72;
    }
    h1 {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
</style>
''', unsafe_allow_html=True)

# ==============================================================================
# CONSTANTES E MAPEAMENTOS
# ==============================================================================
NAME_MAPPING = {
    'Man United': 'Manchester United', 'Man Utd': 'Manchester United',
    'Man City': 'Manchester City', 'Spurs': 'Tottenham',
    'Wolves': 'Wolverhampton', 'Paris SG': 'PSG',
    'Nottm Forest': 'Nottingham Forest', 'Newcastle': 'Newcastle United',
    'West Ham': 'West Ham United', 'Inter': 'Inter Milan', 'Milan': 'AC Milan',
    'Ath Madrid': 'Atletico Madrid', 'Ath Bilbao': 'Athletic Club',
    'Betis': 'Real Betis', 'Sociedad': 'Real Sociedad',
    'Dortmund': 'Borussia Dortmund', 'Leverkusen': 'Bayer Leverkusen',
    'Bayern': 'Bayern Munich', 'Marseille': 'Olympique Marseille',
    'Lyon': 'Olympique Lyon', 'Monaco': 'AS Monaco', 'Lille': 'LOSC Lille'
}

MERCADOS_DISPONIVEIS = [
    "Selecione...",
    # === GOLS ===
    "Over 0.5 Gols", "Over 1.5 Gols", "Over 2.5 Gols", "Over 3.5 Gols",
    "Under 2.5 Gols", "Under 1.5 Gols",
    
    # === ESCANTEIOS - INDIVIDUAIS (Casa/Fora) ===
    "Over 2.5 Cantos Casa", "Over 3.5 Cantos Casa", "Over 4.5 Cantos Casa", "Over 5.5 Cantos Casa",
    "Over 2.5 Cantos Fora", "Over 3.5 Cantos Fora", "Over 4.5 Cantos Fora", "Over 5.5 Cantos Fora",
    
    # === ESCANTEIOS - TOTAIS ===
    "Over 7.5 Cantos Total", "Over 8.5 Cantos Total", "Over 9.5 Cantos Total",
    
    # === CARTÕES - INDIVIDUAIS (Casa/Fora) ===
    "Over 1.5 Cartões Casa", "Over 2.5 Cartões Casa",
    "Over 1.5 Cartões Fora", "Over 2.5 Cartões Fora",
    
    # === CARTÕES - TOTAIS ===
    "Over 2.5 Cartões Total", "Over 3.5 Cartões Total", "Over 4.5 Cartões Total",
    
    # === RESULTADO ===
    "Ambos Marcam (BTTS)", "Vitória Casa", "Vitória Fora", "Empate"
]

PRESSURE_HIGH_THRESHOLD = 6.0
PRESSURE_MED_THRESHOLD = 4.5
VIOLENCE_HIGH_THRESHOLD = 12.5

# ==============================================================================
# FUNÇÕES AUXILIARES
# ==============================================================================

def find_file(filename: str) -> Optional[str]:
    search_paths = [
        Path('/mnt/project') / filename,
        Path('.') / filename,
        Path('./data') / filename,
        BASE_DIR / filename
    ]
    for path in search_paths:
        if path.exists():
            return str(path)
    return None

def normalize_name(name: str, known_teams: List[str]) -> Optional[str]:
    if not name or not known_teams: return None
    name = str(name).strip()
    if name in NAME_MAPPING:
        target = NAME_MAPPING[name]
        if target in known_teams: return target
        name = target
    if name in known_teams: return name
    matches = get_close_matches(name, known_teams, n=1, cutoff=0.6)
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
# CARREGAMENTO DE DADOS (ETL)
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
        if not filepath: continue
        try:
            df = pd.read_csv(filepath, encoding='utf-8')
            cols = {c: c.strip() for c in df.columns}
            df.rename(columns=cols, inplace=True)
            teams = set(df['HomeTeam'].dropna().unique()) | set(df['AwayTeam'].dropna().unique())
            
            for team in teams:
                if pd.isna(team): continue
                h_games = df[df['HomeTeam'] == team]
                a_games = df[df['AwayTeam'] == team]
                
                # Médias
                corners_h = h_games['HC'].mean() if 'HC' in h_games else 5.0
                corners_a = a_games['AC'].mean() if 'AC' in a_games else 4.0
                ch = (h_games['HY'].mean() + h_games['HR'].mean()*2) if 'HY' in h_games else 1.8
                ca = (a_games['AY'].mean() + a_games['AR'].mean()*2) if 'AY' in a_games else 2.2
                fouls_h = h_games['HF'].mean() if 'HF' in h_games else 11.5
                fouls_a = a_games['AF'].mean() if 'AF' in a_games else 12.5
                goals_fh = h_games['FTHG'].mean() if 'FTHG' in h_games else 1.4
                goals_fa = a_games['FTAG'].mean() if 'FTAG' in a_games else 1.1
                goals_ah = h_games['FTAG'].mean() if 'FTAG' in h_games else 1.0
                goals_aa = a_games['FTHG'].mean() if 'FTHG' in a_games else 1.5
                shots_h = h_games['HST'].mean() if 'HST' in h_games else 4.8
                shots_a = a_games['AST'].mean() if 'AST' in a_games else 3.8
                
                stats_db[team] = {
                    'league': league_name,
                    'corners': (corners_h + corners_a) / 2, 'corners_home': corners_h, 'corners_away': corners_a,
                    'cards': (ch + ca) / 2, 'cards_home': ch, 'cards_away': ca,
                    'fouls': (fouls_h + fouls_a) / 2, 'fouls_home': fouls_h, 'fouls_away': fouls_a,
                    'goals_f': (goals_fh + goals_fa) / 2, 'goals_f_home': goals_fh, 'goals_f_away': goals_fa,
                    'goals_a': (goals_ah + goals_aa) / 2, 'goals_a_home': goals_ah, 'goals_a_away': goals_aa,
                    'shots_on_target': (shots_h + shots_a) / 2, 'shots_home': shots_h, 'shots_away': shots_a,
                    'games_played': len(h_games) + len(a_games)
                }
        except: pass

    # Calendário
    cal_path = find_file('calendario_ligas.csv')
    if cal_path:
        try:
            cal = pd.read_csv(cal_path, encoding='utf-8')
            if 'Data' in cal.columns:
                cal['DtObj'] = pd.to_datetime(cal['Data'], dayfirst=True, errors='coerce')
        except: pass
    
    # Árbitros
    ref_path = find_file('arbitros_5_ligas_2025_2026.csv')
    if ref_path:
        try:
            refs_df = pd.read_csv(ref_path, encoding='utf-8')
            for _, row in refs_df.iterrows():
                avg = row.get('Media_Cartoes_Por_Jogo', 4.0)
                referees[row['Arbitro']] = {
                    'factor': avg / 4.0, 'avg_cards': avg,
                    'games': row.get('Jogos_Apitados', 0),
                    'red_rate': row.get('Cartoes_Vermelhos', 0) / (row.get('Jogos_Apitados', 1) or 1)
                }
        except: pass
            
    return stats_db, cal, referees

# ==============================================================================
# MOTOR DE CÁLCULO E SIMULAÇÃO (V31 ENGINE)
# ==============================================================================

def calcular_poisson(media: float, linha: float) -> float:
    """Calcula probabilidade de OVER usando Poisson"""
    if media <= 0: return 0.0
    if SCIPY_AVAILABLE:
        try:
            return (1 - poisson.cdf(int(linha), media)) * 100
        except: pass
    # Fallback
    prob_exact = 0
    k = int(linha)
    for i in range(k + 1):
        prob_exact += (math.exp(-media) * (media ** i)) / math.factorial(i)
    return (1 - prob_exact) * 100

def calcular_jogo_v31(home_stats: Dict, away_stats: Dict, ref_data: Dict) -> Dict:
    """
    CAUSALITY ENGINE V31 - NÚCLEO DO SISTEMA
    Implementa a lógica 'Causa -> Efeito' para previsões.
    """
    if not home_stats or not away_stats:
        return {'corners': {'h':0,'a':0,'t':0}, 'cards': {'h':0,'a':0,'t':0}, 'goals': {'h':0,'a':0}, 'corners_total':0, 'total_goals':0, 'cards_total':0, 'xg_home':0, 'xg_away':0}

    # === ESCANTEIOS (Baseado em Pressão) ===
    base_h = home_stats.get('corners_home', 5.0)
    base_a = away_stats.get('corners_away', 4.0)
    shots_h = home_stats.get('shots_home', 4.5)
    shots_a = away_stats.get('shots_away', 3.5)
    
    # Fator Pressão
    press_h = 1.15 if shots_h > PRESSURE_HIGH_THRESHOLD else 1.05 if shots_h > PRESSURE_MED_THRESHOLD else 1.0
    press_a = 1.10 if shots_a > PRESSURE_MED_THRESHOLD else 1.0
    
    # Fator Casa/Fora (Mandante pressiona mais no fim)
    corners_h = base_h * press_h * 1.10
    corners_a = base_a * press_a * 0.90
    corners_total = corners_h + corners_a
    
    # === CARTÕES (Baseado em Violência e Árbitro) ===
    fouls_h = home_stats.get('fouls_home', 11.0)
    fouls_a = away_stats.get('fouls_away', 12.0)
    
    viol_h = 1.1 if fouls_h > VIOLENCE_HIGH_THRESHOLD else 1.0
    viol_a = 1.1 if fouls_a > VIOLENCE_HIGH_THRESHOLD else 1.0
    
    ref_avg = ref_data.get('avg_cards', 4.0) if ref_data else 4.0
    cards_h_base = home_stats.get('cards_home', 1.8)
    cards_a_base = away_stats.get('cards_away', 2.2)
    
    # Fórmula: (Média Times + Média Juiz) / 2 * Violência
    cards_h_proj = (cards_h_base + (ref_avg/2)) / 2 * viol_h
    cards_a_proj = (cards_a_base + (ref_avg/2)) / 2 * viol_a
    cards_total = cards_h_proj + cards_a_proj
    
    # === GOLS (xG V31) ===
    league_avg = 1.35
    xg_h = (home_stats['goals_f_home'] / league_avg) * (away_stats['goals_a_away'] / league_avg) * league_avg
    xg_a = (away_stats['goals_f_away'] / league_avg) * (home_stats['goals_a_home'] / league_avg) * league_avg
    
    return {
        'corners': {'h': corners_h, 'a': corners_a, 't': corners_total},
        'cards': {'h': cards_h_proj, 'a': cards_a_proj, 't': cards_total},
        'goals': {'h': xg_h, 'a': xg_a},
        'corners_total': corners_total,
        'cards_total': cards_total,
        'total_goals': xg_h + xg_a,
        'xg_home': xg_h, 'xg_away': xg_a
    }

def simulate_game_v31(home_stats: Dict, away_stats: Dict, ref_data: Dict, n_sims: int = 3000) -> Dict:
    """Simulador Monte Carlo (3000 iterações)"""
    calc = calcular_jogo_v31(home_stats, away_stats, ref_data)
    return {
        'corners_total': np.random.poisson(calc['corners_total'], n_sims),
        'cards_total': np.random.poisson(calc['cards_total'], n_sims),
        'goals_h': np.random.poisson(calc['xg_home'], n_sims),
        'goals_a': np.random.poisson(calc['xg_away'], n_sims),
        'goals_total': np.random.poisson(calc['total_goals'], n_sims)
    }

def calcular_probabilidade_mercado(mercado: str, calc: Dict) -> float:
    """Calcula probabilidade baseada no mercado"""
    if mercado == "Selecione...":
        return 0.0
    
    # Mapeamento de mercados
    mercado_map = {
        # GOLS
        "Over 0.5 Gols": ('total_goals', 0.5),
        "Over 1.5 Gols": ('total_goals', 1.5),
        "Over 2.5 Gols": ('total_goals', 2.5),
        "Over 3.5 Gols": ('total_goals', 3.5),
        
        # ESCANTEIOS TOTAIS
        "Over 7.5 Cantos Total": ('corners_total', 7.5),
        "Over 8.5 Cantos Total": ('corners_total', 8.5),
        "Over 9.5 Cantos Total": ('corners_total', 9.5),
        
        # ESCANTEIOS CASA
        "Over 2.5 Cantos Casa": ('corners', 'h', 2.5),
        "Over 3.5 Cantos Casa": ('corners', 'h', 3.5),
        "Over 4.5 Cantos Casa": ('corners', 'h', 4.5),
        "Over 5.5 Cantos Casa": ('corners', 'h', 5.5),
        
        # ESCANTEIOS FORA
        "Over 2.5 Cantos Fora": ('corners', 'a', 2.5),
        "Over 3.5 Cantos Fora": ('corners', 'a', 3.5),
        "Over 4.5 Cantos Fora": ('corners', 'a', 4.5),
        "Over 5.5 Cantos Fora": ('corners', 'a', 5.5),
        
        # CARTÕES TOTAIS
        "Over 2.5 Cartões Total": ('cards_total', 2.5),
        "Over 3.5 Cartões Total": ('cards_total', 3.5),
        "Over 4.5 Cartões Total": ('cards_total', 4.5),
        
        # CARTÕES CASA
        "Over 1.5 Cartões Casa": ('cards', 'h', 1.5),
        "Over 2.5 Cartões Casa": ('cards', 'h', 2.5),
        
        # CARTÕES FORA
        "Over 1.5 Cartões Fora": ('cards', 'a', 1.5),
        "Over 2.5 Cartões Fora": ('cards', 'a', 2.5),
    }
    
    if mercado in mercado_map:
        val = mercado_map[mercado]
        if len(val) == 2: # Chave direta (ex: total_goals)
            return calcular_poisson(calc[val[0]], val[1])
        elif len(val) == 3: # Chave aninhada (ex: corners -> h)
            # Acessa calc['corners']['h']
            media = calc[val[0]][val[1]]
            return calcular_poisson(media, val[2])
    
    # Casos especiais
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
# 7. CHATBOT AI ADVISOR ULTRA (NLP AVANÇADO)
# ==============================================================================

def extrair_entidades(mensagem: str, stats_db: Dict) -> Dict:
    """Motor de NLP: Extrai times, mercados e linhas da mensagem"""
    msg_lower = mensagem.lower()
    entidades = {'times': [], 'mercado': None, 'linha': None}
    
    # Extração de Times (Prioriza compostos)
    known = list(stats_db.keys())
    sorted_teams = sorted(known, key=len, reverse=True)
    msg_clean = msg_lower
    
    for team in sorted_teams:
        if team.lower() in msg_clean:
            # Verifica se não é substring de outro
            is_sub = False
            for ft in entidades['times']:
                if team.lower() in ft.lower(): is_sub = True; break
            if not is_sub:
                entidades['times'].append(team)
                msg_clean = msg_clean.replace(team.lower(), "")
    
    # Mercado
    if any(x in msg_lower for x in ['canto', 'escanteio']): entidades['mercado'] = 'cantos'
    elif any(x in msg_lower for x in ['cartao', 'cartão']): entidades['mercado'] = 'cartoes'
    elif any(x in msg_lower for x in ['gol', 'gols', 'over']): entidades['mercado'] = 'gols'
    
    # Linha
    nums = re.findall(r'\d+\.?\d*', mensagem)
    if nums:
        for n in nums:
            if float(n) < 20: entidades['linha'] = float(n); break
            
    return entidades

def processar_chat_ultra(mensagem: str, stats_db: Dict, cal: pd.DataFrame, refs: Dict) -> str:
    """CÉREBRO DO AI ADVISOR"""
    if not mensagem: return "Olá! Sou o AI Advisor ULTRA. Como posso ajudar?"
    
    entidades = extrair_entidades(mensagem, stats_db)
    times = entidades['times']
    msg_lower = mensagem.lower()
    
    # 1. ANÁLISE DE CONFRONTO (VS)
    if len(times) >= 2:
        t1, t2 = times[0], times[1]
        s1, s2 = stats_db[t1], stats_db[t2]
        calc = calcular_jogo_v31(s1, s2, {})
        
        prob_g = calcular_poisson(calc['total_goals'], 2.5)
        prob_c = calcular_poisson(calc['corners_total'], 9.5)
        prob_card = calcular_poisson(calc['cards_total'], 4.5)
        prob_btts = min((calc['xg_home'] * calc['xg_away'] * 38), 92)
        
        resp = f"📊 **ANÁLISE: {t1} vs {t2}**\n\n"
        resp += "**🔎 Projeções V31:**\n"
        resp += f"• **Gols (xG):** {calc['total_goals']:.2f} (Tendência: {'Aberto' if calc['total_goals']>2.6 else 'Travado'})\n"
        resp += f"• **Cantos:** {calc['corners_total']:.1f}\n"
        resp += f"• **Cartões:** {calc['cards_total']:.1f}\n\n"
        
        # Pergunta específica de probabilidade
        if ('prob' in msg_lower or 'chance' in msg_lower) and entidades['linha']:
            lin = entidades['linha']
            merc = entidades['mercado'] or 'gols'
            media = calc['corners_total'] if merc=='cantos' else calc['cards_total'] if merc=='cartoes' else calc['total_goals']
            prob_user = calcular_poisson(media, lin)
            resp += f"🎲 **Sua Consulta:** Over {lin} {merc}\n{get_prob_emoji(prob_user)} **Prob:** {prob_user:.1f}%\n"
            return resp
            
        resp += "**💡 Sugestões (EV+):**\n"
        found = False
        if prob_g > 65: resp += f"✅ Over 2.5 Gols ({prob_g:.1f}%)\n"; found=True
        elif prob_g < 35: resp += f"✅ Under 2.5 Gols ({100-prob_g:.1f}%)\n"; found=True
        if prob_c > 70: resp += f"✅ Over 9.5 Cantos ({prob_c:.1f}%)\n"; found=True
        if prob_card > 65: resp += f"✅ Over 4.5 Cartões ({prob_card:.1f}%)\n"; found=True
        if prob_btts > 60: resp += f"✅ BTTS - Sim ({prob_btts:.1f}%)\n"; found=True
        
        if not found: resp += "⚠️ Sem valor estatístico claro (Linhas justas)."
        return resp

    # 2. ANÁLISE DE TIME ÚNICO
    elif len(times) == 1:
        t = times[0]
        s = stats_db[t]
        resp = f"📊 **RAIO-X: {t}**\n_(Liga: {s['league']})_\n\n"
        resp += f"**Ataque:** {s['goals_f']:.2f} gols/jogo\n"
        resp += f"**Defesa:** {s['goals_a']:.2f} sofridos/jogo\n"
        resp += f"**Cantos:** {s['corners']:.2f}/jogo\n"
        resp += f"**Cartões:** {s['cards']:.2f}/jogo\n\n"
        
        resp += "**🧠 Veredito:**\n"
        if s['corners'] > 6.0: resp += "🔥 Máquina de Cantos (Over).\n"
        elif s['corners'] < 3.5: resp += "🔻 Time fechado (Under Cantos).\n"
        if s['goals_f'] > 1.8: resp += "⚽ Ataque Poderoso.\n"
        if s['cards'] > 2.5: resp += "🟨 Time Indisciplinado.\n"
        return resp

    # 3. SCANNER / MELHORES JOGOS
    elif any(x in msg_lower for x in ['melhor', 'hoje', 'jogos']):
        hoje = datetime.now().strftime('%d/%m/%Y')
        jogos = cal[cal['Data'] == hoje] if not cal.empty else pd.DataFrame()
        if jogos.empty: return f"📅 Sem jogos hoje ({hoje})."
        
        ranking = []
        for _, r in jogos.iterrows():
            h, a = normalize_name(r['Time_Casa'], list(stats_db.keys())), normalize_name(r['Time_Visitante'], list(stats_db.keys()))
            if h and a:
                c = calcular_jogo_v31(stats_db[h], stats_db[a], {})
                score = c['total_goals']*2 + c['corners_total']
                ranking.append({'j': f"{h} vs {a}", 's': score, 'd': c})
        
        ranking.sort(key=lambda x: x['s'], reverse=True)
        resp = f"🏆 **TOP JOGOS HOJE ({hoje}):**\n\n"
        for i in ranking[:3]:
            resp += f"**{i['j']}**\n   🎯 Gols: {i['d']['total_goals']:.1f} | Cantos: {i['d']['corners_total']:.1f}\n\n"
        return resp

    else:
        return "🤖 **AI Advisor:** Pergunte sobre jogos ('Arsenal vs Chelsea'), times ('Real Madrid') ou 'Jogos de hoje'."

# ==============================================================================
# 8. MÉTODOS FINANCEIROS E GRÁFICOS
# ==============================================================================

def calculate_sharpe_ratio(returns: List[float]) -> float:
    if not returns or len(returns) < 2: return 0.0
    return (np.mean(returns) - 1.0) / np.std(returns) if np.std(returns) > 0 else 0.0

def calculate_max_drawdown(history: List[float]) -> float:
    if len(history) < 2: return 0.0
    peak = history[0]
    max_dd = 0.0
    for v in history:
        if v > peak: peak = v
        dd = (peak - v) / peak * 100
        if dd > max_dd: max_dd = dd
    return max_dd

def calculate_roi(investido: float, retorno: float) -> float:
    return (retorno / investido) * 100 if investido > 0 else 0

def parse_bilhete_texto(texto: str) -> List[Dict]:
    """Parser simplificado para texto"""
    jogos = []
    lines = texto.split('\n')
    for line in lines:
        if ' vs ' in line or ' x ' in line:
            parts = re.split(r' vs | x ', line)
            if len(parts) >= 2:
                jogos.append({'home': parts[0].strip(), 'away': parts[1].strip()})
    return jogos

def validar_jogos_bilhete(jogos_parsed: List[Dict], stats_db: Dict) -> List[Dict]:
    validos = []
    known = list(stats_db.keys())
    for j in jogos_parsed:
        h = normalize_name(j['home'], known)
        a = normalize_name(j['away'], known)
        if h and a:
            validos.append({'home': h, 'away': a, 'home_stats': stats_db[h], 'away_stats': stats_db[a]})
    return validos

# ==============================================================================
# 9. UI PRINCIPAL (MAIN)
# ==============================================================================

def main():
    # 1. CARREGAMENTO INICIAL DE DADOS
    STATS, CAL, REFS = load_all_data()
    
    # 2. Inicialização de Estado
    if 'current_ticket' not in st.session_state: st.session_state.current_ticket = []
    if 'bet_results' not in st.session_state: st.session_state.bet_results = []
    if 'bankroll_history' not in st.session_state: st.session_state.bankroll_history = [1000.0]
    if 'chat_history' not in st.session_state: st.session_state.chat_history = []
    
    # 3. SIDEBAR (DASHBOARD)
    with st.sidebar:
        st.header("📊 Dashboard Profissional")
        c1, c2 = st.columns(2)
        c1.metric("Times", len(STATS))
        c2.metric("Jogos DB", len(CAL) if not CAL.empty else 0)
        
        banca = st.session_state.bankroll_history[-1]
        lucro_total = banca - 1000.0
        st.metric("💰 Banca Atual", format_currency(banca), delta=format_currency(lucro_total))
        
        if st.session_state.current_ticket:
            st.success(f"🎫 {len(st.session_state.current_ticket)} apostas")
            if st.button("Limpar Bilhete", use_container_width=True):
                st.session_state.current_ticket = []
                st.rerun()
                
        # Exportação JSON
        st.markdown("---")
        if st.session_state.current_ticket:
            st.download_button("📥 Baixar Bilhete", json.dumps(st.session_state.current_ticket), "ticket.json")

    # 4. HEADER
    col1, col2, col3 = st.columns([1, 5, 2])
    with col1: st.markdown("# ⚽")
    with col2:
        st.title("FutPrevisão V31 ULTRA")
        st.markdown("**Professional Sports Analytics System** | _Powered by Causality Engine V31_")
    with col3:
        if not CAL.empty:
            hj = datetime.now().strftime('%d/%m/%Y')
            st.metric("Jogos Hoje", len(CAL[CAL['Data'] == hj]))
    
    st.markdown("---")

    # 5. TABS
    tabs = st.tabs([
        "🎫 Construtor", "🛡️ Hedges", "🎲 Simulador", "📊 Métricas", 
        "🎨 Viz", "📝 Registro", "🔍 Scanner", "📋 Importar", "🤖 AI Advisor"
    ])
    
    # ========================================
    # TAB 1: CONSTRUTOR DE BILHETES (HÍBRIDO)
    # ========================================
    with tabs[0]:
        st.subheader("🛠️ Construtor de Bilhetes Profissional")
        
        c_col1, c_col2 = st.columns([1, 1])
        
        # MODO AUTOMÁTICO (CALENDÁRIO)
        with c_col1:
            st.markdown("#### 📅 Seleção Automática")
            if not CAL.empty:
                dates = sorted(CAL['DtObj'].dt.strftime('%d/%m/%Y').unique())
                if dates:
                    data_sel = st.selectbox("📆 Data:", dates)
                    jogos_dia = CAL[CAL['DtObj'].dt.strftime('%d/%m/%Y') == data_sel]
                    
                    if jogos_dia.empty: st.info("Sem jogos.")
                    
                    for idx, row in jogos_dia.iterrows():
                        h, a = normalize_name(row['Time_Casa'], list(STATS.keys())), normalize_name(row['Time_Visitante'], list(STATS.keys()))
                        if h and a:
                            calc = calcular_jogo_v31(STATS[h], STATS[a], {})
                            with st.expander(f"⚽ {h} vs {a} | {row.get('Hora', '-')}"):
                                m1, m2, m3 = st.columns(3)
                                m1.metric("Cantos", f"{calc['corners_total']:.1f}")
                                m2.metric("Gols", f"{calc['total_goals']:.1f}")
                                m3.metric("Cartões", f"{calc['cards_total']:.1f}")
                                
                                b1, b2 = st.columns(2)
                                if b1.button("+ Over 9.5 C", key=f"ac_{idx}"):
                                    prob = calcular_poisson(calc['corners_total'], 9.5)
                                    st.session_state.current_ticket.append({'jogo': f"{h} vs {a}", 'mercado': 'Over 9.5 Cantos', 'odd': 1.85, 'prob': prob, 'tipo': 'Auto'})
                                    st.success("Adicionado!")
                                    st.rerun()
                                if b2.button("+ Over 2.5 G", key=f"ag_{idx}"):
                                    prob = calcular_poisson(calc['total_goals'], 2.5)
                                    st.session_state.current_ticket.append({'jogo': f"{h} vs {a}", 'mercado': 'Over 2.5 Gols', 'odd': 1.90, 'prob': prob, 'tipo': 'Auto'})
                                    st.success("Adicionado!")
                                    st.rerun()

        # MODO MANUAL (DROPDOWNS)
        with c_col2:
            st.markdown("#### 📝 Adicionar Manualmente")
            with st.container():
                st.info("Adicione jogos específicos com cálculo automático.")
                all_teams = sorted(list(STATS.keys()))
                
                tc = st.selectbox("🏠 Casa:", ["Selecione..."] + all_teams, key="m_casa")
                tv = st.selectbox("✈️ Fora:", ["Selecione..."] + all_teams, key="m_fora")
                
                c_mk, c_od = st.columns(2)
                m_mercado = c_mk.selectbox("Mercado:", MERCADOS_DISPONIVEIS)
                m_odd = c_od.number_input("Odd:", 1.01, 100.0, 1.90)
                
                # Previsão em tempo real
                prob_est = 0
                if tc != "Selecione..." and tv != "Selecione..." and m_mercado != "Selecione...":
                    calc_m = calcular_jogo_v31(STATS[tc], STATS[tv], {})
                    prob_est = calcular_probabilidade_mercado(m_mercado, calc_m)
                    
                    st.caption(f"🎲 Probabilidade Calculada V31: **{prob_est:.1f}%**")
                
                if st.button("➕ Adicionar Manual", use_container_width=True):
                    if tc != "Selecione..." and tv != "Selecione..." and m_mercado != "Selecione...":
                        st.session_state.current_ticket.append({
                            'jogo': f"{tc} vs {tv}", 'mercado': m_mercado, 
                            'odd': m_odd, 'prob': prob_est, 'tipo': 'Manual'
                        })
                        st.success("Adicionado!")
                        st.rerun()
                    else:
                        st.error("Selecione os times e o mercado.")

        # VISUALIZAÇÃO DO BILHETE
        st.markdown("---")
        if st.session_state.current_ticket:
            st.subheader("📋 Seu Bilhete")
            df_tick = pd.DataFrame(st.session_state.current_ticket)
            
            # Formatação para exibição
            df_show = df_tick.copy()
            df_show['Prob'] = df_show['prob'].apply(lambda x: f"{x:.1f}%")
            df_show['Emoji'] = df_show['prob'].apply(get_prob_emoji)
            
            st.dataframe(df_show[['Emoji', 'jogo', 'mercado', 'odd', 'Prob', 'tipo']], use_container_width=True)
            
            # Cálculos de EV
            total_odd = np.prod([x['odd'] for x in st.session_state.current_ticket])
            prob_real = np.prod([x['prob']/100 for x in st.session_state.current_ticket]) * 100
            fair_odd = 100/prob_real if prob_real > 0 else 0
            ev = (total_odd - fair_odd) / fair_odd * 100 if fair_odd > 0 else 0
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Odd Total", f"{total_odd:.2f}")
            c2.metric("Prob Real", f"{prob_real:.1f}%")
            c3.metric("Fair Odd", f"{fair_odd:.2f}")
            c4.metric("EV (Valor)", f"{ev:+.1f}%", delta_color="normal" if ev>0 else "inverse")
            
            # Botão de Remover (Corrigido)
            st.markdown("##### ⚙️ Editar Bilhete")
            idx_remove = st.selectbox(
                "Remover aposta:",
                range(len(st.session_state.current_ticket)),
                format_func=lambda i: f"{st.session_state.current_ticket[i]['jogo']} - {st.session_state.current_ticket[i]['mercado']}"
            )
            
            if st.button("🗑️ Remover Selecionada"):
                st.session_state.current_ticket.pop(idx_remove)
                st.rerun()

    # ========================================
    # TAB 2: HEDGES MAXIMUM (COMPLETO)
    # ========================================
    with tabs[1]:
        st.header("🛡️ Sistema de Hedges e Proteção")
        if not st.session_state.current_ticket:
            st.warning("Crie um bilhete no Construtor primeiro.")
        else:
            col1, col2 = st.columns(2)
            stake = col1.number_input("Stake Principal (R$)", value=100.0)
            odd_total = np.prod([x['odd'] for x in st.session_state.current_ticket])
            col2.metric("Odd do Bilhete", f"{odd_total:.2f}")
            
            retorno_max = stake * odd_total
            
            st.markdown("### 🛠️ Estratégias Disponíveis")
            
            # HEDGE 1: SMART
            with st.expander("🛡️ 1. Smart Protection (Zebra)", expanded=True):
                st.write("Cobre o prejuízo se a aposta principal perder.")
                odd_hedge = st.number_input("Odd da Contra-Aposta (Zebra):", 2.0, 100.0, 3.5)
                # Cálculo: Stake necessário na zebra para cobrir (Stake princ + Stake zebra)
                # Lucro Zebra = Stake Zebra * Odd Zebra - Stake Zebra
                # Lucro Zebra >= Stake Principal
                # S_z * (O_z - 1) = S_p  => S_z = S_p / (O_z - 1)
                stake_hedge = stake / (odd_hedge - 1)
                
                c1, c2 = st.columns(2)
                c1.metric("Apostar na Zebra", format_currency(stake_hedge))
                c2.metric("Custo Total", format_currency(stake + stake_hedge))
                
                if (stake + stake_hedge) < retorno_max:
                    st.success(f"✅ Hedge Viável! Lucro se Principal bater: {format_currency(retorno_max - (stake + stake_hedge))}")
                else:
                    st.error("🚫 Hedge Inviável (Custo supera retorno).")

            # HEDGE 2: PARTIAL
            with st.expander("⚖️ 2. Partial Protection (50%)"):
                st.write("Protege metade do stake principal.")
                stake_partial = (stake * 0.5) / (odd_hedge - 1)
                st.metric("Apostar na Zebra", format_currency(stake_partial))
                st.info(f"Se Principal perder e Zebra ganhar, você recupera 50% do investimento.")

            # HEDGE 3: ARBITRAGEM
            with st.expander("💎 3. Guaranteed Profit (Dutching)"):
                # Probabilidade Implícita Total
                imp_prob = (1/odd_total) + (1/odd_hedge)
                if imp_prob < 1:
                    st.success(f"💎 ARBITRAGEM DETECTADA! Margem: {imp_prob*100:.1f}%")
                    # Stakes para lucro igual
                    inv_total = stake + stake_hedge # Exemplo base
                    s1 = (inv_total / odd_total) / imp_prob
                    s2 = (inv_total / odd_hedge) / imp_prob
                    st.write(f"Para investir {format_currency(inv_total)} no total:")
                    st.write(f"- Na Principal (@{odd_total:.2f}): {format_currency(s1)}")
                    st.write(f"- Na Zebra (@{odd_hedge:.2f}): {format_currency(s2)}")
                    st.write(f"Lucro Garantido: {format_currency(inv_total/imp_prob - inv_total)}")
                else:
                    st.warning(f"Sem oportunidade de arbitragem (Prob > 100%: {imp_prob*100:.1f}%)")

    # ========================================
    # TAB 3: SIMULADOR MONTE CARLO
    # ========================================
    with tabs[2]:
        st.header("🎲 Simulador Monte Carlo (3.000 Iterações)")
        
        sc1, sc2 = st.columns(2)
        sim_h = sc1.selectbox("Time Casa", sorted(list(STATS.keys())), key='sh')
        sim_a = sc2.selectbox("Time Visitante", sorted(list(STATS.keys())), key='sa')
        
        if st.button("🚀 Iniciar Simulação", use_container_width=True):
            if sim_h != sim_a:
                with st.spinner("Simulando partidas..."):
                    res = simulate_game_v31(STATS[sim_h], STATS[sim_a], {}, 3000)
                    st.success("Concluído!")
                    
                    # Resultados
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Média Gols", f"{res['goals_total'].mean():.2f}")
                    m2.metric("Média Cantos", f"{res['corners_total'].mean():.2f}")
                    m3.metric("Média Cartões", f"{res['cards_total'].mean():.2f}")
                    m4.metric("Prob Over 2.5", f"{(res['goals_total'] > 2.5).mean()*100:.1f}%")
                    
                    # Gráfico
                    fig = px.histogram(res['goals_total'], nbins=10, title="Distribuição de Gols", color_discrete_sequence=['#1e3c72'])
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Tabela
                    probs = pd.DataFrame({
                        'Mercado': ['Over 1.5 Gols', 'Over 2.5 Gols', 'Over 9.5 Cantos', 'Over 10.5 Cantos'],
                        'Probabilidade': [
                            (res['goals_total'] > 1.5).mean(), (res['goals_total'] > 2.5).mean(),
                            (res['corners_total'] > 9.5).mean(), (res['corners_total'] > 10.5).mean()
                        ]
                    })
                    probs['Probabilidade'] = probs['Probabilidade'].apply(lambda x: f"{x*100:.1f}%")
                    st.table(probs)
            else:
                st.error("Times iguais.")

    # ========================================
    # TAB 4: MÉTRICAS
    # ========================================
    with tabs[3]:
        st.header("📊 Métricas de Performance")
        if st.session_state.bet_results:
            df = pd.DataFrame(st.session_state.bet_results)
            wins = df[df['ganhou']==True]
            win_rate = len(wins)/len(df)*100
            roi = (df['lucro'].sum() / df['stake'].sum()) * 100
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Win Rate", f"{win_rate:.1f}%")
            c2.metric("ROI", f"{roi:.1f}%")
            c3.metric("Lucro Total", format_currency(df['lucro'].sum()))
            
            fig = px.line(y=st.session_state.bankroll_history, title="Evolução da Banca")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Registre apostas para ver métricas.")

    # ========================================
    # TAB 5: VISUALIZAÇÕES
    # ========================================
    with tabs[4]:
        st.header("🎨 Visualizações Avançadas")
        opt = st.selectbox("Tipo:", ["Ranking Cantos", "Ranking Cartões", "Ataque vs Defesa"])
        
        if opt == "Ranking Cantos":
            data = [{'Time': k, 'Cantos': v['corners'], 'Liga': v['league']} for k,v in STATS.items()]
            df = pd.DataFrame(data).sort_values('Cantos', ascending=False).head(20)
            fig = px.bar(df, x='Cantos', y='Time', orientation='h', color='Liga', title="Top 20 Cantos")
            st.plotly_chart(fig, use_container_width=True)
            
        elif opt == "Ataque vs Defesa":
            data = [{'Time': k, 'GF': v['goals_f'], 'GS': v['goals_a'], 'Liga': v['league']} for k,v in STATS.items()]
            df = pd.DataFrame(data)
            fig = px.scatter(df, x='GF', y='GS', color='Liga', hover_name='Time', title="Ataque vs Defesa")
            st.plotly_chart(fig, use_container_width=True)
            
        elif opt == "Ranking Cartões":
            data = [{'Time': k, 'Cartões': v['cards'], 'Liga': v['league']} for k,v in STATS.items()]
            df = pd.DataFrame(data).sort_values('Cartões', ascending=False).head(20)
            fig = px.bar(df, x='Cartões', y='Time', orientation='h', color='Liga', title="Top 20 Cartões")
            st.plotly_chart(fig, use_container_width=True)

    # ========================================
    # TAB 6: REGISTRO
    # ========================================
    with tabs[5]:
        st.header("📝 Registro Manual")
        with st.form("reg"):
            c1, c2 = st.columns(2)
            desc = c1.text_input("Descrição")
            stake = c2.number_input("Stake", 10.0)
            c3, c4 = st.columns(2)
            odd = c3.number_input("Odd", 1.01)
            res = c4.selectbox("Resultado", ["Green", "Red", "Void"])
            
            if st.form_submit_button("Salvar"):
                lucro = (stake * odd - stake) if res == "Green" else -stake if res == "Red" else 0
                ganhou = res == "Green"
                st.session_state.bet_results.append({'data': datetime.now().strftime('%d/%m'), 'descricao': desc, 'stake': stake, 'odd': odd, 'ganhou': ganhou, 'lucro': lucro})
                st.session_state.bankroll_history.append(st.session_state.bankroll_history[-1] + lucro)
                st.success("Salvo!")
                st.rerun()
        
        if st.session_state.bet_results:
            st.dataframe(pd.DataFrame(st.session_state.bet_results))

    # ========================================
    # TAB 7: SCANNER
    # ========================================
    with tabs[6]:
        st.header("🔍 Scanner Inteligente")
        if CAL.empty:
            st.warning("Sem calendário.")
        else:
            c1, c2 = st.columns(2)
            d_scan = c1.selectbox("Data Scan:", sorted(CAL['DtObj'].dt.strftime('%d/%m/%Y').unique()))
            mp = st.slider("Prob Mín", 50, 90, 70)
            
            if st.button("🔎 Escanear"):
                hits = []
                for _, r in CAL[CAL['DtObj'].dt.strftime('%d/%m/%Y')==d_scan].iterrows():
                    h, a = normalize_name(r['Time_Casa'], list(STATS.keys())), normalize_name(r['Time_Visitante'], list(STATS.keys()))
                    if h and a:
                        c = calcular_jogo_v31(STATS[h], STATS[a], {})
                        pc = calcular_poisson(c['corners_total'], 9.5)
                        if pc >= mp: hits.append({'Jogo': f"{h} x {a}", 'M': 'O9.5 C', 'Prob': f"{pc:.1f}%"})
                if hits: st.dataframe(pd.DataFrame(hits))
                else: st.warning("Nada encontrado")

    # ========================================
    # TAB 8: IMPORTAR
    # ========================================
    with tabs[7]:
        st.header("📋 Importar Texto")
        txt = st.text_area("Cole seu bilhete:")
        if st.button("Analisar"):
            jogos = parse_bilhete_texto(txt)
            if jogos:
                st.success(f"{len(jogos)} jogos identificados.")
                vals = validar_jogos_bilhete(jogos, STATS)
                if vals:
                    for v in vals:
                        calc = calcular_jogo_v31(v['home_stats'], v['away_stats'], {})
                        st.write(f"✅ **{v['home']} x {v['away']}**: Previsão {calc['corners_total']:.1f} cantos")
            else:
                st.error("Nenhum jogo identificado.")

    # ========================================
    # TAB 9: AI ADVISOR ULTRA (FINAL)
    # ========================================
    with tabs[8]:
        st.header("🤖 AI Advisor ULTRA")
        st.caption("Powered by Causality Engine V31")
        
        chat_c = st.container()
        with chat_c:
            if not st.session_state.chat_history:
                st.info("👋 Olá! Pergunte sobre 'Arsenal vs Chelsea', 'Como está o Flamengo' ou 'Melhores jogos de hoje'.")
            
            for msg in st.session_state.chat_history:
                role = msg['role']
                av = "👤" if role == 'user' else "🤖"
                st.chat_message(role, avatar=av).markdown(msg['content'])
                
        prompt = st.chat_input("Digite sua pergunta...")
        if prompt:
            st.session_state.chat_history.append({'role': 'user', 'content': prompt})
            with st.spinner("🧠 Analisando dados..."):
                resp = processar_chat_ultra(prompt, STATS, CAL, REFS)
            st.session_state.chat_history.append({'role': 'assistant', 'content': resp})
            st.rerun()

if __name__ == "__main__":
    main()
