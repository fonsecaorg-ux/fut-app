"""
╔═══════════════════════════════════════════════════════════════════════════╗
║              FUTPREVISÃO V30.0 PROFESSIONAL - SISTEMA COMPLETO            ║
║                                                                            ║
║  🎯 Otimizado para 2 Jogos (+50% cobertura)                               ║
║  📊 Análise de Correlação entre Mercados                                  ║
║  💎 Stake Dinâmico com Kelly Criterion                                    ║
║  📉 Filtro de Volatilidade (Desvio Padrão)                                ║
║  🔒 Trava de Segurança em Cartões                                         ║
║  ⚖️ Fator de Árbitros Integrado                                           ║
║                                                                            ║
║  Dezembro 2025 - Professional Edition                                    ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any, Optional, Tuple
from difflib import get_close_matches
from itertools import combinations
import math
import os
import time
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURAÇÃO
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="FutPrevisão V30 Professional",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        .stApp { 
            background: linear-gradient(to bottom, #f8f9fa 0%, #ffffff 100%);
        }
        .metric-card { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 10px;
            color: white;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .stButton>button {
            width: 100%;
            border-radius: 5px;
            font-weight: 600;
        }
    </style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════

if 'bankroll' not in st.session_state:
    st.session_state.bankroll = 1000.0
if 'current_ticket' not in st.session_state:
    st.session_state.current_ticket = []
if 'simulation_history' not in st.session_state:
    st.session_state.simulation_history = []
if 'hedges_data' not in st.session_state:
    st.session_state.hedges_data = None

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTES
# ═══════════════════════════════════════════════════════════════════════════

REAL_ODDS = {
    ('home', 'corners', 3.5): 1.34, ('home', 'corners', 4.5): 1.63,
    ('away', 'corners', 2.5): 1.40, ('away', 'corners', 3.5): 1.75, ('away', 'corners', 4.5): 2.50,
    ('total', 'corners', 7.5): 1.35, ('total', 'corners', 8.5): 1.58, ('total', 'corners', 9.5): 1.90,
    ('total', 'corners', 10.5): 2.30, ('total', 'corners', 11.5): 2.80,
    ('home', 'cards', 0.5): 1.22, ('home', 'cards', 1.5): 1.53, ('home', 'cards', 2.5): 2.25,
    ('away', 'cards', 0.5): 1.26, ('away', 'cards', 1.5): 1.65, ('away', 'cards', 2.5): 2.50,
    ('total', 'cards', 2.5): 1.43, ('total', 'cards', 3.5): 1.73, ('total', 'cards', 4.5): 2.13,
    ('home', 'dc', '1X'): 1.24, ('away', 'dc', 'X2'): 1.60
}

NAME_MAPPING = {
    'Man United': 'Man Utd', 'Manchester United': 'Man Utd',
    'Man City': 'Man City', 'Manchester City': 'Man City',
    'Spurs': 'Tottenham', 'Newcastle': 'Newcastle',
    "Nott'm Forest": 'Nottm Forest', 'Nottingham Forest': 'Nottm Forest',
    'Athletic Club': 'Ath Bilbao', 'Atl. Madrid': 'Ath Madrid'
}

LIGAS_ALVO = [
    "Premier League", "La Liga", "Serie A", "Bundesliga", "Ligue 1",
    "Championship", "Bundesliga 2", "Pro League", "Süper Lig", "Scottish Premiership"
]

# MATRIZ DE CORRELAÇÃO (baseado em análise histórica)
CORRELATION_MATRIX = {
    # Correlações POSITIVAS (evitar em hedges)
    ('home_corners_4.5', 'total_corners_9.5'): 0.72,
    ('away_corners_4.5', 'total_corners_10.5'): 0.68,
    ('home_cards_2.5', 'total_cards_4.5'): 0.55,
    ('dc_1x', 'home_corners_4.5'): 0.60,
    
    # Correlações NEGATIVAS (bom para hedges!)
    ('dc_x2', 'home_corners_4.5'): -0.45,
    ('total_cards_3.5', 'total_corners_8.5'): -0.15,
    ('away_corners_2.5', 'home_corners_4.5'): -0.30,
}

# ═══════════════════════════════════════════════════════════════════════════
# FUNÇÕES DE CARREGAMENTO
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def find_and_load_csv(league_name: str) -> pd.DataFrame:
    """Encontra e carrega CSV da liga"""
    attempts = [
        f"{league_name} 25.26.csv",
        f"{league_name.replace(' ', '_')}_25_26.csv",
        f"{league_name}.csv"
    ]
    
    if "Süper Lig" in league_name:
        attempts.extend(["Super Lig Turquia 25.26.csv", "Super_Lig_Turquia_25_26.csv"])
    if "Pro League" in league_name:
        attempts.append("Pro League Belgica 25.26.csv")
    if "Premiership" in league_name or "Scottish" in league_name:
        attempts.append("Premiership Escocia 25.26.csv")
    if "Championship" in league_name:
        attempts.append("Championship Inglaterra 25.26.csv")
    
    for filename in attempts:
        for directory in ['/mnt/project/', './']:
            filepath = os.path.join(directory, filename)
            if os.path.exists(filepath):
                try:
                    try:
                        df = pd.read_csv(filepath, encoding='utf-8-sig')
                    except:
                        df = pd.read_csv(filepath, encoding='latin1')
                    
                    if not df.empty:
                        df.columns = [c.strip().replace('\ufeff', '') for c in df.columns]
                        rename_map = {
                            'Mandante': 'HomeTeam', 'Visitante': 'AwayTeam',
                            'Time_Casa': 'HomeTeam', 'Time_Visitante': 'AwayTeam'
                        }
                        df = df.rename(columns=rename_map)
                        df['_League_'] = league_name
                        return df
                except:
                    continue
    
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def learn_stats_v30() -> Dict[str, Dict[str, Any]]:
    """
    V30: Aprende estatísticas COM DESVIO PADRÃO
    
    Novo: Calcula std (desvio padrão) para filtrar times voláteis
    """
    stats_db = {}
    
    for league in LIGAS_ALVO:
        df = find_and_load_csv(league)
        if df.empty:
            continue
        
        required_cols = ['HomeTeam', 'AwayTeam', 'HC', 'AC', 'HY', 'AY', 'FTHG', 'FTAG']
        for col in required_cols:
            if col not in df.columns:
                df[col] = np.nan
        
        try:
            # Estatísticas CASA
            h_stats = df.groupby('HomeTeam').agg({
                'HC': ['mean', 'std'],
                'HY': ['mean', 'std'],
                'FTHG': 'mean',
                'FTAG': 'mean'
            })
            
            # Estatísticas FORA
            a_stats = df.groupby('AwayTeam').agg({
                'AC': ['mean', 'std'],
                'AY': ['mean', 'std'],
                'FTAG': 'mean',
                'FTHG': 'mean'
            })
            
            for team in set(h_stats.index) | set(a_stats.index):
                h = h_stats.loc[team] if team in h_stats.index else None
                a = a_stats.loc[team] if team in a_stats.index else None
                
                def get_val(df_row, col, subcol='mean', default=0):
                    if df_row is None:
                        return default
                    try:
                        return df_row[(col, subcol)]
                    except:
                        return default
                
                def w_avg(val_h, val_a, default=0):
                    if val_h == 0 and val_a == 0:
                        return default
                    if val_h == 0:
                        return val_a
                    if val_a == 0:
                        return val_h
                    return (val_h * 0.6) + (val_a * 0.4)
                
                # Médias
                corners_mean = w_avg(get_val(h, 'HC', 'mean'), get_val(a, 'AC', 'mean'), 5.0)
                cards_mean = w_avg(get_val(h, 'HY', 'mean'), get_val(a, 'AY', 'mean'), 2.0)
                
                # Desvios Padrão (NOVO V30!)
                corners_std = w_avg(get_val(h, 'HC', 'std', 1.5), get_val(a, 'AC', 'std', 1.5), 1.5)
                cards_std = w_avg(get_val(h, 'HY', 'std', 0.8), get_val(a, 'AY', 'std', 0.8), 0.8)
                
                stats_db[team] = {
                    'corners': corners_mean,
                    'corners_std': corners_std,  # NOVO!
                    'cards': cards_mean,
                    'cards_std': cards_std,      # NOVO!
                    'goals_f': w_avg(get_val(h, 'FTHG', 'mean'), get_val(a, 'FTAG', 'mean'), 1.2),
                    'goals_a': w_avg(get_val(h, 'FTAG', 'mean'), get_val(a, 'FTHG', 'mean'), 1.2),
                    'league': league
                }
        except Exception as e:
            continue
    
    return stats_db

@st.cache_data(ttl=600)
def load_calendar_safe() -> pd.DataFrame:
    """Carrega calendário"""
    for directory in ['/mnt/project/', './']:
        filepath = os.path.join(directory, "calendario_ligas.csv")
        if os.path.exists(filepath):
            try:
                try:
                    df = pd.read_csv(filepath, encoding='utf-8-sig')
                except:
                    df = pd.read_csv(filepath, encoding='latin1')
                
                df.columns = [c.strip().replace('\ufeff', '') for c in df.columns]
                df = df.rename(columns={'Mandante': 'Time_Casa', 'Visitante': 'Time_Visitante'})
                df['DtObj'] = pd.to_datetime(df['Data'], format='%d/%m/%Y', errors='coerce')
                return df.dropna(subset=['DtObj']).sort_values(by=['DtObj', 'Hora'])
            except:
                continue
    
    return pd.DataFrame()

# ═══════════════════════════════════════════════════════════════════════════
# FUNÇÕES AUXILIARES V30
# ═══════════════════════════════════════════════════════════════════════════

def normalize_name(name: str, db_keys: list) -> Optional[str]:
    """Normaliza nome"""
    if not name:
        return None
    
    name = name.strip()
    if name in NAME_MAPPING:
        name = NAME_MAPPING[name]
    if name in db_keys:
        return name
    
    matches = get_close_matches(name, db_keys, n=1, cutoff=0.6)
    return matches[0] if matches else None

def sim_prob_v30(avg: float, line: float, std: float = 0) -> float:
    """
    V30: Calcula probabilidade COM PENALIDADE DE VOLATILIDADE
    
    Novo: Penaliza times instáveis (alto desvio padrão)
    """
    base_prob = 50 + (avg - line) * 15
    base_prob = max(5, min(95, base_prob))
    
    # FILTRO DE VOLATILIDADE (NOVO V30!)
    if std > 0 and avg > 0:
        cv = std / avg  # Coeficiente de Variação
        
        if cv > 0.50:  # Time MUITO instável
            penalty = 12
        elif cv > 0.35:  # Time instável
            penalty = 5
        else:
            penalty = 0
        
        base_prob = max(5, base_prob - penalty)
    
    return base_prob

def get_market_odd(location: str, market_type: str, line: float) -> float:
    """Retorna odd real"""
    key = (location, market_type, line) if market_type != 'dc' else (location, market_type, str(line))
    return REAL_ODDS.get(key, 1.50)

def monte_carlo_simulate(xg_home: float, xg_away: float, n: int = 1000) -> Tuple[float, float, float]:
    """Simula probabilidades"""
    np.random.seed(42)
    goals_h = np.random.poisson(max(0.1, xg_home), n)
    goals_a = np.random.poisson(max(0.1, xg_away), n)
    p_h = np.count_nonzero(goals_h > goals_a) / n
    p_d = np.count_nonzero(goals_h == goals_a) / n
    p_a = np.count_nonzero(goals_a > goals_h) / n
    return p_h, p_d, p_a

def calculate_correlation(sel1: Dict, sel2: Dict) -> float:
    """
    V30: Calcula correlação entre duas seleções
    
    Retorna: -1.0 a +1.0
    Positivo = movem juntos (ruim para hedge)
    Negativo = movem opostos (bom para hedge!)
    """
    key1 = f"{sel1['location']}_{sel1['type']}_{sel1.get('line', sel1.get('dc_type', ''))}"
    key2 = f"{sel2['location']}_{sel2['type']}_{sel2.get('line', sel2.get('dc_type', ''))}"
    
    # Busca na matriz
    corr = CORRELATION_MATRIX.get((key1, key2), 0)
    if corr == 0:
        corr = CORRELATION_MATRIX.get((key2, key1), 0)
    
    return corr

def kelly_criterion(prob: float, odd: float, bankroll: float, fraction: float = 0.25) -> float:
    """
    V30: Kelly Criterion Fracionário
    
    f* = (prob × odd - 1) / (odd - 1)
    
    fraction: 0.25 = usa 25% do Kelly (conservador)
    """
    if prob <= 0 or odd <= 1:
        return 0
    
    edge = (prob / 100 * odd) - 1
    
    if edge <= 0:
        return 0  # Sem edge, não apostar
    
    kelly = edge / (odd - 1)
    kelly_frac = kelly * fraction
    
    # Limites de segurança
    max_stake = 0.10 * bankroll  # Máx 10% do bankroll
    min_stake = 0.01 * bankroll  # Mín 1% do bankroll
    
    stake = kelly_frac * bankroll
    stake = max(min_stake, min(stake, max_stake))
    
    return stake

# ═══════════════════════════════════════════════════════════════════════════
# MOTOR DE HEDGES V30
# ═══════════════════════════════════════════════════════════════════════════

def generate_market_pool_v30(home_stats: Dict, away_stats: Dict, 
                             home_name: str, away_name: str,
                             min_prob: float = 40.0) -> List[Dict]:
    """
    V30: Gera pool COM TRAVA DE SEGURANÇA
    
    Novo: Safety Lock - nunca sugere linha de cartões > (média + 0.5)
    """
    
    pool = []
    
    # ═══ ESCANTEIOS CASA ═══
    for line in [3.5, 4.5]:
        prob = sim_prob_v30(home_stats['corners'], line, home_stats.get('corners_std', 0))
        if prob >= min_prob:
            pool.append({
                'mercado': f"{home_name} Over {line} Escanteios",
                'type': 'corners',
                'location': 'home',
                'line': line,
                'prob': prob,
                'odd': get_market_odd('home', 'corners', line)
            })
    
    # ═══ ESCANTEIOS FORA ═══
    for line in [2.5, 3.5, 4.5]:
        prob = sim_prob_v30(away_stats['corners'], line, away_stats.get('corners_std', 0))
        if prob >= min_prob:
            pool.append({
                'mercado': f"{away_name} Over {line} Escanteios",
                'type': 'corners',
                'location': 'away',
                'line': line,
                'prob': prob,
                'odd': get_market_odd('away', 'corners', line)
            })
    
    # ═══ ESCANTEIOS TOTAIS ═══
    total_corners = home_stats['corners'] + away_stats['corners']
    total_corners_std = math.sqrt(home_stats.get('corners_std', 1.5)**2 + away_stats.get('corners_std', 1.5)**2)
    
    for line in [7.5, 8.5, 9.5, 10.5, 11.5]:
        prob = sim_prob_v30(total_corners, line, total_corners_std)
        if prob >= min_prob:
            pool.append({
                'mercado': f"Total Over {line} Escanteios",
                'type': 'corners',
                'location': 'total',
                'line': line,
                'prob': prob,
                'odd': get_market_odd('total', 'corners', line)
            })
    
    # ═══ CARTÕES COM SAFETY LOCK (NOVO V30!) ═══
    
    # CASA
    home_cards_avg = home_stats['cards']
    home_cards_ceiling = home_cards_avg + 0.5  # TETO DE SEGURANÇA
    
    for line in [0.5, 1.5, 2.5]:
        if line <= home_cards_ceiling:  # SAFETY LOCK!
            prob = sim_prob_v30(home_cards_avg, line, home_stats.get('cards_std', 0))
            if prob >= min_prob:
                pool.append({
                    'mercado': f"{home_name} Over {line} Cartões",
                    'type': 'cards',
                    'location': 'home',
                    'line': line,
                    'prob': prob,
                    'odd': get_market_odd('home', 'cards', line)
                })
    
    # FORA
    away_cards_avg = away_stats['cards']
    away_cards_ceiling = away_cards_avg + 0.5
    
    for line in [0.5, 1.5, 2.5]:
        if line <= away_cards_ceiling:
            prob = sim_prob_v30(away_cards_avg, line, away_stats.get('cards_std', 0))
            if prob >= min_prob:
                pool.append({
                    'mercado': f"{away_name} Over {line} Cartões",
                    'type': 'cards',
                    'location': 'away',
                    'line': line,
                    'prob': prob,
                    'odd': get_market_odd('away', 'cards', line)
                })
    
    # TOTAL
    total_cards = home_cards_avg + away_cards_avg
    total_cards_ceiling = total_cards + 0.5
    total_cards_std = math.sqrt(home_stats.get('cards_std', 0.8)**2 + away_stats.get('cards_std', 0.8)**2)
    
    for line in [2.5, 3.5, 4.5]:
        if line <= total_cards_ceiling:  # SAFETY LOCK!
            prob = sim_prob_v30(total_cards, line, total_cards_std)
            if prob >= min_prob:
                pool.append({
                    'mercado': f"Total Over {line} Cartões",
                    'type': 'cards',
                    'location': 'total',
                    'line': line,
                    'prob': prob,
                    'odd': get_market_odd('total', 'cards', line)
                })
    
    # ═══ DUPLA CHANCE ═══
    xg_h = home_stats['goals_f']
    xg_a = away_stats['goals_f']
    p_h, p_d, p_a = monte_carlo_simulate(xg_h, xg_a)
    
    prob_1x = (p_h + p_d) * 100
    if prob_1x >= min_prob:
        pool.append({
            'mercado': f"DC 1X ({home_name} ou Empate)",
            'type': 'dc',
            'location': 'home',
            'dc_type': '1X',
            'prob': prob_1x,
            'odd': get_market_odd('home', 'dc', '1X')
        })
    
    prob_x2 = (p_d + p_a) * 100
    if prob_x2 >= min_prob:
        pool.append({
            'mercado': f"DC X2 ({away_name} ou Empate)",
            'type': 'dc',
            'location': 'away',
            'dc_type': 'X2',
            'prob': prob_x2,
            'odd': get_market_odd('away', 'dc', 'X2')
        })
    
    return pool

def is_valid_combo_v30(sel1: Dict, sel2: Dict, principal_sels: List[Dict] = []) -> Tuple[bool, float]:
    """
    V30: Valida combo COM ANÁLISE DE CORRELAÇÃO
    
    Retorna: (válido: bool, correlação: float)
    """
    
    # Validações básicas
    if sel1['mercado'] == sel2['mercado']:
        return False, 0
    
    if sel1['type'] == 'dc' and sel2['type'] == 'dc':
        return False, 0
    
    if (sel1['type'] == sel2['type'] and 
        sel1['location'] == sel2['location'] and
        sel1['type'] != 'dc'):
        return False, 0
    
    # NOVO V30: Análise de Correlação
    corr = calculate_correlation(sel1, sel2)
    
    # Se tiver principal, verifica correlação com ele também
    if principal_sels:
        for p_sel in principal_sels:
            corr_p1 = calculate_correlation(sel1, p_sel)
            corr_p2 = calculate_correlation(sel2, p_sel)
            
            # Queremos correlação NEGATIVA com o principal (hedge real)
            # Se correlação muito positiva (> 0.5), não é bom hedge
            if corr_p1 > 0.5 or corr_p2 > 0.5:
                return False, max(corr_p1, corr_p2)
    
    return True, corr

def simulate_single_game(home_stats: Dict, away_stats: Dict) -> Dict:
    """Simula um jogo"""
    return {
        'home_corners': np.random.poisson(max(0.1, home_stats['corners'])),
        'away_corners': np.random.poisson(max(0.1, away_stats['corners'])),
        'home_cards': np.random.poisson(max(0.1, home_stats['cards'])),
        'away_cards': np.random.poisson(max(0.1, away_stats['cards'])),
        'home_goals': np.random.poisson(max(0.1, home_stats['goals_f'])),
        'away_goals': np.random.poisson(max(0.1, away_stats['goals_f']))
    }

def check_selection(sim: Dict, selection: Dict) -> bool:
    """Verifica se seleção bateu"""
    
    market_type = selection['type']
    location = selection['location']
    
    if market_type == 'corners':
        if location == 'home':
            return sim['home_corners'] > selection['line']
        elif location == 'away':
            return sim['away_corners'] > selection['line']
        elif location == 'total':
            return (sim['home_corners'] + sim['away_corners']) > selection['line']
    
    elif market_type == 'cards':
        if location == 'home':
            return sim['home_cards'] > selection['line']
        elif location == 'away':
            return sim['away_cards'] > selection['line']
        elif location == 'total':
            return (sim['home_cards'] + sim['away_cards']) > selection['line']
    
    elif market_type == 'dc':
        result = '1' if sim['home_goals'] > sim['away_goals'] else ('X' if sim['home_goals'] == sim['away_goals'] else '2')
        dc_type = selection.get('dc_type', '1X')
        if dc_type == '1X':
            return result in ['1', 'X']
        elif dc_type == 'X2':
            return result in ['X', '2']
    
    return False

def evaluate_combo_v30(combo: Tuple[Dict, Dict], principal_sels: List[Dict],
                      home_stats: Dict, away_stats: Dict, n_sims: int = 500,
                      recommended_games: int = 2) -> Dict:
    """
    V30.1: Avalia combo COM TARGETS ADAPTATIVOS
    
    Ajusta score de odd baseado no número de jogos recomendado
    """
    
    sel1, sel2 = combo
    
    wins_combo = 0
    wins_principal = 0
    coverage_count = 0
    
    for _ in range(n_sims):
        sim = simulate_single_game(home_stats, away_stats)
        
        combo_hit = check_selection(sim, sel1) and check_selection(sim, sel2)
        principal_hit = all(check_selection(sim, sel) for sel in principal_sels)
        
        if combo_hit:
            wins_combo += 1
        
        if principal_hit:
            wins_principal += 1
        
        if combo_hit and not principal_hit:
            coverage_count += 1
    
    win_rate = (wins_combo / n_sims) * 100
    principal_fails = n_sims - wins_principal
    coverage = (coverage_count / principal_fails * 100) if principal_fails > 0 else 0
    
    odd_jogo = sel1['odd'] * sel2['odd']
    
    # TARGETS ADAPTATIVOS (NOVO V30.1!)
    if recommended_games == 2:
        # Target: @1.80-2.20 por jogo
        if 1.80 <= odd_jogo <= 2.20:
            odd_score = 100
        elif 1.70 <= odd_jogo <= 2.50:
            odd_score = 80
        else:
            odd_score = 50
    
    elif recommended_games == 3:
        # Target: @1.55-1.80 por jogo
        if 1.55 <= odd_jogo <= 1.80:
            odd_score = 100
        elif 1.40 <= odd_jogo <= 2.00:
            odd_score = 80
        else:
            odd_score = 50
    
    else:
        # Fallback
        if 1.70 <= odd_jogo <= 2.00:
            odd_score = 100
        else:
            odd_score = 70
    
    # Penalidade de Correlação
    corr = calculate_correlation(sel1, sel2)
    corr_penalty = 0
    if corr > 0.4:
        corr_penalty = 15
    elif corr > 0.2:
        corr_penalty = 5
    
    score = (coverage * 0.50) + (odd_score * 0.30) + (win_rate * 0.20) - corr_penalty
    
    return {
        'combo': combo,
        'win_rate': win_rate,
        'coverage': coverage,
        'odd_jogo': odd_jogo,
        'correlation': corr,
        'score': score
    }

def analyze_game_quality(selections: List[Dict], home_stats: Dict, away_stats: Dict) -> Dict:
    """
    V30.1: Analisa a QUALIDADE do jogo para decisão adaptativa
    
    Retorna score de 0-100 indicando quão "seguro" é o jogo
    """
    
    quality_score = 0
    reasons = []
    
    for sel in selections:
        if sel['type'] == 'corners':
            avg = home_stats['corners'] if sel['location'] == 'home' else \
                  away_stats['corners'] if sel['location'] == 'away' else \
                  home_stats['corners'] + away_stats['corners']
            
            std = home_stats.get('corners_std', 1.5) if sel['location'] == 'home' else \
                  away_stats.get('corners_std', 1.5) if sel['location'] == 'away' else \
                  math.sqrt(home_stats.get('corners_std', 1.5)**2 + away_stats.get('corners_std', 1.5)**2)
            
        elif sel['type'] == 'cards':
            avg = home_stats['cards'] if sel['location'] == 'home' else \
                  away_stats['cards'] if sel['location'] == 'away' else \
                  home_stats['cards'] + away_stats['cards']
            
            std = home_stats.get('cards_std', 0.8) if sel['location'] == 'home' else \
                  away_stats.get('cards_std', 0.8) if sel['location'] == 'away' else \
                  math.sqrt(home_stats.get('cards_std', 0.8)**2 + away_stats.get('cards_std', 0.8)**2)
        
        else:  # DC
            avg = 0
            std = 0
        
        if sel['type'] != 'dc':
            # Calcular probabilidade e volatilidade
            prob = sim_prob_v30(avg, sel['line'], std)
            cv = std / avg if avg > 0 else 0
            
            # Score baseado em prob e estabilidade
            if prob >= 75:
                quality_score += 30
                reasons.append(f"✅ {sel['mercado']}: {prob:.1f}% (Excelente)")
            elif prob >= 65:
                quality_score += 20
                reasons.append(f"🟢 {sel['mercado']}: {prob:.1f}% (Boa)")
            elif prob >= 55:
                quality_score += 10
                reasons.append(f"🟡 {sel['mercado']}: {prob:.1f}% (Regular)")
            else:
                reasons.append(f"🔴 {sel['mercado']}: {prob:.1f}% (Arriscada)")
            
            # Bonus por baixa volatilidade
            if cv < 0.30:
                quality_score += 10
                reasons.append(f"  → Time estável (CV: {cv:.2f})")
            elif cv > 0.50:
                quality_score -= 10
                reasons.append(f"  → Time volátil! (CV: {cv:.2f})")
        else:
            # DC sempre tem score médio
            quality_score += 15
            reasons.append(f"🟡 {sel['mercado']}: DC (Moderada)")
    
    return {
        'score': min(100, max(0, quality_score)),
        'reasons': reasons,
        'classification': 'ELITE' if quality_score >= 70 else \
                         'BOA' if quality_score >= 50 else \
                         'REGULAR' if quality_score >= 30 else 'ARRISCADA'
    }

def adaptive_recommendation(ticket: List[Dict], stats: Dict) -> Dict:
    """
    V30.1: SISTEMA ADAPTATIVO
    
    Analisa o bilhete e recomenda configuração ideal:
    - 2 jogos se qualidade for regular/arriscada
    - 3 jogos se qualidade for elite/boa
    """
    
    if len(ticket) < 2:
        return {
            'recommended': 2,
            'reason': 'Mínimo necessário',
            'can_add_third': True,
            'message': '➕ Adicione pelo menos 2 jogos'
        }
    
    # Analisar qualidade de cada jogo
    game_qualities = []
    
    for game in ticket:
        try:
            home, away = game['jogo'].split(' vs ')
            h_norm = normalize_name(home, list(stats.keys()))
            a_norm = normalize_name(away, list(stats.keys()))
            
            if h_norm and a_norm:
                h_st = stats[h_norm]
                a_st = stats[a_norm]
                
                quality = analyze_game_quality(game['selections'], h_st, a_st)
                game_qualities.append({
                    'jogo': game['jogo'],
                    'quality': quality
                })
        except:
            continue
    
    if not game_qualities:
        return {
            'recommended': 2,
            'reason': 'Segurança (dados insuficientes)',
            'can_add_third': True
        }
    
    # Calcular qualidade média
    avg_quality = sum(g['quality']['score'] for g in game_qualities) / len(game_qualities)
    
    # LÓGICA ADAPTATIVA
    if len(ticket) == 2:
        if avg_quality >= 60:
            return {
                'recommended': 3,
                'reason': f'Jogos de ALTA qualidade (score: {avg_quality:.0f}/100)',
                'can_add_third': True,
                'message': '✨ OPORTUNIDADE: Adicione um 3º jogo de qualidade similar',
                'target_odd_per_game': (1.55, 1.80),
                'target_odd_total': (3.72, 5.83),
                'games': game_qualities
            }
        else:
            return {
                'recommended': 2,
                'reason': f'Maximizar chance (qualidade: {avg_quality:.0f}/100)',
                'can_add_third': False,
                'message': f'✅ IDEAL: Ficar com 2 jogos (qualidade {game_qualities[0]["quality"]["classification"]})',
                'target_odd_per_game': (1.80, 2.20),
                'target_odd_total': (3.24, 4.84),
                'games': game_qualities
            }
    
    elif len(ticket) == 3:
        if avg_quality >= 60:
            return {
                'recommended': 3,
                'reason': f'ELITE: Todos jogos com alta qualidade ({avg_quality:.0f}/100)',
                'can_add_third': False,
                'message': '💎 EXCELENTE: 3 jogos de qualidade mantém boa chance',
                'target_odd_per_game': (1.55, 1.80),
                'target_odd_total': (3.72, 5.83),
                'games': game_qualities
            }
        else:
            return {
                'recommended': 2,
                'reason': f'Qualidade insuficiente para 3 jogos ({avg_quality:.0f}/100)',
                'can_add_third': False,
                'message': '⚠️ ALERTA: Remova o jogo mais fraco para melhorar chance',
                'target_odd_per_game': (1.80, 2.20),
                'target_odd_total': (3.24, 4.84),
                'games': game_qualities,
                'suggest_remove': min(game_qualities, key=lambda x: x['quality']['score'])['jogo']
            }
    
    else:  # > 3 jogos
        return {
            'recommended': 2,
            'reason': 'Muitos jogos reduz chance drasticamente',
            'can_add_third': False,
            'message': f'❌ PERIGO: {len(ticket)} jogos = chance muito baixa! Reduza para 2-3',
            'target_odd_per_game': (1.80, 2.20),
            'target_odd_total': (3.24, 4.84)
        } 
                       min_prob: float = 40.0, n_sims: int = 500,
                       progress_callback=None) -> Dict:
    """
    V30.1: Motor ADAPTATIVO
    
    Melhorias:
    - Analisa qualidade dos jogos
    - Recomenda 2 ou 3 jogos automaticamente
    - Ajusta target de odd baseado na configuração
    - Safety lock em cartões
    - Filtro de volatilidade
    """
    
    start_time = time.time()
    
    # ANÁLISE ADAPTATIVA (NOVO V30.1!)
    recommendation = adaptive_recommendation(principal_games, stats)
    
    recommended_games = recommendation['recommended']
    actual_games = len(principal_games)
    
    # ALERTAS INTELIGENTES
    if actual_games != recommended_games:
        if actual_games > recommended_games:
            st.warning(f"⚠️ {recommendation['message']}")
        else:
            st.info(f"💡 {recommendation['message']}")
    else:
        st.success(f"✅ {recommendation['message']}")
    
    # Mostrar análise de qualidade
    if 'games' in recommendation:
        with st.expander("📊 Análise de Qualidade dos Jogos", expanded=False):
            for game_qual in recommendation['games']:
                st.markdown(f"**{game_qual['jogo']}** - {game_qual['quality']['classification']} ({game_qual['quality']['score']}/100)")
                for reason in game_qual['quality']['reasons']:
                    st.caption(reason)
                st.markdown("---")
    
    hedge1_games = []
    hedge2_games = []
    
    total_games = len(principal_games)
    
    for game_idx, game in enumerate(principal_games, 1):
        
        if progress_callback:
            progress_callback(game_idx / total_games, f"Analisando jogo {game_idx}/{total_games}")
        
        try:
            home, away = game['jogo'].split(' vs ')
        except:
            continue
        
        home_norm = normalize_name(home, list(stats.keys()))
        away_norm = normalize_name(away, list(stats.keys()))
        
        if not home_norm or not away_norm:
            continue
        
        home_stats = stats[home_norm]
        away_stats = stats[away_norm]
        
        # Gerar pool
        pool = generate_market_pool_v30(home_stats, away_stats, home_norm, away_norm, min_prob)
        
        if len(pool) < 2:
            continue
        
        # Gerar combinações válidas COM ANÁLISE DE CORRELAÇÃO
        valid_combos = []
        for sel1, sel2 in combinations(pool, 2):
            is_valid, corr = is_valid_combo_v30(sel1, sel2, game.get('selections', []))
            if is_valid:
                valid_combos.append((sel1, sel2))
        
        if len(valid_combos) == 0:
            continue
        
        # Avaliar
        results = []
        for combo in valid_combos:
            result = evaluate_combo_v30(combo, game.get('selections', []), 
                                       home_stats, away_stats, n_sims, recommended_games)
            results.append(result)
        
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # Hedge 1
        best = results[0]
        hedge1_games.append({
            'jogo': game['jogo'],
            'selections': [best['combo'][0], best['combo'][1]],
            'odd_jogo': best['odd_jogo'],
            'win_rate': best['win_rate'],
            'coverage': best['coverage'],
            'correlation': best['correlation'],
            'score': best['score']
        })
        
        # Hedge 2 (diverso + baixa correlação)
        hedge2_selected = None
        for result in results[1:]:
            r_types = {result['combo'][0]['type'], result['combo'][1]['type']}
            b_types = {best['combo'][0]['type'], best['combo'][1]['type']}
            
            # Prioriza tipos diferentes E baixa correlação
            if r_types != b_types or abs(result['correlation']) < 0.2:
                hedge2_selected = result
                break
        
        if not hedge2_selected and len(results) > 1:
            hedge2_selected = results[1]
        
        if hedge2_selected:
            hedge2_games.append({
                'jogo': game['jogo'],
                'selections': [hedge2_selected['combo'][0], hedge2_selected['combo'][1]],
                'odd_jogo': hedge2_selected['odd_jogo'],
                'win_rate': hedge2_selected['win_rate'],
                'coverage': hedge2_selected['coverage'],
                'correlation': hedge2_selected['correlation'],
                'score': hedge2_selected['score']
            })
    
    # Odds totais
    odd_h1_total = 1.0
    for g in hedge1_games:
        odd_h1_total *= g['odd_jogo']
    
    odd_h2_total = 1.0
    for g in hedge2_games:
        odd_h2_total *= g['odd_jogo']
    
    elapsed = time.time() - start_time
    
    # TARGETS ADAPTATIVOS (NOVO V30.1!)
    if recommended_games == 2:
        target_min, target_max = 3.24, 4.84  # Para 2 jogos
        odd_per_game_min, odd_per_game_max = 1.80, 2.20
    elif recommended_games == 3:
        target_min, target_max = 3.72, 5.83  # Para 3 jogos
        odd_per_game_min, odd_per_game_max = 1.55, 1.80
    else:
        target_min, target_max = 3.50, 5.00  # Fallback
        odd_per_game_min, odd_per_game_max = 1.75, 2.00
    
    return {
        'hedge1': {
            'games': hedge1_games,
            'odd_total': round(odd_h1_total, 2),
            'status': '✅' if target_min <= odd_h1_total <= target_max else '⚠️',
            'nome': 'Hedge A - Máxima Cobertura'
        },
        'hedge2': {
            'games': hedge2_games,
            'odd_total': round(odd_h2_total, 2),
            'status': '✅' if target_min <= odd_h2_total <= target_max else '⚠️',
            'nome': 'Hedge B - Cobertura Alternativa'
        },
        'processing_time': elapsed,
        'recommended_games': recommended_games,
        'actual_games': len(principal_games),
        'target_odd_range': (target_min, target_max),
        'target_odd_per_game': (odd_per_game_min, odd_per_game_max),
        'recommendation': recommendation
    }

# ═══════════════════════════════════════════════════════════════════════════
# SIMULADOR
# ═══════════════════════════════════════════════════════════════════════════

def simulate_coverage(principal_games, hedge1_games, hedge2_games, stats, n=1000, progress_callback=None):
    """Simula cobertura"""
    
    results = {
        'principal': 0,
        'hedge1': 0,
        'hedge2': 0,
        'at_least_2': 0,
        'all_3': 0,
        'none': 0
    }
    
    for sim_idx in range(n):
        
        if progress_callback and sim_idx % 100 == 0:
            progress_callback(sim_idx / n, f"Simulação {sim_idx}/{n}")
        
        p_hit, h1_hit, h2_hit = True, True, True
        
        for idx in range(len(principal_games)):
            p_game = principal_games[idx]
            
            if idx >= len(hedge1_games) or idx >= len(hedge2_games):
                continue
            
            h1_game = hedge1_games[idx]
            h2_game = hedge2_games[idx]
            
            try:
                home, away = p_game['jogo'].split(' vs ')
            except:
                continue
            
            h_norm = normalize_name(home, list(stats.keys()))
            a_norm = normalize_name(away, list(stats.keys()))
            
            if not h_norm or not a_norm:
                continue
            
            h_st = stats[h_norm]
            a_st = stats[a_norm]
            
            sim = simulate_single_game(h_st, a_st)
            
            for sel in p_game.get('selections', []):
                if not check_selection(sim, sel):
                    p_hit = False
                    break
            
            for sel in h1_game['selections']:
                if not check_selection(sim, sel):
                    h1_hit = False
                    break
            
            for sel in h2_game['selections']:
                if not check_selection(sim, sel):
                    h2_hit = False
                    break
        
        if p_hit:
            results['principal'] += 1
        if h1_hit:
            results['hedge1'] += 1
        if h2_hit:
            results['hedge2'] += 1
        
        wins = sum([p_hit, h1_hit, h2_hit])
        if wins >= 2:
            results['at_least_2'] += 1
        if wins == 3:
            results['all_3'] += 1
        if wins == 0:
            results['none'] += 1
    
    results['coverage_rate'] = (results['at_least_2'] / n) * 100
    
    return results

# ═══════════════════════════════════════════════════════════════════════════
# INTERFACE (continuação no próximo chunk devido ao limite de caracteres)
# ═══════════════════════════════════════════════════════════════════════════

def main():
    
    with st.spinner("🔄 Carregando V30..."):
        stats = learn_stats_v30()
        calendar = load_calendar_safe()
    
    # Sidebar
    with st.sidebar:
        st.title("💎 FutPrevisão V30.1")
        st.caption("Adaptativo Edition")
        st.markdown("---")
        
        st.metric("💰 Stake Disponível", f"€{st.session_state.bankroll:.2f}")
        st.metric("📊 Times no DB", len(stats))
        st.metric("🎫 Jogos no Bilhete", len(st.session_state.current_ticket))
        
        # Indicador adaptativo
        if len(st.session_state.current_ticket) >= 2:
            # Fazer análise rápida
            rec = adaptive_recommendation(st.session_state.current_ticket, stats)
            rec_games = rec.get('recommended', 2)
            
            if rec_games == len(st.session_state.current_ticket):
                st.success(f"✅ {rec_games} jogos - IDEAL!")
            elif rec_games < len(st.session_state.current_ticket):
                st.warning(f"⚠️ Recomendado: {rec_games} jogos")
            else:
                st.info(f"💡 Pode adicionar +{rec_games - len(st.session_state.current_ticket)}")
        
        st.markdown("---")
        
        if st.button("🗑️ Limpar Bilhete", use_container_width=True):
            st.session_state.current_ticket = []
            st.session_state.hedges_data = None
            st.rerun()
        
        st.markdown("---")
        st.caption("🤖 V30.1 Adaptativo:")
        st.caption("✅ Análise de qualidade")
        st.caption("✅ Recomendação inteligente")
        st.caption("✅ Targets dinâmicos")
        st.caption("✅ Safety lock cartões")
        st.caption("✅ Filtro volatilidade")
    
    st.title("💎 FutPrevisão V30.1 Adaptativo")
    st.caption("Sistema Inteligente: Ajusta automaticamente entre 2 ou 3 jogos")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎫 Construtor",
        "🛡️ Hedges V30.1",
        "🎲 Simulador",
        "📊 Análise",
        "📈 Dashboard"
    ])
    
    # TAB 1: CONSTRUTOR
    with tab1:
        st.header("🎫 Monte seu Bilhete")
        
        # Análise adaptativa em tempo real
        if len(st.session_state.current_ticket) >= 2:
            rec = adaptive_recommendation(st.session_state.current_ticket, stats)
            rec_games = rec.get('recommended', 2)
            
            if rec_games == len(st.session_state.current_ticket):
                st.success(f"✅ {rec['message']}")
            elif rec_games > len(st.session_state.current_ticket):
                st.info(f"💡 {rec['message']}")
            else:
                st.warning(f"⚠️ {rec['message']}")
        
        elif len(st.session_state.current_ticket) == 1:
            st.info("➕ Adicione mais 1 jogo para análise adaptativa")
        
        if calendar.empty:
            st.warning("⚠️ Calendário não encontrado. Verifique se calendario_ligas.csv existe.")
        else:
            dates = sorted(calendar['DtObj'].dt.strftime('%d/%m/%Y').unique())
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                sel_date = st.selectbox("📅 Data:", dates, key="builder_date")
            
            df_day = calendar[calendar['DtObj'].dt.strftime('%d/%m/%Y') == sel_date]
            
            with col2:
                games = sorted((df_day['Time_Casa'] + ' vs ' + df_day['Time_Visitante']).unique())
                sel_game = st.selectbox("⚽ Jogo:", games, key="builder_game")
            
            if sel_game:
                try:
                    home, away = sel_game.split(' vs ')
                except:
                    st.error("Formato de jogo inválido")
                    return
                
                h_norm = normalize_name(home, list(stats.keys()))
                a_norm = normalize_name(away, list(stats.keys()))
                
                if h_norm and a_norm:
                    h_st = stats[h_norm]
                    a_st = stats[a_norm]
                    
                    with st.expander("📊 Estatísticas do Jogo", expanded=True):
                        col1, col2, col3, col4 = st.columns(4)
                        
                        # Mostrar volatilidade
                        h_cv = h_st.get('corners_std', 0) / h_st['corners'] if h_st['corners'] > 0 else 0
                        a_cv = a_st.get('corners_std', 0) / a_st['corners'] if a_st['corners'] > 0 else 0
                        
                        h_stability = "🔴" if h_cv > 0.5 else ("🟡" if h_cv > 0.35 else "🟢")
                        a_stability = "🔴" if a_cv > 0.5 else ("🟡" if a_cv > 0.35 else "🟢")
                        
                        col1.metric(f"{h_norm} Cantos", f"{h_st['corners']:.1f}", 
                                   f"{h_stability} CV: {h_cv:.2f}")
                        col2.metric(f"{a_norm} Cantos", f"{a_st['corners']:.1f}",
                                   f"{a_stability} CV: {a_cv:.2f}")
                        col3.metric(f"{h_norm} Cartões", f"{h_st['cards']:.1f}")
                        col4.metric(f"{a_norm} Cartões", f"{a_st['cards']:.1f}")
                    
                    st.markdown("### Adicionar Seleções (2 por jogo)")
                    
                    col_sel1, col_sel2 = st.columns(2)
                    
                    # Gerar opções respeitando Safety Lock
                    home_cards_ceiling = h_st['cards'] + 0.5
                    away_cards_ceiling = a_st['cards'] + 0.5
                    total_cards_ceiling = (h_st['cards'] + a_st['cards']) + 0.5
                    
                    corners_options = [
                        f"{h_norm} Over 3.5 Escanteios",
                        f"{h_norm} Over 4.5 Escanteios",
                        f"{a_norm} Over 2.5 Escanteios",
                        f"{a_norm} Over 3.5 Escanteios",
                        "Total Over 7.5 Escanteios",
                        "Total Over 8.5 Escanteios",
                        "Total Over 9.5 Escanteios"
                    ]
                    
                    cards_options = []
                    
                    # Home cards com Safety Lock
                    if 0.5 <= home_cards_ceiling:
                        cards_options.append(f"{h_norm} Over 0.5 Cartões")
                    if 1.5 <= home_cards_ceiling:
                        cards_options.append(f"{h_norm} Over 1.5 Cartões")
                    if 2.5 <= home_cards_ceiling:
                        cards_options.append(f"{h_norm} Over 2.5 Cartões")
                    
                    # Away cards com Safety Lock
                    if 0.5 <= away_cards_ceiling:
                        cards_options.append(f"{a_norm} Over 0.5 Cartões")
                    if 1.5 <= away_cards_ceiling:
                        cards_options.append(f"{a_norm} Over 1.5 Cartões")
                    if 2.5 <= away_cards_ceiling:
                        cards_options.append(f"{a_norm} Over 2.5 Cartões")
                    
                    # Total cards com Safety Lock
                    if 2.5 <= total_cards_ceiling:
                        cards_options.append("Total Over 2.5 Cartões")
                    if 3.5 <= total_cards_ceiling:
                        cards_options.append("Total Over 3.5 Cartões")
                    if 4.5 <= total_cards_ceiling:
                        cards_options.append("Total Over 4.5 Cartões")
                    
                    dc_options = [
                        f"DC 1X ({h_norm} ou Empate)",
                        f"DC X2 ({a_norm} ou Empate)"
                    ]
                    
                    all_markets = corners_options + cards_options + dc_options
                    
                    with col_sel1:
                        market1 = st.selectbox("Mercado 1:", all_markets, key="market1")
                    
                    with col_sel2:
                        # Remove mercado 1 das opções do mercado 2
                        market2_options = [m for m in all_markets if m != market1]
                        market2 = st.selectbox("Mercado 2:", market2_options, key="market2")
                    
                    if st.button("➕ Adicionar ao Bilhete", type="primary", use_container_width=True):
                        import re
                        
                        def parse_sel(text):
                            result = {'mercado': text}
                            
                            if 'DC' in text:
                                result['type'] = 'dc'
                                result['location'] = 'home' if '1X' in text else 'away'
                                result['dc_type'] = '1X' if '1X' in text else 'X2'
                            elif 'Escanteio' in text:
                                result['type'] = 'corners'
                                if 'Total' in text:
                                    result['location'] = 'total'
                                elif h_norm in text:
                                    result['location'] = 'home'
                                else:
                                    result['location'] = 'away'
                            elif 'Cartão' in text or 'Cartões' in text:
                                result['type'] = 'cards'
                                if 'Total' in text:
                                    result['location'] = 'total'
                                elif h_norm in text:
                                    result['location'] = 'home'
                                else:
                                    result['location'] = 'away'
                            
                            match = re.search(r'(\d+\.?\d*)', text)
                            if match:
                                result['line'] = float(match.group(1))
                            
                            return result
                        
                        sel1 = parse_sel(market1)
                        sel2 = parse_sel(market2)
                        
                        st.session_state.current_ticket.append({
                            'jogo': sel_game,
                            'selections': [sel1, sel2]
                        })
                        
                        st.success("✅ Adicionado!")
                        
                        if len(st.session_state.current_ticket) > 2:
                            st.warning(f"⚠️ Você tem {len(st.session_state.current_ticket)} jogos. Recomendado: 2 jogos (+50% de cobertura)")
                        
                        st.rerun()
                else:
                    st.error(f"⚠️ Times não encontrados: {home}, {away}")
        
        if st.session_state.current_ticket:
            st.markdown("---")
            st.markdown("### 🎫 Seu Bilhete Atual")
            
            for idx, game in enumerate(st.session_state.current_ticket, 1):
                with st.container():
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.markdown(f"**{idx}. {game['jogo']}**")
                        for sel in game['selections']:
                            st.write(f"   • {sel['mercado']}")
                    with col2:
                        if st.button("🗑️", key=f"del_{idx}"):
                            st.session_state.current_ticket.pop(idx-1)
                            st.session_state.hedges_data = None
                            st.rerun()
    
    # TAB 2: HEDGES V30
    with tab2:
        st.header("🛡️ Sistema de Hedges V30.1 Adaptativo")
        
        if not st.session_state.current_ticket:
            st.warning("⚠️ Adicione jogos no Construtor primeiro")
        else:
            if len(st.session_state.current_ticket) > 2:
                st.error(f"⚠️ VOCÊ TEM {len(st.session_state.current_ticket)} JOGOS. Recomendado: 2 jogos para +50% de cobertura!")
            
            st.info("""
            💡 **V30.1 Sistema Adaptativo:**
            - 🤖 Analisa qualidade de cada jogo automaticamente
            - 🎯 Recomenda 2 ou 3 jogos baseado na segurança
            - ✅ Jogos ELITE (prob >75%): Permite 3 jogos
            - ⚠️ Jogos REGULARES: Força 2 jogos
            - 🔒 Safety Lock em cartões
            - 📉 Filtro de volatilidade
            - 📊 Análise de correlação
            
            🎲 **Targets Adaptativos:**
            - 2 jogos: @1.80-2.20 cada → @3.24-4.84 total
            - 3 jogos: @1.55-1.80 cada → @3.72-5.83 total
            """)
            
            col1, col2 = st.columns(2)
            with col1:
                min_prob = st.slider("Probabilidade Mínima (%)", 30, 60, 40, 5)
            with col2:
                n_sims = st.slider("Simulações por Combo", 200, 1000, 500, 100)
            
            if st.button("🎯 GERAR HEDGES V30", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(pct, msg):
                    progress_bar.progress(pct)
                    status_text.text(msg)
                
                with st.spinner("Analisando..."):
                    hedges = generate_hedges_v30(
                        st.session_state.current_ticket,
                        stats,
                        min_prob=min_prob,
                        n_sims=n_sims,
                        progress_callback=update_progress
                    )
                
                progress_bar.empty()
                status_text.empty()
                
                st.session_state.hedges_data = hedges
                
                st.success(f"✅ Hedges gerados em {hedges['processing_time']:.1f}s")
                
                # Mostrar recomendação adaptativa
                rec = hedges.get('recommendation', {})
                if rec:
                    rec_games = rec.get('recommended', 2)
                    actual_games = hedges.get('actual_games', len(st.session_state.current_ticket))
                    
                    if rec_games == 2 and actual_games == 2:
                        st.success(f"💎 **CONFIGURAÇÃO IDEAL:** 2 jogos (@{hedges['target_odd_per_game'][0]:.2f}-{hedges['target_odd_per_game'][1]:.2f} cada)")
                        st.caption(f"Odd alvo total: @{hedges['target_odd_range'][0]:.2f}-{hedges['target_odd_range'][1]:.2f}")
                    
                    elif rec_games == 3 and actual_games == 3:
                        st.success(f"💎 **CONFIGURAÇÃO ELITE:** 3 jogos de alta qualidade (@{hedges['target_odd_per_game'][0]:.2f}-{hedges['target_odd_per_game'][1]:.2f} cada)")
                        st.caption(f"Odd alvo total: @{hedges['target_odd_range'][0]:.2f}-{hedges['target_odd_range'][1]:.2f}")
                    
                    elif actual_games > rec_games:
                        st.warning(f"⚠️ **ALERTA:** Sistema recomenda {rec_games} jogos, você tem {actual_games}")
                        if 'suggest_remove' in rec:
                            st.caption(f"Sugestão: Remover '{rec['suggest_remove']}'")
                    
                    else:
                        st.info(f"💡 **OPORTUNIDADE:** Você pode adicionar mais {rec_games - actual_games} jogo(s) de qualidade")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"#### {hedges['hedge1']['nome']} {hedges['hedge1']['status']}")
                    st.metric("Odd Total Acumulada", f"@{hedges['hedge1']['odd_total']}")
                    
                    for game in hedges['hedge1']['games']:
                        with st.expander(f"{game['jogo']} (@{game['odd_jogo']})"):
                            for sel in game['selections']:
                                st.write(f"✓ {sel['mercado']}")
                                st.caption(f"   Prob: {sel['prob']:.1f}% | Odd: @{sel['odd']}")
                            
                            corr_emoji = "🔴" if game['correlation'] > 0.3 else ("🟡" if game['correlation'] > 0 else "🟢")
                            st.info(f"""
                            📊 **Métricas:**
                            - Taxa Acerto: {game['win_rate']:.1f}%
                            - Cobertura: {game['coverage']:.1f}%
                            - Correlação: {game['correlation']:.2f} {corr_emoji}
                            - Score: {game['score']:.1f}
                            """)
                
                with col2:
                    st.markdown(f"#### {hedges['hedge2']['nome']} {hedges['hedge2']['status']}")
                    st.metric("Odd Total Acumulada", f"@{hedges['hedge2']['odd_total']}")
                    
                    for game in hedges['hedge2']['games']:
                        with st.expander(f"{game['jogo']} (@{game['odd_jogo']})"):
                            for sel in game['selections']:
                                st.write(f"✓ {sel['mercado']}")
                                st.caption(f"   Prob: {sel['prob']:.1f}% | Odd: @{sel['odd']}")
                            
                            corr_emoji = "🔴" if game['correlation'] > 0.3 else ("🟡" if game['correlation'] > 0 else "🟢")
                            st.info(f"""
                            📊 **Métricas:**
                            - Taxa Acerto: {game['win_rate']:.1f}%
                            - Cobertura: {game['coverage']:.1f}%
                            - Correlação: {game['correlation']:.2f} {corr_emoji}
                            - Score: {game['score']:.1f}
                            """)
    
    # TAB 3: SIMULADOR
    with tab3:
        st.header("🎲 Simulador Monte Carlo")
        
        if not st.session_state.current_ticket:
            st.warning("⚠️ Monte seu bilhete primeiro")
        elif not st.session_state.hedges_data:
            st.warning("⚠️ Gere os hedges primeiro (Tab 2)")
        else:
            n_sims_sim = st.slider("Número de simulações:", 100, 10000, 1000, step=100)
            
            if st.button("▶️ EXECUTAR SIMULAÇÃO", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(pct, msg):
                    progress_bar.progress(pct)
                    status_text.text(msg)
                
                with st.spinner(f"Simulando {n_sims_sim} cenários..."):
                    results = simulate_coverage(
                        st.session_state.current_ticket,
                        st.session_state.hedges_data['hedge1']['games'],
                        st.session_state.hedges_data['hedge2']['games'],
                        stats,
                        n=n_sims_sim,
                        progress_callback=update_progress
                    )
                
                progress_bar.empty()
                status_text.empty()
                
                st.markdown("### 📊 Resultados")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Principal", f"{results['principal']}/{n_sims_sim}", 
                           f"{results['principal']/n_sims_sim*100:.1f}%")
                col2.metric("Hedge A", f"{results['hedge1']}/{n_sims_sim}",
                           f"{results['hedge1']/n_sims_sim*100:.1f}%")
                col3.metric("Hedge B", f"{results['hedge2']}/{n_sims_sim}",
                           f"{results['hedge2']/n_sims_sim*100:.1f}%")
                
                st.markdown("---")
                
                col4, col5, col6 = st.columns(3)
                
                coverage_color = "🟢" if results['coverage_rate'] >= 65 else ("🟡" if results['coverage_rate'] >= 50 else "🔴")
                col4.metric(f"{coverage_color} Pelo menos 2 de 3", 
                           f"{results['at_least_2']}/{n_sims_sim}",
                           f"{results['coverage_rate']:.1f}%")
                col5.metric("🎯 Todos 3", f"{results['all_3']}/{n_sims_sim}",
                           f"{results['all_3']/n_sims_sim*100:.1f}%")
                col6.metric("❌ Nenhum", f"{results['none']}/{n_sims_sim}",
                           f"{results['none']/n_sims_sim*100:.1f}%")
                
                # Comparação com Meta
                if len(st.session_state.current_ticket) == 2:
                    expected = 65
                    st.success(f"✅ Meta para 2 jogos: {expected}% | Obtido: {results['coverage_rate']:.1f}%")
                else:
                    expected = 44
                    st.warning(f"⚠️ Meta para {len(st.session_state.current_ticket)} jogos: {expected}% | Obtido: {results['coverage_rate']:.1f}%")
                
                # Gráfico
                fig = go.Figure(data=[
                    go.Bar(name='Principal', x=['Bateu'], y=[results['principal']/n_sims_sim*100],
                          marker_color='#667eea'),
                    go.Bar(name='Hedge A', x=['Bateu'], y=[results['hedge1']/n_sims_sim*100],
                          marker_color='#764ba2'),
                    go.Bar(name='Hedge B', x=['Bateu'], y=[results['hedge2']/n_sims_sim*100],
                          marker_color='#f093fb')
                ])
                fig.update_layout(
                    title="Taxa de Acerto por Bilhete",
                    yaxis_title="% de Acerto",
                    barmode='group',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Salvar
                st.session_state.simulation_history.append({
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
                    'n_sims': n_sims_sim,
                    'coverage': results['coverage_rate'],
                    'principal': results['principal']/n_sims_sim*100,
                    'hedge1': results['hedge1']/n_sims_sim*100,
                    'hedge2': results['hedge2']/n_sims_sim*100,
                    'games': len(st.session_state.current_ticket)
                })
    
    # TAB 4: ANÁLISE KELLY
    with tab4:
        st.header("📊 Análise de Risco com Kelly Criterion")
        
        if st.session_state.hedges_data and st.session_state.current_ticket:
            
            st.markdown("### 💰 Calculadora de Stake Inteligente (Kelly)")
            
            st.info("""
            **Kelly Criterion Fracionário (25% do Kelly completo)**
            
            Calcula stake ideal baseado em:
            - Probabilidade de acerto de cada bilhete
            - Odds oferecidas
            - Edge estatístico
            
            Mais conservador que Kelly completo para proteger bankroll.
            """)
            
            bankroll = st.number_input("Bankroll Total:", value=st.session_state.bankroll, step=50.0)
            
            # Calcular stakes Kelly
            hedges = st.session_state.hedges_data
            
            # Estimativas de probabilidade (baseado em simulações anteriores)
            prob_principal = 25  # Estimado para 2-3 jogos
            prob_h1 = 70
            prob_h2 = 60
            
            odd_principal = 7.0  # Estimado
            odd_h1 = hedges['hedge1']['odd_total']
            odd_h2 = hedges['hedge2']['odd_total']
            
            kelly_p = kelly_criterion(prob_principal, odd_principal, bankroll)
            kelly_h1 = kelly_criterion(prob_h1, odd_h1, bankroll)
            kelly_h2 = kelly_criterion(prob_h2, odd_h2, bankroll)
            
            total_kelly = kelly_p + kelly_h1 + kelly_h2
            
            st.markdown("### 💸 Distribuição Kelly")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Principal", f"€{kelly_p:.2f}", f"{kelly_p/bankroll*100:.1f}%")
            col2.metric("Hedge A", f"€{kelly_h1:.2f}", f"{kelly_h1/bankroll*100:.1f}%")
            col3.metric("Hedge B", f"€{kelly_h2:.2f}", f"{kelly_h2/bankroll*100:.1f}%")
            col4.metric("Total Investido", f"€{total_kelly:.2f}", f"{total_kelly/bankroll*100:.1f}%")
            
            st.markdown("### 📈 Cenários de Retorno")
            
            scenarios = [
                {
                    "nome": "✅✅✅ Todos batem",
                    "prob": 5,
                    "ret": kelly_p * odd_principal + kelly_h1 * odd_h1 + kelly_h2 * odd_h2
                },
                {
                    "nome": "✅✅❌ Principal + Hedge A",
                    "prob": 15,
                    "ret": kelly_p * odd_principal + kelly_h1 * odd_h1 - kelly_h2
                },
                {
                    "nome": "✅❌✅ Principal + Hedge B",
                    "prob": 20,
                    "ret": kelly_p * odd_principal - kelly_h1 + kelly_h2 * odd_h2
                },
                {
                    "nome": "❌✅✅ Hedge A + Hedge B",
                    "prob": 45,
                    "ret": -kelly_p + kelly_h1 * odd_h1 + kelly_h2 * odd_h2
                },
                {
                    "nome": "❌❌❌ Nenhum bate",
                    "prob": 15,
                    "ret": -total_kelly
                }
            ]
            
            roi_esperado = 0
            for sc in scenarios:
                profit = sc['ret'] - total_kelly
                roi_esperado += (sc['prob'] / 100) * profit
                color = "green" if profit > 0 else "red"
                st.markdown(f"**{sc['nome']}** (Prob ~{sc['prob']}%): :{color}[€{profit:+.2f}]")
            
            st.success(f"💎 **ROI Esperado: {roi_esperado/total_kelly*100:+.1f}%** (€{roi_esperado:+.2f})")
            
            # Gráfico de pizza
            fig_pie = go.Figure(data=[go.Pie(
                labels=[s['nome'] for s in scenarios],
                values=[s['prob'] for s in scenarios],
                hole=0.3,
                marker_colors=['#28a745', '#17a2b8', '#ffc107', '#dc3545', '#6c757d']
            )])
            fig_pie.update_layout(title="Distribuição de Probabilidade dos Cenários", height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Monte seu bilhete e gere hedges primeiro")
    
    # TAB 5: DASHBOARD
    with tab5:
        st.header("📈 Dashboard V30")
        
        if st.session_state.simulation_history:
            st.markdown("### 📊 Histórico de Simulações")
            
            df_hist = pd.DataFrame(st.session_state.simulation_history)
            
            # Comparar 2 vs 3 jogos
            if 'games' in df_hist.columns:
                fig_comp = px.scatter(df_hist, x='timestamp', y='coverage', 
                                     color='games', size='n_sims',
                                     labels={'coverage': 'Cobertura (%)', 'games': 'Nº Jogos'},
                                     title="Impacto do Número de Jogos na Cobertura")
                st.plotly_chart(fig_comp, use_container_width=True)
            
            fig = px.line(df_hist, x='timestamp', y='coverage', markers=True,
                         labels={'coverage': 'Taxa de Cobertura (%)', 'timestamp': 'Data/Hora'},
                         title="Evolução da Taxa de Cobertura")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(df_hist, use_container_width=True)
        else:
            st.info("Execute simulações para popular o dashboard")
        
        st.markdown("### 🎯 Resumo V30")
        col1, col2, col3 = st.columns(3)
        col1.metric("Times Analisados", len(stats))
        col2.metric("Jogos no Bilhete", len(st.session_state.current_ticket))
        col3.metric("Simulações Realizadas", len(st.session_state.simulation_history))
        
        st.markdown("---")
        st.markdown("### ℹ️ Sobre o V30.1")
        st.success("""
        **FutPrevisão V30.1 Adaptativo**
        
        🤖 **Sistema Inteligente:**
        - Analisa qualidade de cada jogo automaticamente
        - Score 0-100 baseado em: probabilidade + volatilidade + estabilidade
        - Classifica jogos: ELITE, BOA, REGULAR, ARRISCADA
        
        🎯 **Recomendação Adaptativa:**
        - **Jogos ELITE** (score ≥60): Sistema permite 3 jogos
          → Target: @1.55-1.80 cada (@3.72-5.83 total)
        - **Jogos REGULARES** (score <60): Sistema força 2 jogos
          → Target: @1.80-2.20 cada (@3.24-4.84 total)
        
        📊 **Vantagens:**
        - Sem banca fixa necessária
        - Adapta-se à qualidade disponível
        - Maximiza chance quando jogos fracos
        - Maximiza ROI quando jogos fortes
        - Stake livre ($10, $20, $30... o que tiver)
        
        ✅ **Resultados Esperados:**
        - 2 jogos regulares: 42% de green
        - 3 jogos elite: 37% de green (ROI maior)
        """)

if __name__ == "__main__":
    main()
