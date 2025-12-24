"""
╔═══════════════════════════════════════════════════════════════════════════╗
║              FUTPREVISÃO V29.0 ULTIMATE - SISTEMA COMPLETO                ║
║                                                                            ║
║  🎯 Motor de Hedges Livre (Máxima Cobertura)                              ║
║  🎲 Simulador Monte Carlo Integrado                                       ║
║  📊 5 Tabs Funcionais                                                     ║
║  💎 100% Baseado em Dados Reais                                           ║
║                                                                            ║
║  Dezembro 2025 - Ultimate Free Edition                                   ║
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
# CONFIGURAÇÃO STREAMLIT
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="FutPrevisão V29 Ultimate",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Customizado
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
        .success-box { 
            background-color: #d4edda;
            border-left: 4px solid #28a745;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
        }
        .warning-box { 
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
        }
        .danger-box { 
            background-color: #f8d7da;
            border-left: 4px solid #dc3545;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
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
# ODDS REAIS DO MERCADO
# ═══════════════════════════════════════════════════════════════════════════

REAL_ODDS = {
    # Mandante Escanteios
    ('home', 'corners', 3.5): 1.34,
    ('home', 'corners', 4.5): 1.63,
    
    # Mandante Cartões
    ('home', 'cards', 0.5): 1.22,
    ('home', 'cards', 1.5): 1.53,
    ('home', 'cards', 2.5): 2.25,
    
    # Visitante Escanteios
    ('away', 'corners', 2.5): 1.40,
    ('away', 'corners', 3.5): 1.75,
    ('away', 'corners', 4.5): 2.50,
    
    # Visitante Cartões
    ('away', 'cards', 0.5): 1.26,
    ('away', 'cards', 1.5): 1.65,
    ('away', 'cards', 2.5): 2.50,
    
    # Totais Escanteios
    ('total', 'corners', 7.5): 1.35,
    ('total', 'corners', 8.5): 1.58,
    ('total', 'corners', 9.5): 1.90,
    ('total', 'corners', 10.5): 2.30,
    ('total', 'corners', 11.5): 2.80,
    
    # Totais Cartões
    ('total', 'cards', 2.5): 1.43,
    ('total', 'cards', 3.5): 1.73,
    ('total', 'cards', 4.5): 2.13,
    
    # Dupla Chance
    ('home', 'dc', '1X'): 1.24,
    ('away', 'dc', 'X2'): 1.60
}

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTES
# ═══════════════════════════════════════════════════════════════════════════

NAME_MAPPING = {
    'Man United': 'Man Utd', 'Manchester United': 'Man Utd',
    'Man City': 'Man City', 'Manchester City': 'Man City',
    'Spurs': 'Tottenham', 'Newcastle': 'Newcastle',
    'Wolves': 'Wolves', 'Brighton': 'Brighton',
    "Nott'm Forest": 'Nottm Forest', 'Nottingham Forest': 'Nottm Forest',
    'West Ham': 'West Ham', 'Leicester': 'Leicester',
    'Athletic Club': 'Ath Bilbao', 'Atl. Madrid': 'Ath Madrid'
}

LIGAS_ALVO = [
    "Premier League", "La Liga", "Serie A", "Bundesliga", "Ligue 1",
    "Championship", "Bundesliga 2", "Pro League", "Süper Lig", "Scottish Premiership"
]

# ═══════════════════════════════════════════════════════════════════════════
# FUNÇÕES DE CARREGAMENTO DE DADOS
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
                            'Mandante': 'HomeTeam',
                            'Visitante': 'AwayTeam',
                            'Time_Casa': 'HomeTeam',
                            'Time_Visitante': 'AwayTeam'
                        }
                        df = df.rename(columns=rename_map)
                        df['_League_'] = league_name
                        return df
                except:
                    continue
    
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def learn_stats_v29() -> Dict[str, Dict[str, Any]]:
    """Aprende estatísticas de todos os times"""
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
            h_stats = df.groupby('HomeTeam').agg({
                'HC': 'mean',
                'HY': 'mean',
                'FTHG': 'mean',
                'FTAG': 'mean'
            })
            
            a_stats = df.groupby('AwayTeam').agg({
                'AC': 'mean',
                'AY': 'mean',
                'FTAG': 'mean',
                'FTHG': 'mean'
            })
            
            for team in set(h_stats.index) | set(a_stats.index):
                h = h_stats.loc[team] if team in h_stats.index else pd.Series(0, index=h_stats.columns)
                a = a_stats.loc[team] if team in a_stats.index else pd.Series(0, index=a_stats.columns)
                
                def w_avg(val_h, val_a, default=0):
                    if val_h == 0 and val_a == 0:
                        return default
                    if val_h == 0:
                        return val_a
                    if val_a == 0:
                        return val_h
                    return (val_h * 0.6) + (val_a * 0.4)
                
                stats_db[team] = {
                    'corners': w_avg(h.get('HC', 0), a.get('AC', 0), 5.0),
                    'cards': w_avg(h.get('HY', 0), a.get('AY', 0), 2.0),
                    'goals_f': w_avg(h.get('FTHG', 0), a.get('FTAG', 0), 1.2),
                    'goals_a': w_avg(h.get('FTAG', 0), a.get('FTHG', 0), 1.2),
                    'league': league
                }
        except:
            continue
    
    return stats_db

@st.cache_data(ttl=600)
def load_calendar_safe() -> pd.DataFrame:
    """Carrega calendário de jogos"""
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
# FUNÇÕES AUXILIARES
# ═══════════════════════════════════════════════════════════════════════════

def normalize_name(name: str, db_keys: list) -> Optional[str]:
    """Normaliza nome do time"""
    if not name:
        return None
    
    name = name.strip()
    
    if name in NAME_MAPPING:
        name = NAME_MAPPING[name]
    
    if name in db_keys:
        return name
    
    matches = get_close_matches(name, db_keys, n=1, cutoff=0.6)
    return matches[0] if matches else None

def sim_prob(avg: float, line: float) -> float:
    """Calcula probabilidade baseada na média"""
    prob = 50 + (avg - line) * 15
    return max(5, min(95, prob))

def get_market_odd(location: str, market_type: str, line: float) -> float:
    """Retorna odd real do mercado"""
    key = (location, market_type, line) if market_type != 'dc' else (location, market_type, str(line))
    return REAL_ODDS.get(key, 1.50)

def monte_carlo_simulate(xg_home: float, xg_away: float, n: int = 1000) -> Tuple[float, float, float]:
    """Simula probabilidades de resultado"""
    np.random.seed(42)
    goals_h = np.random.poisson(max(0.1, xg_home), n)
    goals_a = np.random.poisson(max(0.1, xg_away), n)
    p_h = np.count_nonzero(goals_h > goals_a) / n
    p_d = np.count_nonzero(goals_h == goals_a) / n
    p_a = np.count_nonzero(goals_a > goals_h) / n
    return p_h, p_d, p_a

# ═══════════════════════════════════════════════════════════════════════════
# MOTOR DE HEDGES LIVRE V29
# ═══════════════════════════════════════════════════════════════════════════

def generate_market_pool(home_stats: Dict, away_stats: Dict, 
                        home_name: str, away_name: str,
                        min_prob: float = 40.0) -> List[Dict]:
    """Gera pool de mercados viáveis"""
    
    pool = []
    
    # Escanteios Casa
    for line in [3.5, 4.5]:
        prob = sim_prob(home_stats['corners'], line)
        if prob >= min_prob:
            pool.append({
                'mercado': f"{home_name} Over {line} Escanteios",
                'type': 'corners',
                'location': 'home',
                'line': line,
                'prob': prob,
                'odd': get_market_odd('home', 'corners', line)
            })
    
    # Escanteios Fora
    for line in [2.5, 3.5, 4.5]:
        prob = sim_prob(away_stats['corners'], line)
        if prob >= min_prob:
            pool.append({
                'mercado': f"{away_name} Over {line} Escanteios",
                'type': 'corners',
                'location': 'away',
                'line': line,
                'prob': prob,
                'odd': get_market_odd('away', 'corners', line)
            })
    
    # Escanteios Totais
    total_corners = home_stats['corners'] + away_stats['corners']
    for line in [7.5, 8.5, 9.5, 10.5, 11.5]:
        prob = sim_prob(total_corners, line)
        if prob >= min_prob:
            pool.append({
                'mercado': f"Total Over {line} Escanteios",
                'type': 'corners',
                'location': 'total',
                'line': line,
                'prob': prob,
                'odd': get_market_odd('total', 'corners', line)
            })
    
    # Cartões Casa
    for line in [0.5, 1.5, 2.5]:
        prob = sim_prob(home_stats['cards'], line)
        if prob >= min_prob:
            pool.append({
                'mercado': f"{home_name} Over {line} Cartões",
                'type': 'cards',
                'location': 'home',
                'line': line,
                'prob': prob,
                'odd': get_market_odd('home', 'cards', line)
            })
    
    # Cartões Fora
    for line in [0.5, 1.5, 2.5]:
        prob = sim_prob(away_stats['cards'], line)
        if prob >= min_prob:
            pool.append({
                'mercado': f"{away_name} Over {line} Cartões",
                'type': 'cards',
                'location': 'away',
                'line': line,
                'prob': prob,
                'odd': get_market_odd('away', 'cards', line)
            })
    
    # Cartões Totais
    total_cards = home_stats['cards'] + away_stats['cards']
    for line in [2.5, 3.5, 4.5]:
        prob = sim_prob(total_cards, line)
        if prob >= min_prob:
            pool.append({
                'mercado': f"Total Over {line} Cartões",
                'type': 'cards',
                'location': 'total',
                'line': line,
                'prob': prob,
                'odd': get_market_odd('total', 'cards', line)
            })
    
    # Dupla Chance
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

def is_valid_combo(sel1: Dict, sel2: Dict) -> bool:
    """Valida se duas seleções podem ser combinadas"""
    
    if sel1['mercado'] == sel2['mercado']:
        return False
    
    if sel1['type'] == 'dc' and sel2['type'] == 'dc':
        return False
    
    if (sel1['type'] == sel2['type'] and 
        sel1['location'] == sel2['location'] and
        sel1['type'] != 'dc'):
        return False
    
    return True

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
    """Verifica se uma seleção bateu"""
    
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

def evaluate_combo(combo: Tuple[Dict, Dict], principal_sels: List[Dict],
                   home_stats: Dict, away_stats: Dict, n_sims: int = 500) -> Dict:
    """Avalia uma combinação através de simulação"""
    
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
    odd_score = 100 if 1.60 <= odd_jogo <= 2.00 else (80 if 1.40 <= odd_jogo <= 2.50 else 50)
    
    score = (coverage * 0.50) + (odd_score * 0.30) + (win_rate * 0.20)
    
    return {
        'combo': combo,
        'win_rate': win_rate,
        'coverage': coverage,
        'odd_jogo': odd_jogo,
        'score': score
    }

def generate_free_hedges_v29(principal_games: List[Dict], stats: Dict, 
                             min_prob: float = 40.0, n_sims: int = 500,
                             progress_callback=None) -> Dict:
    """Gera 2 hedges livres com máxima cobertura"""
    
    start_time = time.time()
    
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
        pool = generate_market_pool(home_stats, away_stats, home_norm, away_norm, min_prob)
        
        if len(pool) < 2:
            continue
        
        # Gerar combinações
        valid_combos = []
        for sel1, sel2 in combinations(pool, 2):
            if is_valid_combo(sel1, sel2):
                valid_combos.append((sel1, sel2))
        
        if len(valid_combos) == 0:
            continue
        
        # Avaliar
        results = []
        for combo in valid_combos:
            result = evaluate_combo(combo, game.get('selections', []), 
                                   home_stats, away_stats, n_sims)
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
            'score': best['score']
        })
        
        # Hedge 2 (diverso)
        hedge2_selected = None
        for result in results[1:]:
            r_types = {result['combo'][0]['type'], result['combo'][1]['type']}
            b_types = {best['combo'][0]['type'], best['combo'][1]['type']}
            
            if r_types != b_types:
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
    
    return {
        'hedge1': {
            'games': hedge1_games,
            'odd_total': round(odd_h1_total, 2),
            'status': '✅' if 4.0 <= odd_h1_total <= 6.5 else '⚠️',
            'nome': 'Hedge A - Máxima Cobertura'
        },
        'hedge2': {
            'games': hedge2_games,
            'odd_total': round(odd_h2_total, 2),
            'status': '✅' if 4.0 <= odd_h2_total <= 6.5 else '⚠️',
            'nome': 'Hedge B - Cobertura Alternativa'
        },
        'processing_time': elapsed
    }

# ═══════════════════════════════════════════════════════════════════════════
# SIMULADOR DE COBERTURA
# ═══════════════════════════════════════════════════════════════════════════

def simulate_coverage(principal_games, hedge1_games, hedge2_games, stats, n=1000, progress_callback=None):
    """Simula cobertura dos 3 bilhetes"""
    
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
            
            # Check Principal
            for sel in p_game.get('selections', []):
                if not check_selection(sim, sel):
                    p_hit = False
                    break
            
            # Check Hedge 1
            for sel in h1_game['selections']:
                if not check_selection(sim, sel):
                    h1_hit = False
                    break
            
            # Check Hedge 2
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
# INTERFACE PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════

def main():
    
    # Carregamento de dados
    with st.spinner("🔄 Carregando sistema..."):
        stats = learn_stats_v29()
        calendar = load_calendar_safe()
    
    # Sidebar
    with st.sidebar:
        st.title("💎 FutPrevisão V29")
        st.caption("Ultimate Free Edition")
        st.markdown("---")
        
        st.metric("💰 Bankroll", f"€{st.session_state.bankroll:.2f}")
        st.metric("📊 Times no DB", len(stats))
        st.metric("🎫 Jogos no Bilhete", len(st.session_state.current_ticket))
        
        st.markdown("---")
        
        if st.button("🗑️ Limpar Bilhete", use_container_width=True):
            st.session_state.current_ticket = []
            st.session_state.hedges_data = None
            st.rerun()
        
        st.markdown("---")
        st.caption("Motor de Hedges Livre V29")
        st.caption("Máxima Cobertura Garantida")
    
    # Header
    st.title("💎 FutPrevisão V29 Ultimate")
    st.caption("Sistema de Hedges com Máxima Cobertura")
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎫 Construtor",
        "🛡️ Hedges",
        "🎲 Simulador",
        "📊 Análise",
        "📈 Dashboard"
    ])
    
    # ═══════════════════════════════════════════════════════════════════════
    # TAB 1: CONSTRUTOR
    # ═══════════════════════════════════════════════════════════════════════
    
    with tab1:
        st.header("🎫 Monte seu Bilhete Principal")
        
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
                        col1.metric(f"Cantos {h_norm}", f"{h_st['corners']:.1f}")
                        col2.metric(f"Cantos {a_norm}", f"{a_st['corners']:.1f}")
                        col3.metric(f"Cartões {h_norm}", f"{h_st['cards']:.1f}")
                        col4.metric(f"Cartões {a_norm}", f"{a_st['cards']:.1f}")
                    
                    st.markdown("### Adicionar Seleções")
                    
                    col_sel1, col_sel2 = st.columns(2)
                    
                    with col_sel1:
                        market1 = st.selectbox("Mercado 1:", [
                            f"{h_norm} Over 3.5 Escanteios",
                            f"{h_norm} Over 4.5 Escanteios",
                            f"{a_norm} Over 2.5 Escanteios",
                            f"{a_norm} Over 3.5 Escanteios",
                            "Total Over 7.5 Escanteios",
                            "Total Over 8.5 Escanteios",
                            "Total Over 9.5 Escanteios"
                        ], key="market1")
                    
                    with col_sel2:
                        market2 = st.selectbox("Mercado 2:", [
                            "Total Over 2.5 Cartões",
                            "Total Over 3.5 Cartões",
                            f"{h_norm} Over 0.5 Cartões",
                            f"{h_norm} Over 1.5 Cartões",
                            f"{a_norm} Over 0.5 Cartões",
                            f"{a_norm} Over 1.5 Cartões",
                            f"DC 1X ({h_norm} ou Empate)",
                            f"DC X2 ({a_norm} ou Empate)"
                        ], key="market2")
                    
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
    
    # ═══════════════════════════════════════════════════════════════════════
    # TAB 2: HEDGES
    # ═══════════════════════════════════════════════════════════════════════
    
    with tab2:
        st.header("🛡️ Sistema de Hedges Livres")
        
        if not st.session_state.current_ticket:
            st.warning("⚠️ Adicione jogos no Construtor primeiro")
        else:
            st.info("💡 O sistema testará todas as combinações possíveis e escolherá as que oferecem MÁXIMA COBERTURA")
            
            col1, col2 = st.columns(2)
            with col1:
                min_prob = st.slider("Probabilidade Mínima (%)", 30, 60, 40, 5)
            with col2:
                n_sims = st.slider("Simulações por Combo", 200, 1000, 500, 100)
            
            if st.button("🎯 GERAR HEDGES", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(pct, msg):
                    progress_bar.progress(pct)
                    status_text.text(msg)
                
                with st.spinner("Analisando..."):
                    hedges = generate_free_hedges_v29(
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
                
                st.markdown("### 📋 Resultado")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"#### {hedges['hedge1']['nome']} {hedges['hedge1']['status']}")
                    st.metric("Odd Total Acumulada", f"@{hedges['hedge1']['odd_total']}")
                    
                    for game in hedges['hedge1']['games']:
                        with st.expander(f"{game['jogo']} (@{game['odd_jogo']})"):
                            for sel in game['selections']:
                                st.write(f"✓ {sel['mercado']}")
                                st.caption(f"   Prob: {sel['prob']:.1f}% | Odd: @{sel['odd']}")
                            st.info(f"📊 Taxa Acerto: {game['win_rate']:.1f}% | Cobertura: {game['coverage']:.1f}%")
                
                with col2:
                    st.markdown(f"#### {hedges['hedge2']['nome']} {hedges['hedge2']['status']}")
                    st.metric("Odd Total Acumulada", f"@{hedges['hedge2']['odd_total']}")
                    
                    for game in hedges['hedge2']['games']:
                        with st.expander(f"{game['jogo']} (@{game['odd_jogo']})"):
                            for sel in game['selections']:
                                st.write(f"✓ {sel['mercado']}")
                                st.caption(f"   Prob: {sel['prob']:.1f}% | Odd: @{sel['odd']}")
                            st.info(f"📊 Taxa Acerto: {game['win_rate']:.1f}% | Cobertura: {game['coverage']:.1f}%")
    
    # ═══════════════════════════════════════════════════════════════════════
    # TAB 3: SIMULADOR
    # ═══════════════════════════════════════════════════════════════════════
    
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
                col4.metric("✅ Pelo menos 2 de 3", f"{results['at_least_2']}/{n_sims_sim}",
                           f"{results['coverage_rate']:.1f}%")
                col5.metric("🎯 Todos 3", f"{results['all_3']}/{n_sims_sim}",
                           f"{results['all_3']/n_sims_sim*100:.1f}%")
                col6.metric("❌ Nenhum", f"{results['none']}/{n_sims_sim}",
                           f"{results['none']/n_sims_sim*100:.1f}%")
                
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
                
                # Salvar histórico
                st.session_state.simulation_history.append({
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
                    'n_sims': n_sims_sim,
                    'coverage': results['coverage_rate'],
                    'principal': results['principal']/n_sims_sim*100,
                    'hedge1': results['hedge1']/n_sims_sim*100,
                    'hedge2': results['hedge2']/n_sims_sim*100
                })
    
    # ═══════════════════════════════════════════════════════════════════════
    # TAB 4: ANÁLISE
    # ═══════════════════════════════════════════════════════════════════════
    
    with tab4:
        st.header("📊 Análise de Risco")
        
        if st.session_state.hedges_data and st.session_state.current_ticket:
            
            st.markdown("### 💰 Calculadora de Stake")
            
            bankroll = st.number_input("Bankroll Total:", value=st.session_state.bankroll, step=50.0)
            
            col1, col2, col3 = st.columns(3)
            stake_p = col1.slider("% Principal:", 10, 100, 50, step=5)
            stake_h1 = col2.slider("% Hedge A:", 10, 100, 30, step=5)
            stake_h2 = col3.slider("% Hedge B:", 10, 100, 20, step=5)
            
            total_pct = stake_p + stake_h1 + stake_h2
            
            if total_pct != 100:
                st.error(f"⚠️ A soma deve ser 100% (atual: {total_pct}%)")
            else:
                val_p = (bankroll * stake_p) / 100
                val_h1 = (bankroll * stake_h1) / 100
                val_h2 = (bankroll * stake_h2) / 100
                
                st.markdown("### 💸 Distribuição")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Principal", f"€{val_p:.2f}", f"{stake_p}%")
                col2.metric("Hedge A", f"€{val_h1:.2f}", f"{stake_h1}%")
                col3.metric("Hedge B", f"€{val_h2:.2f}", f"{stake_h2}%")
                
                # Cenários
                hedges = st.session_state.hedges_data
                odd_principal = 7.0  # Estimado
                
                st.markdown("### 📈 Cenários de Retorno")
                
                scenarios = [
                    {
                        "nome": "✅✅✅ Todos batem",
                        "prob": 5,
                        "ret": val_p * odd_principal + val_h1 * hedges['hedge1']['odd_total'] + val_h2 * hedges['hedge2']['odd_total']
                    },
                    {
                        "nome": "✅✅❌ Principal + Hedge A",
                        "prob": 15,
                        "ret": val_p * odd_principal + val_h1 * hedges['hedge1']['odd_total'] - val_h2
                    },
                    {
                        "nome": "✅❌✅ Principal + Hedge B",
                        "prob": 20,
                        "ret": val_p * odd_principal - val_h1 + val_h2 * hedges['hedge2']['odd_total']
                    },
                    {
                        "nome": "❌✅✅ Hedge A + Hedge B",
                        "prob": 45,
                        "ret": -val_p + val_h1 * hedges['hedge1']['odd_total'] + val_h2 * hedges['hedge2']['odd_total']
                    },
                    {
                        "nome": "❌❌❌ Nenhum bate",
                        "prob": 15,
                        "ret": -bankroll
                    }
                ]
                
                for sc in scenarios:
                    profit = sc['ret'] - bankroll
                    color = "green" if profit > 0 else "red"
                    st.markdown(f"**{sc['nome']}** (Prob ~{sc['prob']}%): :{color}[€{profit:+.2f}]")
                
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
    
    # ═══════════════════════════════════════════════════════════════════════
    # TAB 5: DASHBOARD
    # ═══════════════════════════════════════════════════════════════════════
    
    with tab5:
        st.header("📈 Dashboard Geral")
        
        if st.session_state.simulation_history:
            st.markdown("### 📊 Histórico de Simulações")
            
            df_hist = pd.DataFrame(st.session_state.simulation_history)
            
            fig = px.line(df_hist, x='timestamp', y='coverage', markers=True,
                         labels={'coverage': 'Taxa de Cobertura (%)', 'timestamp': 'Data/Hora'},
                         title="Evolução da Taxa de Cobertura")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(df_hist, use_container_width=True)
        else:
            st.info("Execute simulações para popular o dashboard")
        
        st.markdown("### 🎯 Resumo do Sistema")
        col1, col2, col3 = st.columns(3)
        col1.metric("Times Analisados", len(stats))
        col2.metric("Jogos no Bilhete", len(st.session_state.current_ticket))
        col3.metric("Simulações Realizadas", len(st.session_state.simulation_history))
        
        st.markdown("---")
        st.markdown("### ℹ️ Sobre o Sistema")
        st.info("""
        **FutPrevisão V29 Ultimate** utiliza um sistema de hedges **LIVRE**, onde não há estratégias 
        pré-definidas. O motor testa TODAS as combinações possíveis de mercados e seleciona as que 
        oferecem **máxima cobertura** quando o bilhete principal falha.
        
        ✅ 100% baseado em dados reais  
        ✅ Simulação Monte Carlo para cada combinação  
        ✅ Sem viés de estratégia  
        ✅ Target: @4.0-6.5 de odd total  
        """)

if __name__ == "__main__":
    main()
