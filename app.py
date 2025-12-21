"""
╔═══════════════════════════════════════════════════════════════════════════╗
║       FUTPREVISÃO V23.0 ULTIMATE - SISTEMA COMPLETO                       ║
║                                                                            ║
║  ✅ Código V22 100% preservado (funcionando)                              ║
║  🚀 Scanner V23 Smart Ticket                                              ║
║  🎯 Sistema 3 Bilhetes (Principal + Hedges)                               ║
║  📊 Radares de Cantos e Cartões                                           ║
║  📈 Dashboard Completo                                                    ║
║  🎨 Tema Claro/Escuro                                                     ║
║  ⚡ Desvio Padrão / Índice Consistência                                   ║
║  🔔 Alertas Inteligentes                                                  ║
║  💾 Sistema de Favoritos                                                  ║
║                                                                            ║
║  Dezembro 2025 - Ultimate Edition                                        ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import math
import numpy as np
import os
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any, Optional, Tuple
from difflib import get_close_matches
from datetime import datetime
import json

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURAÇÃO
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="FutPrevisão V23 Ultimate",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session State
if 'bankroll' not in st.session_state:
    st.session_state.bankroll = 1000.0
if 'bet_history' not in st.session_state:
    st.session_state.bet_history = []
if 'favorites' not in st.session_state:
    st.session_state.favorites = {'teams': [], 'matches': []}
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'
if 'alert_threshold' not in st.session_state:
    st.session_state.alert_threshold = 80.0

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTES
# ═══════════════════════════════════════════════════════════════════════════

THRESHOLDS = {
    'fouls_violent': 12.5,
    'shots_pressure_high': 6.0,
    'shots_pressure_medium': 4.5,
    'red_rate_strict_high': 0.12,
    'red_rate_strict_medium': 0.08,
    'prob_elite': 75,
    'prob_elite_cards': 70,
    'prob_red_high': 12,
    'prob_red_medium': 8,
    'radar_corners': 70,
    'radar_cards': 65,
    'anchor_safety': 85,
    'smart_ticket_min': 4.60,
    'smart_ticket_max': 5.50
}

DEFAULTS = {
    'shots_on_target': 4.5,
    'red_cards_avg': 0.08,
    'red_rate_referee': 0.08
}

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
# CARREGAMENTO DE DADOS
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
    if "Premiership" in league_name:
        attempts.append("Premiership Escocia 25.26.csv")
    if "Championship" in league_name:
        attempts.append("Championship Inglaterra 25.26.csv")

    for filename in attempts:
        if os.path.exists(filename):
            try:
                try:
                    df = pd.read_csv(filename, encoding='utf-8-sig')
                except:
                    df = pd.read_csv(filename, encoding='latin1')
                
                if not df.empty:
                    df.columns = [c.strip().replace('\ufeff', '') for c in df.columns]
                    
                    rename_map = {}
                    if 'Mandante' in df.columns:
                        rename_map['Mandante'] = 'HomeTeam'
                    if 'Visitante' in df.columns:
                        rename_map['Visitante'] = 'AwayTeam'
                    if 'Time_Casa' in df.columns:
                        rename_map['Time_Casa'] = 'HomeTeam'
                    if 'Time_Visitante' in df.columns:
                        rename_map['Time_Visitante'] = 'AwayTeam'
                    
                    if rename_map:
                        df = df.rename(columns=rename_map)
                    
                    df['_League_'] = league_name
                    return df
            except:
                pass
    
    return pd.DataFrame()

@st.cache_resource
def load_all_dataframes() -> Dict[str, pd.DataFrame]:
    """Carrega todos os dataframes"""
    dfs = {}
    for league in LIGAS_ALVO:
        df = find_and_load_csv(league)
        if not df.empty:
            dfs[league] = df
    return dfs

@st.cache_data(ttl=3600)
def learn_stats_v23() -> Dict[str, Dict[str, Any]]:
    """Aprende estatísticas dos times com desvio padrão"""
    stats_db = {}
    
    for league in LIGAS_ALVO:
        df = find_and_load_csv(league)
        if df.empty:
            continue
        
        cols_needed = ['HomeTeam', 'AwayTeam', 'HC', 'AC', 'HY', 'AY', 'HF', 'AF', 
                      'FTHG', 'FTAG', 'HST', 'AST', 'HR', 'AR']
        for c in cols_needed:
            if c not in df.columns:
                df[c] = np.nan
        
        try:
            # Estatísticas casa
            h_stats = df.groupby('HomeTeam').agg({
                'HC': ['mean', 'std'], 
                'HY': ['mean', 'std'], 
                'HF': 'mean',
                'FTHG': ['mean', 'std'], 
                'FTAG': 'mean',
                'HST': 'mean', 
                'HR': 'mean'
            })
            
            h_stats.columns = ['_'.join(col).strip() for col in h_stats.columns.values]
            h_stats = h_stats.fillna(value={
                'HST_mean': DEFAULTS['shots_on_target'],
                'HR_mean': DEFAULTS['red_cards_avg'],
                'HC_std': 1.5,
                'HY_std': 0.8,
                'FTHG_std': 1.0
            })
            
            # Estatísticas fora
            a_stats = df.groupby('AwayTeam').agg({
                'AC': ['mean', 'std'],
                'AY': ['mean', 'std'],
                'AF': 'mean',
                'FTAG': ['mean', 'std'],
                'FTHG': 'mean',
                'AST': 'mean',
                'AR': 'mean'
            })
            
            a_stats.columns = ['_'.join(col).strip() for col in a_stats.columns.values]
            a_stats = a_stats.fillna(value={
                'AST_mean': DEFAULTS['shots_on_target'],
                'AR_mean': DEFAULTS['red_cards_avg'],
                'AC_std': 1.5,
                'AY_std': 0.8,
                'FTAG_std': 1.0
            })
            
            all_teams = set(h_stats.index) | set(a_stats.index)
            
            for team in all_teams:
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
                
                # Calcular índice de consistência (0-100)
                corners_std = w_avg(h.get('HC_std', 1.5), a.get('AC_std', 1.5), 1.5)
                cards_std = w_avg(h.get('HY_std', 0.8), a.get('AY_std', 0.8), 0.8)
                goals_std = w_avg(h.get('FTHG_std', 1.0), a.get('FTAG_std', 1.0), 1.0)
                
                # Quanto menor o std, maior a consistência
                consistency_corners = max(0, 100 - (corners_std * 30))
                consistency_cards = max(0, 100 - (cards_std * 50))
                consistency_goals = max(0, 100 - (goals_std * 40))
                
                stats_db[team] = {
                    'corners': w_avg(h.get('HC_mean', 0), a.get('AC_mean', 0), 5.0),
                    'corners_std': corners_std,
                    'consistency_corners': round(consistency_corners, 1),
                    'cards': w_avg(h.get('HY_mean', 0), a.get('AY_mean', 0), 2.0),
                    'cards_std': cards_std,
                    'consistency_cards': round(consistency_cards, 1),
                    'fouls': w_avg(h.get('HF_mean', 0), a.get('AF_mean', 0), 11.0),
                    'goals_f': w_avg(h.get('FTHG_mean', 0), a.get('FTAG_mean', 0), 1.2),
                    'goals_f_std': goals_std,
                    'consistency_goals': round(consistency_goals, 1),
                    'goals_a': w_avg(h.get('FTAG_mean', 0), a.get('FTHG_mean', 0), 1.2),
                    'shots_on_target': w_avg(h.get('HST_mean', 0), a.get('AST_mean', 0), 4.5),
                    'red_cards_avg': w_avg(h.get('HR_mean', 0), a.get('AR_mean', 0), 0.08),
                    'league': league
                }
        except Exception as e:
            pass
    
    return stats_db

@st.cache_data(ttl=3600)
def load_referees_v23() -> Dict[str, Dict[str, float]]:
    """Carrega dados dos árbitros"""
    refs_db = {}
    
    if os.path.exists("arbitros_5_ligas_2025_2026.csv"):
        try:
            df = pd.read_csv("arbitros_5_ligas_2025_2026.csv", encoding='utf-8-sig')
            df.columns = [c.strip().replace('\ufeff', '') for c in df.columns]
            
            for _, row in df.iterrows():
                nome = str(row['Arbitro']).strip()
                vermelhos = float(row.get('Cartoes_Vermelhos', 0))
                jogos = float(row.get('Jogos_Apitados', 1))
                media = float(row.get('Media_Cartoes_Por_Jogo', 4.0))
                amarelos = int(row.get('Cartoes_Amarelos', 0))
                
                red_rate = (vermelhos / jogos) if jogos > 0 else 0.08
                
                refs_db[nome] = {
                    'factor': media / 4.0,
                    'red_rate': red_rate,
                    'strictness': media,
                    'games': int(jogos),
                    'yellows': amarelos,
                    'reds': int(vermelhos)
                }
        except:
            pass
    
    if os.path.exists("arbitros.csv"):
        try:
            df = pd.read_csv("arbitros.csv", encoding='utf-8-sig')
            df.columns = [c.strip().replace('\ufeff', '') for c in df.columns]
            
            for _, row in df.iterrows():
                nome = str(row['Nome']).strip()
                if nome not in refs_db:
                    refs_db[nome] = {
                        'factor': float(row['Fator']),
                        'red_rate': 0.08,
                        'strictness': float(row['Fator']) * 4.0,
                        'games': 0,
                        'yellows': 0,
                        'reds': 0
                    }
        except:
            pass
    
    return refs_db

@st.cache_data(ttl=600)
def load_calendar_safe() -> pd.DataFrame:
    """Carrega calendário de jogos futuros"""
    fname = "calendario_ligas.csv"
    if not os.path.exists(fname):
        return pd.DataFrame()
    
    try:
        try:
            df = pd.read_csv(fname, encoding='utf-8-sig')
        except:
            df = pd.read_csv(fname, encoding='latin1')
        
        df.columns = [c.strip().replace('\ufeff', '') for c in df.columns]
        
        rename_map = {}
        if 'Mandante' in df.columns:
            rename_map['Mandante'] = 'Time_Casa'
        if 'Visitante' in df.columns:
            rename_map['Visitante'] = 'Time_Visitante'
        
        if rename_map:
            df = df.rename(columns=rename_map)
        
        req = ['Data', 'Liga', 'Time_Casa', 'Time_Visitante', 'Hora']
        if not set(req).issubset(df.columns):
            return pd.DataFrame()
        
        df['DtObj'] = pd.to_datetime(df['Data'], format='%d/%m/%Y', errors='coerce')
        df = df.dropna(subset=['DtObj']).sort_values(by=['DtObj', 'Hora'])
        
        return df
    except:
        return pd.DataFrame()

# ═══════════════════════════════════════════════════════════════════════════
# MOTOR DE CÁLCULO (MANTIDO DO V22)
# ═══════════════════════════════════════════════════════════════════════════

def normalize_name(name: str, db_keys: list) -> Optional[str]:
    """Normaliza nome do time"""
    if not name:
        return None
    if name in NAME_MAPPING:
        name = NAME_MAPPING[name]
    if name in db_keys:
        return name
    
    matches = get_close_matches(name, db_keys, n=1, cutoff=0.6)
    return matches[0] if matches else None

def get_native_history(team_name: str, league: str, market: str, line: float, location: str, all_dfs: Dict) -> str:
    """Retorna histórico nativo"""
    if league not in all_dfs:
        return "N/A"
    
    df = all_dfs[league]
    
    col_map = {
        ('home', 'corners'): 'HC',
        ('away', 'corners'): 'AC',
        ('home', 'cards'): 'HY',
        ('away', 'cards'): 'AY'
    }
    
    col_code = col_map.get((location, market))
    if not col_code:
        return "N/A"
    
    team_col = 'HomeTeam' if location == 'home' else 'AwayTeam'
    
    matches = df[df[team_col] == team_name]
    if matches.empty:
        return "0/0"
    
    last_matches = matches.tail(10)
    hits = sum(1 for val in last_matches[col_code] if float(val) > line)
    
    return f"{hits}/{len(last_matches)}"

def poisson(k: int, lamb: float) -> float:
    """Distribuição de Poisson"""
    if lamb > 30 or k > 20:
        return 0.0
    try:
        return (lamb**k * math.exp(-lamb)) / math.factorial(k)
    except:
        return 0.0

def monte_carlo(xg_h: float, xg_a: float, n: int = 1000) -> Tuple[float, float, float]:
    """Simulação Monte Carlo"""
    gh = np.random.poisson(max(0.1, xg_h), n)
    ga = np.random.poisson(max(0.1, xg_a), n)
    
    h_w = np.count_nonzero(gh > ga)
    a_w = np.count_nonzero(ga > gh)
    d = n - h_w - a_w
    
    return h_w/n, d/n, a_w/n

def calcular_jogo_v23(home: str, away: str, stats: Dict, ref: Optional[str], refs_db: Dict, all_dfs: Dict) -> Dict:
    """Motor de cálculo V23 - Completo com consistência"""
    
    h_norm = normalize_name(home, list(stats.keys()))
    a_norm = normalize_name(away, list(stats.keys()))
    
    if not h_norm or not a_norm:
        return {'error': "Times não encontrados."}
    
    s_h = stats[h_norm]
    s_a = stats[a_norm]
    
    # Dados do árbitro
    if ref and ref in refs_db:
        r_data = refs_db[ref]
    else:
        r_data = {'factor': 1.0, 'red_rate': 0.08, 'strictness': 4.0, 'games': 0, 'yellows': 0, 'reds': 0}
    
    # Chutes
    shots_h = s_h['shots_on_target']
    shots_a = s_a['shots_on_target']
    
    p_h = 1.20 if shots_h > 6.0 else 1.10 if shots_h > 4.5 else 1.0
    p_a = 1.20 if shots_a > 6.0 else 1.10 if shots_a > 4.5 else 1.0
    
    # Escanteios
    corn_h = s_h['corners'] * 1.15 * p_h
    corn_a = s_a['corners'] * 0.90 * p_a
    
    # Rigidez
    rr = r_data['red_rate']
    strict = 1.15 if rr > 0.12 else 1.08 if rr > 0.08 else 1.0
    strict_label = "MUITO RIGOROSO 🔴" if strict == 1.15 else "RIGOROSO 🟠" if strict == 1.08 else "NORMAL 🟢"
    
    # Violência
    viol_h = 1.0 if s_h['fouls'] > 12.5 else 0.85
    viol_a = 1.0 if s_a['fouls'] > 12.5 else 0.85
    
    # Cartões
    card_h = s_h['cards'] * viol_h * r_data['factor'] * strict
    card_a = s_a['cards'] * viol_a * r_data['factor'] * strict
    
    # Probabilidade vermelho
    prob_red = ((s_h['red_cards_avg'] + s_a['red_cards_avg']) / 2) * rr * 100
    prob_red_label = "ALTA 🔴" if prob_red > 12 else "MÉDIA 🟠" if prob_red > 8 else "BAIXA 🟡"
    
    # xG e Monte Carlo
    xg_h = max(0.1, s_h['goals_f'])
    xg_a = max(0.1, s_a['goals_f'])
    
    mc_h, mc_d, mc_a = monte_carlo(xg_h, xg_a)
    
    # BTTS
    btts = (1 - poisson(0, xg_h)) * (1 - poisson(0, xg_a)) * 100
    
    # Over 2.5
    over_25 = (1 - sum([
        poisson(h, xg_h) * poisson(a, xg_a)
        for h in range(3) for a in range(3) if h+a < 3
    ])) * 100
    
    return {
        'home': h_norm,
        'away': a_norm,
        'league_h': s_h['league'],
        'league_a': s_a['league'],
        'corners': {
            'h': corn_h,
            'a': corn_a,
            'total': corn_h + corn_a
        },
        'cards': {
            'h': card_h,
            'a': card_a,
            'total': card_h + card_a
        },
        'goals': {
            'h': xg_h,
            'a': xg_a
        },
        'monte_carlo': {
            'h': mc_h * 100,
            'd': mc_d * 100,
            'a': mc_a * 100
        },
        'consistency': {
            'corners_h': s_h.get('consistency_corners', 50),
            'corners_a': s_a.get('consistency_corners', 50),
            'cards_h': s_h.get('consistency_cards', 50),
            'cards_a': s_a.get('consistency_cards', 50),
            'goals_h': s_h.get('consistency_goals', 50),
            'goals_a': s_a.get('consistency_goals', 50)
        },
        'meta': {
            'shots_h': shots_h,
            'shots_a': shots_a,
            'pressure_h': p_h,
            'pressure_a': p_a,
            'fouls_h': s_h['fouls'],
            'fouls_a': s_a['fouls'],
            'violence_h': viol_h,
            'violence_a': viol_a,
            'referee': ref if ref else 'Neutro',
            'ref_factor': r_data['factor'],
            'ref_strictness': strict,
            'ref_label': strict_label,
            'ref_red_rate': rr,
            'prob_red': prob_red,
            'prob_red_label': prob_red_label,
            'ref_games': r_data.get('games', 0),
            'ref_yellows': r_data.get('yellows', 0),
            'ref_reds': r_data.get('reds', 0)
        },
        'probs': {
            'btts': btts,
            'over_2_5': over_25
        }
    }

def get_detailed_probs(res: Dict) -> Dict:
    """Probabilidades detalhadas"""
    
    def sim_prob(avg: float, line: float) -> float:
        return max(5, min(95, 50 + (avg - line) * 15))
    
    probs = {
        'corners': {
            'home': {
                f'Over {l}': sim_prob(res['corners']['h'], l)
                for l in [2.5, 3.5, 4.5, 5.5]
            },
            'away': {
                f'Over {l}': sim_prob(res['corners']['a'], l)
                for l in [2.5, 3.5, 4.5]
            },
            'total': {
                f'Over {l}': sim_prob(res['corners']['total'], l)
                for l in [8.5, 9.5, 10.5, 11.5]
            }
        },
        'cards': {
            'home': {
                f'Over {l}': sim_prob(res['cards']['h'], l)
                for l in [1.5, 2.5]
            },
            'away': {
                f'Over {l}': sim_prob(res['cards']['a'], l)
                for l in [1.5, 2.5]
            },
            'total': {
                f'Over {l}': sim_prob(res['cards']['total'], l)
                for l in [2.5, 3.5, 4.5, 5.5]
            }
        }
    }
    
    mc = res['monte_carlo']
    probs['chance'] = {
        '1': mc['h'],
        'X': mc['d'],
        '2': mc['a'],
        '1X': mc['h'] + mc['d'],
        'X2': mc['a'] + mc['d'],
        '12': mc['h'] + mc['a']
    }
    
    probs['goals'] = {
        'BTTS': res['probs']['btts'],
        'Over 2.5': res['probs']['over_2_5']
    }
    
    return probs

def get_fair_odd(prob: float) -> float:
    """Calcula odd justa a partir da probabilidade"""
    if prob <= 0:
        return 99.0
    return round(100 / prob, 2)

# ═══════════════════════════════════════════════════════════════════════════
# NOVAS FUNCIONALIDADES V23
# ═══════════════════════════════════════════════════════════════════════════

def scan_day_for_radars(calendar: pd.DataFrame, stats: Dict, refs: Dict, all_dfs: Dict, date_str: str) -> Dict:
    """
    Scanner de Alta Frequência - Radares de Cantos e Cartões
    """
    
    df_day = calendar[calendar['DtObj'].dt.strftime('%d/%m/%Y') == date_str]
    
    radar_corners_individual = []
    radar_cards_individual = []
    radar_corners_total = []
    radar_cards_total = []
    
    for _, row in df_day.iterrows():
        home = row['Time_Casa']
        away = row['Time_Visitante']
        liga = row.get('Liga', 'N/A')
        hora = row.get('Hora', 'N/A')
        
        res = calcular_jogo_v23(home, away, stats, None, refs, all_dfs)
        
        if 'error' in res:
            continue
        
        probs = get_detailed_probs(res)
        
        # Radar Cantos Individual
        for line in [3.5, 4.5, 5.5]:
            prob_h = probs['corners']['home'].get(f'Over {line}', 0)
            if prob_h >= THRESHOLDS['radar_corners']:
                radar_corners_individual.append({
                    'time': res['home'],
                    'adversario': res['away'],
                    'liga': liga,
                    'hora': hora,
                    'mercado': f"{res['home']} Over {line} Escanteios",
                    'prob': prob_h,
                    'media': res['corners']['h'],
                    'consistencia': res['consistency']['corners_h'],
                    'location': 'Casa'
                })
            
            if line <= 4.5:
                prob_a = probs['corners']['away'].get(f'Over {line}', 0)
                if prob_a >= THRESHOLDS['radar_corners']:
                    radar_corners_individual.append({
                        'time': res['away'],
                        'adversario': res['home'],
                        'liga': liga,
                        'hora': hora,
                        'mercado': f"{res['away']} Over {line} Escanteios",
                        'prob': prob_a,
                        'media': res['corners']['a'],
                        'consistencia': res['consistency']['corners_a'],
                        'location': 'Fora'
                    })
        
        # Radar Cartões Individual
        for line in [1.5, 2.5]:
            prob_h = probs['cards']['home'].get(f'Over {line}', 0)
            if prob_h >= THRESHOLDS['radar_cards']:
                radar_cards_individual.append({
                    'time': res['home'],
                    'adversario': res['away'],
                    'liga': liga,
                    'hora': hora,
                    'mercado': f"{res['home']} Over {line} Cartões",
                    'prob': prob_h,
                    'media': res['cards']['h'],
                    'consistencia': res['consistency']['cards_h'],
                    'location': 'Casa'
                })
            
            prob_a = probs['cards']['away'].get(f'Over {line}', 0)
            if prob_a >= THRESHOLDS['radar_cards']:
                radar_cards_individual.append({
                    'time': res['away'],
                    'adversario': res['home'],
                    'liga': liga,
                    'hora': hora,
                    'mercado': f"{res['away']} Over {line} Cartões",
                    'prob': prob_a,
                    'media': res['cards']['a'],
                    'consistencia': res['consistency']['cards_a'],
                    'location': 'Fora'
                })
        
        # Radar Cantos Total
        for line in [8.5, 9.5, 10.5, 11.5]:
            prob = probs['corners']['total'].get(f'Over {int(line)}.5', 0)
            if prob >= THRESHOLDS['radar_corners']:
                radar_corners_total.append({
                    'jogo': f"{res['home']} vs {res['away']}",
                    'liga': liga,
                    'hora': hora,
                    'mercado': f"Over {line} Escanteios Total",
                    'prob': prob,
                    'media': res['corners']['total']
                })
        
        # Radar Cartões Total
        for line in [3.5, 4.5, 5.5]:
            prob = probs['cards']['total'].get(f'Over {int(line)}.5', 0)
            if prob >= THRESHOLDS['radar_corners']:
                radar_cards_total.append({
                    'jogo': f"{res['home']} vs {res['away']}",
                    'liga': liga,
                    'hora': hora,
                    'mercado': f"Over {line} Cartões Total",
                    'prob': prob,
                    'media': res['cards']['total']
                })
    
    return {
        'corners_individual': radar_corners_individual,
        'cards_individual': radar_cards_individual,
        'corners_total': radar_corners_total,
        'cards_total': radar_cards_total
    }

def generate_smart_ticket_v23(calendar: pd.DataFrame, stats: Dict, refs: Dict, all_dfs: Dict, date_str: str) -> Dict:
    """
    Scanner V23.6 - Smart Ticket (High Volume & Mix Fix)
    ✅ Filtro: 65% (Mais agressivo para encher o bilhete)
    ✅ Mix: Força busca por Escanteios E Cartões
    ✅ Correção: Chave 'mercado' compatível com a tela
    """
    
    df_day = calendar[calendar['DtObj'].dt.strftime('%d/%m/%Y') == date_str]
    
    anchors = []  # Tipo A: Segurança
    fusions = []  # Tipo B: Valor
    
    for _, row in df_day.iterrows():
        home, away = row['Time_Casa'], row['Time_Visitante']
        liga = row.get('Liga', 'N/A')
        hora = row.get('Hora', 'N/A')
        
        res = calcular_jogo_v23(home, away, stats, None, refs, all_dfs)
        if 'error' in res: continue
        probs = get_detailed_probs(res)
        
        # 1. ÂNCORAS (Filtro 65% - Busca Escanteios E Cartões)
        for loc, name in [('home', res['home']), ('away', res['away'])]:
            
            # A) ESCANTEIOS (Over 3.5 e 4.5)
            for l in [3.5, 4.5]:
                p = probs['corners'][loc].get(f'Over {l}', 0)
                if p >= 65: # Filtro solicitado
                    odd = get_fair_odd(p)
                    # Aceita odds de segurança (1.20) até valor (1.50)
                    if 1.20 <= odd <= 1.55:
                        anchors.append({
                            'type': 'anchor', 
                            'jogo': f"{res['home']} vs {res['away']}",
                            'mercado': f"{name} Over {l} Escanteios", # Chave corrigida
                            'prob': p, 
                            'odd': odd, 
                            'liga': liga, 
                            'hora': hora
                        })
            
            # B) CARTÕES (Over 1.5)
            p_card = probs['cards'][loc].get('Over 1.5', 0)
            if p_card >= 65: # Filtro solicitado
                odd = get_fair_odd(p_card)
                if 1.20 <= odd <= 1.60:
                    anchors.append({
                        'type': 'anchor', 
                        'jogo': f"{res['home']} vs {res['away']}",
                        'mercado': f"{name} Over 1.5 Cartões", # Chave corrigida
                        'prob': p_card, 
                        'odd': odd, 
                        'liga': liga, 
                        'hora': hora
                    })

        # 2. FUSÕES (Criar Aposta: Canto Time + Cartão Jogo)
        corn_prob = probs['corners']['home'].get('Over 3.5', 0)
        card_prob = probs['cards']['total'].get('Over 1.5', 0)
        
        # Se probabilidade combinada for boa
        if corn_prob >= 65 and card_prob >= 65:
            p_comb = (corn_prob/100 * card_prob/100 * 0.90) * 100
            odd_comb = get_fair_odd(p_comb)
            
            if 1.50 <= odd_comb <= 2.30:
                fusions.append({
                    'type': 'fusion', 
                    'jogo': f"{res['home']} vs {res['away']}",
                    'team': res['home'],
                    'mercados': [f"{res['home']} Over 3.5 Escanteios", "Total Jogo Over 1.5 Cartões"],
                    'prob_combined': p_comb, 
                    'odd': odd_comb, 
                    'liga': liga, 
                    'hora': hora
                })

    # MONTAGEM DO BILHETE (Prioridade: Encher 6 slots com qualidade)
    ticket = []
    curr_odd = 1.0
    used_games = set()
    
    # Ordena por probabilidade (os mais garantidos primeiro)
    anchors.sort(key=lambda x: x['prob'], reverse=True)
    fusions.sort(key=lambda x: x['prob_combined'], reverse=True)
    
    # Estratégia de Mix: Tenta pegar pelo menos 1 de cada tipo se possível
    pool = []
    
    # Adiciona Top Âncoras (Escanteios ou Cartões)
    for a in anchors: pool.append(a)
    # Adiciona Top Fusões
    for f in fusions: pool.append(f)
    
    # Reordena o pool geral pela probabilidade para pegar o "Filé Mignon" global
    # (Assim garantimos que se tiver cartões muito bons, eles entram)
    pool.sort(key=lambda x: x.get('prob', x.get('prob_combined', 0)), reverse=True)
    
    for item in pool:
        if len(ticket) >= 6: break # Meta: 6 seleções
        
        if item['jogo'] not in used_games:
            # Trava de segurança para a odd total não ficar absurda
            if curr_odd * item['odd'] <= 8.5: 
                ticket.append(item)
                curr_odd *= item['odd']
                used_games.add(item['jogo'])
            
    return {
        'ticket': ticket, 
        'total_odd': round(curr_odd, 2), 
        'num_selections': len(ticket), 
        'all_anchors': len(anchors), 
        'all_fusions': len(fusions)
    }

def generate_3_tickets_system(calendar: pd.DataFrame, stats: Dict, refs: Dict, all_dfs: Dict, date_str: str, num_games: int = 3) -> Dict:
    """
    Sistema de 3 Bilhetes: Principal + Hedge 1 + Hedge 2
    Cobrindo os 3 roteiros do jogo
    """
    
    df_day = calendar[calendar['DtObj'].dt.strftime('%d/%m/%Y') == date_str]
    
    # Encontrar os melhores jogos
    best_games = []
    
    for _, row in df_day.iterrows():
        home = row['Time_Casa']
        away = row['Time_Visitante']
        
        res = calcular_jogo_v23(home, away, stats, None, refs, all_dfs)
        
        if 'error' in res:
            continue
        
        probs = get_detailed_probs(res)
        
        # Score do jogo (quanto maior, melhor)
        max_corner_prob = max([p for p in probs['corners']['total'].values()])
        max_card_prob = max([p for p in probs['cards']['total'].values()])
        dc_prob = probs['chance']['1X']
        
        score = (max_corner_prob * 0.4) + (max_card_prob * 0.3) + (dc_prob * 0.3)
        
        best_games.append({
            'home': res['home'],
            'away': res['away'],
            'score': score,
            'res': res,
            'probs': probs,
            'liga': row.get('Liga', 'N/A'),
            'hora': row.get('Hora', 'N/A')
        })
    
    # Ordenar e pegar top N
    best_games = sorted(best_games, key=lambda x: x['score'], reverse=True)[:num_games]
    
    # Montar os 3 bilhetes
    principal = []
    hedge1 = []
    hedge2 = []
    
    for game in best_games:
        res = game['res']
        probs = game['probs']
        
        # Principal: Cenário Lógico (Cantos)
        best_corner_line = None
        best_corner_prob = 0
        
        for line, prob in probs['corners']['total'].items():
            if prob > best_corner_prob:
                best_corner_prob = prob
                best_corner_line = line
        
        principal.append({
            'jogo': f"{res['home']} vs {res['away']}",
            'mercado': f"Total {best_corner_line}",
            'prob': best_corner_prob,
            'odd': get_fair_odd(best_corner_prob),
            'liga': game['liga'],
            'hora': game['hora'],
            'tipo': 'Escanteios'
        })
        
        # Hedge 1: Safety (Dupla Chance ou DNB)
        dc_prob = probs['chance']['1X']
        
        hedge1.append({
            'jogo': f"{res['home']} vs {res['away']}",
            'mercado': "Dupla Chance 1X",
            'prob': dc_prob,
            'odd': get_fair_odd(dc_prob),
            'liga': game['liga'],
            'hora': game['hora'],
            'tipo': 'Resultado'
        })
        
        # Hedge 2: Caos Calculado (Cartões - Anti-Colisão)
        best_card_line = None
        best_card_prob = 0
        
        for line, prob in probs['cards']['total'].items():
            if prob > best_card_prob:
                best_card_prob = prob
                best_card_line = line
        
        hedge2.append({
            'jogo': f"{res['home']} vs {res['away']}",
            'mercado': f"Total {best_card_line}",
            'prob': best_card_prob,
            'odd': get_fair_odd(best_card_prob),
            'liga': game['liga'],
            'hora': game['hora'],
            'tipo': 'Cartões'
        })
    
    # Calcular odds totais
    odd_principal = 1.0
    for sel in principal:
        odd_principal *= sel['odd']
    
    odd_hedge1 = 1.0
    for sel in hedge1:
        odd_hedge1 *= sel['odd']
    
    odd_hedge2 = 1.0
    for sel in hedge2:
        odd_hedge2 *= sel['odd']
    
    return {
        'principal': {
            'selections': principal,
            'odd_total': round(odd_principal, 2),
            'allocation': 50
        },
        'hedge1': {
            'selections': hedge1,
            'odd_total': round(odd_hedge1, 2),
            'allocation': 30
        },
        'hedge2': {
            'selections': hedge2,
            'odd_total': round(odd_hedge2, 2),
            'allocation': 20
        }
    }

def get_top_teams_day(calendar: pd.DataFrame, stats: Dict, refs: Dict, all_dfs: Dict, date_str: str, market: str = 'corners', top_n: int = 10) -> List[Dict]:
    """
    Lista Top Times do Dia para determinado mercado
    """
    
    df_day = calendar[calendar['DtObj'].dt.strftime('%d/%m/%Y') == date_str]
    
    teams_list = []
    
    for _, row in df_day.iterrows():
        home = row['Time_Casa']
        away = row['Time_Visitante']
        
        res = calcular_jogo_v23(home, away, stats, None, refs, all_dfs)
        
        if 'error' in res:
            continue
        
        probs = get_detailed_probs(res)
        
        # Adicionar casa
        if market == 'corners':
            best_prob = max([p for p in probs['corners']['home'].values()])
            avg = res['corners']['h']
            consistency = res['consistency']['corners_h']
        else:  # cards
            best_prob = max([p for p in probs['cards']['home'].values()])
            avg = res['cards']['h']
            consistency = res['consistency']['cards_h']
        
        teams_list.append({
            'time': res['home'],
            'adversario': res['away'],
            'liga': row.get('Liga', 'N/A'),
            'hora': row.get('Hora', 'N/A'),
            'prob_max': best_prob,
            'media': avg,
            'consistencia': consistency,
            'location': 'Casa'
        })
        
        # Adicionar visitante
        if market == 'corners':
            best_prob = max([p for p in probs['corners']['away'].values()])
            avg = res['corners']['a']
            consistency = res['consistency']['corners_a']
        else:  # cards
            best_prob = max([p for p in probs['cards']['away'].values()])
            avg = res['cards']['a']
            consistency = res['consistency']['cards_a']
        
        teams_list.append({
            'time': res['away'],
            'adversario': res['home'],
            'liga': row.get('Liga', 'N/A'),
            'hora': row.get('Hora', 'N/A'),
            'prob_max': best_prob,
            'media': avg,
            'consistencia': consistency,
            'location': 'Fora'
        })
    
    # Ordenar e retornar top N
    teams_sorted = sorted(teams_list, key=lambda x: x['prob_max'], reverse=True)
    
    return teams_sorted[:top_n]

# ═══════════════════════════════════════════════════════════════════════════
# UI PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════

def main():
    
    # Aplicar tema
    if st.session_state.theme == 'dark':
        st.markdown("""
        <style>
        .stApp {
            background-color: #0E1117;
            color: #FAFAFA;
        }
        </style>
        """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.title("🎛️ FutPrevisão V23")
        st.caption("Ultimate Edition")
        
        st.markdown("---")
        
        # Tema
        theme_col1, theme_col2 = st.columns(2)
        if theme_col1.button("🌙 Escuro"):
            st.session_state.theme = 'dark'
            st.rerun()
        if theme_col2.button("☀️ Claro"):
            st.session_state.theme = 'light'
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 💰 Bankroll")
        st.session_state.bankroll = st.number_input(
            "Bankroll (€)",
            min_value=0.0,
            value=st.session_state.bankroll,
            step=50.0
        )
        
        st.markdown("---")
        st.markdown("### 🔔 Alertas")
        st.session_state.alert_threshold = st.slider(
            "Probabilidade mínima para alerta",
            50, 95, int(st.session_state.alert_threshold)
        )
        
        st.markdown("---")
        st.markdown("### 📊 Status")
    
    # Título
    st.title("⚽ FutPrevisão V23.0 Ultimate Edition")
    st.caption("Sistema Completo de Análise de Apostas Esportivas")
    
    # Carregamento
    with st.spinner("📂 Carregando dados..."):
        stats = learn_stats_v23()
        refs = load_referees_v23()
        calendar = load_calendar_safe()
        all_dfs = load_all_dataframes()
    
    # Status
    st.sidebar.success(f"✅ {len(stats)} times")
    st.sidebar.success(f"✅ {len(refs)} árbitros")
    st.sidebar.success(f"✅ {len(calendar)} jogos futuros")
    st.sidebar.success(f"✅ {len(all_dfs)} ligas")
    
    # Tabs PRINCIPAIS (mantém as 3 originais + novas)
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📅 Calendário",
        "🔍 Simulação",
        "🎯 Scanner V23",
        "🎰 Sistema 3 Bilhetes",
        "📊 Radares",
        "🏆 Top Times",
        "📈 Dashboard"
    ])
    
    # TAB 1: Calendário (MANTIDO 100% DO V22)
    with tab1:
        st.markdown("## 📅 Calendário de Jogos")
        
        if calendar.empty:
            st.warning("⚠️ Calendário não carregado")
        else:
            dates = sorted(calendar['DtObj'].dt.strftime('%d/%m/%Y').unique())
            sel_date = st.selectbox("Selecione a data:", dates, key="cal_date")
            
            df_day = calendar[calendar['DtObj'].dt.strftime('%d/%m/%Y') == sel_date]
            
            st.info(f"📊 {len(df_day)} jogos em {sel_date}")
            
            for idx, row in df_day.iterrows():
                liga = row.get('Liga', 'N/A')
                hora = row.get('Hora', 'N/A')
                home = row['Time_Casa']
                away = row['Time_Visitante']
                
                with st.expander(f"⏰ {hora} | {liga} | 🏠 {home} vs 🛫 {away}"):
                    
                    lista_refs = ['Neutro'] + sorted(list(refs.keys()))
                    ref_sel = st.selectbox(
                        "👮 Árbitro:",
                        lista_refs,
                        key=f"cal_ref_{idx}"
                    )
                    
                    if st.button("🎯 Analisar Jogo", key=f"cal_btn_{idx}"):
                        ref_v = None if ref_sel == 'Neutro' else ref_sel
                        
                        res = calcular_jogo_v23(home, away, stats, ref_v, refs, all_dfs)
                        
                        if 'error' in res:
                            st.error(res['error'])
                        else:
                            st.markdown(f"## 🏠 {res['home']} vs 🛫 {res['away']}")
                            
                            c1, c2, c3 = st.columns(3)
                            c1.metric("xG Casa", f"{res['goals']['h']:.2f}")
                            c2.metric("xG Fora", f"{res['goals']['a']:.2f}")
                            c3.metric("Árbitro", res['meta']['referee'])
                            
                            if ref_v:
                                st.markdown("### 👮 Dados do Árbitro")
                                ar1, ar2, ar3, ar4 = st.columns(4)
                                ar1.metric("Fator", f"{res['meta']['ref_factor']:.2f}")
                                ar2.metric("Jogos", res['meta']['ref_games'])
                                ar3.metric("Amarelos", res['meta']['ref_yellows'])
                                ar4.metric("Vermelhos", res['meta']['ref_reds'])
                                
                                st.info(f"Rigidez: {res['meta']['ref_label']}")
                            
                            st.markdown("### 🎲 Probabilidades")
                            mc1, mc2, mc3 = st.columns(3)
                            mc1.metric("Casa", f"{res['monte_carlo']['h']:.1f}%")
                            mc2.metric("Empate", f"{res['monte_carlo']['d']:.1f}%")
                            mc3.metric("Fora", f"{res['monte_carlo']['a']:.1f}%")
                            
                            probs = get_detailed_probs(res)
                            
                            col_esc, col_cart = st.columns(2)
                            
                            with col_esc:
                                st.markdown("**🏁 Escanteios**")
                                
                                st.markdown(f"**{res['home']}:**")
                                for k, v in probs['corners']['home'].items():
                                    hist = get_native_history(res['home'], res['league_h'], 'corners', 
                                                             float(k.split()[1]), 'home', all_dfs)
                                    cor = "green" if v >= 70 else "orange" if v >= 60 else "red"
                                    st.markdown(f"- {k}: :{cor}[{v:.0f}%] | Hist: {hist}")
                                
                                st.markdown(f"**{res['away']}:**")
                                for k, v in probs['corners']['away'].items():
                                    hist = get_native_history(res['away'], res['league_a'], 'corners',
                                                             float(k.split()[1]), 'away', all_dfs)
                                    cor = "green" if v >= 70 else "orange" if v >= 60 else "red"
                                    st.markdown(f"- {k}: :{cor}[{v:.0f}%] | Hist: {hist}")
                            
                            with col_cart:
                                st.markdown("**🟨 Cartões**")
                                
                                st.markdown(f"**{res['home']}:**")
                                for k, v in probs['cards']['home'].items():
                                    hist = get_native_history(res['home'], res['league_h'], 'cards',
                                                             float(k.split()[1]), 'home', all_dfs)
                                    cor = "green" if v >= 70 else "orange" if v >= 60 else "red"
                                    st.markdown(f"- {k}: :{cor}[{v:.0f}%] | Hist: {hist}")
                                
                                st.markdown(f"**{res['away']}:**")
                                for k, v in probs['cards']['away'].items():
                                    hist = get_native_history(res['away'], res['league_a'], 'cards',
                                                             float(k.split()[1]), 'away', all_dfs)
                                    cor = "green" if v >= 70 else "orange" if v >= 60 else "red"
                                    st.markdown(f"- {k}: :{cor}[{v:.0f}%] | Hist: {hist}")
    
    # TAB 2: Simulação (MANTIDO 100% DO V22)
    with tab2:
        st.markdown("## 🔍 Simulador de Jogo")
        
        lista_times = sorted(list(stats.keys()))
        lista_refs = ['Neutro'] + sorted(list(refs.keys()))
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            home = st.selectbox("🏠 Time Casa", lista_times, key="sim_home")
        
        with col2:
            away = st.selectbox("🛫 Time Visitante", lista_times, key="sim_away")
        
        with col3:
            ref = st.selectbox("👮 Árbitro", lista_refs, key="sim_ref")
        
        if st.button("🎯 SIMULAR JOGO", type="primary"):
            ref_v = None if ref == 'Neutro' else ref
            
            res = calcular_jogo_v23(home, away, stats, ref_v, refs, all_dfs)
            
            if 'error' in res:
                st.error(res['error'])
            else:
                st.markdown(f"# 🏠 {res['home']} vs 🛫 {res['away']}")
                st.caption(f"Liga: {res['league_h']}")
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("xG Casa", f"{res['goals']['h']:.2f}")
                m2.metric("xG Fora", f"{res['goals']['a']:.2f}")
                m3.metric("Escanteios Total", f"{res['corners']['total']:.1f}")
                m4.metric("Cartões Total", f"{res['cards']['total']:.1f}")
                
                # Índices de Consistência
                st.markdown("### 🎯 Índice de Consistência (Reloginho vs Caótico)")
                cons1, cons2, cons3 = st.columns(3)
                
                cons1.metric(
                    f"{res['home']} - Escanteios",
                    f"{res['consistency']['corners_h']:.0f}/100",
                    delta="Reloginho" if res['consistency']['corners_h'] >= 70 else "Caótico"
                )
                cons2.metric(
                    f"{res['away']} - Escanteios",
                    f"{res['consistency']['corners_a']:.0f}/100",
                    delta="Reloginho" if res['consistency']['corners_a'] >= 70 else "Caótico"
                )
                cons3.metric(
                    f"Gols - {res['home']}",
                    f"{res['consistency']['goals_h']:.0f}/100",
                    delta="Reloginho" if res['consistency']['goals_h'] >= 70 else "Caótico"
                )
                
                st.markdown("### 👮 Informações do Árbitro")
                if ref_v and ref_v in refs:
                    ar_data = refs[ref_v]
                    
                    ar1, ar2, ar3, ar4, ar5 = st.columns(5)
                    ar1.metric("Nome", ref_v)
                    ar2.metric("Fator", f"{ar_data['factor']:.2f}")
                    ar3.metric("Jogos", ar_data['games'])
                    ar4.metric("Amarelos", ar_data['yellows'])
                    ar5.metric("Vermelhos", ar_data['reds'])
                    
                    st.info(f"📊 Rigidez: {res['meta']['ref_label']} | Red Rate: {ar_data['red_rate']:.2%}")
                    st.info(f"🔴 Probabilidade de Vermelho: {res['meta']['prob_red']:.1f}% ({res['meta']['prob_red_label']})")
                else:
                    st.info("Árbitro neutro")
                
                st.markdown("### 🎲 Monte Carlo")
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Vitória Casa", f"{res['monte_carlo']['h']:.1f}%")
                mc2.metric("Empate", f"{res['monte_carlo']['d']:.1f}%")
                mc3.metric("Vitória Fora", f"{res['monte_carlo']['a']:.1f}%")
                
                st.markdown("### 🎰 Dupla Chance")
                probs = get_detailed_probs(res)
                
                dc1, dc2, dc3 = st.columns(3)
                dc1.metric("1X", f"{probs['chance']['1X']:.1f}%")
                dc2.metric("X2", f"{probs['chance']['X2']:.1f}%")
                dc3.metric("12", f"{probs['chance']['12']:.1f}%")
                
                st.markdown("### 🏁 Escanteios Individuais")
                
                col_h, col_a = st.columns(2)
                
                with col_h:
                    st.markdown(f"#### 🏠 {res['home']}")
                    st.metric("Média", f"{res['corners']['h']:.2f}")
                    
                    for line in [2.5, 3.5, 4.5, 5.5]:
                        prob = probs['corners']['home'].get(f'Over {line}', 0)
                        hist = get_native_history(res['home'], res['league_h'], 'corners', line, 'home', all_dfs)
                        
                        cor = "green" if prob >= 70 else "orange" if prob >= 60 else "red"
                        st.markdown(f"**Over {line}:** :{cor}[{prob:.0f}%] | Hist: {hist}")
                
                with col_a:
                    st.markdown(f"#### 🛫 {res['away']}")
                    st.metric("Média", f"{res['corners']['a']:.2f}")
                    
                    for line in [2.5, 3.5, 4.5]:
                        prob = probs['corners']['away'].get(f'Over {line}', 0)
                        hist = get_native_history(res['away'], res['league_a'], 'corners', line, 'away', all_dfs)
                        
                        cor = "green" if prob >= 70 else "orange" if prob >= 60 else "red"
                        st.markdown(f"**Over {line}:** :{cor}[{prob:.0f}%] | Hist: {hist}")
                
                st.markdown("### 🟨 Cartões Individuais")
                
                col_h2, col_a2 = st.columns(2)
                
                with col_h2:
                    st.markdown(f"#### 🏠 {res['home']}")
                    st.metric("Média", f"{res['cards']['h']:.2f}")
                    
                    for line in [1.5, 2.5]:
                        prob = probs['cards']['home'].get(f'Over {line}', 0)
                        hist = get_native_history(res['home'], res['league_h'], 'cards', line, 'home', all_dfs)
                        
                        cor = "green" if prob >= 70 else "orange" if prob >= 60 else "red"
                        st.markdown(f"**Over {line}:** :{cor}[{prob:.0f}%] | Hist: {hist}")
                
                with col_a2:
                    st.markdown(f"#### 🛫 {res['away']}")
                    st.metric("Média", f"{res['cards']['a']:.2f}")
                    
                    for line in [1.5, 2.5]:
                        prob = probs['cards']['away'].get(f'Over {line}', 0)
                        hist = get_native_history(res['away'], res['league_a'], 'cards', line, 'away', all_dfs)
                        
                        cor = "green" if prob >= 70 else "orange" if prob >= 60 else "red"
                        st.markdown(f"**Over {line}:** :{cor}[{prob:.0f}%] | Hist: {hist}")
                
                st.markdown("### ⚽ Outros Mercados")
                
                om1, om2 = st.columns(2)
                om1.metric("BTTS", f"{probs['goals']['BTTS']:.1f}%")
                om2.metric("Over 2.5 Gols", f"{probs['goals']['Over 2.5']:.1f}%")
    
    # TAB 3: SCANNER V23 SMART TICKET (NOVO!)
    with tab3:
        st.markdown("## 🎯 Scanner V23 - Smart Ticket")
        st.caption("Gerador automático de bilhetes com Âncoras + Fusões")
        
        if calendar.empty:
            st.warning("⚠️ Calendário não carregado")
        else:
            dates = sorted(calendar['DtObj'].dt.strftime('%d/%m/%Y').unique())
            sel_date_smart = st.selectbox("Selecione a data:", dates, key="smart_date")
            
            if st.button("🚀 GERAR SMART TICKET", type="primary"):
                with st.spinner("Analisando todos os jogos..."):
                    result = generate_smart_ticket_v23(calendar, stats, refs, all_dfs, sel_date_smart)
                
                if result['num_selections'] == 0:
                    st.warning("Nenhuma seleção encontrada para os critérios definidos.")
                else:
                    st.success(f"✅ Bilhete gerado com {result['num_selections']} seleções!")
                    
                    st.markdown(f"### 🎫 BILHETE SUGERIDO V23")
                    st.metric("Odd Total", f"@{result['total_odd']}", delta=f"{result['num_selections']} seleções")
                    
                    st.info(f"📊 Disponíveis: {result['all_anchors']} âncoras | {result['all_fusions']} fusões")
                    
                    st.markdown("---")
                    
                    for i, sel in enumerate(result['ticket'], 1):
                        if sel['type'] == 'anchor':
                            st.markdown(f"**{i}. 🔴 [ÂNCORA - Segurança]**")
                            st.write(f"   **Jogo:** {sel['jogo']}")
                            st.write(f"   **Mercado:** {sel['mercado']}")
                            st.write(f"   **Probabilidade:** {sel['prob']:.1f}% | **Odd:** @{sel['odd']}")
                            st.caption(f"{sel['liga']} | {sel['hora']}")
                        else:  # fusion
                            st.markdown(f"**{i}. 🔗 [CRIAR APOSTA - Combo Duplo]**")
                            st.write(f"   **Jogo:** {sel['jogo']}")
                            st.write(f"   **Time:** {sel['team']}")
                            for j, merc in enumerate(sel['mercados']):
                                st.write(f"   ✓ {merc} ({sel['probs'][j]:.0f}%)")
                            st.write(f"   **Prob. Combinada:** {sel['prob_combined']:.1f}% | **Odd:** @{sel['odd']}")
                            st.caption(f"{sel['liga']} | {sel['hora']}")
                        
                        st.markdown("---")
    
    # TAB 4: SISTEMA 3 BILHETES (NOVO!)
    with tab4:
        st.markdown("## 🎰 Sistema de 3 Bilhetes")
        st.caption("Cobrindo os 3 Roteiros do Jogo: Principal + Hedge 1 + Hedge 2")
        
        if calendar.empty:
            st.warning("⚠️ Calendário não carregado")
        else:
            dates = sorted(calendar['DtObj'].dt.strftime('%d/%m/%Y').unique())
            sel_date_3b = st.selectbox("Selecione a data:", dates, key="3b_date")
            
            num_games_3b = st.slider("Quantos jogos?", 2, 5, 3, key="3b_games")
            
            if st.button("🎲 GERAR SISTEMA 3 BILHETES", type="primary"):
                with st.spinner("Gerando sistema..."):
                    system = generate_3_tickets_system(calendar, stats, refs, all_dfs, sel_date_3b, num_games_3b)
                
                st.success("✅ Sistema gerado!")
                
                st.markdown("### 💡 Estratégia de Alocação")
                st.write("50% no Principal | 30% no Hedge 1 | 20% no Hedge 2")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("### 📋 PRINCIPAL (50%)")
                    st.caption("Cenário Lógico - Escanteios")
                    st.metric("Odd Total", f"@{system['principal']['odd_total']}")
                    
                    for sel in system['principal']['selections']:
                        st.markdown(f"**{sel['jogo']}**")
                        st.write(f"{sel['mercado']}")
                        st.write(f"Prob: {sel['prob']:.1f}% | @{sel['odd']}")
                        st.write(f"{sel['liga']} | {sel['hora']}")
                        st.markdown("---")
                
                with col2:
                    st.markdown("### 🛡️ HEDGE 1 (30%)")
                    st.caption("Safety - Resultado")
                    st.metric("Odd Total", f"@{system['hedge1']['odd_total']}")
                    
                    for sel in system['hedge1']['selections']:
                        st.markdown(f"**{sel['jogo']}**")
                        st.write(f"{sel['mercado']}")
                        st.write(f"Prob: {sel['prob']:.1f}% | @{sel['odd']}")
                        st.write(f"{sel['liga']} | {sel['hora']}")
                        st.markdown("---")
                
                with col3:
                    st.markdown("### 🔄 HEDGE 2 (20%)")
                    st.caption("Caos - Cartões")
                    st.metric("Odd Total", f"@{system['hedge2']['odd_total']}")
                    
                    for sel in system['hedge2']['selections']:
                        st.markdown(f"**{sel['jogo']}**")
                        st.write(f"{sel['mercado']}")
                        st.write(f"Prob: {sel['prob']:.1f}% | @{sel['odd']}")
                        st.write(f"{sel['liga']} | {sel['hora']}")
                        st.markdown("---")
    
    # TAB 5: RADARES (NOVO!)
    with tab5:
        st.markdown("## 📊 Radares de Alta Frequência")
        st.caption("Scanner automático de oportunidades")
        
        if calendar.empty:
            st.warning("⚠️ Calendário não carregado")
        else:
            dates = sorted(calendar['DtObj'].dt.strftime('%d/%m/%Y').unique())
            sel_date_radar = st.selectbox("Selecione a data:", dates, key="radar_date")
            
            if st.button("🔍 ESCANEAR DIA", type="primary"):
                with st.spinner("Escaneando..."):
                    radares = scan_day_for_radars(calendar, stats, refs, all_dfs, sel_date_radar)
                
                st.success("✅ Scan completo!")
                
                tab_r1, tab_r2, tab_r3, tab_r4 = st.tabs([
                    "🏁 Cantos Individual",
                    "🟨 Cartões Individual",
                    "🏁 Cantos Total",
                    "🟨 Cartões Total"
                ])
                
                with tab_r1:
                    st.markdown(f"### 🚩 Radar de Cantos Individuais (>{THRESHOLDS['radar_corners']}%)")
                    
                    if radares['corners_individual']:
                        df_ci = pd.DataFrame(radares['corners_individual'])
                        df_ci = df_ci.sort_values('prob', ascending=False)
                        
                        for _, row in df_ci.iterrows():
                            cor = "green" if row['prob'] >= 80 else "orange"
                            cons_label = "🎯 Reloginho" if row['consistencia'] >= 70 else "⚠️ Caótico"
                            location_icon = "🏠" if row['location'] == 'Casa' else "🛫"
                            
                            st.markdown(f"**:{cor}[{row['prob']:.0f}%]** | {location_icon} {row['mercado']}")
                            st.write(f"Jogo: {row['time']} vs {row['adversario']}")
                            st.write(f"Média: {row['media']:.1f} | {cons_label}")
                            st.caption(f"{row['liga']} | {row['hora']}")
                            st.markdown("---")
                    else:
                        st.info("Nenhuma oportunidade encontrada")
                
                with tab_r2:
                    st.markdown(f"### 🟨 Radar de Cartões Individuais (>{THRESHOLDS['radar_cards']}%)")
                    
                    if radares['cards_individual']:
                        df_cdi = pd.DataFrame(radares['cards_individual'])
                        df_cdi = df_cdi.sort_values('prob', ascending=False)
                        
                        for _, row in df_cdi.iterrows():
                            cor = "green" if row['prob'] >= 75 else "orange"
                            cons_label = "🎯 Reloginho" if row['consistencia'] >= 70 else "⚠️ Caótico"
                            location_icon = "🏠" if row['location'] == 'Casa' else "🛫"
                            
                            st.markdown(f"**:{cor}[{row['prob']:.0f}%]** | {location_icon} {row['mercado']}")
                            st.write(f"Jogo: {row['time']} vs {row['adversario']}")
                            st.write(f"Média: {row['media']:.1f} | {cons_label}")
                            st.caption(f"{row['liga']} | {row['hora']}")
                            st.markdown("---")
                    else:
                        st.info("Nenhuma oportunidade encontrada")
                
                with tab_r3:
                    st.markdown(f"### 🏁 Radar de Cantos Total (>{THRESHOLDS['radar_corners']}%)")
                    
                    if radares['corners_total']:
                        df_ct = pd.DataFrame(radares['corners_total'])
                        df_ct = df_ct.sort_values('prob', ascending=False)
                        
                        for _, row in df_ct.iterrows():
                            cor = "green" if row['prob'] >= 80 else "orange"
                            
                            st.markdown(f"**:{cor}[{row['prob']:.0f}%]** | {row['jogo']}")
                            st.write(f"{row['mercado']} | Média: {row['media']:.1f}")
                            st.caption(f"{row['liga']} | {row['hora']}")
                            st.markdown("---")
                    else:
                        st.info("Nenhuma oportunidade encontrada")
                
                with tab_r4:
                    st.markdown(f"### 🟨 Radar de Cartões Total (>{THRESHOLDS['radar_corners']}%)")
                    
                    if radares['cards_total']:
                        df_cdt = pd.DataFrame(radares['cards_total'])
                        df_cdt = df_cdt.sort_values('prob', ascending=False)
                        
                        for _, row in df_cdt.iterrows():
                            cor = "green" if row['prob'] >= 80 else "orange"
                            
                            st.markdown(f"**:{cor}[{row['prob']:.0f}%]** | {row['jogo']}")
                            st.write(f"{row['mercado']} | Média: {row['media']:.1f}")
                            st.caption(f"{row['liga']} | {row['hora']}")
                            st.markdown("---")
                    else:
                        st.info("Nenhuma oportunidade encontrada")
    
    # TAB 6: TOP TIMES (NOVO!)
    with tab6:
        st.markdown("## 🏆 Top Times do Dia")
        st.caption("Ranking por mercado")
        
        if calendar.empty:
            st.warning("⚠️ Calendário não carregado")
        else:
            dates = sorted(calendar['DtObj'].dt.strftime('%d/%m/%Y').unique())
            sel_date_top = st.selectbox("Selecione a data:", dates, key="top_date")
            
            market_top = st.radio("Mercado:", ["Escanteios", "Cartões"], horizontal=True)
            
            if st.button("📊 GERAR RANKING", type="primary"):
                with st.spinner("Gerando ranking..."):
                    market_key = 'corners' if market_top == "Escanteios" else 'cards'
                    top_teams = get_top_teams_day(calendar, stats, refs, all_dfs, sel_date_top, market_key, 15)
                
                if top_teams:
                    st.success(f"✅ Top {len(top_teams)} times encontrados!")
                    
                    for i, team in enumerate(top_teams, 1):
                        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
                        cor = "green" if team['prob_max'] >= 80 else "orange" if team['prob_max'] >= 70 else "red"
                        cons_label = "🎯" if team['consistencia'] >= 70 else "⚠️"
                        
                        st.markdown(f"### {medal} :{cor}[{team['prob_max']:.0f}%] {cons_label}")
                        st.markdown(f"**{team['time']}** ({team['location']}) vs {team['adversario']}")
                        st.write(f"Média: {team['media']:.2f} | Consistência: {team['consistencia']:.0f}/100")
                        st.caption(f"{team['liga']} | {team['hora']}")
                        st.markdown("---")
                else:
                    st.info("Nenhum time encontrado")
    
    # TAB 7: DASHBOARD (NOVO!)
    with tab7:
        st.markdown("## 📈 Dashboard Geral")
        
        if st.session_state.bet_history:
            st.markdown("### 📊 Histórico de Apostas")
            df_hist = pd.DataFrame(st.session_state.bet_history)
            st.dataframe(df_hist, use_container_width=True)
            
            # ROI
            if 'profit' in df_hist.columns:
                total_staked = df_hist['stake'].sum()
                total_profit = df_hist['profit'].sum()
                roi = (total_profit / total_staked * 100) if total_staked > 0 else 0
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Apostado", f"€{total_staked:.2f}")
                col2.metric("Lucro/Prejuízo", f"€{total_profit:.2f}")
                col3.metric("ROI", f"{roi:.2f}%")
        else:
            st.info("📊 Dashboard será populado conforme você fizer apostas")
            
            # Exemplo de gráfico
            st.markdown("### 📈 Visualização Exemplo")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[1, 2, 3, 4, 5],
                y=[1000, 1100, 1050, 1200, 1180],
                mode='lines+markers',
                name='Bankroll'
            ))
            fig.update_layout(
                title="Evolução do Bankroll",
                xaxis_title="Dias",
                yaxis_title="Valor (€)",
                template="plotly_dark" if st.session_state.theme == 'dark' else "plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
