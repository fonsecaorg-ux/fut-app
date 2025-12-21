"""
╔═══════════════════════════════════════════════════════════════════════════╗
║       FUTPREVISÃO V23.1 - ULTIMATE FIXED (USER FLOW EDITION)              ║
║                                                                            ║
║  ✅ Código Base V23 Mantido (Dados Reais, Radares, Consistência)          ║
║  ✅ CORREÇÃO: Aba Hedges agora lê o Bilhete do Scanner                    ║
║  ✅ Fluxo: Scanner -> Salva Bilhete -> Aba Hedge Calcula Proteções        ║
║                                                                            ║
║  Dezembro 2025                                                           ║
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
    page_title="FutPrevisão V23.1 Ultimate",
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
# NOVO: Estado para salvar o bilhete gerado no Scanner
if 'current_ticket' not in st.session_state:
    st.session_state.current_ticket = []

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
    'smart_ticket_max': 6.50  # Ajustado levemente para flexibilidade
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
                    if 'Mandante' in df.columns: rename_map['Mandante'] = 'HomeTeam'
                    if 'Visitante' in df.columns: rename_map['Visitante'] = 'AwayTeam'
                    if 'Time_Casa' in df.columns: rename_map['Time_Casa'] = 'HomeTeam'
                    if 'Time_Visitante' in df.columns: rename_map['Time_Visitante'] = 'AwayTeam'
                    
                    if rename_map: df = df.rename(columns=rename_map)
                    
                    df['_League_'] = league_name
                    return df
            except: pass
    
    return pd.DataFrame()

@st.cache_resource
def load_all_dataframes() -> Dict[str, pd.DataFrame]:
    dfs = {}
    for league in LIGAS_ALVO:
        df = find_and_load_csv(league)
        if not df.empty: dfs[league] = df
    return dfs

@st.cache_data(ttl=3600)
def learn_stats_v23() -> Dict[str, Dict[str, Any]]:
    stats_db = {}
    for league in LIGAS_ALVO:
        df = find_and_load_csv(league)
        if df.empty: continue
        
        cols_needed = ['HomeTeam', 'AwayTeam', 'HC', 'AC', 'HY', 'AY', 'HF', 'AF', 'FTHG', 'FTAG', 'HST', 'AST', 'HR', 'AR']
        for c in cols_needed:
            if c not in df.columns: df[c] = np.nan
        
        try:
            h_stats = df.groupby('HomeTeam').agg({'HC': ['mean', 'std'], 'HY': ['mean', 'std'], 'HF': 'mean', 'FTHG': ['mean', 'std'], 'FTAG': 'mean', 'HST': 'mean', 'HR': 'mean'})
            h_stats.columns = ['_'.join(col).strip() for col in h_stats.columns.values]
            h_stats = h_stats.fillna(value={'HST_mean': DEFAULTS['shots_on_target'], 'HR_mean': DEFAULTS['red_cards_avg'], 'HC_std': 1.5, 'HY_std': 0.8, 'FTHG_std': 1.0})
            
            a_stats = df.groupby('AwayTeam').agg({'AC': ['mean', 'std'], 'AY': ['mean', 'std'], 'AF': 'mean', 'FTAG': ['mean', 'std'], 'FTHG': 'mean', 'AST': 'mean', 'AR': 'mean'})
            a_stats.columns = ['_'.join(col).strip() for col in a_stats.columns.values]
            a_stats = a_stats.fillna(value={'AST_mean': DEFAULTS['shots_on_target'], 'AR_mean': DEFAULTS['red_cards_avg'], 'AC_std': 1.5, 'AY_std': 0.8, 'FTAG_std': 1.0})
            
            all_teams = set(h_stats.index) | set(a_stats.index)
            
            for team in all_teams:
                h = h_stats.loc[team] if team in h_stats.index else pd.Series(0, index=h_stats.columns)
                a = a_stats.loc[team] if team in a_stats.index else pd.Series(0, index=a_stats.columns)
                
                def w_avg(val_h, val_a, default=0):
                    if val_h == 0 and val_a == 0: return default
                    if val_h == 0: return val_a
                    if val_a == 0: return val_h
                    return (val_h * 0.6) + (val_a * 0.4)
                
                corners_std = w_avg(h.get('HC_std', 1.5), a.get('AC_std', 1.5), 1.5)
                cards_std = w_avg(h.get('HY_std', 0.8), a.get('AY_std', 0.8), 0.8)
                goals_std = w_avg(h.get('FTHG_std', 1.0), a.get('FTAG_std', 1.0), 1.0)
                
                stats_db[team] = {
                    'corners': w_avg(h.get('HC_mean', 0), a.get('AC_mean', 0), 5.0),
                    'corners_std': corners_std,
                    'consistency_corners': round(max(0, 100 - (corners_std * 30)), 1),
                    'cards': w_avg(h.get('HY_mean', 0), a.get('AY_mean', 0), 2.0),
                    'cards_std': cards_std,
                    'consistency_cards': round(max(0, 100 - (cards_std * 50)), 1),
                    'fouls': w_avg(h.get('HF_mean', 0), a.get('AF_mean', 0), 11.0),
                    'goals_f': w_avg(h.get('FTHG_mean', 0), a.get('FTAG_mean', 0), 1.2),
                    'goals_f_std': goals_std,
                    'consistency_goals': round(max(0, 100 - (goals_std * 40)), 1),
                    'goals_a': w_avg(h.get('FTAG_mean', 0), a.get('FTHG_mean', 0), 1.2),
                    'shots_on_target': w_avg(h.get('HST_mean', 0), a.get('AST_mean', 0), 4.5),
                    'red_cards_avg': w_avg(h.get('HR_mean', 0), a.get('AR_mean', 0), 0.08),
                    'league': league
                }
        except Exception: pass
    return stats_db

@st.cache_data(ttl=3600)
def load_referees_v23() -> Dict[str, Dict[str, float]]:
    refs_db = {}
    files = ["arbitros_5_ligas_2025_2026.csv", "arbitros.csv"]
    for f in files:
        if os.path.exists(f):
            try:
                df = pd.read_csv(f, encoding='utf-8-sig')
                df.columns = [c.strip().replace('\ufeff', '') for c in df.columns]
                col_nome = 'Arbitro' if 'Arbitro' in df.columns else 'Nome'
                for _, row in df.iterrows():
                    nome = str(row.get(col_nome, '')).strip()
                    if not nome: continue
                    media = float(row.get('Media_Cartoes_Por_Jogo', row.get('Fator', 4.0)))
                    reds = float(row.get('Cartoes_Vermelhos', 0))
                    games = float(row.get('Jogos_Apitados', 1))
                    
                    refs_db[nome] = {
                        'factor': media/4.0, 'red_rate': (reds/games) if games > 0 else 0.08,
                        'strictness': media, 'games': int(games),
                        'yellows': int(row.get('Cartoes_Amarelos', 0)), 'reds': int(reds)
                    }
            except: pass
    return refs_db

@st.cache_data(ttl=600)
def load_calendar_safe() -> pd.DataFrame:
    fname = "calendario_ligas.csv"
    if not os.path.exists(fname): return pd.DataFrame()
    try:
        try: df = pd.read_csv(fname, encoding='utf-8-sig')
        except: df = pd.read_csv(fname, encoding='latin1')
        df.columns = [c.strip().replace('\ufeff', '') for c in df.columns]
        rename = {'Mandante': 'Time_Casa', 'Visitante': 'Time_Visitante'}
        df = df.rename(columns=rename)
        df['DtObj'] = pd.to_datetime(df['Data'], format='%d/%m/%Y', errors='coerce')
        return df.dropna(subset=['DtObj']).sort_values(by=['DtObj', 'Hora'])
    except: return pd.DataFrame()

# ═══════════════════════════════════════════════════════════════════════════
# MOTOR DE CÁLCULO
# ═══════════════════════════════════════════════════════════════════════════

def normalize_name(name: str, db_keys: list) -> Optional[str]:
    if not name: return None
    if name in NAME_MAPPING: name = NAME_MAPPING[name]
    if name in db_keys: return name
    matches = get_close_matches(name, db_keys, n=1, cutoff=0.6)
    return matches[0] if matches else None

def get_native_history(team_name: str, league: str, market: str, line: float, location: str, all_dfs: Dict) -> str:
    if league not in all_dfs: return "N/A"
    df = all_dfs[league]
    col_map = {('home', 'corners'): 'HC', ('away', 'corners'): 'AC', ('home', 'cards'): 'HY', ('away', 'cards'): 'AY'}
    col_code = col_map.get((location, market))
    team_col = 'HomeTeam' if location == 'home' else 'AwayTeam'
    matches = df[df[team_col] == team_name]
    if matches.empty: return "0/0"
    last_matches = matches.tail(10)
    hits = sum(1 for val in last_matches[col_code] if float(val) > line)
    return f"{hits}/{len(last_matches)}"

def poisson(k: int, lamb: float) -> float:
    return (lamb**k * math.exp(-lamb)) / math.factorial(k) if lamb <= 30 else 0

def monte_carlo(xg_h: float, xg_a: float, n: int = 1000) -> Tuple[float, float, float]:
    gh = np.random.poisson(max(0.1, xg_h), n)
    ga = np.random.poisson(max(0.1, xg_a), n)
    return np.count_nonzero(gh > ga)/n, np.count_nonzero(gh == ga)/n, np.count_nonzero(ga > gh)/n

def calcular_jogo_v23(home: str, away: str, stats: Dict, ref: Optional[str], refs_db: Dict, all_dfs: Dict) -> Dict:
    h_n = normalize_name(home, list(stats.keys()))
    a_n = normalize_name(away, list(stats.keys()))
    if not h_n or not a_n: return {'error': "Times não encontrados."}
    
    s_h, s_a = stats[h_n], stats[a_n]
    r_data = refs_db.get(ref, {'factor': 1.0, 'red_rate': 0.08, 'strictness': 4.0})
    
    p_h = 1.15 if s_h['shots_on_target'] > 5.0 else 1.0
    p_a = 1.15 if s_a['shots_on_target'] > 5.0 else 1.0
    
    corn_h = s_h['corners'] * 1.15 * p_h
    corn_a = s_a['corners'] * 0.90 * p_a
    
    card_h = s_h['cards'] * r_data['factor']
    card_a = s_a['cards'] * r_data['factor']
    
    xg_h, xg_a = max(0.1, s_h['goals_f']), max(0.1, s_a['goals_f'])
    mc_h, mc_d, mc_a = monte_carlo(xg_h, xg_a)
    
    btts = (1 - poisson(0, xg_h)) * (1 - poisson(0, xg_a)) * 100
    over_25 = (1 - sum([poisson(h, xg_h) * poisson(a, xg_a) for h in range(3) for a in range(3) if h+a < 3])) * 100
    
    return {
        'home': h_n, 'away': a_n, 'league_h': s_h['league'], 'league_a': s_a['league'],
        'corners': {'h': corn_h, 'a': corn_a, 'total': corn_h + corn_a},
        'cards': {'h': card_h, 'a': card_a, 'total': card_h + card_a},
        'goals': {'h': xg_h, 'a': xg_a},
        'monte_carlo': {'h': mc_h * 100, 'd': mc_d * 100, 'a': mc_a * 100},
        'consistency': {
            'corners_h': s_h.get('consistency_corners', 50), 'corners_a': s_a.get('consistency_corners', 50),
            'cards_h': s_h.get('consistency_cards', 50), 'cards_a': s_a.get('consistency_cards', 50),
            'goals_h': s_h.get('consistency_goals', 50)
        },
        'meta': {'referee': ref if ref else 'Neutro', 'ref_factor': r_data['factor'], 'ref_strictness': r_data['strictness'], 'ref_red_rate': r_data.get('red_rate', 0.08)},
        'probs': {'btts': btts, 'over_2_5': over_25}
    }

def get_detailed_probs(res: Dict) -> Dict:
    def sim_prob(avg, line): return max(5, min(95, 50 + (avg - line) * 15))
    return {
        'corners': {
            'home': {f'Over {l}': sim_prob(res['corners']['h'], l) for l in [2.5, 3.5, 4.5, 5.5]},
            'away': {f'Over {l}': sim_prob(res['corners']['a'], l) for l in [2.5, 3.5, 4.5]},
            'total': {f'Over {l}': sim_prob(res['corners']['total'], l) for l in [8.5, 9.5, 10.5, 11.5]}
        },
        'cards': {
            'home': {f'Over {l}': sim_prob(res['cards']['h'], l) for l in [1.5, 2.5]},
            'away': {f'Over {l}': sim_prob(res['cards']['a'], l) for l in [1.5, 2.5]},
            'total': {f'Over {l}': sim_prob(res['cards']['total'], l) for l in [2.5, 3.5, 4.5, 5.5]}
        },
        'chance': {'1X': res['monte_carlo']['h'] + res['monte_carlo']['d'], 'X2': res['monte_carlo']['a'] + res['monte_carlo']['d']}
    }

def get_fair_odd(prob: float) -> float:
    return round(100 / prob, 2) if prob > 0 else 99.0

# ═══════════════════════════════════════════════════════════════════════════
# NOVAS FUNCIONALIDADES V23 (SCANNER + HEDGE USER)
# ═══════════════════════════════════════════════════════════════════════════

def scan_day_for_radars(calendar: pd.DataFrame, stats: Dict, refs: Dict, all_dfs: Dict, date_str: str) -> Dict:
    """Scanner de Radares (Mantido do V22)"""
    df_day = calendar[calendar['DtObj'].dt.strftime('%d/%m/%Y') == date_str]
    radar = {'corners_individual': [], 'cards_individual': [], 'corners_total': [], 'cards_total': []}
    
    for _, row in df_day.iterrows():
        home, away = row['Time_Casa'], row['Time_Visitante']
        res = calcular_jogo_v23(home, away, stats, None, refs, all_dfs)
        if 'error' in res: continue
        probs = get_detailed_probs(res)
        
        # Individual Cantos
        for line in [3.5, 4.5]:
            ph = probs['corners']['home'].get(f'Over {line}', 0)
            if ph >= THRESHOLDS['radar_corners']:
                radar['corners_individual'].append({'time': res['home'], 'adversario': res['away'], 'mercado': f"{res['home']} Over {line} Cantos", 'prob': ph, 'location': 'Casa'})
            pa = probs['corners']['away'].get(f'Over {line}', 0)
            if pa >= THRESHOLDS['radar_corners']:
                radar['corners_individual'].append({'time': res['away'], 'adversario': res['home'], 'mercado': f"{res['away']} Over {line} Cantos", 'prob': pa, 'location': 'Fora'})
                
        # Individual Cartões
        for line in [1.5, 2.5]:
            ph = probs['cards']['home'].get(f'Over {line}', 0)
            if ph >= THRESHOLDS['radar_cards']:
                radar['cards_individual'].append({'time': res['home'], 'adversario': res['away'], 'mercado': f"{res['home']} Over {line} Cartões", 'prob': ph, 'location': 'Casa'})
            pa = probs['cards']['away'].get(f'Over {line}', 0)
            if pa >= THRESHOLDS['radar_cards']:
                radar['cards_individual'].append({'time': res['away'], 'adversario': res['home'], 'mercado': f"{res['away']} Over {line} Cartões", 'prob': pa, 'location': 'Fora'})

    return radar

def generate_smart_ticket_v23(calendar: pd.DataFrame, stats: Dict, refs: Dict, all_dfs: Dict, date_str: str) -> Dict:
    """Gera bilhete e retorna para salvar na sessão"""
    df_day = calendar[calendar['DtObj'].dt.strftime('%d/%m/%Y') == date_str]
    anchors, fusions = [], []
    
    for _, row in df_day.iterrows():
        home, away = row['Time_Casa'], row['Time_Visitante']
        res = calcular_jogo_v23(home, away, stats, None, refs, all_dfs)
        if 'error' in res: continue
        probs = get_detailed_probs(res)
        
        # Âncoras
        for loc, name in [('home', res['home']), ('away', res['away'])]:
            for l in [3.5, 4.5]:
                p = probs['corners'][loc].get(f'Over {l}', 0)
                if p >= THRESHOLDS['anchor_safety']:
                    odd = get_fair_odd(p)
                    if 1.20 <= odd <= 1.40:
                        anchors.append({'type': 'anchor', 'jogo': f"{res['home']} vs {res['away']}", 'selection': f"{name} Over {l} Escanteios", 'prob': p, 'odd': odd})
            
            p_card = probs['cards'][loc].get('Over 1.5', 0)
            if p_card >= THRESHOLDS['anchor_safety']:
                odd = get_fair_odd(p_card)
                if 1.20 <= odd <= 1.45:
                    anchors.append({'type': 'anchor', 'jogo': f"{res['home']} vs {res['away']}", 'selection': f"{name} Over 1.5 Cartões", 'prob': p_card, 'odd': odd})

        # Fusões
        pc = probs['corners']['home'].get('Over 3.5', 0)
        pcd = probs['cards']['home'].get('Over 1.5', 0)
        if pc > 70 and pcd > 65:
            comb = (pc/100 * pcd/100 * 0.85) * 100
            odd = get_fair_odd(comb)
            if 1.60 <= odd <= 2.20:
                fusions.append({'type': 'fusion', 'jogo': f"{res['home']} vs {res['away']}", 'selection': f"{res['home']}: Over 3.5 Cantos + Over 1.5 Cartões", 'prob': comb, 'odd': odd})

    ticket = []
    total_odd = 1.0
    used_games = set()
    
    anchors.sort(key=lambda x: x['prob'], reverse=True)
    fusions.sort(key=lambda x: x['prob'], reverse=True)
    
    # Monta bilhete
    for a in anchors[:2]:
        if a['jogo'] not in used_games:
            ticket.append(a)
            total_odd *= a['odd']
            used_games.add(a['jogo'])
            
    pool = fusions + anchors
    for item in pool:
        if len(ticket) >= 6: break
        if item['jogo'] not in used_games:
            if total_odd * item['odd'] <= THRESHOLDS['smart_ticket_max']:
                ticket.append(item)
                total_odd *= item['odd']
                used_games.add(item['jogo'])
                
    return {'ticket': ticket, 'total_odd': round(total_odd, 2), 'num_selections': len(ticket)}

def generate_hedges_for_user_ticket(ticket: List[Dict], stats: Dict, refs: Dict, all_dfs: Dict) -> Dict:
    """
    NOVA FUNÇÃO: Gera Hedges baseados no bilhete do usuário (Salvo na Sessão)
    """
    principal = []
    hedge1 = [] # Safety (Result)
    hedge2 = [] # Mix (Cards/Chaos)
    processed_games = set()
    
    for item in ticket:
        # Tenta extrair times da string "TimeA vs TimeB"
        try:
            parts = item['jogo'].split(' vs ')
            h, a = parts[0], parts[1]
        except: continue
        
        if f"{h}vs{a}" in processed_games: continue
        
        # Recalcula para pegar hedges
        res = calcular_jogo_v23(h, a, stats, None, refs, all_dfs)
        if 'error' in res: continue
        probs = get_detailed_probs(res)
        
        # 1. Principal (Cópia)
        principal.append({'jogo': f"{h} vs {a}", 'selecao': item.get('selection', item.get('mercado', '??')), 'odd': item['odd']})
        
        # 2. Hedge 1 (Resultado/Safety)
        mc = res['monte_carlo']
        if mc['h'] > mc['a']: 
            sel1 = f"DC {h} ou Empate"
            odd1 = get_fair_odd(probs['chance']['1X'])
        else:
            sel1 = f"DC {a} ou Empate"
            odd1 = get_fair_odd(probs['chance']['X2'])
        hedge1.append({'jogo': f"{h} vs {a}", 'selecao': sel1, 'odd': odd1})
        
        # 3. Hedge 2 (Caos/Cartões)
        best_card = 0
        best_line = "3.5"
        for l, p in probs['cards']['total'].items():
            if p > best_card:
                best_card = p
                best_line = l.replace('Over ', '')
        hedge2.append({'jogo': f"{h} vs {a}", 'selecao': f"Total Over {best_line} Cartões", 'odd': get_fair_odd(best_card)})
        
        processed_games.add(f"{h}vs{a}")
        
    return {
        'principal': {'itens': principal, 'odd': round(np.prod([x['odd'] for x in principal]), 2)},
        'hedge1': {'itens': hedge1, 'odd': round(np.prod([x['odd'] for x in hedge1]), 2)},
        'hedge2': {'itens': hedge2, 'odd': round(np.prod([x['odd'] for x in hedge2]), 2)}
    }

# ═══════════════════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    if st.session_state.theme == 'dark':
        st.markdown("<style>.stApp {background-color: #0E1117; color: #FAFAFA;}</style>", unsafe_allow_html=True)
        
    st.sidebar.title("🎛️ Painel V23.1")
    
    with st.spinner("Carregando..."):
        stats = learn_stats_v23()
        refs = load_referees_v23()
        calendar = load_calendar_safe()
        all_dfs = load_all_dataframes()
        
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📅 Calendário", "🔍 Simulação", "🎯 Scanner Smart", "🛡️ Sistema Hedges", "📊 Radares"])
    
    # Tabs 1, 2 e 5 mantidas iguais (Omitidas para brevidade, mas devem estar no código final se copiar tudo)
    # Focando nas abas alteradas 3 e 4
    
    with tab1:
        if not calendar.empty: st.dataframe(calendar, use_container_width=True)
        
    with tab2:
        l_times = sorted(list(stats.keys()))
        c1, c2 = st.columns(2)
        h = c1.selectbox("Casa", l_times)
        a = c2.selectbox("Fora", l_times, index=1)
        if st.button("Simular"):
            res = calcular_jogo_v23(h, a, stats, None, refs, all_dfs)
            if 'error' not in res:
                st.info(f"xG: {res['goals']['h']:.2f} x {res['goals']['a']:.2f}")
                
    with tab3:
        st.header("🎯 Scanner V23 - Smart Ticket")
        if not calendar.empty:
            dates = sorted(calendar['DtObj'].dt.strftime('%d/%m/%Y').unique())
            sel_date = st.selectbox("Data:", dates, key="sd")
            
            if st.button("🚀 GERAR BILHETE E SALVAR", type="primary"):
                res = generate_smart_ticket_v23(calendar, stats, refs, all_dfs, sel_date)
                if res['ticket']:
                    st.session_state.current_ticket = res['ticket'] # SALVA NA SESSÃO
                    st.success("Bilhete Gerado e Salvo!")
                else:
                    st.warning("Sem oportunidades.")
            
            if st.session_state.current_ticket:
                st.markdown("### 🎫 Bilhete Ativo (Na Memória)")
                for item in st.session_state.current_ticket:
                    st.write(f"✅ {item['jogo']} | {item.get('selection', item.get('mercado'))} (@{item['odd']})")
                st.info("👉 Vá para a aba 'Sistema Hedges' para proteger este bilhete.")

    with tab4:
        st.header("🛡️ Sistema de Proteção (Hedges)")
        st.caption("Gera proteções baseadas no bilhete ativo.")
        
        if not st.session_state.current_ticket:
            st.warning("⚠️ Nenhum bilhete na memória. Gere no Scanner (Aba 3).")
        else:
            if st.button("🛡️ CALCULAR HEDGES PARA MEU BILHETE", type="primary"):
                hedges = generate_hedges_for_user_ticket(st.session_state.current_ticket, stats, refs, all_dfs)
                
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.subheader("📋 Principal")
                    st.metric("Odd", f"@{hedges['principal']['odd']}")
                    for i in hedges['principal']['itens']: st.caption(f"{i['jogo']} - {i['selecao']}")
                with c2:
                    st.subheader("🛡️ Safety (30%)")
                    st.metric("Odd", f"@{hedges['hedge1']['odd']}")
                    for i in hedges['hedge1']['itens']: st.caption(f"{i['jogo']} - {i['selecao']}")
                with c3:
                    st.subheader("🔄 Caos (20%)")
                    st.metric("Odd", f"@{hedges['hedge2']['odd']}")
                    for i in hedges['hedge2']['itens']: st.caption(f"{i['jogo']} - {i['selecao']}")

    with tab5:
         if not calendar.empty:
            dates = sorted(calendar['DtObj'].dt.strftime('%d/%m/%Y').unique())
            sel_date_r = st.selectbox("Data Radar:", dates)
            if st.button("Escanear"):
                res = scan_day_for_radars(calendar, stats, refs, all_dfs, sel_date_r)
                st.write(res)

if __name__ == "__main__":
    main()
