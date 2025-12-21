"""
╔═══════════════════════════════════════════════════════════════════════════╗
║       FUTPREVISÃO V25.7 - WHITE THEME & UI FIX                            ║
║                                                                            ║
║  ✅ TEMA: Tela Branca (Modo Claro ativado por padrão)                     ║
║  ✅ FIX MENU: Seletor não reseta mais ao clicar (Adicionado key única)    ║
║  ✅ ODDS CALIBRADAS: Visitante Over 1.5 Cartões fixado em ~1.45           ║
║  ✅ INTEGRIDADE: Prevenção de duplicatas no menu manual                   ║
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
from typing import Dict, List, Any, Optional, Tuple
from difflib import get_close_matches

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURAÇÃO
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="FutPrevisão V25.7 White",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session State
if 'bankroll' not in st.session_state: st.session_state.bankroll = 1000.0
if 'current_ticket' not in st.session_state: st.session_state.current_ticket = []
# TEMA PADRÃO DEFINIDO PARA LIGHT (CLARO)
if 'theme' not in st.session_state: st.session_state.theme = 'light'
if 'bet_history' not in st.session_state: st.session_state.bet_history = []

# Mapeamentos
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

# TABELA DE PREÇOS REAIS (V25.7)
REAL_ODDS = {
    # Mandante Escanteios
    ('home', 'corners', 3.5): 1.34,
    ('home', 'corners', 4.5): 1.62,
    ('home', 'corners', 5.5): 2.10,
    
    # Visitante Escanteios
    ('away', 'corners', 2.5): 1.40,
    ('away', 'corners', 3.5): 1.75,
    ('away', 'corners', 4.5): 2.50,
    
    # Mandante Cartões
    ('home', 'cards', 0.5): 1.20,
    ('home', 'cards', 1.5): 1.52,
    ('home', 'cards', 2.5): 2.25,
    
    # Visitante Cartões (CALIBRADO @1.45)
    ('away', 'cards', 0.5): 1.18,
    ('away', 'cards', 1.5): 1.45,  
    ('away', 'cards', 2.5): 2.30,
    
    # Totais Escanteios
    ('total', 'corners', 7.5): 1.35,
    ('total', 'corners', 8.5): 1.57,
    ('total', 'corners', 9.5): 1.90,
    ('total', 'corners', 10.5): 2.30,
    
    # Totais Cartões
    ('total', 'cards', 2.5): 1.38,
    ('total', 'cards', 3.5): 1.65,
    ('total', 'cards', 4.5): 2.10,
    
    # Dupla Chance
    ('home', 'dc'): 1.25,
    ('away', 'dc'): 1.60
}

# ═══════════════════════════════════════════════════════════════════════════
# CARREGAMENTO DE DADOS
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def find_and_load_csv(league_name: str) -> pd.DataFrame:
    attempts = [f"{league_name} 25.26.csv", f"{league_name.replace(' ', '_')}_25_26.csv", f"{league_name}.csv"]
    if "Süper Lig" in league_name: attempts.extend(["Super Lig Turquia 25.26.csv"])
    
    for filename in attempts:
        if os.path.exists(filename):
            try:
                try: df = pd.read_csv(filename, encoding='utf-8-sig')
                except: df = pd.read_csv(filename, encoding='latin1')
                if not df.empty:
                    df.columns = [c.strip().replace('\ufeff', '') for c in df.columns]
                    rename_map = {'Mandante': 'HomeTeam', 'Visitante': 'AwayTeam', 'Time_Casa': 'HomeTeam', 'Time_Visitante': 'AwayTeam'}
                    df = df.rename(columns=rename_map)
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
        for c in ['HomeTeam', 'AwayTeam', 'HC', 'AC', 'HY', 'AY', 'HF', 'AF', 'FTHG', 'FTAG', 'HST', 'AST', 'HR', 'AR']:
            if c not in df.columns: df[c] = np.nan
            
        try:
            h_stats = df.groupby('HomeTeam').agg({'HC': ['mean','std'], 'HY': ['mean','std'], 'HF': 'mean', 'FTHG': ['mean','std'], 'FTAG': 'mean', 'HST': 'mean', 'HR': 'mean'})
            a_stats = df.groupby('AwayTeam').agg({'AC': ['mean','std'], 'AY': ['mean','std'], 'AF': 'mean', 'FTAG': ['mean','std'], 'FTHG': 'mean', 'AST': 'mean', 'AR': 'mean'})
            h_stats.columns = ['_'.join(col).strip() for col in h_stats.columns.values]
            a_stats.columns = ['_'.join(col).strip() for col in a_stats.columns.values]
            h_stats = h_stats.fillna(value={'HST_mean': 4.5, 'HR_mean': 0.08, 'HC_std': 1.5, 'HY_std': 0.8, 'FTHG_std': 1.0})
            a_stats = a_stats.fillna(value={'AST_mean': 4.5, 'AR_mean': 0.08, 'AC_std': 1.5, 'AY_std': 0.8, 'FTAG_std': 1.0})
            
            for team in set(h_stats.index) | set(a_stats.index):
                h = h_stats.loc[team] if team in h_stats.index else pd.Series(0, index=h_stats.columns)
                a = a_stats.loc[team] if team in a_stats.index else pd.Series(0, index=a_stats.columns)
                def w_avg(val_h, val_a, default=0): return (val_h * 0.6 + val_a * 0.4) if (val_h+val_a) > 0 else default
                
                stats_db[team] = {
                    'corners': w_avg(h.get('HC_mean',0), a.get('AC_mean',0), 5.0),
                    'cards': w_avg(h.get('HY_mean',0), a.get('AY_mean',0), 2.0),
                    'goals_f': w_avg(h.get('FTHG_mean',0), a.get('FTAG_mean',0), 1.2),
                    'goals_a': w_avg(h.get('FTAG_mean',0), a.get('FTHG_mean',0), 1.2),
                    'shots_on_target': w_avg(h.get('HST_mean',0), a.get('AST_mean',0), 4.5),
                    'league': league
                }
        except: pass
    return stats_db

@st.cache_data(ttl=3600)
def load_referees_v23() -> Dict:
    refs_db = {}
    if os.path.exists("arbitros_5_ligas_2025_2026.csv"):
        try:
            df = pd.read_csv("arbitros_5_ligas_2025_2026.csv", encoding='utf-8-sig')
            df.columns = [c.strip().replace('\ufeff', '') for c in df.columns]
            for _, row in df.iterrows():
                nome = str(row.get('Arbitro', '')).strip()
                if nome:
                    media = float(row.get('Media_Cartoes_Por_Jogo', 4.0))
                    refs_db[nome] = {'factor': media/4.0, 'strictness': media}
        except: pass
    return refs_db

@st.cache_data(ttl=600)
def load_calendar_safe() -> pd.DataFrame:
    if not os.path.exists("calendario_ligas.csv"): return pd.DataFrame()
    try:
        try: df = pd.read_csv("calendario_ligas.csv", encoding='utf-8-sig')
        except: df = pd.read_csv("calendario_ligas.csv", encoding='latin1')
        df.columns = [c.strip().replace('\ufeff', '') for c in df.columns]
        df = df.rename(columns={'Mandante': 'Time_Casa', 'Visitante': 'Time_Visitante'})
        df['DtObj'] = pd.to_datetime(df['Data'], format='%d/%m/%Y', errors='coerce')
        return df.dropna(subset=['DtObj']).sort_values(by=['DtObj', 'Hora'])
    except: return pd.DataFrame()

# ═══════════════════════════════════════════════════════════════════════════
# CÁLCULOS
# ═══════════════════════════════════════════════════════════════════════════

def normalize_name(name: str, db_keys: list) -> Optional[str]:
    if not name: return None
    if name in NAME_MAPPING: name = NAME_MAPPING[name]
    if name in db_keys: return name
    matches = get_close_matches(name, db_keys, n=1, cutoff=0.6)
    return matches[0] if matches else None

def get_fair_odd(prob: float) -> float:
    return round(100/prob, 2) if prob > 0 else 99.0

def get_market_price(location: str, market_type: str, line: float, prob_calculated: float) -> float:
    """Busca preço real ou calcula fair odd."""
    real_price = REAL_ODDS.get((location, market_type, line))
    if real_price and prob_calculated >= 40:
        return real_price
    return get_fair_odd(prob_calculated)

def monte_carlo(xg_h: float, xg_a: float, n: int = 1000) -> Tuple[float, float, float]:
    gh = np.random.poisson(max(0.1, xg_h), n)
    ga = np.random.poisson(max(0.1, xg_a), n)
    return np.count_nonzero(gh > ga)/n, np.count_nonzero(gh == ga)/n, np.count_nonzero(ga > gh)/n

def calcular_jogo_v23(home: str, away: str, stats: Dict, ref: Optional[str], refs_db: Dict, all_dfs: Dict) -> Dict:
    h_n = normalize_name(home, list(stats.keys()))
    a_n = normalize_name(away, list(stats.keys()))
    if not h_n or not a_n: return {'error': "Times desconhecidos"}
    
    s_h, s_a = stats[h_n], stats[a_n]
    r_data = refs_db.get(ref, {'factor': 1.0, 'strictness': 4.0})
    
    p_h = 1.15 if s_h['shots_on_target'] > 5.5 else 1.0
    p_a = 1.15 if s_a['shots_on_target'] > 5.5 else 1.0
    
    corn_h = s_h['corners'] * 1.15 * p_h
    corn_a = s_a['corners'] * 0.90 * p_a
    
    card_h = s_h['cards'] * r_data['factor']
    card_a = s_a['cards'] * r_data['factor']
    
    xg_h, xg_a = max(0.1, s_h['goals_f']), max(0.1, s_a['goals_f'])
    mc_h, mc_d, mc_a = monte_carlo(xg_h, xg_a)
    
    return {
        'home': h_n, 'away': a_n, 'league_h': s_h['league'],
        'corners': {'h': corn_h, 'a': corn_a, 'total': corn_h+corn_a},
        'cards': {'h': card_h, 'a': card_a, 'total': card_h+card_a},
        'goals': {'h': xg_h, 'a': xg_a},
        'monte_carlo': {'h': mc_h*100, 'd': mc_d*100, 'a': mc_a*100},
        'probs': {'btts': 50.0, 'over_2_5': 50.0}
    }

def get_detailed_probs(res: Dict) -> Dict:
    def sim_prob(avg: float, line: float) -> float:
        return max(5, min(95, 50 + (avg - line) * 15))
    
    probs = {
        'corners': {
            'home': {f'Over {l}': sim_prob(res['corners']['h'], l) for l in [2.5, 3.5, 4.5, 5.5, 6.5]},
            'away': {f'Over {l}': sim_prob(res['corners']['a'], l) for l in [2.5, 3.5, 4.5]},
            'total': {f'Over {l}': sim_prob(res['corners']['total'], l) for l in [6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5]}
        },
        'cards': {
            'home': {f'Over {l}': sim_prob(res['cards']['h'], l) for l in [0.5, 1.5, 2.5]},
            'away': {f'Over {l}': sim_prob(res['cards']['a'], l) for l in [0.5, 1.5, 2.5]},
            'total': {f'Over {l}': sim_prob(res['cards']['total'], l) for l in [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]}
        }
    }
    mc = res['monte_carlo']
    probs['chance'] = {'1': mc['h'], 'X': mc['d'], '2': mc['a'], '1X': mc['h'] + mc['d'], 'X2': mc['a'] + mc['d']}
    return probs

def get_available_markets_for_game(res: Dict, probs: Dict) -> List[Dict]:
    """Helper para modo manual - LISTA TUDO COM PREÇOS REAIS"""
    markets = []
    
    # --- ESCANTEIOS ---
    for l in [3.5, 4.5, 5.5]:
        p = probs['corners']['home'].get(f'Over {l}', 0)
        odd = get_market_price('home', 'corners', l, p)
        if p > 0: markets.append({'mercado': f"{res['home']} Over {l} Escanteios", 'prob': p, 'odd': odd, 'type': 'Escanteios'})
    
    for l in [2.5, 3.5, 4.5]:
        p = probs['corners']['away'].get(f'Over {l}', 0)
        odd = get_market_price('away', 'corners', l, p)
        if p > 0: markets.append({'mercado': f"{res['away']} Over {l} Escanteios", 'prob': p, 'odd': odd, 'type': 'Escanteios'})
        
    # --- CARTÕES ---
    for l in [1.5, 2.5]:
        p1 = probs['cards']['home'].get(f'Over {l}', 0)
        odd1 = get_market_price('home', 'cards', l, p1)
        if p1 > 0: markets.append({'mercado': f"{res['home']} Over {l} Cartões", 'prob': p1, 'odd': odd1, 'type': 'Cartões'})
        
        p2 = probs['cards']['away'].get(f'Over {l}', 0)
        odd2 = get_market_price('away', 'cards', l, p2)
        if p2 > 0: markets.append({'mercado': f"{res['away']} Over {l} Cartões", 'prob': p2, 'odd': odd2, 'type': 'Cartões'})

    # --- TOTAIS ---
    for l in [7.5, 8.5, 9.5]:
        p = probs['corners']['total'].get(f'Over {l}', 0)
        odd = get_market_price('total', 'corners', l, p)
        if p > 0: markets.append({'mercado': f"Total Jogo Over {l} Escanteios", 'prob': p, 'odd': odd, 'type': 'TotalEscanteios'})
        
    for l in [2.5, 3.5, 4.5]:
        p = probs['cards']['total'].get(f'Over {l}', 0)
        odd = get_market_price('total', 'cards', l, p)
        if p > 0: markets.append({'mercado': f"Total Jogo Over {l} Cartões", 'prob': p, 'odd': odd, 'type': 'TotalCartões'})
        
    # --- DUPLA CHANCE ---
    p1x = probs['chance']['1X']
    odd1x = get_market_price('home', 'dc', 0, p1x)
    markets.append({'mercado': f"DC {res['home']} ou Empate", 'prob': p1x, 'odd': odd1x, 'type': 'DC'})
    
    px2 = probs['chance']['X2']
    oddx2 = get_market_price('away', 'dc', 0, px2)
    markets.append({'mercado': f"DC {res['away']} ou Empate", 'prob': px2, 'odd': oddx2, 'type': 'DC'})

    return markets

# ═══════════════════════════════════════════════════════════════════════════
# LÓGICA V25 - SCANNER E HEDGES
# ═══════════════════════════════════════════════════════════════════════════

def generate_smart_ticket_v23(calendar: pd.DataFrame, stats: Dict, refs: Dict, all_dfs: Dict, date_str: str, 
                              target_leagues: List[str] = None, target_games: List[str] = None) -> Dict:
    """Scanner V25 - DOUBLE LOCK + REAL ODDS"""
    
    df_day = calendar[calendar['DtObj'].dt.strftime('%d/%m/%Y') == date_str].copy()
    if target_leagues: df_day = df_day[df_day['Liga'].isin(target_leagues)]
    df_day['GameID'] = df_day['Time_Casa'] + ' vs ' + df_day['Time_Visitante']
    if target_games: df_day = df_day[df_day['GameID'].isin(target_games)]
    
    ticket = []
    curr_odd = 1.0
    
    if not target_games and len(df_day) > 3: df_day = df_day.head(3)

    for _, row in df_day.iterrows():
        home, away = row['Time_Casa'], row['Time_Visitante']
        liga, hora = row.get('Liga', 'N/A'), row.get('Hora', 'N/A')
        
        res = calcular_jogo_v23(home, away, stats, None, refs, all_dfs)
        if 'error' in res: continue
        probs = get_detailed_probs(res)
        
        candidates = []
        
        # 1. Escanteios
        for loc, name in [('home', res['home']), ('away', res['away'])]:
            for l in [3.5, 4.5]:
                p = probs['corners'][loc].get(f'Over {l}', 0)
                odd = get_market_price(loc, 'corners', l, p)
                if p >= 65: candidates.append({'mercado': f"{name} Over {l} Escanteios", 'prob': p, 'odd': odd, 'type': 'Escanteios'})
        
        # 2. Cartões
        for loc, name in [('home', res['home']), ('away', res['away'])]:
            p = probs['cards'][loc].get('Over 1.5', 0)
            odd = get_market_price(loc, 'cards', 1.5, p)
            if p >= 60: candidates.append({'mercado': f"{name} Over 1.5 Cartões", 'prob': p, 'odd': odd, 'type': 'Cartões'})
        
        # 3. Totais
        p_card_tot = probs['cards']['total'].get('Over 3.5', 0)
        odd_card_tot = get_market_price('total', 'cards', 3.5, p_card_tot)
        if p_card_tot >= 65: candidates.append({'mercado': "Total Jogo Over 3.5 Cartões", 'prob': p_card_tot, 'odd': odd_card_tot, 'type': 'TotalCartões'})

        candidates.sort(key=lambda x: x['prob'], reverse=True)
        selected = []
        types_used = set()
        
        for cand in candidates:
            if len(selected) >= 2: break
            if cand['type'] not in types_used or cand['prob'] > 80:
                if not any(x['mercado'] == cand['mercado'] for x in selected):
                    item = {
                        'type': 'auto_dual', 'jogo': f"{home} vs {away}",
                        'mercado': cand['mercado'], 'prob': cand['prob'], 'odd': cand['odd'],
                        'liga': liga, 'hora': hora
                    }
                    selected.append(item)
                    types_used.add(cand['type'])
        
        for item in selected:
            if curr_odd * item['odd'] <= 80.0:
                ticket.append(item)
                curr_odd *= item['odd']

    return {'ticket': ticket, 'total_odd': round(curr_odd, 2), 'num_selections': len(ticket)}

def generate_hedges_for_user_ticket(ticket: List[Dict], stats: Dict, refs: Dict, all_dfs: Dict) -> Dict:
    """Hedge V25.1 - MIRROR + SAFETY + REAL ODDS"""
    
    games_map = {}
    for item in ticket:
        game_name = item['jogo']
        if game_name not in games_map: games_map[game_name] = []
        games_map[game_name].append(item)
        
    principal_display = []
    hedge1 = []
    hedge2 = []
    processed_games = set()
    
    for game_name, items in games_map.items():
        if game_name in processed_games: continue
        try: parts = game_name.split(' vs '); h, a = parts[0], parts[1]
        except: continue
        
        res = calcular_jogo_v23(h, a, stats, None, refs, all_dfs)
        if 'error' in res: continue
        probs = get_detailed_probs(res)
        
        principal_desc_str = ""
        for it in items:
            desc = it.get('mercado', it.get('selection', 'Aposta'))
            if it.get('type') == 'fusion': desc = " + ".join(it.get('mercados', []))
            principal_display.append({'jogo': it['jogo'], 'selecao': desc, 'odd': it['odd']})
            principal_desc_str += desc + " "
            
        prob_corn_h_react = probs['corners']['home']['Over 4.5']
        prob_corn_a_react = probs['corners']['away']['Over 3.5']
        prob_card_safe = probs['cards']['total']['Over 2.5']
        prob_corn_safe = probs['corners']['total']['Over 7.5']
        
        odd_corn_h_react = get_market_price('home', 'corners', 4.5, prob_corn_h_react)
        odd_corn_a_react = get_market_price('away', 'corners', 3.5, prob_corn_a_react)
        odd_card_tot_35 = get_market_price('total', 'cards', 3.5, probs['cards']['total']['Over 3.5'])
        
        # === HEDGE 1: ESPELHO ===
        has_home_corn = h in principal_desc_str and "Escanteios" in principal_desc_str
        has_away_corn = a in principal_desc_str and "Escanteios" in principal_desc_str
        
        h1_sel = ""
        h1_odd = 1.0
        
        if has_home_corn:
            h1_sel = f"{a} Over 3.5 Escanteios + Total Over 3.5 Cartões"
            h1_odd = round(odd_corn_a_react * odd_card_tot_35 * 0.9, 2)
        elif has_away_corn:
            h1_sel = f"{h} Over 4.5 Escanteios + Total Over 3.5 Cartões"
            h1_odd = round(odd_corn_h_react * odd_card_tot_35 * 0.9, 2)
        else:
            if res['monte_carlo']['h'] > res['monte_carlo']['a']:
                odd_dc = get_market_price('away', 'dc', 0, probs['chance']['X2'])
                h1_sel = f"DC {a} ou Empate + Total Over 3.5 Cartões"
                h1_odd = round(odd_dc * odd_card_tot_35 * 0.9, 2)
            else:
                odd_dc = get_market_price('home', 'dc', 0, probs['chance']['1X'])
                h1_sel = f"DC {h} ou Empate + Total Over 3.5 Cartões"
                h1_odd = round(odd_dc * odd_card_tot_35 * 0.9, 2)
        
        hedge1.append({'jogo': game_name, 'selecao': h1_sel, 'odd': h1_odd})
        
        # === HEDGE 2: SEGURANÇA ===
        fav_home = res['monte_carlo']['h'] > res['monte_carlo']['a']
        if fav_home:
            dc_sel = f"DC {h} ou Empate"
            dc_odd = get_market_price('home', 'dc', 0, probs['chance']['1X'])
        else:
            dc_sel = f"DC {a} ou Empate"
            dc_odd = get_market_price('away', 'dc', 0, probs['chance']['X2'])
        
        safe_card_odd = get_market_price('total', 'cards', 2.5, prob_card_safe)
        h2_sel = f"{dc_sel} + Total Over 2.5 Cartões"
        h2_odd = round(dc_odd * safe_card_odd * 0.95, 2)
        
        if h2_odd < 1.80:
            safe_corn_odd = get_market_price('total', 'corners', 7.5, prob_corn_safe)
            h2_sel += " + Over 7.5 Cantos"
            h2_odd = round(h2_odd * safe_corn_odd * 0.95, 2)
            
        hedge2.append({'jogo': game_name, 'selecao': h2_sel, 'odd': h2_odd})
        processed_games.add(game_name)

    return {
        'principal': {'itens': principal_display, 'odd': round(np.prod([x['odd'] for x in principal_display]), 2)},
        'hedge1': {'itens': hedge1, 'odd': round(np.prod([x['odd'] for x in hedge1]), 2)},
        'hedge2': {'itens': hedge2, 'odd': round(np.prod([x['odd'] for x in hedge2]), 2)}
    }

# ═══════════════════════════════════════════════════════════════════════════
# INTERFACE PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════

def main():
    # REMOVIDO CSS ESCURO. AGORA O TEMA É CLARO (PADRÃO)
    
    with st.spinner("Carregando bases..."):
        stats = learn_stats_v23()
        refs = load_referees_v23()
        calendar = load_calendar_safe()
        all_dfs = load_all_dataframes()
        
    st.title("⚽ FutPrevisão V25.7 White")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📅 Calendário", "🔍 Simulação", "🎯 Scanner/Manual", "🛡️ Hedges", "📊 Radares"])
    
    with tab3:
        st.header("🎫 Construtor de Bilhetes")
        modo = st.radio("Modo:", ["🤖 Robô Scanner", "✍️ Manual"], horizontal=True)
        st.markdown("---")
        
        if not calendar.empty:
            dates = sorted(calendar['DtObj'].dt.strftime('%d/%m/%Y').unique())
            sel_date = st.selectbox("Data:", dates, key="builder_date")
            df_day = calendar[calendar['DtObj'].dt.strftime('%d/%m/%Y') == sel_date]
            
            if "Robô" in modo:
                avail_leagues = sorted(df_day['Liga'].unique())
                sel_leagues = st.multiselect("Ligas:", avail_leagues, default=avail_leagues)
                if st.button("🚀 GERAR BILHETE", type="primary"):
                    res = generate_smart_ticket_v23(calendar, stats, refs, all_dfs, sel_date, target_leagues=sel_leagues)
                    if res['ticket']:
                        st.session_state.current_ticket = res['ticket']
                        st.success(f"Gerado com {len(res['ticket'])} apostas!")
                    else: st.warning("Nada encontrado.")
            else:
                games = sorted((df_day['Time_Casa'] + ' vs ' + df_day['Time_Visitante']).unique())
                # KEY ÚNICA PARA JOGO
                sel_game = st.selectbox("Jogo:", games, key="sel_game_manual")
                
                if sel_game:
                    row = df_day[(df_day['Time_Casa'] + ' vs ' + df_day['Time_Visitante']) == sel_game].iloc[0]
                    res = calcular_jogo_v23(row['Time_Casa'], row['Time_Visitante'], stats, None, refs, all_dfs)
                    if 'error' not in res:
                        mkts = get_available_markets_for_game(res, get_detailed_probs(res))
                        
                        # MAPEAMENTO ÚNICO PARA EVITAR DUPLICATAS
                        options_map = {}
                        for m in mkts:
                            label = f"{m['mercado']} (@{m['odd']})"
                            options_map[label] = m
                        
                        sorted_labels = sorted(options_map.keys())
                        
                        # KEY ÚNICA PARA MERCADO PARA EVITAR RESET
                        sel_mkt_label = st.selectbox("Mercado:", sorted_labels, key="sel_mkt_manual")
                        
                        if st.button("➕ Adicionar"):
                            if sel_mkt_label in options_map:
                                obj = options_map[sel_mkt_label]
                                st.session_state.current_ticket.append({
                                    'type': 'manual', 'jogo': sel_game, 'mercado': obj['mercado'], 
                                    'odd': obj['odd'], 'prob': obj['prob']
                                })
                                st.success("Adicionado!")

            if st.session_state.current_ticket:
                st.markdown("### 🎫 Bilhete Atual")
                if st.button("🗑️ Limpar"): st.session_state.current_ticket = []; st.rerun()
                for it in st.session_state.current_ticket:
                    st.write(f"✅ {it['jogo']} - {it.get('mercado')} (@{it['odd']})")

    with tab4:
        st.header("🛡️ Sistema de Proteção V25")
        if st.session_state.current_ticket:
            if st.button("🛡️ CALCULAR HEDGES (Real Odds)", type="primary"):
                hedges = generate_hedges_for_user_ticket(st.session_state.current_ticket, stats, refs, all_dfs)
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.subheader("📋 Principal")
                    st.metric("Odd", f"@{hedges['principal']['odd']}")
                    for i in hedges['principal']['itens']: st.caption(f"{i['jogo']} - {i['selecao']}")
                with c2:
                    st.subheader("🛡️ Hedge 1 (Espelho)")
                    st.metric("Odd", f"@{hedges['hedge1']['odd']}")
                    for i in hedges['hedge1']['itens']: st.caption(f"{i['jogo']} - {i['selecao']}")
                with c3:
                    st.subheader("🔄 Hedge 2 (Segurança)")
                    st.metric("Odd", f"@{hedges['hedge2']['odd']}")
                    for i in hedges['hedge2']['itens']: st.caption(f"{i['jogo']} - {i['selecao']}")
        else: st.warning("Adicione jogos na Aba 3 primeiro.")

if __name__ == "__main__":
    main()
