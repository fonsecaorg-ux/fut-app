"""
╔═══════════════════════════════════════════════════════════════════════════╗
║       FUTPREVISÃO V17.3 - REAL DATA ONLY (PROFESSIONAL EDITION)           ║
║                          Sistema Profissional de Apostas                   ║
║                                                                            ║
║  Versão: V17.3 Stable                                                     ║
║  Base de Dados: 100% Real (Sem Simulação/Mock)                            ║
║  Funcionalidades: 25+ (Monte Carlo, Kelly, H2H, Elo, TS-Index, Hedge...)  ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import math
import numpy as np
import os
from typing import Dict, List, Any, Optional
from difflib import get_close_matches
from datetime import datetime, timedelta

# Configuração da Página
st.set_page_config(
    page_title="FutPrevisão V17.3 Pro",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTES & CONFIGURAÇÕES
# ═══════════════════════════════════════════════════════════════════════════

DEFAULTS = {'shots_on_target': 4.5, 'red_cards_avg': 0.08}

NAME_MAPPING = {
    'Man United': 'Man Utd', 'Manchester United': 'Man Utd', 'Man City': 'Man City',
    'Manchester City': 'Man City', 'Spurs': 'Tottenham', 'Newcastle': 'Newcastle',
    'Wolves': 'Wolves', 'Brighton': 'Brighton', 'Nott\'m Forest': 'Nottm Forest',
    'West Ham': 'West Ham', 'Leicester': 'Leicester', 'Athletic Club': 'Ath Bilbao',
    'Atl. Madrid': 'Ath Madrid'
}

LIGAS_ALVO = [
    "Premier League", "La Liga", "Serie A", "Bundesliga", "Ligue 1",
    "Championship", "Bundesliga 2", "Pro League", "Süper Lig", "Scottish Premiership"
]

# ═══════════════════════════════════════════════════════════════════════════
# 1. CARREGAMENTO DE DADOS (REAL DATA ENGINE)
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def find_and_load_csv(league_name: str, uploaded_files: List = None) -> pd.DataFrame:
    """Carrega CSVs do disco local ou de uploads manuais."""
    # 1. Tenta carregar de Uploads Manuais (se houver)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if league_name.lower() in uploaded_file.name.lower().replace("_", " "):
                try:
                    df = pd.read_csv(uploaded_file, encoding='latin1')
                    return process_dataframe(df, league_name)
                except: pass

    # 2. Tenta carregar do Disco Local (Repositório)
    attempts = [
        f"{league_name} 25.26.csv", f"{league_name.replace(' ', '_')}_25_26.csv", f"{league_name}.csv"
    ]
    if "Süper Lig" in league_name: attempts.extend(["Super Lig Turquia 25.26.csv"])
    
    for filename in attempts:
        if os.path.exists(filename):
            try:
                # Tenta latin1 primeiro (padrão comum)
                df = pd.read_csv(filename, encoding='latin1')
                return process_dataframe(df, league_name)
            except:
                try:
                    # Tenta utf-8 se falhar
                    df = pd.read_csv(filename, encoding='utf-8')
                    return process_dataframe(df, league_name)
                except: pass
                
    return pd.DataFrame()

def process_dataframe(df: pd.DataFrame, league_name: str) -> pd.DataFrame:
    """Padroniza colunas do DataFrame carregado."""
    if df.empty: return df
    df.columns = [c.strip() for c in df.columns]
    rename_map = {
        'Mandante': 'HomeTeam', 'Visitante': 'AwayTeam', 
        'Time_Casa': 'HomeTeam', 'Time_Visitante': 'AwayTeam',
        'Data': 'Date', 'Liga': 'League'
    }
    df = df.rename(columns=rename_map)
    df['_League_'] = league_name
    return df

@st.cache_resource
def load_all_dataframes(uploaded_files: List = None) -> Dict[str, pd.DataFrame]:
    dfs = {}
    for league in LIGAS_ALVO:
        df = find_and_load_csv(league, uploaded_files)
        if not df.empty: dfs[league] = df
    return dfs

def calculate_elo(df: pd.DataFrame, K=30) -> Dict[str, float]:
    elo_ratings = {}
    teams = set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique())
    for team in teams: elo_ratings[team] = 1500

    if 'Date' in df.columns:
        df['DtObj'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)
        df = df.sort_values('DtObj')

    for _, row in df.iterrows():
        if 'FTHG' not in row or 'FTAG' not in row: continue
        h, a = row['HomeTeam'], row['AwayTeam']
        elo_h, elo_a = elo_ratings.get(h, 1500), elo_ratings.get(a, 1500)
        
        res = 1 if row['FTHG'] > row['FTAG'] else 0.5 if row['FTHG'] == row['FTAG'] else 0
        exp_h = 1 / (1 + 10**((elo_a - elo_h) / 400))
        
        elo_ratings[h] = elo_h + K * (res - exp_h)
        elo_ratings[a] = elo_a + K * ((1 - res) - (1 - exp_h))
    return elo_ratings

def calculate_ts_index(stats_db: Dict) -> Dict:
    ts_index = {}
    for team, stats in stats_db.items():
        elo_norm = (stats.get('elo_rating', 1500) - 1000) / 1000 
        gf_norm = min(1, stats.get('goals_f', 1.5) / 2.5)
        ga_norm = 1 - min(1, stats.get('goals_a', 1.5) / 2.5)
        index = (elo_norm * 0.5) + (gf_norm * 0.3) + (ga_norm * 0.2)
        ts_index[team] = round(index * 100, 1)
    return ts_index

@st.cache_data(ttl=3600)
def learn_stats_v17(uploaded_files: List = None) -> Dict[str, Dict[str, Any]]:
    stats_db = {}
    all_dfs = load_all_dataframes(uploaded_files)
    
    # Elo Global
    global_elo = {}
    for lg, df in all_dfs.items(): global_elo.update(calculate_elo(df))

    for league, df in all_dfs.items():
        if df.empty: continue
        
        # Garante colunas numéricas
        cols = ['HC', 'AC', 'HY', 'AY', 'HF', 'AF', 'FTHG', 'FTAG', 'HST', 'AST', 'HR', 'AR']
        for c in cols:
            if c not in df.columns: df[c] = np.nan
            else: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
            
        # Peso de Recência
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)
            df = df.sort_values(by='Date', ascending=True).dropna(subset=['Date'])
            if not df.empty:
                w = np.exp(np.linspace(0, 1, len(df)))
                df['RecencyWeight'] = w / w.sum() * len(df)
            else: df['RecencyWeight'] = 1.0
        else: df['RecencyWeight'] = 1.0
        
        try:
            # Agregação Vetorial
            metrics = ['HC', 'HY', 'HF', 'FTHG', 'FTAG', 'HST', 'HR']
            metrics_a = ['AC', 'AY', 'AF', 'FTAG', 'FTHG', 'AST', 'AR']
            
            # Prepara dados ponderados
            for m in metrics + metrics_a:
                if m not in df.columns: df[m] = 0
                if f'{m}_W' not in df.columns: df[f'{m}_W'] = df[m] * df['RecencyWeight']
            
            h_sum = df.groupby('HomeTeam')[[f'{m}_W' for m in metrics] + ['RecencyWeight']].sum()
            h_stats = h_sum.div(h_sum['RecencyWeight'], axis=0).fillna(0)
            h_stats.columns = metrics
            
            a_sum = df.groupby('AwayTeam')[[f'{m}_W' for m in metrics_a] + ['RecencyWeight']].sum()
            a_stats = a_sum.div(a_sum['RecencyWeight'], axis=0).fillna(0)
            a_stats.columns = metrics_a 
            
            all_teams = set(h_stats.index) | set(a_stats.index)
            for team in all_teams:
                h = h_stats.loc[team] if team in h_stats.index else pd.Series(0, index=metrics)
                a = a_stats.loc[team] if team in a_stats.index else pd.Series(0, index=metrics_a)
                
                def w_avg(v1, v2): return (v1 * 0.6) + (v2 * 0.4) if v1+v2 > 0 else 0
                
                stats_db[team] = {
                    'corners': w_avg(h.get('HC',0), a.get('AC',0)),
                    'cards': w_avg(h.get('HY',0), a.get('AY',0)),
                    'fouls': w_avg(h.get('HF',0), a.get('AF',0)),
                    'goals_f': w_avg(h.get('FTHG',0), a.get('FTHG',0)), 
                    'goals_a': w_avg(h.get('FTAG',0), a.get('FTAG',0)),
                    'shots_on_target': w_avg(h.get('HST',0), a.get('AST',0)),
                    'red_cards_avg': w_avg(h.get('HR',0), a.get('AR',0)),
                    'league': league,
                    'elo_rating': global_elo.get(team, 1500),
                    'home_goals_f': h.get('FTHG', 0), 'home_goals_a': h.get('FTAG', 0),
                    'away_goals_f': a.get('FTHG', 0), 'away_goals_a': a.get('FTAG', 0),
                }
        except Exception as e:
            # Em produção, logar erro silenciosamente
            pass
            
    ts = calculate_ts_index(stats_db)
    for t in stats_db: stats_db[t]['ts_index'] = ts.get(t, 50.0)
        
    return stats_db

@st.cache_data(ttl=3600)
def load_referees_v15() -> Dict[str, Dict[str, float]]:
    refs_db = {}
    files = ["arbitros_5_ligas_2025_2026.csv", "arbitros.csv"]
    for f in files:
        if os.path.exists(f):
            try:
                df = pd.read_csv(f)
                for _, row in df.iterrows():
                    nome = str(row.get('Arbitro', row.get('Nome', 'Juiz'))).strip()
                    media = float(row.get('Media_Cartoes_Por_Jogo', row.get('Fator', 4.0)))
                    reds = float(row.get('Cartoes_Vermelhos', 0))
                    games = float(row.get('Jogos_Apitados', 1))
                    
                    refs_db[nome] = {
                        'factor': media/4.0, 
                        'red_rate': (reds/games) if games > 0 else 0.08,
                        'strictness_score': media
                    }
            except: pass
    return refs_db

@st.cache_data(ttl=3600)
def load_calendar_safe() -> pd.DataFrame:
    # Tenta carregar CSV real
    files = ["calendario_futuro.csv", "calendario_ligas.csv"]
    for f in files:
        if os.path.exists(f):
            try:
                df = pd.read_csv(f)
                if 'Time_Casa' in df.columns: df = df.rename(columns={'Time_Casa': 'HomeTeam', 'Time_Visitante': 'AwayTeam', 'Liga': 'League', 'Data': 'Date'})
                df['DtObj'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)
                df = df.dropna(subset=['DtObj']).sort_values(by='DtObj')
                if not df.empty: return df
            except: pass
    return pd.DataFrame()

# ═══════════════════════════════════════════════════════════════════════════
# 2. MOTOR V17 (Monte Carlo + Poisson)
# ═══════════════════════════════════════════════════════════════════════════

def normalize_name(name: str, db_keys: list) -> Optional[str]:
    if name in NAME_MAPPING: name = NAME_MAPPING[name]
    if name in db_keys: return name
    matches = get_close_matches(name, db_keys, n=1, cutoff=0.6)
    return matches[0] if matches else None

def poisson_prob(k, lamb):
    if lamb > 30: return 0.0
    return (lamb**k * math.exp(-lamb)) / math.factorial(k)

def monte_carlo_simulation(xg_home, xg_away, iterations=1000):
    gh = np.random.poisson(xg_home, iterations)
    ga = np.random.poisson(xg_away, iterations)
    h_wins = np.count_nonzero(gh > ga)
    a_wins = np.count_nonzero(ga > gh)
    draws = iterations - h_wins - a_wins
    return h_wins/iterations, draws/iterations, a_wins/iterations

def calculate_kelly_criterion(prob_real, odd_casa, bankroll):
    b = odd_casa - 1
    p = prob_real / 100
    q = 1 - p
    f = (b * p - q) / b
    return max(0, f * bankroll * 0.5) 

def calculate_dutching(odds: List[float], target_profit: float) -> Dict[str, float]:
    implied_probs = [1 / odd for odd in odds]
    total_implied_prob = sum(implied_probs)
    if total_implied_prob >= 1: return {'error': "Sem valor (Prob > 100%)"}
    total_stake = target_profit / (1 - total_implied_prob)
    stakes = {f"Odd {odd:.2f}": (total_stake * implied_probs[i]) for i, odd in enumerate(odds)}
    stakes['total_stake'] = sum(stakes.values())
    return stakes

def get_h2h_stats(home: str, away: str, all_dfs: Dict) -> Dict:
    stats = {'games': 0, 'h_wins': 0, 'a_wins': 0, 'draws': 0, 'avg_goals': 0}
    total_goals = 0
    if not all_dfs: return stats

    for league, df in all_dfs.items():
        if df.empty: continue
        # Verifica colunas necessárias
        req_cols = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']
        if not all(col in df.columns for col in req_cols): continue

        mask = ((df['HomeTeam'] == home) & (df['AwayTeam'] == away)) | \
               ((df['HomeTeam'] == away) & (df['AwayTeam'] == home))
        matches = df[mask]
        
        if not matches.empty:
            stats['games'] += len(matches)
            for _, row in matches.iterrows():
                gh, ga = row['FTHG'], row['FTAG']
                total_goals += (gh + ga)
                winner = home if (row['HomeTeam'] == home and gh > ga) or (row['AwayTeam'] == home and ga > gh) else \
                         away if (row['HomeTeam'] == away and gh > ga) or (row['AwayTeam'] == away and ga > gh) else "Draw"
                if winner == home: stats['h_wins'] += 1
                elif winner == away: stats['a_wins'] += 1
                else: stats['draws'] += 1
                
    if stats['games'] > 0: stats['avg_goals'] = total_goals / stats['games']
    return stats

def get_form_analysis(team_name: str, all_dfs: Dict, n_games: int = 5) -> str:
    if not all_dfs: return "N/A"
    matches_found = []
    for league, df in all_dfs.items():
        if df.empty or 'HomeTeam' not in df.columns: continue
        t_games = df[(df['HomeTeam'] == team_name) | (df['AwayTeam'] == team_name)].copy()
        if not t_games.empty and 'Date' in t_games.columns:
            t_games['Date'] = pd.to_datetime(t_games['Date'], errors='coerce', dayfirst=True)
            matches_found.append(t_games)
            
    if not matches_found: return "N/A"
    try:
        full_history = pd.concat(matches_found).sort_values(by='Date', ascending=False).head(n_games)
        form = []
        for _, row in full_history.iterrows():
            if row['HomeTeam'] == team_name: gh, ga = row['FTHG'], row['FTAG']
            else: gh, ga = row['FTAG'], row['FTHG']
            if gh > ga: form.append('V')
            elif ga > gh: form.append('D')
            else: form.append('E')
        return "".join(form[::-1])
    except: return "N/A"

def get_native_history(team_name: str, league: str, market: str, line: float, location: str, all_dfs: Dict, n_games: int = 5) -> str:
    if not all_dfs or league not in all_dfs: return "N/A"
    df = all_dfs[league]
    col_map = {('home', 'corners'): 'HC', ('away', 'corners'): 'AC', ('home', 'cards'): 'HY', ('away', 'cards'): 'AY'}
    col_code = col_map.get((location, market))
    team_col = 'HomeTeam' if location == 'home' else 'AwayTeam'
    
    matches = df[df[team_col] == team_name]
    if matches.empty: return "0/0"
    if 'Date' in matches.columns:
        matches['DtObj'] = pd.to_datetime(matches['Date'], errors='coerce', dayfirst=True)
        matches = matches.sort_values(by='DtObj')
    last_matches = matches.tail(n_games)
    if col_code not in last_matches.columns: return "0/0"
    hits = sum(1 for val in last_matches[col_code] if float(val) > line)
    return f"{hits}/{len(last_matches)}"

def calcular_jogo_v17(home: str, away: str, stats: Dict, ref: Optional[str], refs_db: Dict, weather_bad: bool = False, all_dfs: Dict = None) -> Dict:
    h_norm = normalize_name(home, list(stats.keys()))
    a_norm = normalize_name(away, list(stats.keys()))
    
    if not h_norm or not a_norm: return {'error': "Times não encontrados. Verifique nomes."}
    
    s_h, s_a = stats[h_norm], stats[a_norm]
    r_data = refs_db.get(ref, {'factor': 1.0, 'red_rate': 0.08, 'strictness_score': 4.0}) if ref else {'factor': 1.0, 'red_rate': 0.08, 'strictness_score': 4.0}
    
    weather_factor_goals = 0.9 if weather_bad else 1.0
    weather_factor_cards = 1.2 if weather_bad else 1.0
    
    elo_diff = s_h.get('elo_rating', 1500) - s_a.get('elo_rating', 1500)
    elo_factor = math.log10(max(1, abs(elo_diff))) * 0.05 * (1 if elo_diff > 0 else -1)
    
    corn_h = s_h['corners'] * 1.15
    corn_a = s_a['corners'] * 0.90
    card_h = s_h['cards'] * r_data['factor'] * weather_factor_cards
    card_a = s_a['cards'] * r_data['factor'] * weather_factor_cards
    
    # xG Model
    lg_avg = max(0.1, s_h.get('goals_a', 1.0))
    xg_home_base = (s_h.get('home_goals_f', 1.0) * s_a.get('away_goals_a', 1.0)) / lg_avg
    xg_away_base = (s_a.get('away_goals_f', 1.0) * s_h.get('home_goals_a', 1.0)) / lg_avg
    
    xg_home = max(0.1, (xg_home_base * weather_factor_goals) + elo_factor)
    xg_away = max(0.1, (xg_away_base * weather_factor_goals) - elo_factor)
    
    mc_h, mc_d, mc_a = monte_carlo_simulation(xg_home, xg_away)
    
    # Probs
    prob_over_9_5_corners = min(95, 40 + (corn_h + corn_a) * 5)
    prob_over_4_5_cards = min(95, 50 + (card_h + card_a - 4.5) * 10)
    
    prob_btts = (1 - poisson_prob(0, xg_home)) * (1 - poisson_prob(0, xg_away)) * 100
    prob_over_2_5 = (1 - sum([poisson_prob(h, xg_home) * poisson_prob(a, xg_away) for h in range(3) for a in range(3) if h+a < 3])) * 100
    
    h2h_stats = get_h2h_stats(h_norm, a_norm, all_dfs) if all_dfs else {}
    
    return {
        'home': h_norm, 'away': a_norm, 'league_h': s_h.get('league'), 'league_a': s_a.get('league'),
        'goals': {'h': xg_home, 'a': xg_away},
        'corners': {'h': corn_h, 'a': corn_a, 'total_over_9_5': prob_over_9_5_corners},
        'cards': {'h': card_h, 'a': card_a, 'total_over_4_5': prob_over_4_5_cards},
        'monte_carlo': {'h': mc_h * 100, 'd': mc_d * 100, 'a': mc_a * 100},
        'meta': {'ts_h': s_h.get('ts_index', 50), 'ts_a': s_a.get('ts_index', 50)},
        'advanced_probs': {'btts': prob_btts, 'over_2_5': prob_over_2_5},
        'h2h_stats': h2h_stats,
        'form_h': get_form_analysis(h_norm, all_dfs, 5),
        'form_a': get_form_analysis(a_norm, all_dfs, 5),
        'home_stats': {'gf': s_h.get('home_goals_f',0), 'ga': s_h.get('home_goals_a',0)},
        'away_stats': {'gf': s_a.get('away_goals_f',0), 'ga': s_a.get('away_goals_a',0)},
    }

def get_detailed_probs(res: Dict) -> Dict:
    def sim_prob(avg, line): return max(5, min(95, 50 + (avg - line) * 15))
    
    probs = {
        'corners': {
            'home': {f'Over {l}': sim_prob(res['corners']['h'], l) for l in [3.5, 4.5, 5.5]},
            'away': {f'Over {l}': sim_prob(res['corners']['a'], l) for l in [3.5, 4.5, 5.5]},
            'total': {f'Over {l}': sim_prob(res['corners']['h'] + res['corners']['a'], l) for l in [8.5, 9.5, 10.5]}
        },
        'cards': {
            'home': {f'Over {l}': sim_prob(res['cards']['h'], l) for l in [1.5, 2.5]},
            'away': {f'Over {l}': sim_prob(res['cards']['a'], l) for l in [1.5, 2.5]},
            'total': {f'Over {l}': sim_prob(res['cards']['h'] + res['cards']['a'], l) for l in [3.5, 4.5, 5.5]}
        }
    }
    
    mc = res['monte_carlo']
    probs['chance'] = {
        '1X': mc['h'] + mc['d'], 'X2': mc['a'] + mc['d'], '12': mc['h'] + mc['a'],
        'DNB_1': (mc['h'] / (mc['h'] + mc['a'] + 0.01)) * 100,
        'DNB_2': (mc['a'] / (mc['h'] + mc['a'] + 0.01)) * 100
    }
    return probs

def get_fair_odd(prob_percent: float) -> float:
    return round(100 / prob_percent, 2) if prob_percent > 0 else 99.0

def generate_bet_options(home_team: str, away_team: str, probs: Dict) -> List[Dict]:
    options = []
    # Cantos
    for line in [3.5, 4.5, 5.5]:
        p = probs['corners']['home'].get(f'Over {line}', 0)
        options.append({'label': f"{home_team} Over {line} cantos", 'prob': p, 'market':'corners', 'side':'home', 'min_odd': get_fair_odd(p)})
    for line in [3.5, 4.5]:
        p = probs['corners']['away'].get(f'Over {line}', 0)
        options.append({'label': f"{away_team} Over {line} cantos", 'prob': p, 'market':'corners', 'side':'away', 'min_odd': get_fair_odd(p)})
    for line in [8.5, 9.5, 10.5]:
        p = probs['corners']['total'].get(f'Over {int(line)}.5', 0)
        options.append({'label': f"Total Over {line} cantos", 'prob': p, 'market':'corners', 'side':'total', 'min_odd': get_fair_odd(p)})
    
    # Cartões
    for line in [1.5, 2.5]:
        p = probs['cards']['home'].get(f'Over {line}', 0)
        options.append({'label': f"{home_team} Over {line} cartões", 'prob': p, 'market':'cards', 'side':'home', 'min_odd': get_fair_odd(p)})
        p2 = probs['cards']['away'].get(f'Over {line}', 0)
        options.append({'label': f"{away_team} Over {line} cartões", 'prob': p2, 'market':'cards', 'side':'away', 'min_odd': get_fair_odd(p2)})
    for line in [2.5, 3.5, 4.5, 5.5]:
        p = probs['cards']['total'].get(f'Over {int(line)}.5', 0)
        options.append({'label': f"Total Over {line} cartões", 'prob': p, 'market':'cards', 'side':'total', 'min_odd': get_fair_odd(p)})

    # Chance
    if probs['chance']['DNB_1'] >= 65:
        p = probs['chance']['DNB_1']
        options.append({'label': f"Empate Anula: {home_team}", 'prob': p, 'market':'chance', 'side':'home', 'min_odd': get_fair_odd(p)})
    if probs['chance']['DNB_2'] >= 65:
        p = probs['chance']['DNB_2']
        options.append({'label': f"Empate Anula: {away_team}", 'prob': p, 'market':'chance', 'side':'away', 'min_odd': get_fair_odd(p)})
        
    options.append({'label': f"Dupla Chance: {home_team} ou Empate", 'prob': probs['chance']['1X'], 'market':'chance', 'side':'home', 'min_odd': get_fair_odd(probs['chance']['1X'])})
    options.append({'label': f"Dupla Chance: {away_team} ou Empate", 'prob': probs['chance']['X2'], 'market':'chance', 'side':'away', 'min_odd': get_fair_odd(probs['chance']['X2'])})

    options.sort(key=lambda x: x['prob'], reverse=True)
    return options

def calculate_combined_probability(selections: List[Dict]) -> float:
    if not selections: return 0.0
    prob = 1.0
    for s in selections: prob *= (s['prob']/100)
    return prob * 100

def generate_dual_hedges(main_slip: List[Dict], stats: Dict, refs_db: Dict):
    hedge1 = []
    hedge2 = []
    
    games = {}
    for sel in main_slip:
        gid = sel['game_id']
        if gid not in games: games[gid] = []
        games[gid].append(sel)
        
    for gid, sels in games.items():
        home, away = sels[0]['home'], sels[0]['away']
        # Usa None para all_dfs em hedge para performance
        res = calcular_jogo_v17(home, away, stats, None, refs_db, False, None)
        if 'error' in res: continue
        
        probs = get_detailed_probs(res)
        all_opts = generate_bet_options(home, away, probs)
        valid_opts = [o for o in all_opts if o['prob'] >= 65]
        if len(valid_opts) < 6: valid_opts = all_opts[:12]
        
        main_labels = [s['label'] for s in sels]
        
        # HEDGE 1: SAFETY (Chance + Stat)
        h1_pair = []
        chance_opts = [o for o in valid_opts if o['market'] == 'chance' and o['label'] not in main_labels]
        if chance_opts: h1_pair.append(chance_opts[0])
        else:
            safe = [o for o in valid_opts if o['label'] not in main_labels]
            if safe: h1_pair.append(safe[0])
            
        stat_opts = [o for o in valid_opts if o['market'] in ['corners', 'cards'] and o['label'] not in main_labels and o not in h1_pair]
        if stat_opts: h1_pair.append(stat_opts[0])
            
        if len(h1_pair) < 2:
            leftover = [o for o in valid_opts if o['label'] not in main_labels and o not in h1_pair]
            h1_pair.extend(leftover[:2-len(h1_pair)])
            
        for opt in h1_pair:
            hedge1.append({**opt, 'game_id': gid, 'home': home, 'away': away, 'type': 'Safety'})

        # HEDGE 2: MIX (Canto + Cartão)
        h2_pair = []
        used = main_labels + [o['label'] for o in h1_pair]
        avail = [o for o in valid_opts if o['label'] not in used]
        
        corns = [o for o in avail if o['market'] == 'corners']
        cards = [o for o in avail if o['market'] == 'cards']
        chanc = [o for o in avail if o['market'] == 'chance']
        
        if corns:
            h2_pair.append(corns[0])
            if cards: h2_pair.append(cards[0])
            elif chanc: h2_pair.append(chanc[0])
            else: 
                if len(corns) > 1: h2_pair.append(corns[1])
        elif cards:
            h2_pair.append(cards[0])
            if chanc: h2_pair.append(chanc[0])
            elif len(cards) > 1: h2_pair.append(cards[1])
            
        if len(h2_pair) < 2:
            leftover = [o for o in avail if o not in h2_pair]
            h2_pair.extend(leftover[:2-len(h2_pair)])
            
        for opt in h2_pair:
            hedge2.append({**opt, 'game_id': gid, 'home': home, 'away': away, 'type': 'Mix'})
            
    return hedge1, hedge2

# ═══════════════════════════════════════════════════════════════════════════
# UI PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════

def main():
    st.sidebar.title("🎛️ Painel V17.3")
    weather = st.sidebar.checkbox("🌧️ Clima Ruim", value=False)
    
    st.title("🛡️ FutPrevisão V17.3 (Real Data Only)")
    
    # --- UPLOAD MANUAL (FALLBACK) ---
    uploaded_files = st.sidebar.file_uploader("📂 Upload CSVs (Opcional)", accept_multiple_files=True, type=['csv'])
    
    with st.spinner("Carregando bases de dados..."):
        stats = learn_stats_v17(uploaded_files)
        refs = load_referees_v15()
        calendar = load_calendar_safe()
        all_dfs = load_all_dataframes(uploaded_files)
        
    if not stats:
        st.error("🚨 NENHUM DADO ENCONTRADO!")
        st.warning("O sistema não encontrou arquivos CSV na pasta nem via upload.")
        st.info("👉 Por favor, arraste seus arquivos CSV (ex: 'Premier League 25.26.csv') para a barra lateral.")
        return # Para a execução aqui se não tiver dados

    if uploaded_files: st.sidebar.success(f"{len(uploaded_files)} arquivos carregados manualmente!")

    t1, t2, t3, t4 = st.tabs(["📅 Calendário", "🔍 Simulação", "🎰 Bet Builder", "💰 Gestão"])
    
    with t1:
        if calendar.empty: st.info("Calendário não disponível.")
        else:
            # Filtra calendário apenas com jogos que temos dados
            valid_games = calendar[calendar['HomeTeam'].isin(stats.keys())]
            st.dataframe(valid_games, use_container_width=True)
            
            st.markdown("---")
            st.subheader("Simulação Rápida (Jogos Reais)")
            for i, row in valid_games.head(5).iterrows():
                h, a, l = row['HomeTeam'], row['AwayTeam'], row['League']
                res = calcular_jogo_v17(h, a, stats, None, refs, False, all_dfs)
                if 'error' not in res:
                    mc = res['monte_carlo']
                    st.markdown(f"**{h}** vs **{a}** ({l})")
                    st.info(f"Prob: 🏠 {mc['h']:.0f}% | 🤝 {mc['d']:.0f}% | ✈️ {mc['a']:.0f}% | TS: {res['meta']['ts_h']} vs {res['meta']['ts_a']}")

    with t2:
        l_times = sorted(list(stats.keys()))
        l_refs = ["Neutro"] + sorted(list(refs.keys()))
        c1, c2, c3 = st.columns(3)
        h = c1.selectbox("Casa", l_times, index=0)
        a = c2.selectbox("Fora", l_times, index=min(1, len(l_times)-1))
        r = c3.selectbox("Árbitro", l_refs)
        
        if st.button("Simular Jogo"):
            rf = None if r == "Neutro" else r
            res = calcular_jogo_v17(h, a, stats, rf, refs, weather, all_dfs)
            
            if 'error' in res: st.error(res['error'])
            else:
                probs = get_detailed_probs(res)
                st.subheader(f"{res['home']} vs {res['away']}")
                st.info(f"xG: {res['goals']['h']:.2f} x {res['goals']['a']:.2f}")
                st.write(f"Forma: 🏠 {res['form_h']} | ✈️ {res['form_a']}")
                
                # H2H (Função Corrigida)
                h2h = res['h2h_stats']
                if h2h['games'] > 0:
                    st.write(f"**H2H ({h2h['games']} jogos):** {res['home']} venceu {h2h['h_wins']}x")
                
                c1, c2 = st.columns(2)
                c1.write("**Escanteios**")
                for k, v in probs['corners']['home'].items(): c1.write(f"{res['home']} {k}: {v:.0f}%")
                c2.write("**Cartões**")
                for k, v in probs['cards']['home'].items(): c2.write(f"{res['home']} {k}: {v:.0f}%")

    with t3:
        if 'main_slip' not in st.session_state: st.session_state.main_slip = []
        l_times = sorted(list(stats.keys()))
        num = st.number_input("Jogos no Bilhete", 1, 5, 3)
        
        temp = []
        for i in range(num):
            st.markdown(f"**Jogo {i+1}**")
            c1, c2, c3 = st.columns(3)
            h = c1.selectbox(f"C", l_times, key=f"bh{i}")
            a = c2.selectbox(f"F", l_times, key=f"ba{i}", index=min(1, len(l_times)-1))
            odd = c3.number_input(f"Odd", 1.01, 20.0, 1.01, key=f"odd{i}")
            
            res = calcular_jogo_v17(h, a, stats, None, refs, False, None)
            probs = get_detailed_probs(res)
            opts = generate_bet_options(h, a, probs)
            lbls = [o['label'] for o in opts]
            
            s1 = st.selectbox(f"S1", range(len(opts)), format_func=lambda x: lbls[x], key=f"s1{i}")
            s2 = st.selectbox(f"S2", range(len(opts)), format_func=lambda x: lbls[x], key=f"s2{i}", index=min(1, len(opts)-1))
            
            temp.append({**opts[s1], 'game_id': i, 'home': h, 'away': a, 'user_odd': odd})
            temp.append({**opts[s2], 'game_id': i, 'home': h, 'away': a, 'user_odd': odd})
            
        st.session_state.main_slip = temp
        
        if st.button("🔮 GERAR ESTRATÉGIA", type="primary"):
            h1, h2 = generate_dual_hedges(st.session_state.main_slip, stats, refs)
            st.success("Sucesso!")
            c1, c2, c3 = st.columns(3)
            
            def card(t, bets, col):
                with col:
                    st.info(t)
                    for b in bets: st.write(f"- {b['label']} (@{b['min_odd']})")
                    
            card("Principal", st.session_state.main_slip, c1)
            card("Hedge Safety", h1, c2)
            card("Hedge Mix", h2, c3)

    with t4:
        st.subheader("Calculadora Dutching")
        odds_str = st.text_input("Odds (ex: 2.5, 3.1)", "2.0, 3.5")
        try:
            odds = [float(x) for x in odds_str.split(',')]
            res_d = calculate_dutching(odds, 100)
            if 'error' in res_d: st.error(res_d['error'])
            else: st.success(f"Stake Total: {res_d['total_stake']:.2f}")
        except: pass

if __name__ == "__main__":
    main()
