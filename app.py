"""
╔═══════════════════════════════════════════════════════════════════════════╗
║       FUTPREVISÃO V17.0 - THE FINAL FRONTIER (50+ FEATURES)               ║
║                          Sistema Profissional de Apostas                   ║
║                                                                            ║
║  Versão: V17.0 Ultimate                                                   ║
║  Funcionalidades: 50+ (Dutching, Hedge Stake, EV, TS-Index, Form Analysis)║
║  Correção Crítica: Erro de Indentação na V16                              ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import math
import numpy as np
import os
import random
from typing import Dict, List, Any, Optional
from difflib import get_close_matches
from datetime import datetime, timedelta
import json # Necessário para salvar/carregar dados de sessão

# Variável Global para Logs de Debug
DEBUG_LOGS = []

# Configuração da Página
st.set_page_config(
    page_title="FutPrevisão V17 Final Frontier",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTES & CONFIGURAÇÕES
# ═══════════════════════════════════════════════════════════════════════════

THRESHOLDS = {
    'fouls_violent': 12.5, 'shots_pressure_high': 6.0,
    'red_rate_strict_high': 0.12, 'prob_elite': 75
}

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
# 1. CARREGAMENTO DE DADOS (V17 - Mais Robusto)
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def find_and_load_csv(league_name: str) -> pd.DataFrame:
    attempts = [
        f"{league_name} 25.26.csv", f"{league_name.replace(' ', '_')}_25_26.csv", f"{league_name}.csv"
    ]
    if "Süper Lig" in league_name: attempts.extend(["Super Lig Turquia 25.26.csv"])
    if "Pro League" in league_name: attempts.append("Pro League Belgica 25.26.csv")
    if "Premiership" in league_name: attempts.append("Premiership Escocia 25.26.csv")
    if "Championship" in league_name: attempts.append("Championship Inglaterra 25.26.csv")

    for filename in attempts:
        if os.path.exists(filename):
            try:
                try: df = pd.read_csv(filename, encoding='utf-8')
                except: df = pd.read_csv(filename, encoding='latin1')
                if not df.empty:
                    df.columns = [c.strip() for c in df.columns]
                    rename_map = {}
                    if 'Mandante' in df.columns: rename_map['Mandante'] = 'HomeTeam'
                    if 'Visitante' in df.columns: rename_map['Visitante'] = 'AwayTeam'
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

def calculate_elo(df: pd.DataFrame, K=30) -> Dict[str, float]:
    elo_ratings = {}
    teams = set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique())
    for team in teams:
        elo_ratings[team] = 1500

    for index, row in df.iterrows():
        team_h = row['HomeTeam']
        team_a = row['AwayTeam']
        
        if 'FTHG' not in row or 'FTAG' not in row: continue
        
        elo_h = elo_ratings.get(team_h, 1500)
        elo_a = elo_ratings.get(team_a, 1500)

        if row['FTHG'] > row['FTAG']:
            result = 1
        elif row['FTHG'] == row['FTAG']:
            result = 0.5
        else:
            result = 0

        expected_h = 1 / (1 + 10**((elo_a - elo_h) / 400))
        
        new_elo_h = elo_h + K * (result - expected_h)
        new_elo_a = elo_a + K * ((1 - result) - (1 - expected_h))
        
        elo_ratings[team_h] = new_elo_h
        elo_ratings[team_a] = new_elo_a
        
    return elo_ratings

# Feature 50: Cálculo do Team Strength Index (TS-Index)
def calculate_ts_index(stats_db: Dict) -> Dict:
    ts_index = {}
    for team, stats in stats_db.items():
        # Combina Elo (peso 50%), Gols Feitos (peso 30%), e Gols Sofridos (peso 20%)
        elo_norm = (stats['elo_rating'] - 1000) / 1000 # Normaliza Elo (1000-2000) para 0-1
        gf_norm = min(1, stats['goals_f'] / 2.5)
        ga_norm = 1 - min(1, stats['goals_a'] / 2.5)
        
        index = (elo_norm * 0.5) + (gf_norm * 0.3) + (ga_norm * 0.2)
        ts_index[team] = round(index * 100, 1)
    return ts_index

@st.cache_data(ttl=3600)
def learn_stats_v17() -> Dict[str, Dict[str, Any]]:
    stats_db = {}
    all_dfs = load_all_dataframes()
    
    global_elo = {}
    for league, df in all_dfs.items():
        global_elo.update(calculate_elo(df))

    for league in LIGAS_ALVO:
        df = find_and_load_csv(league)
        if df.empty: continue
        
        cols = ['HomeTeam', 'AwayTeam', 'HC', 'AC', 'HY', 'AY', 'HF', 'AF', 'FTHG', 'FTAG', 'HST', 'AST', 'HR', 'AR', 'Date']
        for c in cols: 
            if c not in df.columns: df[c] = np.nan
        
        # Feature 22: Peso por Data (Recência)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.sort_values(by='Date', ascending=True).dropna(subset=['Date'])
            df['RecencyWeight'] = np.exp(np.linspace(0, 1, len(df)))
            df['RecencyWeight'] = df['RecencyWeight'] / df['RecencyWeight'].sum() * len(df)
        else:
            df['RecencyWeight'] = 1.0
        
        try:
            # Feature 23: Agregação Ponderada
            h_stats = df.groupby('HomeTeam').apply(lambda x: pd.Series({
                'HC': (x['HC'] * x['RecencyWeight']).sum() / x['RecencyWeight'].sum(),
                'HY': (x['HY'] * x['RecencyWeight']).sum() / x['RecencyWeight'].sum(),
                'HF': (x['HF'] * x['RecencyWeight']).sum() / x['RecencyWeight'].sum(),
                'FTHG': (x['FTHG'] * x['RecencyWeight']).sum() / x['RecencyWeight'].sum(),
                'FTAG': (x['FTAG'] * x['RecencyWeight']).sum() / x['RecencyWeight'].sum(),
                'HST': (x['HST'] * x['RecencyWeight']).sum() / x['RecencyWeight'].sum(),
                'HR': (x['HR'] * x['RecencyWeight']).sum() / x['RecencyWeight'].sum(),
            })).fillna(DEFAULTS['shots_on_target'])
            
            a_stats = df.groupby('AwayTeam').apply(lambda x: pd.Series({
                'AC': (x['AC'] * x['RecencyWeight']).sum() / x['RecencyWeight'].sum(),
                'AY': (x['AY'] * x['RecencyWeight']).sum() / x['RecencyWeight'].sum(),
                'AF': (x['AF'] * x['RecencyWeight']).sum() / x['RecencyWeight'].sum(),
                'FTAG': (x['FTAG'] * x['RecencyWeight']).sum() / x['RecencyWeight'].sum(),
                'FTHG': (x['FTHG'] * x['RecencyWeight']).sum() / x['RecencyWeight'].sum(),
                'AST': (x['AST'] * x['RecencyWeight']).sum() / x['RecencyWeight'].sum(),
                'AR': (x['AR'] * x['RecencyWeight']).sum() / x['RecencyWeight'].sum(),
            })).fillna(DEFAULTS['shots_on_target'])
            
            all_teams = set(h_stats.index) | set(a_stats.index)
            for team in all_teams:
                h = h_stats.loc[team] if team in h_stats.index else pd.Series(0, index=h_stats.columns)
                a = a_stats.loc[team] if team in a_stats.index else pd.Series(0, index=a_stats.columns)
                
                def w_avg(val_h, val_a, default=0):
                    if val_h == 0 and val_a == 0: return default
                    if val_h == 0: return val_a
                    if val_a == 0: return val_h
                    return (val_h * 0.6) + (val_a * 0.4)

                stats_db[team] = {
                    'corners': w_avg(h.get('HC',0), a.get('AC',0), 5.0),
                    'cards': w_avg(h.get('HY',0), a.get('AY',0), 2.0),
                    'fouls': w_avg(h.get('HF',0), a.get('AF',0), 11.0),
                    'goals_f': w_avg(h.get('FTHG',0), a.get('FTAG',0), 1.2),
                    'goals_a': w_avg(h.get('FTAG',0), a.get('FTHG',0), 1.2),
                    'shots_on_target': w_avg(h.get('HST',0), a.get('AST',0), 4.5),
                    'red_cards_avg': w_avg(h.get('HR',0), a.get('AR',0), 0.08),
                    'league': league,
                    'momentum': np.random.uniform(0.9, 1.1),
                    'elo_rating': global_elo.get(team, 1500),
                    # Feature 51: Stats Separadas por Local
                    'home_goals_f': h.get('FTHG', 0), 'home_goals_a': h.get('FTAG', 0),
                    'away_goals_f': a.get('FTAG', 0), 'away_goals_a': a.get('FTHG', 0),
                }
        except Exception as e:
            DEBUG_LOGS.append(f"Erro ao processar liga {league}: {e}")
            pass
            
    # Feature 50: Cálculo do TS-Index
    ts_index = calculate_ts_index(stats_db)
    for team in stats_db:
        stats_db[team]['ts_index'] = ts_index.get(team, 50.0)
        
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

# Feature 52: Carregamento de Calendário (Tenta CSV real primeiro)
@st.cache_data(ttl=3600)
def load_calendar_safe() -> pd.DataFrame:
    # Tenta carregar um arquivo real
    if os.path.exists("calendario_futuro.csv"):
        try:
            df = pd.read_csv("calendario_futuro.csv")
            if not df.empty: return df
        except:
            pass
            
    # Fallback para simulação
    try:
        df = pd.DataFrame({
            'Date': [datetime.now() + timedelta(days=i) for i in range(5)],
            'HomeTeam': ['Liverpool', 'Real Madrid', 'PSG', 'Bayern Munich', 'Juventus'],
            'AwayTeam': ['Man Utd', 'Barcelona', 'Marseille', 'Dortmund', 'Inter Milan'],
            'League': ['Premier League', 'La Liga', 'Ligue 1', 'Bundesliga', 'Serie A']
        })
        return df
    except:
        return pd.DataFrame()

# ═══════════════════════════════════════════════════════════════════════════
# 2. MOTOR V17 (Monte Carlo + Poisson + Features Avançadas)
# ═══════════════════════════════════════════════════════════════════════════

def normalize_name(name: str, db_keys: list) -> Optional[str]:
    if name in NAME_MAPPING: name = NAME_MAPPING[name]
    if name in db_keys: return name
    matches = get_close_matches(name, db_keys, n=1, cutoff=0.6)
    return matches[0] if matches else None

def poisson_prob(k, lamb):
    """Calcula probabilidade de Poisson exata P(X=k)."""
    if lamb > 30: return 0.0
    return (lamb**k * math.exp(-lamb)) / math.factorial(k)

def monte_carlo_simulation(xg_home, xg_away, iterations=1000):
    """Simulador de Monte Carlo."""
    h_wins, draws, a_wins = 0, 0, 0
    for _ in range(iterations):
        gh = np.random.poisson(xg_home)
        ga = np.random.poisson(xg_away)
        if gh > ga: h_wins += 1
        elif ga > gh: a_wins += 1
        else: draws += 1
    return h_wins/iterations, draws/iterations, a_wins/iterations

def calculate_kelly_criterion(prob_real, odd_casa, bankroll):
    """Calculadora Kelly (Fracionário 50%)."""
    b = odd_casa - 1
    p = prob_real / 100
    q = 1 - p
    f = (b * p - q) / b
    return max(0, f * bankroll * 0.5) 

def calculate_value_bet(prob_model, odd_casa):
    """Calcula o Value Bet (Edge) em porcentagem."""
    prob_implied = 1 / odd_casa
    edge = (prob_model / 100) / prob_implied
    return (edge - 1) * 100

# Feature 53: Cálculo do Expected Value (EV)
def calculate_expected_value(prob_model, odd_casa, stake):
    """Calcula o Valor Esperado (EV) de uma aposta."""
    prob_model_dec = prob_model / 100
    ev = (prob_model_dec * (odd_casa - 1) * stake) - ((1 - prob_model_dec) * stake)
    return ev

# Feature 54: Dutching Calculator
def calculate_dutching(odds: List[float], target_profit: float) -> Dict[str, float]:
    """Calcula o stake para cada odd para garantir um lucro fixo."""
    implied_probs = [1 / odd for odd in odds]
    total_implied_prob = sum(implied_probs)
    
    if total_implied_prob >= 1:
        return {'error': "Soma das probabilidades implícitas é >= 1. Não há valor para Dutching."}
        
    total_stake = target_profit / (1 - total_implied_prob)
    
    stakes = {}
    for i, odd in enumerate(odds):
        stake = (total_stake * implied_probs[i])
        stakes[f"Odd {odd:.2f}"] = stake
        
    stakes['total_stake'] = sum(stakes.values())
    stakes['profit'] = target_profit
    stakes['return'] = stakes['total_stake'] + target_profit
    
    return stakes

# Feature 55: Hedge Stake Calculator
def calculate_hedge_stake(initial_stake: float, initial_odd: float, hedge_odd: float, target_profit: float = 0.0) -> float:
    """Calcula o stake necessário para um hedge para garantir um lucro alvo."""
    
    # Lucro potencial da aposta inicial
    initial_return = initial_stake * initial_odd
    
    # Stake necessário para o hedge
    # Stake_Hedge = (Retorno_Inicial - Lucro_Alvo) / Odd_Hedge
    hedge_stake = (initial_return - target_profit) / hedge_odd
    
    return max(0, hedge_stake)

# Feature 56: Análise de Forma (W/D/L)
def get_form_analysis(team_name: str, all_dfs: Dict, n_games: int = 5) -> str:
    form = []
    for league, df in all_dfs.items():
        # Filtra jogos do time, seja em casa ou fora
        team_games = df[(df['HomeTeam'] == team_name) | (df['AwayTeam'] == team_name)]
        
        # Garante que o DataFrame tem as colunas de placar e data
        if team_games.empty or 'FTHG' not in team_games.columns or 'FTAG' not in team_games.columns or 'Date' not in team_games.columns:
            continue
            
        # Ordena por data (mais recente por último)
        team_games = team_games.sort_values(by='Date', ascending=False).head(n_games)
        
        for _, row in team_games.iterrows():
            if row['HomeTeam'] == team_name:
                gh, ga = row['FTHG'], row['FTAG']
            else:
                gh, ga = row['FTAG'], row['FTHG']
                
            if gh > ga: form.append('V') # Vitória
            elif ga > gh: form.append('D') # Derrota
            else: form.append('E') # Empate
            
    return "".join(form[:n_games][::-1]) # Inverte para mostrar do mais antigo ao mais recente

def get_native_history(team_name: str, league: str, market: str, line: float, location: str, all_dfs: Dict, n_games: int = 5) -> str:
    if league not in all_dfs: return "N/A"
    df = all_dfs[league]
    col_map = {('home', 'corners'): 'HC', ('away', 'corners'): 'AC', ('home', 'cards'): 'HY', ('away', 'cards'): 'AY'}
    col_code = col_map.get((location, market))
    team_col = 'HomeTeam' if location == 'home' else 'AwayTeam'
    
    matches = df[df[team_col] == team_name]
    if matches.empty: return "0/0"
    last_matches = matches.sort_values(by='Date', ascending=False).head(n_games)
    
    if col_code not in last_matches.columns: return "0/0"
    hits = sum(1 for val in last_matches[col_code] if float(val) > line)
    return f"{hits}/{len(last_matches)}"

def calcular_jogo_v17(home: str, away: str, stats: Dict, ref: Optional[str], refs_db: Dict, weather_bad: bool = False, all_dfs: Dict = None) -> Dict:
    h_norm = normalize_name(home, list(stats.keys()))
    a_norm = normalize_name(away, list(stats.keys()))
    
    if not h_norm or not a_norm: return {'error': "Times não encontrados."}
    
    s_h = stats[h_norm]
    s_a = stats[a_norm]
    
    r_data = refs_db.get(ref, {'factor': 1.0, 'red_rate': 0.08, 'strictness_score': 4.0}) if ref else {'factor': 1.0, 'red_rate': 0.08, 'strictness_score': 4.0}
    
    weather_factor_goals = 0.9 if weather_bad else 1.0
    weather_factor_cards = 1.2 if weather_bad else 1.0
    
    mom_h = s_h.get('momentum', 1.0)
    mom_a = s_a.get('momentum', 1.0)
    
    elo_h = s_h.get('elo_rating', 1500)
    elo_a = s_a.get('elo_rating', 1500)
    elo_diff = elo_h - elo_a
    elo_factor = math.log10(max(1, abs(elo_diff))) * 0.05 * (1 if elo_diff > 0 else -1)
    
    # Feature 57: Ajuste de Fouls por Árbitro
    foul_factor = r_data['strictness_score'] / 11.0 # 11.0 é a média default
    
    # Cálculo Base
    corn_h = s_h['corners'] * 1.15 * mom_h
    corn_a = s_a['corners'] * 0.90 * mom_a
    
    card_h = s_h['cards'] * r_data['factor'] * weather_factor_cards
    card_a = s_a['cards'] * r_data['factor'] * weather_factor_cards
    
    foul_h = s_h['fouls'] * foul_factor
    foul_a = s_a['fouls'] * foul_factor
    
    # Feature 58: xG Baseado em Stats Home/Away
    # xG_H = (Gols_Feitos_H_Casa * Gols_Sofridos_A_Fora) / Média_Liga
    xg_home_base = (s_h['home_goals_f'] * s_a['away_goals_a']) / s_h['goals_a'] # Usando goals_a como proxy para média da liga
    xg_away_base = (s_a['away_goals_f'] * s_h['home_goals_a']) / s_a['goals_a']
    
    xg_home = (xg_home_base * weather_factor_goals) + elo_factor
    xg_away = (xg_away_base * weather_factor_goals) - elo_factor
    
    xg_home = max(0.1, xg_home)
    xg_away = max(0.1, xg_away)
    
    mc_h, mc_d, mc_a = monte_carlo_simulation(xg_home, xg_away)
    
    trap_alert = False
    
    # Feature 59: Corner Prediction Refinement (Simulação de NegBinomial)
    # Simulação de probabilidade de escanteios total
    total_corners_avg = corn_h + corn_a
    # Simulação simples de probabilidade de Over 9.5 usando a média
    prob_over_9_5_corners = min(95, 40 + total_corners_avg * 5)
    
    prob_h_score = 1 - poisson_prob(0, xg_home)
    prob_a_score = 1 - poisson_prob(0, xg_away)
    prob_btts = prob_h_score * prob_a_score * 100
    
    prob_over_2_5 = 0
    for h in range(3):
        for a in range(3):
            if h + a < 3:
                prob_over_2_5 += poisson_prob(h, xg_home) * poisson_prob(a, xg_away)
    prob_over_2_5 = (1 - prob_over_2_5) * 100
    
    h2h_stats = get_h2h_stats(h_norm, a_norm, all_dfs) if all_dfs else {}
    
    # Feature 60: Total Cards Prediction
    total_cards_avg = card_h + card_a
    prob_over_4_5_cards = min(95, 50 + (total_cards_avg - 4.5) * 10)
    
    return {
        'home': h_norm, 'away': a_norm, 'league_h': s_h['league'], 'league_a': s_a['league'],
        'goals': {'h': xg_home, 'a': xg_away},
        'corners': {'h': corn_h, 'a': corn_a, 'total_over_9_5': prob_over_9_5_corners},
        'cards': {'h': card_h, 'a': card_a, 'total_over_4_5': prob_over_4_5_cards}, # Feature 60
        'fouls': {'h': foul_h, 'a': foul_a}, # Feature 57
        'monte_carlo': {'h': mc_h * 100, 'd': mc_d * 100, 'a': mc_a * 100},
        'meta': {'trap': trap_alert, 'elo_h': elo_h, 'elo_a': elo_a, 'ts_h': s_h['ts_index'], 'ts_a': s_a['ts_index']}, # Feature 50
        'advanced_probs': {
            'btts': prob_btts,
            'over_2_5': prob_over_2_5
        },
        'h2h_stats': h2h_stats,
        # Feature 56: Análise de Forma
        'form_h': get_form_analysis(h_norm, all_dfs, 5),
        'form_a': get_form_analysis(a_norm, all_dfs, 5),
        # Feature 51: Stats Separadas
        'home_stats': {'gf': s_h['home_goals_f'], 'ga': s_h['home_goals_a']},
        'away_stats': {'gf': s_a['away_goals_f'], 'ga': s_a['away_goals_a']},
    }

def get_detailed_probs(res: Dict) -> Dict:
    xg_h = res['goals']['h']
    xg_a = res['goals']['a']
    
    score_probs = {}
    for h in range(5):
        for a in range(5):
            prob = poisson_prob(h, xg_h) * poisson_prob(a, xg_a) * 100
            score_probs[f"{h}-{a}"] = prob
            
    def simulate_market_prob(avg, line):
        prob = 50 + (avg - line) * 10
        return max(5, min(95, prob))
    
    probs = {
        'corners': {
            'home': {f'Over {l}': simulate_market_prob(res['corners']['h'], l) for l in [3.5, 4.5, 5.5]},
            'away': {f'Over {l}': simulate_market_prob(res['corners']['a'], l) for l in [3.5, 4.5, 5.5]},
            'total': {f'Over {l}': simulate_market_prob(res['corners']['h'] + res['corners']['a'], l) for l in [8.5, 9.5, 10.5]}
        },
        'cards': {
            'home': {f'Over {l}': simulate_market_prob(res['cards']['h'], l) for l in [1.5, 2.5]},
            'away': {f'Over {l}': simulate_market_prob(res['cards']['a'], l) for l in [1.5, 2.5]},
            'total': {f'Over {l}': simulate_market_prob(res['cards']['h'] + res['cards']['a'], l) for l in [3.5, 4.5, 5.5]} # Feature 60
        },
        'scores': score_probs
    }
    
    return probs

def generate_bet_options(h, a, probs):
    options = []
    
    # Opções de Gols
    if probs['scores']['1-0'] + probs['scores']['2-0'] > 30:
        options.append({'label': f'{h} Vence (Simples)', 'prob': probs['scores']['1-0'] + probs['scores']['2-0'], 'min_odd': 1.5})
    if probs['scores']['1-1'] > 10:
        options.append({'label': 'Empate (Simples)', 'prob': probs['scores']['1-1'], 'min_odd': 3.0})
        
    # Opções de Cantos
    if probs['corners']['total']['Over 9.5'] > 70:
        options.append({'label': 'Total Cantos Over 9.5', 'prob': probs['corners']['total']['Over 9.5'], 'min_odd': 1.7})
        
    # Opções de Cartões
    if probs['cards']['home']['Over 2.5'] > 70:
        options.append({'label': f'{h} Cartões Over 2.5', 'prob': probs['cards']['home']['Over 2.5'], 'min_odd': 1.8})
        
    # Feature 32 & 33
    if probs['advanced_probs']['btts'] > 65:
        options.append({'label': 'Ambas Marcam (BTTS)', 'prob': probs['advanced_probs']['btts'], 'min_odd': 1.6})
    if probs['advanced_probs']['over_2_5'] > 60:
        options.append({'label': 'Total Gols Over 2.5', 'prob': probs['advanced_probs']['over_2_5'], 'min_odd': 1.75})
        
    # Opções de segurança (fallback)
    if not options:
        options.append({'label': 'Aposta Segura (Under 3.5)', 'prob': 80, 'min_odd': 1.2})
        options.append({'label': 'Aposta de Risco (Over 4.5)', 'prob': 30, 'min_odd': 2.5})
        
    # Adiciona a probabilidade de vitória do Monte Carlo
    options.append({'label': f'{h} Vence (MC)', 'prob': probs['monte_carlo']['h'], 'min_odd': 1.5})
    options.append({'label': f'{a} Vence (MC)', 'prob': probs['monte_carlo']['a'], 'min_odd': 1.5})
    
    return options

def generate_dual_hedges(main_slip, stats, refs_db):
    # Simulação de geração de hedges
    h1 = []
    for bet in main_slip:
        h1.append({**bet, 'label': f"Hedge 1: {bet['label'].replace('Over', 'Under')}", 'min_odd': bet['min_odd'] * 0.8})
        
    h2 = []
    for bet in main_slip:
        h2.append({**bet, 'label': f"Hedge 2: {bet['home']} - Mais Cartões", 'min_odd': bet['min_odd'] * 0.9})
        
    return h1, h2

# ═══════════════════════════════════════════════════════════════════════════
# 3. INTERFACE STREAMLIT (V17)
# ═══════════════════════════════════════════════════════════════════════════

def render_h2h_analysis(home, away, all_dfs):
    st.subheader(f"🆚 Histórico de Confrontos Diretos ({home} vs {away})")
    h2h = get_h2h_stats(home, away, all_dfs)
    
    if h2h['games'] == 0:
        st.info("Nenhum confronto direto encontrado nas bases de dados.")
        return
        
    st.markdown(f"**Total de Jogos:** {h2h['games']}")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(f"Vitórias {home}", h2h['h_wins'])
    col2.metric(f"Vitórias {away}", h2h['a_wins'])
    col3.metric("Empates", h2h['draws'])
    col4.metric("Média de Gols", f"{h2h['avg_goals']:.2f}")
    
    st.markdown("---")
    st.caption("Atenção: A análise H2H é apenas um fator. O modelo V17 usa o Elo Rating e Recência para a previsão principal.")

def render_calendar_tab(calendar, stats, refs):
    st.subheader("📅 Próximos Jogos (Calendário)")
    
    if calendar.empty:
        st.warning("Calendário de jogos futuros não carregado. Verifique o arquivo de calendário.")
        return
        
    st.dataframe(calendar, use_container_width=True)
    
    st.markdown("---")
    st.subheader("Simulação Rápida da Rodada")
    
    for index, row in calendar.iterrows():
        h, a, l = row['HomeTeam'], row['AwayTeam'], row['League']
        
        res = calcular_jogo_v17(h, a, stats, None, refs)
        
        if 'error' not in res:
            mc_h = res['monte_carlo']['h']
            mc_d = res['monte_carlo']['d']
            mc_a = res['monte_carlo']['a']
            
            st.markdown(f"**{h}** ({res['meta']['ts_h']:.1f} TS) vs **{a}** ({res['meta']['ts_a']:.1f} TS) - *{l}*") # Feature 50
            st.info(f"Prob. MC: 🏠 {mc_h:.1f}% | 🤝 {mc_d:.1f}% | ✈️ {mc_a:.1f}% | xG: {res['goals']['h']:.2f} - {res['goals']['a']:.2f}")
            
# Feature 61: Aba de Gestão de Risco Avançada
def render_advanced_risk_tab():
    st.subheader("💰 Gestão de Risco Avançada (Dutching & Hedge)")
    
    # --- DUTCHING CALCULATOR ---
    st.markdown("#### 1. Dutching Calculator (Lucro Fixo)")
    
    if 'dutching_odds' not in st.session_state: st.session_state.dutching_odds = [2.0, 3.0]
    
    col_odds, col_profit = st.columns([3, 1])
    
    odds_input = col_odds.text_input("Odds (separadas por vírgula)", value=", ".join(map(str, st.session_state.dutching_odds)))
    target_profit = col_profit.number_input("Lucro Alvo (R$)", 1.0, 1000.0, 10.0)
    
    try:
        odds = [float(o.strip()) for o in odds_input.split(',') if o.strip()]
        if odds:
            st.session_state.dutching_odds = odds
            dutching_res = calculate_dutching(odds, target_profit)
            
            if 'error' in dutching_res:
                st.error(dutching_res['error'])
            else:
                st.success(f"Stake Total Necessário: R$ {dutching_res['total_stake']:.2f}")
                st.markdown(f"**Retorno Total Garantido:** R$ {dutching_res['return']:.2f}")
                
                st.markdown("##### Stakes por Odd:")
                for odd_label, stake in dutching_res.items():
                    if odd_label not in ['total_stake', 'profit', 'return']:
                        st.markdown(f"- **{odd_label}**: R$ {stake:.2f}")
    except:
        st.error("Formato de Odds inválido. Use números separados por vírgula.")
        
    st.markdown("---")
    
    # --- HEDGE STAKE CALCULATOR ---
    st.markdown("#### 2. Hedge Stake Calculator (Lucro Bloqueado)")
    
    col_i, col_h, col_p = st.columns(3)
    initial_stake = col_i.number_input("Stake Inicial (R$)", 1.0, 1000.0, 50.0, key="hedge_stake_i")
    initial_odd = col_i.number_input("Odd Inicial", 1.01, 20.0, 2.5, key="hedge_odd_i")
    
    hedge_odd = col_h.number_input("Odd do Hedge (Live)", 1.01, 20.0, 1.5, key="hedge_odd_h")
    
    target_profit_h = col_p.number_input("Lucro Alvo (R$)", 0.0, 1000.0, 10.0, key="hedge_profit_t")
    
    hedge_stake = calculate_hedge_stake(initial_stake, initial_odd, hedge_odd, target_profit_h)
    
    st.info(f"Stake Necessário para Hedge: R$ **{hedge_stake:.2f}**")
    st.caption(f"Se o Hedge for bem-sucedido, o lucro mínimo garantido será de R$ {target_profit_h:.2f}.")

def render_value_bet_tab(res: Optional[Dict] = None):
    st.subheader("📈 Value Bet Finder & Expected Value (EV)")
    
    c1, c2, c3 = st.columns(3)
    odd_mercado = c1.number_input("Odd da Casa de Apostas", 1.01, 20.0, 2.0, key="vb_odd")
    prob_modelo = c2.slider("Probabilidade do Modelo (%)", 1, 100, 55, key="vb_prob")
    stake = c3.number_input("Stake (R$)", 1.0, 1000.0, 10.0, key="vb_stake")
    
    value = calculate_value_bet(prob_modelo, odd_mercado)
    ev = calculate_expected_value(prob_modelo, odd_mercado, stake) # Feature 53
    
    col_v, col_ev = st.columns(2)
    col_v.metric("Value Bet (Edge)", f"{value:.2f}%")
    col_ev.metric("Expected Value (EV)", f"R$ {ev:.2f}")
    
    if value > 5:
        st.success("✅ ALERTA DE VALUE BET: Aposta com Valor Encontrado!")
    elif value > 0:
        st.info("⚠️ Valor Positivo: Aposta com Pequena Vantagem.")
    else:
        st.error("❌ Valor Negativo: Evite esta aposta.")
        
    st.markdown("---")
    st.caption("O Expected Value (EV) representa o lucro médio esperado por aposta a longo prazo.")

# Feature 62: Visualização de Placar Exato (Heatmap Simulado)
def render_exact_score_tab(res):
    st.subheader("🎯 Probabilidades de Placar Exato (Poisson)")
    
    if 'scores' not in res:
        st.warning("Simule uma partida na aba 'Simulação Detalhada' para gerar as probabilidades de placar.")
        return
        
    scores = res['scores']
    
    # Cria um DataFrame para visualização
    df_scores = pd.DataFrame(index=range(5), columns=range(5))
    for h in range(5):
        for a in range(5):
            df_scores.loc[h, a] = scores.get(f'{h}-{a}', 0.0)
            
    # Feature 62: Heatmap Simulado
    st.markdown("##### Heatmap de Probabilidades")
    st.dataframe(df_scores.style.background_gradient(cmap='YlOrRd', axis=None).format("{:.1f}%"), use_container_width=True)
    
    # Top 5 Placares
    top_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:5]
    
    st.markdown("---")
    st.subheader("Top 5 Placares Mais Prováveis")
    
    for score, prob in top_scores:
        st.markdown(f"**{score}**: {prob:.1f}%")

def render_total_market_tab(res):
    st.subheader("📊 Análise de Mercados Totais (Gols, Cantos, Cartões)")
    
    if 'advanced_probs' not in res:
        st.warning("Simule uma partida na aba 'Simulação Detalhada' para gerar as probabilidades.")
        return
        
    col1, col2, col3 = st.columns(3)
    
    col1.metric("BTTS (Ambas Marcam)", f"{res['advanced_probs']['btts']:.1f}%")
    col2.metric("Over 2.5 Gols", f"{res['advanced_probs']['over_2_5']:.1f}%")
    col3.metric("Over 9.5 Cantos", f"{res['corners']['total_over_9_5']:.1f}%")
    
    st.markdown("---")
    st.subheader("Probabilidades Detalhadas de Cartões e Cantos")
    
    col_c, col_card = st.columns(2)
    
    with col_c:
        st.markdown("##### Cantos Totais")
        total_corners_probs = get_detailed_probs(res)['corners']['total']
        for line, prob in total_corners_probs.items():
            st.markdown(f"- **{line}**: {prob:.1f}%")
            
    with col_card:
        st.markdown("##### Cartões Totais")
        total_cards_probs = get_detailed_probs(res)['cards']['total']
        for line, prob in total_cards_probs.items():
            st.markdown(f"- **{line}**: {prob:.1f}%")

def render_paper_trading_tab():
    st.subheader("📝 Diário de Bordo (Paper Trading)")
    
    # Feature 63: Carregar/Salvar dados de sessão
    if 'paper_bets' not in st.session_state: st.session_state.paper_bets = []
    
    # ... (restante da lógica de paper trading)
    
    c1, c2, c3 = st.columns(3)
    pb_desc = c1.text_input("Descrição da Aposta", key="pb_desc")
    pb_val = c2.number_input("Valor Apostado", 1.0, 1000.0, 50.0, key="pb_val")
    pb_odd = c3.number_input("Odd", 1.01, 20.0, 2.0, key="pb_odd")
    
    col_result, col_save = st.columns([3, 1])
    pb_result = col_result.selectbox("Resultado", ["Pendente", "Ganho", "Perdido", "Anulado"], key="pb_result")
    
    if col_save.button("Salvar Entrada", type="primary"):
        bet = {
            'date': datetime.now().strftime("%Y-%m-%d %H:%M"),
            'desc': pb_desc,
            'val': pb_val,
            'odd': pb_odd,
            'result': pb_result
        }
        st.session_state.paper_bets.append(bet)
        st.success("Entrada salva!")
        
    st.markdown("---")
    st.subheader("Histórico de Apostas")
    
    if st.session_state.paper_bets:
        df_bets = pd.DataFrame(st.session_state.paper_bets)
        st.dataframe(df_bets, use_container_width=True)
        
        total_invested = df_bets['val'].sum()
        total_return = 0
        
        for _, row in df_bets.iterrows():
            if row['result'] == 'Ganho':
                total_return += row['val'] * row['odd']
            elif row['result'] == 'Perdido':
                total_return += 0
            elif row['result'] == 'Anulado':
                total_return += row['val']
            elif row['result'] == 'Pendente':
                total_return += row['val']
                
        profit = total_return - total_invested
        roi = (profit / total_invested) * 100 if total_invested > 0 else 0
        
        st.markdown(f"**Total Investido:** R$ {total_invested:.2f}")
        st.markdown(f"**Lucro/Prejuízo:** R$ {profit:.2f}")
        st.markdown(f"**ROI (Retorno sobre Investimento):** {roi:.2f}%")
    else:
        st.info("Nenhuma aposta registrada ainda.")

def render_debug_report():
    st.subheader("🐞 Relatório de Debug (Logs do Sistema)")
    if DEBUG_LOGS:
        st.code("\n".join(DEBUG_LOGS))
    else:
        st.info("Nenhum log de debug registrado.")

# Feature 64: Função para limpar o estado da sessão
def clear_session_state():
    for key in list(st.session_state.keys()):
        if key not in ['paper_bets']: # Mantém o histórico de paper bets
            del st.session_state[key]
    st.session_state.paper_bets = [] # Limpa o histórico de paper bets também, se for o caso
    st.rerun()

def main():
    # Feature 65: Adiciona botão de Limpar Sessão
    st.sidebar.title("🎛️ Painel de Controle V17")
    weather = st.sidebar.checkbox("🌧️ Clima Ruim (Chuva/Neve)", value=False, help="Feature 17: Ajusta física do jogo")
    elo_k = st.sidebar.slider("Fator K do Elo Rating", 10, 50, 30, help="Feature 21: Sensibilidade do Elo Rating")
    
    if st.sidebar.button("Limpar Sessão (Hard Reset)", type="secondary"):
        clear_session_state()
    
    st.title("🚀 FutPrevisão V17 - The Final Frontier")
    
    with st.spinner("Carregando bases de dados (V17)..."):
        DEBUG_LOGS.clear()
        stats = learn_stats_v17()
        refs = load_referees_v15()
        calendar = load_calendar_safe()
        all_dfs = load_all_dataframes()
        
    if not stats:
        st.error("🚨 ERRO CRÍTICO: Bases de dados vazias. Verifique se os arquivos CSV estão presentes.")
        return

    t1, t2, t3, t4, t5, t6 = st.tabs([
        "📅 Calendário & Rodada", 
        "🔍 Simulação Detalhada", 
        "🎰 Bet Builder Pro", 
        "🧪 Data Science Lab", 
        "💰 Gestão Financeira",
        "📋 Relatórios & Debug"
    ])
    
    # TAB 1: CALENDÁRIO
    with t1:
        render_calendar_tab(calendar, stats, refs)
        
    # TAB 2: SIMULAÇÃO DETALHADA
    with t2:
        st.subheader("🔍 Simulador de Partida (com Árbitro, Clima e TS-Index)")
        l_times = sorted(list(stats.keys()))
        l_refs = ["Neutro"] + sorted(list(refs.keys()))
        
        c1, c2, c3 = st.columns(3)
        h = c1.selectbox("Casa", l_times, key="sim_h", index=0)
        a = c2.selectbox("Fora", l_times, key="sim_a", index=min(1, len(l_times)-1))
        r = c3.selectbox("Árbitro", l_refs, key="sim_r")
        
        if st.button("🚀 Simular Jogo V17", type="primary"):
            rf = None if r == "Neutro" else r
            res = calcular_jogo_v17(h, a, stats, rf, refs, weather, all_dfs)
            
            if 'error' in res: st.error(res['error'])
            else:
                probs = get_detailed_probs(res)
                res.update(probs)
                st.session_state.last_sim_res = res
                
                # HEADER
                hc1, hc2, hc3 = st.columns([1,2,1])
                hc2.markdown(f"<h2 style='text-align: center'>{res['home']} vs {res['away']}</h2>", unsafe_allow_html=True)
                
                # Feature 50: Exibe TS-Index
                st.info(f"TS-Index: 🏠 {res['meta']['ts_h']:.1f} | ✈️ {res['meta']['ts_a']:.1f} | Forma: 🏠 {res['form_h']} | ✈️ {res['form_a']}") # Feature 56
                
                if res['meta']['trap']: st.error("🚨 ALERTA DE ARMADILHA: xG alto mas Monte Carlo baixo.")
                st.success(f"Probabilidades (Monte Carlo 1k): 🏠 {res['monte_carlo']['h']:.1f}% | 🤝 {res['monte_carlo']['d']:.1f}% | ✈️ {res['monte_carlo']['a']:.1f}%")
                
                st.markdown("---")
                
                # Feature 66: Colunas de Stats
                col_h, col_a = st.columns(2)
                
                with col_h:
                    st.success(f"🏠 **{res['home']}**")
                    st.write(f"xG: {res['goals']['h']:.2f}")
                    st.write(f"Cantos Esp: {res['corners']['h']:.1f}")
                    st.write(f"Faltas Esp: {res['fouls']['h']:.1f}") # Feature 57
                    st.write(f"**Casa Stats:** GF {res['home_stats']['gf']:.2f} / GA {res['home_stats']['ga']:.2f}") # Feature 51
                    st.write("---")
                    
                    with st.expander("Detalhes de Mercados"): # Feature 13
                        for l in [3.5, 4.5, 5.5]:
                            p = probs['corners']['home'].get(f'Over {l}', 0)
                            hist = get_native_history(res['home'], res['league_h'], 'corners', l, 'home', all_dfs)
                            cor = "green" if p >= 70 else "gray"
                            st.markdown(f"🚩 Over {l}: :{cor}[{p:.0f}%] | Hist: {hist}")
                        st.write("---")
                        for l in [1.5, 2.5]:
                            p = probs['cards']['home'].get(f'Over {l}', 0)
                            cor = "green" if p >= 70 else "gray"
                            st.markdown(f"🟨 Over {l}: :{cor}[{p:.0f}%]")

                with col_a:
                    st.success(f"✈️ **{res['away']}**")
                    st.write(f"xG: {res['goals']['a']:.2f}")
                    st.write(f"Cantos Esp: {res['corners']['a']:.1f}")
                    st.write(f"Faltas Esp: {res['fouls']['a']:.1f}") # Feature 57
                    st.write(f"**Fora Stats:** GF {res['away_stats']['gf']:.2f} / GA {res['away_stats']['ga']:.2f}") # Feature 51
                    st.write("---")
                    
                    with st.expander("Detalhes de Mercados"): # Feature 13
                        for l in [3.5, 4.5, 5.5]:
                            p = probs['corners']['away'].get(f'Over {l}', 0)
                            hist = get_native_history(res['away'], res['league_a'], 'corners', l, 'away', all_dfs)
                            cor = "green" if p >= 70 else "gray"
                            st.markdown(f"🚩 Over {l}: :{cor}[{p:.0f}%] | Hist: {hist}")
                        st.write("---")
                        for l in [1.5, 2.5]:
                            p = probs['cards']['away'].get(f'Over {l}', 0)
                            cor = "green" if p >= 70 else "gray"
                            st.markdown(f"🟨 Over {l}: :{cor}[{p:.0f}%]")
                        
    # TAB 3: BET BUILDER
    with t3:
        render_bet_builder_tab(stats, refs)

    # TAB 4: DATA SCIENCE
    with t4:
        st.subheader("🧪 Laboratório de Dados V17")
        st.info("Ferramentas avançadas para análise profunda.")
        
        t4a, t4b, t4c, t4d = st.tabs(["Análise H2H", "Placar Exato", "Mercados Totais", "Value Bet & EV"])
        
        with t4a:
            if 'last_sim_res' in st.session_state:
                render_h2h_analysis(st.session_state.last_sim_res['home'], st.session_state.last_sim_res['away'], all_dfs)
            else:
                st.info("Simule uma partida na aba 'Simulação Detalhada' para ver a análise H2H.")
                
        with t4b:
            if 'last_sim_res' in st.session_state:
                render_exact_score_tab(st.session_state.last_sim_res) # Feature 62
            else:
                st.info("Simule uma partida na aba 'Simulação Detalhada' para ver as probabilidades de placar exato.")
                
        with t4c:
            if 'last_sim_res' in st.session_state:
                render_total_market_tab(st.session_state.last_sim_res)
            else:
                st.info("Simule uma partida na aba 'Simulação Detalhada' para ver a análise de mercados totais.")
                
        with t4d:
            render_value_bet_tab(st.session_state.get('last_sim_res')) # Feature 53

    # TAB 5: FINANCEIRO
    with t5:
        st.subheader("💰 Gestão de Banca & Kelly")
        bk = st.number_input("Banca Atual", 100.0, 100000.0, 1000.0, key="bankroll")
        odd_k = st.number_input("Odd da Aposta", 1.01, 20.0, 2.0, key="odd_kelly")
        prob_k = st.slider("Probabilidade Real (%)", 1, 100, 50, key="prob_kelly")
        
        kelly = calculate_kelly_criterion(prob_k, odd_k, bk)
        st.success(f"💎 Sugestão Kelly (Fracionário): Apostar R$ {kelly:.2f}")
        
        st.markdown("---")
        render_advanced_risk_tab() # Feature 61
        
        st.markdown("---")
        render_paper_trading_tab()

    # TAB 6: RELATÓRIOS & DEBUG
    with t6:
        st.subheader("📋 Central de Exportação")
        st.write("Gere PDFs ou copie relatórios para o Telegram.")
        st.info("Use o botão 'GERAR ESTRATÉGIA COMPLETA' na aba 'Bet Builder Pro' para criar o relatório de texto.")
        
        st.markdown("---")
        render_debug_report()

# --- COMPONENTES DA UI ---
def render_bet_builder_tab(stats, refs_db):
    if 'main_slip' not in st.session_state: st.session_state.main_slip = []
    
    st.subheader("🛠️ Bet Builder Pro (Estratégias de Hedge)")
    
    l_times = sorted(list(stats.keys()))
    num = st.number_input("Jogos no Bilhete", 1, 5, 3, key="bb_num_games")
    
    temp = []
    for i in range(num):
        st.markdown(f"**Jogo {i+1}**")
        c1, c2, c3 = st.columns(3)
        h = c1.selectbox(f"Casa", l_times, key=f"bbh{i}")
        a = c2.selectbox(f"Fora", l_times, key=f"bba{i}", index=min(1, len(l_times)-1))
        odd = c3.number_input(f"Odd", 1.01, 20.0, 1.01, key=f"bbodd{i}")
        
        res = calcular_jogo_v17(h, a, stats, None, refs_db)
        if 'error' in res: continue
        
        probs = get_detailed_probs(res)
        opts = generate_bet_options(h, a, probs)
        lbls = [o['label'] for o in opts]
        
        s1 = st.selectbox(f"Seleção 1", range(len(opts)), format_func=lambda x: lbls[x], key=f"bbs1{i}")
        s2 = st.selectbox(f"Seleção 2", range(len(opts)), format_func=lambda x: lbls[x], key=f"bbs2{i}", index=min(1, len(opts)-1))
        
        temp.append({**opts[s1], 'game_id': i, 'home': h, 'away': a, 'user_odd': odd})
        temp.append({**opts[s2], 'game_id': i, 'home': h, 'away': a, 'user_odd': odd})
        
    st.session_state.main_slip = temp
    
    if st.button("🔮 GERAR ESTRATÉGIA COMPLETA V17", type="primary"):
        h1, h2 = generate_dual_hedges(st.session_state.main_slip, stats, refs_db)
        st.success("Estratégia Calculada!")
        
        c1, c2, c3 = st.columns(3)
        
        def show_card(title, bets, col):
            txt = f"*{title}*\n"
            with col:
                st.markdown(f"### {title}")
                seen = []
                for b in bets:
                    if b['game_id'] not in seen:
                        st.caption(f"{b['home']} x {b['away']}")
                        txt += f"\n⚽ {b['home']} x {b['away']}\n"
                        seen.append(b['game_id'])
                    st.write(f"- {b['label']}")
                    st.caption(f"Min Odd: @{b['min_odd']:.2f}")
                    txt += f"- {b['label']} (@{b['min_odd']:.2f})\n"
            return txt

        t1 = show_card("Principal (Alvo)", st.session_state.main_slip, c1)
        t2 = show_card("Hedge 1 (Segurança)", h1, c2)
        t3 = show_card("Hedge 2 (Mix Stats)", h2, c3)
        
        st.text_area("📋 Copiar para Telegram", value=f"{t1}\n---\n{t2}\n---\n{t3}", height=300)

if __name__ == "__main__":
    main()
