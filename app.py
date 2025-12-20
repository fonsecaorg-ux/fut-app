"""
╔═══════════════════════════════════════════════════════════════════════════╗
║       FUTPREVISÃO V16.0 - THE QUANTUM MACHINE (35 FEATURES + STABLE)      ║
║                          Sistema Profissional de Apostas                   ║
║                                                                            ║
║  Versão: V16.0 Ultimate                                                   ║
║  Funcionalidades: 35 (Incluindo H2H, Elo Rating, Value Bet, IA de Escanteio)║
║  Correção Crítica: Erro de Variável Global (DEBUG_LOGS)                   ║
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

# Variável Global para Logs de Debug (Correção do Erro da Linha 410)
DEBUG_LOGS = []

# Configuração da Página
st.set_page_config(
    page_title="FutPrevisão V16 Quantum Machine",
    page_icon="⚛️",
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
# 1. CARREGAMENTO DE DADOS (SAFE LOAD)
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

# Feature 21: Elo Rating System (Novo)
def calculate_elo(df: pd.DataFrame, K=30) -> Dict[str, float]:
    elo_ratings = {}
    teams = set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique())
    for team in teams:
        elo_ratings[team] = 1500  # Elo inicial

    for index, row in df.iterrows():
        team_h = row['HomeTeam']
        team_a = row['AwayTeam']
        
        # Garante que as colunas de placar existam
        if 'FTHG' not in row or 'FTAG' not in row: continue
        
        elo_h = elo_ratings.get(team_h, 1500)
        elo_a = elo_ratings.get(team_a, 1500)

        # Resultado: 1 para vitória H, 0.5 para empate, 0 para vitória A
        if row['FTHG'] > row['FTAG']:
            result = 1
        elif row['FTHG'] == row['FTAG']:
            result = 0.5
        else:
            result = 0

        # Probabilidade de vitória esperada
        expected_h = 1 / (1 + 10**((elo_a - elo_h) / 400))
        
        # Atualização do Elo
        new_elo_h = elo_h + K * (result - expected_h)
        new_elo_a = elo_a + K * ((1 - result) - (1 - expected_h))
        
        elo_ratings[team_h] = new_elo_h
        elo_ratings[team_a] = new_elo_a
        
    return elo_ratings

@st.cache_data(ttl=3600)
def learn_stats_v16() -> Dict[str, Dict[str, Any]]:
    stats_db = {}
    all_dfs = load_all_dataframes()
    
    # Feature 21: Cálculo do Elo Rating
    global_elo = {}
    for league, df in all_dfs.items():
        # Concatena todos os jogos para um Elo global mais robusto
        global_elo.update(calculate_elo(df))

    for league in LIGAS_ALVO:
        df = find_and_load_csv(league)
        if df.empty: continue
        
        # Garante colunas mínimas (fallback para NaN)
        cols = ['HomeTeam', 'AwayTeam', 'HC', 'AC', 'HY', 'AY', 'HF', 'AF', 'FTHG', 'FTAG', 'HST', 'AST', 'HR', 'AR']
        for c in cols: 
            if c not in df.columns: df[c] = np.nan
        
        # Feature 22: Peso por Data (Recência)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.sort_values(by='Date', ascending=True).dropna(subset=['Date'])
            # Aplica um peso exponencial (mais recente > mais peso)
            df['RecencyWeight'] = np.exp(np.linspace(0, 1, len(df)))
            df['RecencyWeight'] = df['RecencyWeight'] / df['RecencyWeight'].sum() * len(df)
        else:
            df['RecencyWeight'] = 1.0
        
        try:
            # Agregação Simples para manter performance
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
                
                # Weighted Average (60% Mandante / 40% Visitante para stats gerais)
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
                    'momentum': np.random.uniform(0.9, 1.1), # Simulação de momentum se não tiver data
                    'elo_rating': global_elo.get(team, 1500) # Feature 21
                }
        except Exception as e:
            DEBUG_LOGS.append(f"Erro ao processar liga {league}: {e}")
            pass
    return stats_db

@st.cache_data(ttl=3600)
def load_referees_v15() -> Dict[str, Dict[str, float]]:
    refs_db = {}
    # Carrega CSVs se existirem
    files = ["arbitros_5_ligas_2025_2026.csv", "arbitros.csv"]
    for f in files:
        if os.path.exists(f):
            try:
                df = pd.read_csv(f)
                for _, row in df.iterrows():
                    # Tenta pegar colunas variadas
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

# Feature 24: Carregamento Seguro de Calendário
@st.cache_data(ttl=3600)
def load_calendar_safe() -> pd.DataFrame:
    # Simulação de carregamento de calendário
    # Em um ambiente real, isso carregaria um CSV de jogos futuros
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
# 2. MOTOR V16 (Monte Carlo + Poisson + Features Avançadas)
# ═══════════════════════════════════════════════════════════════════════════

def normalize_name(name: str, db_keys: list) -> Optional[str]:
    if name in NAME_MAPPING: name = NAME_MAPPING[name]
    if name in db_keys: return name
    matches = get_close_matches(name, db_keys, n=1, cutoff=0.6)
    return matches[0] if matches else None

def poisson_prob(k, lamb):
    """Calcula probabilidade de Poisson exata P(X=k)."""
    # Feature 25: Tratamento de Overflow
    if lamb > 30: return 0.0 # Limite prático para evitar overflow
    return (lamb**k * math.exp(-lamb)) / math.factorial(k)

def monte_carlo_simulation(xg_home, xg_away, iterations=1000):
    """Funcionalidade 11: Simulador de Monte Carlo."""
    h_wins, draws, a_wins = 0, 0, 0
    for _ in range(iterations):
        gh = np.random.poisson(xg_home)
        ga = np.random.poisson(xg_away)
        if gh > ga: h_wins += 1
        elif ga > gh: a_wins += 1
        else: draws += 1
    return h_wins/iterations, draws/iterations, a_wins/iterations

def calculate_kelly_criterion(prob_real, odd_casa, bankroll):
    """Funcionalidade 6: Calculadora Kelly."""
    b = odd_casa - 1
    p = prob_real / 100
    q = 1 - p
    # Feature 26: Kelly Otimizado (Kelly Fracionário 50%)
    f = (b * p - q) / b
    return max(0, f * bankroll * 0.5) 

# Feature 27: Cálculo de Value Bet
def calculate_value_bet(prob_model, odd_casa):
    """Calcula o Value Bet (Edge) em porcentagem."""
    prob_implied = 1 / odd_casa
    edge = (prob_model / 100) / prob_implied
    return (edge - 1) * 100

def get_native_history(team_name: str, league: str, market: str, line: float, location: str, all_dfs: Dict) -> str:
    if league not in all_dfs: return "N/A"
    df = all_dfs[league]
    col_map = {('home', 'corners'): 'HC', ('away', 'corners'): 'AC', ('home', 'cards'): 'HY', ('away', 'cards'): 'AY'}
    col_code = col_map.get((location, market))
    team_col = 'HomeTeam' if location == 'home' else 'AwayTeam'
    
    matches = df[df[team_col] == team_name]
    if matches.empty: return "0/0"
    # Feature 28: Histórico Recente (Últimos 5)
    last_matches = matches.tail(5) 
    if col_code not in last_matches.columns: return "0/0"
    hits = sum(1 for val in last_matches[col_code] if float(val) > line)
    return f"{hits}/{len(last_matches)}"

# Feature 29: Análise Head-to-Head (H2H)
def get_h2h_stats(home: str, away: str, all_dfs: Dict) -> Dict:
    h2h_stats = {'games': 0, 'h_wins': 0, 'a_wins': 0, 'draws': 0, 'avg_goals': 0.0}
    
    for league, df in all_dfs.items():
        # Jogos onde H foi mandante e A foi visitante
        df_h_vs_a = df[(df['HomeTeam'] == home) & (df['AwayTeam'] == away)]
        # Jogos onde A foi mandante e H foi visitante (para histórico completo)
        df_a_vs_h = df[(df['HomeTeam'] == away) & (df['AwayTeam'] == home)]
        
        h2h_df = pd.concat([df_h_vs_a, df_a_vs_h])
        
        if h2h_df.empty: continue
        
        h2h_stats['games'] += len(h2h_df)
        
        # Garante colunas de placar
        if 'FTHG' not in h2h_df.columns or 'FTAG' not in h2h_df.columns: continue
        
        for _, row in h2h_df.iterrows():
            # Normaliza o resultado para a perspectiva do time 'home' da função
            if row['HomeTeam'] == home:
                gh, ga = row['FTHG'], row['FTAG']
            else: # Inverte se o time 'away' da função foi o mandante
                gh, ga = row['FTAG'], row['FTHG']
                
            if gh > ga: h2h_stats['h_wins'] += 1
            elif ga > gh: h2h_stats['a_wins'] += 1
            else: h2h_stats['draws'] += 1
            
            h2h_stats['avg_goals'] += gh + ga
            
    if h2h_stats['games'] > 0:
        h2h_stats['avg_goals'] /= h2h_stats['games']
        
    return h2h_stats

def calcular_jogo_v16(home: str, away: str, stats: Dict, ref: Optional[str], refs_db: Dict, weather_bad: bool = False, all_dfs: Dict = None) -> Dict:
    h_norm = normalize_name(home, list(stats.keys()))
    a_norm = normalize_name(away, list(stats.keys()))
    
    if not h_norm or not a_norm: return {'error': "Times não encontrados."}
    
    s_h = stats[h_norm]
    s_a = stats[a_norm]
    
    r_data = refs_db.get(ref, {'factor': 1.0, 'red_rate': 0.08, 'strictness_score': 4.0}) if ref else {'factor': 1.0, 'red_rate': 0.08, 'strictness_score': 4.0}
    
    # Feature 17: Clima
    weather_factor_goals = 0.9 if weather_bad else 1.0
    weather_factor_cards = 1.2 if weather_bad else 1.0
    
    # Feature 1: Momentum
    mom_h = s_h.get('momentum', 1.0)
    mom_a = s_a.get('momentum', 1.0)
    
    # Feature 21: Fator Elo
    elo_h = s_h.get('elo_rating', 1500)
    elo_a = s_a.get('elo_rating', 1500)
    elo_diff = elo_h - elo_a
    # Ajuste de xG baseado na diferença de Elo (Feature 30)
    elo_factor = math.log10(max(1, abs(elo_diff))) * 0.05 * (1 if elo_diff > 0 else -1)
    
    # Cálculo Base
    corn_h = s_h['corners'] * 1.15 * mom_h
    corn_a = s_a['corners'] * 0.90 * mom_a
    
    card_h = s_h['cards'] * r_data['factor'] * weather_factor_cards
    card_a = s_a['cards'] * r_data['factor'] * weather_factor_cards
 
(Content truncated due to size limit. Use page ranges or line ranges to read remaining content)
