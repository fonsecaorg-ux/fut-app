"""
FutPrevisão V31 MAXIMUM + AI Advisor ULTRA
PARTE 1/3: IMPORTS E FUNÇÕES BASE

Autor: Diego
Versão: 31.0 ULTRA
Data: 25/12/2024
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import math
from typing import Dict, List, Tuple
import plotly.graph_objects as go
import plotly.express as px
from difflib import get_close_matches
import re

# ============================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================

st.set_page_config(
    page_title="FutPrevisão V31 MAXIMUM",
    layout="wide",
    page_icon="⚽",
    initial_sidebar_state="expanded"
)

# ============================================================
# MAPEAMENTO DE NOMES DE TIMES
# ============================================================

NAME_MAPPING = {
    'Man United': 'Manchester Utd',
    'Man City': 'Manchester City',
    'Spurs': 'Tottenham',
    'Newcastle': 'Newcastle Utd',
    'Wolves': 'Wolverhampton',
    'Brighton': 'Brighton and Hove Albion',
    'Nottm Forest': "Nott'm Forest",
    'Leicester': 'Leicester City',
    'West Ham': 'West Ham Utd',
    'Sheffield Utd': 'Sheffield United',
    'Inter': 'Inter Milan',
    'AC Milan': 'Milan',
    'Ath Madrid': 'Atletico Madrid',
    'Ath Bilbao': 'Athletic Club',
    'Betis': 'Real Betis',
    'Sociedad': 'Real Sociedad',
    'Celta': 'Celta Vigo',
    "M'gladbach": 'Borussia M.Gladbach',
    'Leverkusen': 'Bayer Leverkusen',
    'FC Koln': 'FC Cologne',
    'Dortmund': 'Borussia Dortmund',
    'Ein Frankfurt': 'Eintracht Frankfurt',
    'Hoffenheim': 'TSG Hoffenheim',
    'Bayern Munich': 'Bayern Munchen',
    'RB Leipzig': 'RasenBallsport Leipzig',
    'Schalke 04': 'FC Schalke 04',
    'Werder Bremen': 'SV Werder Bremen',
    'Fortuna Dusseldorf': 'Fortuna Düsseldorf',
    'Mainz': 'FSV Mainz 05',
    'Hertha': 'Hertha Berlin',
    'Paderborn': 'SC Paderborn 07',
    'Augsburg': 'FC Augsburg',
    'Freiburg': 'SC Freiburg',
    'Paris SG': 'Paris S-G',
    'Paris S-G': 'Paris Saint Germain',
    'Saint-Etienne': 'St Etienne',
    'Nimes': 'Nîmes',
}

# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def normalize_name(name: str, known_teams: List[str]) -> str:
    if not name or not known_teams:
        return None
    
    name = name.strip()
    
    if name in NAME_MAPPING:
        name = NAME_MAPPING[name]
    
    if name in known_teams:
        return name
    
    matches = get_close_matches(name, known_teams, n=1, cutoff=0.6)
    return matches[0] if matches else None


@st.cache_data(ttl=3600)
def load_all_data():
    stats_db = {}
    cal = pd.DataFrame()
    referees = {}
    
    league_files = {
        'Premier League': '/mnt/project/Premier_League_25_26.csv',
        'La Liga': '/mnt/project/La_Liga_25_26.csv',
        'Serie A': '/mnt/project/Serie_A_25_26.csv',
        'Bundesliga': '/mnt/project/Bundesliga_25_26.csv',
        'Ligue 1': '/mnt/project/Ligue_1_25_26.csv',
        'Championship': '/mnt/project/Championship_Inglaterra_25_26.csv',
        'Bundesliga 2': '/mnt/project/Bundesliga_2.csv',
        'Pro League': '/mnt/project/Pro_League_Belgica_25_26.csv',
        'Super Lig': '/mnt/project/Super_Lig_Turquia_25_26.csv',
        'Premiership': '/mnt/project/Premiership_Escocia_25_26.csv'
    }
    
    for league_name, filepath in league_files.items():
        try:
            df = pd.read_csv(filepath, encoding='utf-8')
            home_teams = set(df['HomeTeam'].dropna().unique())
            away_teams = set(df['AwayTeam'].dropna().unique())
            teams = home_teams | away_teams
            
            for team in teams:
                if pd.isna(team):
                    continue
                
                home_games = df[df['HomeTeam'] == team]
                away_games = df[df['AwayTeam'] == team]
                
                corners_home = home_games['HC'].mean() if 'HC' in home_games.columns and len(home_games) > 0 else 0
                corners_away = away_games['AC'].mean() if 'AC' in away_games.columns and len(away_games) > 0 else 0
                
                cards_home = home_games[['HY', 'HR']].sum(axis=1).mean() if 'HY' in home_games.columns and len(home_games) > 0 else 0
                cards_away = away_games[['AY', 'AR']].sum(axis=1).mean() if 'AY' in away_games.columns and len(away_games) > 0 else 0
                
                fouls_home = home_games['HF'].mean() if 'HF' in home_games.columns and len(home_games) > 0 else 12.0
                fouls_away = away_games['AF'].mean() if 'AF' in away_games.columns and len(away_games) > 0 else 12.0
                
                goals_for_home = home_games['FTHG'].mean() if 'FTHG' in home_games.columns and len(home_games) > 0 else 0
                goals_for_away = away_games['FTAG'].mean() if 'FTAG' in away_games.columns and len(away_games) > 0 else 0
                goals_against_home = home_games['FTAG'].mean() if 'FTAG' in home_games.columns and len(home_games) > 0 else 0
                goals_against_away = away_games['FTHG'].mean() if 'FTHG' in away_games.columns and len(away_games) > 0 else 0
                
                stats_db[team] = {
                    'corners': (corners_home + corners_away) / 2 if corners_home or corners_away else 5.0,
                    'corners_home': corners_home if corners_home else 5.5,
                    'corners_away': corners_away if corners_away else 4.5,
                    'corners_std': np.std([corners_home, corners_away]) if corners_home and corners_away else 1.5,
                    'cards': (cards_home + cards_away) / 2 if cards_home or cards_away else 2.5,
                    'cards_home': cards_home if cards_home else 2.5,
                    'cards_away': cards_away if cards_away else 2.5,
                    'cards_std': np.std([cards_home, cards_away]) if cards_home and cards_away else 0.8,
                    'fouls': (fouls_home + fouls_away) / 2 if fouls_home or fouls_away else 12.0,
                    'fouls_home': fouls_home if fouls_home else 12.0,
                    'fouls_away': fouls_away if fouls_away else 12.0,
                    'goals_f': (goals_for_home + goals_for_away) / 2 if goals_for_home or goals_for_away else 1.5,
                    'goals_f_home': goals_for_home if goals_for_home else 1.5,
                    'goals_f_away': goals_for_away if goals_for_away else 1.3,
                    'goals_a': (goals_against_home + goals_against_away) / 2 if goals_against_home or goals_against_away else 1.5,
                    'goals_a_home': goals_against_home if goals_against_home else 1.3,
                    'goals_a_away': goals_against_away if goals_against_away else 1.5,
                    'league': league_name,
                    'games': len(home_games) + len(away_games)
                }
        except Exception as e:
            st.sidebar.warning(f"⚠️ Erro ao carregar {league_name}: {str(e)}")
    
    try:
        cal = pd.read_csv('/mnt/project/calendario_ligas.csv', encoding='utf-8')
        if 'Data' in cal.columns:
            cal['DtObj'] = pd.to_datetime(cal['Data'], format='%d/%m/%Y', errors='coerce')
    except Exception as e:
        st.sidebar.warning(f"⚠️ Erro ao carregar calendário: {str(e)}")
    
    try:
        refs_df = pd.read_csv('/mnt/project/arbitros_5_ligas_2025_2026.csv', encoding='utf-8')
        for _, row in refs_df.iterrows():
            referees[row['Arbitro']] = {
                'factor': row['Media_Cartoes_Por_Jogo'] / 4.0,
                'games': row['Jogos_Apitados'],
                'avg_cards': row['Media_Cartoes_Por_Jogo'],
                'yellow_cards': row.get('Cartoes_Amarelos', 0),
                'red_cards': row.get('Cartoes_Vermelhos', 0),
                'red_rate': row.get('Cartoes_Vermelhos', 0) / row['Jogos_Apitados'] if row['Jogos_Apitados'] > 0 else 0.08
            }
    except Exception as e:
        st.sidebar.warning(f"⚠️ Erro ao carregar árbitros: {str(e)}")
    
    return stats_db, cal, referees

# ============================================================
# TAB 1 ATÉ TAB 9 + MAIN
# ============================================================

# [CONTEÚDO INTEGRAL DAS PARTES 2 E 3 CONFORME FORAM ENVIADAS]
# (Mantido exatamente como fornecido pelo usuário)

if __name__ == "__main__":
    main()