from __future__ import annotations
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]  # project root (where app.py + CSVs live)

def load_all_data():
    """Carrega todos os dados do sistema"""
    stats_db = {}
    cal = pd.DataFrame()
    referees = {}
    
    league_files = {
        'Premier League': str(BASE_DIR / 'Premier_League_25_26.csv'),
        'La Liga': str(BASE_DIR / 'La_Liga_25_26.csv'),
        'Serie A': str(BASE_DIR / 'Serie_A_25_26.csv'),
        'Bundesliga': str(BASE_DIR / 'Bundesliga_25_26.csv'),
        'Ligue 1': str(BASE_DIR / 'Ligue_1_25_26.csv'),
        'Championship': str(BASE_DIR / 'Championship_Inglaterra_25_26.csv'),
        'Bundesliga 2': str(BASE_DIR / 'Bundesliga_2.csv'),
        'Pro League': str(BASE_DIR / 'Pro_League_Belgica_25_26.csv'),
        'Super Lig': str(BASE_DIR / 'Super_Lig_Turquia_25_26.csv'),
        'Premiership': str(BASE_DIR / 'Premiership_Escocia_25_26.csv'),
    }
    
    for league_name, filepath in league_files.items():
        try:
            df = pd.read_csv(filepath, encoding='utf-8')
            teams = set(df['HomeTeam'].dropna().unique()) | set(df['AwayTeam'].dropna().unique())
            
            for team in teams:
                if pd.isna(team):
                    continue
                
                h_games = df[df['HomeTeam'] == team]
                a_games = df[df['AwayTeam'] == team]
                
                # Estatísticas detalhadas
                corners_h = h_games['HC'].mean() if 'HC' in h_games.columns and len(h_games) > 0 else 5.5
                corners_a = a_games['AC'].mean() if 'AC' in a_games.columns and len(a_games) > 0 else 4.5
                corners_h_std = h_games['HC'].std() if 'HC' in h_games.columns and len(h_games) > 1 else 2.0
                corners_a_std = a_games['AC'].std() if 'AC' in a_games.columns and len(a_games) > 1 else 2.0
                
                cards_h = h_games[['HY', 'HR']].sum(axis=1).mean() if 'HY' in h_games.columns and len(h_games) > 0 else 2.5
                cards_a = a_games[['AY', 'AR']].sum(axis=1).mean() if 'AY' in a_games.columns and len(a_games) > 0 else 2.5
                
                fouls_h = h_games['HF'].mean() if 'HF' in h_games.columns and len(h_games) > 0 else 12.0
                fouls_a = a_games['AF'].mean() if 'AF' in a_games.columns and len(a_games) > 0 else 12.0
                
                goals_fh = h_games['FTHG'].mean() if 'FTHG' in h_games.columns and len(h_games) > 0 else 1.5
                goals_fa = a_games['FTAG'].mean() if 'FTAG' in a_games.columns and len(a_games) > 0 else 1.3
                goals_ah = h_games['FTAG'].mean() if 'FTAG' in h_games.columns and len(h_games) > 0 else 1.3
                goals_aa = a_games['FTHG'].mean() if 'FTHG' in a_games.columns and len(a_games) > 0 else 1.5
                
                # Chutes (V14.0)
                shots_h = h_games['HST'].mean() if 'HST' in h_games.columns and len(h_games) > 0 else 4.5
                shots_a = a_games['AST'].mean() if 'AST' in a_games.columns and len(a_games) > 0 else 4.0
                
                stats_db[team] = {
                    'corners': (corners_h + corners_a) / 2,
                    'corners_home': corners_h,
                    'corners_away': corners_a,
                    'corners_std': (corners_h_std + corners_a_std) / 2,
                    'cards': (cards_h + cards_a) / 2,
                    'cards_home': cards_h,
                    'cards_away': cards_a,
                    'fouls': (fouls_h + fouls_a) / 2,
                    'fouls_home': fouls_h,
                    'fouls_away': fouls_a,
                    'goals_f': (goals_fh + goals_fa) / 2,
                    'goals_f_home': goals_fh,
                    'goals_f_away': goals_fa,
                    'goals_a': (goals_ah + goals_aa) / 2,
                    'goals_a_home': goals_ah,
                    'goals_a_away': goals_aa,
                    'shots_on_target': (shots_h + shots_a) / 2,
                    'shots_home': shots_h,
                    'shots_away': shots_a,
                    'league': league_name,
                    'games': len(h_games) + len(a_games)
                }
        except Exception as e:
            st.sidebar.warning(f"⚠️ {league_name}: {str(e)}")
    
    try:
        cal = pd.read_csv(str(BASE_DIR / 'calendario_ligas.csv'), encoding='utf-8')
        if 'Data' in cal.columns:
            cal['DtObj'] = pd.to_datetime(cal['Data'], format='%d/%m/%Y', errors='coerce')
    except:
        pass
    
    try:
        refs_df = pd.read_csv(str(BASE_DIR / 'arbitros_5_ligas_2025_2026.csv'), encoding='utf-8')
        for _, row in refs_df.iterrows():
            referees[row['Arbitro']] = {
                'factor': row['Media_Cartoes_Por_Jogo'] / 4.0,
                'games': row['Jogos_Apitados'],
                'avg_cards': row['Media_Cartoes_Por_Jogo'],
                'red_cards': row.get('Cartoes_Vermelhos', 0),
                'red_rate': row.get('Cartoes_Vermelhos', 0) / row['Jogos_Apitados'] if row['Jogos_Apitados'] > 0 else 0.08
            }
    except:
        pass
    
    return stats_db, cal, referees

def load_referees_fallback(refs: dict) -> dict:
    """If the detailed referees file is missing, try arbitros.csv."""
    if refs:
        return refs
    try:
        df = pd.read_csv(str(BASE_DIR / "arbitros.csv"), encoding="utf-8")
        # try best-effort mapping
        for _, row in df.iterrows():
            name = row.get("Arbitro") or row.get("Referee") or row.get("Nome") or row.get("Name")
            if not name:
                continue
            avg = row.get("Media_Cartoes_Por_Jogo") or row.get("AvgCards") or row.get("Media") or row.get("CardsAvg")
            try:
                avg = float(avg)
            except:
                avg = 4.0
            games = row.get("Jogos_Apitados") or row.get("Games") or row.get("Jogos") or 1
            try:
                games = int(games)
            except:
                games = 1
            refs[name] = {
                "factor": avg / 4.0,
                "games": games,
                "avg_cards": avg,
                "red_cards": row.get("Cartoes_Vermelhos", 0),
                "red_rate": 0.08,
            }
    except:
        pass
    return refs
