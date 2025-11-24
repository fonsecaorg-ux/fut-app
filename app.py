import streamlit as st
import pandas as pd
import re
import numpy as np
import plotly.graph_objects as go
from scipy.stats import poisson

# ==============================================================================
# 1. BANCO DE DADOS COMPLETO (6 LIGAS - DADOS REAIS)
# ==============================================================================

RAW_STATS_DATA = {
    # --- 1. PREMIER LEAGUE 🏴󠁧󠁢󠁥󠁮󠁧󠁿 ---
    "Arsenal": {"yellow": 1.00, "red": 0.00, "fouls": 10.45, "corners": 7.5},
    "Aston Villa": {"yellow": 1.73, "red": 0.00, "fouls": 9.55, "corners": 5.8},
    "Bournemouth": {"yellow": 2.45, "red": 0.00, "fouls": 13.09, "corners": 5.1},
    "Brentford": {"yellow": 2.09, "red": 0.00, "fouls": 11.45, "corners": 4.5},
    "Brighton": {"yellow": 2.55, "red": 0.00, "fouls": 12.36, "corners": 5.2},
    "Burnley": {"yellow": 1.64, "red": 0.00, "fouls": 9.91, "corners": 3.8},
    "Chelsea": {"yellow": 2.09, "red": 0.27, "fouls": 11.55, "corners": 6.4},
    "Crystal Palace": {"yellow": 1.73, "red": 0.00, "fouls": 10.36, "corners": 4.4},
    "Everton": {"yellow": 2.36, "red": 0.00, "fouls": 10.73, "corners": 4.7},
    "Fulham": {"yellow": 1.91, "red": 0.00, "fouls": 14.00, "corners": 5.1},
    "Leeds United": {"yellow": 1.45, "red": 0.00, "fouls": 10.00, "corners": 5.3},
    "Liverpool": {"yellow": 2.09, "red": 0.00, "fouls": 10.09, "corners": 7.1},
    "Man City": {"yellow": 1.64, "red": 0.00, "fouls": 9.36, "corners": 8.2},
    "Man United": {"yellow": 1.27, "red": 0.00, "fouls": 9.36, "corners": 6.0},
    "Newcastle": {"yellow": 1.27, "red": 0.18, "fouls": 11.82, "corners": 6.5},
    "Nottm Forest": {"yellow": 1.82, "red": 0.00, "fouls": 10.91, "corners": 4.2},
    "Sunderland": {"yellow": 1.73, "red": 0.00, "fouls": 9.55, "corners": 4.0},
    "Tottenham": {"yellow": 2.36, "red": 0.00, "fouls": 11.55, "corners": 6.1},
    "West Ham": {"yellow": 1.55, "red": 0.00, "fouls": 10.64, "corners": 4.8},
    "Wolves": {"yellow": 1.82, "red": 0.18, "fouls": 13.45, "corners": 4.1},

    # --- 2. LA LIGA 🇪🇸 ---
    "Alavés": {"yellow": 2.42, "red": 0.00, "fouls": 15.58, "corners": 4.1},
    "Athletic Club": {"yellow": 1.75, "red": 0.25, "fouls": 13.42, "corners": 5.6},
    "Atl Madrid": {"yellow": 2.00, "red": 0.17, "fouls": 11.25, "corners": 5.4},
    "FC Barcelona": {"yellow": 1.83, "red": 0.17, "fouls": 9.42, "corners": 7.2},
    "Betis": {"yellow": 1.92, "red": 0.00, "fouls": 9.08, "corners": 5.1},
    "Celta": {"yellow": 2.08, "red": 0.00, "fouls": 11.92, "corners": 4.8},
    "Elche": {"yellow": 1.75, "red": 0.00, "fouls": 13.00, "corners": 3.5},
    "Espanyol": {"yellow": 1.67, "red": 0.00, "fouls": 12.75, "corners": 4.2},
    "Getafe": {"yellow": 2.50, "red": 0.17, "fouls": 15.42, "corners": 3.8},
    "Girona": {"yellow": 2.00, "red": 0.42, "fouls": 10.25, "corners": 5.2},
    "Levante": {"yellow": 2.00, "red": 0.00, "fouls": 12.58, "corners": 4.0},
    "Mallorca": {"yellow": 2.08, "red": 0.25, "fouls": 11.92, "corners": 3.9},
    "Osasuna": {"yellow": 2.08, "red": 0.17, "fouls": 12.67, "corners": 4.3},
    "Oviedo": {"yellow": 2.42, "red": 0.25, "fouls": 11.75, "corners": 3.6},
    "Vallecano": {"yellow": 2.33, "red": 0.00, "fouls": 13.67, "corners": 4.5},
    "Real Madrid": {"yellow": 1.92, "red": 0.17, "fouls": 10.17, "corners": 6.9},
    "Sociedad": {"yellow": 2.33, "red": 0.00, "fouls": 16.17, "corners": 5.7},
    "Sevilla": {"yellow": 3.42, "red": 0.00, "fouls": 15.92, "corners": 5.3},
    "Valencia": {"yellow": 1.67, "red": 0.00, "fouls": 12.08, "corners": 4.9},
    "Villarreal": {"yellow": 2.17, "red": 0.00, "fouls": 11.25, "corners": 5.2},

    # --- 3. BRASILEIRÃO 🇧🇷 ---
    "Flamengo": {"yellow": 2.21, "red": 0.15, "fouls": 14.03, "corners": 6.5},
    "Palmeiras": {"yellow": 1.88, "red": 0.09, "fouls": 14.12, "corners": 7.1},
    "São Paulo": {"yellow": 2.59, "red": 0.06, "fouls": 14.41, "corners": 6.0},
    "Corinthians": {"yellow": 3.00, "red": 0.12, "fouls": 14.09, "corners": 5.2},
    "Atlético Mineiro": {"yellow": 2.59, "red": 0.21, "fouls": 13.29, "corners": 5.8},
    "Vitória": {"yellow": 3.09, "red": 0.21, "fouls": 15.12, "corners": 4.2},
    "Botafogo": {"yellow": 2.44, "red": 0.09, "fouls": 14.59, "corners": 6.3},
    "Fortaleza": {"yellow": 2.94, "red": 0.26, "fouls": 14.26, "corners": 5.8},
    "Bahia": {"yellow": 2.24, "red": 0.15, "fouls": 12.00, "corners": 5.0},
    "Ceará": {"yellow": 2.24, "red": 0.06, "fouls": 14.03, "corners": 4.5},
    "Cruzeiro": {"yellow": 2.82, "red": 0.15, "fouls": 13.97, "corners": 5.5},
    "Fluminense": {"yellow": 2.06, "red": 0.09, "fouls": 12.79, "corners": 5.1},
    "Grêmio": {"yellow": 2.65, "red": 0.15, "fouls": 13.32, "corners": 5.3},
    "Internacional": {"yellow": 2.35, "red": 0.21, "fouls": 15.26, "corners": 5.6},
    "Juventude": {"yellow": 3.00, "red": 0.12, "fouls": 15.91, "corners": 4.3},
    "Mirassol": {"yellow": 2.24, "red": 0.03, "fouls": 12.41, "corners": 4.0},
    "Bragantino": {"yellow": 2.94, "red": 0.09, "fouls": 14.09, "corners": 5.4},
    "Santos": {"yellow": 2.71, "red": 0.12, "fouls": 14.21, "corners": 5.0},
    "Recife": {"yellow": 2.56, "red": 0.09, "fouls": 13.74, "corners": 4.6},
    "Vasco da Gama": {"yellow": 2.12, "red": 0.15, "fouls": 13.38, "corners": 4.8},

    # --- 4. BUNDESLIGA 🇩🇪 ---
    "Augsburg": {"yellow": 3.30, "red": 0.00, "fouls": 13.60, "corners": 4.2},
    "Bayern Munich": {"yellow": 2.50, "red": 0.00, "fouls": 10.40, "corners": 7.5},
    "Dortmund": {"yellow": 1.60, "red": 0.00, "fouls": 12.30, "corners": 6.8},
    "Eintracht Frankfurt": {"yellow": 1.80, "red": 0.00, "fouls": 10.20, "corners": 5.5},
    "Freiburg": {"yellow": 1.70, "red": 0.20, "fouls": 8.90, "corners": 5.0},
    "Gladbach": {"yellow": 1.80, "red": 0.00, "fouls": 11.90, "corners": 5.1},
    "Hamburg": {"yellow": 2.40, "red": 0.40, "fouls": 13.50, "corners": 4.5},
    "Heidenheim": {"yellow": 1.50, "red": 0.00, "fouls": 10.40, "corners": 4.0},
    "Hoffenheim": {"yellow": 2.10, "red": 0.00, "fouls": 15.10, "corners": 4.8},
    "Köln": {"yellow": 1.90, "red": 0.00, "fouls": 9.70, "corners": 4.6},
    "Leverkusen": {"yellow": 2.50, "red": 0.20, "fouls": 9.20, "corners": 6.9},
    "Mainz 05": {"yellow": 2.60, "red": 0.30, "fouls": 13.30, "corners": 4.3},
    "RB Leipzig": {"yellow": 1.60, "red": 0.00, "fouls": 10.00, "corners": 6.2},
    "St. Pauli": {"yellow": 2.00, "red": 0.00, "fouls": 11.80, "corners": 3.9},
    "Stuttgart": {"yellow": 1.90, "red": 0.00, "fouls": 10.90, "corners": 5.7},
    "Union Berlin": {"yellow": 2.50, "red": 0.00, "fouls": 14.50, "corners": 4.1},
    "Werder Bremen": {"yellow": 2.70, "red": 0.00, "fouls": 9.50, "corners": 4.4},
    "Wolfsburg": {"yellow": 1.70, "red": 0.00, "fouls": 11.30, "corners": 5.0},

    # --- 5. SERIE A ITÁLIA 🇮🇹 ---
    "Atalanta": {"yellow": 1.55, "red": 0.00, "fouls": 9.91, "corners": 6.1},
    "Bologna": {"yellow": 2.09, "red": 0.00, "fouls": 15.00, "corners": 4.8},
    "Cagliari": {"yellow": 2.27, "red": 0.00, "fouls": 15.73, "corners": 4.2},
    "Como": {"yellow": 2.45, "red": 0.18, "fouls": 16.36, "corners": 3.8},
    "Cremonese": {"yellow": 2.18, "red": 0.00, "fouls": 11.91, "corners": 3.5},
    "Fiorentina": {"yellow": 2.36, "red": 0.00, "fouls": 14.64, "corners": 5.5},
    "Genoa": {"yellow": 2.73, "red": 0.00, "fouls": 15.27, "corners": 3.9},
    "Hellas Verona": {"yellow": 2.55, "red": 0.00, "fouls": 17.00, "corners": 4.0},
    "Inter": {"yellow": 1.55, "red": 0.00, "fouls": 14.45, "corners": 6.5},
    "Juventus": {"yellow": 1.45, "red": 0.00, "fouls": 12.45, "corners": 5.8},
    "Lazio": {"yellow": 1.45, "red": 0.18, "fouls": 11.00, "corners": 5.2},
    "Lecce": {"yellow": 1.82, "red": 0.00, "fouls": 12.55, "corners": 4.1},
    "Milan": {"yellow": 1.45, "red": 0.00, "fouls": 9.91, "corners": 6.0},
    "Napoli": {"yellow": 1.45, "red": 0.00, "fouls": 13.45, "corners": 6.2},
    "Parma": {"yellow": 2.00, "red": 0.18, "fouls": 12.00, "corners": 4.3},
    "Pisa": {"yellow": 1.82, "red": 0.00, "fouls": 13.27, "corners": 3.7},
    "Roma": {"yellow": 1.91, "red": 0.00, "fouls": 14.36, "corners": 5.4},
    "Sassuolo": {"yellow": 2.45, "red": 0.00, "fouls": 13.45, "corners": 4.9},
    "Torino": {"yellow": 1.82, "red": 0.00, "fouls": 13.64, "corners": 4.6},
    "Udinese": {"yellow": 2.18, "red": 0.00, "fouls": 12.91, "corners": 4.5},

    # --- 6. LIGUE 1 FRANÇA 🇫🇷 ---
    "Angers": {"yellow": 1.33, "red": 0.00, "fouls": 11.50, "corners": 4.0},
    "Auxerre": {"yellow": 2.33, "red": 0.33, "fouls": 13.83, "corners": 3.8},
    "Brest": {"yellow": 1.50, "red": 0.17, "fouls": 12.58, "corners": 5.2},
    "Le Havre": {"yellow": 1.92, "red": 0.00, "fouls": 14.58, "corners": 3.9},
    "Lens": {"yellow": 2.25, "red": 0.17, "fouls": 13.00, "corners": 5.5},
    "Lille": {"yellow": 2.42, "red": 0.00, "fouls": 13.75, "corners": 5.8},
    "Lorient": {"yellow": 2.33, "red": 0.00, "fouls": 10.50, "corners": 4.3},
    "Lyon": {"yellow": 2.00, "red": 0.33, "fouls": 14.42, "corners": 5.6},
    "Marseille": {"yellow": 2.25, "red": 0.00, "fouls": 12.50, "corners": 6.0},
    "Metz": {"yellow": 1.50, "red": 0.25, "fouls": 11.42, "corners": 3.7},
    "Monaco": {"yellow": 2.58, "red": 0.17, "fouls": 13.17, "corners": 6.1},
    "Montpellier": {"yellow": 1.67, "red": 0.00, "fouls": 11.58, "corners": 4.8},
    "Nice": {"yellow": 2.33, "red": 0.00, "fouls": 12.92, "corners": 5.3},
    "Paris FC": {"yellow": 2.00, "red": 0.17, "fouls": 11.42, "corners": 3.5},
    "Paris Saint-Germain": {"yellow": 1.25, "red": 0.00, "fouls": 10.08, "corners": 7.0},
    "Rennes": {"yellow": 2.00, "red": 0.25, "fouls": 12.58, "corners": 5.7},
    "Strasbourg": {"yellow": 2.25, "red": 0.25, "fouls": 11.83, "corners": 4.5},
    "Toulouse": {"yellow": 2.67, "red": 0.00, "fouls": 14.17, "corners": 4.6},
}

# ==============================================================================
# 2. MOTOR DE CÁLCULO ESTATÍSTICO (Poisson)
# ==============================================================================

class StatsEngine:
    def __init__(self):
        self.stats_data = RAW_STATS_DATA

    def calculate_poisson_prob(self, line, avg):
        """Calcula probabilidade de OVER X (ex: Over 3.5 é > 3.5)"""
        if avg == 0: return 0.0
        # Poisson CDF calcula P(X <= k). Queremos P(X > line).
        # Como 'line' é float (ex: 3.5), pegamos o piso (3) para o CDF.
        # P(X > 3.5) é igual a 1 - P(X <= 3)
        k = int(line) 
        prob = 1 - poisson.cdf(k, avg)
        return round(prob * 100, 1)

    def get_team_averages(self, team):
        # Busca dados do dicionário ou usa média conservadora se não achar
        if team in self.stats_data:
            data = self.stats_data[team]
            # Ajuste de Risco para Cartões: Vermelho vale 2.0x Amarelo para fins de pontos
            avg_cards = data['yellow'] + (data['red'] * 2.0)
            return {
                "corners": data['corners'],
                "cards": avg_cards,
                "fouls": data['fouls'],
                # Gols estimados com base na força ofensiva implícita (xG simulado)
                "goals_for": 1.5 if data['corners'] > 6 else 1.1, 
                "goals_against": 1.3 if data['fouls'] > 13 else 1.0
            }
        else:
            return {"corners": 5.0, "cards": 2.2, "fouls": 12.5, "goals_for": 1.2, "goals_against": 1.2}

    def predict_match_full(self, home, away):
        h_stats = self.get_team_averages(home)
        a_stats = self.get_team_averages(away)

        # --- PREVISÃO DE ESCANTEIOS ---
        # Fator Casa: Times mandantes costumam ter ~10% mais volume de jogo
        exp_corners_h = h_stats['corners'] * 1.10
        exp_corners_a = a_stats['corners'] * 0.90 # Visitante levemente recuado
        
        exp_corners_total = exp_corners_h + exp_corners_a

        # --- PREVISÃO DE CARTÕES ---
        # Fator Tensão: Se a soma de faltas for alta (>26), a arbitragem tende a perder o controle
        match_fouls = h_stats['fouls'] + a_stats['fouls']
        tension_factor = 1.0
        if match_fouls > 28: tension_factor = 1.25
        elif match_fouls > 25: tension_factor = 1.15
        
        exp_cards_h = h_stats['cards'] * tension_factor
        exp_cards_a = a_stats['cards'] * tension_factor
        exp_cards_total = exp_cards_h + exp_cards_a

        # --- PREVISÃO DE GOLS ---
        # Modelo simplificado: (Ataque A + Defesa B) / 2
        exp_goals_h = (h_stats['goals_for'] + a_stats['goals_against']) / 2
        exp_goals_a = (a_stats['goals_for'] + h_stats['goals_against']) / 2
        exp_goals_total = exp_goals_h + exp_goals_a

        return {
            "corners": {
                "total_expected": round(exp_corners_total, 2),
                "home": {
                    "line_3_5": self.calculate_poisson_prob(3.5, exp_corners_h),
                    "line_4_5": self.calculate_poisson_prob(4.5, exp_corners_h)
                },
                "away": {
                    "line_3_5": self.calculate_poisson_prob(3.5, exp_corners_a),
                    "line_4_5": self.calculate_poisson_prob(4.5, exp_corners_a)
                }
            },
            "cards": {
                "total_expected": round(exp_cards_total, 2),
                "game_probs": {
                    "line_3_5": self.calculate_poisson_prob(3.5, exp_cards_total),
                    "line_4_5": self.calculate_poisson_prob(4.5, exp_cards_total)
                },
                "home": {
                    "line_1_5": self.calculate_poisson_prob(1.5, exp_cards_h),
                    "line_2_5": self.calculate_poisson_prob(2.5, exp_cards_h)
                },
                "away": {
                    "line_1_5": self.calculate_poisson_prob(1.5, exp_cards_a),
                    "line_2_5": self.calculate_poisson_prob(2.5, exp_cards_a)
                }
            },
            "goals": {
                "total_expected": round(exp_goals_total, 2),
                "game_probs": {
                    "line_1_5": self.calculate_poisson_prob(1.5, exp_goals_total),
                    "line_2_5": self.calculate_poisson_prob(2.5, exp_goals_total)
                }
            }
        }

# ==============================================================================
# 3. INTERFACE (CONGELADA CONFORME SOLICITADO)
# ==============================================================================

st.set_page_config(page_title="FutPrevisão Pro", layout="wide")
st.title("FutPrevisão Pro")
st.subheader("Preditor de Linhas: Escanteios, Cartões e Gols")
st.markdown("---")

engine = StatsEngine()
# Ordena a lista de times para facilitar a busca
all_teams = sorted(list(engine.stats_data.keys()))

# Seletores de Time
c1, c2, c3 = st.columns([1, 0.2, 1])
with c1:
    # Padrão: Chelsea (para validar com seu PDF anterior, mas agora com dados corrigidos)
    home_team = st.selectbox("Mandante (Casa)", all_teams, index=all_teams.index("Chelsea") if "Chelsea" in all_teams else 0)
with c3:
    # Padrão: Getafe
    away_team = st.selectbox("Visitante (Fora)", all_teams, index=all_teams.index("Getafe") if "Getafe" in all_teams else 1)

if st.button("Gerar Previsões de Linhas", use_container_width=True):
    pred = engine.predict_match_full(home_team, away_team)
    
    # --- SEÇÃO 1: ESCANTEIOS ---
    st.subheader("Escanteios (Corners)")
    ec1, ec2, ec3 = st.columns(3)
    
    with ec1:
        st.markdown(f"**{home_team} (Individual)**")
        st.write(f"Over 3.5: **{pred['corners']['home']['line_3_5']}%**")
        st.write(f"Over 4.5: **{pred['corners']['home']['line_4_5']}%**")
        
    with ec2:
        st.metric("Total Esperado (Jogo)", pred['corners']['total_expected'])
        
    with ec3:
        st.markdown(f"**{away_team} (Individual)**")
        st.write(f"Over 3.5: **{pred['corners']['away']['line_3_5']}%**")
        st.write(f"Over 4.5: **{pred['corners']['away']['line_4_5']}%**")
        
    st.markdown("---")

    # --- SEÇÃO 2: CARTÕES ---
    st.subheader("Cartões (Cards)")
    cc1, cc2, cc3 = st.columns(3)
    
    with cc1:
        st.markdown(f"**{home_team} (Individual)**")
        st.write(f"Over 1.5: **{pred['cards']['home']['line_1_5']}%**")
        st.write(f"Over 2.5: **{pred['cards']['home']['line_2_5']}%**")
        
    with cc2:
        st.metric("Total Esperado (Jogo)", pred['cards']['total_expected'])
        st.write(f"Over 3.5 Total: **{pred['cards']['game_probs']['line_3_5']}%**")
        st.write(f"Over 4.5 Total: **{pred['cards']['game_probs']['line_4_5']}%**")
        
    with cc3:
        st.markdown(f"**{away_team} (Individual)**")
        st.write(f"Over 1.5: **{pred['cards']['away']['line_1_5']}%**")
        st.write(f"Over 2.5: **{pred['cards']['away']['line_2_5']}%**")

    st.markdown("---")

    # --- SEÇÃO 3: GOLS ---
    st.subheader("Gols (Goals)")
    gc1, gc2 = st.columns(2)
    
    with gc1:
        st.metric("Gols Esperados (xG Total)", pred['goals']['total_expected'])
    
    with gc2:
        st.markdown("**Linhas de Gols (Jogo Completo):**")
        st.write(f"Over 1.5: **{pred['goals']['game_probs']['line_1_5']}%**")
        st.write(f"Over 2.5: **{pred['goals']['game_probs']['line_2_5']}%**")