import streamlit as st
import pandas as pd
import re
import numpy as np
import plotly.graph_objects as go
from scipy.stats import poisson
import hmac

# ==============================================================================
# 0. SISTEMA DE LOGIN
# ==============================================================================
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    def password_entered():
        if "passwords" in st.secrets:
            user = st.session_state["username"]
            password = st.session_state["password"]
            if user in st.secrets["passwords"] and password == st.secrets["passwords"][user]:
                st.session_state["password_correct"] = True
                del st.session_state["password"]
                del st.session_state["username"]
            else:
                st.session_state["password_correct"] = False
        else:
            st.error("Erro: Senhas não configuradas.")

    if st.session_state["password_correct"]:
        return True

    st.set_page_config(page_title="Login FutPrevisão", layout="centered", page_icon="🔒")
    st.title("🔒 Acesso Restrito")
    st.info("Faça login para acessar as previsões.")
    st.text_input("Usuário", key="username")
    st.text_input("Senha", type="password", key="password")
    st.button("Entrar", on_click=password_entered)
    
    if "password_correct" in st.session_state and not st.session_state["password_correct"] and "username" in st.session_state:
        st.error("😕 Usuário ou senha incorretos.")
    return False

if not check_password():
    st.stop()

# ==============================================================================
# 1. CARREGAMENTO DE DADOS
# ==============================================================================
# DADOS DE BACKUP (Caso CSV falhe)
BACKUP_DATA = {
    "Arsenal": {"yellow": 1.00, "red": 0.00, "fouls": 10.45, "corners": 7.5, "g_for": 2.3, "g_against": 0.8},
    "Aston Villa": {"yellow": 1.73, "red": 0.00, "fouls": 9.55, "corners": 5.8, "g_for": 1.9, "g_against": 1.4},
    "Bournemouth": {"yellow": 2.45, "red": 0.00, "fouls": 13.09, "corners": 5.1, "g_for": 1.4, "g_against": 1.6},
    "Brentford": {"yellow": 2.09, "red": 0.00, "fouls": 11.45, "corners": 4.5, "g_for": 1.5, "g_against": 1.7},
    "Brighton": {"yellow": 2.55, "red": 0.00, "fouls": 12.36, "corners": 5.2, "g_for": 1.8, "g_against": 1.6},
    "Burnley": {"yellow": 1.64, "red": 0.00, "fouls": 9.91, "corners": 3.8, "g_for": 0.9, "g_against": 1.9},
    "Chelsea": {"yellow": 2.09, "red": 0.27, "fouls": 11.55, "corners": 6.4, "g_for": 2.1, "g_against": 1.3},
    "Crystal Palace": {"yellow": 1.73, "red": 0.00, "fouls": 10.36, "corners": 4.4, "g_for": 1.2, "g_against": 1.6},
    "Everton": {"yellow": 2.36, "red": 0.00, "fouls": 10.73, "corners": 4.7, "g_for": 1.0, "g_against": 1.5},
    "Fulham": {"yellow": 1.91, "red": 0.00, "fouls": 14.00, "corners": 5.1, "g_for": 1.4, "g_against": 1.5},
    "Leeds United": {"yellow": 1.45, "red": 0.00, "fouls": 10.00, "corners": 5.3, "g_for": 1.5, "g_against": 1.6},
    "Liverpool": {"yellow": 2.09, "red": 0.00, "fouls": 10.09, "corners": 7.1, "g_for": 2.4, "g_against": 1.0},
    "Man City": {"yellow": 1.64, "red": 0.00, "fouls": 9.36, "corners": 8.2, "g_for": 2.6, "g_against": 0.9},
    "Man United": {"yellow": 1.27, "red": 0.00, "fouls": 9.36, "corners": 6.0, "g_for": 1.5, "g_against": 1.3},
    "Newcastle": {"yellow": 1.27, "red": 0.18, "fouls": 11.82, "corners": 6.5, "g_for": 2.0, "g_against": 1.2},
    "Nottm Forest": {"yellow": 1.82, "red": 0.00, "fouls": 10.91, "corners": 4.2, "g_for": 1.2, "g_against": 1.6},
    "Sunderland": {"yellow": 1.73, "red": 0.00, "fouls": 9.55, "corners": 4.0, "g_for": 1.2, "g_against": 1.4},
    "Tottenham": {"yellow": 2.36, "red": 0.00, "fouls": 11.55, "corners": 6.1, "g_for": 2.0, "g_against": 1.4},
    "West Ham": {"yellow": 1.55, "red": 0.00, "fouls": 10.64, "corners": 4.8, "g_for": 1.4, "g_against": 1.6},
    "Wolves": {"yellow": 1.82, "red": 0.18, "fouls": 13.45, "corners": 4.1, "g_for": 1.1, "g_against": 1.6},
    "Alavés": {"yellow": 2.42, "red": 0.00, "fouls": 15.58, "corners": 4.1, "g_for": 1.1, "g_against": 1.4},
    "Athletic Club": {"yellow": 1.75, "red": 0.25, "fouls": 13.42, "corners": 5.6, "g_for": 1.6, "g_against": 1.0},
    "Atl Madrid": {"yellow": 2.00, "red": 0.17, "fouls": 11.25, "corners": 5.4, "g_for": 1.8, "g_against": 0.9},
    "FC Barcelona": {"yellow": 1.83, "red": 0.17, "fouls": 9.42, "corners": 7.2, "g_for": 2.6, "g_against": 1.0},
    "Betis": {"yellow": 1.92, "red": 0.00, "fouls": 9.08, "corners": 5.1, "g_for": 1.4, "g_against": 1.2},
    "Celta": {"yellow": 2.08, "red": 0.00, "fouls": 11.92, "corners": 4.8, "g_for": 1.4, "g_against": 1.5},
    "Elche": {"yellow": 1.75, "red": 0.00, "fouls": 13.00, "corners": 3.5, "g_for": 0.8, "g_against": 1.7},
    "Espanyol": {"yellow": 1.67, "red": 0.00, "fouls": 12.75, "corners": 4.2, "g_for": 1.1, "g_against": 1.5},
    "Getafe": {"yellow": 2.50, "red": 0.17, "fouls": 15.42, "corners": 3.8, "g_for": 1.0, "g_against": 1.3},
    "Girona": {"yellow": 2.00, "red": 0.42, "fouls": 10.25, "corners": 5.2, "g_for": 2.0, "g_against": 1.5},
    "Levante": {"yellow": 2.00, "red": 0.00, "fouls": 12.58, "corners": 4.0, "g_for": 1.2, "g_against": 1.4},
    "Mallorca": {"yellow": 2.08, "red": 0.25, "fouls": 11.92, "corners": 3.9, "g_for": 1.0, "g_against": 1.2},
    "Osasuna": {"yellow": 2.08, "red": 0.17, "fouls": 12.67, "corners": 4.3, "g_for": 1.1, "g_against": 1.4},
    "Oviedo": {"yellow": 2.42, "red": 0.25, "fouls": 11.75, "corners": 3.6, "g_for": 1.0, "g_against": 1.3},
    "Vallecano": {"yellow": 2.33, "red": 0.00, "fouls": 13.67, "corners": 4.5, "g_for": 1.0, "g_against": 1.4},
    "Real Madrid": {"yellow": 1.92, "red": 0.17, "fouls": 10.17, "corners": 6.9, "g_for": 2.4, "g_against": 0.9},
    "Sociedad": {"yellow": 2.33, "red": 0.00, "fouls": 16.17, "corners": 5.5, "g_for": 1.4, "g_against": 1.1},
    "Sevilla": {"yellow": 3.42, "red": 0.00, "fouls": 15.92, "corners": 5.3, "g_for": 1.3, "g_against": 1.3},
    "Valencia": {"yellow": 1.67, "red": 0.00, "fouls": 12.08, "corners": 4.9, "g_for": 1.2, "g_against": 1.3},
    "Villarreal": {"yellow": 2.17, "red": 0.00, "fouls": 11.25, "corners": 5.2, "g_for": 1.7, "g_against": 1.6},
    "Flamengo": {"yellow": 2.21, "red": 0.15, "fouls": 14.03, "corners": 6.5, "g_for": 1.8, "g_against": 0.9},
    "Palmeiras": {"yellow": 1.88, "red": 0.09, "fouls": 14.12, "corners": 7.1, "g_for": 1.9, "g_against": 0.8},
    "São Paulo": {"yellow": 2.59, "red": 0.06, "fouls": 14.41, "corners": 6.0, "g_for": 1.5, "g_against": 0.9},
    "Corinthians": {"yellow": 3.00, "red": 0.12, "fouls": 14.09, "corners": 5.2, "g_for": 1.2, "g_against": 1.1},
    "Atlético Mineiro": {"yellow": 2.59, "red": 0.21, "fouls": 13.29, "corners": 5.8, "g_for": 1.6, "g_against": 1.1},
    "Vitória": {"yellow": 3.09, "red": 0.21, "fouls": 15.12, "corners": 4.2, "g_for": 1.1, "g_against": 1.5},
    "Botafogo": {"yellow": 2.44, "red": 0.09, "fouls": 14.59, "corners": 6.3, "g_for": 1.9, "g_against": 0.9},
    "Fortaleza": {"yellow": 2.94, "red": 0.26, "fouls": 14.26, "corners": 5.8, "g_for": 1.5, "g_against": 1.0},
    "Bahia": {"yellow": 2.24, "red": 0.15, "fouls": 12.00, "corners": 5.0, "g_for": 1.4, "g_against": 1.3},
    "Ceará": {"yellow": 2.24, "red": 0.06, "fouls": 14.03, "corners": 4.5, "g_for": 1.5, "g_against": 1.0},
    "Cruzeiro": {"yellow": 2.82, "red": 0.15, "fouls": 13.97, "corners": 5.5, "g_for": 1.3, "g_against": 1.0},
    "Fluminense": {"yellow": 2.06, "red": 0.09, "fouls": 12.79, "corners": 5.1, "g_for": 1.3, "g_against": 1.1},
    "Grêmio": {"yellow": 2.65, "red": 0.15, "fouls": 13.32, "corners": 5.3, "g_for": 1.4, "g_against": 1.2},
    "Internacional": {"yellow": 2.35, "red": 0.21, "fouls": 15.26, "corners": 5.6, "g_for": 1.4, "g_against": 1.0},
    "Juventude": {"yellow": 3.00, "red": 0.12, "fouls": 15.91, "corners": 4.3, "g_for": 1.2, "g_against": 1.4},
    "Mirassol": {"yellow": 2.24, "red": 0.03, "fouls": 12.41, "corners": 4.0, "g_for": 1.3, "g_against": 0.9},
    "Bragantino": {"yellow": 2.94, "red": 0.09, "fouls": 14.09, "corners": 5.4, "g_for": 1.3, "g_against": 1.2},
    "Santos": {"yellow": 2.71, "red": 0.12, "fouls": 14.21, "corners": 5.0, "g_for": 1.3, "g_against": 1.1},
    "Recife": {"yellow": 2.56, "red": 0.09, "fouls": 13.74, "corners": 4.6, "g_for": 1.4, "g_against": 1.1},
    "Vasco da Gama": {"yellow": 2.12, "red": 0.15, "fouls": 13.38, "corners": 4.8, "g_for": 1.3, "g_against": 1.4},
    "Augsburg": {"yellow": 3.30, "red": 0.00, "fouls": 13.60, "corners": 4.2, "g_for": 1.3, "g_against": 1.7},
    "Bayern Munich": {"yellow": 2.50, "red": 0.00, "fouls": 10.40, "corners": 7.5, "g_for": 2.8, "g_against": 0.9},
    "Dortmund": {"yellow": 1.60, "red": 0.00, "fouls": 12.30, "corners": 6.8, "g_for": 2.2, "g_against": 1.3},
    "Eintracht Frankfurt": {"yellow": 1.80, "red": 0.00, "fouls": 10.20, "corners": 5.5, "g_for": 1.7, "g_against": 1.4},
    "Freiburg": {"yellow": 1.70, "red": 0.20, "fouls": 8.90, "corners": 5.0, "g_for": 1.5, "g_against": 1.4},
    "Gladbach": {"yellow": 1.80, "red": 0.00, "fouls": 11.90, "corners": 5.1, "g_for": 1.6, "g_against": 1.8},
    "Hamburg": {"yellow": 2.40, "red": 0.40, "fouls": 13.50, "corners": 4.5, "g_for": 1.6, "g_against": 1.6},
    "Heidenheim": {"yellow": 1.50, "red": 0.00, "fouls": 10.40, "corners": 4.0, "g_for": 1.2, "g_against": 1.5},
    "Hoffenheim": {"yellow": 2.10, "red": 0.00, "fouls": 15.10, "corners": 4.8, "g_for": 1.6, "g_against": 1.9},
    "Köln": {"yellow": 1.90, "red": 0.00, "fouls": 9.70, "corners": 4.6, "g_for": 1.1, "g_against": 1.7},
    "Leverkusen": {"yellow": 2.50, "red": 0.20, "fouls": 9.20, "corners": 6.9, "g_for": 2.4, "g_against": 1.0},
    "Mainz 05": {"yellow": 2.60, "red": 0.30, "fouls": 13.30, "corners": 4.3, "g_for": 1.2, "g_against": 1.7},
    "RB Leipzig": {"yellow": 1.60, "red": 0.00, "fouls": 10.00, "corners": 6.2, "g_for": 2.2, "g_against": 1.1},
    "St. Pauli": {"yellow": 2.00, "red": 0.00, "fouls": 11.80, "corners": 3.9, "g_for": 1.0, "g_against": 1.5},
    "Stuttgart": {"yellow": 1.90, "red": 0.00, "fouls": 10.90, "corners": 5.7, "g_for": 2.1, "g_against": 1.4},
    "Union Berlin": {"yellow": 2.50, "red": 0.00, "fouls": 14.50, "corners": 4.1, "g_for": 1.1, "g_against": 1.5},
    "Werder Bremen": {"yellow": 2.70, "red": 0.00, "fouls": 9.50, "corners": 4.4, "g_for": 1.4, "g_against": 1.7},
    "Wolfsburg": {"yellow": 1.70, "red": 0.00, "fouls": 11.30, "corners": 5.0, "g_for": 1.3, "g_against": 1.5},
    "Atalanta": {"yellow": 1.55, "red": 0.00, "fouls": 9.91, "corners": 6.1, "g_for": 2.1, "g_against": 1.1},
    "Bologna": {"yellow": 2.09, "red": 0.00, "fouls": 15.00, "corners": 4.8, "g_for": 1.5, "g_against": 1.1},
    "Cagliari": {"yellow": 2.27, "red": 0.00, "fouls": 15.73, "corners": 4.2, "g_for": 1.0, "g_against": 1.6},
    "Como": {"yellow": 2.45, "red": 0.18, "fouls": 16.36, "corners": 3.8, "g_for": 0.9, "g_against": 1.6},
    "Cremonese": {"yellow": 2.18, "red": 0.00, "fouls": 11.91, "corners": 3.5, "g_for": 0.9, "g_against": 1.4},
    "Fiorentina": {"yellow": 2.36, "red": 0.00, "fouls": 14.64, "corners": 5.5, "g_for": 1.6, "g_against": 1.2},
    "Genoa": {"yellow": 2.73, "red": 0.00, "fouls": 15.27, "corners": 3.9, "g_for": 1.0, "g_against": 1.4},
    "Hellas Verona": {"yellow": 2.55, "red": 0.00, "fouls": 17.00, "corners": 4.0, "g_for": 1.0, "g_against": 1.6},
    "Inter": {"yellow": 1.55, "red": 0.00, "fouls": 14.45, "corners": 6.5, "g_for": 2.3, "g_against": 0.7},
    "Juventus": {"yellow": 1.45, "red": 0.00, "fouls": 12.45, "corners": 5.8, "g_for": 1.8, "g_against": 0.8},
    "Lazio": {"yellow": 1.45, "red": 0.18, "fouls": 11.00, "corners": 5.2, "g_for": 1.7, "g_against": 1.1},
    "Lecce": {"yellow": 1.82, "red": 0.00, "fouls": 12.55, "corners": 4.1, "g_for": 0.9, "g_against": 1.6},
    "Milan": {"yellow": 1.45, "red": 0.00, "fouls": 9.91, "corners": 6.0, "g_for": 2.0, "g_against": 1.2},
    "Napoli": {"yellow": 1.45, "red": 0.00, "fouls": 13.45, "corners": 6.2, "g_for": 1.9, "g_against": 1.0},
    "Parma": {"yellow": 2.00, "red": 0.18, "fouls": 12.00, "corners": 4.3, "g_for": 1.2, "g_against": 1.5},
    "Pisa": {"yellow": 1.82, "red": 0.00, "fouls": 13.27, "corners": 3.7, "g_for": 1.1, "g_against": 1.3},
    "Roma": {"yellow": 1.91, "red": 0.00, "fouls": 14.36, "corners": 5.4, "g_for": 1.8, "g_against": 1.3},
    "Sassuolo": {"yellow": 2.45, "red": 0.00, "fouls": 13.45, "corners": 4.9, "g_for": 1.4, "g_against": 1.6},
    "Torino": {"yellow": 1.82, "red": 0.00, "fouls": 13.64, "corners": 4.6, "g_for": 1.1, "g_against": 1.2},
    "Udinese": {"yellow": 2.18, "red": 0.00, "fouls": 12.91, "corners": 4.5, "g_for": 1.1, "g_against": 1.4},
    "Angers": {"yellow": 1.33, "red": 0.00, "fouls": 11.50, "corners": 4.0, "g_for": 1.0, "g_against": 1.5},
    "Auxerre": {"yellow": 2.33, "red": 0.33, "fouls": 13.83, "corners": 3.8, "g_for": 1.1, "g_against": 1.6},
    "Brest": {"yellow": 1.50, "red": 0.17, "fouls": 12.58, "corners": 5.2, "g_for": 1.4, "g_against": 1.2},
    "Le Havre": {"yellow": 1.92, "red": 0.00, "fouls": 14.58, "corners": 3.9, "g_for": 0.9, "g_against": 1.5},
    "Lens": {"yellow": 2.25, "red": 0.17, "fouls": 13.00, "corners": 5.5, "g_for": 1.5, "g_against": 1.1},
    "Lille": {"yellow": 2.42, "red": 0.00, "fouls": 13.75, "corners": 5.8, "g_for": 1.6, "g_against": 1.0},
    "Lorient": {"yellow": 2.33, "red": 0.00, "fouls": 10.50, "corners": 4.3, "g_for": 1.3, "g_against": 1.7},
    "Lyon": {"yellow": 2.00, "red": 0.33, "fouls": 14.42, "corners": 5.6, "g_for": 1.6, "g_against": 1.5},
    "Marseille": {"yellow": 2.25, "red": 0.00, "fouls": 12.50, "corners": 6.0, "g_for": 1.8, "g_against": 1.2},
    "Metz": {"yellow": 1.50, "red": 0.25, "fouls": 11.42, "corners": 3.7, "g_for": 1.0, "g_against": 1.6},
    "Monaco": {"yellow": 2.58, "red": 0.17, "fouls": 13.17, "corners": 6.1, "g_for": 2.0, "g_against": 1.3},
    "Montpellier": {"yellow": 1.67, "red": 0.00, "fouls": 11.58, "corners": 4.8, "g_for": 1.3, "g_against": 1.6},
    "Nice": {"yellow": 2.33, "red": 0.00, "fouls": 12.92, "corners": 5.3, "g_for": 1.3, "g_against": 1.0},
    "Paris FC": {"yellow": 2.00, "red": 0.17, "fouls": 11.42, "corners": 3.5, "g_for": 1.0, "g_against": 1.2},
    "Paris Saint-Germain": {"yellow": 1.25, "red": 0.00, "fouls": 10.08, "corners": 7.0, "g_for": 2.5, "g_against": 0.8},
    "Rennes": {"yellow": 2.00, "red": 0.25, "fouls": 12.58, "corners": 5.7, "g_for": 1.6, "g_against": 1.3},
    "Strasbourg": {"yellow": 2.25, "red": 0.25, "fouls": 11.83, "corners": 4.5, "g_for": 1.3, "g_against": 1.5},
    "Toulouse": {"yellow": 2.67, "red": 0.00, "fouls": 14.17, "corners": 4.6, "g_for": 1.2, "g_against": 1.5}
}

@st.cache_data(ttl=0) 
def load_data():
    data = {"teams": {}, "referees": {}, "error": None}
    try:
        # Tenta carregar do CSV primeiro
        df_teams = pd.read_csv("dados_times.csv")
        if len(df_teams.columns) < 2: df_teams = pd.read_csv("dados_times.csv", sep=";")
        
        csv_dict = {}
        for _, row in df_teams.iterrows():
            g_for = row['GolsFeitos'] if 'GolsFeitos' in df_teams.columns else 1.2
            g_against = row['GolsSofridos'] if 'GolsSofridos' in df_teams.columns else 1.2
            
            csv_dict[row['Time']] = {
                "yellow": row['CartoesAmarelos'],
                "red": row['CartoesVermelhos'],
                "fouls": row['Faltas'],
                "corners": row['Escanteios'],
                "g_for": g_for,
                "g_against": g_against
            }
        data["teams"] = csv_dict
    except Exception as e:
        # Se falhar, usa o BACKUP
        data["teams"] = BACKUP_DATA
        data["error"] = f"Usando Backup. Erro: {str(e)}"

    try:
        df_refs = pd.read_csv("arbitros.csv")
        if len(df_refs.columns) < 2: df_refs = pd.read_csv("arbitros.csv", sep=";")
        for _, row in df_refs.iterrows():
            data["referees"][row['Nome']] = row['Fator']
    except:
        pass

    return data

DB = load_data()
RAW_STATS_DATA = DB["teams"]
REFEREES_DATA = DB["referees"]

# ==============================================================================
# 2. MOTOR DE CÁLCULO
# ==============================================================================
class StatsEngine:
    def __init__(self):
        self.stats_data = RAW_STATS_DATA

    def calculate_poisson_prob(self, line, avg):
        if avg == 0: return 0.0
        k = int(line) 
        prob = 1 - poisson.cdf(k, avg)
        return round(prob * 100, 1)

    def get_team_averages(self, team):
        if team in self.stats_data:
            data = self.stats_data[team]
            avg_cards = data['yellow'] + (data['red'] * 2.5)
            # Garante que pega g_for do dicionário (seja backup ou csv)
            g_for = data.get('g_for', 1.2)
            g_against = data.get('g_against', 1.2)
            
            return {
                "corners": data['corners'],
                "cards": avg_cards,
                "fouls": data['fouls'],
                "goals_for": g_for,
                "goals_against": g_against
            }
        else:
            return {"corners": 5.0, "cards": 2.2, "fouls": 12.5, "goals_for": 1.2, "goals_against": 1.2}

    def predict_match_full(self, home, away, ref_factor, match_multiplier):
        h_stats = self.get_team_averages(home)
        a_stats = self.get_team_averages(away)

        # Escanteios (Com Fator Champions)
        exp_corners_h = h_stats['corners'] * 1.10 * match_multiplier
        exp_corners_a = a_stats['corners'] * 0.85 * match_multiplier
        exp_corners_total = exp_corners_h + exp_corners_a

        # Cartões
        match_fouls = h_stats['fouls'] + a_stats['fouls']
        tension_factor = 1.0
        if match_fouls > 28: tension_factor = 1.25
        elif match_fouls > 25: tension_factor = 1.15
        
        exp_cards_h = h_stats['cards'] * tension_factor * ref_factor
        exp_cards_a = a_stats['cards'] * tension_factor * ref_factor
        exp_cards_total = exp_cards_h + exp_cards_a

        # Gols (Com Fator Champions)
        lambda_home = ((h_stats['goals_for'] + a_stats['goals_against']) / 2) * match_multiplier
        lambda_away = ((a_stats['goals_for'] + h_stats['goals_against']) / 2) * match_multiplier
        exp_goals_total = lambda_home + lambda_away
        
        prob_home_score = 1 - poisson.pmf(0, lambda_home)
        prob_away_score = 1 - poisson.pmf(0, lambda_away)
        prob_btts = prob_home_score * prob_away_score * 100

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
                "prob_btts": round(prob_btts, 1),
                "game_probs": {
                    "line_1_5": self.calculate_poisson_prob(1.5, exp_goals_total),
                    "line_2_5": self.calculate_poisson_prob(2.5, exp_goals_total)
                }
            }
        }

# ==============================================================================
# 3. INTERFACE
# ==============================================================================
st.set_page_config(page_title="FutPrevisão Pro", layout="wide", page_icon="⚽")

col_logo, col_logout = st.columns([4, 1])
with col_logo:
    st.title("FutPrevisão Pro 🚀")
    st.caption("Versão 2.1 (Final) | Inteligência Artificial")
with col_logout:
    if st.button("Sair"):
        st.session_state["password_correct"] = False
        st.rerun()
st.markdown("---")

# Aviso se estiver usando Backup
if DB.get("error"):
    st.warning(f"⚠️ Modo de Segurança Ativo: {DB['error']}")

engine = StatsEngine()
if engine.stats_data:
    all_teams = sorted(list(engine.stats_data.keys()))
else:
    all_teams = ["Erro Crítico: Sem Dados"]

# Seletores
c1, c2 = st.columns([1, 1])
with c1:
    idx_h = all_teams.index("Chelsea") if "Chelsea" in all_teams else 0
    home_team = st.selectbox("Mandante (Casa)", all_teams, index=idx_h)
with c2:
    idx_a = all_teams.index("Getafe") if "Getafe" in all_teams else 0
    away_team = st.selectbox("Visitante (Fora)", all_teams, index=idx_a)

st.write("")

# Configurações Avançadas
c3, c4 = st.columns([1, 1])

with c3:
    st.markdown("### 👮 Arbitragem")
    if REFEREES_DATA:
        ref_options = ["Outro / Não está na lista"] + sorted(list(REFEREES_DATA.keys()))
    else:
        ref_options = ["Outro / Não está na lista"]
    
    selected_ref = st.selectbox("Selecione o Árbitro:", ref_options)
    
    final_ref_factor = 1.0
    if selected_ref == "Outro / Não está na lista":
        manual_profile = st.selectbox("Perfil Manual:", 
                                      ["Normal (Padrão)", "Rigoroso (Cartoeiro)", "Leniente (Deixa Jogar)"])
        if manual_profile == "Rigoroso (Cartoeiro)": final_ref_factor = 1.20
        elif manual_profile == "Leniente (Deixa Jogar)": final_ref_factor = 0.80
    else:
        if final_ref_factor > 1.0: 
            st.info(f"🔴 **{selected_ref}** é Rigoroso (Aumenta cartões).")
        elif final_ref_factor < 1.0: 
            st.info(f"🟢 **{selected_ref}** é Leniente (Diminui cartões).")
        else:
            st.info(f"⚪ **{selected_ref}** é Normal (Padrão da Liga).")

with c4:
    st.markdown("### 🏆 Contexto")
    match_type = st.radio("Tipo de Jogo:", 
                          ["Liga Normal (Pontos Corridos)", "Champions / Clássico / Decisivo"],
                          horizontal=False)
    match_multiplier = 1.0
    if match_type == "Champions / Clássico / Decisivo":
        match_multiplier = 0.85
        st.caption("ℹ️ Modo Elite: Redutor de 15% aplicado.")

st.markdown("---")

if st.button("🎲 Gerar Previsões", use_container_width=True):
    if home_team in engine.stats_data:
        # Passa TODOS os argumentos (times, juiz e contexto)
        pred = engine.predict_match_full(home_team, away_team, final_ref_factor, match_multiplier)
        
        # ESCANTEIOS
        st.subheader("🚩 Escanteios")
        ec1, ec2, ec3 = st.columns(3)
        with ec1:
            st.markdown(f"**{home_team}**")
            st.write(f"Over 3.5: **{pred['corners']['home']['line_3_5']}%**")
            st.write(f"Over 4.5: **{pred['corners']['home']['line_4_5']}%**")
        with ec2:
            st.metric("Total Esperado", pred['corners']['total_expected'])
        with ec3:
            st.markdown(f"**{away_team}**")
            st.write(f"Over 3.5: **{pred['corners']['away']['line_3_5']}%**")
            st.write(f"Over 4.5: **{pred['corners']['away']['line_4_5']}%**")
        st.markdown("---")

        # CARTÕES
        st.subheader("🟨 Cartões")
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            st.markdown(f"**{home_team}**")
            st.write(f"Over 1.5: **{pred['cards']['home']['line_1_5']}%**")
            st.write(f"Over 2.5: **{pred['cards']['home']['line_2_5']}%**")
        with cc2:
            st.metric("Total Esperado", pred['cards']['total_expected'])
            st.write(f"Jogo Over 3.5: **{pred['cards']['game_probs']['line_3_5']}%**")
            st.write(f"Jogo Over 4.5: **{pred['cards']['game_probs']['line_4_5']}%**")
        with cc3:
            st.markdown(f"**{away_team}**")
            st.write(f"Over 1.5: **{pred['cards']['away']['line_1_5']}%**")
            st.write(f"Over 2.5: **{pred['cards']['away']['line_2_5']}%**")
        st.markdown("---")

        # GOLS
        st.subheader("⚽ Gols")
        gc1, gc2, gc3 = st.columns(3)
        with gc1:
            st.metric("xG Total", pred['goals']['total_expected'])
        with gc2:
            st.metric("Ambas Marcam", f"{pred['goals']['prob_btts']}%")
        with gc3:
            st.write(f"Over 1.5 Gols: **{pred['goals']['game_probs']['line_1_5']}%**")
            st.write(f"Over 2.5 Gols: **{pred['goals']['game_probs']['line_2_5']}%**")
    else:
        st.error("Erro: Time não encontrado.")

