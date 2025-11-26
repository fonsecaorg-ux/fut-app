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
# 1. CARREGAMENTO DE DADOS (TIMES E ÁRBITROS)
# ==============================================================================
@st.cache_data(ttl=600)
def load_data():
    data = {"teams": {}, "referees": {}}
    
    # Carrega Times
    try:
        df_teams = pd.read_csv("dados_times.csv")
        for _, row in df_teams.iterrows():
            data["teams"][row['Time']] = {
                "yellow": row['CartoesAmarelos'],
                "red": row['CartoesVermelhos'],
                "fouls": row['Faltas'],
                "corners": row['Escanteios'],
                "g_for": row['GolsFeitos'],
                "g_against": row['GolsSofridos']
            }
    except Exception:
        pass # Se der erro, fica vazio e avisa na interface

    # Carrega Árbitros
    try:
        df_refs = pd.read_csv("arbitros.csv")
        for _, row in df_refs.iterrows():
            data["referees"][row['Nome']] = row['Fator']
    except Exception:
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
            return {
                "corners": data['corners'],
                "cards": avg_cards,
                "fouls": data['fouls'],
                "goals_for": data['g_for'],
                "goals_against": data['g_against']
            }
        else:
            return {"corners": 5.0, "cards": 2.2, "fouls": 12.5, "goals_for": 1.2, "goals_against": 1.2}

    def predict_match_full(self, home, away, ref_factor):
        h_stats = self.get_team_averages(home)
        a_stats = self.get_team_averages(away)

        # Escanteios
        exp_corners_h = h_stats['corners'] * 1.10
        exp_corners_a = a_stats['corners'] * 0.85
        exp_corners_total = exp_corners_h + exp_corners_a

        # Cartões
        match_fouls = h_stats['fouls'] + a_stats['fouls']
        tension_factor = 1.0
        if match_fouls > 28: tension_factor = 1.25
        elif match_fouls > 25: tension_factor = 1.15
        
        # Aplica o fator do árbitro (vindo do CSV ou do manual)
        exp_cards_h = h_stats['cards'] * tension_factor * ref_factor
        exp_cards_a = a_stats['cards'] * tension_factor * ref_factor
        exp_cards_total = exp_cards_h + exp_cards_a

        # Gols
        lambda_home = (h_stats['goals_for'] + a_stats['goals_against']) / 2
        lambda_away = (a_stats['goals_for'] + h_stats['goals_against']) / 2
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
    st.caption("Versão 2.0 | Dados Reais de Árbitros")
with col_logout:
    if st.button("Sair"):
        st.session_state["password_correct"] = False
        st.rerun()
st.markdown("---")

engine = StatsEngine()

# Seletor de Times
if engine.stats_data:
    all_teams = sorted(list(engine.stats_data.keys()))
else:
    all_teams = ["Erro no CSV de Times"]

c1, c2 = st.columns([1, 1])
with c1:
    home_team = st.selectbox("Mandante (Casa)", all_teams, index=0)
with c2:
    away_team = st.selectbox("Visitante (Fora)", all_teams, index=1)

st.write("")

# --- SELETOR DE ÁRBITRO HÍBRIDO ---
st.markdown("### 👮 Arbitragem")
if REFEREES_DATA:
    ref_options = ["Outro / Não está na lista"] + sorted(list(REFEREES_DATA.keys()))
else:
    ref_options = ["Outro / Não está na lista"]

selected_ref = st.selectbox("Selecione o Árbitro da Partida:", ref_options)

# Lógica de Decisão do Fator
final_ref_factor = 1.0

if selected_ref == "Outro / Não está na lista":
    # Se não achou o nome, mostra o seletor manual antigo
    manual_profile = st.selectbox("Defina o Perfil Manualmente:", 
                                  ["Normal (Padrão)", "Rigoroso (Cartoeiro)", "Leniente (Deixa Jogar)"])
    if manual_profile == "Rigoroso (Cartoeiro)": final_ref_factor = 1.20
    elif manual_profile == "Leniente (Deixa Jogar)": final_ref_factor = 0.80
    else: final_ref_factor = 1.0
else:
    # Se achou o nome, usa o fator do CSV e mostra pro usuário
    factor_from_csv = REFEREES_DATA[selected_ref]
    final_ref_factor = factor_from_csv
    
    # Feedback visual
    if factor_from_csv > 1.0:
        st.info(f"ℹ️ **{selected_ref}** é considerado Rigoroso (Fator {factor_from_csv}).")
    elif factor_from_csv < 1.0:
        st.info(f"ℹ️ **{selected_ref}** é considerado Leniente (Fator {factor_from_csv}).")
    else:
        st.success(f"ℹ️ **{selected_ref}** tem estatística Normal.")

st.markdown("---")

if st.button("🎲 Gerar Previsões", use_container_width=True):
    if home_team in engine.stats_data:
        pred = engine.predict_match_full(home_team, away_team, final_ref_factor)
        
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
