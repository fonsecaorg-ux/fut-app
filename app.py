import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import poisson
import extra_streamlit_components as stx # Biblioteca de Cookies
import time

# ==============================================================================
# 0. SISTEMA DE LOGIN COM COOKIES (LOGIN PERSISTENTE)
# ==============================================================================
def check_password():
    st.set_page_config(page_title="FutPrevisão Pro", layout="wide", page_icon="⚽")
    
    # Gerenciador de Cookies
    cookie_manager = stx.CookieManager()
    
    # Verifica se já existe um cookie válido (usuário já logou antes)
    cookie_value = cookie_manager.get(cookie="futprevisao_access")
    
    if cookie_value == "granted":
        return True

    # Se não tem cookie, mostra tela de login
    st.title("🔒 Acesso Restrito")
    st.info("Faça login para acessar as previsões.")
    
    user = st.text_input("Usuário")
    password = st.text_input("Senha", type="password")
    
    if st.button("Entrar"):
        if "passwords" in st.secrets:
            if user in st.secrets["passwords"] and password == st.secrets["passwords"][user]:
                # SENHA CORRETA: Cria o cookie que dura 1 dia
                cookie_manager.set("futprevisao_access", "granted", key="set_access")
                st.success("Login realizado! Recarregando...")
                time.sleep(1) # Espera gravar o cookie
                st.rerun() # Recarrega a página para entrar
            else:
                st.error("😕 Usuário ou senha incorretos.")
        else:
            st.error("Erro de configuração no servidor.")
            
    return False

# Se o login não for validado pelo cookie ou senha, para aqui.
if not check_password():
    st.stop()

# ==============================================================================
# 1. CARREGAMENTO DE DADOS (COM BACKUP)
# ==============================================================================
BACKUP_DATA = {
    "Arsenal": {"yellow": 1.00, "red": 0.00, "fouls": 10.45, "corners": 7.5, "g_for": 2.3, "g_against": 0.8},
    "Man City": {"yellow": 1.64, "red": 0.00, "fouls": 9.36, "corners": 8.2, "g_for": 2.6, "g_against": 0.9},
    "Chelsea": {"yellow": 2.09, "red": 0.27, "fouls": 11.55, "corners": 6.4, "g_for": 2.1, "g_against": 1.3},
}

@st.cache_data(ttl=0) 
def load_data():
    data = {"teams": {}, "referees": {}, "error": None}
    
    # CARREGA TIMES
    try:
        try:
            df_teams = pd.read_csv("dados_times.csv")
            if len(df_teams.columns) < 2: df_teams = pd.read_csv("dados_times.csv", sep=";")
        except:
            df_teams = pd.read_csv("dados_times.csv", sep=";")

        csv_dict = {}
        for _, row in df_teams.iterrows():
            def safe_float(val):
                try: return float(str(val).replace(',', '.'))
                except: return 1.0

            csv_dict[row['Time']] = {
                "yellow": safe_float(row['CartoesAmarelos']),
                "red": safe_float(row['CartoesVermelhos']),
                "fouls": safe_float(row['Faltas']),
                "corners": safe_float(row['Escanteios']),
                "g_for": safe_float(row.get('GolsFeitos', 1.2)),
                "g_against": safe_float(row.get('GolsSofridos', 1.2))
            }
        data["teams"] = csv_dict
    except Exception as e:
        data["teams"] = BACKUP_DATA
        data["error"] = f"Usando Backup: {str(e)}"

    # CARREGA ÁRBITROS
    try:
        df_refs = pd.read_csv("arbitros.csv")
        if len(df_refs.columns) < 2: df_refs = pd.read_csv("arbitros.csv", sep=";")
        for _, row in df_refs.iterrows():
            fator_str = str(row['Fator']).replace(',', '.')
            data["referees"][row['Nome']] = float(fator_str)
    except: pass

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
                "goals_for": data.get('g_for', 1.2),
                "goals_against": data.get('g_against', 1.2)
            }
        else:
            return {"corners": 5.0, "cards": 2.2, "fouls": 12.5, "goals_for": 1.2, "goals_against": 1.2}

    def predict_match_full(self, home, away, ref_factor, match_multiplier):
        h_stats = self.get_team_averages(home)
        a_stats = self.get_team_averages(away)

        # Escanteios
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

        # Gols
        lambda_home = ((h_stats['goals_for'] + a_stats['goals_against']) / 2) * match_multiplier
        lambda_away = ((a_stats['goals_for'] + h_stats['goals_against']) / 2) * match_multiplier
        exp_goals_total = lambda_home + lambda_away
        
        prob_home_score = 1 - poisson.pmf(0, lambda_home)
        prob_away_score = 1 - poisson.pmf(0, lambda_away)
        prob_btts = prob_home_score * prob_away_score * 100

        # --- NOVO: RADAR DE ESCANTEIOS TOTAIS ---
        corners_radar = {}
        for line in [7.5, 8.5, 9.5, 10.5, 11.5, 12.5]:
            corners_radar[f"Over {line}"] = self.calculate_poisson_prob(line, exp_corners_total)

        return {
            "corners": {
                "total_expected": round(exp_corners_total, 2),
                "radar": corners_radar, # Nova saída
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
# O set_page_config foi movido para dentro do check_password para evitar erro de duplicação

col_logo, col_logout = st.columns([4, 1])
with col_logo:
    st.title("FutPrevisão Pro 🚀")
    st.caption("Versão 2.5 | Cookies + Radar de Escanteios")
with col_logout:
    if st.button("Sair / Logout"):
        # Apaga o cookie para sair
        cookie_manager = stx.CookieManager()
        cookie_manager.delete("futprevisao_access")
        st.rerun()

st.markdown("---")

if DB.get("error"):
    st.toast(f"Aviso: {DB['error']}", icon="⚠️")

engine = StatsEngine()
if engine.stats_data:
    all_teams = sorted(list(engine.stats_data.keys()))
else:
    all_teams = ["Erro de Dados"]

c1, c2 = st.columns([1, 1])
with c1:
    idx_h = all_teams.index("Chelsea") if "Chelsea" in all_teams else 0
    home_team = st.selectbox("Mandante (Casa)", all_teams, index=idx_h)
with c2:
    idx_a = all_teams.index("Getafe") if "Getafe" in all_teams else 0
    away_team = st.selectbox("Visitante (Fora)", all_teams, index=idx_a)

st.write("")
c3, c4 = st.columns([1, 1])
with c3:
    st.markdown("### 👮 Arbitragem")
    ref_options = ["Outro / Não está na lista"] + sorted(list(REFEREES_DATA.keys())) if REFEREES_DATA else ["Outro"]
    selected_ref = st.selectbox("Árbitro:", ref_options)
    
    final_ref_factor = 1.0
    if selected_ref == "Outro / Não está na lista":
        manual_profile = st.selectbox("Perfil Manual:", ["Normal", "Rigoroso", "Leniente"])
        if manual_profile == "Rigoroso": final_ref_factor = 1.20
        elif manual_profile == "Leniente": final_ref_factor = 0.80
    else:
        final_ref_factor = REFEREES_DATA[selected_ref]
        if final_ref_factor > 1.0: st.info(f"🔴 {selected_ref}: Rigoroso")
        elif final_ref_factor < 1.0: st.info(f"🟢 {selected_ref}: Leniente")
        else: st.info(f"⚪ {selected_ref}: Normal")

with c4:
    st.markdown("### 🏆 Contexto")
    match_type = st.radio("Competição:", ["Liga Normal", "Champions / Clássico"])
    match_multiplier = 0.85 if match_type == "Champions / Clássico" else 1.0

st.markdown("---")

if st.button("🎲 Gerar Previsões", use_container_width=True):
    if home_team in engine.stats_data:
        pred = engine.predict_match_full(home_team, away_team, final_ref_factor, match_multiplier)
        
        # --- ESCANTEIOS (NOVO RADAR) ---
        st.subheader("🚩 Escanteios")
        
        # Tabela de Probabilidades Totais (Radar)
        st.markdown("##### 📡 Radar de Escanteios (Total do Jogo)")
        radar = pred['corners']['radar']
        cols = st.columns(6)
        # Mostra as linhas de 7.5 a 12.5 lado a lado
        keys = list(radar.keys())
        for i, col in enumerate(cols):
            line_name = keys[i]
            prob = radar[line_name]
            # Destaca se for > 70%
            if prob >= 70:
                col.metric(line_name, f"{prob}%", delta="Alta")
            else:
                col.metric(line_name, f"{prob}%")
        
        st.divider()
        
        # Previsões Individuais (Mantidas)
        ec1, ec2, ec3 = st.columns(3)
        with ec1:
            st.markdown(f"**{home_team}**")
            st.write(f"Over 3.5: **{pred['corners']['home']['line_3_5']}%**")
            st.write(f"Over 4.5: **{pred['corners']['home']['line_4_5']}%**")
        with ec2:
            st.metric("Média Esperada", pred['corners']['total_expected'])
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
