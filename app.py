import streamlit as st
import pandas as pd
from scipy.stats import poisson
import numpy as np

# Configuração da Página
st.set_page_config(page_title="EsporteStats PRO V11.5", page_icon="⚽", layout="wide")

# --- IMPORTAÇÃO BLINDADA DO DATABASE ---
try:
    from database import RAW_CORNERS_DATA, RAW_GOALS_DATA, RAW_CARDS_DATA, RAW_FOULS_DATA
except ImportError:
    st.error("🚨 ERRO: Arquivo 'database.py' não encontrado.")
    st.stop()

# ==============================================================================
# 1. MOTORES E PARSERS
# ==============================================================================

def parse_match_logs(raw_text, market_type="corners"):
    data = []
    if not raw_text or not isinstance(raw_text, str):
        return pd.DataFrame(columns=['Date', 'HomeTeam', 'AwayTeam', 'HomeStats', 'AwayStats'])
    
    lines = raw_text.strip().split('\n')
    if market_type == "corners":
        pattern = r'"(\d{2}-\d{2}-\d{4})".*?"(.*?)"\s*,.*?"(\d+)-(\d+)"\s*,.*?"(.*?)"'
    else:
        pattern = r'(\d{2}[\/-]\d{2}[\/-]\d{4}):\s*(.+?)\s+(\d+)-(\d+)\s+(.+)'

    for line in lines:
        match = re.search(pattern, line)
        if match:
            try:
                data.append({
                    'Date': match.group(1),
                    'HomeTeam': match.group(2).strip(),
                    'AwayTeam': match.group(5).strip(),
                    'HomeStats': int(match.group(3)),
                    'AwayStats': int(match.group(4))
                })
            except: continue
    df = pd.DataFrame(data)
    if not df.empty:
        df.drop_duplicates(subset=['Date', 'HomeTeam', 'AwayTeam'], inplace=True)
    return df

def parse_card_summary(raw_text, fouls_dict, league_name):
    stats = {}
    if not raw_text: return stats
    
    fouls_data = fouls_dict.get(league_name, {})
    lines = [l.strip() for l in raw_text.split('\n') if l.strip()]
    current_team = None
    
    for line in lines:
        if ":" not in line and len(line) > 3:
            name = line.split("(")[0].strip()
            current_team = name
            f_stats = fouls_data.get(current_team, {'cometidas': 12.0, 'sofridas': 12.0})
            stats[current_team] = {'yellow': 0, 'fouls_comm': f_stats['cometidas'], 'fouls_suff': f_stats['sofridas']}
        elif current_team and "Cartões Amarelos:" in line:
            try: stats[current_team]['yellow'] = float(line.split(":")[1].replace("por jogo", "").replace(",", ".").strip())
            except: pass
    return stats

# --- NOVA FUNÇÃO DE TENDÊNCIAS (SEU SISTEMA) ---
def analyze_corner_trends(df, team, side):
    """Gera o relatório de consistência (Trend System)"""
    if df.empty: return "Sem dados suficientes."
    
    # Filtra jogos
    if side == 'Home':
        matches = df[df['HomeTeam'] == team]
        stats = matches['HomeStats']
    else:
        matches = df[df['AwayTeam'] == team]
        stats = matches['AwayStats']
        
    total_games = len(stats)
    if total_games == 0: return "Sem jogos registrados."

    report = f"✅ **{team} ({'Casa' if side == 'Home' else 'Fora'}):**\n"
    report += f"Últimos {total_games} jogos:\n"
    
    best_line = None
    best_rate = -1

    for line in [3.5, 4.5, 5.5, 6.5]:
        hits = sum(1 for x in stats if x > line)
        rate = (hits / total_games) * 100
        
        # Define emoji
        if rate >= 60: icon = "✅"
        elif rate >= 40: icon = "⚠️"
        else: icon = "❌"
        
        report += f"- {side} +{line}: {hits}/{total_games} ({rate:.1f}%) {icon}\n"
        
        # Lógica para "Melhor Aposta"
        if rate >= 60 and line > (best_rate if best_rate else 0):
            best_line = line
            best_rate = rate

    report += "Análise preliminar:\n"
    if best_line:
        report += f"✅ {team} é CONSISTENTE em +{best_line}\n"
        report += f"💡 Melhor aposta: {team} +{best_line}"
    else:
        report += "⚠️ Time inconsistente em linhas altas."
        
    return report

class ContextualEngine:
    def __init__(self, df):
        self.df = df
        self.teams = sorted(list(set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique()))) if not df.empty else []

    def get_lambdas(self, home, away):
        if self.df.empty: return 0, 0, 0, 0, 0, 0
        h_games = self.df[self.df['HomeTeam'] == home]
        h_atk = h_games['HomeStats'].mean() if not h_games.empty else 1.5
        a_games = self.df[self.df['AwayTeam'] == away]
        a_def = a_games['HomeStats'].mean() if not a_games.empty else 1.5 
        l_home = (h_atk + a_def) / 2
        
        a_games_atk = self.df[self.df['AwayTeam'] == away]
        a_atk = a_games_atk['AwayStats'].mean() if not a_games_atk.empty else 1.0
        h_games_def = self.df[self.df['HomeTeam'] == home]
        h_def = h_games_def['AwayStats'].mean() if not h_games_def.empty else 1.0
        l_away = (a_atk + h_def) / 2
        return l_home, l_away, h_atk, h_def, a_atk, a_def

class CardEngineV2:
    def __init__(self, stats_dict):
        self.stats = stats_dict
        self.teams = sorted(list(stats_dict.keys()))
    
    def get_lambdas(self, home, away):
        sh = self.stats.get(home, {'yellow': 2.0, 'fouls_comm': 12.0, 'fouls_suff': 12.0})
        sa = self.stats.get(away, {'yellow': 2.0, 'fouls_comm': 12.0, 'fouls_suff': 12.0})
        lh = sh['yellow'] + (sa['fouls_comm'] * 0.08) 
        la = sa['yellow'] + (sh['fouls_comm'] * 0.08)
        return lh, la, sh, sa

# Helper de Cor
def get_color(prob):
    if prob >= 75: return "green"
    if prob >= 50: return "orange"
    return "red"

# ==============================================================================
# 2. INTERFACE DASHBOARD
# ==============================================================================

with st.sidebar:
    st.header("EsporteStats PRO")
    leagues = list(RAW_CORNERS_DATA.keys())
    league = st.selectbox("Liga", leagues)
    
    df_corn = parse_match_logs(RAW_CORNERS_DATA.get(league, ""), "corners")
    eng_corn = ContextualEngine(df_corn)
    
    df_goals = parse_match_logs(RAW_GOALS_DATA.get(league, ""), "goals")
    eng_goals = ContextualEngine(df_goals)
    
    card_stats = parse_card_summary(RAW_CARDS_DATA, RAW_FOULS_DATA, league)
    eng_card = CardEngineV2(card_stats)
    
    teams = eng_goals.teams if eng_goals.teams else eng_corn.teams
    
    if teams:
        h_team = st.selectbox("Mandante", teams)
        a_team = st.selectbox("Visitante", teams, index=1 if len(teams)>1 else 0)
        btn_calc = st.button("🔥 GERAR ANÁLISE", type="primary")
    else:
        st.error("Sem dados.")
        btn_calc = False
    
    st.info("ℹ️ Sistema Híbrido: Poisson + Tendências Históricas.")

if btn_calc:
    st.markdown(f"<h2 style='text-align: center;'>⚔️ {h_team} x {a_team}</h2>", unsafe_allow_html=True)
    st.markdown("---")

    col_corn, col_goals, col_cards = st.columns(3)

    # --- ESCANTEIOS ---
    with col_corn:
        st.subheader("🚩 Escanteios")
        if not df_corn.empty:
            lc_h, lc_a, _, _, _, _ = eng_corn.get_lambdas(h_team, a_team)
            tot_corn = lc_h + lc_a
            
            st.info(f"**Exp. Total: {tot_corn:.2f}**")
            
            # --- NOVO SISTEMA DE TENDÊNCIAS ---
            st.markdown("#### 📊 Tendências (Últimos Jogos)")
            trend_h = analyze_corner_trends(df_corn, h_team, 'Home')
            trend_a = analyze_corner_trends(df_corn, a_team, 'Away')
            
            with st.expander(f"Análise {h_team} (Casa)", expanded=True):
                st.markdown(trend_h)
            
            with st.expander(f"Análise {a_team} (Fora)", expanded=True):
                st.markdown(trend_a)
            
            st.markdown("---")
            st.markdown("**Probabilidade Futura (Poisson):**")
            
            # Individuais Casa
            p_h_35 = (1-poisson.cdf(3, lc_h))*100
            p_h_45 = (1-poisson.cdf(4, lc_h))*100
            st.markdown(f"🏠 **{h_team}:**")
            st.markdown(f"+3.5: :{get_color(p_h_35)}[**{p_h_35:.0f}%**] | +4.5: :{get_color(p_h_45)}[**{p_h_45:.0f}%**]")
            
            # Individuais Fora
            p_a_35 = (1-poisson.cdf(3, lc_a))*100
            p_a_45 = (1-poisson.cdf(4, lc_a))*100
            st.markdown(f"✈️ **{a_team}:**")
            st.markdown(f"+3.5: :{get_color(p_a_35)}[**{p_a_35:.0f}%**] | +4.5: :{get_color(p_a_45)}[**{p_a_45:.0f}%**]")

        else:
            st.warning("Sem dados.")

    # --- GOLS ---
    with col_goals:
        st.subheader("⚽ Gols")
        if not df_goals.empty:
            lg_h, lg_a, _, _, _, _ = eng_goals.get_lambdas(h_team, a_team)
            tot_goals = lg_h + lg_a
            btts = (1 - poisson.pmf(0, lg_h)) * (1 - poisson.pmf(0, lg_a)) * 100
            
            st.success(f"**Exp. Total: {tot_goals:.2f}**")
            st.markdown(f"**BTTS (Ambas):** :{get_color(btts)}[**{btts:.0f}%**]")
            
            st.markdown("---")
            
            # LINHAS PRINCIPAIS
            p_15 = (1-poisson.cdf(1, tot_goals))*100
            p_25 = (1-poisson.cdf(2, tot_goals))*100
            
            st.markdown(f"**Over 1.5:** :{get_color(p_15)}[**{p_15:.0f}%**]")
            st.markdown(f"**Over 2.5:** :{get_color(p_25)}[**{p_25:.0f}%**]")
            
            st.markdown("---")
            st.write(f"Over 0.5 Casa ({h_team}): :{get_color((1-poisson.cdf(0, lg_h))*100)}[**{(1-poisson.cdf(0, lg_h))*100:.0f}%**]")
            st.write(f"Over 0.5 Fora ({a_team}): :{get_color((1-poisson.cdf(0, lg_a))*100)}[**{(1-poisson.cdf(0, lg_a))*100:.0f}%**]")

    # --- CARTÕES ---
    with col_cards:
        st.subheader("🟨 Cartões")
        if h_team in eng_card.teams and a_team in eng_card.teams:
            lk_h, lk_a, _, _ = eng_card.get_lambdas(h_team, a_team)
            tot_card = lk_h + lk_a
            
            st.warning(f"**Exp. Total: {tot_card:.2f}**")
            
            st.markdown("**Total no Jogo:**")
            p_tot_35 = (1-poisson.cdf(3, tot_card))*100
            p_tot_45 = (1-poisson.cdf(4, tot_card))*100
            
            st.markdown(f"Over 3.5: :{get_color(p_tot_35)}[**{p_tot_35:.0f}%**]")
            st.markdown(f"Over 4.5: :{get_color(p_tot_45)}[**{p_tot_45:.0f}%**]")
            
            st.markdown("---")
            st.markdown("**Individuais:**")
            
            # Individuais Casa
            pk_h_15 = (1-poisson.cdf(1, lk_h))*100
            pk_h_25 = (1-poisson.cdf(2, lk_h))*100
            st.markdown(f"🏠 **{h_team}:**")
            st.markdown(f"+1.5: :{get_color(pk_h_15)}[**{pk_h_15:.0f}%**] | +2.5: :{get_color(pk_h_25)}[**{pk_h_25:.0f}%**]")
            
            # Individuais Fora
            pk_a_15 = (1-poisson.cdf(1, lk_a))*100
            pk_a_25 = (1-poisson.cdf(2, lk_a))*100
            st.markdown(f"✈️ **{a_team}:**")
            st.markdown(f"+1.5: :{get_color(pk_a_15)}[**{pk_a_15:.0f}%**] | +2.5: :{get_color(pk_a_25)}[**{pk_a_25:.0f}%**]")

        else:
            st.warning("Times não encontrados na base.")

elif not league:
    st.info("👈 Selecione a liga para começar.")
