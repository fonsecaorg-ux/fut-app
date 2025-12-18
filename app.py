"""
╔═══════════════════════════════════════════════════════════════════════════╗
║                    FUTPREVISÃO V14.0 - CAUSALITY ENGINE                   ║
║                          Sistema de Análise de Apostas                     ║
║                                                                            ║
║  Desenvolvido por: Diego                                                   ║
║  Versão: V14.0 (Stable)                                                   ║
║  Data: Dezembro 2025                                                      ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import json
import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from difflib import get_close_matches
from datetime import datetime, timedelta

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURAÇÕES GLOBAIS
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="FutPrevisão V14.0",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constantes V14.0
THRESHOLDS = {
    'fouls_violent': 12.5,
    'shots_pressure_high': 6.0,
    'shots_pressure_medium': 4.5,
    'red_rate_strict_high': 0.12,
    'red_rate_strict_medium': 0.08,
    'prob_elite': 75,
    'prob_elite_cards': 70,
    'prob_red_high': 12,
    'prob_red_medium': 8
}

MULTIPLIERS = {
    'home_corners': 1.15,
    'away_corners': 0.90,
    'violence_safe': 0.85,
    'violence_active': 1.0,
    'shots_boost_high': 1.20,
    'shots_boost_medium': 1.10,
    'shots_boost_low': 1.0,
    'ref_strict_high': 1.15,
    'ref_strict_medium': 1.08,
    'ref_strict_normal': 1.0
}

DEFAULTS = {
    'shots_on_target': 4.5,
    'red_cards_avg': 0.08,
    'red_rate_referee': 0.08
}

NAME_MAPPING = {
    'Man United': 'Manchester Utd', 'Manchester United': 'Manchester Utd',
    'Man City': 'Manchester City', 'Spurs': 'Tottenham',
    'Newcastle': 'Newcastle Utd', 'Wolves': 'Wolverhampton',
    'Brighton': 'Brighton & Hove Albion', 'Nott\'m Forest': 'Nottingham Forest',
    'West Ham': 'West Ham Utd', 'Leicester': 'Leicester City',
}

LIGAS_DISPONIVEIS = [
    "Premier League", "La Liga", "Serie A", "Bundesliga", "Ligue 1",
    "Championship", "Bundesliga 2", "Pro League", "Süper Lig", "Scottish Premiership"
]

# ═══════════════════════════════════════════════════════════════════════════
# FUNÇÕES DE CARREGAMENTO
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def load_league_csv(league: str) -> pd.DataFrame:
    mapping = {
        "Premier League": "Premier_League_25_26.csv",
        "La Liga": "La_Liga_25_26.csv",
        "Serie A": "Serie_A_25_26.csv",
        "Bundesliga": "Bundesliga_25_26.csv",
        "Ligue 1": "Ligue_1_25_26.csv",
        "Championship": "Championship_Inglaterra_25_26.csv",
        "Bundesliga 2": "Bundesliga_2.csv",
        "Pro League": "Pro_League_Belgica_25_26.csv",
        "Süper Lig": "Super_Lig_Turquia_25_26.csv",
        "Scottish Premiership": "Premiership_Escocia_25_26.csv"
    }
    
    filename = mapping.get(league)
    if not filename: return pd.DataFrame()
    
    try:
        df = pd.read_csv(filename, encoding='utf-8-sig')
        # Normalização de nomes de colunas caso venha diferente
        return df
    except:
        try:
            df = pd.read_csv(filename, encoding='latin1')
            return df
        except:
            return pd.DataFrame()

@st.cache_data(ttl=3600)
def learn_stats_v14() -> Dict[str, Dict[str, Any]]:
    stats_db = {}
    for league in LIGAS_DISPONIVEIS:
        df = load_league_csv(league)
        if df.empty: continue
        
        # Garante que as colunas existam ou usa default
        cols = df.columns
        has_hst = 'HST' in cols
        has_hr = 'HR' in cols
        
        # Agregação Home
        home_stats = df.groupby('HomeTeam').agg({
            'HC': 'mean', 'HY': 'mean', 'HF': 'mean',
            'FTHG': 'mean', 'FTAG': 'mean',
            'HST': 'mean' if has_hst else lambda x: DEFAULTS['shots_on_target'],
            'HR': 'mean' if has_hr else lambda x: DEFAULTS['red_cards_avg']
        }).rename(columns={'HC':'corners','HY':'cards','HF':'fouls','FTHG':'goals_f','FTAG':'goals_a','HST':'shots_on_target','HR':'red_cards_avg'})
        
        # Agregação Away
        away_stats = df.groupby('AwayTeam').agg({
            'AC': 'mean', 'AY': 'mean', 'AF': 'mean',
            'FTAG': 'mean', 'FTHG': 'mean',
            'AST': 'mean' if 'AST' in cols else lambda x: DEFAULTS['shots_on_target'],
            'AR': 'mean' if 'AR' in cols else lambda x: DEFAULTS['red_cards_avg']
        }).rename(columns={'AC':'corners','AY':'cards','AF':'fouls','FTAG':'goals_f','FTHG':'goals_a','AST':'shots_on_target','AR':'red_cards_avg'})
        
        all_teams = set(home_stats.index) | set(away_stats.index)
        
        for team in all_teams:
            h = home_stats.loc[team] if team in home_stats.index else None
            a = away_stats.loc[team] if team in away_stats.index else None
            
            if h is not None and a is not None:
                stats_db[team] = {
                    'corners': (h['corners']*0.6 + a['corners']*0.4),
                    'cards': (h['cards']*0.6 + a['cards']*0.4),
                    'fouls': (h['fouls']*0.6 + a['fouls']*0.4),
                    'goals_f': (h['goals_f']*0.6 + a['goals_f']*0.4),
                    'goals_a': (h['goals_a']*0.6 + a['goals_a']*0.4),
                    'shots_on_target': (h['shots_on_target']*0.6 + a['shots_on_target']*0.4),
                    'red_cards_avg': (h['red_cards_avg']*0.6 + a['red_cards_avg']*0.4),
                    'league': league
                }
    return stats_db

@st.cache_data(ttl=3600)
def load_referees_v14() -> Dict[str, Dict[str, float]]:
    refs_db = {}
    # Carrega base unificada
    try:
        df = pd.read_csv('arbitros_5_ligas_2025_2026.csv')
        for _, row in df.iterrows():
            nome = str(row['Arbitro']).strip()
            media = float(row['Media_Cartoes_Por_Jogo'])
            jogos = float(row['Jogos_Apitados'])
            vermelhos = float(row.get('Cartoes_Vermelhos', 0))
            
            red_rate = (vermelhos / jogos) if jogos > 0 else DEFAULTS['red_rate_referee']
            refs_db[nome] = {'factor': media/4.0, 'red_rate': red_rate}
    except: pass
    
    # Carrega fallback arbitros.csv
    try:
        df2 = pd.read_csv('arbitros.csv')
        for _, row in df2.iterrows():
            nome = str(row['Nome']).strip()
            if nome not in refs_db:
                refs_db[nome] = {'factor': float(row['Fator']), 'red_rate': DEFAULTS['red_rate_referee']}
    except: pass
    
    return refs_db

@st.cache_data(ttl=600)
def load_scheduled_games() -> pd.DataFrame:
    """Carrega calendário e normaliza colunas para evitar KeyError."""
    try:
        df = pd.read_csv('calendario_ligas.csv', encoding='utf-8-sig')
    except:
        try: df = pd.read_csv('calendario_ligas.csv', encoding='latin1')
        except: return pd.DataFrame()
        
    # Normalização de Colunas (FIX DO BUG)
    df.columns = [c.strip() for c in df.columns]
    
    # Renomeia Mandante->Time_Casa se necessário
    rename_map = {}
    if 'Mandante' in df.columns: rename_map['Mandante'] = 'Time_Casa'
    if 'Visitante' in df.columns: rename_map['Visitante'] = 'Time_Visitante'
    
    if rename_map:
        df = df.rename(columns=rename_map)
    
    # Garante colunas mínimas
    req = ['Data', 'Liga', 'Time_Casa', 'Time_Visitante', 'Hora']
    if not set(req).issubset(df.columns):
        return pd.DataFrame()

    try:
        df['Data'] = pd.to_datetime(df['Data'], format='%d/%m/%Y', errors='coerce')
        hoje = datetime.now()
        df = df[df['Data'] >= hoje].sort_values('Data')
    except: pass
    
    return df

# ═══════════════════════════════════════════════════════════════════════════
# CÁLCULO E LÓGICA
# ═══════════════════════════════════════════════════════════════════════════

def normalize_team_name(team: str, stats_db: Dict) -> Optional[str]:
    if team in NAME_MAPPING: team = NAME_MAPPING[team]
    if team in stats_db: return team
    matches = get_close_matches(team, stats_db.keys(), n=1, cutoff=0.6)
    return matches[0] if matches else None

def get_referee_data(ref_name: Optional[str], refs_db: Dict) -> Dict:
    if not ref_name or ref_name not in refs_db:
        return {'factor': 1.0, 'red_rate': DEFAULTS['red_rate_referee']}
    return refs_db[ref_name]

def calcular_jogo_v14(home_team: str, away_team: str, stats_db: Dict, referee: Optional[str], refs_db: Dict) -> Dict:
    h_norm = normalize_team_name(home_team, stats_db)
    a_norm = normalize_team_name(away_team, stats_db)
    
    if not h_norm or not a_norm:
        return {'error': f"Times não encontrados: {home_team} ou {away_team}"}
        
    s_h = stats_db[h_norm]
    s_a = stats_db[a_norm]
    ref_data = get_referee_data(referee, refs_db)
    
    # 1. ESCANTEIOS (Baseado em CHUTES)
    shots_h = s_h.get('shots_on_target', 4.5)
    shots_a = s_a.get('shots_on_target', 4.5)
    
    if shots_h > THRESHOLDS['shots_pressure_high']: p_h, l_h = 1.20, "ALTO 🔥"
    elif shots_h > THRESHOLDS['shots_pressure_medium']: p_h, l_h = 1.10, "MÉDIO ✅"
    else: p_h, l_h = 1.0, "BAIXO ⚪"
    
    if shots_a > THRESHOLDS['shots_pressure_high']: p_a, l_a = 1.20, "ALTO 🔥"
    elif shots_a > THRESHOLDS['shots_pressure_medium']: p_a, l_a = 1.10, "MÉDIO ✅"
    else: p_a, l_a = 1.0, "BAIXO ⚪"
    
    corn_h = s_h['corners'] * 1.15 * p_h
    corn_a = s_a['corners'] * 0.90 * p_a
    
    # 2. CARTÕES (Baseado em RIGIDEZ e FALTAS)
    viol_h = 1.0 if s_h['fouls'] > THRESHOLDS['fouls_violent'] else 0.85
    viol_a = 1.0 if s_a['fouls'] > THRESHOLDS['fouls_violent'] else 0.85
    
    rr = ref_data['red_rate']
    if rr > THRESHOLDS['red_rate_strict_high']: strict, s_lbl = 1.15, "MUITO RIGOROSO 🔴"
    elif rr > THRESHOLDS['red_rate_strict_medium']: strict, s_lbl = 1.08, "RIGOROSO 🟠"
    else: strict, s_lbl = 1.0, "NORMAL 🟢"
    
    card_h = s_h['cards'] * viol_h * ref_data['factor'] * strict
    card_a = s_a['cards'] * viol_a * ref_data['factor'] * strict
    
    # 3. PROB VERMELHO
    reds_avg = (s_h.get('red_cards_avg', 0.08) + s_a.get('red_cards_avg', 0.08)) / 2
    prob_red = reds_avg * rr * 100
    
    if prob_red > 12: pr_lbl = "ALTA 🔴"
    elif prob_red > 8: pr_lbl = "MÉDIA 🟠"
    else: pr_lbl = "BAIXA 🟡"
    
    return {
        'home_team': h_norm, 'away_team': a_norm, 'referee': referee,
        'corners': {'home': corn_h, 'away': corn_a, 'total': corn_h+corn_a},
        'cards': {'home': card_h, 'away': card_a, 'total': card_h+card_a},
        'goals': {'home': (s_h['goals_f']*s_a['goals_a'])/1.3, 'away': (s_a['goals_f']*s_h['goals_a'])/1.3},
        'metadata': {
            'shots_home': shots_h, 'pressure_home': p_h, 'pressure_label_home': l_h,
            'shots_away': shots_a, 'pressure_away': p_a, 'pressure_label_away': l_a,
            'ref_factor': ref_data['factor'], 'ref_red_rate': rr, 
            'strictness': strict, 'strictness_label': s_lbl,
            'prob_red_card': prob_red, 'prob_red_label': pr_lbl,
            'reds_home_avg': s_h.get('red_cards_avg', 0), 'reds_away_avg': s_a.get('red_cards_avg', 0),
            'violence_label_home': "VIOLENTO 🔴" if viol_h==1.0 else "DISCIPLINADO ✅",
            'violence_label_away': "VIOLENTO 🔴" if viol_a==1.0 else "DISCIPLINADO ✅"
        }
    }

def calcular_probabilidades(pred: Dict) -> Dict:
    def poisson_cdf(k, lam): return sum((lam**i * math.exp(-lam)) / math.factorial(i) for i in range(k + 1))
    
    ct = pred['corners']['total']
    kt = pred['cards']['total']
    
    return {
        'corners': {f'over_{i}_5': (1-poisson_cdf(i, ct))*100 for i in range(8, 13)},
        'cards': {f'over_{i}_5': (1-poisson_cdf(i, kt))*100 for i in range(3, 6)}
    }

# ═══════════════════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════════════════

def render_prediction_card(pred: Dict, probs: Dict):
    m = pred['metadata']
    st.markdown("---")
    
    # Placar e Times
    c1, c2, c3 = st.columns([2,1,2])
    c1.markdown(f"### 🏠 {pred['home_team']}")
    c2.markdown("<h3 style='text-align: center'>VS</h3>", unsafe_allow_html=True)
    c3.markdown(f"<h3 style='text-align: right'>✈️ {pred['away_team']}</h3>", unsafe_allow_html=True)
    if pred.get('referee'): st.caption(f"🧑‍⚖️ Árbitro: {pred['referee']}")
    
    # Métricas Causais (V14)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("xG Mandante", f"{pred['goals']['home']:.2f}")
    k2.metric("xG Visitante", f"{pred['goals']['away']:.2f}")
    k3.metric("Chutes (Casa)", f"{m['shots_home']:.1f}", m['pressure_label_home'])
    k4.metric("Prob. Vermelho", f"{m['prob_red_card']:.1f}%", m['prob_red_label'])
    
    st.info(f"""
    **Indicadores Causais:**
    * **Rigidez do Juiz:** {m['strictness']}x ({m['strictness_label']}) - Taxa de Vermelhos: {m['ref_red_rate']:.3f}
    * **Violência:** 🏠 {m['violence_label_home']} vs ✈️ {m['violence_label_away']}
    """)
    
    # Previsões
    wc1, wc2 = st.columns(2)
    with wc1:
        st.subheader("🏁 Escanteios")
        st.write(f"Esperado: **{pred['corners']['total']:.2f}**")
        for k, v in probs['corners'].items():
            if v > THRESHOLDS['prob_elite']: st.success(f"✅ {k.replace('_', ' ')}: **{v:.1f}%** (ELITE)")
            elif v > 50: st.write(f"{k.replace('_', ' ')}: {v:.1f}%")
            
    with wc2:
        st.subheader("🟨 Cartões")
        st.write(f"Esperado: **{pred['cards']['total']:.2f}**")
        for k, v in probs['cards'].items():
            if v > THRESHOLDS['prob_elite_cards']: st.success(f"✅ {k.replace('_', ' ')}: **{v:.1f}%** (ELITE)")
            elif v > 50: st.write(f"{k.replace('_', ' ')}: {v:.1f}%")

def main():
    st.title("⚽ FutPrevisão V14.0 (Causality Engine)")
    st.caption("Foco em CAUSAS: Chutes no Gol, Faltas e Rigidez do Árbitro")
    
    with st.spinner("Carregando Cérebro V14..."):
        stats = learn_stats_v14()
        refs = load_referees_v14()
    
    tab1, tab2 = st.tabs(["📅 Jogos do Dia", "🧪 Simulação Manual"])
    
    with tab1:
        df = load_scheduled_games()
        if df.empty:
            st.warning("Sem jogos futuros no calendário ou erro de leitura.")
        else:
            ligas = ["Todas"] + sorted(df['Liga'].unique().tolist())
            filtro_liga = st.selectbox("Filtrar Liga:", ligas)
            if filtro_liga != "Todas": df = df[df['Liga'] == filtro_liga]
            
            for i, row in df.iterrows():
                with st.expander(f"⚽ {row['Time_Casa']} x {row['Time_Visitante']} ({row['Hora']})"):
                    if st.button("Analisar", key=f"b_{i}"):
                        res = calcular_jogo_v14(row['Time_Casa'], row['Time_Visitante'], stats, None, refs)
                        if 'error' in res: st.error(res['error'])
                        else: render_prediction_card(res, calcular_probabilidades(res))

    with tab2:
        c1, c2, c3 = st.columns(3)
        h = c1.text_input("Mandante (Ex: Liverpool)")
        a = c2.text_input("Visitante (Ex: Man City)")
        r = c3.text_input("Árbitro (Opcional)")
        if st.button("Simular"):
            res = calcular_jogo_v14(h, a, stats, r, refs)
            if 'error' in res: st.error(res['error'])
            else: render_prediction_card(res, calcular_probabilidades(res))

if __name__ == "__main__":
    main()
