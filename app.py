"""
╔═══════════════════════════════════════════════════════════════════════════╗
║               FUTPREVISÃO V14.3 - UI EXPERIENCE & BET BUILDER             ║
║                          Sistema de Análise de Apostas                     ║
║                                                                            ║
║  Versão: V14.3 (Dropdowns + Linhas Individuais)                           ║
║  Data: Dezembro 2025                                                      ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import math
import numpy as np
from typing import Dict, Optional, Any
from difflib import get_close_matches
from datetime import datetime
import os

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURAÇÕES GLOBAIS
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="FutPrevisão V14.3",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constantes V14
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

DEFAULTS = {
    'shots_on_target': 4.5,
    'red_cards_avg': 0.08,
    'red_rate_referee': 0.08
}

# Mapeamento de nomes para normalizar times
NAME_MAPPING = {
    'Man United': 'Man Utd', 'Manchester United': 'Man Utd',
    'Man City': 'Man City', 'Manchester City': 'Man City',
    'Spurs': 'Tottenham', 'Newcastle': 'Newcastle',
    'Wolves': 'Wolves', 'Brighton': 'Brighton',
    'Nott\'m Forest': 'Nottm Forest', 'Nottingham Forest': 'Nottm Forest',
    'West Ham': 'West Ham', 'Leicester': 'Leicester',
}

LIGAS_ALVO = [
    "Premier League", "La Liga", "Serie A", "Bundesliga", "Ligue 1",
    "Championship", "Bundesliga 2", "Pro League", "Süper Lig", "Scottish Premiership"
]

DEBUG_LOGS = []

def log_status(msg: str, status: str = "info"):
    icon = "✅" if status == "success" else "❌" if status == "error" else "ℹ️"
    DEBUG_LOGS.append(f"{icon} {msg}")

# ═══════════════════════════════════════════════════════════════════════════
# CARREGAMENTO INTELIGENTE (MANTIDO DA V14.2)
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def find_and_load_csv(league_name: str) -> pd.DataFrame:
    attempts = [
        f"{league_name} 25.26.csv",
        f"{league_name.replace(' ', '_')}_25_26.csv",
        f"{league_name}.csv",
        f"{league_name} 2025.csv"
    ]
    if "Süper Lig" in league_name: attempts.extend(["Super Lig Turquia 25.26.csv", "Super_Lig_Turquia_25_26.csv"])
    if "Pro League" in league_name: attempts.append("Pro League Belgica 25.26.csv")
    if "Premiership" in league_name: attempts.append("Premiership Escocia 25.26.csv")
    if "Championship" in league_name: attempts.append("Championship Inglaterra 25.26.csv")

    for filename in attempts:
        if os.path.exists(filename):
            try:
                try: df = pd.read_csv(filename, encoding='utf-8')
                except: df = pd.read_csv(filename, encoding='latin1')
                if not df.empty:
                    log_status(f"Carregado: {filename}", "success")
                    return df
            except Exception as e:
                log_status(f"Erro em {filename}: {e}", "error")
    
    log_status(f"Arquivo não encontrado: {league_name}", "error")
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def learn_stats_v14() -> Dict[str, Dict[str, Any]]:
    stats_db = {}
    total_loaded = 0
    
    for league in LIGAS_ALVO:
        df = find_and_load_csv(league)
        if df.empty: continue
        
        df.columns = [c.strip() for c in df.columns]
        cols_needed = ['HomeTeam', 'AwayTeam', 'HC', 'AC', 'HY', 'AY', 'HF', 'AF', 'FTHG', 'FTAG', 'HST', 'AST', 'HR', 'AR']
        for c in cols_needed:
            if c not in df.columns: df[c] = np.nan
        
        try:
            h_stats = df.groupby('HomeTeam').agg({
                'HC': 'mean', 'HY': 'mean', 'HF': 'mean',
                'FTHG': 'mean', 'FTAG': 'mean', 'HST': 'mean', 'HR': 'mean'
            }).fillna(value={'HST': DEFAULTS['shots_on_target'], 'HR': DEFAULTS['red_cards_avg']})
            
            a_stats = df.groupby('AwayTeam').agg({
                'AC': 'mean', 'AY': 'mean', 'AF': 'mean',
                'FTAG': 'mean', 'FTHG': 'mean', 'AST': 'mean', 'AR': 'mean'
            }).fillna(value={'AST': DEFAULTS['shots_on_target'], 'AR': DEFAULTS['red_cards_avg']})
            
            all_teams = set(h_stats.index) | set(a_stats.index)
            for team in all_teams:
                h = h_stats.loc[team] if team in h_stats.index else pd.Series(0, index=h_stats.columns)
                a = a_stats.loc[team] if team in a_stats.index else pd.Series(0, index=a_stats.columns)
                
                def combine(val_h, val_a, default=0):
                    if val_h == 0 and val_a == 0: return default
                    if val_h == 0: return val_a
                    if val_a == 0: return val_h
                    return (val_h * 0.6) + (val_a * 0.4)

                stats_db[team] = {
                    'corners': combine(h.get('HC',0), a.get('AC',0), 5.0),
                    'cards': combine(h.get('HY',0), a.get('AY',0), 2.0),
                    'fouls': combine(h.get('HF',0), a.get('AF',0), 11.0),
                    'goals_f': combine(h.get('FTHG',0), a.get('FTAG',0), 1.2),
                    'goals_a': combine(h.get('FTAG',0), a.get('FTHG',0), 1.2),
                    'shots_on_target': combine(h.get('HST',0), a.get('AST',0), 4.5),
                    'red_cards_avg': combine(h.get('HR',0), a.get('AR',0), 0.08),
                    'league': league
                }
            total_loaded += 1
        except Exception as e:
            log_status(f"Erro processando {league}: {e}", "error")
            
    if total_loaded == 0: log_status("CRÍTICO: Nenhuma liga carregada!", "error")
    return stats_db

@st.cache_data(ttl=3600)
def load_referees_v14() -> Dict[str, Dict[str, float]]:
    refs_db = {}
    f5 = "arbitros_5_ligas_2025_2026.csv"
    if os.path.exists(f5):
        try:
            df = pd.read_csv(f5)
            for _, row in df.iterrows():
                nome = str(row['Arbitro']).strip()
                media = float(row['Media_Cartoes_Por_Jogo'])
                jogos = float(row['Jogos_Apitados'])
                vermelhos = float(row.get('Cartoes_Vermelhos', 0))
                red_rate = (vermelhos / jogos) if jogos > 0 else DEFAULTS['red_rate_referee']
                refs_db[nome] = {'factor': media/4.0, 'red_rate': red_rate}
            log_status(f"Árbitros: {f5}", "success")
        except: pass
            
    f_gen = "arbitros.csv"
    if os.path.exists(f_gen):
        try:
            df = pd.read_csv(f_gen)
            for _, row in df.iterrows():
                nome = str(row['Nome']).strip()
                if nome not in refs_db:
                    refs_db[nome] = {'factor': float(row['Fator']), 'red_rate': DEFAULTS['red_rate_referee']}
        except: pass
    return refs_db

@st.cache_data(ttl=600)
def load_calendar_safe() -> pd.DataFrame:
    fname = "calendario_ligas.csv"
    if not os.path.exists(fname): return pd.DataFrame()
    try:
        try: df = pd.read_csv(fname, encoding='utf-8')
        except: df = pd.read_csv(fname, encoding='latin1')
        df.columns = [c.strip() for c in df.columns]
        rename_map = {}
        if 'Mandante' in df.columns: rename_map['Mandante'] = 'Time_Casa'
        if 'Visitante' in df.columns: rename_map['Visitante'] = 'Time_Visitante'
        if rename_map: df = df.rename(columns=rename_map)
        
        req = ['Data', 'Liga', 'Time_Casa', 'Time_Visitante', 'Hora']
        if not set(req).issubset(df.columns): return pd.DataFrame()
        
        df['DtObj'] = pd.to_datetime(df['Data'], format='%d/%m/%Y', errors='coerce')
        df = df.dropna(subset=['DtObj'])
        df = df.sort_values(by=['DtObj', 'Hora'])
        return df
    except: return pd.DataFrame()

# ═══════════════════════════════════════════════════════════════════════════
# CÁLCULO E MATEMÁTICA
# ═══════════════════════════════════════════════════════════════════════════

def normalize_name(name: str, db_keys: list) -> Optional[str]:
    if name in NAME_MAPPING: name = NAME_MAPPING[name]
    if name in db_keys: return name
    matches = get_close_matches(name, db_keys, n=1, cutoff=0.6)
    return matches[0] if matches else None

def calcular_jogo_v14(home: str, away: str, stats: Dict, ref: Optional[str], refs_db: Dict) -> Dict:
    h_norm = normalize_name(home, list(stats.keys()))
    a_norm = normalize_name(away, list(stats.keys()))
    
    if not h_norm or not a_norm:
        return {'error': "Times não encontrados."}
    
    s_h = stats[h_norm]
    s_a = stats[a_norm]
    
    r_data = refs_db.get(ref, {'factor': 1.0, 'red_rate': DEFAULTS['red_rate_referee']}) if ref else {'factor': 1.0, 'red_rate': DEFAULTS['red_rate_referee']}
        
    # 1. Chutes -> Escanteios
    shots_h = s_h['shots_on_target']
    shots_a = s_a['shots_on_target']
    
    p_h = 1.20 if shots_h > THRESHOLDS['shots_pressure_high'] else 1.10 if shots_h > THRESHOLDS['shots_pressure_medium'] else 1.0
    l_h = "ALTO 🔥" if p_h == 1.20 else "MÉDIO ✅" if p_h == 1.10 else "BAIXO ⚪"
    
    p_a = 1.20 if shots_a > THRESHOLDS['shots_pressure_high'] else 1.10 if shots_a > THRESHOLDS['shots_pressure_medium'] else 1.0
    l_a = "ALTO 🔥" if p_a == 1.20 else "MÉDIO ✅" if p_a == 1.10 else "BAIXO ⚪"
    
    corn_h = s_h['corners'] * 1.15 * p_h
    corn_a = s_a['corners'] * 0.90 * p_a
    
    # 2. Rigidez -> Cartões
    rr = r_data['red_rate']
    strict = 1.15 if rr > THRESHOLDS['red_rate_strict_high'] else 1.08 if rr > THRESHOLDS['red_rate_strict_medium'] else 1.0
    s_lbl = "MUITO RIGOROSO 🔴" if strict == 1.15 else "RIGOROSO 🟠" if strict == 1.08 else "NORMAL 🟢"
    
    viol_h = 1.0 if s_h['fouls'] > THRESHOLDS['fouls_violent'] else 0.85
    viol_a = 1.0 if s_a['fouls'] > THRESHOLDS['fouls_violent'] else 0.85
    
    card_h = s_h['cards'] * viol_h * r_data['factor'] * strict
    card_a = s_a['cards'] * viol_a * r_data['factor'] * strict
    
    # 3. Vermelhos
    reds_avg = (s_h['red_cards_avg'] + s_a['red_cards_avg']) / 2
    prob_red = reds_avg * rr * 100
    pr_lbl = "ALTA 🔴" if prob_red > 12 else "MÉDIA 🟠" if prob_red > 8 else "BAIXA 🟡"

    return {
        'home': h_norm, 'away': a_norm, 'referee': ref,
        'corners': {'total': corn_h + corn_a, 'h': corn_h, 'a': corn_a},
        'cards': {'total': card_h + card_a, 'h': card_h, 'a': card_a},
        'goals': {'h': (s_h['goals_f'] * s_a['goals_a'])/1.3, 'a': (s_a['goals_f'] * s_h['goals_a'])/1.3},
        'meta': {
            'shots_h': shots_h, 'shots_a': shots_a, 
            'p_label_h': l_h, 'p_label_a': l_a,
            'strict_val': strict, 'strict_lbl': s_lbl, 'red_rate': rr,
            'prob_red': prob_red, 'prob_red_lbl': pr_lbl,
            'viol_h_lbl': "VIOLENTO 🔴" if viol_h == 1.0 else "DISCIPLINADO ✅",
            'viol_a_lbl': "VIOLENTO 🔴" if viol_a == 1.0 else "DISCIPLINADO ✅"
        }
    }

def get_detailed_probs(pred: Dict) -> Dict:
    def p(k, l): return sum((l**i * math.exp(-l)) / math.factorial(i) for i in range(k + 1))
    
    # Médias Esperadas
    cH, cA = pred['corners']['h'], pred['corners']['a']
    kH, kA = pred['cards']['h'], pred['cards']['a']
    
    return {
        'corners': {
            'total': {f"Over {i}.5": (1-p(i, cH+cA))*100 for i in range(8, 13)},
            'home': {
                'Over 3.5': (1-p(3, cH))*100,
                'Over 4.5': (1-p(4, cH))*100
            },
            'away': {
                'Over 3.5': (1-p(3, cA))*100,
                'Over 4.5': (1-p(4, cA))*100
            }
        },
        'cards': {
            'total': {f"Over {i}.5": (1-p(i, kH+kA))*100 for i in range(3, 6)},
            'home': {
                'Over 1.5': (1-p(1, kH))*100,
                'Over 2.5': (1-p(2, kH))*100
            },
            'away': {
                'Over 1.5': (1-p(1, kA))*100,
                'Over 2.5': (1-p(2, kA))*100
            }
        }
    }

# ═══════════════════════════════════════════════════════════════════════════
# UI REFORMULADA (V14.3)
# ═══════════════════════════════════════════════════════════════════════════

def format_x_of_10(prob):
    """Retorna string formatada (X de 10)."""
    score = int(round(prob / 10))
    return f"({score} de 10)"

def render_result_v14_3(res):
    m = res['meta']
    probs = get_detailed_probs(res)
    
    st.markdown("---")
    
    # Header Times
    c1, c2, c3 = st.columns([2,1,2])
    c1.markdown(f"### 🏠 {res['home']}")
    c2.markdown("<h3 style='text-align: center'>VS</h3>", unsafe_allow_html=True)
    c3.markdown(f"<h3 style='text-align: right'>✈️ {res['away']}</h3>", unsafe_allow_html=True)
    
    # KPI Grid
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("xG Casa", f"{res['goals']['h']:.2f}")
    k2.metric("xG Fora", f"{res['goals']['a']:.2f}")
    k3.metric("Chutes Casa", f"{m['shots_h']:.1f}", m['p_label_h'])
    k4.metric("Risco Vermelho", f"{m['prob_red']:.1f}%", m['prob_red_lbl'])
    
    st.caption(f"👮 Juiz: {res['referee'] if res['referee'] else 'Neutro'} | Rigidez: {m['strict_lbl']} | Taxa Vermelhos: {m['red_rate']:.2f}")

    st.markdown("---")

    # SEÇÃO 1: ESCANTEIOS (Com Linhas Individuais)
    st.subheader(f"🏁 Escanteios (Total Esp: {res['corners']['total']:.2f})")
    ec1, ec2, ec3 = st.columns(3)
    
    # Total
    with ec1:
        st.markdown("**Geral**")
        for k, v in probs['corners']['total'].items():
            if v > 65: st.write(f"{k}: **{v:.0f}%** {format_x_of_10(v)}")
    
    # Mandante Individual
    with ec2:
        st.markdown(f"**🏠 {res['home']}** (Esp: {res['corners']['h']:.1f})")
        p35 = probs['corners']['home']['Over 3.5']
        p45 = probs['corners']['home']['Over 4.5']
        
        c35 = "green" if p35 >= 70 else "black"
        c45 = "green" if p45 >= 60 else "black"
        
        st.markdown(f"Over 3.5: :{c35}[**{p35:.0f}%**] {format_x_of_10(p35)}")
        st.markdown(f"Over 4.5: :{c45}[**{p45:.0f}%**] {format_x_of_10(p45)}")

    # Visitante Individual
    with ec3:
        st.markdown(f"**✈️ {res['away']}** (Esp: {res['corners']['a']:.1f})")
        p35 = probs['corners']['away']['Over 3.5']
        p45 = probs['corners']['away']['Over 4.5']
        
        c35 = "green" if p35 >= 70 else "black"
        c45 = "green" if p45 >= 60 else "black"
        
        st.markdown(f"Over 3.5: :{c35}[**{p35:.0f}%**] {format_x_of_10(p35)}")
        st.markdown(f"Over 4.5: :{c45}[**{p45:.0f}%**] {format_x_of_10(p45)}")
        
    st.markdown("---")

    # SEÇÃO 2: CARTÕES (Com Linhas Individuais)
    st.subheader(f"🟨 Cartões (Total Esp: {res['cards']['total']:.2f})")
    kc1, kc2, kc3 = st.columns(3)
    
    # Total
    with kc1:
        st.markdown("**Geral**")
        for k, v in probs['cards']['total'].items():
            if v > 60: st.write(f"{k}: **{v:.0f}%** {format_x_of_10(v)}")

    # Mandante Individual
    with kc2:
        st.markdown(f"**🏠 {res['home']}** (Esp: {res['cards']['h']:.1f})")
        p15 = probs['cards']['home']['Over 1.5']
        p25 = probs['cards']['home']['Over 2.5']
        
        c15 = "green" if p15 >= 75 else "black"
        c25 = "green" if p25 >= 50 else "black"
        
        st.markdown(f"Over 1.5: :{c15}[**{p15:.0f}%**] {format_x_of_10(p15)}")
        st.markdown(f"Over 2.5: :{c25}[**{p25:.0f}%**] {format_x_of_10(p25)}")

    # Visitante Individual
    with kc3:
        st.markdown(f"**✈️ {res['away']}** (Esp: {res['cards']['a']:.1f})")
        p15 = probs['cards']['away']['Over 1.5']
        p25 = probs['cards']['away']['Over 2.5']
        
        c15 = "green" if p15 >= 75 else "black"
        c25 = "green" if p25 >= 50 else "black"
        
        st.markdown(f"Over 1.5: :{c15}[**{p15:.0f}%**] {format_x_of_10(p15)}")
        st.markdown(f"Over 2.5: :{c25}[**{p25:.0f}%**] {format_x_of_10(p25)}")

def main():
    st.title("⚽ FutPrevisão V14.3 (Bet Builder UI)")
    st.caption("Linhas Individuais + Dropdowns Inteligentes")
    
    with st.spinner("Inicializando motores..."):
        DEBUG_LOGS.clear()
        stats = learn_stats_v14()
        refs = load_referees_v14()
        calendar = load_calendar_safe()
    
    # Listas para Dropdowns (Ordenadas)
    lista_times = sorted(list(stats.keys()))
    lista_juizes = ["Neutro"] + sorted(list(refs.keys()))
    
    with st.sidebar:
        with st.expander("🛠️ Status do Sistema", expanded=not bool(stats)):
            st.write(f"Times: {len(stats)}")
            st.write(f"Árbitros: {len(refs)}")
            for log in DEBUG_LOGS: st.write(log)
    
    if not stats:
        st.error("🚨 ERRO: Nenhum dado carregado.")
        return

    tab1, tab2 = st.tabs(["📅 Calendário", "🧪 Simulação Manual (Bet Builder)"])
    
    with tab1:
        if calendar.empty:
            st.warning("Calendário vazio.")
        else:
            dates = calendar['DtObj'].dt.strftime('%d/%m/%Y').unique()
            sel_date = st.selectbox("Data:", dates)
            subset = calendar[calendar['DtObj'].dt.strftime('%d/%m/%Y') == sel_date]
            
            for i, row in subset.iterrows():
                with st.expander(f"⏰ {str(row['Hora'])[:5]} | {row['Liga']} | {row['Time_Casa']} x {row['Time_Visitante']}"):
                    if st.button("Analisar", key=f"btn_{i}"):
                        res = calcular_jogo_v14(row['Time_Casa'], row['Time_Visitante'], stats, None, refs)
                        if 'error' in res: st.error(res['error'])
                        else: render_result_v14_3(res)

    with tab2:
        st.subheader("Simulador Personalizado")
        c1, c2, c3 = st.columns(3)
        
        # DROPDOWNS INTELIGENTES
        idx_h = lista_times.index("Liverpool") if "Liverpool" in lista_times else 0
        idx_a = lista_times.index("Man City") if "Man City" in lista_times else 1
        
        h = c1.selectbox("Mandante", lista_times, index=idx_h)
        a = c2.selectbox("Visitante", lista_times, index=idx_a)
        r = c3.selectbox("Árbitro", lista_juizes, index=0)
        
        if st.button("Simular Jogo"):
            ref_val = None if r == "Neutro" else r
            res = calcular_jogo_v14(h, a, stats, ref_val, refs)
            if 'error' in res: st.error(res['error'])
            else: render_result_v14_3(res)

if __name__ == "__main__":
    main()
