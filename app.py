"""
╔═══════════════════════════════════════════════════════════════════════════╗
║               FUTPREVISÃO V14.5 - NATIVE HISTORY (ZERO TXT)               ║
║                          Sistema de Análise de Apostas                     ║
║                                                                            ║
║  Versão: V14.5 (Histórico calculado direto do CSV - Sem erros de dados)   ║
║  Data: Dezembro 2025                                                      ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import math
import numpy as np
import os
from typing import Dict, Optional, Any
from difflib import get_close_matches
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURAÇÕES GLOBAIS
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="FutPrevisão V14.5",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constantes
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

NAME_MAPPING = {
    'Man United': 'Man Utd', 'Manchester United': 'Man Utd',
    'Man City': 'Man City', 'Manchester City': 'Man City',
    'Spurs': 'Tottenham', 'Newcastle': 'Newcastle',
    'Wolves': 'Wolves', 'Brighton': 'Brighton',
    'Nott\'m Forest': 'Nottm Forest', 'Nottingham Forest': 'Nottm Forest',
    'West Ham': 'West Ham', 'Leicester': 'Leicester',
    'Athletic Club': 'Ath Bilbao', 'Atl. Madrid': 'Ath Madrid'
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
# CARREGAMENTO INTELIGENTE
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def find_and_load_csv(league_name: str) -> pd.DataFrame:
    attempts = [
        f"{league_name} 25.26.csv",
        f"{league_name.replace(' ', '_')}_25_26.csv",
        f"{league_name}.csv"
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
                    # Garante nomes limpos e normaliza colunas
                    df.columns = [c.strip() for c in df.columns]
                    # Adiciona coluna da liga para referência
                    df['_League_'] = league_name 
                    return df
            except: pass
    return pd.DataFrame()

# NOVO: Cache Global de DataFrames para Histórico
@st.cache_resource
def load_all_dataframes() -> Dict[str, pd.DataFrame]:
    """Carrega todos os CSVs em memória para consulta rápida de histórico."""
    dfs = {}
    for league in LIGAS_ALVO:
        df = find_and_load_csv(league)
        if not df.empty:
            dfs[league] = df
            log_status(f"DB Histórico: {league}", "success")
    return dfs

@st.cache_data(ttl=3600)
def learn_stats_v14() -> Dict[str, Dict[str, Any]]:
    stats_db = {}
    
    # Usa a função de find normal
    for league in LIGAS_ALVO:
        df = find_and_load_csv(league)
        if df.empty: continue
        
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
        except: pass
    return stats_db

@st.cache_data(ttl=3600)
def load_referees_v14() -> Dict[str, Dict[str, float]]:
    refs_db = {}
    if os.path.exists("arbitros_5_ligas_2025_2026.csv"):
        try:
            df = pd.read_csv("arbitros_5_ligas_2025_2026.csv")
            for _, row in df.iterrows():
                nome = str(row['Arbitro']).strip()
                media = float(row['Media_Cartoes_Por_Jogo'])
                jogos = float(row['Jogos_Apitados'])
                vermelhos = float(row.get('Cartoes_Vermelhos', 0))
                red_rate = (vermelhos / jogos) if jogos > 0 else DEFAULTS['red_rate_referee']
                refs_db[nome] = {'factor': media/4.0, 'red_rate': red_rate}
        except: pass
            
    if os.path.exists("arbitros.csv"):
        try:
            df = pd.read_csv("arbitros.csv")
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
# CÁLCULO DE HISTÓRICO NATIVO (NOVO!)
# ═══════════════════════════════════════════════════════════════════════════

def get_native_history(team_name: str, league: str, market: str, line: float, location: str, all_dfs: Dict) -> str:
    """
    Calcula histórico real lendo diretamente do DataFrame da liga.
    
    Args:
        team_name: Nome do time
        league: Nome da liga (para buscar o DF correto)
        market: 'corners' ou 'cards'
        line: Linha de corte (ex: 3.5)
        location: 'home' ou 'away'
    """
    if league not in all_dfs:
        return "N/A (Liga?)"
    
    df = all_dfs[league]
    
    # Filtra jogos do time na posição correta
    if location == 'home':
        matches = df[df['HomeTeam'] == team_name]
        col_code = 'HC' if market == 'corners' else 'HY'
    else:
        matches = df[df['AwayTeam'] == team_name]
        col_code = 'AC' if market == 'corners' else 'AY'
    
    if matches.empty:
        # Tenta fallback se o nome no CSV for ligeiramente diferente
        # (Mas o ideal é que normalize_name já tenha cuidado disso antes)
        return "0/0"

    # Pega os últimos 10 jogos (ou menos se não tiver 10)
    last_matches = matches.tail(10)
    total_games = len(last_matches)
    
    if total_games == 0:
        return "0/0"
        
    # Conta quantos bateram a linha
    hits = 0
    for val in last_matches[col_code]:
        try:
            if float(val) > line:
                hits += 1
        except: pass
        
    return f"{hits}/{total_games}"

# ═══════════════════════════════════════════════════════════════════════════
# CÁLCULO E MATEMÁTICA V14
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
        'league_h': s_h.get('league'), 'league_a': s_a.get('league'), # Passa a liga para buscar histórico
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
    cH, cA = pred['corners']['h'], pred['corners']['a']
    kH, kA = pred['cards']['h'], pred['cards']['a']
    
    return {
        'corners': {
            'total': {f"Over {i}.5": (1-p(i, cH+cA))*100 for i in range(8, 13)},
            'home': {'Over 3.5': (1-p(3, cH))*100, 'Over 4.5': (1-p(4, cH))*100},
            'away': {'Over 3.5': (1-p(3, cA))*100, 'Over 4.5': (1-p(4, cA))*100}
        },
        'cards': {
            'total': {f"Over {i}.5": (1-p(i, kH+kA))*100 for i in range(3, 6)},
            'home': {'Over 1.5': (1-p(1, kH))*100, 'Over 2.5': (1-p(2, kH))*100},
            'away': {'Over 1.5': (1-p(1, kA))*100, 'Over 2.5': (1-p(2, kA))*100}
        }
    }

# ═══════════════════════════════════════════════════════════════════════════
# UI V14.5
# ═══════════════════════════════════════════════════════════════════════════

def render_result_v14_5(res, all_dfs):
    m = res['meta']
    probs = get_detailed_probs(res)
    
    st.markdown("---")
    
    # Header
    c1, c2, c3 = st.columns([2,1,2])
    c1.markdown(f"### 🏠 {res['home']}")
    c2.markdown("<h3 style='text-align: center'>VS</h3>", unsafe_allow_html=True)
    c3.markdown(f"<h3 style='text-align: right'>✈️ {res['away']}</h3>", unsafe_allow_html=True)
    
    # Métricas
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("xG Casa", f"{res['goals']['h']:.2f}")
    k2.metric("xG Fora", f"{res['goals']['a']:.2f}")
    k3.metric("Chutes Casa", f"{m['shots_h']:.1f}", m['p_label_h'])
    k4.metric("Risco Vermelho", f"{m['prob_red']:.1f}%", m['prob_red_lbl'])
    
    st.caption(f"👮 Juiz: {res['referee'] if res['referee'] else 'Neutro'} | Rigidez: {m['strict_lbl']} | Taxa Vermelhos: {m['red_rate']:.2f}")

    st.markdown("---")

    # SEÇÃO 1: ESCANTEIOS
    st.subheader(f"🏁 Escanteios (Total Esp: {res['corners']['total']:.2f})")
    ec1, ec2, ec3 = st.columns(3)
    
    with ec1:
        st.markdown("**Geral**")
        for k, v in probs['corners']['total'].items():
            if v > 65: st.write(f"{k}: **{v:.0f}%**")

    # Mandante (Histórico REAL do CSV)
    with ec2:
        st.markdown(f"**🏠 {res['home']}** (Esp: {res['corners']['h']:.1f})")
        p35 = probs['corners']['home']['Over 3.5']
        p45 = probs['corners']['home']['Over 4.5']
        
        # Histórico Nativo
        h35 = get_native_history(res['home'], res['league_h'], 'corners', 3.5, 'home', all_dfs)
        h45 = get_native_history(res['home'], res['league_h'], 'corners', 4.5, 'home', all_dfs)
        
        c35 = "green" if p35 >= 70 else "red"
        c45 = "green" if p45 >= 60 else "red"
        
        st.markdown(f"Over 3.5: :{c35}[**{p35:.0f}%**] | Hist CSV: {h35}")
        st.markdown(f"Over 4.5: :{c45}[**{p45:.0f}%**] | Hist CSV: {h45}")

    # Visitante (Histórico REAL do CSV)
    with ec3:
        st.markdown(f"**✈️ {res['away']}** (Esp: {res['corners']['a']:.1f})")
        p35 = probs['corners']['away']['Over 3.5']
        p45 = probs['corners']['away']['Over 4.5']
        
        h35 = get_native_history(res['away'], res['league_a'], 'corners', 3.5, 'away', all_dfs)
        h45 = get_native_history(res['away'], res['league_a'], 'corners', 4.5, 'away', all_dfs)
        
        c35 = "green" if p35 >= 70 else "red"
        c45 = "green" if p45 >= 60 else "red"
        
        st.markdown(f"Over 3.5: :{c35}[**{p35:.0f}%**] | Hist CSV: {h35}")
        st.markdown(f"Over 4.5: :{c45}[**{p45:.0f}%**] | Hist CSV: {h45}")
        
    st.markdown("---")

    # SEÇÃO 2: CARTÕES
    st.subheader(f"🟨 Cartões (Total Esp: {res['cards']['total']:.2f})")
    kc1, kc2, kc3 = st.columns(3)
    
    with kc1:
        st.markdown("**Geral**")
        for k, v in probs['cards']['total'].items():
            if v > 60: st.write(f"{k}: **{v:.0f}%**")

    with kc2:
        st.markdown(f"**🏠 {res['home']}**")
        p15 = probs['cards']['home']['Over 1.5']
        
        h15 = get_native_history(res['home'], res['league_h'], 'cards', 1.5, 'home', all_dfs)
        c15 = "green" if p15 >= 75 else "green"
        
        st.markdown(f"Over 1.5: :{c15}[**{p15:.0f}%**] | Hist CSV: {h15}")

    with kc3:
        st.markdown(f"**✈️ {res['away']}**")
        p15 = probs['cards']['away']['Over 1.5']
        
        h15 = get_native_history(res['away'], res['league_a'], 'cards', 1.5, 'away', all_dfs)
        c15 = "green" if p15 >= 75 else "red"
        
        st.markdown(f"Over 1.5: :{c15}[**{p15:.0f}%**] | Hist CSV: {h15}")

def main():
    st.title("⚽ FutPrevisão V14.5 (Native Data)")
    st.caption("Histórico 100% verificado via CSV (Sem TXT externo)")
    
    with st.spinner("Carregando bases..."):
        DEBUG_LOGS.clear()
        stats = learn_stats_v14()
        refs = load_referees_v14()
        calendar = load_calendar_safe()
        all_dfs = load_all_dataframes() # Carrega os DF brutos
    
    lista_times = sorted(list(stats.keys()))
    lista_juizes = ["Neutro"] + sorted(list(refs.keys()))
    
    with st.sidebar:
        with st.expander("🛠️ Status do Sistema", expanded=False):
            st.write(f"Times: {len(stats)}")
            st.write(f"Ligas DB: {len(all_dfs)}")
            for log in DEBUG_LOGS: st.write(log)
    
    if not stats:
        st.error("🚨 ERRO: Nenhum dado carregado.")
        return

    tab1, tab2 = st.tabs(["📅 Calendário", "🧪 Simulação Manual"])
    
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
                        else: render_result_v14_5(res, all_dfs)

    with tab2:
        st.subheader("Simulador Personalizado")
        c1, c2, c3 = st.columns(3)
        idx_h = lista_times.index("Liverpool") if "Liverpool" in lista_times else 0
        idx_a = lista_times.index("Man City") if "Man City" in lista_times else 1
        
        h = c1.selectbox("Mandante", lista_times, index=idx_h)
        a = c2.selectbox("Visitante", lista_times, index=idx_a)
        r = c3.selectbox("Árbitro", lista_juizes, index=0)
        
        if st.button("Simular Jogo"):
            ref_val = None if r == "Neutro" else r
            res = calcular_jogo_v14(h, a, stats, ref_val, refs)
            if 'error' in res: st.error(res['error'])
            else: render_result_v14_5(res, all_dfs)

if __name__ == "__main__":
    main()
