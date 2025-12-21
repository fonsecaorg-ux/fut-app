"""
╔═══════════════════════════════════════════════════════════════════════════╗
║       FUTPREVISÃO V24.0 - CONSTRUTOR HÍBRIDO (MANUAL + AUTO)              ║
║                                                                            ║
║  ✅ Aba 3: Scanner Automático (com filtros) OU Manual (passo-a-passo)     ║
║  ✅ Aba 4: Hedges calculados sobre o bilhete gerado na Aba 3              ║
║  ✅ Helper: Função de mercados manuais corrigida e posicionada            ║
║  ✅ Dados: Base V23 mantida 100%                                          ║
║                                                                            ║
║  Dezembro 2025                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import math
import numpy as np
import os
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any, Optional, Tuple
from difflib import get_close_matches
from datetime import datetime
import json

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURAÇÃO
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="FutPrevisão V24 Híbrido",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session State
if 'bankroll' not in st.session_state: st.session_state.bankroll = 1000.0
if 'bet_history' not in st.session_state: st.session_state.bet_history = []
if 'favorites' not in st.session_state: st.session_state.favorites = {'teams': [], 'matches': []}
if 'theme' not in st.session_state: st.session_state.theme = 'dark'
if 'alert_threshold' not in st.session_state: st.session_state.alert_threshold = 80.0
if 'current_ticket' not in st.session_state: st.session_state.current_ticket = []

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTES
# ═══════════════════════════════════════════════════════════════════════════

THRESHOLDS = {
    'radar_corners': 70,
    'radar_cards': 65,
    'anchor_safety': 85,
    'smart_ticket_min': 4.60,
    'smart_ticket_max': 5.50
}

DEFAULTS = {'shots_on_target': 4.5, 'red_cards_avg': 0.08, 'red_rate_referee': 0.08}

NAME_MAPPING = {
    'Man United': 'Man Utd', 'Manchester United': 'Man Utd',
    'Man City': 'Man City', 'Manchester City': 'Man City',
    'Spurs': 'Tottenham', 'Newcastle': 'Newcastle',
    'Wolves': 'Wolves', 'Brighton': 'Brighton',
    "Nott'm Forest": 'Nottm Forest', 'Nottingham Forest': 'Nottm Forest',
    'West Ham': 'West Ham', 'Leicester': 'Leicester',
    'Athletic Club': 'Ath Bilbao', 'Atl. Madrid': 'Ath Madrid'
}

LIGAS_ALVO = [
    "Premier League", "La Liga", "Serie A", "Bundesliga", "Ligue 1",
    "Championship", "Bundesliga 2", "Pro League", "Süper Lig", "Scottish Premiership"
]

# ═══════════════════════════════════════════════════════════════════════════
# FUNÇÕES AUXILIARES (POSICIONADAS AQUI PARA EVITAR ERROS)
# ═══════════════════════════════════════════════════════════════════════════

def get_fair_odd(prob: float) -> float:
    """Calcula odd justa"""
    return round(100/prob, 2) if prob > 0 else 99.0

def get_available_markets_for_game(res: Dict, probs: Dict) -> List[Dict]:
    """Retorna lista de mercados disponíveis para seleção manual"""
    markets = []
    
    # Cantos Casa
    for l in [3.5, 4.5, 5.5]:
        p = probs['corners']['home'].get(f'Over {l}', 0)
        markets.append({'mercado': f"{res['home']} Over {l} Escanteios", 'prob': p, 'odd': get_fair_odd(p), 'type': 'Escanteios'})
        
    # Cantos Fora
    for l in [2.5, 3.5, 4.5]:
        p = probs['corners']['away'].get(f'Over {l}', 0)
        markets.append({'mercado': f"{res['away']} Over {l} Escanteios", 'prob': p, 'odd': get_fair_odd(p), 'type': 'Escanteios'})
        
    # Cartões
    for l in [1.5, 2.5]:
        p1 = probs['cards']['home'].get(f'Over {l}', 0)
        markets.append({'mercado': f"{res['home']} Over {l} Cartões", 'prob': p1, 'odd': get_fair_odd(p1), 'type': 'Cartões'})
        
        p2 = probs['cards']['away'].get(f'Over {l}', 0)
        markets.append({'mercado': f"{res['away']} Over {l} Cartões", 'prob': p2, 'odd': get_fair_odd(p2), 'type': 'Cartões'})
        
    return markets

# ═══════════════════════════════════════════════════════════════════════════
# CARREGAMENTO DE DADOS
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def find_and_load_csv(league_name: str) -> pd.DataFrame:
    attempts = [f"{league_name} 25.26.csv", f"{league_name.replace(' ', '_')}_25_26.csv", f"{league_name}.csv"]
    if "Süper Lig" in league_name: attempts.extend(["Super Lig Turquia 25.26.csv"])
    
    for filename in attempts:
        if os.path.exists(filename):
            try:
                try: df = pd.read_csv(filename, encoding='utf-8-sig')
                except: df = pd.read_csv(filename, encoding='latin1')
                
                if not df.empty:
                    df.columns = [c.strip().replace('\ufeff', '') for c in df.columns]
                    rename_map = {'Mandante': 'HomeTeam', 'Visitante': 'AwayTeam', 
                                  'Time_Casa': 'HomeTeam', 'Time_Visitante': 'AwayTeam'}
                    df = df.rename(columns=rename_map)
                    df['_League_'] = league_name
                    return df
            except: pass
    return pd.DataFrame()

@st.cache_resource
def load_all_dataframes() -> Dict[str, pd.DataFrame]:
    dfs = {}
    for league in LIGAS_ALVO:
        df = find_and_load_csv(league)
        if not df.empty: dfs[league] = df
    return dfs

@st.cache_data(ttl=3600)
def learn_stats_v23() -> Dict[str, Dict[str, Any]]:
    stats_db = {}
    for league in LIGAS_ALVO:
        df = find_and_load_csv(league)
        if df.empty: continue
        
        for c in ['HomeTeam', 'AwayTeam', 'HC', 'AC', 'HY', 'AY', 'HF', 'AF', 'FTHG', 'FTAG', 'HST', 'AST', 'HR', 'AR']:
            if c not in df.columns: df[c] = np.nan
            
        try:
            h_stats = df.groupby('HomeTeam').agg({'HC': ['mean','std'], 'HY': ['mean','std'], 'HF': 'mean', 'FTHG': ['mean','std'], 'FTAG': 'mean', 'HST': 'mean', 'HR': 'mean'})
            a_stats = df.groupby('AwayTeam').agg({'AC': ['mean','std'], 'AY': ['mean','std'], 'AF': 'mean', 'FTAG': ['mean','std'], 'FTHG': 'mean', 'AST': 'mean', 'AR': 'mean'})
            
            h_stats.columns = ['_'.join(col).strip() for col in h_stats.columns.values]
            a_stats.columns = ['_'.join(col).strip() for col in a_stats.columns.values]
            h_stats = h_stats.fillna(value={'HST_mean': 4.5, 'HR_mean': 0.08, 'HC_std': 1.5, 'HY_std': 0.8, 'FTHG_std': 1.0})
            a_stats = a_stats.fillna(value={'AST_mean': 4.5, 'AR_mean': 0.08, 'AC_std': 1.5, 'AY_std': 0.8, 'FTAG_std': 1.0})
            
            for team in set(h_stats.index) | set(a_stats.index):
                h = h_stats.loc[team] if team in h_stats.index else pd.Series(0, index=h_stats.columns)
                a = a_stats.loc[team] if team in a_stats.index else pd.Series(0, index=a_stats.columns)
                
                def w_avg(val_h, val_a, default=0): return (val_h * 0.6 + val_a * 0.4) if (val_h+val_a) > 0 else default
                
                c_std = w_avg(h.get('HC_std', 1.5), a.get('AC_std', 1.5), 1.5)
                stats_db[team] = {
                    'corners': w_avg(h.get('HC_mean',0), a.get('AC_mean',0), 5.0),
                    'consistency_corners': round(max(0, 100 - (c_std * 30)), 1),
                    'cards': w_avg(h.get('HY_mean',0), a.get('AY_mean',0), 2.0),
                    'consistency_cards': max(0, 100 - (w_avg(h.get('HY_std',0.8), a.get('AY_std',0.8), 0.8)*50)),
                    'fouls': w_avg(h.get('HF_mean',0), a.get('AF_mean',0), 11.0),
                    'goals_f': w_avg(h.get('FTHG_mean',0), a.get('FTAG_mean',0), 1.2),
                    'goals_a': w_avg(h.get('FTAG_mean',0), a.get('FTHG_mean',0), 1.2),
                    'shots_on_target': w_avg(h.get('HST_mean',0), a.get('AST_mean',0), 4.5),
                    'red_cards_avg': w_avg(h.get('HR_mean',0), a.get('AR_mean',0), 0.08),
                    'league': league
                }
        except: pass
    return stats_db

@st.cache_data(ttl=3600)
def load_referees_v23() -> Dict:
    refs_db = {}
    files = ["arbitros_5_ligas_2025_2026.csv", "arbitros.csv"]
    for f in files:
        if os.path.exists(f):
            try:
                df = pd.read_csv(f, encoding='utf-8-sig')
                df.columns = [c.strip().replace('\ufeff', '') for c in df.columns]
                for _, row in df.iterrows():
                    nome = str(row.get('Arbitro', row.get('Nome', ''))).strip()
                    if not nome: continue
                    media = float(row.get('Media_Cartoes_Por_Jogo', row.get('Fator', 4.0)))
                    if media < 2: media *= 4.0 # Correção para Fator
                    refs_db[nome] = {'factor': media/4.0, 'red_rate': 0.08, 'strictness': media}
            except: pass
    return refs_db

@st.cache_data(ttl=600)
def load_calendar_safe() -> pd.DataFrame:
    if not os.path.exists("calendario_ligas.csv"): return pd.DataFrame()
    try:
        try: df = pd.read_csv("calendario_ligas.csv", encoding='utf-8-sig')
        except: df = pd.read_csv("calendario_ligas.csv", encoding='latin1')
        df.columns = [c.strip().replace('\ufeff', '') for c in df.columns]
        df = df.rename(columns={'Mandante': 'Time_Casa', 'Visitante': 'Time_Visitante'})
        df['DtObj'] = pd.to_datetime(df['Data'], format='%d/%m/%Y', errors='coerce')
        return df.dropna(subset=['DtObj']).sort_values(by=['DtObj', 'Hora'])
    except: return pd.DataFrame()

# ═══════════════════════════════════════════════════════════════════════════
# MOTOR DE CÁLCULO
# ═══════════════════════════════════════════════════════════════════════════

def normalize_name(name: str, db_keys: list) -> Optional[str]:
    if not name: return None
    if name in NAME_MAPPING: name = NAME_MAPPING[name]
    if name in db_keys: return name
    matches = get_close_matches(name, db_keys, n=1, cutoff=0.6)
    return matches[0] if matches else None

def get_native_history(team_name: str, league: str, market: str, line: float, location: str, all_dfs: Dict) -> str:
    if league not in all_dfs: return "N/A"
    df = all_dfs[league]
    col_map = {('home', 'corners'): 'HC', ('away', 'corners'): 'AC', ('home', 'cards'): 'HY', ('away', 'cards'): 'AY'}
    col_code = col_map.get((location, market))
    team_col = 'HomeTeam' if location == 'home' else 'AwayTeam'
    matches = df[df[team_col] == team_name]
    if matches.empty: return "0/0"
    last = matches.tail(10)
    hits = sum(1 for val in last[col_code] if float(val) > line)
    return f"{hits}/{len(last)}"

def poisson(k: int, lamb: float) -> float:
    return (lamb**k * math.exp(-lamb)) / math.factorial(k) if lamb <= 30 else 0

def monte_carlo(xg_h: float, xg_a: float, n: int = 1000) -> Tuple[float, float, float]:
    gh = np.random.poisson(max(0.1, xg_h), n)
    ga = np.random.poisson(max(0.1, xg_a), n)
    return np.count_nonzero(gh > ga)/n, np.count_nonzero(gh == ga)/n, np.count_nonzero(ga > gh)/n

def calcular_jogo_v23(home: str, away: str, stats: Dict, ref: Optional[str], refs_db: Dict, all_dfs: Dict) -> Dict:
    h_n = normalize_name(home, list(stats.keys()))
    a_n = normalize_name(away, list(stats.keys()))
    if not h_n or not a_n: return {'error': "Times desconhecidos"}
    
    s_h, s_a = stats[h_n], stats[a_n]
    r_data = refs_db.get(ref, {'factor': 1.0, 'red_rate': 0.08, 'strictness': 4.0})
    
    p_h = 1.15 if s_h['shots_on_target'] > 5.5 else 1.0
    p_a = 1.15 if s_a['shots_on_target'] > 5.5 else 1.0
    
    corn_h = s_h['corners'] * 1.15 * p_h
    corn_a = s_a['corners'] * 0.90 * p_a
    
    card_h = s_h['cards'] * r_data['factor']
    card_a = s_a['cards'] * r_data['factor']
    
    xg_h, xg_a = max(0.1, s_h['goals_f']), max(0.1, s_a['goals_f'])
    mc_h, mc_d, mc_a = monte_carlo(xg_h, xg_a)
    
    return {
        'home': h_n, 'away': a_n, 'league_h': s_h['league'], 'league_a': s_a['league'],
        'corners': {'h': corn_h, 'a': corn_a, 'total': corn_h+corn_a},
        'cards': {'h': card_h, 'a': card_a, 'total': card_h+card_a},
        'goals': {'h': xg_h, 'a': xg_a},
        'monte_carlo': {'h': mc_h*100, 'd': mc_d*100, 'a': mc_a*100},
        'consistency': {'corners_h': s_h.get('consistency_corners', 50), 'corners_a': s_a.get('consistency_corners', 50), 'goals_h': s_h.get('consistency_goals', 50)},
        'meta': {'referee': ref or 'Neutro', 'ref_factor': r_data['factor'], 'ref_label': "Rigoroso" if r_data['strictness'] > 4.5 else "Normal", 'prob_red': 8.0, 'prob_red_label': "Baixa"},
        'probs': {'btts': 50.0, 'over_2_5': 50.0}
    }

def get_detailed_probs(res: Dict) -> Dict:
    def sim(avg, line): return max(5, min(95, 50 + (avg - line) * 15))
    return {
        'corners': {
            'home': {f'Over {l}': sim(res['corners']['h'], l) for l in [2.5, 3.5, 4.5, 5.5]},
            'away': {f'Over {l}': sim(res['corners']['a'], l) for l in [2.5, 3.5, 4.5]},
            'total': {f'Over {l}': sim(res['corners']['total'], l) for l in [8.5, 9.5, 10.5, 11.5]}
        },
        'cards': {
            'home': {f'Over {l}': sim(res['cards']['h'], l) for l in [1.5, 2.5]},
            'away': {f'Over {l}': sim(res['cards']['a'], l) for l in [1.5, 2.5]},
            'total': {f'Over {l}': sim(res['cards']['total'], l) for l in [2.5, 3.5, 4.5, 5.5]}
        },
        'chance': {'1X': res['monte_carlo']['h'] + res['monte_carlo']['d'], 'X2': res['monte_carlo']['a'] + res['monte_carlo']['d'], '12': res['monte_carlo']['h'] + res['monte_carlo']['a']},
        'goals': {'BTTS': 50.0, 'Over 2.5': 50.0}
    }

# ═══════════════════════════════════════════════════════════════════════════
# LÓGICA V24.1 - SCANNER DIVERSIFICADO E HEDGE INTELIGENTE
# ═══════════════════════════════════════════════════════════════════════════

def generate_smart_ticket_v23(calendar: pd.DataFrame, stats: Dict, refs: Dict, all_dfs: Dict, date_str: str, 
                              target_leagues: List[str] = None, target_games: List[str] = None) -> Dict:
    """Scanner V24.6 - FULL STATS (2 Sugestões/Jogo)"""
    
    df_day = calendar[calendar['DtObj'].dt.strftime('%d/%m/%Y') == date_str].copy()
    if target_leagues: df_day = df_day[df_day['Liga'].isin(target_leagues)]
    df_day['GameID'] = df_day['Time_Casa'] + ' vs ' + df_day['Time_Visitante']
    if target_games: df_day = df_day[df_day['GameID'].isin(target_games)]
    
    ticket = []
    curr_odd = 1.0
    
    # Limita jogos auto se não houver filtro
    if not target_games and len(df_day) > 3: df_day = df_day.head(3)

    for _, row in df_day.iterrows():
        home, away = row['Time_Casa'], row['Time_Visitante']
        liga, hora = row.get('Liga', 'N/A'), row.get('Hora', 'N/A')
        
        res = calcular_jogo_v23(home, away, stats, None, refs, all_dfs)
        if 'error' in res: continue
        probs = get_detailed_probs(res)
        
        candidates = []
        
        # 1. Escanteios (Times)
        for loc, name in [('home', res['home']), ('away', res['away'])]:
            for l in [3.5, 4.5]:
                p = probs['corners'][loc].get(f'Over {l}', 0)
                if p >= 65: candidates.append({'mercado': f"{name} Over {l} Escanteios", 'prob': p, 'odd': get_fair_odd(p), 'type': 'Escanteios'})
        
        # 2. Cartões (Times)
        for loc, name in [('home', res['home']), ('away', res['away'])]:
            p = probs['cards'][loc].get('Over 1.5', 0)
            if p >= 60: candidates.append({'mercado': f"{name} Over 1.5 Cartões", 'prob': p, 'odd': get_fair_odd(p), 'type': 'Cartões'})
        
        # 3. Totais
        p_card_tot = probs['cards']['total'].get('Over 3.5', 0)
        if p_card_tot >= 65: candidates.append({'mercado': "Total Jogo Over 3.5 Cartões", 'prob': p_card_tot, 'odd': get_fair_odd(p_card_tot), 'type': 'TotalCartões'})

        candidates.sort(key=lambda x: x['prob'], reverse=True)
        selected = []
        types_used = set()
        
        for cand in candidates:
            if len(selected) >= 2: break
            if cand['type'] not in types_used or cand['prob'] > 80:
                if not any(x['mercado'] == cand['mercado'] for x in selected):
                    item = {
                        'type': 'auto_dual', 'jogo': f"{home} vs {away}",
                        'mercado': cand['mercado'], 'prob': cand['prob'], 'odd': cand['odd'],
                        'liga': liga, 'hora': hora
                    }
                    selected.append(item)
                    types_used.add(cand['type'])
        
        for item in selected:
            if curr_odd * item['odd'] <= 60.0:
                ticket.append(item)
                curr_odd *= item['odd']

    return {'ticket': ticket, 'total_odd': round(curr_odd, 2), 'num_selections': len(ticket)}

def get_detailed_probs(res: Dict) -> Dict:
    """
    Gera probabilidades detalhadas para várias linhas.
    ✅ ATUALIZADO: Inclui linhas baixas (Safety) como Over 6.5/7.5 Cantos.
    """
    
    def sim_prob(avg: float, line: float) -> float:
        # Fórmula de Simulação Logística Simples
        diff = avg - line
        # Se a média for muito maior que a linha, prob tende a 99%
        # Se a média for muito menor, prob tende a 1%
        # Ajuste de sensibilidade (15)
        prob = 50 + (diff * 15)
        return max(1.0, min(99.0, prob))
    
    probs = {
        'corners': {
            'home': {f'Over {l}': sim_prob(res['corners']['h'], l) for l in [2.5, 3.5, 4.5, 5.5, 6.5]},
            'away': {f'Over {l}': sim_prob(res['corners']['a'], l) for l in [2.5, 3.5, 4.5, 5.5]},
            # GERA LINHAS DE SEGURANÇA (6.5 e 7.5) ATÉ LINHAS ALTAS
            'total': {f'Over {l}': sim_prob(res['corners']['total'], l) for l in [6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5]}
        },
        'cards': {
            'home': {f'Over {l}': sim_prob(res['cards']['h'], l) for l in [0.5, 1.5, 2.5]},
            'away': {f'Over {l}': sim_prob(res['cards']['a'], l) for l in [0.5, 1.5, 2.5]},
            # GERA LINHAS DE SEGURANÇA (1.5 e 2.5)
            'total': {f'Over {l}': sim_prob(res['cards']['total'], l) for l in [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]}
        }
    }
    
    mc = res['monte_carlo']
    probs['chance'] = {
        '1': mc['h'],
        'X': mc['d'],
        '2': mc['a'],
        '1X': mc['h'] + mc['d'],
        'X2': mc['a'] + mc['d'],
        '12': mc['h'] + mc['a']
    }
    
    probs['goals'] = {
        'BTTS': res['probs']['btts'],
        'Over 2.5': res['probs']['over_2_5']
    }
    
    return probs
# ═══════════════════════════════════════════════════════════════════════════
# UI PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════

def main():
    if st.session_state.theme == 'dark':
        st.markdown("<style>.stApp {background-color: #0E1117; color: #FAFAFA;}</style>", unsafe_allow_html=True)
        
    st.sidebar.title("🎛️ Painel V24.0")
    
    col_t1, col_t2 = st.sidebar.columns(2)
    if col_t1.button("🌙 Escuro"): st.session_state.theme = 'dark'; st.rerun()
    if col_t2.button("☀️ Claro"): st.session_state.theme = 'light'; st.rerun()
    
    with st.spinner("Carregando bases..."):
        stats = learn_stats_v23()
        refs = load_referees_v23()
        calendar = load_calendar_safe()
        all_dfs = load_all_dataframes()
    
    st.title("⚽ FutPrevisão V24.0 Híbrido")
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📅 Calendário", "🔍 Simulação", "🎯 Construtor", "🛡️ Hedges", "📊 Radares", "📈 Dashboard"])
    
    with tab1:
        if not calendar.empty:
            dates = sorted(calendar['DtObj'].dt.strftime('%d/%m/%Y').unique())
            sel_date = st.selectbox("Data:", dates, key="cal_date")
            df_d = calendar[calendar['DtObj'].dt.strftime('%d/%m/%Y') == sel_date]
            st.dataframe(df_d, use_container_width=True)
            
    with tab2:
        l_times = sorted(list(stats.keys()))
        c1, c2 = st.columns(2)
        h = c1.selectbox("Casa", l_times)
        a = c2.selectbox("Fora", l_times, index=1)
        if st.button("Simular"):
            res = calcular_jogo_v23(h, a, stats, None, refs, all_dfs)
            if 'error' not in res:
                st.info(f"xG: {res['goals']['h']:.2f} x {res['goals']['a']:.2f}")
                
    with tab3:
        st.header("🎫 Construtor de Bilhetes")
        st.caption("Escolha como quer montar seu Bilhete Principal. O Hedge (Aba 4) protegerá o resultado final.")
        
        modo_construcao = st.radio("Como você quer montar o bilhete?", ["🤖 Robô Scanner (Automático)", "✍️ Manual (Passo-a-passo)"], horizontal=True)
        st.markdown("---")
        
        if not calendar.empty:
            dates = sorted(calendar['DtObj'].dt.strftime('%d/%m/%Y').unique())
            sel_date = st.selectbox("Data dos Jogos:", dates, key="builder_date")
            df_day = calendar[calendar['DtObj'].dt.strftime('%d/%m/%Y') == sel_date]
            
            if "Robô" in modo_construcao:
                st.subheader("🤖 Scanner Inteligente")
                c_filters1, c_filters2 = st.columns(2)
                available_leagues = sorted(df_day['Liga'].unique())
                with c_filters1: sel_leagues = st.multiselect("Filtrar Ligas:", available_leagues, default=available_leagues)
                
                if sel_leagues: df_filtered = df_day[df_day['Liga'].isin(sel_leagues)]
                else: df_filtered = df_day
                available_games = sorted((df_filtered['Time_Casa'] + ' vs ' + df_filtered['Time_Visitante']).unique())
                with c_filters2: sel_games = st.multiselect("Filtrar Jogos Específicos:", available_games)
                
                if st.button("🚀 ROBÔ: GERAR BILHETE AGORA", type="primary"):
                    res = generate_smart_ticket_v23(calendar, stats, refs, all_dfs, sel_date, target_leagues=sel_leagues, target_games=sel_games)
                    if res['ticket']:
                        st.session_state.current_ticket = res['ticket']
                        st.success(f"O Robô montou um bilhete com {len(res['ticket'])} seleções!")
                    else: st.warning("Nenhuma oportunidade encontrada.")

            else:
                st.subheader("✍️ Montagem Manual")
                available_games_man = sorted((df_day['Time_Casa'] + ' vs ' + df_day['Time_Visitante']).unique())
                selected_game_str = st.selectbox("1. Escolha a Partida:", available_games_man)
                
                if selected_game_str:
                    row = df_day[(df_day['Time_Casa'] + ' vs ' + df_day['Time_Visitante']) == selected_game_str].iloc[0]
                    res_man = calcular_jogo_v23(row['Time_Casa'], row['Time_Visitante'], stats, None, refs, all_dfs)
                    
                    if 'error' not in res_man:
                        probs_man = get_detailed_probs(res_man)
                        available_markets = get_available_markets_for_game(res_man, probs_man)
                        st.write("2. Escolha o Mercado:")
                        market_options = [f"{m['mercado']} | Prob: {m['prob']:.0f}% | Odd: @{m['odd']}" for m in available_markets if m['prob'] > 50]
                        selected_market_str = st.selectbox("Mercados Disponíveis:", market_options)
                        
                        if st.button("➕ Adicionar ao Bilhete"):
                            sel_obj = next(m for m in available_markets if f"{m['mercado']} | Prob: {m['prob']:.0f}% | Odd: @{m['odd']}" == selected_market_str)
                            new_item = {'type': 'manual', 'jogo': selected_game_str, 'mercado': sel_obj['mercado'], 'prob': sel_obj['prob'], 'odd': sel_obj['odd'], 'liga': row['Liga'], 'hora': row['Hora']}
                            if 'current_ticket' not in st.session_state: st.session_state.current_ticket = []
                            st.session_state.current_ticket.append(new_item)
                            st.success("Adicionado!")
            
            st.markdown("---")
            if st.session_state.current_ticket:
                if st.button("🗑️ Limpar Bilhete"): st.session_state.current_ticket = []; st.rerun()
                ticket = st.session_state.current_ticket
                total_odd = np.prod([b['odd'] for b in ticket])
                st.metric("Odd Total Atual", f"@{total_odd:.2f}", delta=f"{len(ticket)} seleções")
                for i, item in enumerate(ticket, 1):
                    desc = item.get('mercado', item.get('selection'))
                    if item.get('type') == 'fusion': desc = " + ".join(item['mercados'])
                    st.write(f"**{i}. {item['jogo']}** - {desc} (@{item['odd']})")
                st.info("Vá para a aba 'Hedges' para proteger.")

    with tab4:
        st.header("🛡️ Sistema de Proteção (Hedges)")
        if not st.session_state.current_ticket:
            st.warning("Gere um bilhete no Scanner ou monte manualmente primeiro.")
        else:
            if st.button("🛡️ CALCULAR HEDGES", type="primary"):
                hedges = generate_hedges_for_user_ticket(st.session_state.current_ticket, stats, refs, all_dfs)
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.subheader("📋 Principal")
                    st.metric("Odd", f"@{hedges['principal']['odd']}")
                    for i in hedges['principal']['itens']: st.caption(f"{i['jogo']} - {i['selecao']}")
                with c2:
                    st.subheader("🛡️ Safety")
                    st.metric("Odd", f"@{hedges['hedge1']['odd']}")
                    for i in hedges['hedge1']['itens']: st.caption(f"{i['jogo']} - {i['selecao']}")
                with c3:
                    st.subheader("🔄 Caos")
                    st.metric("Odd", f"@{hedges['hedge2']['odd']}")
                    for i in hedges['hedge2']['itens']: st.caption(f"{i['jogo']} - {i['selecao']}")

    with tab5:
         if not calendar.empty:
            dates = sorted(calendar['DtObj'].dt.strftime('%d/%m/%Y').unique())
            sel = st.selectbox("Data Radar:", dates, key="rad_d")
            if st.button("Escanear"):
                res = scan_day_for_radars(calendar, stats, refs, all_dfs, sel)
                st.write(res)
                
    with tab6:
        st.write("Dashboard V24")

if __name__ == "__main__":
    main()
