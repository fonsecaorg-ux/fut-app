"""
╔═══════════════════════════════════════════════════════════════════════════╗
║       FUTPREVISÃO V23.4 - AUTO TICKET GENERATOR (ODD TARGET 4.5-5.5)      ║
║                                                                            ║
║  ✅ Scanner Automático: Gera bilhete pronto com 1 clique.                 ║
║  ✅ Alvo de Odd: Busca combinações que somem entre 4.50 e 5.50.           ║
║  ✅ Lógica Híbrida: Mistura Âncoras (@1.25) com Fusões (@1.70).           ║
║  ✅ Hedge Automático: Já calcula a proteção para esse bilhete gerado.     ║
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
    page_title="FutPrevisão V23.4",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'bankroll' not in st.session_state: st.session_state.bankroll = 1000.0
if 'bet_history' not in st.session_state: st.session_state.bet_history = []
if 'theme' not in st.session_state: st.session_state.theme = 'dark'
if 'current_ticket' not in st.session_state: st.session_state.current_ticket = []

THRESHOLDS = {
    'radar_corners': 70,
    'radar_cards': 65,
    'anchor_safety': 85,
    'smart_ticket_min': 4.50, # ALVO MÍNIMO
    'smart_ticket_max': 5.50  # ALVO MÁXIMO
}

DEFAULTS = {'shots_on_target': 4.5, 'red_cards_avg': 0.08, 'red_rate_referee': 0.08}

NAME_MAPPING = {
    'Man United': 'Man Utd', 'Manchester United': 'Man Utd', 'Man City': 'Man City',
    'Manchester City': 'Man City', 'Spurs': 'Tottenham', 'Newcastle': 'Newcastle',
    'Wolves': 'Wolves', 'Brighton': 'Brighton', "Nott'm Forest": 'Nottm Forest',
    'West Ham': 'West Ham', 'Leicester': 'Leicester', 'Athletic Club': 'Ath Bilbao',
    'Atl. Madrid': 'Ath Madrid'
}

LIGAS_ALVO = [
    "Premier League", "La Liga", "Serie A", "Bundesliga", "Ligue 1",
    "Championship", "Bundesliga 2", "Pro League", "Süper Lig", "Scottish Premiership"
]

# ═══════════════════════════════════════════════════════════════════════════
# CARREGAMENTO DE DADOS (MANTIDO)
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
            
            all_teams = set(h_stats.index) | set(a_stats.index)
            
            for team in all_teams:
                h = h_stats.loc[team] if team in h_stats.index else pd.Series(0, index=h_stats.columns)
                a = a_stats.loc[team] if team in a_stats.index else pd.Series(0, index=a_stats.columns)
                
                def w_avg(vh, va, deft=0): return (vh * 0.6 + va * 0.4) if (vh+va) > 0 else deft
                
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
                col_nome = 'Arbitro' if 'Arbitro' in df.columns else 'Nome'
                for _, row in df.iterrows():
                    nome = str(row.get(col_nome, '')).strip()
                    if not nome: continue
                    media = float(row.get('Media_Cartoes_Por_Jogo', row.get('Fator', 4.0)))
                    if media < 2: media *= 4.0
                    refs_db[nome] = {'factor': media/4.0, 'red_rate': 0.08, 'strictness': media}
            except: pass
    return refs_db

@st.cache_data(ttl=600)
def load_calendar_safe() -> pd.DataFrame:
    if not os.path.exists("calendario_ligas.csv"): return pd.DataFrame()
    try:
        df = pd.read_csv("calendario_ligas.csv", encoding='utf-8-sig')
        df.columns = [c.strip().replace('\ufeff', '') for c in df.columns]
        rename = {'Mandante': 'Time_Casa', 'Visitante': 'Time_Visitante'}
        df = df.rename(columns=rename)
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

def monte_carlo(xg_h, xg_a, n=1000):
    gh = np.random.poisson(max(0.1, xg_h), n)
    ga = np.random.poisson(max(0.1, xg_a), n)
    return np.count_nonzero(gh > ga)/n, np.count_nonzero(gh == ga)/n, np.count_nonzero(ga > gh)/n

def calcular_jogo_v23(home: str, away: str, stats: Dict, ref: Optional[str], refs: Dict, all_dfs: Dict) -> Dict:
    h_n, a_n = normalize_name(home, list(stats.keys())), normalize_name(away, list(stats.keys()))
    if not h_n or not a_n: return {'error': "Times desconhecidos"}
    
    s_h, s_a = stats[h_n], stats[a_n]
    r_data = refs.get(ref, {'factor': 1.0, 'red_rate': 0.08, 'strictness': 4.0})
    
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
        'meta': {'referee': ref or 'Neutro', 'ref_factor': r_data['factor']}
    }

def get_detailed_probs(pred: Dict) -> Dict:
    def sim(avg, line): return max(5, min(95, 50 + (avg - line) * 15))
    return {
        'corners': {
            'home': {f'Over {l}': sim(pred['corners']['h'], l) for l in [3.5, 4.5]},
            'away': {f'Over {l}': sim(pred['corners']['a'], l) for l in [3.5, 4.5]},
            'total': {f'Over {l}': sim(pred['corners']['total'], l) for l in [8.5, 9.5]}
        },
        'cards': {
            'home': {f'Over {l}': sim(pred['cards']['h'], l) for l in [1.5]},
            'away': {f'Over {l}': sim(pred['cards']['a'], l) for l in [1.5]},
            'total': {f'Over {l}': sim(pred['cards']['total'], l) for l in [3.5]}
        },
        'chance': {'1X': pred['monte_carlo']['h'] + pred['monte_carlo']['d'], 'X2': pred['monte_carlo']['a'] + pred['monte_carlo']['d']}
    }

def get_fair_odd(prob: float) -> float:
    return round(100/prob, 2) if prob > 0 else 99.0

# ═══════════════════════════════════════════════════════════════════════════
# 3. LÓGICA DE AUTO TICKET V23.4 (ODD TARGET 4.5-5.5)
# ═══════════════════════════════════════════════════════════════════════════

def generate_smart_ticket_v23_auto(calendar: pd.DataFrame, stats: Dict, refs: Dict, all_dfs: Dict, date_str: str) -> Dict:
    """
    Gera um bilhete AUTOMÁTICO com Odd entre 4.50 e 5.50.
    Mistura: 2-3 Âncoras (1.25) + 2-3 Fusões (1.60+)
    """
    df_day = calendar[calendar['DtObj'].dt.strftime('%d/%m/%Y') == date_str]
    anchors = []
    fusions = []
    
    for _, row in df_day.iterrows():
        h, a = row['Time_Casa'], row['Time_Visitante']
        res = calcular_jogo_v23(h, a, stats, None, refs, all_dfs)
        if 'error' in res: continue
        probs = get_detailed_probs(res)
        
        # 1. Âncoras (Safety)
        # Cantos Casa/Fora > 3.5
        for loc, name in [('home', res['home']), ('away', res['away'])]:
            p = probs['corners'][loc].get('Over 3.5', 0)
            if p >= 85: # Probabilidade muito alta
                odd = get_fair_odd(p)
                if 1.20 <= odd <= 1.35: # Odd de segurança
                    anchors.append({'type': 'anchor', 'jogo': f"{h} x {a}", 'selection': f"{name} Over 3.5 Cantos", 'prob': p, 'odd': odd})
        
        # Cartões > 1.5
        for loc, name in [('home', res['home']), ('away', res['away'])]:
            p = probs['cards'][loc].get('Over 1.5', 0)
            if p >= 80:
                odd = get_fair_odd(p)
                if 1.25 <= odd <= 1.40:
                    anchors.append({'type': 'anchor', 'jogo': f"{h} x {a}", 'selection': f"{name} Over 1.5 Cartões", 'prob': p, 'odd': odd})

        # 2. Fusões (Valor)
        # Canto > 3.5 E Cartão > 1.5 no mesmo jogo
        pc = probs['corners']['home'].get('Over 3.5', 0)
        pk = probs['cards']['total'].get('Over 3.5', 0)
        
        if pc > 70 and pk > 70:
            comb = (pc/100 * pk/100 * 0.9) * 100
            odd = get_fair_odd(comb)
            if 1.60 <= odd <= 2.10:
                fusions.append({'type': 'fusion', 'jogo': f"{h} x {a}", 'selection': f"Criar Aposta: {res['home']} +3.5 Cantos & Jogo +3.5 Cartões", 'prob': comb, 'odd': odd})

    # Montagem Inteligente do Bilhete
    ticket = []
    curr_odd = 1.0
    used_games = set()
    
    anchors.sort(key=lambda x: x['prob'], reverse=True)
    fusions.sort(key=lambda x: x['prob'], reverse=True)
    
    # Passo A: Pega 2 Âncoras
    for a in anchors:
        if len(ticket) >= 2: break
        if a['jogo'] not in used_games:
            ticket.append(a)
            curr_odd *= a['odd']
            used_games.add(a['jogo'])
            
    # Passo B: Enche de Fusões até Odd ~4.5
    for f in fusions:
        if len(ticket) >= 6: break
        if f['jogo'] not in used_games:
            # Só adiciona se não estourar 6.0
            if curr_odd * f['odd'] <= 6.0:
                ticket.append(f)
                curr_odd *= f['odd']
                used_games.add(f['jogo'])
    
    # Passo C: Ajuste fino (Se ficou baixo, põe mais âncora)
    if curr_odd < 4.0:
        for a in anchors:
            if len(ticket) >= 6: break
            if a['jogo'] not in used_games:
                ticket.append(a)
                curr_odd *= a['odd']
                used_games.add(a['jogo'])
                
    return {'ticket': ticket, 'total_odd': round(curr_odd, 2)}

# Reutilizando a função de Hedge V23.1
def generate_hedges_for_user_ticket(ticket: List[Dict], stats: Dict, refs: Dict, all_dfs: Dict) -> Dict:
    principal, hedge1, hedge2 = [], [], []
    processed = set()
    
    for item in ticket:
        try: parts = item['jogo'].split(' x '); h, a = parts[0], parts[1]
        except: continue
        if f"{h}x{a}" in processed: continue
        
        res = calcular_jogo_v23(h, a, stats, None, refs, all_dfs)
        if 'error' in res: continue
        probs = get_detailed_probs(res)
        
        # Principal
        principal.append({'jogo': item['jogo'], 'selecao': item['selection'], 'odd': item['odd']})
        
        # Hedge 1: Safety (DC)
        sel1 = f"DC {h}" if probs['chance']['1X'] > 60 else f"DC {a}"
        odd1 = 1.35 # Est
        hedge1.append({'jogo': item['jogo'], 'selecao': sel1, 'odd': odd1})
        
        # Hedge 2: Mix (Cartões/Total)
        hedge2.append({'jogo': item['jogo'], 'selecao': "Total Over 3.5 Cartões", 'odd': 1.40})
        
        processed.add(f"{h}x{a}")
        
    return {
        'principal': {'itens': principal, 'odd': round(np.prod([x['odd'] for x in principal]), 2)},
        'hedge1': {'itens': hedge1, 'odd': round(np.prod([x['odd'] for x in hedge1]), 2)},
        'hedge2': {'itens': hedge2, 'odd': round(np.prod([x['odd'] for x in hedge2]), 2)}
    }

# ═══════════════════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════════════════

def render_result_v14_5(res, all_dfs):
    m = res['meta']
    probs = get_detailed_probs(res)
    st.markdown("---")
    st.subheader(f"🏠 {res['home']} vs ✈️ {res['away']}")
    
    c1, c2 = st.columns(2)
    with c1:
        st.info("🏁 **Escanteios**")
        for line in [3.5, 4.5, 5.5]:
            p = probs['corners']['home'].get(f'Over {line}', 0)
            h = get_native_history(res['home'], res['league_h'], 'corners', line, 'home', all_dfs)
            st.write(f"🏠 Over {line}: **{p:.0f}%** (Hist: {h})")
            
    with c2:
        st.warning("🟨 **Cartões**")
        for line in [1.5, 2.5]:
            p = probs['cards']['home'].get(f'Over {line}', 0)
            h = get_native_history(res['home'], res['league_h'], 'cards', line, 'home', all_dfs)
            st.write(f"🏠 Over {line}: **{p:.0f}%** (Hist: {h})")

def main():
    st.title("⚽ FutPrevisão V23.4 (Auto Ticket)")
    
    with st.spinner("Carregando..."):
        stats = learn_stats_v23()
        refs = load_referees_v23()
        calendar = load_calendar_safe()
        all_dfs = load_all_dataframes()
    
    if not stats:
        st.error("🚨 ERRO: Dados não carregados.")
        return

    tab1, tab2, tab3, tab4 = st.tabs(["📅 Calendário", "🎯 Scanner Auto", "🛡️ Sistema Hedges", "🔮 Simulação"])
    
    with tab1:
        if calendar.empty: st.warning("Vazio")
        else: st.dataframe(calendar, use_container_width=True)

    with tab2:
        st.header("🎯 Gerador de Bilhete Automático")
        st.caption("Meta: Odd 4.50 ~ 5.50 | 6 Seleções | Âncoras + Fusões")
        
        if not calendar.empty:
            dates = sorted(calendar['DtObj'].dt.strftime('%d/%m/%Y').unique())
            sel_date = st.selectbox("Data:", dates, key="sd")
            
            if st.button("🚀 GERAR BILHETE DO DIA", type="primary"):
                res = generate_smart_ticket_v23_auto(calendar, stats, refs, all_dfs, sel_date)
                
                if res['ticket']:
                    st.session_state.current_ticket = res['ticket']
                    st.success(f"Bilhete Gerado! Odd Total: @{res['total_odd']}")
                else:
                    st.warning("Não encontrei jogos suficientes para bater a meta de odd hoje.")
            
            if st.session_state.current_ticket:
                st.markdown("### 🎫 Bilhete Sugerido")
                for item in st.session_state.current_ticket:
                    icon = "🔴" if item['type'] == 'anchor' else "🔗"
                    st.write(f"{icon} {item['jogo']} | **{item['selection']}** (@{item['odd']})")
                st.info("Vá para a aba 'Sistema Hedges' para proteger este bilhete.")

    with tab3:
        st.header("🛡️ Sistema de Proteção")
        if not st.session_state.current_ticket:
            st.warning("Gere um bilhete na aba Scanner primeiro.")
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
                    st.subheader("🔄 Mix")
                    st.metric("Odd", f"@{hedges['hedge2']['odd']}")
                    for i in hedges['hedge2']['itens']: st.caption(f"{i['jogo']} - {i['selecao']}")

    with tab4:
        l = sorted(list(stats.keys()))
        c1, c2 = st.columns(2)
        h = c1.selectbox("Casa", l, index=0)
        a = c2.selectbox("Fora", l, index=1)
        if st.button("Simular"):
            res = calcular_jogo_v23(h, a, stats, None, refs, all_dfs)
            if 'error' not in res: render_result_v14_5(res, all_dfs)

if __name__ == "__main__":
    main()
