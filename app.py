"""
╔═══════════════════════════════════════════════════════════════════════════╗
║       FUTPREVISÃO V23.2 - FINAL STABLE (CORREÇÃO TELA BRANCA)             ║
║                                                                            ║
║  ✅ Base: V23.0 Ultimate (Dados Reais + Dashboard)                        ║
║  ✅ Correção: Integração Scanner -> Hedge via Session State               ║
║  ✅ Ajuste: Scanner focado em Âncoras (1.25) + Fusões (1.60-1.90)         ║
║  ✅ Fix: Prevenção de Tela Branca (Erros de Renderização)                 ║
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
from typing import Dict, List, Any, Optional, Tuple
from difflib import get_close_matches
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURAÇÃO
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="FutPrevisão V23.2",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inicialização do Session State
if 'bankroll' not in st.session_state: st.session_state.bankroll = 1000.0
if 'bet_history' not in st.session_state: st.session_state.bet_history = []
if 'theme' not in st.session_state: st.session_state.theme = 'dark'
if 'current_ticket' not in st.session_state: st.session_state.current_ticket = [] # O Bilhete salvo fica aqui

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTES
# ═══════════════════════════════════════════════════════════════════════════

THRESHOLDS = {
    'radar_corners': 70,
    'radar_cards': 65,
    'anchor_safety': 85,     # Probabilidade alta para âncoras
    'smart_ticket_min': 4.60,
    'smart_ticket_max': 6.50
}

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
# CARREGAMENTO DE DADOS (BASE V23)
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
        
        # Garante colunas necessárias
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
        df = df.rename(columns={'Mandante': 'Time_Casa', 'Visitante': 'Time_Visitante'})
        df['DtObj'] = pd.to_datetime(df['Data'], format='%d/%m/%Y', errors='coerce')
        return df.dropna(subset=['DtObj']).sort_values(by=['DtObj', 'Hora'])
    except: return pd.DataFrame()

# ═══════════════════════════════════════════════════════════════════════════
# MOTOR DE CÁLCULO
# ═══════════════════════════════════════════════════════════════════════════

def normalize_name(name: str, db_keys: list) -> Optional[str]:
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
    h_n = normalize_name(home, list(stats.keys()))
    a_n = normalize_name(away, list(stats.keys()))
    if not h_n or not a_n: return {'error': "Times desconhecidos"}
    
    s_h, s_a = stats[h_n], stats[a_n]
    r = refs.get(ref, {'factor': 1.0, 'red_rate': 0.08, 'strictness': 4.0})
    
    p_h = 1.15 if s_h['shots_on_target'] > 5.5 else 1.0
    p_a = 1.15 if s_a['shots_on_target'] > 5.5 else 1.0
    
    corn_h = s_h['corners'] * 1.15 * p_h
    corn_a = s_a['corners'] * 0.90 * p_a
    
    card_h = s_h['cards'] * r['factor']
    card_a = s_a['cards'] * r['factor']
    
    xg_h, xg_a = max(0.1, s_h['goals_f']), max(0.1, s_a['goals_f'])
    mc_h, mc_d, mc_a = monte_carlo(xg_h, xg_a)
    
    btts = (1 - poisson(0, xg_h)) * (1 - poisson(0, xg_a)) * 100
    
    return {
        'home': h_n, 'away': a_n, 'league_h': s_h['league'], 'league_a': s_a['league'],
        'corners': {'h': corn_h, 'a': corn_a, 'total': corn_h+corn_a},
        'cards': {'h': card_h, 'a': card_a, 'total': card_h+card_a},
        'goals': {'h': xg_h, 'a': xg_a},
        'monte_carlo': {'h': mc_h*100, 'd': mc_d*100, 'a': mc_a*100},
        'consistency': {
            'corners_h': s_h.get('consistency_corners', 50), 'corners_a': s_a.get('consistency_corners', 50),
            'cards_h': s_h.get('consistency_cards', 50), 'cards_a': s_a.get('consistency_cards', 50)
        },
        'meta': {'referee': ref or 'Neutro', 'ref_factor': r['factor'], 'probs': {'btts': btts}}
    }

def get_detailed_probs(res: Dict) -> Dict:
    def sim(avg, line): return max(5, min(95, 50 + (avg - line) * 15))
    return {
        'corners': {
            'home': {f'Over {l}': sim(res['corners']['h'], l) for l in [2.5, 3.5, 4.5]},
            'away': {f'Over {l}': sim(res['corners']['a'], l) for l in [2.5, 3.5]},
            'total': {f'Over {l}': sim(res['corners']['total'], l) for l in [8.5, 9.5, 10.5]}
        },
        'cards': {
            'home': {f'Over {l}': sim(res['cards']['h'], l) for l in [1.5, 2.5]},
            'away': {f'Over {l}': sim(res['cards']['a'], l) for l in [1.5, 2.5]},
            'total': {f'Over {l}': sim(res['cards']['total'], l) for l in [3.5, 4.5]}
        },
        'chance': {'1X': res['monte_carlo']['h'] + res['monte_carlo']['d'], 'X2': res['monte_carlo']['a'] + res['monte_carlo']['d']}
    }

def get_fair_odd(prob: float) -> float:
    return round(100/prob, 2) if prob > 0 else 99.0

# ═══════════════════════════════════════════════════════════════════════════
# SCANNER V23 (CORRIGIDO PARA ÂNCORAS + FUSÕES)
# ═══════════════════════════════════════════════════════════════════════════

def scan_day_for_radars(calendar: pd.DataFrame, stats: Dict, refs: Dict, all_dfs: Dict, date_str: str) -> Dict:
    df_day = calendar[calendar['DtObj'].dt.strftime('%d/%m/%Y') == date_str]
    radar = {'corners_individual': [], 'cards_individual': [], 'corners_total': [], 'cards_total': []}
    
    for _, row in df_day.iterrows():
        home, away = row['Time_Casa'], row['Time_Visitante']
        res = calcular_jogo_v23(home, away, stats, None, refs, all_dfs)
        if 'error' in res: continue
        probs = get_detailed_probs(res)
        
        # Individual Cantos
        for line in [3.5, 4.5]:
            ph = probs['corners']['home'].get(f'Over {line}', 0)
            if ph >= THRESHOLDS['radar_corners']:
                radar['corners_individual'].append({'time': res['home'], 'mercado': f"{res['home']} Over {line} Cantos", 'prob': ph})
            
            if line <= 4.5:
                pa = probs['corners']['away'].get(f'Over {line}', 0)
                if pa >= THRESHOLDS['radar_corners']:
                    radar['corners_individual'].append({'time': res['away'], 'mercado': f"{res['away']} Over {line} Cantos", 'prob': pa})
        
        # Individual Cartões
        for line in [1.5, 2.5]:
            ph = probs['cards']['home'].get(f'Over {line}', 0)
            if ph >= THRESHOLDS['radar_cards']:
                radar['cards_individual'].append({'time': res['home'], 'mercado': f"{res['home']} Over {line} Cartões", 'prob': ph})
    return radar

def generate_smart_ticket_v23(calendar: pd.DataFrame, stats: Dict, refs: Dict, all_dfs: Dict, date_str: str) -> Dict:
    """Scanner que gera 1 bilhete híbrido: Âncoras (1.25) + Fusões (1.60-1.90)"""
    df_day = calendar[calendar['DtObj'].dt.strftime('%d/%m/%Y') == date_str]
    anchors = []
    fusions = []
    
    for _, row in df_day.iterrows():
        home, away = row['Time_Casa'], row['Time_Visitante']
        res = calcular_jogo_v23(home, away, stats, None, refs, all_dfs)
        if 'error' in res: continue
        probs = get_detailed_probs(res)
        
        # 1. Âncoras (Safety) - Odds 1.22 a 1.40
        for loc, name in [('home', res['home']), ('away', res['away'])]:
            # Cantos
            for l in [3.5, 4.5]:
                p = probs['corners'][loc].get(f'Over {l}', 0)
                if p >= 85: 
                    odd = get_fair_odd(p)
                    if 1.22 <= odd <= 1.40:
                        anchors.append({'type': 'anchor', 'jogo': f"{res['home']} vs {res['away']}", 'selection': f"{name} Over {l} Escanteios", 'prob': p, 'odd': odd})
            # Cartões
            p_card = probs['cards'][loc].get('Over 1.5', 0)
            if p_card >= 80:
                odd = get_fair_odd(p_card)
                if 1.22 <= odd <= 1.45:
                    anchors.append({'type': 'anchor', 'jogo': f"{res['home']} vs {res['away']}", 'selection': f"{name} Over 1.5 Cartões", 'prob': p_card, 'odd': odd})

        # 2. Fusões (Criar Aposta) - Odds 1.55 a 2.10
        # Regra: Mandante Canto > 3.5 E Jogo Cartão > 1.5
        corn_prob = probs['corners']['home'].get('Over 3.5', 0)
        card_prob = probs['cards']['total'].get('Over 1.5', 0)
        
        if corn_prob >= 75 and card_prob >= 75:
            p_comb = (corn_prob/100 * card_prob/100 * 0.90) * 100
            odd_comb = get_fair_odd(p_comb)
            
            if 1.55 <= odd_comb <= 2.10:
                fusions.append({
                    'type': 'fusion', 'jogo': f"{res['home']} vs {res['away']}",
                    'team': res['home'],
                    'mercados': [f"{res['home']} Over 3.5 Escanteios", "Total Jogo Over 1.5 Cartões"],
                    'prob_combined': p_comb, 'odd': odd_comb
                })

    # Montagem do Bilhete
    ticket = []
    curr_odd = 1.0
    used_games = set()
    
    anchors.sort(key=lambda x: x['prob'], reverse=True)
    fusions.sort(key=lambda x: x['prob_combined'], reverse=True)
    
    # Pega 2 Âncoras
    for a in anchors:
        if len(ticket) >= 2: break
        if a['jogo'] not in used_games:
            ticket.append(a)
            curr_odd *= a['odd']
            used_games.add(a['jogo'])
            
    # Pega Fusões até Odd ~4.60
    for f in fusions:
        if len(ticket) >= 6: break
        if f['jogo'] not in used_games:
            if curr_odd * f['odd'] <= 6.50:
                ticket.append(f)
                curr_odd *= f['odd']
                used_games.add(f['jogo'])
                
    # Completa com âncoras se precisar
    if curr_odd < 4.0:
        for a in anchors:
            if len(ticket) >= 6: break
            if a['jogo'] not in used_games:
                ticket.append(a)
                curr_odd *= a['odd']
                used_games.add(a['jogo'])

    return {'ticket': ticket, 'total_odd': round(curr_odd, 2), 'num_selections': len(ticket)}

def generate_hedges_for_user_ticket(ticket: List[Dict], stats: Dict, refs: Dict, all_dfs: Dict) -> Dict:
    """Gera Hedge para os jogos do bilhete salvo"""
    principal, hedge1, hedge2 = [], [], []
    processed = set()
    
    for item in ticket:
        try:
            parts = item['jogo'].split(' vs ')
            h, a = parts[0], parts[1]
        except: continue
        
        if f"{h}vs{a}" in processed: continue
        
        res = calcular_jogo_v23(h, a, stats, None, refs, all_dfs)
        if 'error' in res: continue
        probs = get_detailed_probs(res)
        
        # Principal (Cópia)
        sel_txt = item.get('selection', 'Criar Aposta')
        if item.get('type') == 'fusion': sel_txt = " + ".join(item['mercados'])
        principal.append({'jogo': item['jogo'], 'selecao': sel_txt, 'odd': item['odd']})
        
        # Hedge 1 (Safety - Resultado)
        mc = res['monte_carlo']
        sel1 = f"DC {h} ou Empate" if mc['h'] > mc['a'] else f"DC {a} ou Empate"
        odd1 = get_fair_odd(probs['chance']['1X'] if mc['h'] > mc['a'] else probs['chance']['X2'])
        hedge1.append({'jogo': item['jogo'], 'selecao': sel1, 'odd': odd1})
        
        # Hedge 2 (Caos - Cartões)
        hedge2.append({'jogo': item['jogo'], 'selecao': "Total Over 3.5 Cartões", 'odd': get_fair_odd(probs['cards']['total']['Over 3.5'])})
        
        processed.add(f"{h}vs{a}")
        
    return {
        'principal': {'itens': principal, 'odd': round(np.prod([x['odd'] for x in principal]), 2)},
        'hedge1': {'itens': hedge1, 'odd': round(np.prod([x['odd'] for x in hedge1]), 2)},
        'hedge2': {'itens': hedge2, 'odd': round(np.prod([x['odd'] for x in hedge2]), 2)}
    }

# ═══════════════════════════════════════════════════════════════════════════
# UI PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════

def main():
    if st.session_state.theme == 'dark':
        st.markdown("<style>.stApp {background-color: #0E1117; color: #FAFAFA;}</style>", unsafe_allow_html=True)
        
    st.sidebar.title("🎛️ Painel V23.2")
    
    with st.spinner("Carregando bases..."):
        stats = learn_stats_v23()
        refs = load_referees_v23()
        calendar = load_calendar_safe()
        all_dfs = load_all_dataframes()
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📅 Calendário", "🔍 Simulação", "🎯 Scanner Smart", "🛡️ Sistema Hedges", "📊 Radares", "📈 Dashboard"])
    
    with tab1:
        if not calendar.empty: st.dataframe(calendar, use_container_width=True)
        else: st.warning("Calendário vazio")
        
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
        st.header("🎯 Scanner V23 - Smart Ticket")
        if not calendar.empty:
            dates = sorted(calendar['DtObj'].dt.strftime('%d/%m/%Y').unique())
            sel_date = st.selectbox("Data:", dates, key="sd")
            
            if st.button("🚀 GERAR BILHETE E SALVAR", type="primary"):
                res = generate_smart_ticket_v23(calendar, stats, refs, all_dfs, sel_date)
                if res['ticket']:
                    st.session_state.current_ticket = res['ticket']
                    st.success("Bilhete Gerado e Salvo!")
                else: st.warning("Sem oportunidades.")
            
            if st.session_state.current_ticket:
                st.markdown("### 🎫 Bilhete Ativo (Na Memória)")
                for item in st.session_state.current_ticket:
                    desc = item.get('selection')
                    if item['type'] == 'fusion': desc = " + ".join(item['mercados'])
                    st.write(f"✅ {item['jogo']} | {desc} (@{item['odd']})")
                st.info("👉 Vá para a aba 'Sistema Hedges' para proteger este bilhete.")

    with tab4:
        st.header("🛡️ Sistema de Proteção (Hedges)")
        if not st.session_state.current_ticket:
            st.warning("⚠️ Nenhum bilhete na memória. Gere no Scanner (Aba 3).")
        else:
            if st.button("🛡️ CALCULAR HEDGES", type="primary"):
                hedges = generate_hedges_for_user_ticket(st.session_state.current_ticket, stats, refs, all_dfs)
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.subheader("📋 Principal")
                    st.metric("Odd Total", f"@{hedges['principal']['odd']}")
                    for i in hedges['principal']['itens']: st.caption(f"{i['jogo']} - {i['selecao']}")
                with c2:
                    st.subheader("🛡️ Safety")
                    st.metric("Odd Total", f"@{hedges['hedge1']['odd']}")
                    for i in hedges['hedge1']['itens']: st.caption(f"{i['jogo']} - {i['selecao']}")
                with c3:
                    st.subheader("🔄 Caos")
                    st.metric("Odd Total", f"@{hedges['hedge2']['odd']}")
                    for i in hedges['hedge2']['itens']: st.caption(f"{i['jogo']} - {i['selecao']}")

    with tab5:
         if not calendar.empty:
            dates = sorted(calendar['DtObj'].dt.strftime('%d/%m/%Y').unique())
            sel = st.selectbox("Data Radar:", dates)
            if st.button("Escanear"):
                res = scan_day_for_radars(calendar, stats, refs, all_dfs, sel)
                st.write(res)
    
    with tab6:
        st.markdown("### 📊 Histórico")
        if st.session_state.bet_history:
            st.dataframe(pd.DataFrame(st.session_state.bet_history))
        else: st.info("Sem apostas registradas.")

if __name__ == "__main__":
    main()
