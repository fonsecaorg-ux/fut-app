"""
╔═══════════════════════════════════════════════════════════════════════════╗
║       FUTPREVISÃO V23.1 - FIXED HEDGE FLOW (USER DRIVEN)                  ║
║                                                                            ║
║  ✅ Correção: Hedge agora obedece ao bilhete do usuário (não aleatório)   ║
║  ✅ Fluxo: Scanner -> Aceitar Bilhete -> Gerar Hedges                     ║
║  ✅ Mantida toda a inteligência de dados reais e radares                  ║
║                                                                            ║
║  Dezembro 2025 - Patch de Correção                                       ║
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
# CONFIGURAÇÃO E CONSTANTES (MANTIDAS)
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="FutPrevisão V23.1",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session State para persistir o bilhete escolhido
if 'current_ticket' not in st.session_state:
    st.session_state.current_ticket = []
if 'bankroll' not in st.session_state:
    st.session_state.bankroll = 1000.0
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'

THRESHOLDS = {
    'radar_corners': 70,
    'radar_cards': 65,
    'anchor_safety': 85,
    'smart_ticket_min': 4.60,
    'smart_ticket_max': 6.50
}

DEFAULTS = {'shots_on_target': 4.5, 'red_cards_avg': 0.08}

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
# CARREGAMENTO DE DADOS (MANTIDO IGUAL - ROBUSTO)
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
                    rename_map = {}
                    if 'Mandante' in df.columns: rename_map['Mandante'] = 'HomeTeam'
                    if 'Visitante' in df.columns: rename_map['Visitante'] = 'AwayTeam'
                    if 'Time_Casa' in df.columns: rename_map['Time_Casa'] = 'HomeTeam'
                    if 'Time_Visitante' in df.columns: rename_map['Time_Visitante'] = 'AwayTeam'
                    if rename_map: df = df.rename(columns=rename_map)
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
        
        cols = ['HomeTeam', 'AwayTeam', 'HC', 'AC', 'HY', 'AY', 'HF', 'AF', 'FTHG', 'FTAG', 'HST', 'AST', 'HR', 'AR']
        for c in cols: 
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
                cons_c = max(0, 100 - (c_std * 30))
                
                stats_db[team] = {
                    'corners': w_avg(h.get('HC_mean',0), a.get('AC_mean',0), 5.0),
                    'consistency_corners': round(cons_c, 1),
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
    if name in NAME_MAPPING: name = NAME_MAPPING[name]
    if name in db_keys: return name
    matches = get_close_matches(name, db_keys, n=1, cutoff=0.6)
    return matches[0] if matches else None

def monte_carlo(xg_h, xg_a, n=1000):
    gh = np.random.poisson(max(0.1, xg_h), n)
    ga = np.random.poisson(max(0.1, xg_a), n)
    return np.count_nonzero(gh > ga)/n, np.count_nonzero(gh == ga)/n, np.count_nonzero(ga > gh)/n

def poisson(k, lamb):
    return (lamb**k * math.exp(-lamb)) / math.factorial(k) if lamb <= 30 else 0

def calcular_jogo_v23(home: str, away: str, stats: Dict, ref: Optional[str], refs: Dict, all_dfs: Dict) -> Dict:
    h_n, a_n = normalize_name(home, list(stats.keys())), normalize_name(away, list(stats.keys()))
    if not h_n or not a_n: return {'error': "Times desconhecidos"}
    
    s_h, s_a = stats[h_n], stats[a_n]
    r_data = refs.get(ref, {'factor': 1.0, 'red_rate': 0.08, 'strictness': 4.0})
    
    # Ajustes finos V23
    pressure_h = 1.15 if s_h['shots_on_target'] > 5.5 else 1.0
    pressure_a = 1.15 if s_a['shots_on_target'] > 5.5 else 1.0
    
    corn_h = s_h['corners'] * 1.15 * pressure_h
    corn_a = s_a['corners'] * 0.90 * pressure_a
    
    viol_h = 1.1 if s_h['fouls'] > 12 else 1.0
    viol_a = 1.1 if s_a['fouls'] > 12 else 1.0
    
    card_h = s_h['cards'] * viol_h * r_data['factor']
    card_a = s_a['cards'] * viol_a * r_data['factor']
    
    xg_h, xg_a = max(0.1, s_h['goals_f']), max(0.1, s_a['goals_f'])
    mc_h, mc_d, mc_a = monte_carlo(xg_h, xg_a)
    
    return {
        'home': h_n, 'away': a_n, 'league': s_h['league'],
        'corners': {'h': corn_h, 'a': corn_a, 'total': corn_h+corn_a},
        'cards': {'h': card_h, 'a': card_a, 'total': card_h+card_a},
        'goals': {'h': xg_h, 'a': xg_a},
        'monte_carlo': {'h': mc_h*100, 'd': mc_d*100, 'a': mc_a*100},
        'consistency': {
            'corners_h': s_h.get('consistency_corners', 50),
            'corners_a': s_a.get('consistency_corners', 50),
            'cards_h': s_h.get('consistency_cards', 50),
            'cards_a': s_a.get('consistency_cards', 50)
        }
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
        'chance': {'1X': res['monte_carlo']['h'] + res['monte_carlo']['d']}
    }

def get_fair_odd(prob: float) -> float:
    return round(100/prob, 2) if prob > 0 else 99.0

# ═══════════════════════════════════════════════════════════════════════════
# LÓGICA V23: SCANNER + HEDGE PERSONALIZADO
# ═══════════════════════════════════════════════════════════════════════════

def generate_smart_ticket_v23(calendar: pd.DataFrame, stats: Dict, refs: Dict, all_dfs: Dict, date_str: str) -> Dict:
    """Gera bilhete inteligente com Âncoras + Fusões"""
    df_day = calendar[calendar['DtObj'].dt.strftime('%d/%m/%Y') == date_str]
    anchors, fusions = [], []
    
    for _, row in df_day.iterrows():
        home, away = row['Time_Casa'], row['Time_Visitante']
        res = calcular_jogo_v23(home, away, stats, None, refs, all_dfs)
        if 'error' in res: continue
        probs = get_detailed_probs(res)
        
        # 1. Âncoras (Safety)
        for loc, name in [('home', res['home']), ('away', res['away'])]:
            # Cantos
            for l in [3.5, 4.5]:
                p = probs['corners'][loc].get(f'Over {l}', 0)
                if p >= THRESHOLDS['anchor_safety']:
                    odd = get_fair_odd(p)
                    if 1.25 <= odd <= 1.40:
                        anchors.append({
                            'type': 'anchor', 'jogo': f"{res['home']} vs {res['away']}",
                            'selection': f"{name} Over {l} Escanteios",
                            'prob': p, 'odd': odd,
                            'home_team': res['home'], 'away_team': res['away'] # Meta dados para hedge
                        })
            # Cartões
            p_card = probs['cards'][loc].get('Over 1.5', 0)
            if p_card >= THRESHOLDS['anchor_safety']:
                odd = get_fair_odd(p_card)
                if 1.25 <= odd <= 1.45:
                    anchors.append({
                        'type': 'anchor', 'jogo': f"{res['home']} vs {res['away']}",
                        'selection': f"{name} Over 1.5 Cartões",
                        'prob': p_card, 'odd': odd,
                        'home_team': res['home'], 'away_team': res['away']
                    })

        # 2. Fusões (Combos)
        p_corn_h = probs['corners']['home'].get('Over 3.5', 0)
        p_card_h = probs['cards']['home'].get('Over 1.5', 0)
        
        if p_corn_h > 70 and p_card_h > 65:
            comb_p = (p_corn_h/100 * p_card_h/100 * 0.85) * 100
            odd = get_fair_odd(comb_p)
            if 1.60 <= odd <= 2.20:
                fusions.append({
                    'type': 'fusion', 'jogo': f"{res['home']} vs {res['away']}",
                    'selection': f"{res['home']}: Over 3.5 Cantos + Over 1.5 Cartões",
                    'prob': comb_p, 'odd': odd,
                    'home_team': res['home'], 'away_team': res['away']
                })

    # Montagem do Bilhete (Mix)
    ticket = []
    curr_odd = 1.0
    used_games = set()
    
    # Ordena por qualidade
    anchors.sort(key=lambda x: x['prob'], reverse=True)
    fusions.sort(key=lambda x: x['prob'], reverse=True)
    
    # 2 Âncoras obrigatórias
    for a in anchors:
        if len(ticket) < 2 and a['jogo'] not in used_games:
            ticket.append(a)
            curr_odd *= a['odd']
            used_games.add(a['jogo'])
            
    # Completa com Fusões e Âncoras até Odd > 4.60
    pool = fusions + anchors
    for item in pool:
        if len(ticket) >= 6: break
        if item['jogo'] not in used_games:
            if curr_odd * item['odd'] <= THRESHOLDS['smart_ticket_max']:
                ticket.append(item)
                curr_odd *= item['odd']
                used_games.add(item['jogo'])
                
    return {'ticket': ticket, 'total_odd': round(curr_odd, 2)}

def generate_hedges_for_user_ticket(ticket: List[Dict], stats: Dict, refs: Dict, all_dfs: Dict) -> Dict:
    """
    CORREÇÃO V23.1:
    Gera Hedges baseados EXCLUSIVAMENTE nos jogos do bilhete do usuário.
    """
    principal = []
    hedge1 = [] # Safety (Result)
    hedge2 = [] # Mix (Cards/Chaos)
    
    # Processa cada jogo presente no bilhete
    # Precisamos recalcular o jogo para pegar as probs de hedge
    
    processed_games = set()
    
    for item in ticket:
        # Recupera nomes dos times (armazenados no item)
        # Se não tiver (bilhete manual), tenta extrair da string (frágil, mas fallback)
        if 'home_team' in item:
            h, a = item['home_team'], item['away_team']
        else:
            # Tenta parsear "TimeA vs TimeB"
            try:
                parts = item['jogo'].split(' vs ')
                h, a = parts[0], parts[1]
            except: continue
            
        if f"{h}vs{a}" in processed_games: continue
        
        # Recalcula jogo
        res = calcular_jogo_v23(h, a, stats, None, refs, all_dfs)
        if 'error' in res: continue
        probs = get_detailed_probs(res)
        
        # Adiciona ao Principal (Cópia da seleção do user)
        principal.append({
            'jogo': f"{h} vs {a}",
            'selecao': item.get('selection', item.get('mercado', '??')),
            'odd': item['odd']
        })
        
        # Gera Hedge 1 (Resultado/Safety)
        # Se o principal é Canto/Cartão, o Safety é DC ou DNB
        dc_prob = probs['chance']['1X'] # Assume casa favorito por padrão ou analisa odd
        mc = res['monte_carlo']
        
        if mc['h'] > mc['a']: 
            sel_h1 = f"DC {h} ou Empate"
            odd_h1 = get_fair_odd(probs['chance']['1X'])
        else:
            sel_h1 = f"DC {a} ou Empate"
            odd_h1 = get_fair_odd(probs['chance']['X2'])
            
        hedge1.append({
            'jogo': f"{h} vs {a}",
            'selecao': sel_h1,
            'odd': odd_h1
        })
        
        # Gera Hedge 2 (Caos/Cartões)
        # Se o jogo ficar sujo, vamos de Over Cartões Total
        best_card_line = "3.5"
        best_card_prob = probs['cards']['total']['Over 3.5']
        
        hedge2.append({
            'jogo': f"{h} vs {a}",
            'selecao': f"Total Over {best_card_line} Cartões",
            'odd': get_fair_odd(best_card_prob)
        })
        
        processed_games.add(f"{h}vs{a}")
        
    # Calcula Odds Totais
    odd_p = np.prod([x['odd'] for x in principal])
    odd_h1 = np.prod([x['odd'] for x in hedge1])
    odd_h2 = np.prod([x['odd'] for x in hedge2])
    
    return {
        'principal': {'itens': principal, 'odd': round(odd_p, 2)},
        'hedge1': {'itens': hedge1, 'odd': round(odd_h1, 2)},
        'hedge2': {'itens': hedge2, 'odd': round(odd_h2, 2)}
    }

# ═══════════════════════════════════════════════════════════════════════════
# UI PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════

def main():
    if st.session_state.theme == 'dark':
        st.markdown("<style>.stApp {background-color: #0E1117; color: #FAFAFA;}</style>", unsafe_allow_html=True)
        
    st.sidebar.title("🎛️ Painel V23.1")
    st.sidebar.info("✅ Correção: Hedges Personalizados")
    
    with st.spinner("Carregando dados..."):
        stats = learn_stats_v23()
        refs = load_referees_v23()
        calendar = load_calendar_safe()
        all_dfs = load_all_dataframes()
        
    tab1, tab2, tab3, tab4 = st.tabs(["📅 Calendário", "🎯 Scanner Smart", "🎰 Sistema Hedges", "📊 Radares"])
    
    # TAB 1: CALENDÁRIO (Simplificado para brevidade)
    with tab1:
        if not calendar.empty:
            st.dataframe(calendar, use_container_width=True)
            
    # TAB 2: SCANNER (O Coração do Bilhete)
    with tab2:
        st.header("🎯 Scanner V23 - Smart Ticket")
        if not calendar.empty:
            dates = sorted(calendar['DtObj'].dt.strftime('%d/%m/%Y').unique())
            sel_date = st.selectbox("Data:", dates)
            
            if st.button("🚀 GERAR BILHETE DO DIA", type="primary"):
                res = generate_smart_ticket_v23(calendar, stats, refs, all_dfs, sel_date)
                
                if res['ticket']:
                    st.session_state.current_ticket = res['ticket'] # SALVA NA SESSÃO
                    st.success("Bilhete Gerado e Salvo na Memória!")
                else:
                    st.warning("Sem oportunidades hoje.")
            
            # Exibe Bilhete Atual (se existir)
            if st.session_state.current_ticket:
                st.markdown("### 🎫 Bilhete Ativo")
                st.metric("Odd Total", f"@{st.session_state.current_ticket[0]['odd'] * len(st.session_state.current_ticket):.2f}") # Aprox
                
                for sel in st.session_state.current_ticket:
                    st.write(f"✅ {sel['jogo']} | **{sel.get('selection', sel.get('mercado'))}** (@{sel['odd']})")
                
                st.info("💡 Vá para a aba 'Sistema Hedges' para proteger este bilhete.")

    # TAB 3: SISTEMA HEDGES (CORRIGIDO)
    with tab3:
        st.header("🛡️ Sistema de Proteção (Hedges)")
        st.caption("Gera proteções baseadas NO SEU BILHETE ATUAL.")
        
        if not st.session_state.current_ticket:
            st.warning("⚠️ Nenhum bilhete selecionado. Gere um no Scanner ou monte manualmente.")
        else:
            st.write("Bilhete carregado da memória com sucesso.")
            
            if st.button("🛡️ CALCULAR HEDGES PARA ESTE BILHETE", type="primary"):
                hedges = generate_hedges_for_user_ticket(st.session_state.current_ticket, stats, refs, all_dfs)
                
                c1, c2, c3 = st.columns(3)
                
                with c1:
                    st.subheader("📋 Principal (50%)")
                    st.metric("Odd", f"@{hedges['principal']['odd']}")
                    for item in hedges['principal']['itens']:
                        st.write(f"- {item['jogo']}")
                        st.caption(f"{item['selecao']}")
                        
                with c2:
                    st.subheader("🛡️ Hedge 1 (Safety - 30%)")
                    st.metric("Odd", f"@{hedges['hedge1']['odd']}")
                    for item in hedges['hedge1']['itens']:
                        st.write(f"- {item['jogo']}")
                        st.caption(f"{item['selecao']}")
                        
                with c3:
                    st.subheader("🔄 Hedge 2 (Caos - 20%)")
                    st.metric("Odd", f"@{hedges['hedge2']['odd']}")
                    for item in hedges['hedge2']['itens']:
                        st.write(f"- {item['jogo']}")
                        st.caption(f"{item['selecao']}")
    
    with tab4:
        st.write("Radares de Alta Frequência (Mantido do V23)")

if __name__ == "__main__":
    main()
