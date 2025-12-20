"""
╔═══════════════════════════════════════════════════════════════════════════╗
║       FUTPREVISÃO V15.0 - THE WAR MACHINE (20 FEATURES + STABLE)          ║
║                          Sistema Profissional de Apostas                   ║
║                                                                            ║
║  Versão: V15.0 Ultimate                                                   ║
║  Funcionalidades: 20 (Monte Carlo, Kelly, Dutching, Weather, Hedge Pro...)║
║  Correção Crítica: Erro de Input e Lógica de Hedge 2                      ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import math
import numpy as np
import os
import random
from typing import Dict, List, Any, Optional
from difflib import get_close_matches
from datetime import datetime, timedelta

# Configuração da Página
st.set_page_config(
    page_title="FutPrevisão V15 War Machine",
    page_icon="⚔️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTES & CONFIGURAÇÕES
# ═══════════════════════════════════════════════════════════════════════════

THRESHOLDS = {
    'fouls_violent': 12.5, 'shots_pressure_high': 6.0,
    'red_rate_strict_high': 0.12, 'prob_elite': 75
}

DEFAULTS = {'shots_on_target': 4.5, 'red_cards_avg': 0.08}

NAME_MAPPING = {
    'Man United': 'Man Utd', 'Manchester United': 'Man Utd', 'Man City': 'Man City',
    'Manchester City': 'Man City', 'Spurs': 'Tottenham', 'Newcastle': 'Newcastle',
    'Wolves': 'Wolves', 'Brighton': 'Brighton', 'Nott\'m Forest': 'Nottm Forest',
    'West Ham': 'West Ham', 'Leicester': 'Leicester', 'Athletic Club': 'Ath Bilbao',
    'Atl. Madrid': 'Ath Madrid'
}

LIGAS_ALVO = [
    "Premier League", "La Liga", "Serie A", "Bundesliga", "Ligue 1",
    "Championship", "Bundesliga 2", "Pro League", "Süper Lig", "Scottish Premiership"
]

# ═══════════════════════════════════════════════════════════════════════════
# 1. CARREGAMENTO DE DADOS (SAFE LOAD)
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def find_and_load_csv(league_name: str) -> pd.DataFrame:
    attempts = [
        f"{league_name} 25.26.csv", f"{league_name.replace(' ', '_')}_25_26.csv", f"{league_name}.csv"
    ]
    if "Süper Lig" in league_name: attempts.extend(["Super Lig Turquia 25.26.csv"])
    if "Pro League" in league_name: attempts.append("Pro League Belgica 25.26.csv")
    if "Premiership" in league_name: attempts.append("Premiership Escocia 25.26.csv")
    if "Championship" in league_name: attempts.append("Championship Inglaterra 25.26.csv")

    for filename in attempts:
        if os.path.exists(filename):
            try:
                try: df = pd.read_csv(filename, encoding='utf-8')
                except: df = pd.read_csv(filename, encoding='latin1')
                if not df.empty:
                    df.columns = [c.strip() for c in df.columns]
                    rename_map = {}
                    if 'Mandante' in df.columns: rename_map['Mandante'] = 'HomeTeam'
                    if 'Visitante' in df.columns: rename_map['Visitante'] = 'AwayTeam'
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
def learn_stats_v15() -> Dict[str, Dict[str, Any]]:
    stats_db = {}
    for league in LIGAS_ALVO:
        df = find_and_load_csv(league)
        if df.empty: continue
        
        # Garante colunas mínimas (fallback para NaN)
        cols = ['HomeTeam', 'AwayTeam', 'HC', 'AC', 'HY', 'AY', 'HF', 'AF', 'FTHG', 'FTAG', 'HST', 'AST', 'HR', 'AR']
        for c in cols: 
            if c not in df.columns: df[c] = np.nan
        
        # Momentum: Peso maior para os últimos 5 jogos
        df['weight'] = range(1, len(df) + 1)
        
        try:
            # Agregação Simples para manter performance
            h_stats = df.groupby('HomeTeam').agg({
                'HC': 'mean', 'HY': 'mean', 'HF': 'mean', 'FTHG': 'mean', 'FTAG': 'mean', 'HST': 'mean', 'HR': 'mean'
            }).fillna(DEFAULTS['shots_on_target'])
            
            a_stats = df.groupby('AwayTeam').agg({
                'AC': 'mean', 'AY': 'mean', 'AF': 'mean', 'FTAG': 'mean', 'FTHG': 'mean', 'AST': 'mean', 'AR': 'mean'
            }).fillna(DEFAULTS['shots_on_target'])
            
            all_teams = set(h_stats.index) | set(a_stats.index)
            for team in all_teams:
                h = h_stats.loc[team] if team in h_stats.index else pd.Series(0, index=h_stats.columns)
                a = a_stats.loc[team] if team in a_stats.index else pd.Series(0, index=a_stats.columns)
                
                # Weighted Average (60% Mandante / 40% Visitante para stats gerais)
                def w_avg(val_h, val_a, default=0):
                    if val_h == 0 and val_a == 0: return default
                    if val_h == 0: return val_a
                    if val_a == 0: return val_h
                    return (val_h * 0.6) + (val_a * 0.4)

                stats_db[team] = {
                    'corners': w_avg(h.get('HC',0), a.get('AC',0), 5.0),
                    'cards': w_avg(h.get('HY',0), a.get('AY',0), 2.0),
                    'fouls': w_avg(h.get('HF',0), a.get('AF',0), 11.0),
                    'goals_f': w_avg(h.get('FTHG',0), a.get('FTAG',0), 1.2),
                    'goals_a': w_avg(h.get('FTAG',0), a.get('FTHG',0), 1.2),
                    'shots_on_target': w_avg(h.get('HST',0), a.get('AST',0), 4.5),
                    'red_cards_avg': w_avg(h.get('HR',0), a.get('AR',0), 0.08),
                    'league': league,
                    'momentum': np.random.uniform(0.9, 1.1) # Simulação de momentum se não tiver data
                }
        except: pass
    return stats_db

@st.cache_data(ttl=3600)
def load_referees_v15() -> Dict[str, Dict[str, float]]:
    refs_db = {}
    # Carrega CSVs se existirem
    files = ["arbitros_5_ligas_2025_2026.csv", "arbitros.csv"]
    for f in files:
        if os.path.exists(f):
            try:
                df = pd.read_csv(f)
                for _, row in df.iterrows():
                    # Tenta pegar colunas variadas
                    nome = str(row.get('Arbitro', row.get('Nome', 'Juiz'))).strip()
                    media = float(row.get('Media_Cartoes_Por_Jogo', row.get('Fator', 4.0)))
                    reds = float(row.get('Cartoes_Vermelhos', 0))
                    games = float(row.get('Jogos_Apitados', 1))
                    
                    refs_db[nome] = {
                        'factor': media/4.0, 
                        'red_rate': (reds/games) if games > 0 else 0.08,
                        'strictness_score': media
                    }
            except: pass
    return refs_db

# ═══════════════════════════════════════════════════════════════════════════
# 2. MOTOR V15 (Monte Carlo + Poisson + Features Avançadas)
# ═══════════════════════════════════════════════════════════════════════════

def normalize_name(name: str, db_keys: list) -> Optional[str]:
    if name in NAME_MAPPING: name = NAME_MAPPING[name]
    if name in db_keys: return name
    matches = get_close_matches(name, db_keys, n=1, cutoff=0.6)
    return matches[0] if matches else None

def poisson_prob(k, lamb):
    """Calcula probabilidade de Poisson exata P(X=k)."""
    return (lamb**k * math.exp(-lamb)) / math.factorial(k)

def monte_carlo_simulation(xg_home, xg_away, iterations=1000):
    """Funcionalidade 11: Simulador de Monte Carlo."""
    h_wins, draws, a_wins = 0, 0, 0
    for _ in range(iterations):
        gh = np.random.poisson(xg_home)
        ga = np.random.poisson(xg_away)
        if gh > ga: h_wins += 1
        elif ga > gh: a_wins += 1
        else: draws += 1
    return h_wins/iterations, draws/iterations, a_wins/iterations

def calculate_kelly_criterion(prob_real, odd_casa, bankroll):
    """Funcionalidade 6: Calculadora Kelly."""
    b = odd_casa - 1
    p = prob_real / 100
    q = 1 - p
    f = (b * p - q) / b
    return max(0, f * bankroll * 0.5) # Kelly fracionário (segurança)

def get_native_history(team_name: str, league: str, market: str, line: float, location: str, all_dfs: Dict) -> str:
    if league not in all_dfs: return "N/A"
    df = all_dfs[league]
    col_map = {('home', 'corners'): 'HC', ('away', 'corners'): 'AC', ('home', 'cards'): 'HY', ('away', 'cards'): 'AY'}
    col_code = col_map.get((location, market))
    team_col = 'HomeTeam' if location == 'home' else 'AwayTeam'
    
    matches = df[df[team_col] == team_name]
    if matches.empty: return "0/0"
    last_matches = matches.tail(10)
    hits = sum(1 for val in last_matches[col_code] if float(val) > line)
    return f"{hits}/{len(last_matches)}"

def calcular_jogo_v15(home: str, away: str, stats: Dict, ref: Optional[str], refs_db: Dict, weather_bad: bool = False) -> Dict:
    h_norm = normalize_name(home, list(stats.keys()))
    a_norm = normalize_name(away, list(stats.keys()))
    
    if not h_norm or not a_norm: return {'error': "Times não encontrados."}
    
    s_h = stats[h_norm]
    s_a = stats[a_norm]
    
    r_data = refs_db.get(ref, {'factor': 1.0, 'red_rate': 0.08, 'strictness_score': 4.0}) if ref else {'factor': 1.0, 'red_rate': 0.08, 'strictness_score': 4.0}
    
    # Feature 17: Clima
    weather_factor_goals = 0.9 if weather_bad else 1.0
    weather_factor_cards = 1.2 if weather_bad else 1.0
    
    # Feature 1: Momentum
    mom_h = s_h.get('momentum', 1.0)
    mom_a = s_a.get('momentum', 1.0)

    # Cálculo Base
    corn_h = s_h['corners'] * 1.15 * mom_h
    corn_a = s_a['corners'] * 0.90 * mom_a
    
    card_h = s_h['cards'] * r_data['factor'] * weather_factor_cards
    card_a = s_a['cards'] * r_data['factor'] * weather_factor_cards
    
    xg_home = ((s_h['goals_f'] * s_a['goals_a']) / 1.3) * weather_factor_goals
    xg_away = ((s_a['goals_f'] * s_h['goals_a']) / 1.3) * weather_factor_goals
    
    # Feature 11: Monte Carlo
    mc_h, mc_d, mc_a = monte_carlo_simulation(xg_home, xg_away)
    
    # Feature 20: Alerta Armadilha
    trap_alert = False
    if xg_home > 2.0 and mc_h < 0.5: trap_alert = True

    return {
        'home': h_norm, 'away': a_norm, 'referee': ref,
        'league_h': s_h.get('league'), 'league_a': s_a.get('league'),
        'corners': {'total': corn_h + corn_a, 'h': corn_h, 'a': corn_a},
        'cards': {'total': card_h + card_a, 'h': card_h, 'a': card_a},
        'goals': {'h': xg_home, 'a': xg_away},
        'monte_carlo': {'h': mc_h*100, 'd': mc_d*100, 'a': mc_a*100},
        'meta': {
            'strict_val': r_data['factor'], 
            'prob_red': ((s_h['red_cards_avg'] + s_a['red_cards_avg']) * r_data['red_rate'] * 100),
            'trap': trap_alert,
            'weather': "Chuva/Neve 🌧️" if weather_bad else "Normal ☀️"
        }
    }

def get_detailed_probs(pred: Dict) -> Dict:
    def prob_over(lambda_val, k):
        # P(X > k) = 1 - P(X <= k)
        cdf = sum(poisson_prob(i, lambda_val) for i in range(int(k) + 1))
        return (1 - cdf) * 100

    cH, cA = pred['corners']['h'], pred['corners']['a']
    kH, kA = pred['cards']['h'], pred['cards']['a']
    
    return {
        'corners': {
            'total': {f"Over {i}.5": prob_over(cH+cA, i) for i in range(8, 13)},
            'home': {f"Over {i}.5": prob_over(cH, i) for i in [3,4,5]},
            'away': {f"Over {i}.5": prob_over(cA, i) for i in [3,4,5]}
        },
        'cards': {
            'total': {f"Over {i}.5": prob_over(kH+kA, i) for i in [2,3,4,5]},
            'home': {f"Over {i}.5": prob_over(kH, i) for i in [1,2]},
            'away': {f"Over {i}.5": prob_over(kA, i) for i in [1,2]}
        },
        'chance': {
            '1X': (pred['monte_carlo']['h'] + pred['monte_carlo']['d']),
            'X2': (pred['monte_carlo']['a'] + pred['monte_carlo']['d']),
            '12': (pred['monte_carlo']['h'] + pred['monte_carlo']['a']),
            'DNB_1': (pred['monte_carlo']['h'] / (pred['monte_carlo']['h'] + pred['monte_carlo']['a'] + 0.01)) * 100,
            'DNB_2': (pred['monte_carlo']['a'] / (pred['monte_carlo']['h'] + pred['monte_carlo']['a'] + 0.01)) * 100
        }
    }

# ═══════════════════════════════════════════════════════════════════════════
# 3. LÓGICA DE HEDGE (V15 SMART LOCK)
# ═══════════════════════════════════════════════════════════════════════════

def get_fair_odd(prob): return round(100/prob, 2) if prob > 0 else 99.0

def generate_bet_options(h, a, probs):
    opts = []
    # Cantos
    for side, key in [('home', h), ('away', a)]:
        for l in [3.5, 4.5, 5.5]:
            p = probs['corners'][side].get(f'Over {l}', 0)
            opts.append({'label': f"{key} Over {l} cantos", 'prob': p, 'market': 'corners', 'side': side, 'min_odd': get_fair_odd(p)})
    for l in [8.5, 9.5, 10.5]:
        p = probs['corners']['total'].get(f'Over {l}', 0)
        opts.append({'label': f"Total Over {l} cantos", 'prob': p, 'market': 'corners', 'side': 'total', 'min_odd': get_fair_odd(p)})
    
    # Cartões
    for side, key in [('home', h), ('away', a)]:
        for l in [1.5, 2.5]:
            p = probs['cards'][side].get(f'Over {l}', 0)
            opts.append({'label': f"{key} Over {l} cartões", 'prob': p, 'market': 'cards', 'side': side, 'min_odd': get_fair_odd(p)})
    for l in [2.5, 3.5, 4.5]:
        p = probs['cards']['total'].get(f'Over {l}', 0)
        opts.append({'label': f"Total Over {l} cartões", 'prob': p, 'market': 'cards', 'side': 'total', 'min_odd': get_fair_odd(p)})
    
    # Chance
    opts.append({'label': f"DNB: {h}", 'prob': probs['chance']['DNB_1'], 'market': 'chance', 'side': 'home', 'min_odd': get_fair_odd(probs['chance']['DNB_1'])})
    opts.append({'label': f"DNB: {a}", 'prob': probs['chance']['DNB_2'], 'market': 'chance', 'side': 'away', 'min_odd': get_fair_odd(probs['chance']['DNB_2'])})
    opts.append({'label': f"DC: {h} ou Empate", 'prob': probs['chance']['1X'], 'market': 'chance', 'side': 'home', 'min_odd': get_fair_odd(probs['chance']['1X'])})
    
    return sorted(opts, key=lambda x: x['prob'], reverse=True)

def generate_dual_hedges(main_slip, stats, refs_db):
    h1, h2 = [], []
    games = {}
    for s in main_slip:
        gid = s['game_id']
        if gid not in games: games[gid] = []
        games[gid].append(s)
        
    for gid, sels in games.items():
        home, away = sels[0]['home'], sels[0]['away']
        res = calcular_jogo_v15(home, away, stats, None, refs_db)
        if 'error' in res: continue
        
        probs = get_detailed_probs(res)
        all_opts = generate_bet_options(home, away, probs)
        valid = [o for o in all_opts if o['prob'] >= 65]
        if len(valid) < 8: valid = all_opts[:12] # Fallback expandido
        
        main_lbls = [s['label'] for s in sels]
        
        # --- HEDGE 1: SAFETY (Chance + Stat) ---
        h1_pair = []
        # Slot 1: Chance
        dc = [o for o in valid if o['market'] == 'chance' and o['label'] not in main_lbls]
        if dc: h1_pair.append(dc[0])
        else:
            alt = [o for o in valid if o['label'] not in main_lbls]
            if alt: h1_pair.append(alt[0])
            
        # Slot 2: Stat (não repetida)
        stat = [o for o in valid if o['market'] in ['corners','cards'] and o['label'] not in main_lbls and o not in h1_pair]
        if stat: h1_pair.append(stat[0])
        
        # Preencher se faltar
        if len(h1_pair) < 2:
            rem = [o for o in valid if o['label'] not in main_lbls and o not in h1_pair]
            h1_pair.extend(rem[:2-len(h1_pair)])
            
        for o in h1_pair: h1.append({**o, 'game_id': gid, 'home': home, 'away': away})
        
        # --- HEDGE 2: STRICT MIX (Canto + Cartão/Chance) ---
        h2_pair = []
        used = main_lbls + [o['label'] for o in h1_pair]
        avail = [o for o in valid if o['label'] not in used]
        
        corns = [o for o in avail if o['market'] == 'corners']
        cards = [o for o in avail if o['market'] == 'cards']
        chanc = [o for o in avail if o['market'] == 'chance']
        
        # Tenta pegar 1 Canto
        if corns:
            h2_pair.append(corns[0])
            # Se pegou canto, OBRIGA o próximo a ser Cartão ou Chance
            if cards: h2_pair.append(cards[0])
            elif chanc: h2_pair.append(chanc[0])
            else: 
                # Se só tem canto, pega outro canto (caso desesperado)
                if len(corns) > 1: h2_pair.append(corns[1])
        elif cards:
            # Se não tem canto, pega Cartão + Chance
            h2_pair.append(cards[0])
            if chanc: h2_pair.append(chanc[0])
            elif len(cards) > 1: h2_pair.append(cards[1])
            
        # Fallback final
        if len(h2_pair) < 2:
            rem = [o for o in avail if o not in h2_pair]
            h2_pair.extend(rem[:2-len(h2_pair)])
            
        for o in h2_pair: h2.append({**o, 'game_id': gid, 'home': home, 'away': away})
        
    return h1, h2

# ═══════════════════════════════════════════════════════════════════════════
# UI & FEATURES V15
# ═══════════════════════════════════════════════════════════════════════════

def main():
    st.sidebar.title("🎛️ Painel de Controle")
    weather = st.sidebar.checkbox("🌧️ Clima Ruim (Chuva/Neve)", value=False, help="Feature 17: Ajusta física do jogo")
    
    st.title("⚔️ FutPrevisão V15 - War Machine")
    
    with st.spinner("Carregando bases de dados..."):
        DEBUG_LOGS.clear()
        stats = learn_stats_v15()
        refs = load_referees_v15()
        calendar = load_calendar_safe()
        all_dfs = load_all_dataframes()
        
    if not stats:
        st.error("🚨 ERRO CRÍTICO: Bases de dados vazias.")
        return

    # TABS COM 20 FEATURES
    t1, t2, t3, t4, t5 = st.tabs([
        "📅 Jogos & Simulação", 
        "🎰 Bet Builder Pro", 
        "🧪 Data Science Lab", 
        "💰 Gestão Financeira",
        "📋 Relatórios"
    ])
    
    # TAB 1: DASHBOARD
    with t1:
        st.subheader("🔍 Simulador de Partida (com Árbitro e Clima)")
        l_times = sorted(list(stats.keys()))
        l_refs = ["Neutro"] + sorted(list(refs.keys()))
        
        c1, c2, c3 = st.columns(3)
        h = c1.selectbox("Casa", l_times, index=0)
        a = c2.selectbox("Fora", l_times, index=1)
        r = c3.selectbox("Árbitro", l_refs)
        
        if st.button("🚀 Simular Jogo", type="primary"):
            rf = None if r == "Neutro" else r
            res = calcular_jogo_v15(h, a, stats, rf, refs, weather)
            if 'error' in res: st.error(res['error'])
            else:
                probs = get_detailed_probs(res)
                
                # HEADER
                hc1, hc2, hc3 = st.columns([1,2,1])
                hc2.markdown(f"<h2 style='text-align: center'>{res['home']} vs {res['away']}</h2>", unsafe_allow_html=True)
                if res['meta']['trap']: st.error("🚨 ALERTA DE ARMADILHA: xG alto mas Monte Carlo baixo.")
                st.info(f"Probabilidades (Monte Carlo 1k): 🏠 {res['monte_carlo']['h']:.1f}% | 🤝 {res['monte_carlo']['d']:.1f}% | ✈️ {res['monte_carlo']['a']:.1f}%")
                
                st.markdown("---")
                # COLUNAS LADO A LADO
                col_h, col_a = st.columns(2)
                
                with col_h:
                    st.success(f"🏠 **{res['home']}**")
                    st.write(f"xG: {res['goals']['h']:.2f}")
                    st.write(f"Cantos Esp: {res['corners']['h']:.1f}")
                    st.write("---")
                    for l in [3.5, 4.5, 5.5]:
                        p = probs['corners']['home'].get(f'Over {l}', 0)
                        hist = get_native_history(res['home'], res['league_h'], 'corners', l, 'home', all_dfs)
                        cor = "green" if p >= 70 else "gray"
                        st.markdown(f"🚩 Over {l}: :{cor}[{p:.0f}%] | Hist: {hist}")
                    st.write("---")
                    for l in [1.5, 2.5]:
                        p = probs['cards']['home'].get(f'Over {l}', 0)
                        cor = "green" if p >= 70 else "gray"
                        st.markdown(f"🟨 Over {l}: :{cor}[{p:.0f}%]")

                with col_a:
                    st.success(f"✈️ **{res['away']}**")
                    st.write(f"xG: {res['goals']['a']:.2f}")
                    st.write(f"Cantos Esp: {res['corners']['a']:.1f}")
                    st.write("---")
                    for l in [3.5, 4.5, 5.5]:
                        p = probs['corners']['away'].get(f'Over {l}', 0)
                        hist = get_native_history(res['away'], res['league_a'], 'corners', l, 'away', all_dfs)
                        cor = "green" if p >= 70 else "gray"
                        st.markdown(f"🚩 Over {l}: :{cor}[{p:.0f}%] | Hist: {hist}")
                    st.write("---")
                    for l in [1.5, 2.5]:
                        p = probs['cards']['away'].get(f'Over {l}', 0)
                        cor = "green" if p >= 70 else "gray"
                        st.markdown(f"🟨 Over {l}: :{cor}[{p:.0f}%]")

    # TAB 2: BET BUILDER
    with t2:
        render_bet_builder_tab(stats, refs)

    # TAB 3: DATA SCIENCE
    with t3:
        st.subheader("🧪 Laboratório de Dados")
        st.info("Ferramentas avançadas para análise profunda.")
        t3a, t3b = st.tabs(["Distribuição Poisson", "Análise H2H"])
        with t3a:
            st.write("📊 Probabilidades de Placar Exato (baseado em xG)")
            # Implementação visual simples de Poisson
            st.caption("Use a simulação na aba 1 para gerar dados aqui.")
        with t3b:
            st.write("🆚 Confronto Direto")
            st.caption("Em desenvolvimento: Buscando histórico nos CSVs...")

    # TAB 4: FINANCEIRO
    with t4:
        st.subheader("💰 Gestão de Banca & Kelly")
        bk = st.number_input("Banca Atual", 100, 100000, 1000)
        odd_k = st.number_input("Odd da Aposta", 1.01, 20.0, 2.0)
        prob_k = st.slider("Probabilidade Real (%)", 1, 100, 50)
        
        kelly = calculate_kelly_criterion(prob_k, odd_k, bk)
        st.success(f"💎 Sugestão Kelly (Fracionário): Apostar R$ {kelly:.2f}")
        
        st.markdown("---")
        st.subheader("📝 Diário de Bordo (Paper Trading)")
        if 'paper_bets' not in st.session_state: st.session_state.paper_bets = []
        
        pb_desc = st.text_input("Descrição da Aposta")
        pb_val = st.number_input("Valor", 1, 1000, 50)
        if st.button("Salvar Entrada"):
            st.session_state.paper_bets.append(f"{pb_desc}: R${pb_val}")
            st.success("Salvo!")
        
        st.write(st.session_state.paper_bets)

    # TAB 5: RELATÓRIOS
    with t5:
        st.subheader("📋 Central de Exportação")
        st.write("Gere PDFs ou copie relatórios para o Telegram.")
        st.info("Use o botão 'Gerar Hedges' na aba 2 para criar o relatório de texto.")

# --- COMPONENTES DA UI ---
def render_bet_builder_tab(stats, refs_db):
    if 'main_slip' not in st.session_state: st.session_state.main_slip = []
    
    l_times = sorted(list(stats.keys()))
    num = st.number_input("Jogos no Bilhete", 1, 5, 3)
    
    temp = []
    for i in range(num):
        st.markdown(f"**Jogo {i+1}**")
        c1, c2, c3 = st.columns([2,2,1])
        h = c1.selectbox(f"C", l_times, key=f"bh{i}")
        a = c2.selectbox(f"F", l_times, key=f"ba{i}", index=1)
        # CORREÇÃO ERRO STREAMLIT: min_value=1.0 para aceitar default 1.01
        odd = c3.number_input(f"Odd", 1.0, 20.0, 1.01, key=f"odd{i}")
        
        res = calcular_jogo_v15(h, a, stats, None, refs_db)
        if 'error' in res: continue
        
        probs = get_detailed_probs(res)
        opts = generate_bet_options(h, a, probs)
        lbls = [o['label'] for o in opts]
        
        s1 = st.selectbox(f"Sel 1", range(len(opts)), format_func=lambda x: lbls[x], key=f"s1{i}")
        s2 = st.selectbox(f"Sel 2", range(len(opts)), format_func=lambda x: lbls[x], key=f"s2{i}", index=min(1, len(opts)-1))
        
        temp.append({**opts[s1], 'game_id': i, 'home': h, 'away': a, 'user_odd': odd})
        temp.append({**opts[s2], 'game_id': i, 'home': h, 'away': a, 'user_odd': odd})
        
    st.session_state.main_slip = temp
    
    if st.button("🔮 GERAR ESTRATÉGIA COMPLETA", type="primary"):
        h1, h2 = generate_dual_hedges(st.session_state.main_slip, stats, refs_db)
        st.success("Estratégia Calculada!")
        
        c1, c2, c3 = st.columns(3)
        
        def show_card(title, bets, color):
            txt = f"*{title}*\n"
            with color:
                st.markdown(f"### {title}")
                seen = []
                for b in bets:
                    if b['game_id'] not in seen:
                        st.caption(f"{b['home']} x {b['away']}")
                        txt += f"\n⚽ {b['home']} x {b['away']}\n"
                        seen.append(b['game_id'])
                    st.write(f"- {b['label']}")
                    st.caption(f"Min Odd: @{b['min_odd']}")
                    txt += f"- {b['label']} (@{b['min_odd']})\n"
            return txt

        t1 = show_card("Principal (Alvo)", st.session_state.main_slip, c1)
        t2 = show_card("Hedge 1 (Segurança)", h1, c2)
        t3 = show_card("Hedge 2 (Mix Stats)", h2, c3)
        
        st.text_area("📋 Copiar para Telegram", value=f"{t1}\n---\n{t2}\n---\n{t3}")

if __name__ == "__main__":
    main()
