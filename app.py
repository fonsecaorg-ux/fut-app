"""
╔═══════════════════════════════════════════════════════════════════════════╗
║       FUTPREVISÃO V15 - ULTIMATE PROFESSIONAL (EV+, COPY, STAKE, HEDGE)   ║
║                          Sistema de Análise de Apostas                     ║
║                                                                            ║
║  Versão: V15.0                                                            ║
║  Data: Dezembro 2025                                                      ║
║  Novidades:                                                               ║
║  1. Hedge com Trava Cruzada Rígida (Canto+Cartão Obrigatório).            ║
║  2. Calculadora EV+ (Valor Esperado).                                     ║
║  3. Gestão de Banca (Stake Calculator).                                   ║
║  4. Gerador de Texto para Copiar.                                         ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import math
import numpy as np
import os
from typing import Dict, List, Any, Optional
from difflib import get_close_matches
from datetime import datetime

# Configuração da Página
st.set_page_config(
    page_title="FutPrevisão V15 Pro",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTES & CONFIG
# ═══════════════════════════════════════════════════════════════════════════

THRESHOLDS = {
    'fouls_violent': 12.5, 'shots_pressure_high': 6.0, 'shots_pressure_medium': 4.5,
    'red_rate_strict_high': 0.12, 'red_rate_strict_medium': 0.08,
    'prob_elite': 75, 'prob_elite_cards': 70, 'prob_red_high': 12, 'prob_red_medium': 8
}

DEFAULTS = {'shots_on_target': 4.5, 'red_cards_avg': 0.08, 'red_rate_referee': 0.08}

NAME_MAPPING = {
    'Man United': 'Man Utd', 'Manchester United': 'Man Utd', 'Man City': 'Man City',
    'Manchester City': 'Man City', 'Spurs': 'Tottenham', 'Newcastle': 'Newcastle',
    'Wolves': 'Wolves', 'Brighton': 'Brighton', 'Nott\'m Forest': 'Nottm Forest',
    'Nottingham Forest': 'Nottm Forest', 'West Ham': 'West Ham', 'Leicester': 'Leicester',
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
# 1. CARREGAMENTO DE DADOS
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def find_and_load_csv(league_name: str) -> pd.DataFrame:
    attempts = [
        f"{league_name} 25.26.csv", f"{league_name.replace(' ', '_')}_25_26.csv", f"{league_name}.csv"
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
                    df.columns = [c.strip() for c in df.columns]
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
def learn_stats_v14() -> Dict[str, Dict[str, Any]]:
    stats_db = {}
    for league in LIGAS_ALVO:
        df = find_and_load_csv(league)
        if df.empty: continue
        
        cols_needed = ['HomeTeam', 'AwayTeam', 'HC', 'AC', 'HY', 'AY', 'HF', 'AF', 'FTHG', 'FTAG', 'HST', 'AST', 'HR', 'AR']
        for c in cols_needed:
            if c not in df.columns: df[c] = np.nan
        
        try:
            h_stats = df.groupby('HomeTeam').agg({
                'HC': 'mean', 'HY': 'mean', 'HF': 'mean', 'FTHG': 'mean', 'FTAG': 'mean', 
                'HST': 'mean', 'HR': 'mean'
            }).fillna(value={'HST': DEFAULTS['shots_on_target'], 'HR': DEFAULTS['red_cards_avg']})
            
            a_stats = df.groupby('AwayTeam').agg({
                'AC': 'mean', 'AY': 'mean', 'AF': 'mean', 'FTAG': 'mean', 'FTHG': 'mean', 
                'AST': 'mean', 'AR': 'mean'
            }).fillna(value={'AST': DEFAULTS['shots_on_target'], 'AR': DEFAULTS['red_cards_avg']})
            
            all_teams = set(h_stats.index) | set(a_stats.index)
            for team in all_teams:
                h = h_stats.loc[team] if team in h_stats.index else pd.Series(0, index=h_stats.columns)
                a = a_stats.loc[team] if team in a_stats.index else pd.Series(0, index=a_stats.columns)
                
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
                vermelhos = float(row.get('Cartoes_Vermelhos', 0))
                jogos = float(row.get('Jogos_Apitados', 1))
                media = float(row.get('Media_Cartoes_Por_Jogo', 4.0))
                red_rate = (vermelhos / jogos) if jogos > 0 else 0.08
                refs_db[nome] = {'factor': media/4.0, 'red_rate': red_rate}
        except: pass
    if os.path.exists("arbitros.csv"):
        try:
            df = pd.read_csv("arbitros.csv")
            for _, row in df.iterrows():
                nome = str(row['Nome']).strip()
                if nome not in refs_db:
                    refs_db[nome] = {'factor': float(row['Fator']), 'red_rate': 0.08}
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
        df = df.dropna(subset=['DtObj']).sort_values(by=['DtObj', 'Hora'])
        return df
    except: return pd.DataFrame()

# ═══════════════════════════════════════════════════════════════════════════
# 2. MOTOR DE CÁLCULO V14 (CAUSAL ENGINE)
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

    last_matches = matches.tail(10)
    hits = sum(1 for val in last_matches[col_code] if float(val) > line)
    return f"{hits}/{len(last_matches)}"

def calcular_jogo_v14(home: str, away: str, stats: Dict, ref: Optional[str], refs_db: Dict) -> Dict:
    h_norm = normalize_name(home, list(stats.keys()))
    a_norm = normalize_name(away, list(stats.keys()))
    
    if not h_norm or not a_norm: return {'error': "Times não encontrados."}
    
    s_h = stats[h_norm]
    s_a = stats[a_norm]
    
    r_data = refs_db.get(ref, {'factor': 1.0, 'red_rate': 0.08}) if ref else {'factor': 1.0, 'red_rate': 0.08}
        
    shots_h, shots_a = s_h['shots_on_target'], s_a['shots_on_target']
    p_h = 1.20 if shots_h > 6.0 else 1.10 if shots_h > 4.5 else 1.0
    p_a = 1.20 if shots_a > 6.0 else 1.10 if shots_a > 4.5 else 1.0
    
    corn_h = s_h['corners'] * 1.15 * p_h
    corn_a = s_a['corners'] * 0.90 * p_a
    
    rr = r_data['red_rate']
    strict = 1.15 if rr > 0.12 else 1.08 if rr > 0.08 else 1.0
    
    viol_h = 1.0 if s_h['fouls'] > 12.5 else 0.85
    viol_a = 1.0 if s_a['fouls'] > 12.5 else 0.85
    
    card_h = s_h['cards'] * viol_h * r_data['factor'] * strict
    card_a = s_a['cards'] * viol_a * r_data['factor'] * strict
    
    prob_red = ((s_h['red_cards_avg'] + s_a['red_cards_avg']) / 2) * rr * 100
    pr_lbl = "ALTA 🔴" if prob_red > 12 else "MÉDIA 🟠" if prob_red > 8 else "BAIXA 🟡"

    # xG para Dupla Chance
    xg_home = (s_h['goals_f'] * s_a['goals_a']) / 1.3
    xg_away = (s_a['goals_f'] * s_h['goals_a']) / 1.3

    return {
        'home': h_norm, 'away': a_norm, 'referee': ref,
        'league_h': s_h.get('league'), 'league_a': s_a.get('league'),
        'corners': {'total': corn_h + corn_a, 'h': corn_h, 'a': corn_a},
        'cards': {'total': card_h + card_a, 'h': card_h, 'a': card_a},
        'goals': {'h': xg_home, 'a': xg_away},
        'meta': {
            'shots_h': shots_h, 'p_label_h': "ALTO 🔥" if p_h > 1.0 else "BAIXO ⚪",
            'shots_a': shots_a,
            'strict_val': strict, 'prob_red': prob_red, 'prob_red_lbl': pr_lbl,
            'red_rate': rr
        }
    }

def get_detailed_probs(pred: Dict) -> Dict:
    def p(k, l): return sum((l**i * math.exp(-l)) / math.factorial(i) for i in range(k + 1))
    cH, cA = pred['corners']['h'], pred['corners']['a']
    kH, kA = pred['cards']['h'], pred['cards']['a']
    
    xG_H, xG_A = pred['goals']['h'], pred['goals']['a']
    total_strength = xG_H + xG_A
    if total_strength == 0: p_home, p_away = 0.33, 0.33
    else:
        p_home = xG_H / (total_strength * 1.2)
        p_away = xG_A / (total_strength * 1.2)
    p_draw = max(0, 1 - (p_home + p_away))
    
    dnb_home = (p_home / (p_home + p_away)) * 100 if (p_home + p_away) > 0 else 50
    dnb_away = (p_away / (p_home + p_away)) * 100 if (p_home + p_away) > 0 else 50
    
    return {
        'corners': {
            'total': {f"Over {i}.5": (1-p(i, cH+cA))*100 for i in range(8, 13)},
            'home': {'Over 3.5': (1-p(3, cH))*100, 'Over 4.5': (1-p(4, cH))*100, 'Over 5.5': (1-p(5, cH))*100},
            'away': {'Over 3.5': (1-p(3, cA))*100, 'Over 4.5': (1-p(4, cA))*100, 'Over 5.5': (1-p(5, cA))*100}
        },
        'cards': {
            'total': {f"Over {i}.5": (1-p(i, kH+kA))*100 for i in range(2, 6)},
            'home': {'Over 1.5': (1-p(1, kH))*100, 'Over 2.5': (1-p(2, kH))*100},
            'away': {'Over 1.5': (1-p(1, kA))*100, 'Over 2.5': (1-p(2, kA))*100}
        },
        'chance': {
            '1X': (p_home + p_draw) * 100,
            'X2': (p_away + p_draw) * 100,
            '12': (p_home + p_away) * 100,
            'DNB_1': dnb_home,
            'DNB_2': dnb_away
        }
    }

# ═══════════════════════════════════════════════════════════════════════════
# 3. LÓGICA DE BET BUILDER & HEDGE (PROFESSIONAL MODE)
# ═══════════════════════════════════════════════════════════════════════════

def get_fair_odd(prob_percent: float) -> float:
    if prob_percent <= 0: return 99.0
    return round(100 / prob_percent, 2)

def generate_bet_options(home_team: str, away_team: str, probs: Dict) -> List[Dict]:
    options = []
    
    # 1. ESCANTEIOS
    for line in [3.5, 4.5, 5.5]:
        p = probs['corners']['home'].get(f'Over {line}', 0)
        options.append({'label': f"{home_team} Over {line} cantos", 'prob': p, 'market':'corners', 'side':'home', 'min_odd': get_fair_odd(p)})
    for line in [3.5, 4.5]:
        p = probs['corners']['away'].get(f'Over {line}', 0)
        options.append({'label': f"{away_team} Over {line} cantos", 'prob': p, 'market':'corners', 'side':'away', 'min_odd': get_fair_odd(p)})
    for line in [8.5, 9.5, 10.5]:
        p = probs['corners']['total'].get(f'Over {int(line)}.5', 0)
        options.append({'label': f"Total Over {line} cantos", 'prob': p, 'market':'corners', 'side':'total', 'min_odd': get_fair_odd(p)})
    
    # 2. CARTÕES
    for line in [1.5, 2.5]:
        p = probs['cards']['home'].get(f'Over {line}', 0)
        options.append({'label': f"{home_team} Over {line} cartões", 'prob': p, 'market':'cards', 'side':'home', 'min_odd': get_fair_odd(p)})
        p2 = probs['cards']['away'].get(f'Over {line}', 0)
        options.append({'label': f"{away_team} Over {line} cartões", 'prob': p2, 'market':'cards', 'side':'away', 'min_odd': get_fair_odd(p2)})
    for line in [2.5, 3.5, 4.5, 5.5]:
        p = probs['cards']['total'].get(f'Over {int(line)}.5', 0)
        options.append({'label': f"Total Over {line} cartões", 'prob': p, 'market':'cards', 'side':'total', 'min_odd': get_fair_odd(p)})

    # 3. RESULTADO
    if probs['chance']['DNB_1'] >= 65:
        p = probs['chance']['DNB_1']
        options.append({'label': f"Empate Anula: {home_team}", 'prob': p, 'market':'chance', 'side':'home', 'min_odd': get_fair_odd(p)})
    if probs['chance']['DNB_2'] >= 65:
        p = probs['chance']['DNB_2']
        options.append({'label': f"Empate Anula: {away_team}", 'prob': p, 'market':'chance', 'side':'away', 'min_odd': get_fair_odd(p)})
        
    options.append({'label': f"Dupla Chance: {home_team} ou Empate", 'prob': probs['chance']['1X'], 'market':'chance', 'side':'home', 'min_odd': get_fair_odd(probs['chance']['1X'])})
    options.append({'label': f"Dupla Chance: {away_team} ou Empate", 'prob': probs['chance']['X2'], 'market':'chance', 'side':'away', 'min_odd': get_fair_odd(probs['chance']['X2'])})

    options.sort(key=lambda x: x['prob'], reverse=True)
    return options

def calculate_combined_probability(selections: List[Dict]) -> float:
    if not selections: return 0.0
    prob = 1.0
    for s in selections: prob *= (s['prob']/100)
    return prob * 100

def generate_dual_hedges(main_slip: List[Dict], stats: Dict, refs_db: Dict):
    hedge1 = []
    hedge2 = []
    
    games = {}
    for sel in main_slip:
        gid = sel['game_id']
        if gid not in games: games[gid] = []
        games[gid].append(sel)
        
    for gid, sels in games.items():
        home, away = sels[0]['home'], sels[0]['away']
        res = calcular_jogo_v14(home, away, stats, None, refs_db)
        if 'error' in res: continue
        
        probs = get_detailed_probs(res)
        all_opts = generate_bet_options(home, away, probs)
        
        # Filtro de Segurança
        valid_opts = [o for o in all_opts if o['prob'] >= 65]
        if len(valid_opts) < 6: valid_opts = all_opts[:12]
        
        main_labels = [s['label'] for s in sels]
        
        # ══════════════════════════════════════════════════════════════════
        # HEDGE 1: [DUPLA CHANCE/DNB] + [STAT]
        # ══════════════════════════════════════════════════════════════════
        h1_pair = []
        
        # Slot 1: Chance (DNB ou DC)
        chance_opts = [o for o in valid_opts if o['market'] == 'chance' and o['label'] not in main_labels]
        if chance_opts: h1_pair.append(chance_opts[0])
        else:
            # Fallback seguro
            safe = [o for o in valid_opts if o['label'] not in main_labels]
            if safe: h1_pair.append(safe[0])
            
        # Slot 2: Stat complementar
        stat_opts = [o for o in valid_opts if o['market'] in ['corners', 'cards'] and o['label'] not in main_labels and o not in h1_pair]
        if stat_opts: h1_pair.append(stat_opts[0])
            
        # Fallback
        if len(h1_pair) < 2:
            leftover = [o for o in valid_opts if o['label'] not in main_labels and o not in h1_pair]
            h1_pair.extend(leftover[:2-len(h1_pair)])
            
        for opt in h1_pair:
            hedge1.append({**opt, 'game_id': gid, 'home': home, 'away': away, 'type': 'Safety'})

        # ══════════════════════════════════════════════════════════════════
        # HEDGE 2: STRICT MIX [CANTO] + [CARTÃO]
        # ══════════════════════════════════════════════════════════════════
        h2_pair = []
        
        used_labels = main_labels + [o['label'] for o in h1_pair]
        avail_opts = [o for o in valid_opts if o['label'] not in used_labels]
        
        corn_opts = [o for o in avail_opts if o['market'] == 'corners']
        card_opts = [o for o in avail_opts if o['market'] == 'cards']
        
        # FORÇA A MISTURA
        if corn_opts and card_opts:
            h2_pair.append(corn_opts[0])
            h2_pair.append(card_opts[0])
        elif corn_opts:
            # Só tem cantos? Pega 1 Canto + 1 Chance (NUNCA Canto+Canto)
            h2_pair.append(corn_opts[0])
            bkp = [o for o in valid_opts if o['market'] == 'chance' and o not in h2_pair and o['label'] not in used_labels]
            if bkp: h2_pair.append(bkp[0])
        elif card_opts:
            # Só tem cartões? Pega 1 Cartão + 1 Chance
            h2_pair.append(card_opts[0])
            bkp = [o for o in valid_opts if o['market'] == 'chance' and o not in h2_pair and o['label'] not in used_labels]
            if bkp: h2_pair.append(bkp[0])
            
        # Fallback final (se ainda faltar)
        if len(h2_pair) < 2:
            leftover = [o for o in avail_opts if o not in h2_pair]
            h2_pair.extend(leftover[:2-len(h2_pair)])
            
        for opt in h2_pair:
            hedge2.append({**opt, 'game_id': gid, 'home': home, 'away': away, 'type': 'Mix'})
            
    return hedge1, hedge2

def render_bet_builder_tab(stats, refs_db):
    st.markdown("## 🎰 Bet Builder V15 (Professional Suite)")
    
    if 'main_slip' not in st.session_state: st.session_state.main_slip = []
    
    # 💰 GESTÃO DE BANCA
    with st.expander("💰 Gestão de Banca (Stake Calculator)", expanded=False):
        c1, c2 = st.columns(2)
        bankroll = c1.number_input("Sua Banca Total (R$)", 100, 100000, 1000)
        unit_size = c2.slider("Unidade de Risco (%)", 1.0, 10.0, 2.5)
        total_stake = bankroll * (unit_size/100)
        st.info(f"Aposta Sugerida Total: **R$ {total_stake:.2f}**")
        st.write(f"- Principal (50%): R$ {total_stake*0.5:.2f}")
        st.write(f"- Hedge 1 (30%): R$ {total_stake*0.3:.2f}")
        st.write(f"- Hedge 2 (20%): R$ {total_stake*0.2:.2f}")

    lista_times = sorted(list(stats.keys()))
    num_games = st.number_input("Quantos Jogos?", 1, 5, 3)
    
    temp_slip = []
    
    for i in range(num_games):
        st.markdown(f"---")
        st.markdown(f"### ⚽ Jogo {i+1}")
        c1, c2 = st.columns(2)
        h = c1.selectbox(f"Casa", lista_times, key=f"h_{i}")
        a = c2.selectbox(f"Fora", lista_times, key=f"a_{i}", index=min(1, len(lista_times)-1))
        
        # Inputs de Odd (EV+)
        odd_h = st.number_input(f"Odd Mercado (Opcional) - Jogo {i+1}", 1.01, 20.0, 1.0, key=f"odd_{i}")
        
        res = calcular_jogo_v14(h, a, stats, None, refs_db)
        if 'error' in res: continue
            
        probs = get_detailed_probs(res)
        opts = generate_bet_options(h, a, probs)
        opt_fmt = [f"{o['label']}" for o in opts]
        
        s1 = st.selectbox(f"Seleção 1", range(len(opts)), format_func=lambda x: opt_fmt[x], key=f"s1_{i}")
        s2 = st.selectbox(f"Seleção 2", range(len(opts)), format_func=lambda x: opt_fmt[x], key=f"s2_{i}", index=min(1, len(opts)-1))
        
        temp_slip.append({**opts[s1], 'game_id': i, 'home': h, 'away': a})
        temp_slip.append({**opts[s2], 'game_id': i, 'home': h, 'away': a})
    
    st.session_state.main_slip = temp_slip
    
    if st.button("🔮 Gerar Hedges & Relatório", type="primary"):
        h1, h2 = generate_dual_hedges(st.session_state.main_slip, stats, refs_db)
        
        st.success("✅ Hedges Gerados!")
        
        # DISPLAY PROFISSIONAL
        c_main, c_h1, c_h2 = st.columns(3)
        
        with c_main:
            st.info("📋 **Principal (Alvo)**")
            txt_main = "📋 *Bilhete Principal*\n"
            games_seen = []
            for s in st.session_state.main_slip:
                if s['game_id'] not in games_seen:
                    st.caption(f"{s['home']} x {s['away']}")
                    txt_main += f"\n⚽ {s['home']} x {s['away']}\n"
                    games_seen.append(s['game_id'])
                st.write(f"- {s['label']}")
                txt_main += f"- {s['label']} (Min Odd: {s['min_odd']})\n"
                
                # EV Check
                if 'odd' in locals() and odd_h > 1.0:
                    if odd_h > s['min_odd']: st.success(f"💎 VALOR: {odd_h} > {s['min_odd']}")
            
            st.metric("Prob Comb.", f"{calculate_combined_probability(st.session_state.main_slip):.1f}%")
            
        with c_h1:
            st.warning("🛡️ **Hedge #1 (Segurança)**")
            txt_h1 = "🛡️ *Hedge Segurança*\n"
            games_seen = []
            for s in h1:
                if s['game_id'] not in games_seen:
                    st.caption(f"{s['home']} x {s['away']}")
                    txt_h1 += f"\n⚽ {s['home']} x {s['away']}\n"
                    games_seen.append(s['game_id'])
                st.write(f"- {s['label']}")
                txt_h1 += f"- {s['label']} (Min Odd: {s['min_odd']})\n"
            st.metric("Prob Comb.", f"{calculate_combined_probability(h1):.1f}%")

        with c_h2:
            st.success("🔄 **Hedge #2 (Mix)**")
            txt_h2 = "🔄 *Hedge Mix*\n"
            games_seen = []
            for s in h2:
                if s['game_id'] not in games_seen:
                    st.caption(f"{s['home']} x {s['away']}")
                    txt_h2 += f"\n⚽ {s['home']} x {s['away']}\n"
                    games_seen.append(s['game_id'])
                st.write(f"- {s['label']}")
                txt_h2 += f"- {s['label']} (Min Odd: {s['min_odd']})\n"
            st.metric("Prob Comb.", f"{calculate_combined_probability(h2):.1f}%")

        # 📋 COPY PASTE AREA
        st.markdown("---")
        st.subheader("📋 Copiar para Telegram/WhatsApp")
        full_text = f"{txt_main}\n---\n{txt_h1}\n---\n{txt_h2}\n\n🤖 Gerado por FutPrevisão Pro"
        st.code(full_text, language='text')

# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def render_result_v14_5(res, all_dfs):
    m = res['meta']
    probs = get_detailed_probs(res)
    st.markdown("---")
    st.subheader(f"🏠 {res['home']} vs ✈️ {res['away']}")
    
    if res.get('referee'):
        st.caption(f"👮 Árbitro: {res['referee']} | Rigidez: {m['strict_val']}x")
    
    risk = []
    if m['strict_val'] < 1.05 and m['prob_red'] > 8: risk.append("⚠️ Conflito: Times violentos com árbitro leniente.")
    if risk:
        with st.expander("🛡️ Radar de Risco", expanded=True):
            for r in risk: st.error(r)
    
    c1, c2 = st.columns(2)
    with c1:
        st.info("🏁 **Escanteios Individuais**")
        st.markdown(f"**🏠 {res['home']}**")
        for line in [3.5, 4.5, 5.5]:
            p = probs['corners']['home'].get(f'Over {line}', 0)
            h = get_native_history(res['home'], res['league_h'], 'corners', line, 'home', all_dfs)
            c = "green" if p >= 70 else "gray"
            st.markdown(f"Over {line}: :{c}[**{p:.0f}%**] | Hist: {h}")
            
    with c2:
        st.markdown(f"**✈️ {res['away']}**")
        for line in [3.5, 4.5, 5.5]:
            p = probs['corners']['away'].get(f'Over {line}', 0)
            h = get_native_history(res['away'], res['league_a'], 'corners', line, 'away', all_dfs)
            c = "green" if p >= 70 else "gray"
            st.markdown(f"Over {line}: :{c}[**{p:.0f}%**] | Hist: {h}")

    st.markdown("---")

    c3, c4 = st.columns(2)
    with c3:
        st.warning("🟨 **Cartões Individuais**")
        st.markdown(f"**🏠 {res['home']}**")
        for line in [1.5, 2.5]:
            p = probs['cards']['home'].get(f'Over {line}', 0)
            h = get_native_history(res['home'], res['league_h'], 'cards', line, 'home', all_dfs)
            c = "green" if p >= 70 else "gray"
            st.markdown(f"Over {line}: :{c}[**{p:.0f}%**] | Hist: {h}")
            
    with c4:
        st.markdown(f"**✈️ {res['away']}**")
        for line in [1.5, 2.5]:
            p = probs['cards']['away'].get(f'Over {line}', 0)
            h = get_native_history(res['away'], res['league_a'], 'cards', line, 'away', all_dfs)
            c = "green" if p >= 70 else "gray"
            st.markdown(f"Over {line}: :{c}[**{p:.0f}%**] | Hist: {h}")

def main():
    st.title("⚽ FutPrevisão V15.0 (Ultimate)")
    
    with st.spinner("Carregando..."):
        DEBUG_LOGS.clear()
        stats = learn_stats_v14()
        refs = load_referees_v14()
        calendar = load_calendar_safe()
        all_dfs = load_all_dataframes()
    
    if not stats:
        st.error("🚨 ERRO: Dados não carregados.")
        return

    tab1, tab2, tab3 = st.tabs(["📅 Calendário", "🧪 Simulação", "🎰 Bet Builder"])
    
    with tab1:
        if calendar.empty: st.warning("Vazio")
        else:
            dates = calendar['DtObj'].dt.strftime('%d/%m/%Y').unique()
            sel_date = st.selectbox("Data", dates)
            subset = calendar[calendar['DtObj'].dt.strftime('%d/%m/%Y') == sel_date]
            for i, row in subset.iterrows():
                tc = row.get('Time_Casa', row.get('Mandante', 'Time A'))
                tv = row.get('Time_Visitante', row.get('Visitante', 'Time B'))
                if st.button(f"{tc} x {tv}", key=f"b{i}"):
                    res = calcular_jogo_v14(tc, tv, stats, None, refs)
                    if 'error' not in res: render_result_v14_5(res, all_dfs)

    with tab2:
        l_times = sorted(list(stats.keys()))
        l_refs = ["Neutro"] + sorted(list(refs.keys()))
        c1, c2, c3 = st.columns(3)
        h = c1.selectbox("Casa", l_times, index=0)
        a = c2.selectbox("Fora", l_times, index=1)
        r = c3.selectbox("Árbitro", l_refs)
        
        if st.button("Simular"):
            rf = None if r == "Neutro" else r
            res = calcular_jogo_v14(h, a, stats, rf, refs)
            if 'error' not in res: render_result_v14_5(res, all_dfs)
            
    with tab3:
        render_bet_builder_tab(stats, refs)

if __name__ == "__main__":
    main()
