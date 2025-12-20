
"""
╔═══════════════════════════════════════════════════════════════════════════╗
║          FUTPREVISÃO V14.5 - HEDGE SYSTEM (2 SELEÇÕES POR JOGO)          ║
║                          Sistema de Análise de Apostas                     ║
║                                                                            ║
║  Versão: V14.5 + Hedge v2                                                 ║
║  Data: Dezembro 2025                                                      ║
║  🆕 Atualização: 2 seleções por jogo no Bet Builder                       ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import math
import numpy as np
import os
from typing import Dict, Optional, Any, List
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
# CARREGAMENTO (MANTIDO DO V14.5)
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
                    df.columns = [c.strip() for c in df.columns]
                    df['_League_'] = league_name 
                    return df
            except: pass
    return pd.DataFrame()

@st.cache_resource
def load_all_dataframes() -> Dict[str, pd.DataFrame]:
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

def get_native_history(team_name: str, league: str, market: str, line: float, location: str, all_dfs: Dict) -> str:
    if league not in all_dfs:
        return "N/A"
    
    df = all_dfs[league]
    
    if location == 'home':
        matches = df[df['HomeTeam'] == team_name]
        col_code = 'HC' if market == 'corners' else 'HY'
    else:
        matches = df[df['AwayTeam'] == team_name]
        col_code = 'AC' if market == 'corners' else 'AY'
    
    if matches.empty:
        return "0/0"

    last_matches = matches.tail(10)
    total_games = len(last_matches)
    
    if total_games == 0:
        return "0/0"
        
    hits = 0
    for val in last_matches[col_code]:
        try:
            if float(val) > line:
                hits += 1
        except: pass
        
    return f"{hits}/{total_games}"

# ═══════════════════════════════════════════════════════════════════════════
# CÁLCULO V14
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
        
    shots_h = s_h['shots_on_target']
    shots_a = s_a['shots_on_target']
    
    p_h = 1.20 if shots_h > THRESHOLDS['shots_pressure_high'] else 1.10 if shots_h > THRESHOLDS['shots_pressure_medium'] else 1.0
    l_h = "ALTO 🔥" if p_h == 1.20 else "MÉDIO ✅" if p_h == 1.10 else "BAIXO ⚪"
    
    p_a = 1.20 if shots_a > THRESHOLDS['shots_pressure_high'] else 1.10 if shots_a > THRESHOLDS['shots_pressure_medium'] else 1.0
    l_a = "ALTO 🔥" if p_a == 1.20 else "MÉDIO ✅" if p_a == 1.10 else "BAIXO ⚪"
    
    corn_h = s_h['corners'] * 1.15 * p_h
    corn_a = s_a['corners'] * 0.90 * p_a
    
    rr = r_data['red_rate']
    strict = 1.15 if rr > THRESHOLDS['red_rate_strict_high'] else 1.08 if rr > THRESHOLDS['red_rate_strict_medium'] else 1.0
    s_lbl = "MUITO RIGOROSO 🔴" if strict == 1.15 else "RIGOROSO 🟠" if strict == 1.08 else "NORMAL 🟢"
    
    viol_h = 1.0 if s_h['fouls'] > THRESHOLDS['fouls_violent'] else 0.85
    viol_a = 1.0 if s_a['fouls'] > THRESHOLDS['fouls_violent'] else 0.85
    
    card_h = s_h['cards'] * viol_h * r_data['factor'] * strict
    card_a = s_a['cards'] * viol_a * r_data['factor'] * strict
    
    reds_avg = (s_h['red_cards_avg'] + s_a['red_cards_avg']) / 2
    prob_red = reds_avg * rr * 100
    pr_lbl = "ALTA 🔴" if prob_red > 12 else "MÉDIA 🟠" if prob_red > 8 else "BAIXA 🟡"

    return {
        'home': h_norm, 'away': a_norm, 'referee': ref,
        'league_h': s_h.get('league'), 'league_a': s_a.get('league'),
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
            'home': {'Over 3.5': (1-p(3, cH))*100, 'Over 4.5': (1-p(4, cH))*100, 'Over 2.5': (1-p(2, cH))*100},
            'away': {'Over 3.5': (1-p(3, cA))*100, 'Over 4.5': (1-p(4, cA))*100, 'Over 2.5': (1-p(2, cA))*100}
        },
        'cards': {
            'total': {f"Over {i}.5": (1-p(i, kH+kA))*100 for i in range(3, 6)},
            'home': {'Over 1.5': (1-p(1, kH))*100, 'Over 2.5': (1-p(2, kH))*100},
            'away': {'Over 1.5': (1-p(1, kA))*100, 'Over 2.5': (1-p(2, kA))*100}
        }
    }

# ═══════════════════════════════════════════════════════════════════════════
# 🆕 GERADOR DE OPÇÕES PARA DROPDOWN
# ═══════════════════════════════════════════════════════════════════════════

def generate_bet_options(home_team: str, away_team: str, probs: Dict) -> List[Dict]:
    """
    Gera TODAS as opções de aposta possíveis para o dropdown.
    Retorna lista de dicts com: label, market, side, line, prob
    """
    options = []
    
    # ESCANTEIOS CASA
    for line in [2.5, 3.5, 4.5, 5.5]:
        prob = probs['corners']['home'].get(f'Over {line}', 0)
        options.append({
            'label': f"{home_team} Over {line} escanteios ({prob:.0f}%)",
            'market': 'corners',
            'side': 'home',
            'line': line,
            'prob': prob,
            'display': f"{home_team} Over {line} escanteios"
        })
    
    # ESCANTEIOS FORA
    for line in [2.5, 3.5, 4.5]:
        prob = probs['corners']['away'].get(f'Over {line}', 0)
        options.append({
            'label': f"{away_team} Over {line} escanteios ({prob:.0f}%)",
            'market': 'corners',
            'side': 'away',
            'line': line,
            'prob': prob,
            'display': f"{away_team} Over {line} escanteios"
        })
    
    # ESCANTEIOS TOTAL
    for line in [8.5, 9.5, 10.5, 11.5]:
        prob = probs['corners']['total'].get(f'Over {int(line)}.5', 0)
        options.append({
            'label': f"Total Over {line} escanteios ({prob:.0f}%)",
            'market': 'corners',
            'side': 'total',
            'line': line,
            'prob': prob,
            'display': f"Total Over {line} escanteios"
        })
    
    # CARTÕES CASA
    for line in [1.5, 2.5]:
        prob = probs['cards']['home'].get(f'Over {line}', 0)
        options.append({
            'label': f"{home_team} Over {line} cartões ({prob:.0f}%)",
            'market': 'cards',
            'side': 'home',
            'line': line,
            'prob': prob,
            'display': f"{home_team} Over {line} cartões"
        })
    
    # CARTÕES FORA
    for line in [1.5, 2.5]:
        prob = probs['cards']['away'].get(f'Over {line}', 0)
        options.append({
            'label': f"{away_team} Over {line} cartões ({prob:.0f}%)",
            'market': 'cards',
            'side': 'away',
            'line': line,
            'prob': prob,
            'display': f"{away_team} Over {line} cartões"
        })
    
    # CARTÕES TOTAL
    for line in [3.5, 4.5, 5.5]:
        prob = probs['cards']['total'].get(f'Over {int(line)}.5', 0)
        options.append({
            'label': f"Total Over {line} cartões ({prob:.0f}%)",
            'market': 'cards',
            'side': 'total',
            'line': line,
            'prob': prob,
            'display': f"Total Over {line} cartões"
        })
    
    # Ordenar por probabilidade (maior primeiro)
    options.sort(key=lambda x: x['prob'], reverse=True)
    
    return options

# ═══════════════════════════════════════════════════════════════════════════
# SISTEMA DE HEDGE (SIMPLIFICADO)
# ═══════════════════════════════════════════════════════════════════════════

def generate_hedge_bets_v3(main_slip: List[Dict], stats: Dict, refs_db: Dict) -> tuple:
    """
    Gera 2 bilhetes de hedge mantendo os MESMOS JOGOS.
    GARANTIA: Cada hedge tem EXATAMENTE 2 seleções por jogo.
    """
    
    hedge_1 = []
    hedge_2 = []
    
    # Agrupar por jogo
    games = {}
    for sel in main_slip:
        game_id = sel['game_id']
        if game_id not in games:
            games[game_id] = []
        games[game_id].append(sel)
    
    # Para cada jogo, gerar EXATAMENTE 2 variações
    for game_id, selections in games.items():
        ref_sel = selections[0]
        
        res = calcular_jogo_v14(ref_sel['home'], ref_sel['away'], stats, None, refs_db)
        if 'error' in res:
            continue
            
        probs = get_detailed_probs(res)
        all_options = generate_bet_options(res['home'], res['away'], probs)
        
        # Filtrar opções >= 70%
        valid_options = [opt for opt in all_options if opt['prob'] >= 70]
        
        # Pegar principais seleções do bilhete principal
        main_labels = [s['display'] for s in selections]
        
        # ═══════════════════════════════════════════════════════════════
        # HEDGE #1: Escolher EXATAMENTE 2 opções diferentes do principal
        # ═══════════════════════════════════════════════════════════════
        h1_opts = [opt for opt in valid_options if opt['display'] not in main_labels]
        
        # Garantir que tem pelo menos 2 opções
        while len(h1_opts) < 2 and len(valid_options) > 0:
            # Se não tem 2 opções diferentes, pega da lista geral (pode repetir)
            for opt in valid_options:
                if opt not in h1_opts:
                    h1_opts.append(opt)
                    if len(h1_opts) >= 2:
                        break
            break
        
        # Pegar EXATAMENTE 2
        for opt in h1_opts[:2]:
            hedge_1.append({
                'home': res['home'],
                'away': res['away'],
                'market': opt['market'],
                'side': opt['side'],
                'line': opt['line'],
                'prob': opt['prob'],
                'label': opt['display'],
                'display': opt['display'],
                'change': '🔄 variação',
                'game_id': game_id
            })
        
        # ═══════════════════════════════════════════════════════════════
        # HEDGE #2: Escolher EXATAMENTE 2 outras opções
        # ═══════════════════════════════════════════════════════════════
        h2_opts = []
        
        # Estratégia: Permite repetir se >= 80%, senão pega diferentes
        for opt in valid_options:
            if opt in h1_opts[:2]:
                continue  # Não repetir as do hedge #1
            
            if opt['display'] in main_labels and opt['prob'] >= 80:
                h2_opts.append(opt)  # Pode repetir do principal se >= 80%
            elif opt['display'] not in main_labels:
                h2_opts.append(opt)  # Adiciona diferentes
            
            if len(h2_opts) >= 2:
                break
        
        # Se ainda não tem 2, completa com qualquer opção válida
        if len(h2_opts) < 2:
            for opt in valid_options:
                if opt not in h2_opts and opt not in h1_opts[:2]:
                    h2_opts.append(opt)
                    if len(h2_opts) >= 2:
                        break
        
        # Pegar EXATAMENTE 2
        for opt in h2_opts[:2]:
            hedge_2.append({
                'home': res['home'],
                'away': res['away'],
                'market': opt['market'],
                'side': opt['side'],
                'line': opt['line'],
                'prob': opt['prob'],
                'label': opt['display'],
                'display': opt['display'],
                'change': '✅ mantido' if opt['display'] in main_labels else '🔄 variação',
                'game_id': game_id
            })
    
    return hedge_1, hedge_2


def calculate_combined_probability(selections: List[Dict]) -> float:
    if not selections:
        return 0.0
    prob_combined = 1.0
    for sel in selections:
        prob_combined *= (sel.get('prob', 70) / 100)
    return prob_combined * 100


def calculate_coverage_scenarios(main_slip, hedge_1, hedge_2, stakes: Dict) -> List[Dict]:
    scenarios = []
    
    p_main = calculate_combined_probability(main_slip)
    odd_main = (100 / p_main) * 0.85 if p_main > 0 else 2.0
    profit_main = (stakes['main'] * odd_main) - sum(stakes.values())
    
    scenarios.append({
        'scenario': '✅ Bilhete Principal ACERTA',
        'probability': p_main,
        'profit': profit_main,
        'color': 'green'
    })
    
    p_h1 = calculate_combined_probability(hedge_1)
    odd_h1 = (100 / p_h1) * 0.85 if p_h1 > 0 else 2.0
    profit_h1 = (stakes['hedge_1'] * odd_h1) - sum(stakes.values())
    
    scenarios.append({
        'scenario': '⚠️ Hedge #1 ACERTA',
        'probability': p_h1,
        'profit': profit_h1,
        'color': 'orange'
    })
    
    p_h2 = calculate_combined_probability(hedge_2)
    odd_h2 = (100 / p_h2) * 0.85 if p_h2 > 0 else 2.0
    profit_h2 = (stakes['hedge_2'] * odd_h2) - sum(stakes.values())
    
    scenarios.append({
        'scenario': '🟡 Hedge #2 ACERTA',
        'probability': p_h2,
        'profit': profit_h2,
        'color': 'blue'
    })
    
    p_all_miss = max(5, 100 - (p_main + p_h1 + p_h2))
    
    scenarios.append({
        'scenario': '❌ TODOS ERRAM',
        'probability': p_all_miss,
        'profit': -sum(stakes.values()),
        'color': 'red'
    })
    
    return scenarios

# ═══════════════════════════════════════════════════════════════════════════
# UI V14.5 (MANTIDA)
# ═══════════════════════════════════════════════════════════════════════════

def render_result_v14_5(res, all_dfs):
    m = res['meta']
    probs = get_detailed_probs(res)
    
    st.markdown("---")
    
    c1, c2, c3 = st.columns([2,1,2])
    c1.markdown(f"### 🏠 {res['home']}")
    c2.markdown("<h3 style='text-align: center'>VS</h3>", unsafe_allow_html=True)
    c3.markdown(f"<h3 style='text-align: right'>✈️ {res['away']}</h3>", unsafe_allow_html=True)
    
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("xG Casa", f"{res['goals']['h']:.2f}")
    k2.metric("xG Fora", f"{res['goals']['a']:.2f}")
    k3.metric("Chutes Casa", f"{m['shots_h']:.1f}", m['p_label_h'])
    k4.metric("Risco Vermelho", f"{m['prob_red']:.1f}%", m['prob_red_lbl'])
    
    st.caption(f"👮 Juiz: {res['referee'] if res['referee'] else 'Neutro'} | Rigidez: {m['strict_lbl']} | Taxa Vermelhos: {m['red_rate']:.2f}")

    st.markdown("---")

    st.subheader(f"🏁 Escanteios (Total Esp: {res['corners']['total']:.2f})")
    ec1, ec2, ec3 = st.columns(3)
    
    with ec1:
        st.markdown("**Geral**")
        for k, v in probs['corners']['total'].items():
            if v > 65: st.write(f"{k}: **{v:.0f}%**")

    with ec2:
        st.markdown(f"**🏠 {res['home']}** (Esp: {res['corners']['h']:.1f})")
        p35 = probs['corners']['home']['Over 3.5']
        p45 = probs['corners']['home']['Over 4.5']
        
        h35 = get_native_history(res['home'], res['league_h'], 'corners', 3.5, 'home', all_dfs)
        h45 = get_native_history(res['home'], res['league_h'], 'corners', 4.5, 'home', all_dfs)
        
        c35 = "green" if p35 >= 70 else "red"
        c45 = "green" if p45 >= 60 else "red"
        
        st.markdown(f"Over 3.5: :{c35}[**{p35:.0f}%**] | Hist: {h35}")
        st.markdown(f"Over 4.5: :{c45}[**{p45:.0f}%**] | Hist: {h45}")

    with ec3:
        st.markdown(f"**✈️ {res['away']}** (Esp: {res['corners']['a']:.1f})")
        p35 = probs['corners']['away']['Over 3.5']
        p45 = probs['corners']['away']['Over 4.5']
        
        h35 = get_native_history(res['away'], res['league_a'], 'corners', 3.5, 'away', all_dfs)
        h45 = get_native_history(res['away'], res['league_a'], 'corners', 4.5, 'away', all_dfs)
        
        c35 = "green" if p35 >= 70 else "red"
        c45 = "green" if p45 >= 60 else "red"
        
        st.markdown(f"Over 3.5: :{c35}[**{p35:.0f}%**] | Hist: {h35}")
        st.markdown(f"Over 4.5: :{c45}[**{p45:.0f}%**] | Hist: {h45}")
        
    st.markdown("---")

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
        c15 = "green" if p15 >= 75 else "red"
        
        st.markdown(f"Over 1.5: :{c15}[**{p15:.0f}%**] | Hist: {h15}")

    with kc3:
        st.markdown(f"**✈️ {res['away']}**")
        p15 = probs['cards']['away']['Over 1.5']
        
        h15 = get_native_history(res['away'], res['league_a'], 'cards', 1.5, 'away', all_dfs)
        c15 = "green" if p15 >= 75 else "red"
        
        st.markdown(f"Over 1.5: :{c15}[**{p15:.0f}%**] | Hist: {h15}")


def render_hedge_builder_tab_v3(stats, refs_db):
    """🆕 Interface SIMPLIFICADA - 1 dropdown por seleção"""
    
    st.markdown("## 🎰 Bet Builder + Sistema de Cobertura")
    st.caption("Interface simplificada: 2 dropdowns por jogo com TODAS as opções")
    
    st.markdown("---")
    
    with st.expander("📋 BILHETE PRINCIPAL", expanded=True):
        st.markdown("**Monte seu bilhete (2 seleções por jogo):**")
        
        if 'main_slip' not in st.session_state:
            st.session_state.main_slip = []
        
        lista_times = sorted(list(stats.keys()))
        
        num_games = st.number_input("Quantos jogos?", 1, 5, 3, key="num_games")
        
        main_slip_temp = []
        
        for i in range(num_games):
            st.markdown(f"### ⚽ Jogo {i+1}")
            
            col_teams = st.columns(2)
            with col_teams[0]:
                home = st.selectbox(f"Time Casa", lista_times, key=f"home_{i}")
            with col_teams[1]:
                away = st.selectbox(f"Time Visitante", lista_times, key=f"away_{i}", 
                                   index=min(i+1, len(lista_times)-1))
            
            # Calcular probabilidades UMA VEZ
            res = calcular_jogo_v14(home, away, stats, None, refs_db)
            if 'error' in res:
                st.error(f"Erro: {res['error']}")
                continue
                
            probs = get_detailed_probs(res)
            
            # Gerar TODAS as opções
            all_options = generate_bet_options(res['home'], res['away'], probs)
            option_labels = [opt['label'] for opt in all_options]
            
            # ═══════════════════════════════════════════════════════════
            # SELEÇÃO #1 - DROPDOWN ÚNICO
            # ═══════════════════════════════════════════════════════════
            st.markdown("#### 🎯 Seleção #1")
            selected_idx_1 = st.selectbox(
                "Escolha a aposta:",
                range(len(option_labels)),
                format_func=lambda x: option_labels[x],
                key=f"sel_1_{i}"
            )
            
            sel_1 = all_options[selected_idx_1]
            
            # Cor da probabilidade
            color_1 = "green" if sel_1['prob'] >= 70 else "orange" if sel_1['prob'] >= 60 else "red"
            st.markdown(f"**Probabilidade:** :{color_1}[{sel_1['prob']:.1f}%]")
            
            main_slip_temp.append({
                'home': home,
                'away': away,
                'market': sel_1['market'],
                'side': sel_1['side'],
                'line': sel_1['line'],
                'prob': sel_1['prob'],
                'label': sel_1['display'],
                'display': sel_1['display'],
                'referee': None,
                'game_id': i
            })
            
            # ═══════════════════════════════════════════════════════════
            # SELEÇÃO #2 - DROPDOWN ÚNICO
            # ═══════════════════════════════════════════════════════════
            st.markdown("#### 🎯 Seleção #2")
            selected_idx_2 = st.selectbox(
                "Escolha a aposta:",
                range(len(option_labels)),
                format_func=lambda x: option_labels[x],
                key=f"sel_2_{i}",
                index=min(1, len(option_labels)-1)  # Default = 2ª opção
            )
            
            sel_2 = all_options[selected_idx_2]
            
            color_2 = "green" if sel_2['prob'] >= 70 else "orange" if sel_2['prob'] >= 60 else "red"
            st.markdown(f"**Probabilidade:** :{color_2}[{sel_2['prob']:.1f}%]")
            
            main_slip_temp.append({
                'home': home,
                'away': away,
                'market': sel_2['market'],
                'side': sel_2['side'],
                'line': sel_2['line'],
                'prob': sel_2['prob'],
                'label': sel_2['display'],
                'display': sel_2['display'],
                'referee': None,
                'game_id': i
            })
            
            st.markdown("---")
        
        st.session_state.main_slip = main_slip_temp
        
        # Resumo
        if st.session_state.main_slip:
            st.markdown("### 📊 Resumo do Bilhete Principal:")
            
            games = {}
            for sel in st.session_state.main_slip:
                game_id = sel['game_id']
                if game_id not in games:
                    games[game_id] = []
                games[game_id].append(sel)
            
            for game_id, selections in sorted(games.items()):
                st.markdown(f"**Jogo {game_id + 1}: {selections[0]['home']} vs {selections[0]['away']}**")
                for j, sel in enumerate(selections, 1):
                    prob_color = "green" if sel['prob'] >= 70 else "orange"
                    st.markdown(f"  {j}. **{sel['display']}** - :{prob_color}[{sel['prob']:.1f}%]")
            
            prob_combinada = calculate_combined_probability(st.session_state.main_slip)
            st.metric("Probabilidade Combinada", f"{prob_combinada:.1f}%")
            st.caption(f"Total de seleções: {len(st.session_state.main_slip)}")
    
    # Gerar Hedges
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        budget = st.number_input("💰 Orçamento Total (€)", 10, 1000, 50, step=10)
    
    with col2:
        st.markdown("**Distribuição Sugerida:**")
        st.write(f"Principal: €{budget * 0.5:.0f} (50%)")
        st.write(f"Hedge #1: €{budget * 0.3:.0f} (30%)")
        st.write(f"Hedge #2: €{budget * 0.2:.0f} (20%)")
    
    if st.button("🔮 GERAR BILHETES DE COBERTURA", use_container_width=True, type="primary"):
        if not st.session_state.main_slip or len(st.session_state.main_slip) == 0:
            st.error("❌ Monte o bilhete principal primeiro!")
        else:
            with st.spinner("⚙️ Gerando hedges..."):
                hedge_1, hedge_2 = generate_hedge_bets_v3(
                    st.session_state.main_slip, 
                    stats, 
                    refs_db
                )
                
                if not hedge_1 or not hedge_2:
                    st.warning("⚠️ Não foi possível gerar hedges com 70%+")
                else:
                    stakes = {
                        'main': budget * 0.5,
                        'hedge_1': budget * 0.3,
                        'hedge_2': budget * 0.2
                    }
                    
                    st.success("✅ Hedges gerados!")
                    
                    st.markdown("---")
                    
                    # HEDGE #1
                    st.markdown("## 🤖 BILHETE HEDGE #1")
                    
                    h1_games = {}
                    for sel in hedge_1:
                        game_id = sel['game_id']
                        if game_id not in h1_games:
                            h1_games[game_id] = []
                        h1_games[game_id].append(sel)
                    
                    for game_id, selections in sorted(h1_games.items()):
                        st.markdown(f"**Jogo {game_id + 1}: {selections[0]['home']} vs {selections[0]['away']}**")
                        for j, sel in enumerate(selections, 1):
                            prob_color = "green" if sel['prob'] >= 70 else "orange"
                            st.markdown(f"  {j}. **{sel['display']}** - :{prob_color}[{sel['prob']:.1f}%] {sel['change']}")
                    
                    prob_h1 = calculate_combined_probability(hedge_1)
                    st.metric("Probabilidade Combinada", f"{prob_h1:.1f}%")
                    st.info(f"💵 Stake: €{stakes['hedge_1']:.0f}")
                    
                    st.markdown("---")
                    
                    # HEDGE #2
                    st.markdown("## 🤖 BILHETE HEDGE #2")
                    
                    h2_games = {}
                    for sel in hedge_2:
                        game_id = sel['game_id']
                        if game_id not in h2_games:
                            h2_games[game_id] = []
                        h2_games[game_id].append(sel)
                    
                    for game_id, selections in sorted(h2_games.items()):
                        st.markdown(f"**Jogo {game_id + 1}: {selections[0]['home']} vs {selections[0]['away']}**")
                        for j, sel in enumerate(selections, 1):
                            prob_color = "green" if sel['prob'] >= 70 else "orange"
                            st.markdown(f"  {j}. **{sel['display']}** - :{prob_color}[{sel['prob']:.1f}%] {sel['change']}")
                    
                    prob_h2 = calculate_combined_probability(hedge_2)
                    st.metric("Probabilidade Combinada", f"{prob_h2:.1f}%")
                    st.info(f"💵 Stake: €{stakes['hedge_2']:.0f}")
                    
                    st.markdown("---")
                    
                    # ANÁLISE
                    st.markdown("## 📊 ANÁLISE DE COBERTURA")
                    
                    scenarios = calculate_coverage_scenarios(
                        st.session_state.main_slip,
                        hedge_1,
                        hedge_2,
                        stakes
                    )
                    
                    for scenario in scenarios:
                        if scenario['profit'] > 0:
                            st.success(f"{scenario['scenario']}: **{scenario['probability']:.1f}%** → Lucro **€{scenario['profit']:.2f}** 💰")
                        else:
                            st.error(f"{scenario['scenario']}: **{scenario['probability']:.1f}%** → Perda **€{abs(scenario['profit']):.2f}** 💸")
                    
                    prob_ganho = sum(s['probability'] for s in scenarios if s['profit'] > 0)
                    st.metric("🎯 Probabilidade de GANHAR algo", f"{prob_ganho:.1f}%")
                    
                    if prob_ganho >= 85:
                        st.success("✅ Excelente cobertura!")
                    elif prob_ganho >= 70:
                        st.info("👍 Boa cobertura!")
                    else:
                        st.warning("⚠️ Cobertura moderada")


def main():
    st.title("⚽ FutPrevisão V14.5 + Hedge v3")
    st.caption("Interface Simplificada - Dropdown Único")
    
    with st.spinner("Carregando..."):
        DEBUG_LOGS.clear()
        stats = learn_stats_v14()
        refs = load_referees_v14()
        calendar = load_calendar_safe()
        all_dfs = load_all_dataframes()
    
    lista_times = sorted(list(stats.keys()))
    lista_juizes = ["Neutro"] + sorted(list(refs.keys()))
    
    with st.sidebar:
        with st.expander("🛠️ Status", expanded=False):
            st.write(f"Times: {len(stats)}")
            st.write(f"Ligas: {len(all_dfs)}")
            for log in DEBUG_LOGS: st.write(log)
    
    if not stats:
        st.error("🚨 Erro ao carregar dados")
        return

    tab1, tab2, tab3 = st.tabs(["📅 Calendário", "🧪 Simulação", "🎰 Bet Builder"])
    
    with tab1:
        if calendar.empty:
            st.warning("Calendário vazio")
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
        st.subheader("Simulador")
        c1, c2, c3 = st.columns(3)
        idx_h = lista_times.index("Liverpool") if "Liverpool" in lista_times else 0
        idx_a = lista_times.index("Man City") if "Man City" in lista_times else 1
        
        h = c1.selectbox("Mandante", lista_times, index=idx_h)
        a = c2.selectbox("Visitante", lista_times, index=idx_a)
        r = c3.selectbox("Árbitro", lista_juizes, index=0)
        
        if st.button("Simular"):
            ref_val = None if r == "Neutro" else r
            res = calcular_jogo_v14(h, a, stats, ref_val, refs)
            if 'error' in res: st.error(res['error'])
            else: render_result_v14_5(res, all_dfs)
    
    with tab3:
        render_hedge_builder_tab_v3(stats, refs_db)

if __name__ == "__main__":
    main()
