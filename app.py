"""
╔═══════════════════════════════════════════════════════════════════════════╗
║               FUTPREVISÃO V14.5 - NATIVE HISTORY + BET BUILDER            ║
║                          Sistema de Análise de Apostas                     ║
║                                                                            ║
║  Versão: V14.5 + Hedge                                                    ║
║  Data: Dezembro 2025                                                      ║
║  🆕 Nova funcionalidade: Sistema de Cobertura Inteligente                ║
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
                    df.columns = [c.strip() for c in df.columns]
                    df['_League_'] = league_name 
                    return df
            except: pass
    return pd.DataFrame()

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
# CÁLCULO DE HISTÓRICO NATIVO
# ═══════════════════════════════════════════════════════════════════════════

def get_native_history(team_name: str, league: str, market: str, line: float, location: str, all_dfs: Dict) -> str:
    """Calcula histórico real lendo diretamente do DataFrame da liga."""
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
# 🆕 SISTEMA DE HEDGE BETTING
# ═══════════════════════════════════════════════════════════════════════════

def generate_hedge_bets(main_slip: List[Dict], stats: Dict, refs_db: Dict) -> tuple:
    """
    Gera 2 bilhetes de hedge mantendo os MESMOS JOGOS mas com SELEÇÕES DIFERENTES.
    
    Regras:
    - Sempre usar os mesmos jogos do bilhete principal
    - Variar seleções: Individual↔Total, Escanteios↔Cartões, Linha Alta↔Baixa
    - Manter TODAS as seleções com >= 70% probabilidade
    - Nunca usar Under (apenas Over)
    """
    
    hedge_1 = []
    hedge_2 = []
    
    for sel in main_slip:
        # Calcular resultado completo do jogo
        res = calcular_jogo_v14(sel['home'], sel['away'], stats, sel.get('referee'), refs_db)
        if 'error' in res:
            continue
            
        probs = get_detailed_probs(res)
        
        # ESTRATÉGIA HEDGE #1: Prioriza mudança Individual→Total + Linha Baixa
        h1 = generate_hedge_option_1(sel, probs, res)
        if h1:
            hedge_1.append(h1)
        
        # ESTRATÉGIA HEDGE #2: Prioriza troca Escanteios↔Cartões + Lado oposto
        h2 = generate_hedge_option_2(sel, probs, res)
        if h2:
            hedge_2.append(h2)
    
    return hedge_1, hedge_2


def generate_hedge_option_1(main_sel: Dict, probs: Dict, game_res: Dict) -> Optional[Dict]:
    """
    HEDGE #1: Individual → Total ou Linha Baixa
    Mantém mercado (escanteios/cartões) mas muda escopo ou linha
    """
    market = main_sel['market']
    side = main_sel['side']
    line = main_sel['line']
    
    # Se seleção principal é individual, tenta total
    if side in ['home', 'away']:
        # Tentar total do mesmo mercado
        if market == 'corners':
            for total_line in [10.5, 9.5, 8.5]:
                prob = probs['corners']['total'].get(f'Over {int(total_line)}.5', 0)
                if prob >= 70:
                    return {
                        'home': game_res['home'],
                        'away': game_res['away'],
                        'market': 'corners',
                        'side': 'total',
                        'line': total_line,
                        'prob': prob,
                        'label': f"Total Over {total_line} escanteios",
                        'change': '🔄 individual→total'
                    }
        
        elif market == 'cards':
            for total_line in [4.5, 3.5]:
                prob = probs['cards']['total'].get(f'Over {int(total_line)}.5', 0)
                if prob >= 70:
                    return {
                        'home': game_res['home'],
                        'away': game_res['away'],
                        'market': 'cards',
                        'side': 'total',
                        'line': total_line,
                        'prob': prob,
                        'label': f"Total Over {total_line} cartões",
                        'change': '🔄 individual→total'
                    }
    
    # Se não conseguiu total, tenta linha mais baixa do mesmo lado
    if market == 'corners':
        for lower_line in [2.5, 3.5, 4.5]:
            if lower_line < line:
                prob = probs['corners'][side].get(f'Over {lower_line}', 0)
                if prob >= 70:
                    side_label = game_res['home'] if side == 'home' else game_res['away']
                    return {
                        'home': game_res['home'],
                        'away': game_res['away'],
                        'market': 'corners',
                        'side': side,
                        'line': lower_line,
                        'prob': prob,
                        'label': f"{side_label} Over {lower_line} escanteios",
                        'change': f'🔄 {line}→{lower_line}'
                    }
    
    elif market == 'cards':
        for lower_line in [1.5, 2.5]:
            if lower_line < line:
                prob = probs['cards'][side].get(f'Over {lower_line}', 0)
                if prob >= 70:
                    side_label = game_res['home'] if side == 'home' else game_res['away']
                    return {
                        'home': game_res['home'],
                        'away': game_res['away'],
                        'market': 'cards',
                        'side': side,
                        'line': lower_line,
                        'prob': prob,
                        'label': f"{side_label} Over {lower_line} cartões",
                        'change': f'🔄 {line}→{lower_line}'
                    }
    
    # Fallback: mantém original se tiver >= 80%
    original_prob = main_sel.get('prob', 0)
    if original_prob >= 80:
        return {**main_sel, 'change': '✅ mantido (alta confiança)'}
    
    return None


def generate_hedge_option_2(main_sel: Dict, probs: Dict, game_res: Dict) -> Optional[Dict]:
    """
    HEDGE #2: Troca Mercado (Escanteios↔Cartões) ou Lado (Casa↔Fora)
    """
    market = main_sel['market']
    side = main_sel['side']
    
    # Tenta trocar mercado primeiro
    if market == 'corners':
        # Muda para cartões do mesmo lado (ou total)
        if side == 'home':
            prob = probs['cards']['home'].get('Over 1.5', 0)
            if prob >= 70:
                return {
                    'home': game_res['home'],
                    'away': game_res['away'],
                    'market': 'cards',
                    'side': 'home',
                    'line': 1.5,
                    'prob': prob,
                    'label': f"{game_res['home']} Over 1.5 cartões",
                    'change': '🔄 escanteios→cartões'
                }
        elif side == 'away':
            prob = probs['cards']['away'].get('Over 1.5', 0)
            if prob >= 70:
                return {
                    'home': game_res['home'],
                    'away': game_res['away'],
                    'market': 'cards',
                    'side': 'away',
                    'line': 1.5,
                    'prob': prob,
                    'label': f"{game_res['away']} Over 1.5 cartões",
                    'change': '🔄 escanteios→cartões'
                }
        else:  # total corners
            prob = probs['cards']['total'].get('Over 4.5', 0)
            if prob >= 70:
                return {
                    'home': game_res['home'],
                    'away': game_res['away'],
                    'market': 'cards',
                    'side': 'total',
                    'line': 4.5,
                    'prob': prob,
                    'label': f"Total Over 4.5 cartões",
                    'change': '🔄 escanteios→cartões'
                }
    
    elif market == 'cards':
        # Muda para escanteios
        if side == 'home':
            for corner_line in [3.5, 4.5]:
                prob = probs['corners']['home'].get(f'Over {corner_line}', 0)
                if prob >= 70:
                    return {
                        'home': game_res['home'],
                        'away': game_res['away'],
                        'market': 'corners',
                        'side': 'home',
                        'line': corner_line,
                        'prob': prob,
                        'label': f"{game_res['home']} Over {corner_line} escanteios",
                        'change': '🔄 cartões→escanteios'
                    }
        elif side == 'away':
            for corner_line in [2.5, 3.5]:
                prob = probs['corners']['away'].get(f'Over {corner_line}', 0)
                if prob >= 70:
                    return {
                        'home': game_res['home'],
                        'away': game_res['away'],
                        'market': 'corners',
                        'side': 'away',
                        'line': corner_line,
                        'prob': prob,
                        'label': f"{game_res['away']} Over {corner_line} escanteios",
                        'change': '🔄 cartões→escanteios'
                    }
        else:  # total cards
            prob = probs['corners']['total'].get('Over 9.5', 0)
            if prob >= 70:
                return {
                    'home': game_res['home'],
                    'away': game_res['away'],
                    'market': 'corners',
                    'side': 'total',
                    'line': 9.5,
                    'prob': prob,
                    'label': f"Total Over 9.5 escanteios",
                    'change': '🔄 cartões→escanteios'
                }
    
    # Tenta trocar lado (casa↔fora) mantendo mercado
    if side == 'home':
        opposite_side = 'away'
        opposite_team = game_res['away']
        # Ajusta linha (visitante joga mais recuado)
        adjusted_line = main_sel['line'] - 1.0 if market == 'corners' else main_sel['line']
    elif side == 'away':
        opposite_side = 'home'
        opposite_team = game_res['home']
        # Aumenta linha (casa joga mais ofensivo)
        adjusted_line = main_sel['line'] + 1.0 if market == 'corners' else main_sel['line']
    else:
        # Se é total, mantém
        original_prob = main_sel.get('prob', 0)
        if original_prob >= 80:
            return {**main_sel, 'change': '✅ mantido (total)'}
        return None
    
    prob = probs[market][opposite_side].get(f'Over {adjusted_line}', 0)
    if prob >= 70:
        market_label = 'escanteios' if market == 'corners' else 'cartões'
        return {
            'home': game_res['home'],
            'away': game_res['away'],
            'market': market,
            'side': opposite_side,
            'line': adjusted_line,
            'prob': prob,
            'label': f"{opposite_team} Over {adjusted_line} {market_label}",
            'change': f'🔄 trocou lado'
        }
    
    return None


def calculate_combined_probability(selections: List[Dict]) -> float:
    """Calcula probabilidade combinada de um bilhete."""
    if not selections:
        return 0.0
    prob_combined = 1.0
    for sel in selections:
        prob_combined *= (sel.get('prob', 70) / 100)
    return prob_combined * 100


def calculate_coverage_scenarios(main_slip, hedge_1, hedge_2, stakes: Dict) -> List[Dict]:
    """Simula todos os cenários possíveis de resultado."""
    scenarios = []
    
    # Cenário 1: Principal acerta
    p_main = calculate_combined_probability(main_slip)
    # Estimar odd combinada (simplificado)
    odd_main = (100 / p_main) * 0.85  # Margem da casa
    profit_main = (stakes['main'] * odd_main) - sum(stakes.values())
    
    scenarios.append({
        'scenario': '✅ Bilhete Principal ACERTA',
        'probability': p_main,
        'profit': profit_main,
        'color': 'green'
    })
    
    # Cenário 2: Hedge #1 acerta
    p_h1 = calculate_combined_probability(hedge_1)
    odd_h1 = (100 / p_h1) * 0.85
    profit_h1 = (stakes['hedge_1'] * odd_h1) - sum(stakes.values())
    
    scenarios.append({
        'scenario': '⚠️ Hedge #1 ACERTA',
        'probability': p_h1,
        'profit': profit_h1,
        'color': 'orange'
    })
    
    # Cenário 3: Hedge #2 acerta
    p_h2 = calculate_combined_probability(hedge_2)
    odd_h2 = (100 / p_h2) * 0.85
    profit_h2 = (stakes['hedge_2'] * odd_h2) - sum(stakes.values())
    
    scenarios.append({
        'scenario': '🟡 Hedge #2 ACERTA',
        'probability': p_h2,
        'profit': profit_h2,
        'color': 'blue'
    })
    
    # Cenário 4: Todos erram
    p_all_miss = 100 - (p_main + p_h1 + p_h2)  # Aproximação
    if p_all_miss < 0:
        p_all_miss = 5  # Mínimo 5%
    
    scenarios.append({
        'scenario': '❌ TODOS ERRAM',
        'probability': p_all_miss,
        'profit': -sum(stakes.values()),
        'color': 'red'
    })
    
    return scenarios

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
        c15 = "green" if p15 >= 75 else "red"
        
        st.markdown(f"Over 1.5: :{c15}[**{p15:.0f}%**] | Hist: {h15}")

    with kc3:
        st.markdown(f"**✈️ {res['away']}**")
        p15 = probs['cards']['away']['Over 1.5']
        
        h15 = get_native_history(res['away'], res['league_a'], 'cards', 1.5, 'away', all_dfs)
        c15 = "green" if p15 >= 75 else "red"
        
        st.markdown(f"Over 1.5: :{c15}[**{p15:.0f}%**] | Hist: {h15}")


def render_hedge_builder_tab(stats, refs_db):
    """🆕 Nova aba: Bet Builder + Hedge System"""
    
    st.markdown("## 🎰 Bet Builder + Sistema de Cobertura")
    st.caption("Monte seu bilhete principal e o sistema gera 2 hedges automáticos com os MESMOS jogos")
    
    st.markdown("---")
    
    # SEÇÃO 1: Input do Bilhete Principal
    with st.expander("📋 BILHETE PRINCIPAL", expanded=True):
        st.markdown("**Escolha 3 jogos para sua múltipla:**")
        
        # Inicializar session state
        if 'main_slip' not in st.session_state:
            st.session_state.main_slip = []
        
        lista_times = sorted(list(stats.keys()))
        
        num_games = st.number_input("Quantos jogos?", 1, 5, 3, key="num_games")
        
        main_slip_temp = []
        
        for i in range(num_games):
            st.markdown(f"### ⚽ Jogo {i+1}")
            col1, col2 = st.columns(2)
            
            with col1:
                home = st.selectbox(f"Time Casa", lista_times, key=f"home_{i}")
                away = st.selectbox(f"Time Visitante", lista_times, key=f"away_{i}", 
                                   index=min(i+1, len(lista_times)-1))
            
            with col2:
                market = st.selectbox("Mercado", 
                    ["Escanteios", "Cartões"], key=f"market_{i}")
                
                side = st.selectbox("Seleção", 
                    ["Casa", "Fora", "Total"], key=f"side_{i}")
                
                if market == "Escanteios":
                    line_options = [2.5, 3.5, 4.5, 5.5] if side != "Total" else [8.5, 9.5, 10.5, 11.5]
                else:  # Cartões
                    line_options = [1.5, 2.5] if side != "Total" else [3.5, 4.5, 5.5]
                
                line = st.selectbox("Linha", line_options, key=f"line_{i}")
            
            # Calcular probabilidade real
            res = calcular_jogo_v14(home, away, stats, None, refs_db)
            if 'error' not in res:
                probs = get_detailed_probs(res)
                
                market_key = 'corners' if market == "Escanteios" else 'cards'
                side_key = 'home' if side == "Casa" else 'away' if side == "Fora" else 'total'
                
                if side_key == 'total':
                    prob = probs[market_key]['total'].get(f'Over {int(line)}.5', 0)
                else:
                    prob = probs[market_key][side_key].get(f'Over {line}', 0)
                
                # Display da probabilidade
                color = "green" if prob >= 70 else "orange" if prob >= 60 else "red"
                st.markdown(f"Probabilidade: :{color}[**{prob:.1f}%**]")
                
                # Adicionar à lista
                side_label = res['home'] if side == "Casa" else res['away'] if side == "Fora" else "Total"
                market_label = "escanteios" if market == "Escanteios" else "cartões"
                
                main_slip_temp.append({
                    'home': home,
                    'away': away,
                    'market': market_key,
                    'side': side_key,
                    'line': line,
                    'prob': prob,
                    'label': f"{side_label} Over {line} {market_label}",
                    'referee': None
                })
            
            st.markdown("---")
        
        # Salvar no session state
        st.session_state.main_slip = main_slip_temp
        
        # Mostrar resumo do bilhete
        if st.session_state.main_slip:
            st.markdown("### 📊 Resumo do Bilhete Principal:")
            for i, sel in enumerate(st.session_state.main_slip, 1):
                prob_color = "green" if sel['prob'] >= 70 else "orange"
                st.markdown(f"{i}. {sel['home']} vs {sel['away']}: **{sel['label']}** - :{prob_color}[{sel['prob']:.1f}%]")
            
            prob_combinada = calculate_combined_probability(st.session_state.main_slip)
            st.metric("Probabilidade Combinada", f"{prob_combinada:.1f}%")
    
    # SEÇÃO 2: Gerar Hedges
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
            with st.spinner("⚙️ Gerando hedges inteligentes..."):
                hedge_1, hedge_2 = generate_hedge_bets(
                    st.session_state.main_slip, 
                    stats, 
                    refs_db
                )
                
                # Verificar se conseguiu gerar hedges válidos
                if not hedge_1 or not hedge_2:
                    st.warning("⚠️ Não foi possível gerar hedges mantendo 70%+ em todas seleções.")
                    st.info("💡 Tente reduzir o número de jogos ou escolher seleções com probabilidades mais altas.")
                else:
                    # Calcular stakes
                    stakes = {
                        'main': budget * 0.5,
                        'hedge_1': budget * 0.3,
                        'hedge_2': budget * 0.2
                    }
                    
                    # Renderizar resultados
                    st.success("✅ Hedges gerados com sucesso!")
                    
                    st.markdown("---")
                    
                    # BILHETE HEDGE #1
                    st.markdown("## 🤖 BILHETE HEDGE #1 (Cobertura A)")
                    
                    for i, sel in enumerate(hedge_1, 1):
                        prob_color = "green" if sel['prob'] >= 70 else "orange"
                        st.markdown(
                            f"{i}. {sel['home']} vs {sel['away']}: **{sel['label']}** "
                            f"- :{prob_color}[{sel['prob']:.1f}%] {sel['change']}"
                        )
                    
                    prob_h1 = calculate_combined_probability(hedge_1)
                    st.metric("Probabilidade Combinada", f"{prob_h1:.1f}%")
                    st.info(f"💵 Stake Sugerido: €{stakes['hedge_1']:.0f}")
                    
                    st.markdown("---")
                    
                    # BILHETE HEDGE #2
                    st.markdown("## 🤖 BILHETE HEDGE #2 (Cobertura B)")
                    
                    for i, sel in enumerate(hedge_2, 1):
                        prob_color = "green" if sel['prob'] >= 70 else "orange"
                        st.markdown(
                            f"{i}. {sel['home']} vs {sel['away']}: **{sel['label']}** "
                            f"- :{prob_color}[{sel['prob']:.1f}%] {sel['change']}"
                        )
                    
                    prob_h2 = calculate_combined_probability(hedge_2)
                    st.metric("Probabilidade Combinada", f"{prob_h2:.1f}%")
                    st.info(f"💵 Stake Sugerido: €{stakes['hedge_2']:.0f}")
                    
                    st.markdown("---")
                    
                    # ANÁLISE DE CENÁRIOS
                    st.markdown("## 📊 ANÁLISE DE COBERTURA")
                    
                    scenarios = calculate_coverage_scenarios(
                        st.session_state.main_slip,
                        hedge_1,
                        hedge_2,
                        stakes
                    )
                    
                    for scenario in scenarios:
                        if scenario['profit'] > 0:
                            st.success(
                                f"{scenario['scenario']}: **{scenario['probability']:.1f}%** prob "
                                f"→ Lucro de **€{scenario['profit']:.2f}** 💰"
                            )
                        else:
                            st.error(
                                f"{scenario['scenario']}: **{scenario['probability']:.1f}%** prob "
                                f"→ Perda de **€{abs(scenario['profit']):.2f}** 💸"
                            )
                    
                    # Probabilidade de sucesso
                    prob_ganho = sum(s['probability'] for s in scenarios if s['profit'] > 0)
                    st.metric("🎯 Probabilidade de GANHAR algo", f"{prob_ganho:.1f}%")
                    
                    if prob_ganho >= 85:
                        st.success("✅ Excelente cobertura! Mais de 85% de chance de lucro.")
                    elif prob_ganho >= 70:
                        st.info("👍 Boa cobertura! Risco controlado.")
                    else:
                        st.warning("⚠️ Cobertura moderada. Considere ajustar as seleções.")


def main():
    st.title("⚽ FutPrevisão V14.5 + Hedge")
    st.caption("Histórico 100% verificado + Sistema de Cobertura Inteligente")
    
    with st.spinner("Carregando bases..."):
        DEBUG_LOGS.clear()
        stats = learn_stats_v14()
        refs = load_referees_v14()
        calendar = load_calendar_safe()
        all_dfs = load_all_dataframes()
    
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

    tab1, tab2, tab3 = st.tabs(["📅 Calendário", "🧪 Simulação Manual", "🎰 Bet Builder + Hedge"])
    
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
    
    with tab3:
        render_hedge_builder_tab(stats, refs)

if __name__ == "__main__":
    main()
