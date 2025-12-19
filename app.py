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
# CARREGAMENTO INTELIGENTE (MANTIDO DO V14.5 ORIGINAL)
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
# CÁLCULO E MATEMÁTICA V14 (MANTIDO)
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
# 🆕 SISTEMA DE HEDGE BETTING (ATUALIZADO PARA 2 SELEÇÕES POR JOGO)
# ═══════════════════════════════════════════════════════════════════════════

def generate_hedge_bets_v2(main_slip: List[Dict], stats: Dict, refs_db: Dict) -> tuple:
    """
    Gera 2 bilhetes de hedge mantendo os MESMOS JOGOS mas com SELEÇÕES DIFERENTES.
    
    NOVO: Agora trabalha com 2 seleções por jogo (total de 6 seleções para 3 jogos)
    """
    
    hedge_1 = []
    hedge_2 = []
    
    # Agrupar seleções por jogo
    games = {}
    for sel in main_slip:
        game_id = sel['game_id']
        if game_id not in games:
            games[game_id] = []
        games[game_id].append(sel)
    
    # Para cada jogo, gerar 2 variações (uma para cada hedge)
    for game_id, selections in games.items():
        # Pegar primeiro jogo como referência
        ref_sel = selections[0]
        
        # Calcular resultado completo do jogo
        res = calcular_jogo_v14(ref_sel['home'], ref_sel['away'], stats, None, refs_db)
        if 'error' in res:
            continue
            
        probs = get_detailed_probs(res)
        
        # HEDGE #1: Priorizaintermediário troca de mercado + total
        h1_options = generate_hedge_options_for_game(selections, probs, res, game_id, strategy=1)
        hedge_1.extend(h1_options)
        
        # HEDGE #2: Prioriza linha baixa + lado oposto
        h2_options = generate_hedge_options_for_game(selections, probs, res, game_id, strategy=2)
        hedge_2.extend(h2_options)
    
    return hedge_1, hedge_2


def generate_hedge_options_for_game(main_selections: List[Dict], probs: Dict, game_res: Dict, game_id: int, strategy: int) -> List[Dict]:
    """
    Gera 2 seleções alternativas para um jogo específico.
    
    strategy=1: Individual→Total, Escanteios→Cartões
    strategy=2: Linha Baixa, Casa→Fora
    """
    
    hedge_options = []
    
    for main_sel in main_selections:
        market = main_sel['market']
        side = main_sel['side']
        line = main_sel['line']
        
        best_option = None
        
        if strategy == 1:
            # Estratégia 1: Mudar para total ou trocar mercado
            
            # Tentar total primeiro
            if side in ['home', 'away']:
                if market == 'corners':
                    for total_line in [10.5, 9.5, 8.5]:
                        prob = probs['corners']['total'].get(f'Over {int(total_line)}.5', 0)
                        if prob >= 70:
                            best_option = {
                                'home': game_res['home'],
                                'away': game_res['away'],
                                'market': 'corners',
                                'side': 'total',
                                'line': total_line,
                                'prob': prob,
                                'label': f"Total Over {total_line} escanteios",
                                'change': '🔄 individual→total',
                                'game_id': game_id
                            }
                            break
                
                elif market == 'cards':
                    for total_line in [4.5, 3.5]:
                        prob = probs['cards']['total'].get(f'Over {int(total_line)}.5', 0)
                        if prob >= 70:
                            best_option = {
                                'home': game_res['home'],
                                'away': game_res['away'],
                                'market': 'cards',
                                'side': 'total',
                                'line': total_line,
                                'prob': prob,
                                'label': f"Total Over {total_line} cartões",
                                'change': '🔄 individual→total',
                                'game_id': game_id
                            }
                            break
            
            # Se não conseguiu total, tenta trocar mercado
            if not best_option:
                if market == 'corners':
                    # Mudar para cartões
                    if side == 'home':
                        prob = probs['cards']['home'].get('Over 1.5', 0)
                        if prob >= 70:
                            best_option = {
                                'home': game_res['home'],
                                'away': game_res['away'],
                                'market': 'cards',
                                'side': 'home',
                                'line': 1.5,
                                'prob': prob,
                                'label': f"{game_res['home']} Over 1.5 cartões",
                                'change': '🔄 escanteios→cartões',
                                'game_id': game_id
                            }
                    elif side == 'away':
                        prob = probs['cards']['away'].get('Over 1.5', 0)
                        if prob >= 70:
                            best_option = {
                                'home': game_res['home'],
                                'away': game_res['away'],
                                'market': 'cards',
                                'side': 'away',
                                'line': 1.5,
                                'prob': prob,
                                'label': f"{game_res['away']} Over 1.5 cartões",
                                'change': '🔄 escanteios→cartões',
                                'game_id': game_id
                            }
                
                elif market == 'cards':
                    # Mudar para escanteios
                    if side == 'home':
                        for corner_line in [3.5, 4.5]:
                            prob = probs['corners']['home'].get(f'Over {corner_line}', 0)
                            if prob >= 70:
                                best_option = {
                                    'home': game_res['home'],
                                    'away': game_res['away'],
                                    'market': 'corners',
                                    'side': 'home',
                                    'line': corner_line,
                                    'prob': prob,
                                    'label': f"{game_res['home']} Over {corner_line} escanteios",
                                    'change': '🔄 cartões→escanteios',
                                    'game_id': game_id
                                }
                                break
                    elif side == 'away':
                        for corner_line in [2.5, 3.5]:
                            prob = probs['corners']['away'].get(f'Over {corner_line}', 0)
                            if prob >= 70:
                                best_option = {
                                    'home': game_res['home'],
                                    'away': game_res['away'],
                                    'market': 'corners',
                                    'side': 'away',
                                    'line': corner_line,
                                    'prob': prob,
                                    'label': f"{game_res['away']} Over {corner_line} escanteios",
                                    'change': '🔄 cartões→escanteios',
                                    'game_id': game_id
                                }
                                break
        
        elif strategy == 2:
            # Estratégia 2: Baixar linha ou trocar lado
            
            # Tentar linha mais baixa primeiro
            if market == 'corners':
                for lower_line in [2.5, 3.5, 4.5]:
                    if lower_line < line:
                        prob = probs['corners'][side].get(f'Over {lower_line}', 0) if side != 'total' else probs['corners']['total'].get(f'Over {int(lower_line)}.5', 0)
                        if prob >= 70:
                            side_label = game_res['home'] if side == 'home' else game_res['away'] if side == 'away' else "Total"
                            best_option = {
                                'home': game_res['home'],
                                'away': game_res['away'],
                                'market': 'corners',
                                'side': side,
                                'line': lower_line,
                                'prob': prob,
                                'label': f"{side_label} Over {lower_line} escanteios",
                                'change': f'🔽 {line}→{lower_line}',
                                'game_id': game_id
                            }
                            break
            
            elif market == 'cards':
                for lower_line in [1.5]:
                    if lower_line < line:
                        prob = probs['cards'][side].get(f'Over {lower_line}', 0) if side != 'total' else probs['cards']['total'].get(f'Over {int(lower_line)}.5', 0)
                        if prob >= 70:
                            side_label = game_res['home'] if side == 'home' else game_res['away'] if side == 'away' else "Total"
                            best_option = {
                                'home': game_res['home'],
                                'away': game_res['away'],
                                'market': 'cards',
                                'side': side,
                                'line': lower_line,
                                'prob': prob,
                                'label': f"{side_label} Over {lower_line} cartões",
                                'change': f'🔽 {line}→{lower_line}',
                                'game_id': game_id
                            }
                            break
            
            # Se não conseguiu linha baixa, tenta trocar lado
            if not best_option and side in ['home', 'away']:
                opposite_side = 'away' if side == 'home' else 'home'
                opposite_team = game_res['away'] if side == 'home' else game_res['home']
                
                # Ajustar linha (visitante = -1.0 em escanteios)
                if market == 'corners' and opposite_side == 'away':
                    adjusted_line = max(2.5, line - 1.0)
                elif market == 'corners' and opposite_side == 'home':
                    adjusted_line = line + 1.0
                else:
                    adjusted_line = line
                
                prob = probs[market][opposite_side].get(f'Over {adjusted_line}', 0)
                if prob >= 70:
                    market_label = 'escanteios' if market == 'corners' else 'cartões'
                    best_option = {
                        'home': game_res['home'],
                        'away': game_res['away'],
                        'market': market,
                        'side': opposite_side,
                        'line': adjusted_line,
                        'prob': prob,
                        'label': f"{opposite_team} Over {adjusted_line} {market_label}",
                        'change': '🔄 trocou lado',
                        'game_id': game_id
                    }
        
        # Fallback: repetir se >= 80%
        if not best_option and main_sel['prob'] >= 80:
            best_option = {**main_sel, 'change': '✅ mantido (alta confiança)'}
        
        # Adicionar opção encontrada
        if best_option:
            hedge_options.append(best_option)
    
    return hedge_options


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
    odd_main = (100 / p_main) * 0.85 if p_main > 0 else 2.0
    profit_main = (stakes['main'] * odd_main) - sum(stakes.values())
    
    scenarios.append({
        'scenario': '✅ Bilhete Principal ACERTA',
        'probability': p_main,
        'profit': profit_main,
        'color': 'green'
    })
    
    # Cenário 2: Hedge #1 acerta
    p_h1 = calculate_combined_probability(hedge_1)
    odd_h1 = (100 / p_h1) * 0.85 if p_h1 > 0 else 2.0
    profit_h1 = (stakes['hedge_1'] * odd_h1) - sum(stakes.values())
    
    scenarios.append({
        'scenario': '⚠️ Hedge #1 ACERTA',
        'probability': p_h1,
        'profit': profit_h1,
        'color': 'orange'
    })
    
    # Cenário 3: Hedge #2 acerta
    p_h2 = calculate_combined_probability(hedge_2)
    odd_h2 = (100 / p_h2) * 0.85 if p_h2 > 0 else 2.0
    profit_h2 = (stakes['hedge_2'] * odd_h2) - sum(stakes.values())
    
    scenarios.append({
        'scenario': '🟡 Hedge #2 ACERTA',
        'probability': p_h2,
        'profit': profit_h2,
        'color': 'blue'
    })
    
    # Cenário 4: Todos erram
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


def render_hedge_builder_tab_v2(stats, refs_db):
    """🆕 Nova aba: Bet Builder + Hedge System - 2 SELEÇÕES POR JOGO"""
    
    st.markdown("## 🎰 Bet Builder + Sistema de Cobertura")
    st.caption("Monte seu bilhete principal (2 seleções por jogo) e o sistema gera 2 hedges automáticos")
    
    st.markdown("---")
    
    # SEÇÃO 1: Input do Bilhete Principal
    with st.expander("📋 BILHETE PRINCIPAL", expanded=True):
        st.markdown("**Escolha seus jogos (2 seleções por jogo):**")
        
        # Inicializar session state
        if 'main_slip' not in st.session_state:
            st.session_state.main_slip = []
        
        lista_times = sorted(list(stats.keys()))
        
        num_games = st.number_input("Quantos jogos?", 1, 5, 3, key="num_games")
        
        main_slip_temp = []
        
        for i in range(num_games):
            st.markdown(f"### ⚽ Jogo {i+1}")
            
            # Seleção dos times
            col_teams = st.columns(2)
            with col_teams[0]:
                home = st.selectbox(f"Time Casa", lista_times, key=f"home_{i}")
            with col_teams[1]:
                away = st.selectbox(f"Time Visitante", lista_times, key=f"away_{i}", 
                                   index=min(i+1, len(lista_times)-1))
            
            # Calcular probabilidades do jogo UMA VEZ
            res = calcular_jogo_v14(home, away, stats, None, refs_db)
            if 'error' in res:
                st.error(f"Erro ao calcular jogo: {res['error']}")
                continue
                
            probs = get_detailed_probs(res)
            
            # ═══════════════════════════════════════════════════════════
            # SELEÇÃO #1
            # ═══════════════════════════════════════════════════════════
            st.markdown("#### 🎯 Seleção #1")
            col1_1, col1_2, col1_3 = st.columns(3)
            
            with col1_1:
                market_1 = st.selectbox("Mercado", 
                    ["Escanteios", "Cartões"], key=f"market_1_{i}")
            
            with col1_2:
                side_1 = st.selectbox("Lado", 
                    ["Casa", "Fora", "Total"], key=f"side_1_{i}")
            
            with col1_3:
                if market_1 == "Escanteios":
                    line_options_1 = [2.5, 3.5, 4.5, 5.5] if side_1 != "Total" else [8.5, 9.5, 10.5, 11.5]
                else:
                    line_options_1 = [1.5, 2.5] if side_1 != "Total" else [3.5, 4.5, 5.5]
                
                line_1 = st.selectbox("Linha", line_options_1, key=f"line_1_{i}")
            
            # Calcular probabilidade seleção #1
            market_key_1 = 'corners' if market_1 == "Escanteios" else 'cards'
            side_key_1 = 'home' if side_1 == "Casa" else 'away' if side_1 == "Fora" else 'total'
            
            if side_key_1 == 'total':
                prob_1 = probs[market_key_1]['total'].get(f'Over {int(line_1)}.5', 0)
            else:
                prob_1 = probs[market_key_1][side_key_1].get(f'Over {line_1}', 0)
            
            color_1 = "green" if prob_1 >= 70 else "orange" if prob_1 >= 60 else "red"
            st.markdown(f"Probabilidade: :{color_1}[**{prob_1:.1f}%**]")
            
            side_label_1 = res['home'] if side_1 == "Casa" else res['away'] if side_1 == "Fora" else "Total"
            market_label_1 = "escanteios" if market_1 == "Escanteios" else "cartões"
            
            main_slip_temp.append({
                'home': home,
                'away': away,
                'market': market_key_1,
                'side': side_key_1,
                'line': line_1,
                'prob': prob_1,
                'label': f"{side_label_1} Over {line_1} {market_label_1}",
                'referee': None,
                'game_id': i
            })
            
            # ═══════════════════════════════════════════════════════════
            # SELEÇÃO #2
            # ═══════════════════════════════════════════════════════════
            st.markdown("#### 🎯 Seleção #2")
            col2_1, col2_2, col2_3 = st.columns(3)
            
            with col2_1:
                market_2 = st.selectbox("Mercado", 
                    ["Escanteios", "Cartões"], key=f"market_2_{i}")
            
            with col2_2:
                side_2 = st.selectbox("Lado", 
                    ["Casa", "Fora", "Total"], key=f"side_2_{i}")
            
            with col2_3:
                if market_2 == "Escanteios":
                    line_options_2 = [2.5, 3.5, 4.5, 5.5] if side_2 != "Total" else [8.5, 9.5, 10.5, 11.5]
                else:
                    line_options_2 = [1.5, 2.5] if side_2 != "Total" else [3.5, 4.5, 5.5]
                
                line_2 = st.selectbox("Linha", line_options_2, key=f"line_2_{i}")
            
            # Calcular probabilidade seleção #2
            market_key_2 = 'corners' if market_2 == "Escanteios" else 'cards'
            side_key_2 = 'home' if side_2 == "Casa" else 'away' if side_2 == "Fora" else 'total'
            
            if side_key_2 == 'total':
                prob_2 = probs[market_key_2]['total'].get(f'Over {int(line_2)}.5', 0)
            else:
                prob_2 = probs[market_key_2][side_key_2].get(f'Over {line_2}', 0)
            
            color_2 = "green" if prob_2 >= 70 else "orange" if prob_2 >= 60 else "red"
            st.markdown(f"Probabilidade: :{color_2}[**{prob_2:.1f}%**]")
            
            side_label_2 = res['home'] if side_2 == "Casa" else res['away'] if side_2 == "Fora" else "Total"
            market_label_2 = "escanteios" if market_2 == "Escanteios" else "cartões"
            
            main_slip_temp.append({
                'home': home,
                'away': away,
                'market': market_key_2,
                'side': side_key_2,
                'line': line_2,
                'prob': prob_2,
                'label': f"{side_label_2} Over {line_2} {market_label_2}",
                'referee': None,
                'game_id': i
            })
            
            st.markdown("---")
        
        # Salvar no session state
        st.session_state.main_slip = main_slip_temp
        
        # Mostrar resumo do bilhete
        if st.session_state.main_slip:
            st.markdown("### 📊 Resumo do Bilhete Principal:")
            
            # Agrupar por jogo
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
                    st.markdown(f"  {j}. **{sel['label']}** - :{prob_color}[{sel['prob']:.1f}%]")
            
            prob_combinada = calculate_combined_probability(st.session_state.main_slip)
            st.metric("Probabilidade Combinada", f"{prob_combinada:.1f}%")
            st.caption(f"Total de seleções: {len(st.session_state.main_slip)}")
    
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
                hedge_1, hedge_2 = generate_hedge_bets_v2(
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
                    
                    # Agrupar por jogo
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
                            st.markdown(
                                f"  {j}. **{sel['label']}** - :{prob_color}[{sel['prob']:.1f}%] {sel['change']}"
                            )
                    
                    prob_h1 = calculate_combined_probability(hedge_1)
                    st.metric("Probabilidade Combinada", f"{prob_h1:.1f}%")
                    st.info(f"💵 Stake Sugerido: €{stakes['hedge_1']:.0f}")
                    
                    st.markdown("---")
                    
                    # BILHETE HEDGE #2
                    st.markdown("## 🤖 BILHETE HEDGE #2 (Cobertura B)")
                    
                    # Agrupar por jogo
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
                            st.markdown(
                                f"  {j}. **{sel['label']}** - :{prob_color}[{sel['prob']:.1f}%] {sel['change']}"
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
    st.title("⚽ FutPrevisão V14.5 + Hedge v2")
    st.caption("Histórico 100% verificado + Sistema de Cobertura (2 seleções/jogo)")
    
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
        render_hedge_builder_tab_v2(stats, refs_db)

if __name__ == "__main__":
    main()
