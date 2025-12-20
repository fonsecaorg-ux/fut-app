"""
╔═══════════════════════════════════════════════════════════════════════════╗
║       FUTPREVISÃO V14.12 - TEMPLATE HEDGE (DUPLA CHANCE & MIX)            ║
║                          Sistema de Análise de Apostas                     ║
║                                                                            ║
║  Versão: V14.12                                                           ║
║  Data: Dezembro 2025                                                      ║
║  Novidade: Hedges baseados em Templates (DC+Stat ou Canto+Cartão)         ║
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
    page_title="FutPrevisão V14.12",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTES
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
# 2. MOTOR DE CÁLCULO
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
    s_lbl = "MUITO RIGOROSO 🔴" if strict == 1.15 else "RIGOROSO 🟠" if strict == 1.08 else "NORMAL 🟢"
    
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
            'strict_val': strict, 'strict_lbl': s_lbl, 'prob_red': prob_red, 'prob_red_lbl': pr_lbl,
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
            '12': (p_home + p_away) * 100
        }
    }

# ═══════════════════════════════════════════════════════════════════════════
# 3. LÓGICA DE HEDGE REFINADA (TEMPLATES)
# ═══════════════════════════════════════════════════════════════════════════

def generate_bet_options(home_team: str, away_team: str, probs: Dict) -> List[Dict]:
    options = []
    
    # 1. ESCANTEIOS
    for line in [3.5, 4.5, 5.5]:
        options.append({'label': f"{home_team} Over {line} cantos", 'prob': probs['corners']['home'].get(f'Over {line}', 0), 'market':'corners', 'side':'home'})
    for line in [3.5, 4.5]:
        options.append({'label': f"{away_team} Over {line} cantos", 'prob': probs['corners']['away'].get(f'Over {line}', 0), 'market':'corners', 'side':'away'})
    for line in [8.5, 9.5, 10.5]:
        options.append({'label': f"Total Over {line} cantos", 'prob': probs['corners']['total'].get(f'Over {int(line)}.5', 0), 'market':'corners', 'side':'total'})
    
    # 2. CARTÕES
    for line in [1.5, 2.5]:
        options.append({'label': f"{home_team} Over {line} cartões", 'prob': probs['cards']['home'].get(f'Over {line}', 0), 'market':'cards', 'side':'home'})
        options.append({'label': f"{away_team} Over {line} cartões", 'prob': probs['cards']['away'].get(f'Over {line}', 0), 'market':'cards', 'side':'away'})
    for line in [2.5, 3.5, 4.5, 5.5]:
        val = probs['cards']['total'].get(f'Over {int(line)}.5', 0)
        options.append({'label': f"Total Over {line} cartões", 'prob': val, 'market':'cards', 'side':'total'})

    # 3. DUPLA CHANCE
    options.append({'label': f"Dupla Chance: {home_team} ou Empate", 'prob': probs['chance']['1X'], 'market':'chance', 'side':'home'})
    options.append({'label': f"Dupla Chance: {away_team} ou Empate", 'prob': probs['chance']['X2'], 'market':'chance', 'side':'away'})
    options.append({'label': f"Dupla Chance: {home_team} ou {away_team}", 'prob': probs['chance']['12'], 'market':'chance', 'side':'any'})

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
        
        # Filtra opções viáveis (>55% para hedge é ok se for dupla chance)
        valid_opts = [o for o in all_opts if o['prob'] >= 55]
        
        main_labels = [s['label'] for s in sels]
        
        # ══════════════════════════════════════════════════════════════════
        # HEDGE #1: TEMPLATE [DUPLA CHANCE] + [ESTATÍSTICA (Cartão/Canto)]
        # Foco: Segurança do Resultado
        # ══════════════════════════════════════════════════════════════════
        h1_pair = []
        
        # 1. Achar melhor Dupla Chance (que não esteja no principal)
        dc_opts = [o for o in valid_opts if o['market'] == 'chance' and o['label'] not in main_labels]
        if dc_opts:
            h1_pair.append(dc_opts[0])
        else:
            # Fallback: Se não tem DC, pega a aposta mais segura de qualquer tipo
            safe_opts = [o for o in valid_opts if o['label'] not in main_labels]
            if safe_opts: h1_pair.append(safe_opts[0])
            
        # 2. Achar estatística complementar (Total Cartões, Total Cantos, Individual)
        # Tenta evitar o mesmo mercado da Principal se possível, mas prioriza probabilidade
        stat_opts = [o for o in valid_opts if o['market'] in ['corners', 'cards'] and o['label'] not in main_labels and o not in h1_pair]
        
        # Filtra para evitar conflito direto (ex: Over 9.5 cantos se principal já tem Over Cantos)
        # Mas aqui vamos permitir desde que não seja a mesma aposta
        if stat_opts:
            h1_pair.append(stat_opts[0])
            
        # Adicionar ao Hedge 1
        for opt in h1_pair:
            hedge1.append({**opt, 'game_id': gid, 'home': home, 'away': away, 'type': 'Safety'})

        # ══════════════════════════════════════════════════════════════════
        # HEDGE #2: TEMPLATE [CANTO] + [CARTÃO] (Cruzamento Estatístico)
        # Foco: Diversificação de Mercados
        # ══════════════════════════════════════════════════════════════════
        h2_pair = []
        
        # Usados até agora (Main + H1)
        used_labels = main_labels + [o['label'] for o in h1_pair]
        
        # 1. Pegar melhor de Escanteios (que não foi usado)
        corn_opts = [o for o in valid_opts if o['market'] == 'corners' and o['label'] not in used_labels]
        if corn_opts:
            h2_pair.append(corn_opts[0])
            
        # 2. Pegar melhor de Cartões (que não foi usado)
        card_opts = [o for o in valid_opts if o['market'] == 'cards' and o['label'] not in used_labels]
        if card_opts:
            h2_pair.append(card_opts[0])
            
        # Fallback: Se faltou um dos dois, completa com o que tiver melhor
        if len(h2_pair) < 2:
            leftover = [o for o in valid_opts if o['label'] not in used_labels and o not in h2_pair]
            count_needed = 2 - len(h2_pair)
            h2_pair.extend(leftover[:count_needed])
            
        # Adicionar ao Hedge 2
        for opt in h2_pair:
            hedge2.append({**opt, 'game_id': gid, 'home': home, 'away': away, 'type': 'Mix'})
            
    return hedge1, hedge2

def render_bet_builder_tab(stats, refs_db):
    st.markdown("## 🎰 Bet Builder (Template Hedge)")
    st.caption("Geração de Hedges com estrutura fixa: Dupla Chance + Stat | Canto + Cartão")
    
    if 'main_slip' not in st.session_state: st.session_state.main_slip = []
    
    lista_times = sorted(list(stats.keys()))
    num_games = st.number_input("Quantos Jogos?", 1, 5, 3)
    
    temp_slip = []
    
    for i in range(num_games):
        st.markdown(f"---")
        st.markdown(f"### ⚽ Jogo {i+1}")
        c1, c2 = st.columns(2)
        h = c1.selectbox(f"Casa", lista_times, key=f"h_{i}")
        a = c2.selectbox(f"Fora", lista_times, key=f"a_{i}", index=min(1, len(lista_times)-1))
        
        res = calcular_jogo_v14(h, a, stats, None, refs_db)
        if 'error' in res: 
            st.error("Erro nos times")
            continue
            
        probs = get_detailed_probs(res)
        opts = generate_bet_options(h, a, probs)
        opt_fmt = [f"{o['label']}" for o in opts]
        
        s1 = st.selectbox(f"Seleção 1", range(len(opts)), format_func=lambda x: opt_fmt[x], key=f"s1_{i}")
        s2 = st.selectbox(f"Seleção 2", range(len(opts)), format_func=lambda x: opt_fmt[x], key=f"s2_{i}", index=min(1, len(opts)-1))
        
        temp_slip.append({**opts[s1], 'game_id': i, 'home': h, 'away': a})
        temp_slip.append({**opts[s2], 'game_id': i, 'home': h, 'away': a})
    
    st.session_state.main_slip = temp_slip
    
    if st.button("🔮 Gerar Hedges (Template)", type="primary"):
        h1, h2 = generate_dual_hedges(st.session_state.main_slip, stats, refs_db)
        
        st.success("✅ Hedges Gerados!")
        
        c_main, c_h1, c_h2 = st.columns(3)
        
        with c_main:
            st.info("📋 **Principal**")
            games_seen = []
            for s in st.session_state.main_slip:
                if s['game_id'] not in games_seen:
                    st.caption(f"{s['home']} x {s['away']}")
                    games_seen.append(s['game_id'])
                st.write(f"- {s['label']}")
            st.metric("Prob Comb.", f"{calculate_combined_probability(st.session_state.main_slip):.1f}%")
            
        with c_h1:
            st.warning("🛡️ **Hedge #1 (Segurança DC)**")
            st.caption("Foco: Dupla Chance + Estatística")
            games_seen = []
            for s in h1:
                if s['game_id'] not in games_seen:
                    st.caption(f"{s['home']} x {s['away']}")
                    games_seen.append(s['game_id'])
                st.write(f"- {s['label']}")
            st.metric("Prob Comb.", f"{calculate_combined_probability(h1):.1f}%")

        with c_h2:
            st.success("🔄 **Hedge #2 (Mix Stats)**")
            st.caption("Foco: Escanteios + Cartões")
            games_seen = []
            for s in h2:
                if s['game_id'] not in games_seen:
                    st.caption(f"{s['home']} x {s['away']}")
                    games_seen.append(s['game_id'])
                st.write(f"- {s['label']}")
            st.metric("Prob Comb.", f"{calculate_combined_probability(h2):.1f}%")

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
    
    # 🆕 RADAR DE RISCO
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
        st.warning("🟨 **Cartões Individuais**")
        st.markdown(f"**🏠 {res['home']}**")
        for line in [1.5, 2.5]:
            p = probs['cards']['home'].get(f'Over {line}', 0)
            h = get_native_history(res['home'], res['league_h'], 'cards', line, 'home', all_dfs)
            c = "green" if p >= 70 else "gray"
            st.markdown(f"Over {line}: :{c}[**{p:.0f}%**] | Hist: {h}")

def main():
    st.title("⚽ FutPrevisão V14.12 (Template Hedge)")
    
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
