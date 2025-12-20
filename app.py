"""
╔═══════════════════════════════════════════════════════════════════════════╗
║          FUTPREVISÃO V14.9 - STRUCTURE MIRROR (HEDGE PERFEITO)            ║
║                          Sistema de Análise de Apostas                     ║
║                                                                            ║
║  Versão: V14.9 (Correção de Hedge e Linhas de Cartões Totais)             ║
║  Data: Dezembro 2025                                                      ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import math
import numpy as np
import os
from typing import Dict, List, Any
from difflib import get_close_matches
from datetime import datetime

# Configuração da Página
st.set_page_config(
    page_title="FutPrevisão V14.9",
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

def normalize_name(name: str, db_keys: list) -> str:
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

    return {
        'home': h_norm, 'away': a_norm, 'referee': ref,
        'league_h': s_h.get('league'), 'league_a': s_a.get('league'),
        'corners': {'total': corn_h + corn_a, 'h': corn_h, 'a': corn_a},
        'cards': {'total': card_h + card_a, 'h': card_h, 'a': card_a},
        'goals': {'h': (s_h['goals_f'] * s_a['goals_a'])/1.3, 'a': (s_a['goals_f'] * s_h['goals_a'])/1.3},
        'meta': {
            'shots_h': shots_h, 'p_label_h': "ALTO 🔥" if p_h > 1.0 else "BAIXO ⚪",
            'strict_val': strict, 'prob_red': prob_red, 'prob_red_lbl': pr_lbl
        }
    }

def get_detailed_probs(pred: Dict) -> Dict:
    def p(k, l): return sum((l**i * math.exp(-l)) / math.factorial(i) for i in range(k + 1))
    cH, cA = pred['corners']['h'], pred['corners']['a']
    kH, kA = pred['cards']['h'], pred['cards']['a']
    
    return {
        'corners': {
            'total': {f"Over {i}.5": (1-p(i, cH+cA))*100 for i in range(8, 13)},
            'home': {'Over 3.5': (1-p(3, cH))*100, 'Over 4.5': (1-p(4, cH))*100},
            'away': {'Over 3.5': (1-p(3, cA))*100, 'Over 4.5': (1-p(4, cA))*100}
        },
        'cards': {
            'total': {f"Over {i}.5": (1-p(i, kH+kA))*100 for i in range(2, 6)}, # Agora inclui 2.5
            'home': {'Over 1.5': (1-p(1, kH))*100, 'Over 2.5': (1-p(2, kH))*100},
            'away': {'Over 1.5': (1-p(1, kA))*100, 'Over 2.5': (1-p(2, kA))*100}
        }
    }

# ═══════════════════════════════════════════════════════════════════════════
# 3. LÓGICA DE BET BUILDER & HEDGE DUPLO
# ═══════════════════════════════════════════════════════════════════════════

def generate_bet_options(home_team: str, away_team: str, probs: Dict) -> List[Dict]:
    options = []
    # Corners
    for line in [2.5, 3.5, 4.5, 5.5]:
        options.append({'label': f"{home_team} Over {line} cantos", 'prob': probs['corners']['home'].get(f'Over {line}', 0), 'market':'corners'})
    for line in [2.5, 3.5, 4.5]:
        options.append({'label': f"{away_team} Over {line} cantos", 'prob': probs['corners']['away'].get(f'Over {line}', 0), 'market':'corners'})
    for line in [8.5, 9.5, 10.5]:
        options.append({'label': f"Total Over {line} cantos", 'prob': probs['corners']['total'].get(f'Over {int(line)}.5', 0), 'market':'corners'})
    
    # Cards (Incluindo Totais)
    for line in [1.5, 2.5]:
        options.append({'label': f"{home_team} Over {line} cartões", 'prob': probs['cards']['home'].get(f'Over {line}', 0), 'market':'cards'})
        options.append({'label': f"{away_team} Over {line} cartões", 'prob': probs['cards']['away'].get(f'Over {line}', 0), 'market':'cards'})
    
    # NOVAS LINHAS DE TOTAIS DE CARTÕES
    for line in [2.5, 3.5, 4.5, 5.5]:
        key = f'Over {int(line)}.5'
        val = probs['cards']['total'].get(key, 0)
        options.append({'label': f"Total Over {line} cartões", 'prob': val, 'market':'cards'})

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
    
    # Agrupar seleções por jogo
    games = {}
    for sel in main_slip:
        gid = sel['game_id']
        if gid not in games: games[gid] = []
        games[gid].append(sel)
        
    # Para cada jogo do bilhete principal...
    for gid, sels in games.items():
        home, away = sels[0]['home'], sels[0]['away']
        res = calcular_jogo_v14(home, away, stats, None, refs_db)
        if 'error' in res: continue
        
        probs = get_detailed_probs(res)
        all_opts = generate_bet_options(home, away, probs)
        
        # 1. Tenta filtrar opções >= 70%
        valid_opts = [o for o in all_opts if o['prob'] >= 70]
        
        # 2. Se não tiver opções suficientes, pega as melhores disponíveis (FALLBACK)
        if len(valid_opts) < 4:
            valid_opts = all_opts[:6] # Pega as top 6, independente da %
        
        main_labels = [s['label'] for s in sels]
        
        # --- HEDGE 1: Opções diferentes das principais ---
        h1_candidates = [o for o in valid_opts if o['label'] not in main_labels]
        # Pega as 2 melhores
        current_h1 = h1_candidates[:2]
        
        # Garante 2 opções (mesmo que tenha que repetir se estiver desesperado)
        if len(current_h1) < 2:
            remaining = [o for o in valid_opts if o not in current_h1]
            current_h1.extend(remaining[:2-len(current_h1)])
            
        for opt in current_h1:
            hedge1.append({**opt, 'game_id': gid, 'home': home, 'away': away, 'type': 'Variação'})
            
        # --- HEDGE 2: Opções diferentes de H1 (pode repetir Main se for >80%) ---
        h1_labels = [o['label'] for o in current_h1]
        h2_candidates = []
        for o in valid_opts:
            if o['label'] in h1_labels: continue # Evita H1
            if o['label'] in main_labels and o['prob'] < 80: continue # Evita Main fraca
            h2_candidates.append(o)
            
        current_h2 = h2_candidates[:2]
        if len(current_h2) < 2: # Completa se faltar
             remaining = [o for o in valid_opts if o not in current_h2 and o not in current_h1]
             current_h2.extend(remaining[:2-len(current_h2)])
             
        for opt in current_h2:
            hedge2.append({**opt, 'game_id': gid, 'home': home, 'away': away, 'type': 'Alternativa'})
            
    return hedge1, hedge2

def render_bet_builder_tab(stats, refs_db):
    st.markdown("## 🎰 Bet Builder (Structure Mirror)")
    st.caption("Cada jogo do principal terá 2 coberturas nos Hedges (Total de Cartões incluído)")
    
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
        opt_fmt = [f"{o['label']} ({o['prob']:.0f}%)" for o in opts]
        
        s1 = st.selectbox(f"Seleção 1", range(len(opts)), format_func=lambda x: opt_fmt[x], key=f"s1_{i}")
        s2 = st.selectbox(f"Seleção 2", range(len(opts)), format_func=lambda x: opt_fmt[x], key=f"s2_{i}", index=min(1, len(opts)-1))
        
        temp_slip.append({**opts[s1], 'game_id': i, 'home': h, 'away': a})
        temp_slip.append({**opts[s2], 'game_id': i, 'home': h, 'away': a})
    
    st.session_state.main_slip = temp_slip
    
    if st.button("🔮 Gerar Hedges (Espelho)", type="primary"):
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
            st.warning("🛡️ **Hedge #1**")
            games_seen = []
            for s in h1:
                if s['game_id'] not in games_seen:
                    st.caption(f"{s['home']} x {s['away']}")
                    games_seen.append(s['game_id'])
                st.write(f"- {s['label']}")
            st.metric("Prob Comb.", f"{calculate_combined_probability(h1):.1f}%")

        with c_h2:
            st.success("🔄 **Hedge #2**")
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
    # Função simplificada de exibição manual
    m = res['meta']
    probs = get_detailed_probs(res)
    st.markdown("---")
    st.subheader(f"🏠 {res['home']} vs ✈️ {res['away']}")
    
    c1, c2 = st.columns(2)
    with c1:
        st.info("🏁 **Escanteios**")
        st.write(f"Total Esp: {res['corners']['total']:.1f}")
        for k,v in probs['corners']['total'].items(): 
            if v>65: st.write(f"{k}: {v:.0f}%")
        
        h35 = get_native_history(res['home'], res['league_h'], 'corners', 3.5, 'home', all_dfs)
        p35h = probs['corners']['home']['Over 3.5']
        st.write(f"🏠 Over 3.5: {p35h:.0f}% | Hist: {h35}")
        
    with c2:
        st.warning("🟨 **Cartões**")
        st.write(f"Total Esp: {res['cards']['total']:.1f}")
        for k,v in probs['cards']['total'].items(): 
            if v>60: st.write(f"{k}: {v:.0f}%")
            
        h15 = get_native_history(res['home'], res['league_h'], 'cards', 1.5, 'home', all_dfs)
        p15h = probs['cards']['home']['Over 1.5']
        st.write(f"🏠 Over 1.5: {p15h:.0f}% | Hist: {h15}")

def main():
    st.title("⚽ FutPrevisão V14.9 (Structure Mirror)")
    
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
        l = sorted(list(stats.keys()))
        c1, c2 = st.columns(2)
        h = c1.selectbox("Casa", l, index=0)
        a = c2.selectbox("Fora", l, index=1)
        if st.button("Simular"):
            res = calcular_jogo_v14(h, a, stats, None, refs)
            if 'error' not in res: render_result_v14_5(res, all_dfs)
            
    with tab3:
        render_bet_builder_tab(stats, refs)

if __name__ == "__main__":
    main()
