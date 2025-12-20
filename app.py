"""
╔═══════════════════════════════════════════════════════════════════════════╗
║       FUTPREVISÃO V22.0 FINAL - SISTEMA COMPLETO DE ANÁLISE               ║
║                                                                            ║
║  ✅ Baseado no código original funcionando                                ║
║  ✅ Árbitro na simulação (CORRIGIDO)                                      ║
║  ✅ Previsões para AMBOS os times (CORRIGIDO)                             ║
║  ✅ Todas as funcionalidades profissionais                                ║
║                                                                            ║
║  Dezembro 2025 - Versão Final                                            ║
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
    page_title="FutPrevisão V22 Final",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session State
if 'bankroll' not in st.session_state:
    st.session_state.bankroll = 1000.0
if 'bet_history' not in st.session_state:
    st.session_state.bet_history = []

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTES
# ═══════════════════════════════════════════════════════════════════════════

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
    "Nott'm Forest": 'Nottm Forest', 'Nottingham Forest': 'Nottm Forest',
    'West Ham': 'West Ham', 'Leicester': 'Leicester',
    'Athletic Club': 'Ath Bilbao', 'Atl. Madrid': 'Ath Madrid'
}

LIGAS_ALVO = [
    "Premier League", "La Liga", "Serie A", "Bundesliga", "Ligue 1",
    "Championship", "Bundesliga 2", "Pro League", "Süper Lig", "Scottish Premiership"
]

# ═══════════════════════════════════════════════════════════════════════════
# CARREGAMENTO DE DADOS
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def find_and_load_csv(league_name: str) -> pd.DataFrame:
    """Encontra e carrega CSV da liga"""
    attempts = [
        f"{league_name} 25.26.csv",
        f"{league_name.replace(' ', '_')}_25_26.csv",
        f"{league_name}.csv"
    ]
    
    # Casos especiais
    if "Süper Lig" in league_name:
        attempts.extend(["Super Lig Turquia 25.26.csv", "Super_Lig_Turquia_25_26.csv"])
    if "Pro League" in league_name:
        attempts.append("Pro League Belgica 25.26.csv")
    if "Premiership" in league_name:
        attempts.append("Premiership Escocia 25.26.csv")
    if "Championship" in league_name:
        attempts.append("Championship Inglaterra 25.26.csv")

    for filename in attempts:
        if os.path.exists(filename):
            try:
                try:
                    df = pd.read_csv(filename, encoding='utf-8-sig')
                except:
                    df = pd.read_csv(filename, encoding='latin1')
                
                if not df.empty:
                    df.columns = [c.strip().replace('\ufeff', '') for c in df.columns]
                    
                    # Normalizar nomes
                    rename_map = {}
                    if 'Mandante' in df.columns:
                        rename_map['Mandante'] = 'HomeTeam'
                    if 'Visitante' in df.columns:
                        rename_map['Visitante'] = 'AwayTeam'
                    if 'Time_Casa' in df.columns:
                        rename_map['Time_Casa'] = 'HomeTeam'
                    if 'Time_Visitante' in df.columns:
                        rename_map['Time_Visitante'] = 'AwayTeam'
                    
                    if rename_map:
                        df = df.rename(columns=rename_map)
                    
                    df['_League_'] = league_name
                    return df
            except:
                pass
    
    return pd.DataFrame()

@st.cache_resource
def load_all_dataframes() -> Dict[str, pd.DataFrame]:
    """Carrega todos os dataframes"""
    dfs = {}
    for league in LIGAS_ALVO:
        df = find_and_load_csv(league)
        if not df.empty:
            dfs[league] = df
    return dfs

@st.cache_data(ttl=3600)
def learn_stats_v22() -> Dict[str, Dict[str, Any]]:
    """Aprende estatísticas dos times"""
    stats_db = {}
    
    for league in LIGAS_ALVO:
        df = find_and_load_csv(league)
        if df.empty:
            continue
        
        # Garantir colunas necessárias
        cols_needed = ['HomeTeam', 'AwayTeam', 'HC', 'AC', 'HY', 'AY', 'HF', 'AF', 
                      'FTHG', 'FTAG', 'HST', 'AST', 'HR', 'AR']
        for c in cols_needed:
            if c not in df.columns:
                df[c] = np.nan
        
        try:
            # Estatísticas casa
            h_stats = df.groupby('HomeTeam').agg({
                'HC': 'mean', 'HY': 'mean', 'HF': 'mean',
                'FTHG': 'mean', 'FTAG': 'mean',
                'HST': 'mean', 'HR': 'mean'
            }).fillna(value={
                'HST': DEFAULTS['shots_on_target'],
                'HR': DEFAULTS['red_cards_avg']
            })
            
            # Estatísticas fora
            a_stats = df.groupby('AwayTeam').agg({
                'AC': 'mean', 'AY': 'mean', 'AF': 'mean',
                'FTAG': 'mean', 'FTHG': 'mean',
                'AST': 'mean', 'AR': 'mean'
            }).fillna(value={
                'AST': DEFAULTS['shots_on_target'],
                'AR': DEFAULTS['red_cards_avg']
            })
            
            all_teams = set(h_stats.index) | set(a_stats.index)
            
            for team in all_teams:
                h = h_stats.loc[team] if team in h_stats.index else pd.Series(0, index=h_stats.columns)
                a = a_stats.loc[team] if team in a_stats.index else pd.Series(0, index=a_stats.columns)
                
                def w_avg(val_h, val_a, default=0):
                    if val_h == 0 and val_a == 0:
                        return default
                    if val_h == 0:
                        return val_a
                    if val_a == 0:
                        return val_h
                    return (val_h * 0.6) + (val_a * 0.4)
                
                stats_db[team] = {
                    'corners': w_avg(h.get('HC', 0), a.get('AC', 0), 5.0),
                    'cards': w_avg(h.get('HY', 0), a.get('AY', 0), 2.0),
                    'fouls': w_avg(h.get('HF', 0), a.get('AF', 0), 11.0),
                    'goals_f': w_avg(h.get('FTHG', 0), a.get('FTAG', 0), 1.2),
                    'goals_a': w_avg(h.get('FTAG', 0), a.get('FTHG', 0), 1.2),
                    'shots_on_target': w_avg(h.get('HST', 0), a.get('AST', 0), 4.5),
                    'red_cards_avg': w_avg(h.get('HR', 0), a.get('AR', 0), 0.08),
                    'league': league
                }
        except:
            pass
    
    return stats_db

@st.cache_data(ttl=3600)
def load_referees_v22() -> Dict[str, Dict[str, float]]:
    """Carrega dados dos árbitros"""
    refs_db = {}
    
    # Arquivo 1: arbitros_5_ligas_2025_2026.csv
    if os.path.exists("arbitros_5_ligas_2025_2026.csv"):
        try:
            df = pd.read_csv("arbitros_5_ligas_2025_2026.csv", encoding='utf-8-sig')
            df.columns = [c.strip().replace('\ufeff', '') for c in df.columns]
            
            for _, row in df.iterrows():
                nome = str(row['Arbitro']).strip()
                vermelhos = float(row.get('Cartoes_Vermelhos', 0))
                jogos = float(row.get('Jogos_Apitados', 1))
                media = float(row.get('Media_Cartoes_Por_Jogo', 4.0))
                amarelos = int(row.get('Cartoes_Amarelos', 0))
                
                red_rate = (vermelhos / jogos) if jogos > 0 else 0.08
                
                refs_db[nome] = {
                    'factor': media / 4.0,
                    'red_rate': red_rate,
                    'strictness': media,
                    'games': int(jogos),
                    'yellows': amarelos,
                    'reds': int(vermelhos)
                }
        except:
            pass
    
    # Arquivo 2: arbitros.csv
    if os.path.exists("arbitros.csv"):
        try:
            df = pd.read_csv("arbitros.csv", encoding='utf-8-sig')
            df.columns = [c.strip().replace('\ufeff', '') for c in df.columns]
            
            for _, row in df.iterrows():
                nome = str(row['Nome']).strip()
                if nome not in refs_db:
                    refs_db[nome] = {
                        'factor': float(row['Fator']),
                        'red_rate': 0.08,
                        'strictness': float(row['Fator']) * 4.0,
                        'games': 0,
                        'yellows': 0,
                        'reds': 0
                    }
        except:
            pass
    
    return refs_db

@st.cache_data(ttl=600)
def load_calendar_safe() -> pd.DataFrame:
    """Carrega calendário de jogos futuros"""
    fname = "calendario_ligas.csv"
    if not os.path.exists(fname):
        return pd.DataFrame()
    
    try:
        try:
            df = pd.read_csv(fname, encoding='utf-8-sig')
        except:
            df = pd.read_csv(fname, encoding='latin1')
        
        df.columns = [c.strip().replace('\ufeff', '') for c in df.columns]
        
        # Normalizar nomes
        rename_map = {}
        if 'Mandante' in df.columns:
            rename_map['Mandante'] = 'Time_Casa'
        if 'Visitante' in df.columns:
            rename_map['Visitante'] = 'Time_Visitante'
        
        if rename_map:
            df = df.rename(columns=rename_map)
        
        req = ['Data', 'Liga', 'Time_Casa', 'Time_Visitante', 'Hora']
        if not set(req).issubset(df.columns):
            return pd.DataFrame()
        
        df['DtObj'] = pd.to_datetime(df['Data'], format='%d/%m/%Y', errors='coerce')
        df = df.dropna(subset=['DtObj']).sort_values(by=['DtObj', 'Hora'])
        
        return df
    except:
        return pd.DataFrame()

# ═══════════════════════════════════════════════════════════════════════════
# MOTOR DE CÁLCULO
# ═══════════════════════════════════════════════════════════════════════════

def normalize_name(name: str, db_keys: list) -> Optional[str]:
    """Normaliza nome do time"""
    if not name:
        return None
    if name in NAME_MAPPING:
        name = NAME_MAPPING[name]
    if name in db_keys:
        return name
    
    matches = get_close_matches(name, db_keys, n=1, cutoff=0.6)
    return matches[0] if matches else None

def get_native_history(team_name: str, league: str, market: str, line: float, location: str, all_dfs: Dict) -> str:
    """Retorna histórico nativo (últimos 10 jogos)"""
    if league not in all_dfs:
        return "N/A"
    
    df = all_dfs[league]
    
    col_map = {
        ('home', 'corners'): 'HC',
        ('away', 'corners'): 'AC',
        ('home', 'cards'): 'HY',
        ('away', 'cards'): 'AY'
    }
    
    col_code = col_map.get((location, market))
    if not col_code:
        return "N/A"
    
    team_col = 'HomeTeam' if location == 'home' else 'AwayTeam'
    
    matches = df[df[team_col] == team_name]
    if matches.empty:
        return "0/0"
    
    last_matches = matches.tail(10)
    hits = sum(1 for val in last_matches[col_code] if float(val) > line)
    
    return f"{hits}/{len(last_matches)}"

def poisson(k: int, lamb: float) -> float:
    """Distribuição de Poisson"""
    if lamb > 30 or k > 20:
        return 0.0
    try:
        return (lamb**k * math.exp(-lamb)) / math.factorial(k)
    except:
        return 0.0

def monte_carlo(xg_h: float, xg_a: float, n: int = 1000) -> Tuple[float, float, float]:
    """Simulação Monte Carlo"""
    gh = np.random.poisson(max(0.1, xg_h), n)
    ga = np.random.poisson(max(0.1, xg_a), n)
    
    h_w = np.count_nonzero(gh > ga)
    a_w = np.count_nonzero(ga > gh)
    d = n - h_w - a_w
    
    return h_w/n, d/n, a_w/n

def calcular_jogo_v22(home: str, away: str, stats: Dict, ref: Optional[str], refs_db: Dict, all_dfs: Dict) -> Dict:
    """
    Motor de cálculo V22 - COMPLETO
    
    CORREÇÕES:
    - ✅ Calcula para AMBOS os times (home e away)
    - ✅ Suporte a árbitro
    - ✅ Histórico nativo
    - ✅ Monte Carlo
    """
    
    h_norm = normalize_name(home, list(stats.keys()))
    a_norm = normalize_name(away, list(stats.keys()))
    
    if not h_norm or not a_norm:
        return {'error': "Times não encontrados."}
    
    s_h = stats[h_norm]
    s_a = stats[a_norm]
    
    # Dados do árbitro
    if ref and ref in refs_db:
        r_data = refs_db[ref]
    else:
        r_data = {'factor': 1.0, 'red_rate': 0.08, 'strictness': 4.0, 'games': 0, 'yellows': 0, 'reds': 0}
    
    # Chutes (pressure)
    shots_h = s_h['shots_on_target']
    shots_a = s_a['shots_on_target']
    
    p_h = 1.20 if shots_h > 6.0 else 1.10 if shots_h > 4.5 else 1.0
    p_a = 1.20 if shots_a > 6.0 else 1.10 if shots_a > 4.5 else 1.0
    
    # Escanteios
    corn_h = s_h['corners'] * 1.15 * p_h
    corn_a = s_a['corners'] * 0.90 * p_a
    
    # Rigidez do árbitro
    rr = r_data['red_rate']
    strict = 1.15 if rr > 0.12 else 1.08 if rr > 0.08 else 1.0
    strict_label = "MUITO RIGOROSO 🔴" if strict == 1.15 else "RIGOROSO 🟠" if strict == 1.08 else "NORMAL 🟢"
    
    # Violência (faltas)
    viol_h = 1.0 if s_h['fouls'] > 12.5 else 0.85
    viol_a = 1.0 if s_a['fouls'] > 12.5 else 0.85
    
    # Cartões
    card_h = s_h['cards'] * viol_h * r_data['factor'] * strict
    card_a = s_a['cards'] * viol_a * r_data['factor'] * strict
    
    # Probabilidade de vermelho
    prob_red = ((s_h['red_cards_avg'] + s_a['red_cards_avg']) / 2) * rr * 100
    prob_red_label = "ALTA 🔴" if prob_red > 12 else "MÉDIA 🟠" if prob_red > 8 else "BAIXA 🟡"
    
    # xG e Monte Carlo
    xg_h = max(0.1, s_h['goals_f'])
    xg_a = max(0.1, s_a['goals_f'])
    
    mc_h, mc_d, mc_a = monte_carlo(xg_h, xg_a)
    
    # BTTS
    btts = (1 - poisson(0, xg_h)) * (1 - poisson(0, xg_a)) * 100
    
    # Over 2.5
    over_25 = (1 - sum([
        poisson(h, xg_h) * poisson(a, xg_a)
        for h in range(3) for a in range(3) if h+a < 3
    ])) * 100
    
    return {
        'home': h_norm,
        'away': a_norm,
        'league_h': s_h['league'],
        'league_a': s_a['league'],
        'corners': {
            'h': corn_h,
            'a': corn_a,
            'total': corn_h + corn_a
        },
        'cards': {
            'h': card_h,
            'a': card_a,
            'total': card_h + card_a
        },
        'goals': {
            'h': xg_h,
            'a': xg_a
        },
        'monte_carlo': {
            'h': mc_h * 100,
            'd': mc_d * 100,
            'a': mc_a * 100
        },
        'meta': {
            'shots_h': shots_h,
            'shots_a': shots_a,
            'pressure_h': p_h,
            'pressure_a': p_a,
            'fouls_h': s_h['fouls'],
            'fouls_a': s_a['fouls'],
            'violence_h': viol_h,
            'violence_a': viol_a,
            'referee': ref if ref else 'Neutro',
            'ref_factor': r_data['factor'],
            'ref_strictness': strict,
            'ref_label': strict_label,
            'ref_red_rate': rr,
            'prob_red': prob_red,
            'prob_red_label': prob_red_label,
            'ref_games': r_data.get('games', 0),
            'ref_yellows': r_data.get('yellows', 0),
            'ref_reds': r_data.get('reds', 0)
        },
        'probs': {
            'btts': btts,
            'over_2_5': over_25
        }
    }

def get_detailed_probs(res: Dict) -> Dict:
    """Probabilidades detalhadas"""
    
    def sim_prob(avg: float, line: float) -> float:
        return max(5, min(95, 50 + (avg - line) * 15))
    
    probs = {
        'corners': {
            'home': {
                f'Over {l}': sim_prob(res['corners']['h'], l)
                for l in [2.5, 3.5, 4.5, 5.5]
            },
            'away': {
                f'Over {l}': sim_prob(res['corners']['a'], l)
                for l in [2.5, 3.5, 4.5]
            },
            'total': {
                f'Over {l}': sim_prob(res['corners']['total'], l)
                for l in [8.5, 9.5, 10.5, 11.5]
            }
        },
        'cards': {
            'home': {
                f'Over {l}': sim_prob(res['cards']['h'], l)
                for l in [1.5, 2.5]
            },
            'away': {
                f'Over {l}': sim_prob(res['cards']['a'], l)
                for l in [1.5, 2.5]
            },
            'total': {
                f'Over {l}': sim_prob(res['cards']['total'], l)
                for l in [2.5, 3.5, 4.5, 5.5]
            }
        }
    }
    
    mc = res['monte_carlo']
    probs['chance'] = {
        '1': mc['h'],
        'X': mc['d'],
        '2': mc['a'],
        '1X': mc['h'] + mc['d'],
        'X2': mc['a'] + mc['d'],
        '12': mc['h'] + mc['a']
    }
    
    probs['goals'] = {
        'BTTS': res['probs']['btts'],
        'Over 2.5': res['probs']['over_2_5']
    }
    
    return probs

# ═══════════════════════════════════════════════════════════════════════════
# UI PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════

def main():
    
    # Sidebar
    with st.sidebar:
        st.title("🎛️ FutPrevisão V22")
        st.caption("Versão Final Completa")
        
        st.markdown("---")
        st.markdown("### 💰 Bankroll")
        st.session_state.bankroll = st.number_input(
            "Bankroll (€)",
            min_value=0.0,
            value=st.session_state.bankroll,
            step=50.0
        )
        
        st.markdown("---")
        st.markdown("### 📊 Status")
    
    # Título
    st.title("⚽ FutPrevisão V22.0 Final Edition")
    st.caption("Sistema Completo de Análise de Apostas Esportivas")
    
    # Carregamento
    with st.spinner("📂 Carregando dados..."):
        stats = learn_stats_v22()
        refs = load_referees_v22()
        calendar = load_calendar_safe()
        all_dfs = load_all_dataframes()
    
    # Status
    st.sidebar.success(f"✅ {len(stats)} times")
    st.sidebar.success(f"✅ {len(refs)} árbitros")
    st.sidebar.success(f"✅ {len(calendar)} jogos futuros")
    st.sidebar.success(f"✅ {len(all_dfs)} ligas")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📅 Calendário", "🔍 Simulação", "📊 Analytics"])
    
    # TAB 1: Calendário
    with tab1:
        st.markdown("## 📅 Calendário de Jogos")
        
        if calendar.empty:
            st.warning("⚠️ Calendário não carregado")
        else:
            dates = sorted(calendar['DtObj'].dt.strftime('%d/%m/%Y').unique())
            sel_date = st.selectbox("Selecione a data:", dates)
            
            df_day = calendar[calendar['DtObj'].dt.strftime('%d/%m/%Y') == sel_date]
            
            st.info(f"📊 {len(df_day)} jogos em {sel_date}")
            
            for idx, row in df_day.iterrows():
                liga = row.get('Liga', 'N/A')
                hora = row.get('Hora', 'N/A')
                home = row['Time_Casa']
                away = row['Time_Visitante']
                
                with st.expander(f"⏰ {hora} | {liga} | 🏠 {home} vs 🛫 {away}"):
                    
                    # Seleção de árbitro
                    lista_refs = ['Neutro'] + sorted(list(refs.keys()))
                    ref_sel = st.selectbox(
                        "👮 Árbitro:",
                        lista_refs,
                        key=f"cal_ref_{idx}"
                    )
                    
                    if st.button("🎯 Analisar Jogo", key=f"cal_btn_{idx}"):
                        ref_v = None if ref_sel == 'Neutro' else ref_sel
                        
                        res = calcular_jogo_v22(home, away, stats, ref_v, refs, all_dfs)
                        
                        if 'error' in res:
                            st.error(res['error'])
                        else:
                            # Resultado
                            st.markdown(f"## 🏠 {res['home']} vs 🛫 {res['away']}")
                            
                            # Métricas
                            c1, c2, c3 = st.columns(3)
                            c1.metric("xG Casa", f"{res['goals']['h']:.2f}")
                            c2.metric("xG Fora", f"{res['goals']['a']:.2f}")
                            c3.metric("Árbitro", res['meta']['referee'])
                            
                            # Info árbitro
                            if ref_v:
                                st.markdown("### 👮 Dados do Árbitro")
                                ar1, ar2, ar3, ar4 = st.columns(4)
                                ar1.metric("Fator", f"{res['meta']['ref_factor']:.2f}")
                                ar2.metric("Jogos", res['meta']['ref_games'])
                                ar3.metric("Amarelos", res['meta']['ref_yellows'])
                                ar4.metric("Vermelhos", res['meta']['ref_reds'])
                                
                                st.info(f"Rigidez: {res['meta']['ref_label']}")
                            
                            # Monte Carlo
                            st.markdown("### 🎲 Probabilidades (Monte Carlo)")
                            mc1, mc2, mc3 = st.columns(3)
                            mc1.metric("Casa", f"{res['monte_carlo']['h']:.1f}%")
                            mc2.metric("Empate", f"{res['monte_carlo']['d']:.1f}%")
                            mc3.metric("Fora", f"{res['monte_carlo']['a']:.1f}%")
                            
                            # Probabilidades detalhadas
                            probs = get_detailed_probs(res)
                            
                            st.markdown("### 📊 Probabilidades Detalhadas")
                            
                            col_esc, col_cart = st.columns(2)
                            
                            with col_esc:
                                st.markdown("**🏁 Escanteios**")
                                
                                st.markdown(f"**{res['home']}:**")
                                for k, v in probs['corners']['home'].items():
                                    hist = get_native_history(res['home'], res['league_h'], 'corners', 
                                                             float(k.split()[1]), 'home', all_dfs)
                                    cor = "green" if v >= 70 else "orange" if v >= 60 else "red"
                                    st.markdown(f"- {k}: :{cor}[{v:.0f}%] | Hist: {hist}")
                                
                                st.markdown(f"**{res['away']}:**")
                                for k, v in probs['corners']['away'].items():
                                    hist = get_native_history(res['away'], res['league_a'], 'corners',
                                                             float(k.split()[1]), 'away', all_dfs)
                                    cor = "green" if v >= 70 else "orange" if v >= 60 else "red"
                                    st.markdown(f"- {k}: :{cor}[{v:.0f}%] | Hist: {hist}")
                            
                            with col_cart:
                                st.markdown("**🟨 Cartões**")
                                
                                st.markdown(f"**{res['home']}:**")
                                for k, v in probs['cards']['home'].items():
                                    hist = get_native_history(res['home'], res['league_h'], 'cards',
                                                             float(k.split()[1]), 'home', all_dfs)
                                    cor = "green" if v >= 70 else "orange" if v >= 60 else "red"
                                    st.markdown(f"- {k}: :{cor}[{v:.0f}%] | Hist: {hist}")
                                
                                st.markdown(f"**{res['away']}:**")
                                for k, v in probs['cards']['away'].items():
                                    hist = get_native_history(res['away'], res['league_a'], 'cards',
                                                             float(k.split()[1]), 'away', all_dfs)
                                    cor = "green" if v >= 70 else "orange" if v >= 60 else "red"
                                    st.markdown(f"- {k}: :{cor}[{v:.0f}%] | Hist: {hist}")
    
    # TAB 2: Simulação
    with tab2:
        st.markdown("## 🔍 Simulador de Jogo")
        
        lista_times = sorted(list(stats.keys()))
        lista_refs = ['Neutro'] + sorted(list(refs.keys()))
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            home = st.selectbox("🏠 Time Casa", lista_times, key="sim_home")
        
        with col2:
            away = st.selectbox("🛫 Time Visitante", lista_times, key="sim_away")
        
        with col3:
            ref = st.selectbox("👮 Árbitro", lista_refs, key="sim_ref")
        
        if st.button("🎯 SIMULAR JOGO", type="primary"):
            ref_v = None if ref == 'Neutro' else ref
            
            res = calcular_jogo_v22(home, away, stats, ref_v, refs, all_dfs)
            
            if 'error' in res:
                st.error(res['error'])
            else:
                # Header
                st.markdown(f"# 🏠 {res['home']} vs 🛫 {res['away']}")
                st.caption(f"Liga: {res['league_h']}")
                
                # Métricas principais
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("xG Casa", f"{res['goals']['h']:.2f}")
                m2.metric("xG Fora", f"{res['goals']['a']:.2f}")
                m3.metric("Escanteios Total", f"{res['corners']['total']:.1f}")
                m4.metric("Cartões Total", f"{res['cards']['total']:.1f}")
                
                # Árbitro
                st.markdown("### 👮 Informações do Árbitro")
                if ref_v and ref_v in refs:
                    ar_data = refs[ref_v]
                    
                    ar1, ar2, ar3, ar4, ar5 = st.columns(5)
                    ar1.metric("Nome", ref_v)
                    ar2.metric("Fator", f"{ar_data['factor']:.2f}")
                    ar3.metric("Jogos", ar_data['games'])
                    ar4.metric("Amarelos", ar_data['yellows'])
                    ar5.metric("Vermelhos", ar_data['reds'])
                    
                    st.info(f"📊 Rigidez: {res['meta']['ref_label']} | Red Rate: {ar_data['red_rate']:.2%}")
                    st.info(f"🔴 Probabilidade de Vermelho no Jogo: {res['meta']['prob_red']:.1f}% ({res['meta']['prob_red_label']})")
                else:
                    st.info("Árbitro neutro selecionado (sem ajustes)")
                
                # Monte Carlo
                st.markdown("### 🎲 Probabilidades (Monte Carlo)")
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Vitória Casa", f"{res['monte_carlo']['h']:.1f}%")
                mc2.metric("Empate", f"{res['monte_carlo']['d']:.1f}%")
                mc3.metric("Vitória Fora", f"{res['monte_carlo']['a']:.1f}%")
                
                # Dupla Chance
                st.markdown("### 🎰 Dupla Chance")
                probs = get_detailed_probs(res)
                
                dc1, dc2, dc3 = st.columns(3)
                dc1.metric("1X (Casa ou Empate)", f"{probs['chance']['1X']:.1f}%")
                dc2.metric("X2 (Fora ou Empate)", f"{probs['chance']['X2']:.1f}%")
                dc3.metric("12 (Casa ou Fora)", f"{probs['chance']['12']:.1f}%")
                
                # Escanteios Individuais
                st.markdown("### 🏁 Escanteios Individuais")
                
                col_h, col_a = st.columns(2)
                
                with col_h:
                    st.markdown(f"#### 🏠 {res['home']}")
                    st.metric("Média de Escanteios", f"{res['corners']['h']:.2f}")
                    
                    for line in [2.5, 3.5, 4.5, 5.5]:
                        prob = probs['corners']['home'].get(f'Over {line}', 0)
                        hist = get_native_history(res['home'], res['league_h'], 'corners', line, 'home', all_dfs)
                        
                        cor = "green" if prob >= 70 else "orange" if prob >= 60 else "red"
                        st.markdown(f"**Over {line}:** :{cor}[{prob:.0f}%] | Hist: {hist}")
                
                with col_a:
                    st.markdown(f"#### 🛫 {res['away']}")
                    st.metric("Média de Escanteios", f"{res['corners']['a']:.2f}")
                    
                    for line in [2.5, 3.5, 4.5]:
                        prob = probs['corners']['away'].get(f'Over {line}', 0)
                        hist = get_native_history(res['away'], res['league_a'], 'corners', line, 'away', all_dfs)
                        
                        cor = "green" if prob >= 70 else "orange" if prob >= 60 else "red"
                        st.markdown(f"**Over {line}:** :{cor}[{prob:.0f}%] | Hist: {hist}")
                
                # Cartões Individuais
                st.markdown("### 🟨 Cartões Individuais")
                
                col_h2, col_a2 = st.columns(2)
                
                with col_h2:
                    st.markdown(f"#### 🏠 {res['home']}")
                    st.metric("Média de Cartões", f"{res['cards']['h']:.2f}")
                    
                    for line in [1.5, 2.5]:
                        prob = probs['cards']['home'].get(f'Over {line}', 0)
                        hist = get_native_history(res['home'], res['league_h'], 'cards', line, 'home', all_dfs)
                        
                        cor = "green" if prob >= 70 else "orange" if prob >= 60 else "red"
                        st.markdown(f"**Over {line}:** :{cor}[{prob:.0f}%] | Hist: {hist}")
                
                with col_a2:
                    st.markdown(f"#### 🛫 {res['away']}")
                    st.metric("Média de Cartões", f"{res['cards']['a']:.2f}")
                    
                    for line in [1.5, 2.5]:
                        prob = probs['cards']['away'].get(f'Over {line}', 0)
                        hist = get_native_history(res['away'], res['league_a'], 'cards', line, 'away', all_dfs)
                        
                        cor = "green" if prob >= 70 else "orange" if prob >= 60 else "red"
                        st.markdown(f"**Over {line}:** :{cor}[{prob:.0f}%] | Hist: {hist}")
                
                # Outros mercados
                st.markdown("### ⚽ Outros Mercados")
                
                om1, om2 = st.columns(2)
                
                with om1:
                    st.metric("BTTS (Ambos Marcam)", f"{probs['goals']['BTTS']:.1f}%")
                
                with om2:
                    st.metric("Over 2.5 Gols", f"{probs['goals']['Over 2.5']:.1f}%")
    
    # TAB 3: Analytics
    with tab3:
        st.markdown("## 📊 Analytics & Estatísticas")
        
        st.info("🚧 Módulo em desenvolvimento")
        
        if st.session_state.bet_history:
            st.markdown("### 📈 Histórico de Apostas")
            df_hist = pd.DataFrame(st.session_state.bet_history)
            st.dataframe(df_hist, use_container_width=True)

if __name__ == "__main__":
    main()
