"""
╔═══════════════════════════════════════════════════════════════════════════╗
║                  FUTPREVISÃO V14.2 - ROBUST DATA LOADER                   ║
║                          Sistema de Análise de Apostas                     ║
║                                                                            ║
║  Versão: V14.2 (Correção de Dados Vazios)                                 ║
║  Data: Dezembro 2025                                                      ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import math
import numpy as np
from typing import Dict, Optional, Any
from difflib import get_close_matches
from datetime import datetime
import os

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURAÇÕES GLOBAIS
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="FutPrevisão V14.2",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constantes V14
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

# Mapeamento de nomes para normalizar times
NAME_MAPPING = {
    'Man United': 'Man Utd', 'Manchester United': 'Man Utd',
    'Man City': 'Man City', 'Manchester City': 'Man City',
    'Spurs': 'Tottenham', 'Newcastle': 'Newcastle',
    'Wolves': 'Wolves', 'Brighton': 'Brighton',
    'Nott\'m Forest': 'Nottm Forest', 'Nottingham Forest': 'Nottm Forest',
    'West Ham': 'West Ham', 'Leicester': 'Leicester',
}

# Lista de ligas esperadas
LIGAS_ALVO = [
    "Premier League", "La Liga", "Serie A", "Bundesliga", "Ligue 1",
    "Championship", "Bundesliga 2", "Pro League", "Süper Lig", "Scottish Premiership"
]

# Variável global para debug na interface
DEBUG_LOGS = []

def log_status(msg: str, status: str = "info"):
    """Registra logs para exibir na sidebar."""
    icon = "✅" if status == "success" else "❌" if status == "error" else "ℹ️"
    DEBUG_LOGS.append(f"{icon} {msg}")

# ═══════════════════════════════════════════════════════════════════════════
# CARREGAMENTO INTELIGENTE DE ARQUIVOS
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def find_and_load_csv(league_name: str) -> pd.DataFrame:
    """
    Tenta encontrar o CSV da liga testando várias variações de nome.
    """
    # Variações possíveis de nome de arquivo
    attempts = [
        f"{league_name} 25.26.csv",          # Com espaços (Padrão Windows)
        f"{league_name.replace(' ', '_')}_25_26.csv", # Com underlines
        f"{league_name}.csv",                # Nome simples
        f"{league_name} 2025.csv"            # Variação de ano
    ]
    
    # Tentativas específicas para ligas com nomes complexos
    if "Süper Lig" in league_name:
        attempts.append("Super Lig Turquia 25.26.csv")
        attempts.append("Super_Lig_Turquia_25_26.csv")
    if "Pro League" in league_name:
        attempts.append("Pro League Belgica 25.26.csv")
    if "Premiership" in league_name:
        attempts.append("Premiership Escocia 25.26.csv")
    if "Championship" in league_name:
        attempts.append("Championship Inglaterra 25.26.csv")

    for filename in attempts:
        if os.path.exists(filename):
            try:
                # Tenta ler com encoding utf-8 primeiro, depois latin1
                try:
                    df = pd.read_csv(filename, encoding='utf-8')
                except:
                    df = pd.read_csv(filename, encoding='latin1')
                
                if not df.empty:
                    log_status(f"Carregado: {filename}", "success")
                    return df
            except Exception as e:
                log_status(f"Erro ao ler {filename}: {e}", "error")
    
    log_status(f"Arquivo não encontrado para: {league_name}", "error")
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def learn_stats_v14() -> Dict[str, Dict[str, Any]]:
    stats_db = {}
    total_loaded = 0
    
    for league in LIGAS_ALVO:
        df = find_and_load_csv(league)
        if df.empty: continue
        
        # Normaliza colunas (remove espaços extras)
        df.columns = [c.strip() for c in df.columns]
        
        # Garante colunas mínimas (preenche com NaN se faltar)
        cols_needed = ['HomeTeam', 'AwayTeam', 'HC', 'AC', 'HY', 'AY', 'HF', 'AF', 'FTHG', 'FTAG', 'HST', 'AST', 'HR', 'AR']
        for c in cols_needed:
            if c not in df.columns:
                df[c] = np.nan
        
        # Agregações
        try:
            # Home Stats
            h_stats = df.groupby('HomeTeam').agg({
                'HC': 'mean', 'HY': 'mean', 'HF': 'mean',
                'FTHG': 'mean', 'FTAG': 'mean',
                'HST': 'mean', 'HR': 'mean'
            }).fillna(value={
                'HST': DEFAULTS['shots_on_target'],
                'HR': DEFAULTS['red_cards_avg']
            })
            
            # Away Stats
            a_stats = df.groupby('AwayTeam').agg({
                'AC': 'mean', 'AY': 'mean', 'AF': 'mean',
                'FTAG': 'mean', 'FTHG': 'mean',
                'AST': 'mean', 'AR': 'mean'
            }).fillna(value={
                'AST': DEFAULTS['shots_on_target'],
                'AR': DEFAULTS['red_cards_avg']
            })
            
            # Unificação
            all_teams = set(h_stats.index) | set(a_stats.index)
            for team in all_teams:
                # Pega dados ou usa padrão zerado se time só jogou de um lado
                h = h_stats.loc[team] if team in h_stats.index else pd.Series(0, index=h_stats.columns)
                a = a_stats.loc[team] if team in a_stats.index else pd.Series(0, index=a_stats.columns)
                
                # Para evitar divisão por zero ou média errada, usamos média simples ponderada se existir
                # Assumindo que se o dado existe, é > 0 ou válido.
                
                # Helper para combinar casa/fora (60% casa / 40% fora é o peso ideal)
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
            
            total_loaded += 1
            
        except Exception as e:
            log_status(f"Erro processando dados da liga {league}: {e}", "error")
            
    if total_loaded == 0:
        log_status("CRÍTICO: Nenhuma liga foi carregada corretamente!", "error")
        
    return stats_db

@st.cache_data(ttl=3600)
def load_referees_v14() -> Dict[str, Dict[str, float]]:
    refs_db = {}
    
    # 1. Tenta carregar árbitros das 5 ligas (arquivo mais completo)
    f5 = "arbitros_5_ligas_2025_2026.csv"
    if os.path.exists(f5):
        try:
            df = pd.read_csv(f5)
            for _, row in df.iterrows():
                nome = str(row['Arbitro']).strip()
                media = float(row['Media_Cartoes_Por_Jogo'])
                jogos = float(row['Jogos_Apitados'])
                vermelhos = float(row.get('Cartoes_Vermelhos', 0))
                red_rate = (vermelhos / jogos) if jogos > 0 else DEFAULTS['red_rate_referee']
                
                refs_db[nome] = {'factor': media/4.0, 'red_rate': red_rate}
            log_status(f"Árbitros carregados: {f5}", "success")
        except Exception as e:
            log_status(f"Erro em {f5}: {e}", "error")
            
    # 2. Tenta carregar árbitros gerais (fallback)
    f_gen = "arbitros.csv"
    if os.path.exists(f_gen):
        try:
            df = pd.read_csv(f_gen)
            for _, row in df.iterrows():
                nome = str(row['Nome']).strip()
                if nome not in refs_db:
                    refs_db[nome] = {'factor': float(row['Fator']), 'red_rate': DEFAULTS['red_rate_referee']}
            log_status(f"Árbitros carregados: {f_gen}", "success")
        except: pass
        
    return refs_db

@st.cache_data(ttl=600)
def load_calendar_safe() -> pd.DataFrame:
    """Carrega calendário, trata erros e ordena por data."""
    fname = "calendario_ligas.csv"
    if not os.path.exists(fname):
        log_status(f"Faltando arquivo: {fname}", "error")
        return pd.DataFrame()
        
    try:
        # Tenta ler com diferentes encodings
        try: df = pd.read_csv(fname, encoding='utf-8')
        except: df = pd.read_csv(fname, encoding='latin1')
        
        # Limpa nomes das colunas
        df.columns = [c.strip() for c in df.columns]
        
        # Padroniza nomes das colunas para evitar KeyError
        rename_map = {}
        if 'Mandante' in df.columns: rename_map['Mandante'] = 'Time_Casa'
        if 'Visitante' in df.columns: rename_map['Visitante'] = 'Time_Visitante'
        if rename_map: df = df.rename(columns=rename_map)
        
        # Verifica colunas essenciais
        req = ['Data', 'Liga', 'Time_Casa', 'Time_Visitante', 'Hora']
        missing = [c for c in req if c not in df.columns]
        if missing:
            log_status(f"Calendário com colunas faltando: {missing}", "error")
            return pd.DataFrame()
            
        # Ordenação
        df['DtObj'] = pd.to_datetime(df['Data'], format='%d/%m/%Y', errors='coerce')
        df = df.dropna(subset=['DtObj']) # Remove datas inválidas
        df = df.sort_values(by=['DtObj', 'Hora'])
        
        log_status("Calendário carregado e ordenado", "success")
        return df
        
    except Exception as e:
        log_status(f"Erro fatal no calendário: {e}", "error")
        return pd.DataFrame()

# ═══════════════════════════════════════════════════════════════════════════
# CÁLCULO V14 (CAUSALITY ENGINE)
# ═══════════════════════════════════════════════════════════════════════════

def normalize_name(name: str, db_keys: list) -> Optional[str]:
    if name in NAME_MAPPING: name = NAME_MAPPING[name]
    if name in db_keys: return name
    matches = get_close_matches(name, db_keys, n=1, cutoff=0.6)
    return matches[0] if matches else None

def calcular_jogo_v14(home: str, away: str, stats: Dict, ref: Optional[str], refs_db: Dict) -> Dict:
    # Busca Times
    h_norm = normalize_name(home, list(stats.keys()))
    a_norm = normalize_name(away, list(stats.keys()))
    
    if not h_norm or not a_norm:
        return {'error': f"Times não encontrados no banco de dados. Verifique a grafia."}
    
    s_h = stats[h_norm]
    s_a = stats[a_norm]
    
    # Busca Árbitro
    if ref and ref in refs_db:
        r_data = refs_db[ref]
    else:
        r_data = {'factor': 1.0, 'red_rate': DEFAULTS['red_rate_referee']} # Default Neutro
        
    # --- MOTOR V14 ---
    
    # 1. Chutes -> Escanteios
    shots_h = s_h['shots_on_target']
    shots_a = s_a['shots_on_target']
    
    # Boost Casa
    if shots_h > THRESHOLDS['shots_pressure_high']: p_h, l_h = 1.20, "ALTO 🔥"
    elif shots_h > THRESHOLDS['shots_pressure_medium']: p_h, l_h = 1.10, "MÉDIO ✅"
    else: p_h, l_h = 1.0, "BAIXO ⚪"
    
    # Boost Fora
    if shots_a > THRESHOLDS['shots_pressure_high']: p_a, l_a = 1.20, "ALTO 🔥"
    elif shots_a > THRESHOLDS['shots_pressure_medium']: p_a, l_a = 1.10, "MÉDIO ✅"
    else: p_a, l_a = 1.0, "BAIXO ⚪"
    
    corn_h = s_h['corners'] * 1.15 * p_h
    corn_a = s_a['corners'] * 0.90 * p_a
    
    # 2. Rigidez -> Cartões
    rr = r_data['red_rate']
    if rr > THRESHOLDS['red_rate_strict_high']: strict, s_lbl = 1.15, "MUITO RIGOROSO 🔴"
    elif rr > THRESHOLDS['red_rate_strict_medium']: strict, s_lbl = 1.08, "RIGOROSO 🟠"
    else: strict, s_lbl = 1.0, "NORMAL 🟢"
    
    # Violência (Faltas)
    viol_h = 1.0 if s_h['fouls'] > THRESHOLDS['fouls_violent'] else 0.85
    viol_a = 1.0 if s_a['fouls'] > THRESHOLDS['fouls_violent'] else 0.85
    
    card_h = s_h['cards'] * viol_h * r_data['factor'] * strict
    card_a = s_a['cards'] * viol_a * r_data['factor'] * strict
    
    # 3. Vermelhos
    reds_avg = (s_h['red_cards_avg'] + s_a['red_cards_avg']) / 2
    prob_red = reds_avg * rr * 100
    
    if prob_red > 12: pr_lbl = "ALTA 🔴"
    elif prob_red > 8: pr_lbl = "MÉDIA 🟠"
    else: pr_lbl = "BAIXA 🟡"

    return {
        'home': h_norm, 'away': a_norm, 'referee': ref,
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

def get_probs(corners, cards):
    def p(k, l): return sum((l**i * math.exp(-l)) / math.factorial(i) for i in range(k + 1))
    return {
        'corn': {f"Over {i}.5": (1-p(i, corners))*100 for i in range(8, 13)},
        'card': {f"Over {i}.5": (1-p(i, cards))*100 for i in range(3, 6)}
    }

# ═══════════════════════════════════════════════════════════════════════════
# INTERFACE
# ═══════════════════════════════════════════════════════════════════════════

def render_result(res):
    m = res['meta']
    st.markdown("---")
    
    # Header Times
    c1, c2, c3 = st.columns([2,1,2])
    c1.markdown(f"### 🏠 {res['home']}")
    c2.markdown("<h3 style='text-align: center'>VS</h3>", unsafe_allow_html=True)
    c3.markdown(f"<h3 style='text-align: right'>✈️ {res['away']}</h3>", unsafe_allow_html=True)
    
    # Métricas Causais
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("xG Casa", f"{res['goals']['h']:.2f}")
    k2.metric("xG Fora", f"{res['goals']['a']:.2f}")
    k3.metric("Chutes Casa", f"{m['shots_h']:.1f}", m['p_label_h'])
    k4.metric("Risco Vermelho", f"{m['prob_red']:.1f}%", m['prob_red_lbl'])
    
    st.caption(f"👮 Juiz: {res['referee'] if res['referee'] else 'Neutro'} | Rigidez: {m['strict_lbl']} ({m['strict_val']}x) | Taxa Vermelhos: {m['red_rate']:.2f}")
    st.caption(f"⚔️ Estilo: Casa {m['viol_h_lbl']} vs Fora {m['viol_a_lbl']}")

    # Resultados e Probabilidades
    probs = get_probs(res['corners']['total'], res['cards']['total'])
    
    wc1, wc2 = st.columns(2)
    with wc1:
        st.subheader(f"🏁 Escanteios: {res['corners']['total']:.2f}")
        for k, v in probs['corn'].items():
            color = "green" if v >= 75 else "black"
            weight = "bold" if v >= 75 else "normal"
            if v >= 75: st.success(f"**{k}: {v:.1f}%** (ELITE)")
            elif v > 50: st.write(f"{k}: {v:.1f}%")
            
    with wc2:
        st.subheader(f"🟨 Cartões: {res['cards']['total']:.2f}")
        for k, v in probs['card'].items():
            if v >= 70: st.success(f"**{k}: {v:.1f}%** (ELITE)")
            elif v > 50: st.write(f"{k}: {v:.1f}%")

def main():
    st.title("⚽ FutPrevisão V14.2 (Robust Data)")
    st.caption("Causality Engine + Smart File Loader")
    
    # Carregamento com logs visíveis
    with st.spinner("Inicializando motores..."):
        DEBUG_LOGS.clear()
        stats = learn_stats_v14()
        refs = load_referees_v14()
        calendar = load_calendar_safe()
        
    # Sidebar de Status (Fundamental para debug)
    with st.sidebar:
        with st.expander("🛠️ Status do Sistema", expanded=not bool(stats)):
            st.write(f"Times Carregados: {len(stats)}")
            st.write(f"Árbitros: {len(refs)}")
            st.markdown("---")
            for log in DEBUG_LOGS:
                st.write(log)
    
    if not stats:
        st.error("🚨 ERRO CRÍTICO: Nenhum dado estatístico foi carregado. Verifique o painel 'Status do Sistema' na esquerda para ver quais arquivos falharam.")
        return

    tab1, tab2 = st.tabs(["📅 Calendário (Ordenado)", "🧪 Simulação Manual"])
    
    with tab1:
        if calendar.empty:
            st.warning("Calendário vazio ou inválido.")
        else:
            dates = calendar['DtObj'].dt.strftime('%d/%m/%Y').unique()
            sel_date = st.selectbox("Selecione a Data:", dates)
            subset = calendar[calendar['DtObj'].dt.strftime('%d/%m/%Y') == sel_date]
            
            st.write(f"{len(subset)} jogos encontrados.")
            for i, row in subset.iterrows():
                with st.expander(f"⏰ {str(row['Hora'])[:5]} | {row['Liga']} | {row['Time_Casa']} x {row['Time_Visitante']}"):
                    if st.button("Analisar", key=f"btn_{i}"):
                        res = calcular_jogo_v14(row['Time_Casa'], row['Time_Visitante'], stats, None, refs)
                        if 'error' in res: st.error(res['error'])
                        else: render_result(res)

    with tab2:
        c1, c2, c3 = st.columns(3)
        h = c1.text_input("Mandante", "Liverpool")
        a = c2.text_input("Visitante", "Man City")
        r = c3.text_input("Árbitro")
        if st.button("Simular Manual"):
            res = calcular_jogo_v14(h, a, stats, r, refs)
            if 'error' in res: st.error(res['error'])
            else: render_result(res)

if __name__ == "__main__":
    main()
