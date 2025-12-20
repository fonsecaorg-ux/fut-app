"""
╔═══════════════════════════════════════════════════════════════════════════╗
║              FUTPREVISÃO V21.1 FINAL STABLE EDITION                       ║
║           100% Baseado Exclusivamente nos Seus Dados                      ║
║                                                                           ║
║  ✅ Correção de Caminhos (Path Fix)                                       ║
║  ✅ Normalização de Colunas (Mandante/Visitante -> HomeTeam/AwayTeam)     ║
║  ✅ Robustez contra arquivos vazios ou mal formatados                     ║
║                                                                           ║
║  Dezembro 2025                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import math
import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from difflib import get_close_matches

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURAÇÃO
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="FutPrevisão V21.1",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'bankroll' not in st.session_state:
    st.session_state.bankroll = 1000.0

# ═══════════════════════════════════════════════════════════════════════════
# MAPEAMENTO DE ARQUIVOS (CORRIGIDO PARA DIRETÓRIO LOCAL)
# ═══════════════════════════════════════════════════════════════════════════

# Se os arquivos estiverem na mesma pasta do script, usamos apenas o nome do arquivo.
# Caso use subpastas, altere aqui (ex: "dados/Premier_League...").

LEAGUE_FILES = {
    "Premier League": "Premier_League_25_26.csv",
    "La Liga": "La_Liga_25_26.csv",
    "Serie A": "Serie_A_25_26.csv",
    "Bundesliga": "Bundesliga_25_26.csv",
    "Ligue 1": "Ligue_1_25_26.csv",
    "Championship": "Championship_Inglaterra_25_26.csv",
    "Bundesliga 2": "Bundesliga_2.csv",
    "Pro League": "Pro_League_Belgica_25_26.csv",
    "Süper Lig": "Super_Lig_Turquia_25_26.csv",
    "Scottish Premiership": "Premiership_Escocia_25_26.csv"
}

REFEREE_FILES = [
    "arbitros_5_ligas_2025_2026.csv",
    "arbitros.csv"
]

CALENDAR_FILE = "calendario_ligas.csv"

NAME_MAPPING = {
    'Man United': 'Man Utd', 'Manchester United': 'Man Utd',
    'Man City': 'Man City', 'Manchester City': 'Man City',
    'Spurs': 'Tottenham', 'Athletic Club': 'Ath Bilbao',
    'Nottm Forest': "Nott'm Forest", 'Nottingham': "Nott'm Forest",
    'Internazionale': 'Inter', 'Milan': 'AC Milan'
}

# ═══════════════════════════════════════════════════════════════════════════
# UTILITÁRIOS
# ═══════════════════════════════════════════════════════════════════════════

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Padroniza nomes de colunas para inglês (padrão do sistema)"""
    df.columns = [c.strip() for c in df.columns]
    
    # Mapa de tradução e normalização
    rename_map = {
        'Mandante': 'HomeTeam', 'Time_Casa': 'HomeTeam', 'Home': 'HomeTeam', 'Time Casa': 'HomeTeam',
        'Visitante': 'AwayTeam', 'Time_Visitante': 'AwayTeam', 'Away': 'AwayTeam', 'Time Visitante': 'AwayTeam',
        'Data': 'Date', 'Liga': 'League',
        'Gols_Mandante': 'FTHG', 'Gols_Visitante': 'FTAG', 'HG': 'FTHG', 'AG': 'FTAG',
        'Cantos_Mandante': 'HC', 'Cantos_Visitante': 'AC',
        'Cartoes_Mandante': 'HY', 'Cartoes_Visitante': 'AY' # Assumindo amarelos como base se não tiver específicos
    }
    
    return df.rename(columns=rename_map)

# ═══════════════════════════════════════════════════════════════════════════
# CARREGAMENTO DE DADOS
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def load_leagues() -> Dict[str, pd.DataFrame]:
    """Carrega as ligas com tratamento de erro e normalização"""
    leagues = {}
    
    for league_name, filename in LEAGUE_FILES.items():
        if os.path.exists(filename):
            try:
                # Tenta UTF-8 primeiro, depois Latin-1
                try:
                    df = pd.read_csv(filename, encoding='utf-8')
                except UnicodeDecodeError:
                    df = pd.read_csv(filename, encoding='latin1')
                
                df = normalize_columns(df)
                
                if not df.empty:
                    leagues[league_name] = df
            except Exception as e:
                # Silencioso no log, mas não para a execução
                print(f"Erro ao ler {filename}: {e}")
                pass
    
    return leagues

@st.cache_data(ttl=3600)
def load_referees() -> Dict[str, Dict]:
    """Carrega árbitros mesclando fontes"""
    refs = {}
    
    for filepath in REFEREE_FILES:
        if os.path.exists(filepath):
            try:
                try:
                    df = pd.read_csv(filepath, encoding='utf-8')
                except:
                    df = pd.read_csv(filepath, encoding='latin1')
                
                df.columns = [c.strip() for c in df.columns]
                
                # Adaptação para diferentes formatos de CSV de árbitros
                col_nome = 'Arbitro' if 'Arbitro' in df.columns else 'Nome'
                col_media = 'Media_Cartoes_Por_Jogo' if 'Media_Cartoes_Por_Jogo' in df.columns else 'Fator'
                
                if col_nome not in df.columns: continue

                for _, row in df.iterrows():
                    nome = str(row[col_nome]).strip()
                    
                    try:
                        # Se tiver a coluna fator/média
                        if col_media in row:
                            media = float(row[col_media])
                            # Se for o arquivo de "Fator" (geralmente escala 1.0), converte para média (~4.0)
                            if media < 2.0: media = media * 4.0
                        else:
                            media = 4.0 # Padrão
                            
                        vermelhos = float(row.get('Cartoes_Vermelhos', 0))
                        jogos = float(row.get('Jogos_Apitados', 1))
                        
                        refs[nome] = {
                            'factor': media / 4.0,
                            'red_rate': vermelhos / jogos if jogos > 0 else 0.08,
                            'strictness': media
                        }
                    except:
                        continue
            except:
                pass
                
    if not refs:
        # Fallback se nenhum arquivo for encontrado
        refs['Neutro'] = {'factor': 1.0, 'red_rate': 0.08, 'strictness': 4.0}
        
    return refs

@st.cache_data(ttl=600)
def load_calendar() -> pd.DataFrame:
    """Carrega calendário futuro"""
    if not os.path.exists(CALENDAR_FILE):
        return pd.DataFrame()
        
    try:
        try:
            df = pd.read_csv(CALENDAR_FILE, encoding='utf-8')
        except:
            df = pd.read_csv(CALENDAR_FILE, encoding='latin1')
            
        df = normalize_columns(df)
        
        # Parse de datas flexível
        df['DtObj'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
        # Tenta formato ISO se falhar
        mask = df['DtObj'].isna()
        if mask.any():
            df.loc[mask, 'DtObj'] = pd.to_datetime(df.loc[mask, 'Date'], errors='coerce')
            
        df = df.dropna(subset=['DtObj']).sort_values('DtObj')
        return df
    except:
        return pd.DataFrame()

# ═══════════════════════════════════════════════════════════════════════════
# PROCESSAMENTO ESTATÍSTICO
# ═══════════════════════════════════════════════════════════════════════════

def calculate_elo(df: pd.DataFrame) -> Dict[str, float]:
    elo = {}
    teams = set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique())
    for t in teams: elo[t] = 1500
    
    if 'Date' in df.columns:
        # Garante datetime
        if not pd.api.types.is_datetime64_any_dtype(df['Date']):
             df['DtSort'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        else:
             df['DtSort'] = df['Date']
        df = df.sort_values('DtSort')
    
    for _, row in df.iterrows():
        if pd.isna(row.get('FTHG')) or pd.isna(row.get('FTAG')): continue
        
        h, a = row['HomeTeam'], row['AwayTeam']
        elo_h, elo_a = elo.get(h, 1500), elo.get(a, 1500)
        
        res = 1 if row['FTHG'] > row['FTAG'] else 0.5 if row['FTHG'] == row['FTAG'] else 0
        exp = 1 / (1 + 10**((elo_a - elo_h) / 400))
        
        elo[h] = elo_h + 30 * (res - exp)
        elo[a] = elo_a + 30 * ((1 - res) - (1 - exp))
        
    return elo

@st.cache_data(ttl=3600)
def learn_stats() -> Dict[str, Dict]:
    stats_db = {}
    leagues = load_leagues()
    
    # ELO Global
    global_elo = {}
    for _, df in leagues.items():
        global_elo.update(calculate_elo(df))
        
    for league_name, df in leagues.items():
        # Garante colunas numéricas essenciais
        num_cols = ['HC', 'AC', 'HY', 'AY', 'HF', 'AF', 'FTHG', 'FTAG']
        for c in num_cols:
            if c not in df.columns: df[c] = np.nan
        
        # Peso Recência
        df['Weight'] = 1.0
        if 'Date' in df.columns:
            df['DtCalc'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
            df = df.sort_values('DtCalc').reset_index(drop=True)
            # Pesos exponenciais para os jogos mais recentes
            weights = np.exp(np.linspace(0, 1, len(df)))
            df['Weight'] = weights / weights.sum() * len(df) # Normaliza para média 1
            
        try:
            # Função auxiliar de média ponderada
            def w_agg(x):
                # Alinha os índices para multiplicar corretamente
                valid_idx = x.index.intersection(df.index)
                if len(valid_idx) == 0: return 0
                vals = x.loc[valid_idx]
                ws = df.loc[valid_idx, 'Weight']
                if ws.sum() == 0: return 0
                return (vals * ws).sum() / ws.sum()

            # Agregação por time
            h_stats = df.groupby('HomeTeam')[['HC','HY','HF','FTHG','FTAG']].agg(w_agg).fillna(0)
            a_stats = df.groupby('AwayTeam')[['AC','AY','AF','FTAG','FTHG']].agg(w_agg).fillna(0)
            
            all_teams = set(h_stats.index) | set(a_stats.index)
            
            for team in all_teams:
                # Recupera stats ou 0 se não existir
                h = h_stats.loc[team] if team in h_stats.index else pd.Series(0, index=h_stats.columns)
                a = a_stats.loc[team] if team in a_stats.index else pd.Series(0, index=a_stats.columns)
                
                # Média Ponderada (60% Casa / 40% Fora)
                def mix(vh, va): return (vh * 0.6) + (va * 0.4)
                
                stats_db[team] = {
                    'corners': mix(h.get('HC', 0), a.get('AC', 0)),
                    'cards': mix(h.get('HY', 0), a.get('AY', 0)),
                    'fouls': mix(h.get('HF', 0), a.get('AF', 0)),
                    'goals_f': mix(h.get('FTHG', 0), a.get('FTAG', 0)), # Gols feitos
                    'goals_a': mix(h.get('FTAG', 0), a.get('FTHG', 0)), # Gols sofridos
                    'elo': global_elo.get(team, 1500),
                    'league': league_name
                }
                
        except Exception as e:
            # Log silencioso para não poluir a UI
            print(f"Erro processando liga {league_name}: {e}")
            continue
            
    return stats_db

# ═══════════════════════════════════════════════════════════════════════════
# CÁLCULO DE PROBABILIDADES
# ═══════════════════════════════════════════════════════════════════════════

def normalize_name(name: str, keys: list) -> Optional[str]:
    if name in NAME_MAPPING: name = NAME_MAPPING[name]
    if name in keys: return name
    matches = get_close_matches(name, keys, n=1, cutoff=0.6)
    return matches[0] if matches else None

def poisson(k: int, lam: float) -> float:
    if lam <= 0 or k < 0: return 0.0
    return (lam**k * math.exp(-lam)) / math.factorial(k)

def monte_carlo(xg_h: float, xg_a: float, n: int = 1000) -> Tuple[float, float, float]:
    gh = np.random.poisson(max(0.1, xg_h), n)
    ga = np.random.poisson(max(0.1, xg_a), n)
    h_w = np.count_nonzero(gh > ga)
    a_w = np.count_nonzero(ga > gh)
    d = n - h_w - a_w
    return h_w/n, d/n, a_w/n

def calculate_game(home: str, away: str, stats: Dict, ref: Optional[str], refs: Dict) -> Dict:
    h_norm = normalize_name(home, list(stats.keys()))
    a_norm = normalize_name(away, list(stats.keys()))
    
    if not h_norm or not a_norm:
        return {'error': "Times não encontrados na base de dados."}
    
    s_h = stats[h_norm]
    s_a = stats[a_norm]
    
    # Dados do Árbitro
    r = refs.get(ref, {'factor': 1.0, 'red_rate': 0.08, 'strictness': 4.0}) if ref else {'factor': 1.0, 'red_rate': 0.08, 'strictness': 4.0}
    
    # Ajuste por ELO
    elo_diff = s_h['elo'] - s_a['elo']
    elo_factor = math.log10(max(1, abs(elo_diff))) * 0.05 * (1 if elo_diff > 0 else -1)
    
    # Previsões Base
    corn_h = s_h['corners'] * 1.15
    corn_a = s_a['corners'] * 0.90
    card_h = s_h['cards'] * r['factor']
    card_a = s_a['cards'] * r['factor']
    
    # xG Ajustado
    xg_h = max(0.1, s_h['goals_f'] + elo_factor)
    xg_a = max(0.1, s_a['goals_f'] - elo_factor)
    
    mc_h, mc_d, mc_a = monte_carlo(xg_h, xg_a)
    
    return {
        'home': h_norm, 'away': a_norm,
        'league_h': s_h['league'], 'league_a': s_a['league'],
        'goals': {'h': xg_h, 'a': xg_a},
        'corners': {'h': corn_h, 'a': corn_a, 'total': corn_h + corn_a},
        'cards': {'h': card_h, 'a': card_a, 'total': card_h + card_a},
        'monte_carlo': {'h': mc_h*100, 'd': mc_d*100, 'a': mc_a*100},
        'meta': {
            'ts_h': round(s_h['elo']/20, 1), 'ts_a': round(s_a['elo']/20, 1), # TS simplificado
            'ref_factor': r['factor'], 'ref_strictness': r['strictness']
        }
    }

def get_probs(res: Dict) -> Dict:
    def sim(avg, line): return max(5, min(95, 50 + (avg - line) * 15))
    
    return {
        'corners': {
            'home': {f'Over {l}': sim(res['corners']['h'], l) for l in [2.5, 3.5, 4.5, 5.5]},
            'away': {f'Over {l}': sim(res['corners']['a'], l) for l in [2.5, 3.5, 4.5]},
            'total': {f'Over {l}': sim(res['corners']['total'], l) for l in [8.5, 9.5, 10.5]}
        },
        'cards': {
            'home': {f'Over {l}': sim(res['cards']['h'], l) for l in [1.5, 2.5]},
            'away': {f'Over {l}': sim(res['cards']['a'], l) for l in [1.5, 2.5]},
            'total': {f'Over {l}': sim(res['cards']['total'], l) for l in [2.5, 3.5, 4.5]}
        }
    }

def get_fair_odd(prob: float) -> float:
    return round(100/prob, 2) if prob > 0 else 99.0

def gen_options(home: str, away: str, probs: Dict) -> List[Dict]:
    opts = []
    # Cria lista de opções de aposta
    for market in ['corners', 'cards']:
        for side, name in [('home', home), ('away', away), ('total', 'Total')]:
            if side not in probs[market]: continue
            for k, p in probs[market][side].items():
                if p > 0:
                    lbl = f"{name} {k} {market}"
                    opts.append({
                        'label': f"{lbl} ({p:.0f}%)", 'display': lbl,
                        'prob': p, 'min_odd': get_fair_odd(p),
                        'market': market, 'side': side
                    })
    return sorted(opts, key=lambda x: x['prob'], reverse=True)

def calc_combined_prob(sels: List[Dict]) -> float:
    if not sels: return 0.0
    p = 1.0
    for s in sels: p *= (s['prob'] / 100)
    return p * 100

def gen_hedges(main: List[Dict], stats: Dict, refs: Dict) -> Tuple[List[Dict], List[Dict]]:
    h1, h2 = [], []
    games = {}
    for s in main:
        gid = s['game_id']
        if gid not in games: games[gid] = []
        games[gid].append(s)
        
    for gid, sels in games.items():
        home, away = sels[0]['home'], sels[0]['away']
        res = calculate_game(home, away, stats, None, refs)
        if 'error' in res: continue
        
        probs = get_probs(res)
        all_opts = gen_options(home, away, probs)
        valid = [o for o in all_opts if o['prob'] >= 65]
        if len(valid) < 6: valid = all_opts[:10]
        
        main_disp = [s['display'] for s in sels]
        
        # Hedge 1: Opções diferentes das principais
        h1_opts = []
        for o in valid:
            if o['display'] not in main_disp:
                h1_opts.append(o)
                if len(h1_opts) >= 2: break
        
        for o in h1_opts: h1.append({**o, 'game_id': gid, 'home': home, 'away': away, 'change': '🔄'})
        
        # Hedge 2: Mix Stats (Prioriza mercados diferentes)
        h2_opts = []
        used = main_disp + [o['display'] for o in h1_opts]
        
        # Tenta pegar um de cada mercado se possível
        corns = [o for o in valid if o['market'] == 'corners' and o['display'] not in used]
        cards = [o for o in valid if o['market'] == 'cards' and o['display'] not in used]
        
        if corns and cards:
            h2_opts = [corns[0], cards[0]]
        else:
            # Fallback
            avail = [o for o in valid if o['display'] not in used]
            h2_opts = avail[:2]
            
        for o in h2_opts: h2.append({**o, 'game_id': gid, 'home': home, 'away': away, 'change': '🔄'})
            
    return h1, h2

# ═══════════════════════════════════════════════════════════════════════════
# INTERFACE PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════

def main():
    st.title("⚽ FutPrevisão V21.1 (Stable Local)")
    
    with st.spinner("Carregando bases de dados..."):
        stats = learn_stats()
        refs = load_referees()
        calendar = load_calendar()
        
    # Verificação de Dados
    if not stats:
        st.error("🚨 ERRO CRÍTICO: Não foi possível carregar estatísticas.")
        st.warning("Certifique-se de que os arquivos CSV (ex: Premier_League_25_26.csv) estão na mesma pasta do script.")
        return

    st.sidebar.success(f"✅ Dados Carregados: {len(stats)} times")

    t1, t2, t3 = st.tabs(["📅 Calendário", "🔍 Simulação", "🎰 Bet Builder"])
    
    with t1:
        if calendar.empty:
            st.warning("Calendário vazio ou não encontrado.")
        else:
            dates = sorted(calendar['DtObj'].dt.strftime('%d/%m/%Y').unique())
            sel_date = st.selectbox("Selecione a Data:", dates)
            subset = calendar[calendar['DtObj'].dt.strftime('%d/%m/%Y') == sel_date]
            
            for _, row in subset.iterrows():
                h, a = row['HomeTeam'], row['AwayTeam']
                if st.button(f"Analisar {h} x {a}"):
                    res = calculate_game(h, a, stats, None, refs)
                    if 'error' in res:
                        st.error(res['error'])
                    else:
                        st.info(f"Prob. Vitória: 🏠 {res['monte_carlo']['h']:.0f}% | 🤝 {res['monte_carlo']['d']:.0f}% | ✈️ {res['monte_carlo']['a']:.0f}%")
                        st.write(f"xG: {res['goals']['h']:.2f} x {res['goals']['a']:.2f}")

    with t2:
        l_times = sorted(list(stats.keys()))
        c1, c2 = st.columns(2)
        h = c1.selectbox("Casa", l_times, key='s_h')
        a = c2.selectbox("Fora", l_times, key='s_a', index=min(1, len(l_times)-1))
        
        if st.button("Simular Partida"):
            res = calculate_game(h, a, stats, None, refs)
            if 'error' in res:
                st.error(res['error'])
            else:
                probs = get_probs(res)
                cc1, cc2 = st.columns(2)
                with cc1:
                    st.write("**Escanteios**")
                    for k, v in probs['corners']['home'].items(): st.write(f"{h} {k}: {v:.0f}%")
                with cc2:
                    st.write("**Cartões**")
                    for k, v in probs['cards']['home'].items(): st.write(f"{h} {k}: {v:.0f}%")

    with t3:
        if 'main_slip' not in st.session_state: st.session_state.main_slip = []
        l_times = sorted(list(stats.keys()))
        num = st.number_input("Qtd Jogos", 1, 5, 3)
        
        temp = []
        for i in range(num):
            st.markdown(f"**Jogo {i+1}**")
            c1, c2 = st.columns(2)
            h = c1.selectbox("C", l_times, key=f"b_h_{i}")
            a = c2.selectbox("F", l_times, key=f"b_a_{i}", index=1)
            
            res = calculate_game(h, a, stats, None, refs)
            if 'error' not in res:
                probs = get_probs(res)
                opts = gen_options(h, a, probs)
                lbls = [o['display'] for o in opts]
                
                s1 = st.selectbox("Sel 1", range(len(lbls)), format_func=lambda x: lbls[x], key=f"sel1_{i}")
                s2 = st.selectbox("Sel 2", range(len(lbls)), format_func=lambda x: lbls[x], key=f"sel2_{i}", index=min(1, len(lbls)-1))
                
                temp.append({**opts[s1], 'game_id': i, 'home': h, 'away': a})
                temp.append({**opts[s2], 'game_id': i, 'home': h, 'away': a})
        
        st.session_state.main_slip = temp
        
        if st.button("Gerar Estratégia"):
            h1, h2 = gen_hedges(st.session_state.main_slip, stats, refs)
            c1, c2, c3 = st.columns(3)
            
            def card(title, bets, col):
                with col:
                    st.info(title)
                    for b in bets: st.write(f"- {b['display']} (@{b['min_odd']})")
            
            card("Principal", st.session_state.main_slip, c1)
            card("Hedge 1", h1, c2)
            card("Hedge 2", h2, c3)

if __name__ == "__main__":
    main()
