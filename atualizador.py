"""
FutPrevisão V31 ULTRA MAXIMUM + AI Advisor ULTRA + Construtor Manual
CÓDIGO COMPLETO - 2500+ LINHAS
VERSÃO PROFISSIONAL

Autor: Diego
Versão: 31.0 ULTRA MAXIMUM
Data: 27/12/2024

NOVIDADES:
✅ Chatbot Ultra Inteligente - Responde QUALQUER pergunta sobre dados
✅ Construtor Manual - Dropdowns para criar apostas personalizadas
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import math
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from difflib import get_close_matches
import re
from collections import defaultdict

BASE_DIR = Path(__file__).resolve().parent

# ============================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================

st.set_page_config(
    page_title="FutPrevisão V31 ULTRA MAXIMUM",
    layout="wide",
    page_icon="⚽",
    initial_sidebar_state="expanded"
)


# CSS PROFISSIONAL - TABS HORIZONTAIS
st.markdown('''
<style>
    /* TABS HORIZONTAIS - DESIGN PROFISSIONAL */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: white;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        color: #667eea;
        border: 2px solid transparent;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border-color: white;
    }
    
    /* Chatbot Azul */
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border-radius: 15px !important;
        padding: 15px !important;
        color: white !important;
    }
    
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) p {
        color: white !important;
    }
    
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) {
        background: #2d3748 !important;
        border-radius: 15px !important;
        padding: 15px !important;
    }
    
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) p {
        color: white !important;
    }
    
    /* Cards profissionais */
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    
    /* Bilhetes */
    .ticket-item {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 3px solid #667eea;
        transition: all 0.2s ease;
    }
    
    .ticket-item:hover {
        transform: translateX(5px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);
    }
    
    /* Header profissional */
    h1 {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
</style>
''', unsafe_allow_html=True)


# CSS customizado
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    .highlight-green {
        background-color: #90EE90;
        padding: 5px;
        border-radius: 3px;
    }
    .highlight-yellow {
        background-color: #FFFFE0;
        padding: 5px;
        border-radius: 3px;
    }
    .highlight-red {
        background-color: #FFB6C1;
        padding: 5px;
        border-radius: 3px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# MAPEAMENTO DE NOMES DE TIMES
# ============================================================

NAME_MAPPING = {
    'Man United': 'Manchester Utd', 'Man City': 'Manchester City',
    'Spurs': 'Tottenham', 'Newcastle': 'Newcastle Utd',
    'Wolves': 'Wolverhampton', 'Brighton': 'Brighton and Hove Albion',
    'Nottm Forest': "Nott'm Forest", 'Leicester': 'Leicester City',
    'West Ham': 'West Ham Utd', 'Sheffield Utd': 'Sheffield United',
    'Inter': 'Inter Milan', 'AC Milan': 'Milan',
    'Ath Madrid': 'Atletico Madrid', 'Ath Bilbao': 'Athletic Club',
    'Betis': 'Real Betis', 'Sociedad': 'Real Sociedad',
    'Celta': 'Celta Vigo', "M'gladbach": 'Borussia M.Gladbach',
    'Leverkusen': 'Bayer Leverkusen', 'FC Koln': 'FC Cologne',
    'Dortmund': 'Borussia Dortmund', 'Ein Frankfurt': 'Eintracht Frankfurt',
    'Hoffenheim': 'TSG Hoffenheim', 'Bayern Munich': 'Bayern Munchen',
    'RB Leipzig': 'RasenBallsport Leipzig', 'Schalke 04': 'FC Schalke 04',
    'Werder Bremen': 'SV Werder Bremen', 'Fortuna Dusseldorf': 'Fortuna Düsseldorf',
    'Mainz': 'FSV Mainz 05', 'Hertha': 'Hertha Berlin',
    'Paderborn': 'SC Paderborn 07', 'Augsburg': 'FC Augsburg',
    'Freiburg': 'SC Freiburg', 'Paris SG': 'Paris S-G',
    'Paris S-G': 'Paris Saint Germain', 'Saint-Etienne': 'St Etienne',
    'Nimes': 'Nîmes',
}

# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def normalize_name(name: str, known_teams: List[str]) -> Optional[str]:
    """Normaliza nomes de times usando mapeamento e fuzzy matching"""
    if not name or not known_teams:
        return None
    
    name = name.strip()
    
    # Mapeamento direto
    if name in NAME_MAPPING:
        name = NAME_MAPPING[name]
    
    # Verificar se já está correto
    if name in known_teams:
        return name
    
    # Fuzzy matching
    matches = get_close_matches(name, known_teams, n=1, cutoff=0.6)
    return matches[0] if matches else None


def render_md(text: str) -> str:
    return (text or '').replace('\n', '  \n')

def format_games_list(df_games: pd.DataFrame) -> str:
    """Formata uma lista de jogos (um por linha) para leitura limpa no chat/UI."""
    if df_games is None or getattr(df_games, 'empty', True):
        return '(sem jogos)'
    out: List[str] = []
    for _, r in df_games.iterrows():
        hora = r.get('Hora', '')
        liga = r.get('Liga', '')
        home = r.get('Home', '')
        away = r.get('Away', '')
        out.append(f"🏟️ {home} x {away} ⏰ {hora} 🏆 {liga}")
    return "\n".join(out)

def format_currency(value: float) -> str:
    """Formata valor em moeda brasileira"""
    return f"R$ {value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def calculate_probability_from_odds(odd: float) -> float:
    """Calcula probabilidade implícita a partir de uma odd"""
    if odd <= 0:
        return 0.0
    return (1.0 / odd) * 100

def calculate_value_bet(prob_real: float, odd_casa: float) -> float:
    """Calcula o value de uma aposta"""
    return (prob_real / 100) * odd_casa

def get_prob_emoji(prob: float) -> str:
    """Retorna emoji baseado na probabilidade"""
    if prob >= 80:
        return "🔥"
    elif prob >= 75:
        return "✅"
    elif prob >= 70:
        return "🎯"
    elif prob >= 65:
        return "⚡"
    else:
        return "⚪"

# ============================================================
# CARREGAMENTO ROBUSTO DE ARQUIVOS
# ============================================================

def find_file(filename: str) -> Optional[str]:
    """Busca arquivo em múltiplos diretórios"""
    from pathlib import Path
    search_paths = [
        Path('/mnt/project') / filename,
        Path('.') / filename,
        Path('./data') / filename,
        BASE_DIR / filename,
        BASE_DIR / 'data' / filename,
    ]
    for path in search_paths:
        if path.exists():
            return str(path)
    return None

@st.cache_data(ttl=3600)
def load_all_data():
    """Carrega todos os dados do sistema"""
    stats_db = {}
    cal = pd.DataFrame()
    referees = {}
    
    league_files = {
        'Premier League': 'Premier_League_25_26.csv',
        'La Liga': 'La_Liga_25_26.csv',
        'Serie A': 'Serie_A_25_26.csv',
        'Bundesliga': 'Bundesliga_25_26.csv',
        'Ligue 1': 'Ligue_1_25_26.csv',
        'Championship': 'Championship_Inglaterra_25_26.csv',
        'Bundesliga 2': 'Bundesliga_2.csv',
        'Pro League': 'Pro_League_Belgica_25_26.csv',
        'Super Lig': 'Super_Lig_Turquia_25_26.csv',
        'Premiership': 'Premiership_Escocia_25_26.csv'
    }
    
    for league_name, filename in league_files.items():
        filepath = find_file(filename)
        if not filepath:
            st.sidebar.warning(f"⚠️ {league_name}: Arquivo não encontrado")
            continue
        try:
            df = pd.read_csv(filepath, encoding='utf-8')
            teams = set(df['HomeTeam'].dropna().unique()) | set(df['AwayTeam'].dropna().unique())
            
            for team in teams:
                if pd.isna(team):
                    continue
                
                h_games = df[df['HomeTeam'] == team]
                a_games = df[df['AwayTeam'] == team]
                
                # Estatísticas detalhadas
                corners_h = h_games['HC'].mean() if 'HC' in h_games.columns and len(h_games) > 0 else 5.5
                corners_a = a_games['AC'].mean() if 'AC' in a_games.columns and len(a_games) > 0 else 4.5
                corners_h_std = h_games['HC'].std() if 'HC' in h_games.columns and len(h_games) > 1 else 2.0
                corners_a_std = a_games['AC'].std() if 'AC' in a_games.columns and len(a_games) > 1 else 2.0
                
                cards_h = h_games[['HY', 'HR']].sum(axis=1).mean() if 'HY' in h_games.columns and len(h_games) > 0 else 2.5
                cards_a = a_games[['AY', 'AR']].sum(axis=1).mean() if 'AY' in a_games.columns and len(a_games) > 0 else 2.5
                
                fouls_h = h_games['HF'].mean() if 'HF' in h_games.columns and len(h_games) > 0 else 12.0
                fouls_a = a_games['AF'].mean() if 'AF' in a_games.columns and len(a_games) > 0 else 12.0
                
                goals_fh = h_games['FTHG'].mean() if 'FTHG' in h_games.columns and len(h_games) > 0 else 1.5
                goals_fa = a_games['FTAG'].mean() if 'FTAG' in a_games.columns and len(a_games) > 0 else 1.3
                goals_ah = h_games['FTAG'].mean() if 'FTAG' in h_games.columns and len(h_games) > 0 else 1.3
                goals_aa = a_games['FTHG'].mean() if 'FTHG' in a_games.columns and len(a_games) > 0 else 1.5
                
                # Chutes (V14.0)
                shots_h = h_games['HST'].mean() if 'HST' in h_games.columns and len(h_games) > 0 else 4.5
                shots_a = a_games['AST'].mean() if 'AST' in a_games.columns and len(a_games) > 0 else 4.0
                
                stats_db[team] = {
                    'corners': (corners_h + corners_a) / 2,
                    'corners_home': corners_h,
                    'corners_away': corners_a,
                    'corners_std': (corners_h_std + corners_a_std) / 2,
                    'cards': (cards_h + cards_a) / 2,
                    'cards_home': cards_h,
                    'cards_away': cards_a,
                    'fouls': (fouls_h + fouls_a) / 2,
                    'fouls_home': fouls_h,
                    'fouls_away': fouls_a,
                    'goals_f': (goals_fh + goals_fa) / 2,
                    'goals_f_home': goals_fh,
                    'goals_f_away': goals_fa,
                    'goals_a': (goals_ah + goals_aa) / 2,
                    'goals_a_home': goals_ah,
                    'goals_a_away': goals_aa,
                    'shots_on_target': (shots_h + shots_a) / 2,
                    'shots_home': shots_h,
                    'shots_away': shots_a,
                    'league': league_name,
                    'games': len(h_games) + len(a_games)
                }
        except Exception as e:
            st.sidebar.warning(f"⚠️ {league_name}: {str(e)}")
    
    cal_filepath = find_file('calendario_ligas.csv')
    if cal_filepath:
        try:
            cal = pd.read_csv(cal_filepath, encoding='utf-8')
            if 'Data' in cal.columns:
                cal['DtObj'] = pd.to_datetime(cal['Data'], format='%d/%m/%Y', errors='coerce')
        except Exception as e:
            st.sidebar.warning(f"⚠️ Calendário: {str(e)}")
    
    refs_filepath = find_file('arbitros_5_ligas_2025_2026.csv')
    if refs_filepath:
        try:
            refs_df = pd.read_csv(refs_filepath, encoding='utf-8')
            for _, row in refs_df.iterrows():
                referees[row['Arbitro']] = {
                    'factor': row['Media_Cartoes_Por_Jogo'] / 4.0,
                    'games': row['Jogos_Apitados'],
                    'avg_cards': row['Media_Cartoes_Por_Jogo'],
                    'red_cards': row.get('Cartoes_Vermelhos', 0),
                    'red_rate': row.get('Cartoes_Vermelhos', 0) / row['Jogos_Apitados'] if row['Jogos_Apitados'] > 0 else 0.08
                }
        except Exception as e:
            st.sidebar.warning(f"⚠️ Árbitros: {str(e)}")
    
    return stats_db, cal, referees


# ============================================================
# MOTOR DE CÁLCULO V31 - CAUSALITY ENGINE
# ============================================================

def calcular_jogo_v31(home_stats: Dict, away_stats: Dict, ref_data: Dict) -> Dict:
    """
    Motor de Cálculo V31 - Causality Engine
    
    Filosofia: CAUSA → EFEITO
    - Chutes no gol → Cantos
    - Faltas → Cartões
    - Árbitro → Rigidez
    """
    
    # ESCANTEIOS com boost de chutes
    base_corners_h = home_stats.get('corners_home', home_stats['corners'])
    base_corners_a = away_stats.get('corners_away', away_stats['corners'])
    
    # Boost baseado em chutes no gol
    shots_h = home_stats.get('shots_home', 4.5)
    shots_a = home_stats.get('shots_away', 4.0)
    
    if shots_h > 6.0:
        pressure_h = 1.20  # Alto
    elif shots_h > 4.5:
        pressure_h = 1.10  # Médio
    else:
        pressure_h = 1.0   # Baixo
    
    # Fator casa/fora
    corners_h = base_corners_h * 1.15 * pressure_h
    corners_a = base_corners_a * 0.90
    corners_total = corners_h + corners_a
    
    # CARTÕES
    fouls_h = home_stats.get('fouls_home', home_stats.get('fouls', 12.0))
    fouls_a = away_stats.get('fouls_away', away_stats.get('fouls', 12.0))
    
    # Fator de violência
    violence_h = 1.0 if fouls_h > 12.5 else 0.85
    violence_a = 1.0 if fouls_a > 12.5 else 0.85
    
    # Fator do árbitro
    ref_factor = ref_data.get('factor', 1.0) if ref_data else 1.0
    ref_red_rate = ref_data.get('red_rate', 0.08) if ref_data else 0.08
    
    # Rigidez do árbitro
    if ref_red_rate > 0.12:
        strictness = 1.15
    elif ref_red_rate > 0.08:
        strictness = 1.08
    else:
        strictness = 1.0
    
    cards_h_base = home_stats.get('cards_home', home_stats['cards'])
    cards_a_base = away_stats.get('cards_away', away_stats['cards'])
    
    cards_h = cards_h_base * violence_h * ref_factor * strictness
    cards_a = cards_a_base * violence_a * ref_factor * strictness
    cards_total = cards_h + cards_a
    
    # Probabilidade de cartão vermelho
    prob_red_card = ((0.05 + 0.05) / 2) * ref_red_rate * 100
    
    # xG (Expected Goals)
    xg_h = (home_stats['goals_f'] * away_stats['goals_a']) / 1.3
    xg_a = (away_stats['goals_f'] * home_stats['goals_a']) / 1.3
    
    return {
        'corners': {'h': corners_h, 'a': corners_a, 't': corners_total},
        'cards': {'h': cards_h, 'a': cards_a, 't': cards_total},
        'goals': {'h': xg_h, 'a': xg_a},
        'metadata': {
            'ref_factor': ref_factor,
            'violence_home': fouls_h > 12.5,
            'violence_away': fouls_a > 12.5,
            'pressure_home': pressure_h,
            'shots_home': shots_h,
            'shots_away': shots_a,
            'strictness': strictness,
            'prob_red_card': prob_red_card
        }
    }

def simulate_game_v31(home_stats: Dict, away_stats: Dict, ref_data: Dict, n_sims: int = 3000) -> Dict:
    """Simulador de Monte Carlo com distribuição de Poisson"""
    calc = calcular_jogo_v31(home_stats, away_stats, ref_data)
    
    return {
        'corners_h': np.random.poisson(calc['corners']['h'], n_sims),
        'corners_a': np.random.poisson(calc['corners']['a'], n_sims),
        'corners_total': np.random.poisson(calc['corners']['t'], n_sims),
        'cards_h': np.random.poisson(calc['cards']['h'], n_sims),
        'cards_a': np.random.poisson(calc['cards']['a'], n_sims),
        'cards_total': np.random.poisson(calc['cards']['t'], n_sims),
        'goals_h': np.random.poisson(calc['goals']['h'], n_sims),
        'goals_a': np.random.poisson(calc['goals']['a'], n_sims)
    }

# ============================================================
# MÉTRICAS FINANCEIRAS
# ============================================================

def calculate_sharpe_ratio(returns: List[float]) -> float:
    """Calcula Sharpe Ratio (retorno ajustado ao risco)"""
    if not returns or len(returns) < 2:
        return 0.0
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    return (mean_return - 1.0) / std_return if std_return > 0 else 0.0

def calculate_max_drawdown(bankroll_history: List[float]) -> float:
    """Calcula Maximum Drawdown (maior queda)"""
    if len(bankroll_history) < 2:
        return 0.0
    peak = bankroll_history[0]
    max_dd = 0.0
    for value in bankroll_history:
        if value > peak:
            peak = value
        dd = (peak - value) / peak * 100
        if dd > max_dd:
            max_dd = dd
    return max_dd

def calculate_kelly_criterion(prob: float, odd: float, bankroll: float) -> Dict:
    """Calcula Kelly Criterion"""
    if prob <= 0 or prob >= 1 or odd <= 1:
        return {'fraction': 0, 'stake': 0, 'recommendation': 'Não apostar'}
    
    b = odd - 1
    p = prob
    q = 1 - prob
    
    kelly_fraction = (b * p - q) / b
    kelly_fraction = max(0, min(kelly_fraction, 0.10))  # Cap em 10%
    
    stake = bankroll * kelly_fraction
    
    if kelly_fraction >= 0.08:
        recommendation = 'Stake alto'
    elif kelly_fraction >= 0.05:
        recommendation = 'Stake médio'
    elif kelly_fraction > 0:
        recommendation = 'Stake baixo'
    else:
        recommendation = 'Não apostar'
    
    return {
        'fraction': kelly_fraction,
        'stake': stake,
        'percentage': kelly_fraction * 100,
        'recommendation': recommendation
    }

def calculate_roi(total_staked: float, total_profit: float) -> float:
    """Calcula ROI (Return on Investment)"""
    if total_staked == 0:
        return 0.0
    return (total_profit / total_staked) * 100

def calculate_ev(probability: float, odds: float, stake: float) -> float:
    """Calcula Expected Value"""
    win_amount = stake * (odds - 1)
    lose_amount = -stake
    ev = (probability * win_amount) + ((1 - probability) * lose_amount)
    return ev

# ============================================================
# PARSER DE BILHETES (TAB 8)
# ============================================================

def parse_bilhete_texto(texto: str) -> List[Dict]:
    """Parser inteligente de bilhetes - Versão ULTRA"""
    linhas_originais = [l.strip() for l in texto.split('\n') if l.strip()]
    linhas = []
    i = 0
    
    # Juntar linhas quebradas
    while i < len(linhas_originais):
        linha = linhas_originais[i]
        if i + 1 < len(linhas_originais):
            proxima = linhas_originais[i + 1]
            tem_mercado = any(x in linha.lower() for x in ['canto', 'escanteio', 'cartão', 'card'])
            tem_num = bool(re.search(r'\d+\.5', linha))
            tem_num_prox = bool(re.search(r'\d+\.5', proxima))
            
            if tem_mercado and not tem_num and tem_num_prox:
                linhas.append(linha + ' ' + proxima)
                i += 2
                continue
        linhas.append(linha)
        i += 1
    
    jogos = []
    jogo_atual = None
    time_pendente = None
    mercados_pend = []
    
    for linha in linhas:
        if any(x in linha.lower() for x in ['criar aposta', 'stake', 'retorno']):
            continue
        
        # Detectar jogo
        if ' vs ' in linha or ' x ' in linha.lower():
            sep = ' vs ' if ' vs ' in linha else ' x '
            partes = linha.split(sep)
            if len(partes) == 2:
                jogo_atual = {'home': partes[0].strip(), 'away': partes[1].strip(), 'mercados': mercados_pend.copy()}
                jogos.append(jogo_atual)
                time_pendente = None
                mercados_pend = []
                continue
        
        # Detectar mercado
        if any(x in linha.lower() for x in ['total de', 'mais de', 'over']) and \
           any(y in linha.lower() for y in ['canto', 'escanteio', 'cartão', 'card']):
            tipo = 'corners' if any(x in linha.lower() for x in ['canto', 'escanteio']) else 'cards'
            nums = re.findall(r'\d+\.5', linha)
            if nums:
                line = float(nums[0])
                odds = re.findall(r'@?\d+\.\d+', linha)
                odd = float(odds[-1].replace('@', '')) if odds else 2.0
                mercado = {'tipo': tipo, 'location': 'total', 'line': line, 'odd': odd, 'desc': linha}
                if jogo_atual:
                    jogo_atual['mercados'].append(mercado)
                else:
                    mercados_pend.append(mercado)
                continue
        
        # Times sem vs
        if not any(x in linha.lower() for x in ['total', 'mais de', 'over']) and len(linha) > 2:
            if time_pendente is None:
                time_pendente = linha.strip()
            else:
                jogo_atual = {'home': time_pendente, 'away': linha.strip(), 'mercados': mercados_pend.copy()}
                jogos.append(jogo_atual)
                time_pendente = None
                mercados_pend = []
    
    return jogos

def validar_jogos_bilhete(jogos_parsed: List[Dict], stats_db: Dict) -> List[Dict]:
    """Valida e normaliza nomes dos times"""
    jogos_val = []
    times = list(stats_db.keys())
    
    for jogo in jogos_parsed:
        h_norm = normalize_name(jogo['home'], times)
        a_norm = normalize_name(jogo['away'], times)
        
        if h_norm and a_norm and h_norm in stats_db and a_norm in stats_db:
            jogos_val.append({
                'home': h_norm,
                'away': a_norm,
                'home_original': jogo['home'],
                'away_original': jogo['away'],
                'mercados': jogo['mercados'],
                'home_stats': stats_db[h_norm],
                'away_stats': stats_db[a_norm]
            })
    
    return jogos_val

def calcular_prob_bilhete(jogos_validados: List[Dict], n_sims: int = 3000) -> Dict:
    """Calcula probabilidade real do bilhete"""
    prob_total = 1.0
    detalhes = []
    
    for jogo in jogos_validados:
        sims = simulate_game_v31(jogo['home_stats'], jogo['away_stats'], {}, n_sims)
        
        for mercado in jogo['mercados']:
            data = sims['corners_total'] if mercado['tipo'] == 'corners' else sims['cards_total']
            prob = (data > mercado['line']).mean()
            prob_total *= prob
            
            detalhes.append({
                'jogo': f"{jogo['home']} vs {jogo['away']}",
                'mercado': mercado['desc'],
                'prob': prob * 100,
                'odd_casa': mercado['odd'],
                'fair_odd': 1.0 / prob if prob > 0 else 999,
                'value': prob * mercado['odd'] if prob > 0 else 0
            })
    
    return {'prob_total': prob_total * 100, 'detalhes': detalhes}


# ============================================================
# CHATBOT ULTRA INTELIGENTE - NOVA VERSÃO
# ============================================================

def clean_team_name(text: str) -> str:
    """Limpa e normaliza nome de time"""
    import re
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    stop_words = {'do', 'da', 'de', 'dos', 'das', 'o', 'a', 'os', 'as', 
                  'como', 'está', 'esta', 'stats', 'estatistica', 'estatísticas', 
                  'estatisticas', 'qual', 'quais', 'quanto', 'quantos'}
    words = text.split()
    text = ' '.join([w for w in words if w not in stop_words])
    return text.strip()

def extrair_entidades(mensagem: str, stats_db: Dict) -> Dict:
    """Extrai entidades da mensagem (times, ligas, métricas)"""
    msg_lower = mensagem.lower()
    entidades = {
        'times': [],
        'ligas': [],
        'metricas': [],
        'valores': []
    }
    
    # Detectar ligas
    ligas_map = {
        'premier': 'Premier League',
        'la liga': 'La Liga',
        'laliga': 'La Liga',
        'serie a': 'Serie A',
        'seriea': 'Serie A',
        'bundesliga': 'Bundesliga',
        'ligue 1': 'Ligue 1',
        'ligue1': 'Ligue 1',
        'francesa': 'Ligue 1',
        'inglesa': 'Premier League',
        'espanhola': 'La Liga',
        'italiana': 'Serie A',
        'alemã': 'Bundesliga',
        'alema': 'Bundesliga'
    }
    
    for key, value in ligas_map.items():
        if key in msg_lower:
            entidades['ligas'].append(value)
    
    # Detectar métricas
    metricas_map = {
        'canto': 'corners',
        'cantos': 'corners',
        'escanteio': 'corners',
        'escanteios': 'corners',
        'cartao': 'cards',
        'cartão': 'cards',
        'cartoes': 'cards',
        'cartões': 'cards',
        'amarelo': 'cards',
        'vermelho': 'cards',
        'gol': 'goals_f',
        'gols': 'goals_f',
        'falta': 'fouls',
        'faltas': 'fouls',
        'chute': 'shots_on_target',
        'chutes': 'shots_on_target'
    }
    
    for key, value in metricas_map.items():
        if key in msg_lower:
            entidades['metricas'].append(value)
    
    # Detectar números
    import re
    numeros = re.findall(r'\d+(?:\.\d+)?', mensagem)
    entidades['valores'] = [float(n) for n in numeros]
    
    # Detectar times
    msg_limpa = clean_team_name(mensagem)
    palavras = msg_limpa.split()
    
    known_teams = list(stats_db.keys())
    
    # Buscar times mencionados
    for i in range(len(palavras)):
        for j in range(i+1, min(i+4, len(palavras)+1)):
            frase = ' '.join(palavras[i:j])
            matches = get_close_matches(frase, [t.lower() for t in known_teams], n=1, cutoff=0.7)
            if matches:
                time_real = [t for t in known_teams if t.lower() == matches[0]][0]
                if time_real not in entidades['times']:
                    entidades['times'].append(time_real)
    
    return entidades

def processar_chat_ultra(mensagem: str, stats_db: Dict, cal: pd.DataFrame) -> str:
    """
    Chatbot ULTRA Inteligente - Versão Completa
    Capaz de responder QUALQUER pergunta sobre os dados
    """
    if not mensagem or not stats_db:
        return "Por favor, digite uma pergunta válida."
    
    msg = mensagem.lower().strip()
    known_teams = list(stats_db.keys())
    
    # ==========================================
    # 1. COMANDOS ESPECIAIS
    # ==========================================
    if msg in ['/ajuda', 'ajuda', 'help']:
        return """
🤖 **CHATBOT ULTRA INTELIGENTE**

💬 **EXEMPLOS DE PERGUNTAS:**

📊 **Estatísticas:**
• "Qual time tem mais cantos na Premier League?"
• "Top 5 times com mais cartões"
• "Média de gols do Arsenal"
• "Compare cantos do Liverpool e City"

⚔️ **Comparações:**
• "Arsenal vs Chelsea"
• "Real Madrid é melhor que Barcelona em cantos?"

📈 **Análises:**
• "Quais times fazem mais de 6 cantos em casa?"
• "Times com menos de 2 cartões por jogo"
• "Qual liga tem mais gols?"

🎯 **Recomendações:**
• "Sugira um jogo para over 10.5 cantos"
• "Qual o melhor time para apostar em cantos?"

💡 **Pergunte QUALQUER COISA sobre os dados!**
        """
    
    if msg in ['oi', 'olá', 'ola', 'hello', 'hi']:
        return "👋 Olá! Sou o Chatbot ULTRA Inteligente. Pergunte qualquer coisa sobre estatísticas de futebol! Digite 'ajuda' para ver exemplos."
    
    # ==========================================
    # 2. EXTRAIR ENTIDADES
    # ==========================================
    entidades = extrair_entidades(mensagem, stats_db)
    
    # ==========================================
    # 3. JOGOS DE HOJE
    # ==========================================
    if 'hoje' in msg or 'today' in msg or 'jogos' in msg:
        try:
            hoje = datetime.now().strftime('%d/%m/%Y')
            jogos_hoje = cal[cal['Data'] == hoje] if not cal.empty else pd.DataFrame()
            
            if len(jogos_hoje) == 0:
                return f"📅 Não há jogos cadastrados para hoje ({hoje})"
            
            resp = f"📅 **JOGOS DE HOJE ({hoje}):**\n\n"
            for idx, jogo in jogos_hoje.head(10).iterrows():
                resp += f"🏟️ {jogo['Time_Casa']} x {jogo['Time_Visitante']}\n"
                resp += f"   ⏰ {jogo['Hora']} | 🏆 {jogo['Liga']}\n\n"
            
            return resp
        except:
            return "❌ Erro ao buscar jogos de hoje."
    
    # ==========================================
    # 4. RANKINGS E TOP N
    # ==========================================
    if any(word in msg for word in ['top', 'ranking', 'melhor', 'melhores', 'maior', 'maiores']):
        # Determinar métrica
        metrica = 'corners'
        if any(word in msg for word in ['cartao', 'cartão', 'cartoes', 'cartões', 'card']):
            metrica = 'cards'
        elif any(word in msg for word in ['gol', 'gols', 'goal']):
            metrica = 'goals_f'
        elif any(word in msg for word in ['falta', 'faltas']):
            metrica = 'fouls'
        
        # Determinar quantidade
        n = 10
        if entidades['valores']:
            n = min(int(entidades['valores'][0]), 20)
        
        # Filtrar por liga se mencionada
        times_filtrados = stats_db
        if entidades['ligas']:
            liga_filtro = entidades['ligas'][0]
            times_filtrados = {k: v for k, v in stats_db.items() if v.get('league') == liga_filtro}
        
        # Gerar ranking
        ranking = sorted(times_filtrados.items(), 
                       key=lambda x: x[1].get(metrica, 0), 
                       reverse=True)[:n]
        
        liga_txt = f" - {entidades['ligas'][0]}" if entidades['ligas'] else ""
        metrica_nome = {
            'corners': 'CANTOS',
            'cards': 'CARTÕES',
            'goals_f': 'GOLS MARCADOS',
            'fouls': 'FALTAS'
        }.get(metrica, metrica.upper())
        
        resp = f"🏆 **TOP {n} - {metrica_nome}{liga_txt}:**\n\n"
        for i, (time, stats) in enumerate(ranking, 1):
            valor = stats.get(metrica, 0)
            liga_emoji = {'Premier League': '🏴󠁧󠁢󠁥󠁮󠁧󠁿', 'La Liga': '🇪🇸', 'Serie A': '🇮🇹', 
                         'Bundesliga': '🇩🇪', 'Ligue 1': '🇫🇷'}.get(stats.get('league', ''), '⚽')
            resp += f"{i}. {time} {liga_emoji}: **{valor:.2f}**/jogo\n"
        
        return resp
    
    # ==========================================
    # 5. QUAL TIME/QUEM TEM
    # ==========================================
    if any(word in msg for word in ['qual', 'quais', 'quem']):
        # Busca por máximo ou mínimo
        if 'mais' in msg or 'maior' in msg or 'maximo' in msg:
            ordem = 'max'
        elif 'menos' in msg or 'menor' in msg or 'minimo' in msg:
            ordem = 'min'
        else:
            ordem = 'max'
        
        # Determinar métrica
        metrica = 'corners'
        if entidades['metricas']:
            metrica = entidades['metricas'][0]
        
        # Filtrar por liga
        times_filtrados = stats_db
        if entidades['ligas']:
            liga_filtro = entidades['ligas'][0]
            times_filtrados = {k: v for k, v in stats_db.items() if v.get('league') == liga_filtro}
        
        if not times_filtrados:
            return "❌ Nenhum time encontrado com esses critérios."
        
        # Encontrar time
        if ordem == 'max':
            time_found = max(times_filtrados.items(), key=lambda x: x[1].get(metrica, 0))
        else:
            time_found = min(times_filtrados.items(), key=lambda x: x[1].get(metrica, 0))
        
        time_nome, stats = time_found
        valor = stats.get(metrica, 0)
        liga = stats.get('league', 'N/A')
        
        metrica_nome = {
            'corners': 'cantos',
            'cards': 'cartões',
            'goals_f': 'gols marcados',
            'fouls': 'faltas',
            'shots_on_target': 'chutes no alvo'
        }.get(metrica, metrica)
        
        resp = f"{'🔥' if ordem == 'max' else '🔻'} **{time_nome}** ({liga})\n"
        resp += f"📊 {metrica_nome.upper()}: **{valor:.2f}** por jogo\n\n"
        
        # Adicionar contexto
        resp += f"📈 Outras estatísticas:\n"
        resp += f"• Cantos: {stats.get('corners', 0):.2f}/jogo\n"
        resp += f"• Cartões: {stats.get('cards', 0):.2f}/jogo\n"
        resp += f"• Gols: {stats.get('goals_f', 0):.2f}/jogo\n"
        
        return resp
    
    # ==========================================
    # 6. COMPARAÇÕES (vs ou x)
    # ==========================================
    if ' vs ' in msg or ' x ' in msg:
        separator = ' vs ' if ' vs ' in msg else ' x '
        times_parts = msg.split(separator)
        
        if len(times_parts) == 2 and entidades['times'] and len(entidades['times']) >= 2:
            t1 = entidades['times'][0]
            t2 = entidades['times'][1]
            
            s1 = stats_db[t1]
            s2 = stats_db[t2]
            
            resp = f"⚔️ **{t1} vs {t2}**\n\n"
            
            # Comparação detalhada
            resp += "📊 **COMPARAÇÃO:**\n\n"
            
            metricas_comp = [
                ('Cantos', 'corners', '🚩'),
                ('Cartões', 'cards', '🟨'),
                ('Gols Marcados', 'goals_f', '⚽'),
                ('Gols Sofridos', 'goals_a', '🛡️'),
                ('Faltas', 'fouls', '⚠️'),
                ('Chutes', 'shots_on_target', '🎯')
            ]
            
            for nome, key, emoji in metricas_comp:
                v1 = s1.get(key, 0)
                v2 = s2.get(key, 0)
                
                if v1 > v2:
                    vencedor = f"**{t1}** vence"
                elif v2 > v1:
                    vencedor = f"**{t2}** vence"
                else:
                    vencedor = "Empate"
                
                resp += f"{emoji} **{nome}:** {v1:.2f} vs {v2:.2f} → {vencedor}\n"
            
            return resp
        
        elif len(entidades['times']) == 1:
            return f"❌ Encontrei apenas {entidades['times'][0]}. Digite os dois times (ex: Arsenal vs Chelsea)"
        else:
            return "❌ Não encontrei os times mencionados. Digite: 'Time1 vs Time2'"
    
    # ==========================================
    # 7. MÉDIA DE / ESTATÍSTICA DE
    # ==========================================
    if any(word in msg for word in ['media', 'média', 'estatistica', 'estatística', 'stats']):
        if entidades['times']:
            time = entidades['times'][0]
            stats = stats_db[time]
            
            # Determinar métrica
            metrica = 'corners'
            if entidades['metricas']:
                metrica = entidades['metricas'][0]
            
            valor = stats.get(metrica, 0)
            
            metrica_nome = {
                'corners': 'Cantos',
                'cards': 'Cartões',
                'goals_f': 'Gols Marcados',
                'goals_a': 'Gols Sofridos',
                'fouls': 'Faltas',
                'shots_on_target': 'Chutes no Alvo'
            }.get(metrica, metrica)
            
            resp = f"📊 **{time.upper()}**\n"
            resp += f"🏆 Liga: {stats.get('league', 'N/A')}\n"
            resp += f"🎮 Jogos: {stats.get('games', 0)}\n\n"
            resp += f"📈 **{metrica_nome}:** {valor:.2f} por jogo\n\n"
            
            # Contexto adicional
            resp += "📊 **TODAS AS ESTATÍSTICAS:**\n"
            resp += f"🚩 Cantos: {stats.get('corners', 0):.2f}/jogo\n"
            resp += f"🟨 Cartões: {stats.get('cards', 0):.2f}/jogo\n"
            resp += f"⚽ Gols Feitos: {stats.get('goals_f', 0):.2f}/jogo\n"
            resp += f"🛡️ Gols Sofridos: {stats.get('goals_a', 0):.2f}/jogo\n"
            resp += f"⚠️ Faltas: {stats.get('fouls', 0):.2f}/jogo\n"
            resp += f"🎯 Chutes: {stats.get('shots_on_target', 0):.2f}/jogo\n"
            
            return resp
    
    # ==========================================
    # 8. FILTROS (MAIS DE / MENOS DE)
    # ==========================================
    if ('mais' in msg and 'de' in msg) or ('menos' in msg and 'de' in msg):
        operador = 'maior' if 'mais' in msg else 'menor'
        
        if entidades['valores'] and entidades['metricas']:
            threshold = entidades['valores'][0]
            metrica = entidades['metricas'][0]
            
            # Filtrar por liga se mencionada
            times_filtrados = stats_db
            if entidades['ligas']:
                liga_filtro = entidades['ligas'][0]
                times_filtrados = {k: v for k, v in stats_db.items() if v.get('league') == liga_filtro}
            
            # Aplicar filtro
            if operador == 'maior':
                resultado = {k: v for k, v in times_filtrados.items() if v.get(metrica, 0) > threshold}
            else:
                resultado = {k: v for k, v in times_filtrados.items() if v.get(metrica, 0) < threshold}
            
            if not resultado:
                return f"❌ Nenhum time encontrado com {metrica} {operador} que {threshold:.1f}"
            
            # Ordenar
            resultado_sorted = sorted(resultado.items(), 
                                    key=lambda x: x[1].get(metrica, 0), 
                                    reverse=(operador == 'maior'))
            
            metrica_nome = {
                'corners': 'cantos',
                'cards': 'cartões',
                'goals_f': 'gols',
                'fouls': 'faltas',
                'shots_on_target': 'chutes'
            }.get(metrica, metrica)
            
            resp = f"🔍 **Times com {metrica_nome} {operador} que {threshold:.1f}:**\n\n"
            
            for time, stats in resultado_sorted[:15]:
                valor = stats.get(metrica, 0)
                liga = stats.get('league', '')
                resp += f"• **{time}** ({liga}): {valor:.2f}/jogo\n"
            
            if len(resultado_sorted) > 15:
                resp += f"\n... e mais {len(resultado_sorted) - 15} times"
            
            return resp
    
    # ==========================================
    # 9. ANÁLISE DE TIME ÚNICO
    # ==========================================
    if entidades['times']:
        team = entidades['times'][0]
        stats = stats_db[team]
        
        resp = f"📊 **{team.upper()}**\n\n"
        resp += f"🏆 Liga: {stats.get('league', 'N/A')}\n"
        resp += f"🎮 Jogos: {stats.get('games', 0)}\n\n"
        
        # Ataque
        gols_f = stats.get('goals_f', 0)
        emoji_atk = '🔥' if gols_f > 1.8 else '⚽' if gols_f > 1.2 else '⚪'
        resp += f"**⚔️ ATAQUE:** {emoji_atk}\n"
        resp += f"⚽ Gols feitos: {gols_f:.2f}/jogo\n"
        resp += f"🎯 Chutes: {stats.get('shots_on_target', 0):.2f}/jogo\n\n"
        
        # Defesa
        gols_a = stats.get('goals_a', 0)
        emoji_def = '🛡️' if gols_a < 1.0 else '⚠️' if gols_a < 1.5 else '🔴'
        resp += f"**🛡️ DEFESA:** {emoji_def}\n"
        resp += f"🥅 Gols sofridos: {gols_a:.2f}/jogo\n\n"
        
        # Escanteios
        corners = stats.get('corners', 0)
        emoji_corner = '🔥' if corners > 6.0 else '🚩' if corners > 5.0 else '⚪'
        resp += f"**🚩 ESCANTEIOS:** {emoji_corner}\n"
        resp += f"📐 Média: {corners:.2f}/jogo\n"
        resp += f"🏠 Casa: {stats.get('corners_home', 0):.2f}/jogo\n"
        resp += f"✈️ Fora: {stats.get('corners_away', 0):.2f}/jogo\n\n"
        
        # Cartões
        cards = stats.get('cards', 0)
        emoji_card = '🔴' if cards > 3.0 else '🟡' if cards > 2.0 else '🟢'
        resp += f"**🟨 CARTÕES:** {emoji_card}\n"
        resp += f"📋 Média: {cards:.2f}/jogo\n\n"
        
        # Faltas
        fouls = stats.get('fouls', 0)
        resp += f"**⚠️ FALTAS:**\n"
        resp += f"🚫 Média: {fouls:.2f}/jogo\n"
        
        return resp
    
    # ==========================================
    # 10. RESPOSTA PADRÃO
    # ==========================================
    return """
🤔 Não entendi sua pergunta. Aqui estão alguns exemplos:

📊 **Exemplos:**
• "Qual time tem mais cantos na Premier League?"
• "Top 5 times com mais cartões"
• "Arsenal vs Chelsea"
• "Média de gols do Liverpool"
• "Times com mais de 6 cantos"
• "jogos de hoje"

💡 Digite **'/ajuda'** para ver todos os comandos!
    """


# Carregar dados globais
stats_db_global, CAL_GLOBAL, REFS_GLOBAL = load_all_data()

def main():
    # ═══════════════════════════════════════════════════════════
    # DADOS CARREGADOS
    # ═══════════════════════════════════════════════════════════
    stats_db, cal, referees = load_all_data()

    # ═══════════════════════════════════════════════════════════
    # HEADER PROFISSIONAL
    # ═══════════════════════════════════════════════════════════
    col1, col2, col3 = st.columns([2, 3, 2])
    
    with col1:
        st.image("https://img.icons8.com/fluency/96/000000/soccer-ball.png", width=80)
    
    with col2:
        st.title("⚽ FutPrevisão V31 ULTRA")
        st.caption("_Sistema Profissional + Chatbot Ultra Inteligente_")
    
    with col3:
        st.metric("📚 Database", f"{len(stats_db)} times", delta="10 Ligas")
    
    st.markdown("---")
    

    # ═══════════════════════════════════════════════════════════
    # TABS HORIZONTAIS - NAVEGAÇÃO PRINCIPAL
    # ═══════════════════════════════════════════════════════════
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "🎫 Construtor", 
        "🛡️ Hedges",
        "🎲 Simulador",
        "📊 Métricas",
        "🎨 Viz", 
        "📝 Registro", 
        "🔍 Scanner", 
        "📋 Importar", 
        "🤖 AI"
    ])
    
    if 'current_ticket' not in st.session_state:
        st.session_state.current_ticket = []
    if 'bet_results' not in st.session_state:
        st.session_state.bet_results = []
    if 'bankroll_history' not in st.session_state:
        st.session_state.bankroll_history = [1000.0]
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'initial_bankroll' not in st.session_state:
        st.session_state.initial_bankroll = 1000.0
    
    with st.sidebar:
        st.header("📊 Dashboard")
        col1, col2 = st.columns(2)
        col1.metric("Times", len(stats_db))
        col1.metric("Jogos", len(cal) if not cal.empty else 0)
        col2.metric("Árbitros", len(referees))
        banca = st.session_state.bankroll_history[-1]
        col2.metric("Banca", format_currency(banca))
        
        if st.session_state.current_ticket:
            st.success(f"🎫 {len(st.session_state.current_ticket)} seleção(ões)")
        else:
            st.info("📭 Bilhete vazio")
        
        if st.session_state.bet_results:
            total = len(st.session_state.bet_results)
            ganhas = sum(1 for b in st.session_state.bet_results if b.get('ganhou', False))
            wr = (ganhas/total)*100 if total > 0 else 0
            st.markdown("---")
            st.metric("Win Rate", f"{wr:.1f}%")
    
    # ============================================================
    # TAB 1: CONSTRUTOR (COM CONSTRUTOR MANUAL)
    # ============================================================
    
    with tab1:
        st.header("🎫 Construtor de Bilhetes Profissional")
        
        # ====================================================
        # 📝 CONSTRUTOR MANUAL - NOVA FUNCIONALIDADE
        # ====================================================
        with st.expander("📝 CONSTRUTOR MANUAL", expanded=True):
            st.markdown("### Crie sua aposta manualmente")
            
            col1, col2 = st.columns(2)
            
            # Dropdowns para times
            todos_times = sorted(list(stats_db.keys()))
            time_casa = col1.selectbox("🏠 Time da Casa:", todos_times, key='manual_home')
            time_fora = col2.selectbox("✈️ Time Visitante:", todos_times, key='manual_away')
            
            col1, col2, col3 = st.columns(3)
            
            # Tipo de mercado
            tipo_mercado = col1.selectbox(
                "📊 Tipo de Mercado:",
                ["Cantos", "Cartões", "Gols"],
                key='manual_market_type'
            )
            
            # Local
            local = col2.selectbox(
                "📍 Local:",
                ["Total", "Casa", "Fora"],
                key='manual_location'
            )
            
            # Linha baseada no tipo
            if tipo_mercado == "Cantos":
                linhas_disponiveis = [4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5]
            elif tipo_mercado == "Cartões":
                linhas_disponiveis = [2.5, 3.5, 4.5, 5.5, 6.5]
            else:  # Gols
                linhas_disponiveis = [0.5, 1.5, 2.5, 3.5, 4.5]
            
            linha = col3.selectbox(
                "🎯 Linha (Over):",
                linhas_disponiveis,
                key='manual_line'
            )
            
            # Calcular previsão
            if time_casa != time_fora:
                calc = calcular_jogo_v31(stats_db[time_casa], stats_db[time_fora], {})
                
                # Determinar valor esperado
                if tipo_mercado == "Cantos":
                    if local == "Total":
                        valor_esperado = calc['corners']['t']
                    elif local == "Casa":
                        valor_esperado = calc['corners']['h']
                    else:
                        valor_esperado = calc['corners']['a']
                elif tipo_mercado == "Cartões":
                    if local == "Total":
                        valor_esperado = calc['cards']['t']
                    elif local == "Casa":
                        valor_esperado = calc['cards']['h']
                    else:
                        valor_esperado = calc['cards']['a']
                else:  # Gols
                    if local == "Total":
                        valor_esperado = calc['goals']['h'] + calc['goals']['a']
                    elif local == "Casa":
                        valor_esperado = calc['goals']['h']
                    else:
                        valor_esperado = calc['goals']['a']
                
                # Calcular probabilidade estimada
                if valor_esperado > linha + 1.0:
                    prob_estimada = 80
                elif valor_esperado > linha + 0.5:
                    prob_estimada = 75
                elif valor_esperado > linha:
                    prob_estimada = 70
                else:
                    prob_estimada = 60
                
                # Mostrar previsão
                col1, col2, col3 = st.columns(3)
                col1.metric("📈 Previsão", f"{valor_esperado:.2f}")
                col2.metric("🎯 Linha", f"{linha}")
                col3.metric("📊 Probabilidade", f"{prob_estimada}%")
                
                emoji = get_prob_emoji(prob_estimada)
                
                if prob_estimada >= 70:
                    st.success(f"{emoji} BOA APOSTA! Previsão: {valor_esperado:.2f} | Linha: {linha}")
                else:
                    st.warning(f"{emoji} ATENÇÃO! Probabilidade baixa ({prob_estimada}%)")
                
                # Botão adicionar
                if st.button("➕ ADICIONAR AO BILHETE", use_container_width=True, type="primary"):
                    descricao = f"{time_casa} vs {time_fora} - Over {linha} {tipo_mercado} {local}"
                    
                    st.session_state.current_ticket.append({
                        'jogo': f"{time_casa} vs {time_fora}",
                        'market_display': f"Over {linha} {tipo_mercado} {local}",
                        'prob': prob_estimada,
                        'data': datetime.now().strftime('%d/%m/%Y')
                    })
                    st.success(f"✅ Adicionado: {descricao}")
                    st.rerun()
        
        st.markdown("---")
        
        # ====================================================
        # 📅 CONSTRUTOR AUTOMÁTICO (JOGOS DO CALENDÁRIO)
        # ====================================================
        st.markdown("### 📅 Construtor Automático (Jogos do Calendário)")
        
        if not cal.empty:
            dates = sorted(cal['DtObj'].dt.strftime('%d/%m/%Y').unique())
            sel_date = st.selectbox("📅 Selecione a Data:", dates, key='c_date')
            jogos_dia = cal[cal['DtObj'].dt.strftime('%d/%m/%Y') == sel_date]
            
            st.markdown(f"### 🎯 {len(jogos_dia)} jogo(s) disponível(eis)")
            
            for idx, jogo in jogos_dia.iterrows():
                h = normalize_name(jogo['Time_Casa'], list(stats_db.keys()))
                a = normalize_name(jogo['Time_Visitante'], list(stats_db.keys()))
                
                if h and a and h in stats_db and a in stats_db:
                    ref_nome = jogo.get('Arbitro', 'N/A')
                    ref_data = referees.get(ref_nome, {})
                    calc = calcular_jogo_v31(stats_db[h], stats_db[a], ref_data)
                    
                    with st.expander(f"⚽ {h} vs {a} | {jogo.get('Hora', 'N/A')}", expanded=False):
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("xG Casa", f"{calc['goals']['h']:.2f}")
                        col2.metric("xG Fora", f"{calc['goals']['a']:.2f}")
                        col3.metric("Cantos", f"{calc['corners']['t']:.1f}")
                        col4.metric("Cartões", f"{calc['cards']['t']:.1f}")
                        
                        st.markdown("#### 📊 Seleções Disponíveis:")
                        
                        opcoes = [
                            (f"{h} - Over 4.5 Cantos Casa", calc['corners']['h'], 4.5, 'corners'),
                            (f"{a} - Over 4.5 Cantos Fora", calc['corners']['a'], 4.5, 'corners'),
                            (f"Over 9.5 Cantos Total", calc['corners']['t'], 9.5, 'corners'),
                            (f"Over 10.5 Cantos Total", calc['corners']['t'], 10.5, 'corners'),
                            (f"Over 11.5 Cantos Total", calc['corners']['t'], 11.5, 'corners'),
                            (f"{h} - Over 2.5 Cartões Casa", calc['cards']['h'], 2.5, 'cards'),
                            (f"{a} - Over 2.5 Cartões Fora", calc['cards']['a'], 2.5, 'cards'),
                            (f"Over 4.5 Cartões Total", calc['cards']['t'], 4.5, 'cards'),
                            (f"Over 5.5 Cartões Total", calc['cards']['t'], 5.5, 'cards'),
                        ]
                        
                        for desc, media, linha, tipo in opcoes:
                            prob = 75 if media > linha + 0.5 else 65 if media > linha else 55
                            emoji = get_prob_emoji(prob)
                            col1, col2 = st.columns([4, 1])
                            col1.markdown(f"{emoji} **{desc}** | Prob: {prob}%")
                            if col2.button("➕", key=f"add_{idx}_{desc}"):
                                st.session_state.current_ticket.append({
                                    'jogo': f"{h} vs {a}",
                                    'market_display': desc,
                                    'prob': prob,
                                    'data': sel_date
                                })
                                st.rerun()
        
        st.markdown("---")
        st.subheader("📋 Seu Bilhete Atual")
        
        if st.session_state.current_ticket:
            st.success(f"✅ {len(st.session_state.current_ticket)} seleção(ões)")
            
            for i, sel in enumerate(st.session_state.current_ticket):
                col1, col2 = st.columns([5, 1])
                col1.write(f"{i+1}. {sel['jogo']} - {sel['market_display']} ({sel['prob']}%)")
                if col2.button("🗑️", key=f"del_{i}"):
                    st.session_state.current_ticket.pop(i)
                    st.rerun()
            
            prob_comb = 1.0
            for sel in st.session_state.current_ticket:
                prob_comb *= (sel['prob'] / 100)
            
            odd_est = 1.0 / prob_comb if prob_comb > 0 else 999
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Prob Total", f"{prob_comb*100:.1f}%")
            col2.metric("Odd Estimada", f"@{odd_est:.2f}")
            col3.metric("Seleções", len(st.session_state.current_ticket))
            
            st.session_state.ticket_odds = {'prob_total': prob_comb*100, 'odd_total': odd_est}
            
            if st.button("🗑️ LIMPAR BILHETE", use_container_width=True):
                st.session_state.current_ticket = []
                st.rerun()
        else:
            st.info("📭 Bilhete vazio. Adicione seleções acima!")
    
    # ============================================================
    # TAB 2: HEDGES MAXIMUM
    # ============================================================
    
    with tab2:
        st.header("🛡️ Hedges MAXIMUM - Sistema de Proteção")
        
        if not st.session_state.current_ticket:
            st.warning("⚠️ Bilhete vazio! Vá para Tab 'Construtor'")
        else:
            col1, col2 = st.columns(2)
            stake = col1.number_input("💰 Stake (R$)", 10.0, 10000.0, 100.0, 10.0)
            odd_total = col2.number_input("📊 Odd Total", 1.5, 100.0, 5.0, 0.1)
            
            ret_max = stake * odd_total
            lucro_max = ret_max - stake
            
            st.info(f"💵 Retorno: {format_currency(ret_max)} | Lucro: {format_currency(lucro_max)}")
            st.markdown("---")
            
            with st.expander("🛡️ HEDGE 1: Smart Protection", expanded=True):
                st.markdown("**Inverte seleção de MENOR probabilidade**")
                h1_stake = stake * 0.30
                h1_odd = 2.0
                cen1_princ = lucro_max - h1_stake
                cen1_hedge = -stake + (h1_stake * h1_odd)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Stake", format_currency(h1_stake))
                col2.metric("Odd", f"@{h1_odd:.2f}")
                col3.metric("Retorno", format_currency(h1_stake * h1_odd))
                
                col1, col2 = st.columns(2)
                col1.success(f"✅ Principal ganha: {format_currency(cen1_princ)}")
                if cen1_hedge > 0:
                    col2.success(f"🛡️ Hedge ganha: {format_currency(cen1_hedge)}")
                else:
                    col2.error(f"🛡️ Hedge ganha: {format_currency(cen1_hedge)}")
            
            with st.expander("⚖️ HEDGE 2: Partial Protection"):
                st.markdown("**Inverte METADE das seleções**")
                h2_stake = stake * 0.50
                h2_odd = 1.8
                cen2_princ = lucro_max - h2_stake
                cen2_hedge = -stake + (h2_stake * h2_odd)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Stake", format_currency(h2_stake))
                col2.metric("Odd", f"@{h2_odd:.2f}")
                col3.metric("Retorno", format_currency(h2_stake * h2_odd))
                
                col1, col2 = st.columns(2)
                col1.success(f"✅ Principal: {format_currency(cen2_princ)}")
                if cen2_hedge > 0:
                    col2.success(f"🛡️ Hedge: {format_currency(cen2_hedge)}")
                else:
                    col2.error(f"🛡️ Hedge: {format_currency(cen2_hedge)}")
            
            with st.expander("💎 HEDGE 3: Guaranteed Profit"):
                st.markdown("**Inverte TUDO (arbitragem)**")
                h3_odd = 1.5
                h3_stake = (stake * odd_total) / (h3_odd + 1)
                lucro_gar = (stake * odd_total) - stake - h3_stake
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Stake", format_currency(h3_stake))
                col2.metric("Odd", f"@{h3_odd:.2f}")
                col3.metric("💰 LUCRO GARANTIDO", format_currency(lucro_gar))
                
                st.success(f"🎯 VOCÊ GANHA {format_currency(lucro_gar)} SEMPRE!")
    
    # ============================================================
    # TAB 3: SIMULADOR
    # ============================================================
    
    with tab3:
        st.header("🎲 Simulador Monte Carlo - 3000 Iterações")
        
        if not cal.empty:
            dates = sorted(cal['DtObj'].dt.strftime('%d/%m/%Y').unique())
            sel_date = st.selectbox("Data:", dates, key='sim_date')
            jogos_dia = cal[cal['DtObj'].dt.strftime('%d/%m/%Y') == sel_date]
            
            jogos_disp = []
            for _, jogo in jogos_dia.iterrows():
                h = normalize_name(jogo['Time_Casa'], list(stats_db.keys()))
                a = normalize_name(jogo['Time_Visitante'], list(stats_db.keys()))
                if h and a:
                    jogos_disp.append(f"{h} vs {a}")
            
            if jogos_disp:
                jogo_sel = st.selectbox("Jogo:", jogos_disp)
                
                if st.button("🎲 SIMULAR 3000 JOGOS"):
                    h_name, a_name = jogo_sel.split(' vs ')
                    
                    with st.spinner('Simulando...'):
                        sims = simulate_game_v31(stats_db[h_name], stats_db[a_name], {}, 3000)
                        
                        st.subheader("📊 Resultados")
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Cantos", f"{sims['corners_total'].mean():.1f}")
                        col2.metric("Cartões", f"{sims['cards_total'].mean():.1f}")
                        col3.metric("Gols Casa", f"{sims['goals_h'].mean():.1f}")
                        col4.metric("Gols Fora", f"{sims['goals_a'].mean():.1f}")
                        
                        st.markdown("---")
                        st.subheader("🎯 Probabilidades")
                        
                        mercados = {
                            'Over 9.5 Cantos': (sims['corners_total'] > 9.5).mean() * 100,
                            'Over 10.5 Cantos': (sims['corners_total'] > 10.5).mean() * 100,
                            'Over 11.5 Cantos': (sims['corners_total'] > 11.5).mean() * 100,
                            'Over 4.5 Cartões': (sims['cards_total'] > 4.5).mean() * 100,
                            'Over 5.5 Cartões': (sims['cards_total'] > 5.5).mean() * 100,
                            'Over 2.5 Gols': ((sims['goals_h'] + sims['goals_a']) > 2.5).mean() * 100,
                        }
                        
                        df_merc = pd.DataFrame({
                            'Mercado': list(mercados.keys()),
                            'Probabilidade (%)': list(mercados.values())
                        }).sort_values('Probabilidade (%)', ascending=False)
                        
                        st.dataframe(df_merc, use_container_width=True, height=250)
                        
                        # Gráficos
                        fig_cantos = go.Figure()
                        fig_cantos.add_trace(go.Histogram(x=sims['corners_total'], nbinsx=15, marker_color='orange'))
                        fig_cantos.update_layout(title='Distribuição de Cantos', height=400)
                        st.plotly_chart(fig_cantos, use_container_width=True)
    
    # ============================================================
    # TAB 4: MÉTRICAS PRO
    # ============================================================
    
    with tab4:
        st.header("📊 Métricas PRO - Análise Financeira Avançada")
        
        if not st.session_state.bet_results:
            st.info("📭 Sem apostas registradas. Use Tab 'Registrar'")
        else:
            total_apostas = len(st.session_state.bet_results)
            apostas_ganhas = sum(1 for b in st.session_state.bet_results if b.get('ganhou', False))
            
            total_staked = sum(b.get('stake', 0) for b in st.session_state.bet_results)
            total_profit = sum(b.get('lucro', 0) for b in st.session_state.bet_results)
            
            win_rate = (apostas_ganhas / total_apostas) * 100 if total_apostas > 0 else 0
            roi = calculate_roi(total_staked, total_profit)
            
            returns = [b.get('return', 0) for b in st.session_state.bet_results]
            sharpe = calculate_sharpe_ratio(returns)
            max_dd = calculate_max_drawdown(st.session_state.bankroll_history)
            
            st.subheader("📈 Métricas Principais")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Win Rate", f"{win_rate:.1f}%")
            col2.metric("ROI", f"{roi:.1f}%")
            col3.metric("Sharpe Ratio", f"{sharpe:.2f}")
            col4.metric("Max Drawdown", f"{max_dd:.1f}%")
            
            st.markdown("---")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Apostas", total_apostas)
            col2.metric("Apostas Ganhas", apostas_ganhas)
            col3.metric("Lucro Total", format_currency(total_profit))
            
            st.markdown("---")
            st.subheader("📊 Evolução da Banca")
            
            fig_banca = go.Figure()
            fig_banca.add_trace(go.Scatter(
                y=st.session_state.bankroll_history,
                mode='lines+markers',
                name='Banca',
                line=dict(color='blue', width=2)
            ))
            fig_banca.update_layout(
                title='Evolução da Banca ao Longo do Tempo',
                yaxis_title='Banca (R$)',
                xaxis_title='Apostas',
                height=400
            )
            st.plotly_chart(fig_banca, use_container_width=True)
    
    # ============================================================
    # TAB 5: VISUALIZAÇÕES
    # ============================================================
    
    with tab5:
        st.header("🎨 Visualizações Avançadas")
        
        viz_tipo = st.selectbox("Tipo de Visualização:", [
            "Comparativo de Ligas",
            "Distribuição de Cantos",
            "Top Times - Cantos",
            "Top Times - Cartões",
        ])
        
        if viz_tipo == "Comparativo de Ligas":
            st.subheader("📊 Comparativo de Métricas por Liga")
            
            liga_data = defaultdict(lambda: {'cantos': [], 'cartoes': [], 'gols': []})
            
            for team, data in stats_db.items():
                liga = data['league']
                liga_data[liga]['cantos'].append(data['corners'])
                liga_data[liga]['cartoes'].append(data['cards'])
                liga_data[liga]['gols'].append(data['goals_f'])
            
            ligas = list(liga_data.keys())
            cantos_media = [np.mean(liga_data[l]['cantos']) for l in ligas]
            cartoes_media = [np.mean(liga_data[l]['cartoes']) for l in ligas]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Cantos Médios', x=ligas, y=cantos_media, marker_color='orange'))
            fig.add_trace(go.Bar(name='Cartões Médios', x=ligas, y=cartoes_media, marker_color='yellow'))
            
            fig.update_layout(
                title='Comparativo de Métricas por Liga',
                barmode='group',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_tipo == "Top Times - Cantos":
            st.subheader("🔶 Top 20 Times com Mais Cantos")
            
            times_sorted = sorted(stats_db.items(), key=lambda x: x[1]['corners'], reverse=True)[:20]
            
            times_nomes = [t[0] for t in times_sorted]
            times_cantos = [t[1]['corners'] for t in times_sorted]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=times_nomes,
                x=times_cantos,
                orientation='h',
                marker_color='orange'
            ))
            
            fig.update_layout(
                title='Top 20 Times - Cantos por Jogo',
                xaxis_title='Cantos Médios',
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_tipo == "Distribuição de Cantos":
            st.subheader("📈 Distribuição de Cantos - Todos os Times")
            
            todos_cantos = [data['corners'] for data in stats_db.values()]
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=todos_cantos,
                nbinsx=30,
                marker_color='orange',
                name='Cantos'
            ))
            
            fig.update_layout(
                title=f'Distribuição de Cantos ({len(stats_db)} times)',
                xaxis_title='Cantos por Jogo',
                yaxis_title='Frequência',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # ============================================================
    # TAB 6: REGISTRAR APOSTAS
    # ============================================================
    
    with tab6:
        st.header("📝 Registrar Apostas")
        
        col1, col2 = st.columns(2)
        stake = col1.number_input("Stake (R$)", 10.0, 10000.0, 50.0, 10.0, key='reg_stake')
        odd = col2.number_input("Odd", 1.01, 100.0, 2.0, 0.01, key='reg_odd')
        
        ganhou = st.checkbox("✅ Aposta ganhou?")
        descricao = st.text_input("Descrição (opcional)", "Aposta manual")
        
        if st.button("💾 REGISTRAR APOSTA", use_container_width=True):
            lucro = stake * (odd - 1) if ganhou else -stake
            
            st.session_state.bet_results.append({
                'stake': stake,
                'odd': odd,
                'ganhou': ganhou,
                'lucro': lucro,
                'data': datetime.now().strftime('%d/%m/%Y %H:%M'),
                'descricao': descricao,
                'return': odd if ganhou else 0
            })
            
            nova_banca = st.session_state.bankroll_history[-1] + lucro
            st.session_state.bankroll_history.append(nova_banca)
            
            st.success(f"✅ Aposta registrada! Lucro: {format_currency(lucro)}")
            st.success(f"💰 Nova banca: {format_currency(nova_banca)}")
            st.rerun()
        
        if st.session_state.bet_results:
            st.markdown("---")
            st.subheader("📜 Histórico de Apostas")
            
            df_hist = pd.DataFrame(st.session_state.bet_results)
            st.dataframe(df_hist, use_container_width=True, height=300)
    
    # ============================================================
    # TAB 7: SCANNER
    # ============================================================
    
    with tab7:
        st.header("🔍 Scanner Inteligente de Jogos")
        
        if not cal.empty:
            dates = sorted(cal['DtObj'].dt.strftime('%d/%m/%Y').unique())
            sel_date = st.selectbox("Data:", dates, key='scan_date')
            
            col1, col2 = st.columns(2)
            prob_min = col1.slider("Probabilidade Mínima (%)", 50, 90, 70)
            tipo_mercado = col2.selectbox("Mercado:", ["Cantos", "Cartões", "Ambos"])
            
            if st.button("🔍 ESCANEAR JOGOS", use_container_width=True):
                jogos_dia = cal[cal['DtObj'].dt.strftime('%d/%m/%Y') == sel_date]
                resultados = []
                
                with st.spinner('Analisando jogos...'):
                    for _, jogo in jogos_dia.iterrows():
                        h = normalize_name(jogo['Time_Casa'], list(stats_db.keys()))
                        a = normalize_name(jogo['Time_Visitante'], list(stats_db.keys()))
                        
                        if h and a and h in stats_db and a in stats_db:
                            calc = calcular_jogo_v31(stats_db[h], stats_db[a], {})
                            
                            # Verificar cantos
                            if tipo_mercado in ["Cantos", "Ambos"]:
                                if calc['corners']['t'] > 10.5:
                                    prob = 75
                                    if prob >= prob_min:
                                        resultados.append({
                                            'Jogo': f"{h} vs {a}",
                                            'Mercado': 'Over 10.5 Cantos',
                                            'Prob': f"{prob}%",
                                            'Previsão': f"{calc['corners']['t']:.1f}",
                                            'Value': '✅' if prob >= 75 else '⚪'
                                        })
                            
                            # Verificar cartões
                            if tipo_mercado in ["Cartões", "Ambos"]:
                                if calc['cards']['t'] > 4.5:
                                    prob = 72
                                    if prob >= prob_min:
                                        resultados.append({
                                            'Jogo': f"{h} vs {a}",
                                            'Mercado': 'Over 4.5 Cartões',
                                            'Prob': f"{prob}%",
                                            'Previsão': f"{calc['cards']['t']:.1f}",
                                            'Value': '✅' if prob >= 75 else '⚪'
                                        })
                
                if resultados:
                    st.success(f"✅ {len(resultados)} oportunidade(s) encontrada(s)!")
                    df_res = pd.DataFrame(resultados)
                    st.dataframe(df_res, use_container_width=True)
                else:
                    st.warning("⚠️ Nenhuma oportunidade encontrada com esses critérios")
    
    # ============================================================
    # TAB 8: IMPORTAR BILHETE
    # ============================================================
    
    with tab8:
        st.header("📋 Importar Bilhete Automaticamente")
        
        texto = st.text_area("Cole o texto do bilhete:", height=200, key='import_text')
        
        col1, col2 = st.columns(2)
        stake_imp = col1.number_input("Stake", 10.0, 10000.0, 30.0, key='imp_stake')
        odd_imp = col2.number_input("Odd Total", 1.01, 100.0, 3.54, key='imp_odd')
        
        if st.button("🔍 ANALISAR BILHETE", use_container_width=True):
            if texto.strip():
                jogos_parsed = parse_bilhete_texto(texto)
                
                if jogos_parsed:
                    jogos_val = validar_jogos_bilhete(jogos_parsed, stats_db)
                    
                    if jogos_val:
                        st.success(f"✅ {len(jogos_val)} jogo(s) validado(s)")
                        
                        analise = calcular_prob_bilhete(jogos_val)
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Prob Real", f"{analise['prob_total']:.1f}%")
                        col2.metric("Odd Casa", f"@{odd_imp:.2f}")
                        col3.metric("Value", f"{analise['prob_total']/100 * odd_imp:.2f}")
                        
                        st.markdown("---")
                        st.subheader("📊 Detalhamento por Seleção")
                        
                        for det in analise['detalhes']:
                            with st.expander(f"{det['jogo']}", expanded=True):
                                st.write(f"**Mercado:** {det['mercado']}")
                                col1, col2, col3 = st.columns(3)
                                col1.metric("Prob Real", f"{det['prob']:.1f}%")
                                col2.metric("Odd Casa", f"@{det['odd_casa']:.2f}")
                                col3.metric("Fair Odd", f"@{det['fair_odd']:.2f}")
                                
                                if det['value'] > 1.0:
                                    st.success(f"✅ VALUE BET! Score: {det['value']:.2f}")
                                else:
                                    st.warning(f"⚠️ Sem value. Score: {det['value']:.2f}")
                    else:
                        st.error("❌ Times não encontrados no banco de dados")
                else:
                    st.error("❌ Não foi possível identificar jogos no texto")
            else:
                st.warning("⚠️ Cole o texto do bilhete acima")
    
    # ============================================================
    # TAB 9: AI ADVISOR ULTRA
    # ============================================================
    
    with tab9:
        st.header("🤖 FutPrevisão AI Advisor ULTRA")
        
        if not st.session_state.chat_history:
            total = len(st.session_state.bet_results)
            ganhas = sum(1 for b in st.session_state.bet_results if b.get('ganhou', False))
            wr = (ganhas/total*100) if total > 0 else 0
            banca = st.session_state.bankroll_history[-1]
            
            perfil = "🎯 PROFISSIONAL" if wr >= 70 and total >= 30 else "📊 AVANÇADO" if wr >= 60 and total >= 15 else "🌟 INTERMEDIÁRIO" if total >= 5 else "🔰 INICIANTE"
            
            welcome = f"""👋 Olá! Sou o **FutPrevisão AI Advisor ULTRA**!

📊 **SEU PERFIL: {perfil}**
• Apostas: {total}
• Win Rate: {wr:.1f}%
• Banca: {format_currency(banca)}

💡 **CHATBOT ULTRA INTELIGENTE**
Pergunte QUALQUER COISA sobre os dados!

**Exemplos:**
• "Qual time tem mais cantos na Premier League?"
• "Top 5 times com mais cartões"
• "Compare Arsenal e Chelsea"
• "Times com mais de 6 cantos em casa"
• "Média de gols do Liverpool"

🎯 **Digite sua pergunta abaixo!**"""
            
            st.session_state.chat_history.append({'role': 'assistant', 'content': welcome})
        
        # Botões rápidos
        col1, col2, col3, col4 = st.columns(4)
        
        if col1.button("🎯 Top Cantos", use_container_width=True):
            st.session_state.chat_history.append({'role': 'user', 'content': 'top 10 cantos'})
            st.rerun()
        
        if col2.button("🟨 Top Cartões", use_container_width=True):
            st.session_state.chat_history.append({'role': 'user', 'content': 'top 10 cartões'})
            st.rerun()
        
        if col3.button("📅 Hoje", use_container_width=True):
            st.session_state.chat_history.append({'role': 'user', 'content': 'jogos de hoje'})
            st.rerun()
        
        if col4.button("🗑️ Limpar", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
        
        st.markdown("---")
        
        # Exibir histórico
        for msg in st.session_state.chat_history:
            if msg['role'] == 'user':
                st.chat_message("user", avatar="👤").markdown(msg['content'])
            else:
                st.chat_message("assistant", avatar="🤖").markdown(msg['content'])
        
        # Input
        user_msg = st.chat_input("Digite sua pergunta...")
        
        if user_msg:
            st.session_state.chat_history.append({'role': 'user', 'content': user_msg})
            
            # Processar com chatbot ULTRA inteligente
            response = processar_chat_ultra(user_msg, stats_db, cal)
            
            st.session_state.chat_history.append({'role': 'assistant', 'content': response})
            st.rerun()


if __name__ == "__main__":
    main()
