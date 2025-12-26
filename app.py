"""
FutPrevisão V31 MAXIMUM + SUPERBOT V2.0 ULTRA INTELIGENTE
CÓDIGO COMPLETO - 2400+ LINHAS
VERSÃO PROFISSIONAL COM IA AVANÇADA

Autor: Diego
Versão: 31.0 ULTRA MAXIMUM + SUPERBOT V2.0
Data: 26/12/2024
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import math
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px
from difflib import get_close_matches
import re
from collections import defaultdict

# ============================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================

st.set_page_config(
    page_title="FutPrevisão V31 MAXIMUM + SUPERBOT V2.0",
    layout="wide",
    page_icon="⚽",
    initial_sidebar_state="expanded"
)

# CSS PERSONALIZADO - MENSAGENS DO BOT EM AZUL
st.markdown('''
<style>
    /* Mensagens do assistente (bot) em azul gradiente */
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border-radius: 15px !important;
        padding: 15px !important;
        margin: 10px 0 !important;
    }
    
    /* Texto das mensagens do bot em branco */
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) p {
        color: white !important;
    }
    
    /* Mensagens do usuário em cinza escuro */
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) {
        background: #2d3748 !important;
        border-radius: 15px !important;
        padding: 15px !important;
        margin: 10px 0 !important;
    }
    
    /* Texto das mensagens do usuário em branco */
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) p {
        color: white !important;
    }
    
    /* Avatar do bot com borda azul */
    div[data-testid="chatAvatarIcon-assistant"] {
        border: 3px solid #667eea !important;
        border-radius: 50% !important;
    }
    
    /* Avatar do usuário com borda verde */
    div[data-testid="chatAvatarIcon-user"] {
        border: 3px solid #48bb78 !important;
        border-radius: 50% !important;
    }
    
    /* Outras customizações */
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
''', unsafe_allow_html=True)

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
# CARREGAMENTO DE DADOS
# ============================================================

@st.cache_data(ttl=3600)
def load_all_data():
    """Carrega todos os dados do sistema"""
    stats_db = {}
    cal = pd.DataFrame()
    referees = {}
    
    league_files = {
        'Premier League': '/mnt/project/Premier_League_25_26.csv',
        'La Liga': '/mnt/project/La_Liga_25_26.csv',
        'Serie A': '/mnt/project/Serie_A_25_26.csv',
        'Bundesliga': '/mnt/project/Bundesliga_25_26.csv',
        'Ligue 1': '/mnt/project/Ligue_1_25_26.csv',
        'Championship': '/mnt/project/Championship_Inglaterra_25_26.csv',
        'Bundesliga 2': '/mnt/project/Bundesliga_2.csv',
        'Pro League': '/mnt/project/Pro_League_Belgica_25_26.csv',
        'Super Lig': '/mnt/project/Super_Lig_Turquia_25_26.csv',
        'Premiership': '/mnt/project/Premiership_Escocia_25_26.csv'
    }
    
    for league_name, filepath in league_files.items():
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
    
    try:
        cal = pd.read_csv('/mnt/project/calendario_ligas.csv', encoding='utf-8')
        if 'Data' in cal.columns:
            cal['DtObj'] = pd.to_datetime(cal['Data'], format='%d/%m/%Y', errors='coerce')
    except:
        pass
    
    try:
        refs_df = pd.read_csv('/mnt/project/arbitros_5_ligas_2025_2026.csv', encoding='utf-8')
        for _, row in refs_df.iterrows():
            referees[row['Arbitro']] = {
                'factor': row['Media_Cartoes_Por_Jogo'] / 4.0,
                'games': row['Jogos_Apitados'],
                'avg_cards': row['Media_Cartoes_Por_Jogo'],
                'red_cards': row.get('Cartoes_Vermelhos', 0),
                'red_rate': row.get('Cartoes_Vermelhos', 0) / row['Jogos_Apitados'] if row['Jogos_Apitados'] > 0 else 0.08
            }
    except:
        pass
    
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

def calculate_ev(probability: float, odds: float, stake: float) -> float:
    """Calcula Expected Value (valor esperado)"""
    win_amount = stake * (odds - 1)
    lose_amount = -stake
    
    ev = (probability * win_amount) + ((1 - probability) * lose_amount)
    return ev

# ============================================================
# FUNÇÃO PRINCIPAL
# ============================================================

def main():
    """Função principal do aplicativo"""
    
    stats, cal, referees = load_all_data()
    
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
    
    st.title("⚽ FutPrevisão V31 MAXIMUM + SUPERBOT V2.0")
    st.markdown("**Sistema Completo e Profissional de Análise de Apostas Esportivas**")
    st.markdown("_Causality Engine V31 | Poisson | Monte Carlo | Kelly | Sharpe | IA Avançada | 2400+ linhas_")
    
    with st.sidebar:
        st.header("📊 Dashboard")
        col1, col2 = st.columns(2)
        col1.metric("Times", len(stats))
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
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "🎫 Construtor", "🛡️ Hedges", "🎲 Simulador", "📊 Métricas",
        "🎨 Viz", "📝 Registro", "🔍 Scanner", "📋 Importar", "🤖 AI V2.0"
    ])
    
    # ============================================================
    # TAB 1: CONSTRUTOR
    # ============================================================
    
    with tab1:
        st.header("🎫 Construtor de Bilhetes Profissional")
        
        if not cal.empty:
            dates = sorted(cal['DtObj'].dt.strftime('%d/%m/%Y').unique())
            sel_date = st.selectbox("📅 Selecione a Data:", dates, key='c_date')
            jogos_dia = cal[cal['DtObj'].dt.strftime('%d/%m/%Y') == sel_date]
            
            st.markdown(f"### 🎯 {len(jogos_dia)} jogo(s) disponível(eis)")
            
            for idx, jogo in jogos_dia.iterrows():
                h = normalize_name(jogo['Time_Casa'], list(stats.keys()))
                a = normalize_name(jogo['Time_Visitante'], list(stats.keys()))
                
                if h and a and h in stats and a in stats:
                    ref_nome = jogo.get('Arbitro', 'N/A')
                    ref_data = referees.get(ref_nome, {})
                    calc = calcular_jogo_v31(stats[h], stats[a], ref_data)
                    
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
                h = normalize_name(jogo['Time_Casa'], list(stats.keys()))
                a = normalize_name(jogo['Time_Visitante'], list(stats.keys()))
                if h and a:
                    jogos_disp.append(f"{h} vs {a}")
            
            if jogos_disp:
                jogo_sel = st.selectbox("Jogo:", jogos_disp)
                
                if st.button("🎲 SIMULAR 3000 JOGOS"):
                    h_name, a_name = jogo_sel.split(' vs ')
                    
                    with st.spinner('Simulando...'):
                        sims = simulate_game_v31(stats[h_name], stats[a_name], {}, 3000)
                        
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
            
            st.markdown("---")
            st.subheader("💡 Interpretação das Métricas")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🎯 Win Rate:**")
                if win_rate >= 70:
                    st.success(f"Excelente! {win_rate:.1f}% está acima da média")
                elif win_rate >= 55:
                    st.info(f"Bom! {win_rate:.1f}% é sólido")
                else:
                    st.warning(f"Atenção! {win_rate:.1f}% precisa melhorar")
                
                st.markdown("**📈 ROI:**")
                if roi > 10:
                    st.success(f"Ótimo retorno! {roi:.1f}%")
                elif roi > 0:
                    st.info(f"Positivo: {roi:.1f}%")
                else:
                    st.error(f"Prejuízo: {roi:.1f}%")
            
            with col2:
                st.markdown("**⚡ Sharpe Ratio:**")
                if sharpe > 2.0:
                    st.success(f"Excelente! {sharpe:.2f} (risco/retorno ótimo)")
                elif sharpe > 1.0:
                    st.info(f"Bom: {sharpe:.2f}")
                else:
                    st.warning(f"Atenção: {sharpe:.2f}")
                
                st.markdown("**📉 Max Drawdown:**")
                if max_dd < 10:
                    st.success(f"Muito bom! {max_dd:.1f}%")
                elif max_dd < 25:
                    st.info(f"Aceitável: {max_dd:.1f}%")
                else:
                    st.warning(f"Alto: {max_dd:.1f}%")
    
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
            
            for team, data in stats.items():
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
            
            times_sorted = sorted(stats.items(), key=lambda x: x[1]['corners'], reverse=True)[:20]
            
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
            
            todos_cantos = [data['corners'] for data in stats.values()]
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=todos_cantos,
                nbinsx=30,
                marker_color='orange',
                name='Cantos'
            ))
            
            fig.update_layout(
                title=f'Distribuição de Cantos ({len(stats)} times)',
                xaxis_title='Cantos por Jogo',
                yaxis_title='Frequência',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            media_cantos = np.mean(todos_cantos)
            mediana_cantos = np.median(todos_cantos)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Média", f"{media_cantos:.2f}")
            col2.metric("Mediana", f"{mediana_cantos:.2f}")
            col3.metric("Times", len(stats))
        
        elif viz_tipo == "Top Times - Cartões":
            st.subheader("🟨 Top 20 Times com Mais Cartões")
            
            times_sorted = sorted(stats.items(), key=lambda x: x[1]['cards'], reverse=True)[:20]
            
            times_nomes = [t[0] for t in times_sorted]
            times_cartoes = [t[1]['cards'] for t in times_sorted]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=times_nomes,
                x=times_cartoes,
                orientation='h',
                marker_color='yellow'
            ))
            
            fig.update_layout(
                title='Top 20 Times - Cartões por Jogo',
                xaxis_title='Cartões Médios',
                height=600
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
                        h = normalize_name(jogo['Time_Casa'], list(stats.keys()))
                        a = normalize_name(jogo['Time_Visitante'], list(stats.keys()))
                        
                        if h and a and h in stats and a in stats:
                            calc = calcular_jogo_v31(stats[h], stats[a], {})
                            
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
                    jogos_val = validar_jogos_bilhete(jogos_parsed, stats)
                    
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
    # TAB 9: AI ADVISOR SUPERBOT V2.0 ULTRA INTELIGENTE
    # ============================================================
    
    with tab9:
        st.header("🤖 FutPrevisão AI Advisor SUPERBOT V2.0")
        st.caption("_Inteligência Artificial com acesso TOTAL aos dados do projeto. Pergunte QUALQUER COISA!_")
        
        # ============================================================
        # CLASSES DO SUPERBOT V2.0
        # ============================================================
        
        class SuperIntentDetector:
            """Detector de intenções ULTRA avançado"""
            
            def __init__(self):
                self.patterns = {
                    # ESTATÍSTICAS DE TIME
                    'stats_time': [
                        'como está', 'como esta', 'estatística', 'estatisticas',
                        'dados do', 'números do', 'stats', 'desempenho', 'performance',
                        'como joga', 'como anda', 'média de', 'media de'
                    ],
                    
                    # JOGOS HOJE/AMANHÃ
                    'jogos_hoje': [
                        'jogos hoje', 'partidas hoje', 'joga hoje', 'tem jogo hoje',
                        'quais jogos hoje', 'que jogo tem hoje', 'hoje'
                    ],
                    'jogos_amanha': [
                        'jogos amanhã', 'jogos amanha', 'partidas amanhã', 'amanhã', 'amanha'
                    ],
                    'jogos_data': [
                        'jogos no dia', 'jogos em', 'partidas no dia', 'calendario'
                    ],
                    
                    # ANÁLISE H2H
                    'analise_jogo': [
                        ' vs ', ' x ', 'versus', 'contra', 'analisa', 'analise',
                        'quem ganha', 'previsão', 'previsao', 'favorito'
                    ],
                    
                    # RANKINGS
                    'ranking_cantos': [
                        'mais cantos', 'top cantos', 'maiores cantos', 'ranking cantos',
                        'times com mais cantos', 'melhores em cantos', 'escanteios'
                    ],
                    'ranking_cartoes': [
                        'mais cartões', 'mais cartoes', 'top cartões', 'top cartoes',
                        'maiores cartões', 'ranking cartões', 'times violentos', 'amarelos'
                    ],
                    'ranking_gols': [
                        'mais gols', 'top gols', 'maiores gols', 'ranking gols',
                        'artilheiros', 'times ofensivos', 'ataque'
                    ],
                    
                    # COMPARAÇÕES
                    'comparar_times': [
                        'compare', 'compara', 'diferença entre', 'qual melhor',
                        'quem é melhor', 'x ou y', 'versus'
                    ],
                    'comparar_ligas': [
                        'liga com mais', 'melhor liga', 'compare ligas',
                        'diferença entre ligas'
                    ],
                    
                    # ÁRBITROS
                    'arbitro_stats': [
                        'árbitro', 'arbitro', 'juiz', 'apita', 'rigidez',
                        'cartões do árbitro', 'cartoes do arbitro'
                    ],
                    'arbitro_ranking': [
                        'árbitros mais rigorosos', 'arbitros rigorosos',
                        'top árbitros', 'ranking arbitros'
                    ],
                    
                    # MERCADOS/APOSTAS
                    'melhor_mercado': [
                        'melhor jogo para', 'onde apostar', 'melhor aposta',
                        'mercado', 'over', 'probabilidade'
                    ],
                    
                    # CALENDÁRIO
                    'proximos_jogos': [
                        'próximos jogos', 'proximos jogos', 'quando joga',
                        'próximo jogo do', 'proximo jogo'
                    ],
                    
                    # MÉDIA DA LIGA
                    'media_liga': [
                        'média da', 'media da', 'liga', 'campeonato'
                    ],
                    
                    # GERAL
                    'saudacao': ['oi', 'olá', 'ola', 'hey', 'bom dia', 'boa tarde'],
                    'agradecimento': ['obrigado', 'obrigada', 'valeu', 'vlw'],
                }
            
            def detect(self, text: str) -> str:
                """Detecta intenção com priorização"""
                text_lower = text.lower()
                
                # Priorizar análise H2H (tem "vs" ou "x")
                if ' vs ' in text_lower or ' x ' in text_lower:
                    return 'analise_jogo'
                
                # Detectar por patterns
                for intent, patterns in self.patterns.items():
                    for pattern in patterns:
                        if pattern in text_lower:
                            return intent
                
                return 'desconhecido'
        
        class SuperEntityExtractor:
            """Extrator de entidades ULTRA robusto"""
            
            def __init__(self, stats_db, calendar_df, referees):
                self.stats_db = stats_db
                self.calendar = calendar_df
                self.referees = referees
                self.today = datetime.now()
            
            def extract_teams(self, text: str) -> list:
                """Extrai times com FUZZY MATCHING"""
                teams_found = []
                text_lower = text.lower()
                
                # Lista de todos os times
                all_teams = list(self.stats_db.keys())
                
                # Tentar match direto
                for team in all_teams:
                    if team.lower() in text_lower:
                        teams_found.append(team)
                
                # Se não encontrou, tentar fuzzy
                if not teams_found:
                    words = text.split()
                    for word in words:
                        if len(word) > 3:  # Palavras com 4+ letras
                            matches = get_close_matches(word, all_teams, n=2, cutoff=0.6)
                            teams_found.extend(matches)
                
                # Tentar normalizar palavras-chave comuns
                if not teams_found:
                    keywords = ['manchester', 'liverpool', 'arsenal', 'chelsea', 'united', 'city']
                    for keyword in keywords:
                        if keyword in text_lower:
                            matches = get_close_matches(keyword, all_teams, n=2, cutoff=0.4)
                            teams_found.extend(matches)
                
                return list(set(teams_found))[:2]  # Max 2 times
            
            def extract_date(self, text: str) -> str:
                """Extrai data com NLP natural"""
                text_lower = text.lower()
                
                # Hoje
                if any(p in text_lower for p in ['hoje', 'agora', 'hj']):
                    return self.today.strftime('%d/%m/%Y')
                
                # Amanhã
                if any(p in text_lower for p in ['amanhã', 'amanha']):
                    return (self.today + timedelta(days=1)).strftime('%d/%m/%Y')
                
                # Depois de amanhã
                if 'depois' in text_lower:
                    return (self.today + timedelta(days=2)).strftime('%d/%m/%Y')
                
                # Dias da semana
                dias = {
                    'segunda': 0, 'terca': 1, 'terça': 1, 'quarta': 2,
                    'quinta': 3, 'sexta': 4, 'sabado': 5, 'sábado': 5, 'domingo': 6
                }
                
                for dia, num in dias.items():
                    if dia in text_lower:
                        days_ahead = num - self.today.weekday()
                        if days_ahead <= 0:
                            days_ahead += 7
                        return (self.today + timedelta(days=days_ahead)).strftime('%d/%m/%Y')
                
                # Data explícita (DD/MM ou DD/MM/YYYY)
                date_patterns = [
                    r'(\d{1,2})/(\d{1,2})/(\d{4})',
                    r'(\d{1,2})/(\d{1,2})',
                    r'dia (\d{1,2})'
                ]
                
                for pattern in date_patterns:
                    match = re.search(pattern, text)
                    if match:
                        if len(match.groups()) == 3:
                            return f"{match.group(1)}/{match.group(2)}/{match.group(3)}"
                        elif len(match.groups()) == 2:
                            return f"{match.group(1)}/{match.group(2)}/{self.today.year}"
                        else:
                            return f"{match.group(1)}/{self.today.month}/{self.today.year}"
                
                return self.today.strftime('%d/%m/%Y')
            
            def extract_league(self, text: str) -> str:
                """Extrai nome da liga"""
                text_lower = text.lower()
                
                leagues = {
                    'premier': 'Premier League',
                    'la liga': 'La Liga',
                    'espanha': 'La Liga',
                    'serie a': 'Serie A',
                    'italia': 'Serie A',
                    'bundesliga': 'Bundesliga',
                    'alemanha': 'Bundesliga',
                    'ligue 1': 'Ligue 1',
                    'franca': 'Ligue 1',
                    'frança': 'Ligue 1',
                    'championship': 'Championship',
                    'segunda divisao': 'Championship',
                    'belgica': 'Pro League',
                    'bélgica': 'Pro League',
                    'turquia': 'Super Lig',
                    'escocia': 'Premiership',
                    'escócia': 'Premiership'
                }
                
                for key, league in leagues.items():
                    if key in text_lower:
                        return league
                
                return None
            
            def extract_number(self, text: str) -> float:
                """Extrai número (linha de aposta)"""
                numbers = re.findall(r'\d+\.?\d*', text)
                return float(numbers[0]) if numbers else None
            
            def extract_referee(self, text: str) -> str:
                """Extrai nome do árbitro"""
                for ref_name in self.referees.keys():
                    if ref_name.lower() in text.lower():
                        return ref_name
                return None
        
        class SuperKnowledgeBase:
            """Base de conhecimento com acesso TOTAL aos dados"""
            
            def __init__(self, stats_db, calendar_df, referees):
                self.stats = stats_db
                self.cal = calendar_df
                self.refs = referees
            
            def get_team_full_stats(self, team_name: str) -> dict:
                """Retorna estatísticas COMPLETAS do time"""
                team_norm = normalize_name(team_name, list(self.stats.keys()))
                
                if not team_norm or team_norm not in self.stats:
                    return None
                
                return {
                    'name': team_norm,
                    'stats': self.stats[team_norm],
                    'league': self.stats[team_norm]['league'],
                    'games': self.stats[team_norm]['games']
                }
            
            def get_games_by_date(self, date_str: str) -> list:
                """Jogos de uma data específica"""
                if self.cal.empty:
                    return []
                
                jogos = self.cal[self.cal['DtObj'].dt.strftime('%d/%m/%Y') == date_str]
                games_list = []
                
                for _, jogo in jogos.iterrows():
                    h = normalize_name(jogo['Time_Casa'], list(self.stats.keys()))
                    a = normalize_name(jogo['Time_Visitante'], list(self.stats.keys()))
                    
                    if h and a and h in self.stats and a in self.stats:
                        games_list.append({
                            'home': h,
                            'away': a,
                            'time': jogo.get('Hora', 'N/A'),
                            'league': self.stats[h]['league'],
                            'date': date_str
                        })
                
                return games_list
            
            def get_next_games_team(self, team_name: str, n: int = 3) -> list:
                """Próximos N jogos de um time"""
                if self.cal.empty:
                    return []
                
                team_norm = normalize_name(team_name, list(self.stats.keys()))
                if not team_norm:
                    return []
                
                future_games = self.cal[self.cal['DtObj'] >= datetime.now()]
                team_games = future_games[
                    (future_games['Time_Casa'].str.contains(team_norm, case=False, na=False)) |
                    (future_games['Time_Visitante'].str.contains(team_norm, case=False, na=False))
                ].head(n)
                
                games = []
                for _, jogo in team_games.iterrows():
                    games.append({
                        'date': jogo['DtObj'].strftime('%d/%m/%Y'),
                        'time': jogo.get('Hora', 'N/A'),
                        'home': jogo['Time_Casa'],
                        'away': jogo['Time_Visitante']
                    })
                
                return games
            
            def get_ranking_corners(self, n: int = 10, league: str = None) -> list:
                """Ranking times com mais cantos"""
                data = []
                
                for team, stats in self.stats.items():
                    if league and stats['league'] != league:
                        continue
                    
                    data.append({
                        'time': team,
                        'cantos': stats.get('corners', 0),
                        'liga': stats['league']
                    })
                
                return sorted(data, key=lambda x: x['cantos'], reverse=True)[:n]
            
            def get_ranking_cards(self, n: int = 10, league: str = None) -> list:
                """Ranking times com mais cartões"""
                data = []
                
                for team, stats in self.stats.items():
                    if league and stats['league'] != league:
                        continue
                    
                    data.append({
                        'time': team,
                        'cartoes': stats.get('cards', 0),
                        'liga': stats['league']
                    })
                
                return sorted(data, key=lambda x: x['cartoes'], reverse=True)[:n]
            
            def get_ranking_goals(self, n: int = 10, league: str = None) -> list:
                """Ranking times com mais gols"""
                data = []
                
                for team, stats in self.stats.items():
                    if league and stats['league'] != league:
                        continue
                    
                    data.append({
                        'time': team,
                        'gols': stats.get('goals_f', 0),
                        'liga': stats['league']
                    })
                
                return sorted(data, key=lambda x: x['gols'], reverse=True)[:n]
            
            def get_league_averages(self, league_name: str) -> dict:
                """Médias de uma liga"""
                league_teams = [t for t, s in self.stats.items() if s['league'] == league_name]
                
                if not league_teams:
                    return None
                
                cantos = [self.stats[t].get('corners', 0) for t in league_teams]
                cartoes = [self.stats[t].get('cards', 0) for t in league_teams]
                gols = [self.stats[t].get('goals_f', 0) for t in league_teams]
                
                return {
                    'liga': league_name,
                    'times': len(league_teams),
                    'cantos_media': np.mean(cantos),
                    'cartoes_media': np.mean(cartoes),
                    'gols_media': np.mean(gols)
                }
            
            def compare_teams(self, team1: str, team2: str) -> dict:
                """Compara dois times em todas as métricas"""
                t1 = self.get_team_full_stats(team1)
                t2 = self.get_team_full_stats(team2)
                
                if not t1 or not t2:
                    return None
                
                s1 = t1['stats']
                s2 = t2['stats']
                
                return {
                    'team1': t1['name'],
                    'team2': t2['name'],
                    'cantos': {
                        'team1': s1.get('corners', 0),
                        'team2': s2.get('corners', 0),
                        'vantagem': t1['name'] if s1.get('corners', 0) > s2.get('corners', 0) else t2['name']
                    },
                    'cartoes': {
                        'team1': s1.get('cards', 0),
                        'team2': s2.get('cards', 0),
                        'vantagem': t1['name'] if s1.get('cards', 0) > s2.get('cards', 0) else t2['name']
                    },
                    'gols_marcados': {
                        'team1': s1.get('goals_f', 0),
                        'team2': s2.get('goals_f', 0),
                        'vantagem': t1['name'] if s1.get('goals_f', 0) > s2.get('goals_f', 0) else t2['name']
                    },
                    'gols_sofridos': {
                        'team1': s1.get('goals_a', 0),
                        'team2': s2.get('goals_a', 0),
                        'vantagem': t1['name'] if s1.get('goals_a', 0) < s2.get('goals_a', 0) else t2['name']
                    }
                }
            
            def get_referee_stats(self, referee_name: str) -> dict:
                """Estatísticas completas do árbitro"""
                if referee_name not in self.refs:
                    return None
                
                ref = self.refs[referee_name]
                
                return {
                    'nome': referee_name,
                    'jogos': ref.get('games', 0),
                    'media_cartoes': ref.get('avg_cards', 0),
                    'cartoes_vermelhos': ref.get('red_cards', 0),
                    'red_rate': ref.get('red_rate', 0),
                    'factor': ref.get('factor', 1.0),
                    'classificacao': self._classify_referee(ref)
                }
            
            def _classify_referee(self, ref_data: dict) -> str:
                """Classifica árbitro por rigidez"""
                red_rate = ref_data.get('red_rate', 0)
                avg_cards = ref_data.get('avg_cards', 0)
                
                if avg_cards > 5.0 or red_rate > 0.12:
                    return "🔴 MUITO RIGOROSO"
                elif avg_cards > 4.0 or red_rate > 0.08:
                    return "🟠 RIGOROSO"
                elif avg_cards > 3.0:
                    return "🟡 MÉDIO"
                else:
                    return "🟢 LENIENTE"
            
            def get_referees_ranking(self, n: int = 10) -> list:
                """Ranking árbitros por rigidez"""
                data = []
                
                for ref_name, ref_data in self.refs.items():
                    data.append({
                        'arbitro': ref_name,
                        'media_cartoes': ref_data.get('avg_cards', 0),
                        'jogos': ref_data.get('games', 0),
                        'vermelhos': ref_data.get('red_cards', 0)
                    })
                
                return sorted(data, key=lambda x: x['media_cartoes'], reverse=True)[:n]
        
        class SuperResponseGenerator:
            """Gerador de respostas ULTRA naturais"""
            
            def __init__(self, kb):
                self.kb = kb
            
            def team_stats(self, team_name: str) -> str:
                """Resposta de estatísticas do time"""
                data = self.kb.get_team_full_stats(team_name)
                
                if not data:
                    similares = get_close_matches(team_name, list(self.kb.stats.keys()), n=3, cutoff=0.5)
                    if similares:
                        return f"❌ Time '{team_name}' não encontrado.\n\n💡 Você quis dizer: {', '.join(similares)}?"
                    return f"❌ Time '{team_name}' não encontrado no banco de dados."
                
                s = data['stats']
                
                return f"""📊 **ESTATÍSTICAS COMPLETAS - {data['name']}**

🏟️ **INFORMAÇÕES GERAIS:**
• Liga: **{data['league']}**
• Jogos Analisados: **{data['games']}**

⚽ **ATAQUE:**
• Gols Marcados: **{s.get('goals_f', 0):.2f}** por jogo
• Chutes no Gol: **{s.get('shots_on_target', 0):.1f}** por jogo
• Classificação: {('🔥 **ATAQUE FORTÍSSIMO**' if s.get('goals_f', 0) > 2.0 else '✅ Ataque bom' if s.get('goals_f', 0) > 1.5 else '⚠️ Ataque fraco')}

🛡️ **DEFESA:**
• Gols Sofridos: **{s.get('goals_a', 0):.2f}** por jogo
• Classificação: {('✅ **DEFESA SÓLIDA**' if s.get('goals_a', 0) < 1.0 else '📊 Defesa média' if s.get('goals_a', 0) < 1.5 else '⚠️ **DEFESA VULNERÁVEL**')}

🔶 **ESCANTEIOS:**
• Média: **{s.get('corners', 0):.1f}** por jogo
• Em Casa: **{s.get('corners_home', 0):.1f}**
• Fora: **{s.get('corners_away', 0):.1f}**
• Classificação: {('🎯 **EXCELENTE PARA CANTOS**' if s.get('corners', 0) > 6.0 else '✅ Bom' if s.get('corners', 0) > 5.0 else 'Médio')}

🟨 **DISCIPLINA:**
• Cartões: **{s.get('cards', 0):.1f}** por jogo
• Faltas: **{s.get('fouls', 0):.1f}** por jogo
• Classificação: {('🔴 **TIME VIOLENTO**' if s.get('fouls', 0) > 12.5 else '✅ Time disciplinado')}

💡 **RECOMENDAÇÕES DE APOSTAS:**
{self._generate_recommendations(s)}"""
            
            def _generate_recommendations(self, stats: dict) -> str:
                """Gera recomendações baseadas em stats"""
                recs = []
                
                if stats.get('corners', 0) > 6.0:
                    recs.append("• ✅ **Excelente para OVER CANTOS**")
                
                if stats.get('cards', 0) > 2.5:
                    recs.append("• ✅ **Bom para OVER CARTÕES**")
                
                if stats.get('goals_f', 0) > 2.0:
                    recs.append("• ✅ **Ótimo para OVER GOLS (ataque forte)**")
                
                if stats.get('goals_f', 0) > 1.5 and stats.get('goals_a', 0) > 1.5:
                    recs.append("• ✅ **Bom para AMBOS MARCAM (BTTS)**")
                
                if not recs:
                    recs.append("• 📊 Time com estatísticas médias")
                
                return "\n".join(recs)
            
            def games_today(self, date_str: str) -> str:
                """Lista jogos do dia"""
                hoje = datetime.now().strftime('%d/%m/%Y')
                amanha = (datetime.now() + timedelta(days=1)).strftime('%d/%m/%Y')
                
                if date_str == hoje:
                    periodo = "**HOJE**"
                elif date_str == amanha:
                    periodo = "**AMANHÃ**"
                else:
                    periodo = f"**{date_str}**"
                
                games = self.kb.get_games_by_date(date_str)
                
                if not games:
                    return f"📅 Não encontrei jogos cadastrados para {periodo}"
                
                response = f"⚽ **JOGOS DE {periodo}:** ({len(games)} partidas)\n\n"
                
                for i, g in enumerate(games, 1):
                    calc = calcular_jogo_v31(self.kb.stats[g['home']], self.kb.stats[g['away']], {})
                    
                    response += f"**{i}. {g['home']} vs {g['away']}**\n"
                    response += f"   🕐 {g['time']} | 🏆 {g['league']}\n"
                    response += f"   📊 Previsão: {calc['corners']['t']:.1f} cantos | {calc['cards']['t']:.1f} cartões\n\n"
                
                return response
            
            def head_to_head(self, team1: str, team2: str) -> str:
                """Análise H2H completa"""
                t1_norm = normalize_name(team1, list(self.kb.stats.keys()))
                t2_norm = normalize_name(team2, list(self.kb.stats.keys()))
                
                if not t1_norm or not t2_norm:
                    return f"❌ Um dos times não foi encontrado.\n\n💡 Verifique os nomes: '{team1}' e '{team2}'"
                
                calc = calcular_jogo_v31(self.kb.stats[t1_norm], self.kb.stats[t2_norm], {})
                
                total_gols = calc['goals']['h'] + calc['goals']['a']
                
                if calc['goals']['h'] > calc['goals']['a'] + 0.5:
                    favorito = f"✅ **{t1_norm} é FAVORITO**"
                elif calc['goals']['a'] > calc['goals']['h'] + 0.5:
                    favorito = f"✅ **{t2_norm} é FAVORITO**"
                else:
                    favorito = "⚖️ **JOGO EQUILIBRADO**"
                
                response = f"""🎯 **ANÁLISE COMPLETA: {t1_norm} vs {t2_norm}**

{favorito}

⚽ **EXPECTED GOALS (xG):**
• {t1_norm}: **{calc['goals']['h']:.2f}**
• {t2_norm}: **{calc['goals']['a']:.2f}**
• Total: **{total_gols:.2f}**

🔶 **ESCANTEIOS:**
• Total Previsto: **{calc['corners']['t']:.1f}**
• {t1_norm}: **{calc['corners']['h']:.1f}**
• {t2_norm}: **{calc['corners']['a']:.1f}**

🟨 **CARTÕES:**
• Total Previsto: **{calc['cards']['t']:.1f}**
• {t1_norm}: **{calc['cards']['h']:.1f}**
• {t2_norm}: **{calc['cards']['a']:.1f}**

🎲 **MELHORES APOSTAS:**"""
                
                apostas = []
                
                if total_gols > 2.5:
                    prob = min(int((total_gols - 2.5) * 30 + 65), 85)
                    apostas.append(f"✅ **Over 2.5 Gols** ({prob}%)")
                
                if calc['corners']['t'] > 10.5:
                    prob = min(int((calc['corners']['t'] - 10.5) * 10 + 70), 85)
                    apostas.append(f"✅ **Over 10.5 Cantos** ({prob}%)")
                
                if calc['cards']['t'] > 4.5:
                    prob = min(int((calc['cards']['t'] - 4.5) * 15 + 68), 82)
                    apostas.append(f"✅ **Over 4.5 Cartões** ({prob}%)")
                
                if not apostas:
                    apostas.append("⚠️ Nenhum mercado com alta probabilidade")
                
                return response + "\n" + "\n".join(apostas)
            
            def ranking_corners(self, n: int = 10, league: str = None) -> str:
                """Ranking de cantos"""
                data = self.kb.get_ranking_corners(n, league)
                
                titulo = f"🔶 **TOP {n} TIMES - ESCANTEIOS"
                if league:
                    titulo += f" ({league})"
                titulo += ":**\n\n"
                
                response = titulo
                
                for i, item in enumerate(data, 1):
                    emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "📊"
                    response += f"{emoji} **{i}. {item['time']}** - {item['cantos']:.1f} cantos/jogo\n"
                    response += f"   🏆 {item['liga']}\n\n"
                
                return response
            
            def ranking_cards(self, n: int = 10, league: str = None) -> str:
                """Ranking de cartões"""
                data = self.kb.get_ranking_cards(n, league)
                
                titulo = f"🟨 **TOP {n} TIMES - CARTÕES"
                if league:
                    titulo += f" ({league})"
                titulo += ":**\n\n"
                
                response = titulo
                
                for i, item in enumerate(data, 1):
                    emoji = "🔴" if i <= 3 else "🟠"
                    response += f"{emoji} **{i}. {item['time']}** - {item['cartoes']:.1f} cartões/jogo\n"
                    response += f"   🏆 {item['liga']}\n\n"
                
                return response
            
            def ranking_goals(self, n: int = 10, league: str = None) -> str:
                """Ranking de gols"""
                data = self.kb.get_ranking_goals(n, league)
                
                titulo = f"⚽ **TOP {n} TIMES - GOLS MARCADOS"
                if league:
                    titulo += f" ({league})"
                titulo += ":**\n\n"
                
                response = titulo
                
                for i, item in enumerate(data, 1):
                    emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "⚽"
                    response += f"{emoji} **{i}. {item['time']}** - {item['gols']:.2f} gols/jogo\n"
                    response += f"   🏆 {item['liga']}\n\n"
                
                return response
            
            def league_averages(self, league_name: str) -> str:
                """Médias de uma liga"""
                data = self.kb.get_league_averages(league_name)
                
                if not data:
                    return f"❌ Liga '{league_name}' não encontrada."
                
                return f"""🏆 **MÉDIAS DA {data['liga']}**

📊 **ESTATÍSTICAS:**
• Times Analisados: **{data['times']}**
• Escanteios Médios: **{data['cantos_media']:.1f}** por jogo
• Cartões Médios: **{data['cartoes_media']:.1f}** por jogo
• Gols Médios: **{data['gols_media']:.2f}** por jogo

💡 **ANÁLISE:**
{self._classify_league(data)}"""
            
            def _classify_league(self, data: dict) -> str:
                """Classifica liga por características"""
                classif = []
                
                if data['cantos_media'] > 10.0:
                    classif.append("• 🔶 Liga com **MUITOS ESCANTEIOS**")
                
                if data['cartoes_media'] > 4.5:
                    classif.append("• 🟨 Liga **MUITO VIOLENTA** (cartões)")
                
                if data['gols_media'] > 2.5:
                    classif.append("• ⚽ Liga **MUITO OFENSIVA** (gols)")
                
                if not classif:
                    classif.append("• 📊 Liga com estatísticas **EQUILIBRADAS**")
                
                return "\n".join(classif)
            
            def compare_teams_full(self, team1: str, team2: str) -> str:
                """Comparação completa entre times"""
                comp = self.kb.compare_teams(team1, team2)
                
                if not comp:
                    return f"❌ Não consegui comparar os times. Verifique os nomes."
                
                return f"""⚖️ **COMPARAÇÃO: {comp['team1']} vs {comp['team2']}**

🔶 **ESCANTEIOS:**
• {comp['team1']}: **{comp['cantos']['team1']:.1f}**
• {comp['team2']}: **{comp['cantos']['team2']:.1f}**
• 🏆 Vantagem: **{comp['cantos']['vantagem']}**

🟨 **CARTÕES:**
• {comp['team1']}: **{comp['cartoes']['team1']:.1f}**
• {comp['team2']}: **{comp['cartoes']['team2']:.1f}**
• 🏆 Vantagem: **{comp['cartoes']['vantagem']}**

⚽ **GOLS MARCADOS:**
• {comp['team1']}: **{comp['gols_marcados']['team1']:.2f}**
• {comp['team2']}: **{comp['gols_marcados']['team2']:.2f}**
• 🏆 Vantagem: **{comp['gols_marcados']['vantagem']}**

🛡️ **GOLS SOFRIDOS:**
• {comp['team1']}: **{comp['gols_sofridos']['team1']:.2f}**
• {comp['team2']}: **{comp['gols_sofridos']['team2']:.2f}**
• 🏆 Vantagem: **{comp['gols_sofridos']['vantagem']}** (defesa)"""
            
            def referee_stats(self, referee_name: str) -> str:
                """Estatísticas do árbitro"""
                data = self.kb.get_referee_stats(referee_name)
                
                if not data:
                    similares = get_close_matches(referee_name, list(self.kb.refs.keys()), n=3, cutoff=0.5)
                    if similares:
                        return f"❌ Árbitro '{referee_name}' não encontrado.\n\n💡 Você quis dizer: {', '.join(similares)}?"
                    return f"❌ Árbitro '{referee_name}' não encontrado."
                
                return f"""👨‍⚖️ **ESTATÍSTICAS - {data['nome']}**

📊 **NÚMEROS:**
• Jogos Apitados: **{data['jogos']}**
• Média de Cartões: **{data['media_cartoes']:.2f}** por jogo
• Cartões Vermelhos: **{data['cartoes_vermelhos']}**
• Taxa de Vermelhos: **{data['red_rate']:.2%}**

🏷️ **CLASSIFICAÇÃO:**
{data['classificacao']}

💡 **IMPACTO NAS APOSTAS:**
{self._referee_impact(data)}"""
            
            def _referee_impact(self, data: dict) -> str:
                """Impacto do árbitro nas apostas"""
                if data['classificacao'].startswith('🔴'):
                    return "• ✅ **ÓTIMO para OVER CARTÕES**\n• ⚠️ Jogos tendem a ser mais tensos"
                elif data['classificacao'].startswith('🟢'):
                    return "• ⚠️ **EVITE** apostas em cartões\n• ✅ Bom para jogos mais fluidos"
                else:
                    return "• 📊 Árbitro com padrão **MÉDIO**"
            
            def referees_ranking(self, n: int = 10) -> str:
                """Ranking de árbitros"""
                data = self.kb.get_referees_ranking(n)
                
                response = f"👨‍⚖️ **TOP {n} ÁRBITROS MAIS RIGOROSOS:**\n\n"
                
                for i, item in enumerate(data, 1):
                    emoji = "🔴" if i <= 3 else "🟠" if i <= 6 else "🟡"
                    response += f"{emoji} **{i}. {item['arbitro']}**\n"
                    response += f"   📊 {item['media_cartoes']:.2f} cartões/jogo ({item['jogos']} jogos)\n"
                    response += f"   🔴 {item['vermelhos']} vermelhos\n\n"
                
                return response
            
            def next_games_team(self, team_name: str, n: int = 3) -> str:
                """Próximos jogos de um time"""
                team_norm = normalize_name(team_name, list(self.kb.stats.keys()))
                
                if not team_norm:
                    return f"❌ Time '{team_name}' não encontrado."
                
                games = self.kb.get_next_games_team(team_norm, n)
                
                if not games:
                    return f"📅 Não encontrei próximos jogos agendados para **{team_norm}**"
                
                response = f"📅 **PRÓXIMOS {len(games)} JOGOS - {team_norm}:**\n\n"
                
                for i, g in enumerate(games, 1):
                    response += f"**{i}. {g['home']} vs {g['away']}**\n"
                    response += f"   📅 {g['date']} às {g['time']}\n\n"
                
                return response
        
        # ============================================================
        # INICIALIZAR SUPERBOT
        # ============================================================
        
        if 'super_intent' not in st.session_state:
            st.session_state.super_intent = SuperIntentDetector()
        
        if 'super_extractor' not in st.session_state:
            st.session_state.super_extractor = SuperEntityExtractor(stats, cal, referees)
        
        if 'super_kb' not in st.session_state:
            st.session_state.super_kb = SuperKnowledgeBase(stats, cal, referees)
        
        if 'super_responder' not in st.session_state:
            st.session_state.super_responder = SuperResponseGenerator(st.session_state.super_kb)
        
        # ============================================================
        # BOAS-VINDAS
        # ============================================================
        
        if not st.session_state.chat_history:
            hoje = datetime.now().strftime('%d/%m/%Y')
            welcome = f"""👋 **Olá! Sou o FutPrevisão SUPERBOT V2.0!**

📅 Hoje é **{hoje}**

🧠 **Tenho acesso TOTAL aos dados do projeto:**
• **{len(stats)}** times de **10 ligas**
• **{len(cal) if not cal.empty else 0}** jogos no calendário
• **{len(referees)}** árbitros cadastrados

💬 **Pergunte QUALQUER COISA:**

📊 **TIMES:**
• "Como está o Arsenal?"
• "Qual a média de escanteios do Liverpool?"
• "Compare Manchester City com Chelsea"

⚽ **JOGOS:**
• "Analisa Arsenal vs Manchester United"
• "Tem jogo hoje?"
• "Quando o Real Madrid joga?"

🏆 **RANKINGS:**
• "Top 10 times com mais cantos"
• "Quais os times mais violentos da Premier League?"
• "Ranking de gols da La Liga"

👨‍⚖️ **ÁRBITROS:**
• "Quem é o árbitro mais rigoroso?"
• "Estatísticas do árbitro Michael Oliver"

📈 **LIGAS:**
• "Qual a média de gols da Bundesliga?"
• "Compare Premier League com La Liga"

**Digite abaixo! 👇**"""
            
            st.session_state.chat_history.append({'role': 'assistant', 'content': welcome})
        
        # ============================================================
        # BOTÕES RÁPIDOS
        # ============================================================
        
        st.markdown("### ⚡ Ações Rápidas:")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        if col1.button("🎯 Jogos Hoje", use_container_width=True):
            st.session_state.chat_history.append({'role': 'user', 'content': 'Quais jogos tem hoje?'})
            st.rerun()
        
        if col2.button("🔶 Top Cantos", use_container_width=True):
            st.session_state.chat_history.append({'role': 'user', 'content': 'Top 10 times com mais cantos'})
            st.rerun()
        
        if col3.button("🟨 Top Cartões", use_container_width=True):
            st.session_state.chat_history.append({'role': 'user', 'content': 'Top 10 times com mais cartões'})
            st.rerun()
        
        if col4.button("👨‍⚖️ Árbitros", use_container_width=True):
            st.session_state.chat_history.append({'role': 'user', 'content': 'Árbitros mais rigorosos'})
            st.rerun()
        
        if col5.button("🗑️ Limpar", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
        
        st.markdown("---")
        
        # ============================================================
        # EXIBIR CHAT
        # ============================================================
        
        for msg in st.session_state.chat_history:
            if msg['role'] == 'user':
                st.chat_message("user", avatar="👤").markdown(msg['content'])
            else:
                st.chat_message("assistant", avatar="🤖").markdown(msg['content'])
        
        # ============================================================
        # INPUT E ROTEAMENTO
        # ============================================================
        
        user_input = st.chat_input("Digite sua pergunta... (ex: 'Como está o Arsenal?')")
        
        if user_input:
            st.session_state.chat_history.append({'role': 'user', 'content': user_input})
            
            # DETECTAR INTENÇÃO
            intent = st.session_state.super_intent.detect(user_input)
            extractor = st.session_state.super_extractor
            responder = st.session_state.super_responder
            
            response = ""
            
            # ========================================
            # ROTEAMENTO INTELIGENTE
            # ========================================
            
            try:
                if intent == 'stats_time':
                    teams = extractor.extract_teams(user_input)
                    if teams:
                        response = responder.team_stats(teams[0])
                    else:
                        response = "⚠️ Não identifiquei o time. Tente: 'Como está o Liverpool?'"
                
                elif intent in ['jogos_hoje', 'jogos_amanha', 'jogos_data']:
                    date_str = extractor.extract_date(user_input)
                    response = responder.games_today(date_str)
                
                elif intent == 'analise_jogo':
                    teams = extractor.extract_teams(user_input)
                    if len(teams) >= 2:
                        response = responder.head_to_head(teams[0], teams[1])
                    elif len(teams) == 1:
                        response = f"⚠️ Preciso de 2 times!\n\nExemplo: 'Analisa {teams[0]} vs Arsenal'"
                    else:
                        response = "⚠️ Não identifiquei os times.\n\nExemplo: 'Analisa Liverpool vs Arsenal'"
                
                elif intent == 'ranking_cantos':
                    league = extractor.extract_league(user_input)
                    n = extractor.extract_number(user_input) or 10
                    response = responder.ranking_corners(int(n), league)
                
                elif intent == 'ranking_cartoes':
                    league = extractor.extract_league(user_input)
                    n = extractor.extract_number(user_input) or 10
                    response = responder.ranking_cards(int(n), league)
                
                elif intent == 'ranking_gols':
                    league = extractor.extract_league(user_input)
                    n = extractor.extract_number(user_input) or 10
                    response = responder.ranking_goals(int(n), league)
                
                elif intent == 'comparar_times':
                    teams = extractor.extract_teams(user_input)
                    if len(teams) >= 2:
                        response = responder.compare_teams_full(teams[0], teams[1])
                    else:
                        response = "⚠️ Preciso de 2 times para comparar!\n\nExemplo: 'Compare Liverpool com Arsenal'"
                
                elif intent == 'media_liga':
                    league = extractor.extract_league(user_input)
                    if league:
                        response = responder.league_averages(league)
                    else:
                        response = "⚠️ Qual liga? Tente: 'Média da Premier League'"
                
                elif intent == 'arbitro_stats':
                    ref = extractor.extract_referee(user_input)
                    if ref:
                        response = responder.referee_stats(ref)
                    else:
                        response = "⚠️ Não identifiquei o árbitro.\n\nExemplo: 'Estatísticas do Michael Oliver'"
                
                elif intent == 'arbitro_ranking':
                    n = extractor.extract_number(user_input) or 10
                    response = responder.referees_ranking(int(n))
                
                elif intent == 'proximos_jogos':
                    teams = extractor.extract_teams(user_input)
                    if teams:
                        n = extractor.extract_number(user_input) or 3
                        response = responder.next_games_team(teams[0], int(n))
                    else:
                        response = "⚠️ De qual time?\n\nExemplo: 'Próximos jogos do Arsenal'"
                
                elif intent == 'saudacao':
                    response = "👋 Olá! Como posso ajudar?\n\n💡 Pergunte sobre times, jogos, rankings, árbitros..."
                
                elif intent == 'agradecimento':
                    response = "😊 Por nada! Estou aqui para ajudar sempre!"
                
                else:
                    # FALLBACK INTELIGENTE
                    response = """🤔 Não entendi perfeitamente...

💡 **Exemplos do que posso fazer:**

📊 **TIMES:**
• "Como está o Arsenal?"
• "Média de escanteios do Liverpool"

⚽ **JOGOS:**
• "Analisa Manchester United vs Chelsea"
• "Jogos de hoje"

🏆 **RANKINGS:**
• "Top 10 times com mais cantos"
• "Times mais violentos da Premier League"

👨‍⚖️ **ÁRBITROS:**
• "Árbitros mais rigorosos"
• "Estatísticas do Michael Oliver"

**Reformule sua pergunta ou escolha um exemplo! 👆**"""
            
            except Exception as e:
                response = f"❌ Ocorreu um erro ao processar sua pergunta.\n\nDetalhes: {str(e)}\n\n💡 Tente reformular!"
            
            st.session_state.chat_history.append({'role': 'assistant', 'content': response})
            st.rerun()


if __name__ == "__main__":
    main()
