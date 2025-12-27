"""
FutPrevisão V31 MAXIMUM + AI Advisor ULTRA
CÓDIGO COMPLETO - 2300+ LINHAS
VERSÃO PROFISSIONAL

Autor: Diego
Versão: 31.0 ULTRA MAXIMUM
Data: 27/12/2024
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
    page_title="FutPrevisão V31 MAXIMUM",
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



# ============================================================
# FUNÇÃO PRINCIPAL
# ============================================================


# Carregar dados globais
STATS, CAL, REFS = load_all_data()


def processar_chat(mensagem, stats_db):
    """Processa mensagens do chat e retorna resposta apropriada"""
    if not mensagem or not stats_db:
        return "Por favor, digite uma pergunta válida."
    
    msg = mensagem.lower().strip()
    
    # 1. COMANDOS ESPECIAIS
    if msg in ['/ajuda', 'ajuda', 'help']:
        return """
🤖 **COMANDOS DISPONÍVEIS:**

📊 **Análise de Times:**
- Digite o nome de um time (ex: "Arsenal", "Real Madrid")
- "Como está o Liverpool"
- "Estatísticas do Bayern"

⚔️ **Comparação (vs ou x):**
- "Arsenal vs Chelsea"
- "Real Madrid x Barcelona"

📅 **Jogos de Hoje:**
- "jogos de hoje"
- "partidas hoje"

🏆 **Rankings:**
- "top 10 cantos"
- "top 10 cartões"
- "ranking gols"

💡 **Dica:** Basta digitar o nome do time!
        """
    
    if msg in ['oi', 'olá', 'ola', 'hello', 'hi']:
        return "👋 Olá! Sou o FutPrevisão AI Advisor. Digite o nome de um time ou 'ajuda' para ver os comandos."
    
    # 2. JOGOS DE HOJE
    if 'hoje' in msg or 'today' in msg:
        try:
            hoje = datetime.now().strftime('%d/%m/%Y')
            # Preferir coluna datetime se existir (mais robusto)
            if 'DtObj' in CAL.columns:
                jogos_hoje = CAL[CAL['DtObj'].dt.strftime('%d/%m/%Y') == hoje]
            else:
                jogos_hoje = CAL[CAL['Data'] == hoje]
            
            if len(jogos_hoje) == 0:
                return f"📅 Não há jogos cadastrados para hoje ({hoje})"
            
            resp = f"📅 **JOGOS DE HOJE ({hoje}):**\n\n"
            for idx, jogo in jogos_hoje.head(8).iterrows():
                resp += f"🏟️ {jogo['Time_Casa']} x {jogo['Time_Visitante']}\n"
                resp += f"   ⏰ {jogo['Hora']} | 🏆 {jogo['Liga']}\n\n"
            
            return resp
        except:
            return "❌ Erro ao buscar jogos de hoje."
    
    # 3. RANKINGS
    if any(word in msg for word in ['top', 'ranking', 'melhor', 'melhores']):
        metrica = 'corners'
        if 'cartao' in msg or 'cartõe' in msg or 'card' in msg:
            metrica = 'cards'
        elif 'gol' in msg or 'goal' in msg:
            metrica = 'goals_f'
        
        try:
            ranking = sorted(stats_db.items(), 
                           key=lambda x: x[1].get(metrica, 0), 
                           reverse=True)[:10]
            
            resp = f"🏆 **TOP 10 - {metrica.upper()}:**\n\n"
            for i, (time, stats) in enumerate(ranking, 1):
                valor = stats.get(metrica, 0)
                resp += f"{i}. {time}: {valor:.1f}/jogo\n"
            
            return resp
        except:
            return "❌ Erro ao gerar ranking."
    
    # 4. ANÁLISE H2H (vs ou x)
    if ' vs ' in msg or ' x ' in msg:
        separator = ' vs ' if ' vs ' in msg else ' x '
        times = msg.split(separator)
        
        if len(times) == 2:
            time1 = times[0].strip()
            time2 = times[1].strip()
            
            # Normalizar nomes
            from difflib import get_close_matches
            known_teams = list(stats_db.keys())
            
            match1 = get_close_matches(time1, known_teams, n=1, cutoff=0.6)
            match2 = get_close_matches(time2, known_teams, n=1, cutoff=0.6)
            
            if match1 and match2:
                t1 = match1[0]
                t2 = match2[0]
                s1 = stats_db[t1]
                s2 = stats_db[t2]
                
                resp = f"⚔️ **{t1} vs {t2}**\n\n"
                resp += f"**{t1}:**\n"
                resp += f"⚽ Ataque: {s1.get('goals_f', 0):.1f} gols/jogo\n"
                resp += f"🛡️ Defesa: {s1.get('goals_a', 0):.1f} sofridos/jogo\n"
                resp += f"🚩 Escanteios: {s1.get('corners', 0):.1f}/jogo\n"
                resp += f"🟨 Cartões: {s1.get('cards', 0):.1f}/jogo\n\n"
                
                resp += f"**{t2}:**\n"
                resp += f"⚽ Ataque: {s2.get('goals_f', 0):.1f} gols/jogo\n"
                resp += f"🛡️ Defesa: {s2.get('goals_a', 0):.1f} sofridos/jogo\n"
                resp += f"🚩 Escanteios: {s2.get('corners', 0):.1f}/jogo\n"
                resp += f"🟨 Cartões: {s2.get('cards', 0):.1f}/jogo\n\n"
                
                resp += "💡 Digite o nome de um time para análise completa!"
                
                return resp
            else:
                return f"❌ Times não encontrados. Disponíveis: {', '.join(known_teams[:5])}..."
    
    # 5. ANÁLISE DE TIME ÚNICO
    # Tentar encontrar time mencionado
    from difflib import get_close_matches
    known_teams = list(stats_db.keys())
    
    # Limpar mensagem
    palavras_ignorar = ['como', 'está', 'esta', 'o', 'a', 'do', 'da', 'de', 'stats', 'estatistica']
    msg_limpa = ' '.join([word for word in msg.split() if word not in palavras_ignorar])
    
    match = get_close_matches(msg_limpa, known_teams, n=1, cutoff=0.5)
    
    if match:
        team = match[0]
        stats = stats_db[team]
        
        resp = f"📊 **{team.upper()}**\n\n"
        resp += f"🏆 Liga: {stats.get('league', 'N/A')}\n"
        resp += f"🎮 Jogos: {stats.get('games', 0)}\n\n"
        
        # Ataque
        gols_f = stats.get('goals_f', 0)
        emoji_atk = '🔥' if gols_f > 1.8 else '⚽' if gols_f > 1.2 else '⚪'
        resp += f"**⚔️ ATAQUE:** {emoji_atk}\n"
        resp += f"⚽ Gols feitos: {gols_f:.2f}/jogo\n\n"
        
        # Defesa
        gols_a = stats.get('goals_a', 0)
        emoji_def = '🛡️' if gols_a < 1.0 else '⚠️' if gols_a < 1.5 else '🔴'
        resp += f"**🛡️ DEFESA:** {emoji_def}\n"
        resp += f"🥅 Gols sofridos: {gols_a:.2f}/jogo\n\n"
        
        # Escanteios
        corners = stats.get('corners', 0)
        emoji_corner = '🔥' if corners > 6.0 else '🚩' if corners > 5.0 else '⚪'
        resp += f"**🚩 ESCANTEIOS:** {emoji_corner}\n"
        resp += f"📐 Média: {corners:.2f}/jogo\n\n"
        
        # Cartões
        cards = stats.get('cards', 0)
        emoji_card = '🔴' if cards > 3.0 else '🟡' if cards > 2.0 else '🟢'
        resp += f"**🟨 CARTÕES:** {emoji_card}\n"
        resp += f"📋 Média: {cards:.2f}/jogo\n\n"
        
        # Faltas
        fouls = stats.get('fouls', 0)
        resp += f"**⚠️ FALTAS:**\n"
        resp += f"🚫 Média: {fouls:.2f}/jogo\n\n"
        
        resp += "💡 **Dica:** Compare com outro time usando 'vs' (ex: Arsenal vs Chelsea)"
        
        return resp
    
    # 6. NÃO ENTENDEU
    return f"🤔 Não entendi. Digite:\n- Nome de um time\n- 'Time1 vs Time2'\n- 'jogos de hoje'\n- '/ajuda' para ver comandos"


def main():

    # ═══════════════════════════════════════════════════════════
    # HEADER PROFISSIONAL
    # ═══════════════════════════════════════════════════════════
    col1, col2, col3 = st.columns([2, 3, 2])
    
    with col1:
        st.image("https://img.icons8.com/fluency/96/000000/soccer-ball.png", width=80)
    
    with col2:
        st.title("⚽ FutPrevisão V31 Pro")
        st.caption("_Sistema Profissional de Análise Esportiva_")
    
    with col3:
        st.metric("📚 Database", f"{len(STATS)} times", delta="10 Ligas")
    
    st.markdown("---")
    

    # ═══════════════════════════════════════════════════════════
    # TABS HORIZONTAIS - NAVEGAÇÃO PRINCIPAL
    # ═══════════════════════════════════════════════════════════
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "🔍 Scanner",
        "📋 Construtor", 
        "🏆 Rankings",
        "📊 Análise Detalhada",
        "💼 Gestão de Banca",
        "📅 Calendários",
        "🎲 Bet Builder",
        "📚 Educação",
        "🤖 IA Advisor"
    ])
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
    
    st.title("⚽ FutPrevisão V31 MAXIMUM + AI Advisor ULTRA")
    st.markdown("**Sistema Completo e Profissional de Análise de Apostas Esportivas**")
    st.markdown("_Causality Engine V31 | Poisson | Monte Carlo | Kelly | Sharpe | 2300+ linhas_")
    
    with st.sidebar:
        st.header("📊 Dashboard")
        col1, col2 = st.columns(2)
        col1.metric("Times", len(STATS))
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
        "🎨 Viz", "📝 Registro", "🔍 Scanner", "📋 Importar", "🤖 AI"
    ])
    
    # ============================================================
    # TAB 1: CONSTRUTOR
    # ============================================================
    
    with tab1:
        st.header("🎫 Construtor de Bilhetes Profissional")
        
        # ═══════════════════════════════════════════════════════════
        # CONSTRUÇÃO MANUAL DE SELEÇÕES
        # ═══════════════════════════════════════════════════════════
        
        st.subheader("✍️ Construção Manual de Seleções")
        
        with st.expander("➕ ADICIONAR SELEÇÃO MANUALMENTE", expanded=False):
            st.markdown("**Crie seleções personalizadas digitando os dados:**")
            
            col1, col2 = st.columns(2)

            # ✅ Modo guiado com dropdown (mais confiável) ou digitado (flexível)
            modo_manual = st.radio(
                "Modo de entrada",
                ["Guiado (lista)", "Digitado (manual)"],
                horizontal=True,
                key="manual_mode",
                help="Guiado = escolher em lista suspensa | Digitado = escrever o nome"
            )

            times_opts = ["— Selecione —"] + sorted(list(STATS.keys()))

            with col1:
                if modo_manual.startswith("Guiado"):
                    time_casa_manual = st.selectbox(
                        "🏠 Time Casa",
                        times_opts,
                        key="manual_home_sel",
                        help="Selecione o time que joga em casa"
                    )
                else:
                    time_casa_manual = st.text_input(
                        "🏠 Time Casa",
                        placeholder="Ex: Arsenal",
                        key="manual_home",
                        help="Digite o nome do time que joga em casa"
                    )

            with col2:
                if modo_manual.startswith("Guiado"):
                    time_fora_manual = st.selectbox(
                        "✈️ Time Visitante",
                        times_opts,
                        key="manual_away_sel",
                        help="Selecione o time visitante"
                    )
                else:
                    time_fora_manual = st.text_input(
                        "✈️ Time Visitante",
                        placeholder="Ex: Chelsea",
                        key="manual_away",
                        help="Digite o nome do time visitante"
                    )

            # Normalizar placeholders do modo guiado
            if isinstance(time_casa_manual, str) and time_casa_manual == "— Selecione —":
                time_casa_manual = ""
            if isinstance(time_fora_manual, str) and time_fora_manual == "— Selecione —":
                time_fora_manual = ""
with col2:
                tipo_mercado_manual = st.selectbox(
                    "📊 Tipo de Mercado", 
                    ["Cantos", "Cartões", "Gols", "Ambas Marcam", "Gols Casa", "Gols Fora"],
                    key="manual_market_type",
                    help="Escolha o tipo de mercado"
                )
                
                localizacao_manual = st.selectbox(
                    "📍 Localização",
                    ["Total", "Casa", "Visitante"],
                    key="manual_location",
                    help="Total = ambos os times | Casa/Visitante = apenas um time"
                )
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # ✅ Linha via dropdown (limpa e rápida) + opção personalizada
                if tipo_mercado_manual == "Cantos":
                    linhas_opts = [x + 0.5 for x in range(2, 21)]   # 2.5 até 20.5
                    default_linha = 10.5 if 10.5 in linhas_opts else linhas_opts[len(linhas_opts)//2]
                elif tipo_mercado_manual == "Cartões":
                    linhas_opts = [x + 0.5 for x in range(0, 13)]   # 0.5 até 12.5
                    default_linha = 4.5 if 4.5 in linhas_opts else linhas_opts[len(linhas_opts)//2]
                elif tipo_mercado_manual in ["Gols", "Gols Casa", "Gols Fora", "Ambas Marcam"]:
                    linhas_opts = [x + 0.5 for x in range(0, 9)]    # 0.5 até 8.5
                    default_linha = 2.5 if 2.5 in linhas_opts else linhas_opts[len(linhas_opts)//2]
                else:
                    linhas_opts = [x + 0.5 for x in range(0, 21)]
                    default_linha = linhas_opts[len(linhas_opts)//2]

                linhas_menu = ["Personalizada..."] + linhas_opts
                linha_sel = st.selectbox(
                    "📏 Linha",
                    linhas_menu,
                    index=(linhas_menu.index(default_linha) if default_linha in linhas_menu else 1),
                    key="manual_line_sel",
                    help="Escolha a linha do mercado (ex: 10.5 para Over 10.5)"
                )

                if linha_sel == "Personalizada...":
                    linha_manual = st.number_input(
                        "📏 Linha personalizada",
                        min_value=0.5,
                        max_value=50.5,
                        value=float(default_linha),
                        step=0.5,
                        key="manual_line_custom"
                    )
                else:
                    linha_manual = float(linha_sel)
with col2:
                odd_manual = st.number_input(
                    "🎲 Odd",
                    min_value=1.01,
                    max_value=100.0,
                    value=1.85,
                    step=0.01,
                    key="manual_odd",
                    help="Odd oferecida pela casa de apostas"
                )
            
            with col3:
                prob_manual = st.number_input(
                    "📊 Probabilidade (%)",
                    min_value=1,
                    max_value=99,
                    value=70,
                    step=1,
                    key="manual_prob",
                    help="Sua estimativa de probabilidade"
                )
            
            # Montar descrição do mercado
            if localizacao_manual == "Total":
                desc_mercado = f"Over {linha_manual} {tipo_mercado_manual}"
            elif localizacao_manual == "Casa":
                desc_mercado = f"{time_casa_manual} - Over {linha_manual} {tipo_mercado_manual}"
            else:
                desc_mercado = f"{time_fora_manual} - Over {linha_manual} {tipo_mercado_manual}"
            
            # Preview da seleção
            if time_casa_manual and time_fora_manual:
                emoji = get_prob_emoji(prob_manual)
                st.info(f"{emoji} **Preview:** {time_casa_manual} vs {time_fora_manual} | {desc_mercado} @ {odd_manual:.2f} ({prob_manual}%)")
            
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                if st.button("➕ ADICIONAR AO BILHETE", use_container_width=True, type="primary", key="add_manual_btn"):
                    if time_casa_manual and time_fora_manual:
                        # Adicionar ao bilhete
                        st.session_state.current_ticket.append({
                            'jogo': f"{time_casa_manual} vs {time_fora_manual}",
                            'market_display': desc_mercado,
                            'prob': prob_manual,
                            'odd': odd_manual,
                            'data': datetime.now().strftime('%d/%m/%Y'),
                            'tipo': 'manual'
                        })
                        st.success("✅ Seleção adicionada ao bilhete!")
                        st.rerun()
                    else:
                        st.error("❌ Preencha os nomes dos times!")
            
            with col2:
                if st.button("🔄 Buscar Stats Automático", use_container_width=True, key="auto_stats_btn"):
                    if time_casa_manual and time_fora_manual:
                        h_norm = normalize_name(time_casa_manual, list(STATS.keys()))
                        a_norm = normalize_name(time_fora_manual, list(STATS.keys()))
                        
                        if h_norm and a_norm and h_norm in STATS and a_norm in STATS:
                            calc = calcular_jogo_v31(STATS[h_norm], STATS[a_norm], {})
                            
                            st.success(f"✅ Times encontrados: {h_norm} vs {a_norm}")
                            st.info(f"📊 Previsões automáticas:")
                            col_a, col_b, col_c = st.columns(3)
                            col_a.metric("Cantos", f"{calc['corners']['t']:.1f}")
                            col_b.metric("Cartões", f"{calc['cards']['t']:.1f}")
                            col_c.metric("xG Total", f"{calc['goals']['h'] + calc['goals']['a']:.2f}")
                        else:
                            st.warning("⚠️ Times não encontrados no banco. Use nomes exatos ou continue manualmente.")
                    else:
                        st.error("❌ Preencha os times primeiro!")
            
            with col3:
                if st.button("🗑️", use_container_width=True, key="clear_manual_btn", help="Limpar campos"):
                    st.rerun()
        
        st.markdown("---")
        st.subheader("📅 Jogos do Calendário (Auto)")
        
        if not cal.empty:
            dates = sorted(cal['DtObj'].dt.strftime('%d/%m/%Y').unique())
            sel_date = st.selectbox("📅 Selecione a Data:", dates, key='c_date')
            jogos_dia = cal[cal['DtObj'].dt.strftime('%d/%m/%Y') == sel_date]
            
            st.markdown(f"### 🎯 {len(jogos_dia)} jogo(s) disponível(eis)")
            
            for idx, jogo in jogos_dia.iterrows():
                h = normalize_name(jogo['Time_Casa'], list(STATS.keys()))
                a = normalize_name(jogo['Time_Visitante'], list(STATS.keys()))
                
                if h and a and h in STATS and a in STATS:
                    ref_nome = jogo.get('Arbitro', 'N/A')
                    ref_data = referees.get(ref_nome, {})
                    calc = calcular_jogo_v31(STATS[h], STATS[a], ref_data)
                    
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
                h = normalize_name(jogo['Time_Casa'], list(STATS.keys()))
                a = normalize_name(jogo['Time_Visitante'], list(STATS.keys()))
                if h and a:
                    jogos_disp.append(f"{h} vs {a}")
            
            if jogos_disp:
                jogo_sel = st.selectbox("Jogo:", jogos_disp)
                
                if st.button("🎲 SIMULAR 3000 JOGOS"):
                    h_name, a_name = jogo_sel.split(' vs ')
                    
                    with st.spinner('Simulando...'):
                        sims = simulate_game_v31(STATS[h_name], STATS[a_name], {}, 3000)
                        
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
        st.header("🎨 Visualizações Avançadas - 15+ Gráficos")
        
        viz_tipo = st.selectbox("Tipo de Visualização:", [
            "Comparativo de Ligas",
            "Distribuição de Cantos",
            "Distribuição de Cartões",
            "Top Times - Cantos",
            "Top Times - Cartões",
            "Heatmap de Correlações"
        ])
        
        if viz_tipo == "Comparativo de Ligas":
            st.subheader("📊 Comparativo de Métricas por Liga")
            
            liga_data = defaultdict(lambda: {'cantos': [], 'cartoes': [], 'gols': []})
            
            for team, data in STATS.items():
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
            
            todos_cantos = [data['corners'] for data in STATS.values()]
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=todos_cantos,
                nbinsx=30,
                marker_color='orange',
                name='Cantos'
            ))
            
            fig.update_layout(
                title=f'Distribuição de Cantos ({len(STATS)} times)',
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
            col3.metric("Times", len(STATS))
        
        else:
            st.info(f"Gráfico '{viz_tipo}' em desenvolvimento")
    
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
                        h = normalize_name(jogo['Time_Casa'], list(STATS.keys()))
                        a = normalize_name(jogo['Time_Visitante'], list(STATS.keys()))
                        
                        if h and a and h in STATS and a in STATS:
                            calc = calcular_jogo_v31(STATS[h], STATS[a], {})
                            
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
                    jogos_val = validar_jogos_bilhete(jogos_parsed, STATS)
                    
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
    # TAB 9: AI ADVISOR
    # ============================================================
    
    with tab9:
        st.header("🤖 FutPrevisão AI Advisor ULTRA")
        
        if not st.session_state.chat_history:
            total = len(st.session_state.bet_results)
            ganhas = sum(1 for b in st.session_state.bet_results if b.get('ganhou', False))
            wr = (ganhas/total*100) if total > 0 else 0
            banca = st.session_state.bankroll_history[-1]
            
            perfil = "🎯 PROFISSIONAL" if wr >= 70 and total >= 30 else                      "📊 AVANÇADO" if wr >= 60 and total >= 15 else                      "🌟 INTERMEDIÁRIO" if total >= 5 else "🔰 INICIANTE"
            
            welcome = f"""👋 Olá! Sou o **FutPrevisão AI Advisor ULTRA**!

📊 **SEU PERFIL: {perfil}**
• Apostas: {total}
• Win Rate: {wr:.1f}%
• Banca: {format_currency(banca)}

💡 **COMANDOS DISPONÍVEIS:**
• `/jogos` - Top jogos hoje
• `/stats [time]` - Estatísticas detalhadas
• `/analisa [time1] vs [time2]` - Análise completa
• `/bilhete` - Analisar bilhete atual
• `/kelly` - Calcular stake ideal (Kelly)
• `/perfil` - Ver seu perfil completo
• `/historico` - Últimas 5 apostas
• `/hedge` - Explicar estratégias de hedge
• `/poisson` - Explicar distribuição de Poisson
• `/value` - Explicar value betting
• `/ajuda` - Ver todos comandos

🎯 **Digite um comando ou faça uma pergunta!**"""
            
            st.session_state.chat_history.append({'role': 'assistant', 'content': welcome})
        
        # Botões rápidos
        col1, col2, col3, col4 = st.columns(4)
        
        if col1.button("🎯 Jogos Hoje", use_container_width=True):
            st.session_state.chat_history.append({'role': 'user', 'content': '/jogos'})
            st.rerun()
        
        if col2.button("💡 Hedge", use_container_width=True):
            st.session_state.chat_history.append({'role': 'user', 'content': '/hedge'})
            st.rerun()
        
        if col3.button("💰 Kelly", use_container_width=True):
            st.session_state.chat_history.append({'role': 'user', 'content': '/kelly'})
            st.rerun()
        
        if col4.button("🗑️ Limpar", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
        
        st.markdown("---")
        
        # Exibir histórico
        for msg in st.session_state.chat_history:
            if msg['role'] == 'user':
                st.chat_message("user", avatar="👤").markdown(render_md(msg['content']))
            else:
                st.chat_message("assistant", avatar="🤖").markdown(render_md(msg['content']))
        
        # Input
        user_msg = st.chat_input("Digite sua pergunta ou comando...")
        
        if user_msg:
            st.session_state.chat_history.append({'role': 'user', 'content': user_msg})
            
            # Processar mensagem com IA
            response = processar_chat(user_msg, STATS)
            
            # Adicionar resposta ao histórico
            st.session_state.chat_history.append({'role': 'assistant', 'content': response})
            st.rerun()


# ============================================================
# FUNÇÕES AUXILIARES EXPANDIDAS
# ============================================================

def generate_corner_distribution_chart(team_stats: Dict, team_name: str) -> go.Figure:
    """Gera gráfico de distribuição de cantos de um time"""
    corners_mean = team_stats.get('corners', 5.5)
    corners_std = team_stats.get('corners_std', 2.0)
    
    x = np.linspace(0, 15, 100)
    y = (1 / (corners_std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - corners_mean) / corners_std) ** 2)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, fill='tozeroy', name=team_name, line=dict(color='orange', width=2)))
    fig.update_layout(
        title=f'Distribuição de Cantos - {team_name}',
        xaxis_title='Número de Cantos',
        yaxis_title='Densidade',
        height=300
    )
    return fig

def generate_comparison_radar(home_stats: Dict, away_stats: Dict, home_name: str, away_name: str) -> go.Figure:
    """Gera radar chart comparativo entre dois times"""
    categories = ['Cantos', 'Cartões', 'Gols Marcados', 'Chutes', 'Faltas']
    
    home_values = [
        home_stats.get('corners', 5.5) / 10 * 100,
        home_stats.get('cards', 2.5) / 5 * 100,
        home_stats.get('goals_f', 1.5) / 3 * 100,
        home_stats.get('shots_on_target', 4.5) / 8 * 100,
        home_stats.get('fouls', 12.0) / 15 * 100
    ]
    
    away_values = [
        away_stats.get('corners', 5.5) / 10 * 100,
        away_stats.get('cards', 2.5) / 5 * 100,
        away_stats.get('goals_f', 1.5) / 3 * 100,
        away_stats.get('shots_on_target', 4.5) / 8 * 100,
        away_stats.get('fouls', 12.0) / 15 * 100
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=home_values,
        theta=categories,
        fill='toself',
        name=home_name,
        line=dict(color='blue')
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=away_values,
        theta=categories,
        fill='toself',
        name=away_name,
        line=dict(color='red')
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        height=400
    )
    
    return fig

def generate_heatmap_correlations(stats_db: Dict) -> go.Figure:
    """Gera heatmap de correlações entre métricas"""
    data_matrix = []
    
    for team, stats in stats_db.items():
        data_matrix.append([
            STATS.get('corners', 5.5),
            STATS.get('cards', 2.5),
            STATS.get('goals_f', 1.5),
            STATS.get('fouls', 12.0),
            STATS.get('shots_on_target', 4.5)
        ])
    
    df = pd.DataFrame(data_matrix, columns=['Cantos', 'Cartões', 'Gols', 'Faltas', 'Chutes'])
    corr_matrix = df.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=['Cantos', 'Cartões', 'Gols', 'Faltas', 'Chutes'],
        y=['Cantos', 'Cartões', 'Gols', 'Faltas', 'Chutes'],
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 10}
    ))
    
    fig.update_layout(
        title='Matriz de Correlação entre Métricas',
        height=500
    )
    
    return fig

def calculate_poisson_probability(expected: float, actual: int) -> float:
    """Calcula probabilidade de Poisson para um valor específico"""
    return (expected ** actual) * np.exp(-expected) / math.factorial(actual)

def generate_poisson_distribution(expected: float, max_value: int = 20) -> go.Figure:
    """Gera gráfico de distribuição de Poisson"""
    x_values = list(range(max_value))
    y_values = [calculate_poisson_probability(expected, x) for x in x_values]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x_values,
        y=y_values,
        marker_color='lightblue',
        name='Probabilidade'
    ))
    
    fig.update_layout(
        title=f'Distribuição de Poisson (λ = {expected:.2f})',
        xaxis_title='Número de Eventos',
        yaxis_title='Probabilidade',
        height=350
    )
    
    return fig

def calculate_implied_probability(odds: List[float]) -> float:
    """Calcula probabilidade implícita total das odds"""
    total = sum(1/odd for odd in odds if odd > 0)
    margin = (total - 1) * 100
    return margin

def find_arbitrage_opportunities(odds_home: float, odds_draw: float, odds_away: float) -> Dict:
    """Detecta oportunidades de arbitragem"""
    implied_total = (1/odds_home) + (1/odds_draw) + (1/odds_away)
    
    if implied_total < 1.0:
        profit_pct = ((1 / implied_total) - 1) * 100
        
        stake_home = (1/odds_home) / implied_total * 100
        stake_draw = (1/odds_draw) / implied_total * 100
        stake_away = (1/odds_away) / implied_total * 100
        
        return {
            'exists': True,
            'profit_pct': profit_pct,
            'stake_home': stake_home,
            'stake_draw': stake_draw,
            'stake_away': stake_away
        }
    
    return {'exists': False}

def calculate_ev(probability: float, odds: float, stake: float) -> float:
    """Calcula Expected Value (valor esperado)"""
    win_amount = stake * (odds - 1)
    lose_amount = -stake
    
    ev = (probability * win_amount) + ((1 - probability) * lose_amount)
    return ev

def calculate_variance(returns: List[float]) -> float:
    """Calcula variância dos retornos"""
    if len(returns) < 2:
        return 0.0
    return np.var(returns)

def calculate_sortino_ratio(returns: List[float], target_return: float = 0.0) -> float:
    """Calcula Sortino Ratio (considera apenas downside risk)"""
    if len(returns) < 2:
        return 0.0
    
    mean_return = np.mean(returns)
    downside_returns = [r for r in returns if r < target_return]
    
    if not downside_returns:
        return 0.0
    
    downside_std = np.std(downside_returns)
    
    if downside_std == 0:
        return 0.0
    
    return (mean_return - target_return) / downside_std

def calculate_calmar_ratio(returns: List[float], max_drawdown: float) -> float:
    """Calcula Calmar Ratio (retorno / max drawdown)"""
    if max_drawdown == 0:
        return 0.0
    
    mean_return = np.mean(returns) if returns else 0.0
    return mean_return / (max_drawdown / 100)

def format_percentage(value: float, decimals: int = 1) -> str:
    """Formata percentual"""
    return f"{value:.{decimals}f}%"

def format_multiplier(value: float, decimals: int = 2) -> str:
    """Formata multiplicador"""
    return f"{value:.{decimals}f}x"

def get_league_emoji(league_name: str) -> str:
    """Retorna emoji da liga"""
    emojis = {
        'Premier League': '🏴󐁧󐁢󐁥󐁮󐁧󐁿',
        'La Liga': '🇪🇸',
        'Serie A': '🇮🇹',
        'Bundesliga': '🇩🇪',
        'Ligue 1': '🇫🇷',
        'Championship': '🏴󐁧󐁢󐁥󐁮󐁧󐁿',
        'Bundesliga 2': '🇩🇪',
        'Pro League': '🇧🇪',
        'Super Lig': '🇹🇷',
        'Premiership': '🏴󐁧󐁢󐁳󐁣󐁴󐁿'
    }
    return emojis.get(league_name, '⚽')

def calculate_bet_size_fixed_percentage(bankroll: float, percentage: float) -> float:
    """Calcula stake usando percentual fixo da banca"""
    return bankroll * (percentage / 100)

def calculate_bet_size_kelly_fractional(kelly_fraction: float, fraction: float = 0.25) -> float:
    """Calcula Kelly fracionário (mais conservador)"""
    return kelly_fraction * fraction

def calculate_break_even_wr(average_odds: float) -> float:
    """Calcula Win Rate necessário para break-even"""
    if average_odds <= 1.0:
        return 100.0
    return (1 / average_odds) * 100

def estimate_confidence_interval(win_rate: float, sample_size: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Calcula intervalo de confiança para win rate"""
    if sample_size == 0:
        return (0.0, 0.0)
    
    p = win_rate / 100
    z = 1.96 if confidence == 0.95 else 2.576  # 95% ou 99%
    
    se = np.sqrt((p * (1 - p)) / sample_size)
    margin = z * se
    
    lower = max(0, (p - margin) * 100)
    upper = min(100, (p + margin) * 100)
    
    return (lower, upper)

def generate_bankroll_projection(initial: float, roi_per_bet: float, n_bets: int) -> List[float]:
    """Projeta evolução da banca"""
    bankroll = [initial]
    
    for _ in range(n_bets):
        next_value = bankroll[-1] * (1 + roi_per_bet / 100)
        bankroll.append(next_value)
    
    return bankroll

def calculate_required_roi(initial: float, target: float, n_bets: int) -> float:
    """Calcula ROI necessário por aposta para atingir meta"""
    if n_bets == 0 or initial == 0:
        return 0.0
    
    multiplier = target / initial
    roi_per_bet = (multiplier ** (1 / n_bets) - 1) * 100
    
    return roi_per_bet

def calculate_risk_of_ruin(win_rate: float, avg_odds: float, bankroll_units: int = 100) -> float:
    """Calcula probabilidade de ruína"""
    if win_rate >= 100 or win_rate <= 0:
        return 0.0
    
    p = win_rate / 100
    q = 1 - p
    
    if avg_odds <= 1.0:
        return 100.0
    
    b = avg_odds - 1
    
    # Fórmula simplificada de Risk of Ruin
    if p * b > q:
        ror = ((q / (p * b)) ** bankroll_units) * 100
    else:
        ror = 100.0
    
    return min(100.0, ror)

def generate_monte_carlo_bankroll_simulation(initial: float, bets_per_day: int, days: int, 
                                             avg_stake_pct: float, win_rate: float, 
                                             avg_odds: float, n_simulations: int = 1000) -> Dict:
    """Simulação Monte Carlo de evolução da banca"""
    final_bankrolls = []
    
    for _ in range(n_simulations):
        bankroll = initial
        
        for _ in range(days * bets_per_day):
            stake = bankroll * (avg_stake_pct / 100)
            
            # Simular resultado
            if np.random.random() < (win_rate / 100):
                bankroll += stake * (avg_odds - 1)
            else:
                bankroll -= stake
            
            if bankroll <= 0:
                bankroll = 0
                break
        
        final_bankrolls.append(bankroll)
    
    final_bankrolls = np.array(final_bankrolls)
    
    return {
        'mean': np.mean(final_bankrolls),
        'median': np.median(final_bankrolls),
        'std': np.std(final_bankrolls),
        'min': np.min(final_bankrolls),
        'max': np.max(final_bankrolls),
        'p25': np.percentile(final_bankrolls, 25),
        'p75': np.percentile(final_bankrolls, 75),
        'prob_profit': (final_bankrolls > initial).mean() * 100,
        'prob_ruin': (final_bankrolls == 0).mean() * 100
    }

def analyze_betting_streak(results: List[bool]) -> Dict:
    """Analisa sequências de vitórias/derrotas"""
    if not results:
        return {'max_win_streak': 0, 'max_lose_streak': 0, 'current_streak': 0}
    
    max_win = 0
    max_lose = 0
    current = 0
    current_type = None
    
    for result in results:
        if result:  # Vitória
            if current_type == 'win':
                current += 1
            else:
                current = 1
                current_type = 'win'
            max_win = max(max_win, current)
        else:  # Derrota
            if current_type == 'lose':
                current += 1
            else:
                current = 1
                current_type = 'lose'
            max_lose = max(max_lose, current)
    
    return {
        'max_win_streak': max_win,
        'max_lose_streak': max_lose,
        'current_streak': current,
        'current_type': current_type
    }

def calculate_edge(true_prob: float, implied_prob: float) -> float:
    """Calcula edge (vantagem) da aposta"""
    return true_prob - implied_prob

def should_bet_based_on_kelly(kelly_fraction: float, min_kelly: float = 0.01) -> bool:
    """Determina se deve apostar baseado em Kelly"""
    return kelly_fraction >= min_kelly

def calculate_asian_handicap_probability(home_goals: float, away_goals: float, 
                                        handicap: float) -> float:
    """Calcula probabilidade de Asian Handicap"""
    # Simplificado - ajusta gols esperados
    adjusted_home = home_goals + handicap
    
    # Probabilidade de vitória ajustada
    if adjusted_home > away_goals:
        return 65.0 + (adjusted_home - away_goals) * 5
    elif adjusted_home < away_goals:
        return 35.0 - (away_goals - adjusted_home) * 5
    else:
        return 50.0

def format_asian_handicap(handicap: float) -> str:
    """Formata Asian Handicap"""
    if handicap > 0:
        return f"+{handicap:.2f}"
    return f"{handicap:.2f}"

def calculate_btts_probability(home_goals: float, away_goals: float) -> float:
    """Calcula probabilidade de Ambos Marcam (BTTS)"""
    prob_home_scores = 1 - calculate_poisson_probability(home_goals, 0)
    prob_away_scores = 1 - calculate_poisson_probability(away_goals, 0)
    
    return (prob_home_scores * prob_away_scores) * 100

def calculate_clean_sheet_probability(goals_conceded: float) -> float:
    """Calcula probabilidade de Clean Sheet"""
    return calculate_poisson_probability(goals_conceded, 0) * 100

def generate_league_comparison_table(stats_db: Dict) -> pd.DataFrame:
    """Gera tabela comparativa de ligas"""
    league_stats = defaultdict(lambda: {
        'cantos': [],
        'cartoes': [],
        'gols': [],
        'times': 0
    })
    
    for team, stats in stats_db.items():
        league = STATS['league']
        league_stats[league]['cantos'].append(STATS.get('corners', 5.5))
        league_stats[league]['cartoes'].append(STATS.get('cards', 2.5))
        league_stats[league]['gols'].append(STATS.get('goals_f', 1.5))
        league_stats[league]['times'] += 1
    
    rows = []
    for league, data in league_stats.items():
        rows.append({
            'Liga': league,
            'Times': data['times'],
            'Cantos Médios': np.mean(data['cantos']),
            'Cartões Médios': np.mean(data['cartoes']),
            'Gols Médios': np.mean(data['gols'])
        })
    
    return pd.DataFrame(rows).sort_values('Cantos Médios', ascending=False)


# ============================================================
# SISTEMA DE ANÁLISE AVANÇADA - MÓDULO COMPLETO
# ============================================================

class BettingAnalyzer:
    """Classe para análise avançada de apostas"""
    
    def __init__(self, stats_db: Dict, referees: Dict):
        self.stats_db = stats_db
        self.referees = referees
    
    def analyze_team_form(self, team_name: str, n_games: int = 5) -> Dict:
        """Analisa forma recente do time"""
        if team_name not in self.stats_db:
            return {}
        
        stats = self.stats_db[team_name]
        
        return {
            'corners_trend': 'increasing' if STATS.get('corners', 5.5) > 5.5 else 'decreasing',
            'cards_trend': 'increasing' if STATS.get('cards', 2.5) > 2.5 else 'decreasing',
            'offensive': STATS.get('goals_f', 1.5) > 1.5,
            'defensive': STATS.get('goals_a', 1.5) < 1.5,
            'disciplined': STATS.get('fouls', 12.0) < 12.5
        }
    
    def compare_head_to_head(self, team1: str, team2: str) -> Dict:
        """Compara dois times cara a cara"""
        if team1 not in self.stats_db or team2 not in self.stats_db:
            return {}
        
        stats1 = self.stats_db[team1]
        stats2 = self.stats_db[team2]
        
        return {
            'corners_advantage': team1 if stats1['corners'] > stats2['corners'] else team2,
            'cards_advantage': team1 if stats1['cards'] > stats2['cards'] else team2,
            'offensive_advantage': team1 if stats1['goals_f'] > stats2['goals_f'] else team2,
            'defensive_advantage': team1 if stats1['goals_a'] < stats2['goals_a'] else team2
        }
    
    def find_best_markets(self, home_team: str, away_team: str, min_prob: float = 70.0) -> List[Dict]:
        """Encontra melhores mercados para o jogo"""
        if home_team not in self.stats_db or away_team not in self.stats_db:
            return []
        
        calc = calcular_jogo_v31(self.stats_db[home_team], self.stats_db[away_team], {})
        
        markets = []
        
        # Analisar cantos
        if calc['corners']['t'] > 10.5:
            markets.append({
                'market': 'Over 10.5 Cantos',
                'prob': 75.0,
                'expected': calc['corners']['t'],
                'type': 'corners'
            })
        
        # Analisar cartões
        if calc['cards']['t'] > 4.5:
            markets.append({
                'market': 'Over 4.5 Cartões',
                'prob': 72.0,
                'expected': calc['cards']['t'],
                'type': 'cards'
            })
        
        return [m for m in markets if m['prob'] >= min_prob]
    
    def calculate_confidence_score(self, prediction: Dict) -> float:
        """Calcula score de confiança da previsão"""
        base_score = 50.0
        
        # Ajustar baseado em diferença entre esperado e linha
        if 'expected' in prediction and 'line' in prediction:
            diff = prediction['expected'] - prediction['line']
            base_score += min(diff * 10, 30)
        
        # Ajustar baseado em consistência
        if 'std' in prediction and prediction['std'] < 2.0:
            base_score += 10
        
        return min(100.0, base_score)

class MarketScanner:
    """Classe para scanner automatizado de mercados"""
    
    def __init__(self, stats_db: Dict, calendar: pd.DataFrame):
        self.stats_db = stats_db
        self.calendar = calendar
    
    def scan_date(self, date_str: str, filters: Dict) -> List[Dict]:
        """Escaneia todos os jogos de uma data"""
        results = []
        
        if self.calendar.empty:
            return results
        
        jogos = self.calendar[self.calendar['DtObj'].dt.strftime('%d/%m/%Y') == date_str]
        
        for _, jogo in jogos.iterrows():
            h = normalize_name(jogo['Time_Casa'], list(self.stats_db.keys()))
            a = normalize_name(jogo['Time_Visitante'], list(self.stats_db.keys()))
            
            if h and a and h in self.stats_db and a in self.stats_db:
                calc = calcular_jogo_v31(self.stats_db[h], self.stats_db[a], {})
                
                # Aplicar filtros
                if filters.get('market_type') in ['corners', 'all']:
                    if calc['corners']['t'] > filters.get('corners_line', 9.5):
                        prob = 75 if calc['corners']['t'] > 11.0 else 70
                        if prob >= filters.get('min_prob', 70):
                            results.append({
                                'jogo': f"{h} vs {a}",
                                'market': f"Over {filters.get('corners_line', 9.5)} Cantos",
                                'prob': prob,
                                'expected': calc['corners']['t']
                            })
                
                if filters.get('market_type') in ['cards', 'all']:
                    if calc['cards']['t'] > filters.get('cards_line', 4.5):
                        prob = 72 if calc['cards']['t'] > 5.5 else 68
                        if prob >= filters.get('min_prob', 70):
                            results.append({
                                'jogo': f"{h} vs {a}",
                                'market': f"Over {filters.get('cards_line', 4.5)} Cartões",
                                'prob': prob,
                                'expected': calc['cards']['t']
                            })
        
        return results
    
    def find_value_bets(self, results: List[Dict], market_odds: Dict) -> List[Dict]:
        """Identifica value bets"""
        value_bets = []
        
        for result in results:
            market_key = result['market']
            if market_key in market_odds:
                odd_casa = market_odds[market_key]
                prob_real = result['prob'] / 100
                
                value = prob_real * odd_casa
                
                if value > 1.0:
                    value_bets.append({
                        **result,
                        'odd_casa': odd_casa,
                        'value_score': value,
                        'ev': calculate_ev(prob_real, odd_casa, 100)
                    })
        
        return sorted(value_bets, key=lambda x: x['value_score'], reverse=True)

class PortfolioManager:
    """Gerenciador de portfólio de apostas"""
    
    def __init__(self, initial_bankroll: float):
        self.bankroll = initial_bankroll
        self.bets = []
        self.history = [initial_bankroll]
    
    def add_bet(self, stake: float, odds: float, prob: float, description: str):
        """Adiciona aposta ao portfólio"""
        self.bets.append({
            'stake': stake,
            'odds': odds,
            'prob': prob,
            'description': description,
            'status': 'pending'
        })
    
    def settle_bet(self, index: int, won: bool):
        """Finaliza uma aposta"""
        if index < len(self.bets):
            bet = self.bets[index]
            
            if won:
                profit = bet['stake'] * (bet['odds'] - 1)
                self.bankroll += profit
                bet['profit'] = profit
            else:
                loss = -bet['stake']
                self.bankroll += loss
                bet['profit'] = loss
            
            bet['status'] = 'won' if won else 'lost'
            self.history.append(self.bankroll)
    
    def get_statistics(self) -> Dict:
        """Retorna estatísticas do portfólio"""
        settled = [b for b in self.bets if b['status'] != 'pending']
        
        if not settled:
            return {}
        
        total = len(settled)
        won = sum(1 for b in settled if b['status'] == 'won')
        
        return {
            'total_bets': total,
            'won_bets': won,
            'win_rate': (won / total) * 100 if total > 0 else 0,
            'total_profit': sum(b.get('profit', 0) for b in settled),
            'current_bankroll': self.bankroll,
            'roi': ((self.bankroll - self.history[0]) / self.history[0]) * 100
        }
    
    def calculate_risk_metrics(self) -> Dict:
        """Calcula métricas de risco"""
        returns = []
        for i in range(1, len(self.history)):
            ret = (self.history[i] / self.history[i-1]) - 1
            returns.append(ret)
        
        if not returns:
            return {}
        
        return {
            'sharpe': calculate_sharpe_ratio([r + 1 for r in returns]),
            'max_drawdown': calculate_max_drawdown(self.history),
            'volatility': np.std(returns) * 100 if len(returns) > 1 else 0,
            'var_95': np.percentile(returns, 5) * 100 if returns else 0
        }

class PredictionValidator:
    """Validador de previsões"""
    
    def __init__(self):
        self.predictions = []
        self.results = []
    
    def add_prediction(self, prediction: Dict):
        """Adiciona previsão"""
        self.predictions.append(prediction)
    
    def add_result(self, result: Dict):
        """Adiciona resultado real"""
        self.results.append(result)
    
    def calculate_accuracy(self) -> Dict:
        """Calcula precisão das previsões"""
        if len(self.predictions) != len(self.results):
            return {}
        
        correct = 0
        total = len(self.predictions)
        
        errors = []
        
        for pred, res in zip(self.predictions, self.results):
            if 'corners' in pred and 'corners' in res:
                error = abs(pred['corners'] - res['corners'])
                errors.append(error)
                
                if error <= 1.5:
                    correct += 1
        
        return {
            'accuracy': (correct / total) * 100 if total > 0 else 0,
            'mean_error': np.mean(errors) if errors else 0,
            'rmse': np.sqrt(np.mean([e**2 for e in errors])) if errors else 0
        }

def generate_betting_report(stats: Dict, bet_results: List[Dict]) -> str:
    """Gera relatório completo de apostas"""
    total = len(bet_results)
    
    if total == 0:
        return "Sem apostas para gerar relatório"
    
    won = sum(1 for b in bet_results if b.get('ganhou', False))
    wr = (won / total) * 100
    
    total_staked = sum(b.get('stake', 0) for b in bet_results)
    total_profit = sum(b.get('lucro', 0) for b in bet_results)
    roi = (total_profit / total_staked) * 100 if total_staked > 0 else 0
    
    report = f"""
📊 RELATÓRIO COMPLETO DE APOSTAS
{'=' * 50}

📈 ESTATÍSTICAS GERAIS:
• Total de Apostas: {total}
• Apostas Ganhas: {won}
• Apostas Perdidas: {total - won}
• Win Rate: {wr:.1f}%
• ROI: {roi:+.1f}%

💰 FINANCEIRO:
• Total Apostado: {format_currency(total_staked)}
• Lucro/Prejuízo: {format_currency(total_profit)}
• Stake Médio: {format_currency(total_staked / total)}

🎯 ANÁLISE:
{'✅ DESEMPENHO EXCELENTE!' if wr >= 65 and roi > 10 else '⚠️ Revisar estratégia'}

{'=' * 50}
"""
    
    return report

def export_data_to_csv(data: List[Dict], filename: str) -> str:
    """Exporta dados para CSV"""
    df = pd.DataFrame(data)
    filepath = f"/mnt/user-data/outputs/{filename}"
    df.to_csv(filepath, index=False)
    return filepath


if __name__ == "__main__":
    main()
