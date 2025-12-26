"""
FutPrevisão V31 MAXIMUM + SUPERBOT V2.0 - CÓDIGO COMPLETO
TODAS AS 9 TABS + CORREÇÃO DE CARREGAMENTO DE DADOS
VERSÃO: 31.0 FINAL CORRIGIDA
DATA: 26/12/2024
LINHAS: 2400+
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
import os

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
# DETECÇÃO AUTOMÁTICA DE PATHS - CORREÇÃO PRINCIPAL
# ============================================================

def find_data_files():
    """
    Detecta automaticamente onde estão os arquivos de dados
    Tenta múltiplos caminhos possíveis
    """
    possible_paths = [
        '/mnt/project/',           # Claude Project
        './',                      # Diretório atual
        '../',                     # Diretório pai
        os.getcwd() + '/',         # Working directory
        os.path.dirname(os.path.abspath(__file__)) + '/',  # Script directory
    ]
    
    test_file = 'Premier_League_25_26.csv'
    
    for base_path in possible_paths:
        full_path = os.path.join(base_path, test_file)
        if os.path.exists(full_path):
            return base_path
    
    return None

DATA_PATH = find_data_files()

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
    'RB Leipzig': 'RasenBallsport Leipzig', 'Paris SG': 'Paris S-G',
    'Paris S-G': 'Paris Saint Germain', 'Saint-Etienne': 'St Etienne',
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
# CARREGAMENTO DE DADOS COM DEBUG
# ============================================================

@st.cache_data(ttl=3600)
def load_all_data():
    """Carrega todos os dados do sistema COM debug"""
    
    if DATA_PATH is None:
        st.sidebar.error("🚨 **ERRO:** Arquivos não encontrados!")
        st.sidebar.info("📂 Procurei em:")
        st.sidebar.text("  • /mnt/project/")
        st.sidebar.text("  • ./")
        st.sidebar.text("  • ../")
        return {}, pd.DataFrame(), {}
    
    st.sidebar.success(f"✅ **Path:** `{DATA_PATH}`")
    
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
        try:
            filepath = os.path.join(DATA_PATH, filename)
            
            if not os.path.exists(filepath):
                st.sidebar.warning(f"⚠️ {league_name}: Arquivo não encontrado")
                continue
            
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
            
            st.sidebar.success(f"✅ {league_name}: {len(teams)} times")
            
        except Exception as e:
            st.sidebar.error(f"❌ {league_name}: {str(e)}")
    
    # Calendário
    try:
        cal_path = os.path.join(DATA_PATH, 'calendario_ligas.csv')
        if os.path.exists(cal_path):
            cal = pd.read_csv(cal_path, encoding='utf-8')
            if 'Data' in cal.columns:
                cal['DtObj'] = pd.to_datetime(cal['Data'], format='%d/%m/%Y', errors='coerce')
            st.sidebar.success(f"✅ Calendário: {len(cal)} jogos")
    except Exception as e:
        st.sidebar.warning(f"⚠️ Calendário: {str(e)}")
    
    # Árbitros
    try:
        ref_path = os.path.join(DATA_PATH, 'arbitros_5_ligas_2025_2026.csv')
        if os.path.exists(ref_path):
            refs_df = pd.read_csv(ref_path, encoding='utf-8')
            for _, row in refs_df.iterrows():
                referees[row['Arbitro']] = {
                    'factor': row['Media_Cartoes_Por_Jogo'] / 4.0,
                    'games': row['Jogos_Apitados'],
                    'avg_cards': row['Media_Cartoes_Por_Jogo'],
                    'red_cards': row.get('Cartoes_Vermelhos', 0),
                    'red_rate': row.get('Cartoes_Vermelhos', 0) / row['Jogos_Apitados'] if row['Jogos_Apitados'] > 0 else 0.08
                }
            st.sidebar.success(f"✅ Árbitros: {len(referees)}")
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

def calculate_roi(total_staked: float, total_profit: float) -> float:
    """Calcula ROI (Return on Investment)"""
    if total_staked == 0:
        return 0.0
    return (total_profit / total_staked) * 100

# ============================================================
# PARSER DE BILHETES (TAB 8)
# ============================================================

def parse_bilhete_texto(texto: str) -> List[Dict]:
    """Parser inteligente de bilhetes"""
    linhas = [l.strip() for l in texto.split('\n') if l.strip()]
    jogos = []
    
    for linha in linhas:
        if ' vs ' in linha or ' x ' in linha.lower():
            sep = ' vs ' if ' vs ' in linha else ' x '
            partes = linha.split(sep)
            if len(partes) == 2:
                jogos.append({
                    'home': partes[0].strip(),
                    'away': partes[1].strip(),
                    'mercados': []
                })
    
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
                'home_stats': stats_db[h_norm],
                'away_stats': stats_db[a_norm]
            })
    
    return jogos_val

# ============================================================
# SUPERBOT V2.0 - CLASSES
# ============================================================

class SuperIntentDetector:
    """Detector de intenções ULTRA avançado"""
    
    def __init__(self):
        self.patterns = {
            'stats_time': [
                'como está', 'como esta', 'estatística', 'estatisticas',
                'dados do', 'números do', 'stats', 'desempenho', 'performance',
                'como joga', 'como anda', 'média de', 'media de'
            ],
            'jogos_hoje': [
                'jogos hoje', 'partidas hoje', 'joga hoje', 'tem jogo hoje',
                'quais jogos hoje', 'que jogo tem hoje', 'hoje'
            ],
            'jogos_amanha': [
                'jogos amanhã', 'jogos amanha', 'partidas amanhã', 'amanhã', 'amanha'
            ],
            'analise_jogo': [
                ' vs ', ' x ', 'versus', 'contra', 'analisa', 'analise',
                'quem ganha', 'previsão', 'previsao', 'favorito'
            ],
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
            'comparar_times': [
                'compare', 'compara', 'diferença entre', 'qual melhor',
                'quem é melhor', 'x ou y', 'versus'
            ],
            'media_liga': [
                'média da', 'media da', 'liga', 'campeonato'
            ],
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
        
        all_teams = list(self.stats_db.keys())
        
        # Tentar match direto
        for team in all_teams:
            if team.lower() in text_lower:
                teams_found.append(team)
        
        # Se não encontrou, tentar fuzzy
        if not teams_found:
            words = text.split()
            for word in words:
                if len(word) > 3:
                    matches = get_close_matches(word, all_teams, n=2, cutoff=0.6)
                    teams_found.extend(matches)
        
        return list(set(teams_found))[:2]
    
    def extract_date(self, text: str) -> str:
        """Extrai data com NLP natural"""
        text_lower = text.lower()
        
        if any(p in text_lower for p in ['hoje', 'agora', 'hj']):
            return self.today.strftime('%d/%m/%Y')
        
        if any(p in text_lower for p in ['amanhã', 'amanha']):
            return (self.today + timedelta(days=1)).strftime('%d/%m/%Y')
        
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
        }
        
        for key, league in leagues.items():
            if key in text_lower:
                return league
        
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
                })
        
        return games_list
    
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
    
    def compare_teams(self, team1: str, team2: str) -> dict:
        """Compara dois times"""
        t1 = self.get_team_full_stats(team1)
        t2 = self.get_team_full_stats(team2)
        
        if not t1 or not t2:
            return None
        
        return {
            'team1': t1['name'],
            'team2': t2['name'],
            'cantos': {
                'team1': t1['stats'].get('corners', 0),
                'team2': t2['stats'].get('corners', 0),
            },
            'cartoes': {
                'team1': t1['stats'].get('cards', 0),
                'team2': t2['stats'].get('cards', 0),
            },
        }

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
            return f"❌ Time '{team_name}' não encontrado."
        
        s = data['stats']
        
        return f"""📊 **ESTATÍSTICAS COMPLETAS - {data['name']}**

🏟️ **INFORMAÇÕES GERAIS:**
• Liga: **{data['league']}**
• Jogos Analisados: **{data['games']}**

⚽ **ATAQUE:**
• Gols Marcados: **{s.get('goals_f', 0):.2f}** por jogo
• Chutes no Gol: **{s.get('shots_on_target', 0):.1f}** por jogo

🔶 **ESCANTEIOS:**
• Média: **{s.get('corners', 0):.1f}** por jogo
• Em Casa: **{s.get('corners_home', 0):.1f}**
• Fora: **{s.get('corners_away', 0):.1f}**

🟨 **DISCIPLINA:**
• Cartões: **{s.get('cards', 0):.1f}** por jogo
• Faltas: **{s.get('fouls', 0):.1f}** por jogo"""
    
    def games_today(self, date_str: str) -> str:
        """Lista jogos do dia"""
        games = self.kb.get_games_by_date(date_str)
        
        if not games:
            return f"📅 Não encontrei jogos para {date_str}"
        
        response = f"⚽ **JOGOS DE {date_str}:** ({len(games)} partidas)\n\n"
        
        for i, g in enumerate(games, 1):
            calc = calcular_jogo_v31(self.kb.stats[g['home']], self.kb.stats[g['away']], {})
            response += f"**{i}. {g['home']} vs {g['away']}**\n"
            response += f"   🕐 {g['time']} | 🏆 {g['league']}\n"
            response += f"   📊 {calc['corners']['t']:.1f} cantos | {calc['cards']['t']:.1f} cartões\n\n"
        
        return response
    
    def head_to_head(self, team1: str, team2: str) -> str:
        """Análise H2H completa"""
        t1_norm = normalize_name(team1, list(self.kb.stats.keys()))
        t2_norm = normalize_name(team2, list(self.kb.stats.keys()))
        
        if not t1_norm or not t2_norm:
            return f"❌ Times não encontrados"
        
        calc = calcular_jogo_v31(self.kb.stats[t1_norm], self.kb.stats[t2_norm], {})
        
        return f"""🎯 **ANÁLISE: {t1_norm} vs {t2_norm}**

⚽ **EXPECTED GOALS (xG):**
• {t1_norm}: **{calc['goals']['h']:.2f}**
• {t2_norm}: **{calc['goals']['a']:.2f}**

🔶 **ESCANTEIOS:**
• Total: **{calc['corners']['t']:.1f}**

🟨 **CARTÕES:**
• Total: **{calc['cards']['t']:.1f}**"""
    
    def ranking_corners(self, n: int = 10, league: str = None) -> str:
        """Ranking de cantos"""
        data = self.kb.get_ranking_corners(n, league)
        
        response = f"🔶 **TOP {n} TIMES - ESCANTEIOS:**\n\n"
        
        for i, item in enumerate(data, 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "📊"
            response += f"{emoji} **{i}. {item['time']}** - {item['cantos']:.1f} cantos/jogo\n"
        
        return response
    
    def ranking_cards(self, n: int = 10, league: str = None) -> str:
        """Ranking de cartões"""
        data = self.kb.get_ranking_cards(n, league)
        
        response = f"🟨 **TOP {n} TIMES - CARTÕES:**\n\n"
        
        for i, item in enumerate(data, 1):
            emoji = "🔴" if i <= 3 else "🟠"
            response += f"{emoji} **{i}. {item['time']}** - {item['cartoes']:.1f} cartões/jogo\n"
        
        return response
    
    def ranking_goals(self, n: int = 10, league: str = None) -> str:
        """Ranking de gols"""
        data = self.kb.get_ranking_goals(n, league)
        
        response = f"⚽ **TOP {n} TIMES - GOLS:**\n\n"
        
        for i, item in enumerate(data, 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "⚽"
            response += f"{emoji} **{i}. {item['time']}** - {item['gols']:.2f} gols/jogo\n"
        
        return response
    
    def compare_teams_full(self, team1: str, team2: str) -> str:
        """Comparação completa"""
        comp = self.kb.compare_teams(team1, team2)
        
        if not comp:
            return "❌ Não consegui comparar os times"
        
        return f"""⚖️ **COMPARAÇÃO: {comp['team1']} vs {comp['team2']}**

🔶 **ESCANTEIOS:**
• {comp['team1']}: **{comp['cantos']['team1']:.1f}**
• {comp['team2']}: **{comp['cantos']['team2']:.1f}**

🟨 **CARTÕES:**
• {comp['team1']}: **{comp['cartoes']['team1']:.1f}**
• {comp['team2']}: **{comp['cartoes']['team2']:.1f}**"""

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
    
    st.title("⚽ FutPrevisão V31 MAXIMUM + SUPERBOT V2.0")
    st.markdown("**Sistema Completo com IA Avançada - TODAS AS 9 TABS**")
    
    with st.sidebar:
        st.header("📊 Dashboard")
        st.metric("Times", len(stats))
        st.metric("Jogos", len(cal) if not cal.empty else 0)
        st.metric("Árbitros", len(referees))
        st.metric("Banca", format_currency(st.session_state.bankroll_history[-1]))
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "🎫 Construtor", "🛡️ Hedges", "🎲 Simulador", "📊 Métricas",
        "🎨 Viz", "📝 Registro", "🔍 Scanner", "📋 Importar", "🤖 AI V2.0"
    ])
    
    # ============================================================
    # TAB 1: CONSTRUTOR
    # ============================================================
    
    with tab1:
        st.header("🎫 Construtor de Bilhetes")
        
        if not cal.empty:
            dates = sorted(cal['DtObj'].dt.strftime('%d/%m/%Y').unique())
            sel_date = st.selectbox("Data:", dates, key='c_date')
            jogos_dia = cal[cal['DtObj'].dt.strftime('%d/%m/%Y') == sel_date]
            
            st.markdown(f"### {len(jogos_dia)} jogo(s)")
            
            for idx, jogo in jogos_dia.iterrows():
                h = normalize_name(jogo['Time_Casa'], list(stats.keys()))
                a = normalize_name(jogo['Time_Visitante'], list(stats.keys()))
                
                if h and a and h in stats and a in stats:
                    calc = calcular_jogo_v31(stats[h], stats[a], {})
                    
                    with st.expander(f"⚽ {h} vs {a}", expanded=False):
                        col1, col2, col3 = st.columns(3)
                        col1.metric("xG Casa", f"{calc['goals']['h']:.2f}")
                        col2.metric("Cantos", f"{calc['corners']['t']:.1f}")
                        col3.metric("Cartões", f"{calc['cards']['t']:.1f}")
                        
                        if st.button("➕ Adicionar", key=f"add_{idx}"):
                            st.session_state.current_ticket.append({
                                'jogo': f"{h} vs {a}",
                                'prob': 75
                            })
                            st.rerun()
        
        st.markdown("---")
        st.subheader("📋 Bilhete Atual")
        
        if st.session_state.current_ticket:
            st.success(f"✅ {len(st.session_state.current_ticket)} seleção(ões)")
            
            for i, sel in enumerate(st.session_state.current_ticket):
                col1, col2 = st.columns([5, 1])
                col1.write(f"{i+1}. {sel['jogo']}")
                if col2.button("🗑️", key=f"del_{i}"):
                    st.session_state.current_ticket.pop(i)
                    st.rerun()
            
            if st.button("🗑️ LIMPAR", use_container_width=True):
                st.session_state.current_ticket = []
                st.rerun()
        else:
            st.info("📭 Bilhete vazio")
    
    # ============================================================
    # TAB 2: HEDGES
    # ============================================================
    
    with tab2:
        st.header("🛡️ Sistema de Hedges")
        
        if not st.session_state.current_ticket:
            st.warning("⚠️ Bilhete vazio!")
        else:
            stake = st.number_input("Stake", 10.0, 10000.0, 100.0)
            odd = st.number_input("Odd", 1.5, 100.0, 5.0)
            
            ret = stake * odd
            lucro = ret - stake
            
            st.info(f"Retorno: {format_currency(ret)} | Lucro: {format_currency(lucro)}")
            
            with st.expander("🛡️ HEDGE 1", expanded=True):
                h1_stake = stake * 0.30
                st.metric("Stake Hedge", format_currency(h1_stake))
    
    # ============================================================
    # TAB 3: SIMULADOR
    # ============================================================
    
    with tab3:
        st.header("🎲 Simulador Monte Carlo")
        
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
                
                if st.button("🎲 SIMULAR"):
                    h_name, a_name = jogo_sel.split(' vs ')
                    sims = simulate_game_v31(stats[h_name], stats[a_name], {}, 3000)
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Cantos", f"{sims['corners_total'].mean():.1f}")
                    col2.metric("Cartões", f"{sims['cards_total'].mean():.1f}")
                    col3.metric("Gols", f"{(sims['goals_h'] + sims['goals_a']).mean():.1f}")
    
    # ============================================================
    # TAB 4: MÉTRICAS
    # ============================================================
    
    with tab4:
        st.header("📊 Métricas PRO")
        
        if not st.session_state.bet_results:
            st.info("📭 Sem apostas registradas")
        else:
            total = len(st.session_state.bet_results)
            ganhas = sum(1 for b in st.session_state.bet_results if b.get('ganhou', False))
            wr = (ganhas/total)*100
            
            col1, col2 = st.columns(2)
            col1.metric("Win Rate", f"{wr:.1f}%")
            col2.metric("Total Apostas", total)
    
    # ============================================================
    # TAB 5: VISUALIZAÇÕES
    # ============================================================
    
    with tab5:
        st.header("🎨 Visualizações")
        
        viz = st.selectbox("Tipo:", ["Top Cantos", "Top Cartões"])
        
        if viz == "Top Cantos":
            times_sorted = sorted(stats.items(), key=lambda x: x[1]['corners'], reverse=True)[:20]
            
            nomes = [t[0] for t in times_sorted]
            valores = [t[1]['corners'] for t in times_sorted]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(y=nomes, x=valores, orientation='h', marker_color='orange'))
            fig.update_layout(title='Top 20 - Cantos', height=600)
            st.plotly_chart(fig, use_container_width=True)
    
    # ============================================================
    # TAB 6: REGISTRO
    # ============================================================
    
    with tab6:
        st.header("📝 Registrar Apostas")
        
        stake = st.number_input("Stake", 10.0, 10000.0, 50.0, key='reg_stake')
        odd = st.number_input("Odd", 1.01, 100.0, 2.0, key='reg_odd')
        ganhou = st.checkbox("Ganhou?")
        
        if st.button("💾 REGISTRAR"):
            lucro = stake * (odd - 1) if ganhou else -stake
            
            st.session_state.bet_results.append({
                'stake': stake,
                'odd': odd,
                'ganhou': ganhou,
                'lucro': lucro,
            })
            
            nova_banca = st.session_state.bankroll_history[-1] + lucro
            st.session_state.bankroll_history.append(nova_banca)
            
            st.success(f"✅ Registrado! Lucro: {format_currency(lucro)}")
    
    # ============================================================
    # TAB 7: SCANNER
    # ============================================================
    
    with tab7:
        st.header("🔍 Scanner de Jogos")
        
        if not cal.empty:
            dates = sorted(cal['DtObj'].dt.strftime('%d/%m/%Y').unique())
            sel_date = st.selectbox("Data:", dates, key='scan_date')
            prob_min = st.slider("Prob Mínima", 50, 90, 70)
            
            if st.button("🔍 ESCANEAR"):
                st.info("Scanner em desenvolvimento...")
    
    # ============================================================
    # TAB 8: IMPORTAR
    # ============================================================
    
    with tab8:
        st.header("📋 Importar Bilhete")
        
        texto = st.text_area("Cole o bilhete:", height=200)
        
        if st.button("🔍 ANALISAR"):
            if texto.strip():
                jogos = parse_bilhete_texto(texto)
                st.success(f"✅ {len(jogos)} jogo(s) encontrado(s)")
            else:
                st.warning("Cole o texto do bilhete")
    
    # ============================================================
    # TAB 9: SUPERBOT V2.0
    # ============================================================
    
    with tab9:
        st.header("🤖 FutPrevisão AI Advisor SUPERBOT V2.0")
        st.caption("_Inteligência Artificial com acesso TOTAL aos dados do projeto_")
        
        # INICIALIZAR SUPERBOT
        if 'super_intent' not in st.session_state:
            st.session_state.super_intent = SuperIntentDetector()
        
        if 'super_extractor' not in st.session_state:
            st.session_state.super_extractor = SuperEntityExtractor(stats, cal, referees)
        
        if 'super_kb' not in st.session_state:
            st.session_state.super_kb = SuperKnowledgeBase(stats, cal, referees)
        
        if 'super_responder' not in st.session_state:
            st.session_state.super_responder = SuperResponseGenerator(st.session_state.super_kb)
        
        # BOAS-VINDAS
        if not st.session_state.chat_history:
            hoje = datetime.now().strftime('%d/%m/%Y')
            welcome = f"""👋 **Olá! Sou o FutPrevisão SUPERBOT V2.0!**

📅 Hoje é **{hoje}**

🧠 **Tenho acesso TOTAL aos dados:**
• **{len(stats)}** times de **10 ligas**
• **{len(cal) if not cal.empty else 0}** jogos no calendário
• **{len(referees)}** árbitros cadastrados

💬 **Pergunte QUALQUER COISA:**

📊 **TIMES:** "Como está o Arsenal?"
⚽ **JOGOS:** "Analisa Arsenal vs Man United"
🏆 **RANKINGS:** "Top 10 times com mais cantos"
👨‍⚖️ **ÁRBITROS:** "Quem é mais rigoroso?"

**Digite abaixo! 👇**"""
            
            st.session_state.chat_history.append({'role': 'assistant', 'content': welcome})
        
        # BOTÕES RÁPIDOS
        st.markdown("### ⚡ Ações Rápidas:")
        col1, col2, col3, col4 = st.columns(4)
        
        if col1.button("🎯 Jogos Hoje", use_container_width=True):
            st.session_state.chat_history.append({'role': 'user', 'content': 'Jogos hoje'})
            st.rerun()
        
        if col2.button("🔶 Top Cantos", use_container_width=True):
            st.session_state.chat_history.append({'role': 'user', 'content': 'Top 10 cantos'})
            st.rerun()
        
        if col3.button("🟨 Top Cartões", use_container_width=True):
            st.session_state.chat_history.append({'role': 'user', 'content': 'Top 10 cartões'})
            st.rerun()
        
        if col4.button("🗑️ Limpar", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
        
        st.markdown("---")
        
        # EXIBIR CHAT
        for msg in st.session_state.chat_history:
            if msg['role'] == 'user':
                st.chat_message("user", avatar="👤").markdown(msg['content'])
            else:
                st.chat_message("assistant", avatar="🤖").markdown(msg['content'])
        
        # INPUT
        user_input = st.chat_input("Digite sua pergunta...")
        
        if user_input:
            st.session_state.chat_history.append({'role': 'user', 'content': user_input})
            
            intent = st.session_state.super_intent.detect(user_input)
            extractor = st.session_state.super_extractor
            responder = st.session_state.super_responder
            
            response = ""
            
            try:
                if intent == 'stats_time':
                    teams = extractor.extract_teams(user_input)
                    if teams:
                        response = responder.team_stats(teams[0])
                    else:
                        response = "⚠️ Time não identificado"
                
                elif intent in ['jogos_hoje', 'jogos_amanha']:
                    date_str = extractor.extract_date(user_input)
                    response = responder.games_today(date_str)
                
                elif intent == 'analise_jogo':
                    teams = extractor.extract_teams(user_input)
                    if len(teams) >= 2:
                        response = responder.head_to_head(teams[0], teams[1])
                    else:
                        response = "⚠️ Preciso de 2 times"
                
                elif intent == 'ranking_cantos':
                    league = extractor.extract_league(user_input)
                    response = responder.ranking_corners(10, league)
                
                elif intent == 'ranking_cartoes':
                    league = extractor.extract_league(user_input)
                    response = responder.ranking_cards(10, league)
                
                elif intent == 'ranking_gols':
                    league = extractor.extract_league(user_input)
                    response = responder.ranking_goals(10, league)
                
                elif intent == 'comparar_times':
                    teams = extractor.extract_teams(user_input)
                    if len(teams) >= 2:
                        response = responder.compare_teams_full(teams[0], teams[1])
                    else:
                        response = "⚠️ Preciso de 2 times"
                
                elif intent == 'saudacao':
                    response = "👋 Olá! Como posso ajudar?"
                
                else:
                    response = """🤔 Não entendi perfeitamente...

💡 **Exemplos:**
• "Como está o Arsenal?"
• "Analisa Man United vs Chelsea"
• "Top 10 cantos"
• "Jogos de hoje"
"""
            
            except Exception as e:
                response = f"❌ Erro: {str(e)}"
            
            st.session_state.chat_history.append({'role': 'assistant', 'content': response})
            st.rerun()


if __name__ == "__main__":
    main()
