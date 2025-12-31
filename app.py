"""
FutPrevisão V33.1 REFATORADO
Versão com TODAS as correções implementadas

Mudanças principais:
- Tratamento de erros robusto
- Função main() modularizada
- Estrutura de dados consistente
- Validações fortes
- Performance otimizada
- Código limpo e testável

Autor: Diego & Claude AI
Versão: 33.1 REFACTORED
Data: 31/12/2024
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import math
from typing import Dict, List, Optional, Tuple, Any, Union
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from difflib import get_close_matches
import re
from collections import defaultdict
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Scipy (opcional)
try:
    from scipy.stats import poisson, norm
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("Scipy não disponível. Usando fallback para Poisson.")

# ==============================================================================
# CONFIGURAÇÕES E CONSTANTES
# ==============================================================================

# Diretório base
BASE_DIR = Path(__file__).resolve().parent

# Configuração Streamlit
st.set_page_config(
    page_title="FutPrevisão V33.1",
    layout="wide",
    page_icon="⚽",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# ESTILIZAÇÃO CSS
# ==============================================================================

st.markdown('''
<style>
    /* ANIMAÇÕES */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* TABS MODERNAS */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 15px 15px 0px 15px;
        border-radius: 15px 15px 0 0;
        box-shadow: 0 8px 25px rgba(30, 60, 114, 0.3);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.1);
        border-radius: 8px 8px 0 0;
        padding: 12px 24px;
        font-weight: 600;
        color: #e0e0e0;
        border: 1px solid rgba(255,255,255,0.1);
        border-bottom: none;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        backdrop-filter: blur(5px);
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255,255,255,0.25);
        transform: translateY(-2px);
        color: #ffffff;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%) !important;
        color: #1a1a1a !important;
        border-color: #FFD700;
        font-weight: 800;
        transform: scale(1.02);
        box-shadow: 0 -4px 15px rgba(255, 215, 0, 0.3);
    }
    
    /* CHATBOT STYLE */
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 0px 18px 18px 18px;
        padding: 20px;
        border-left: 5px solid #1e3c72;
        box-shadow: 2px 4px 12px rgba(0,0,0,0.08);
        margin-bottom: 12px;
        animation: fadeIn 0.5s ease;
    }
    
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) {
        background: linear-gradient(135deg, #e0f2fe 0%, #bae6fd 100%);
        border-radius: 18px 0px 18px 18px;
        padding: 20px;
        text-align: right;
        margin-bottom: 12px;
        border-right: 5px solid #0284c7;
        box-shadow: 2px 4px 12px rgba(0,0,0,0.08);
        animation: fadeIn 0.5s ease;
    }
    
    /* MÉTRICAS */
    div[data-testid="metric-container"] {
        background: #ffffff;
        padding: 18px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
        border-top: 4px solid #1e3c72;
        transition: all 0.3s;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(30, 60, 114, 0.15);
    }
    
    /* BOTÕES */
    div.stButton > button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.12);
    }
    
    /* ALERTS */
    .stAlert {
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        animation: fadeIn 0.3s ease;
    }
</style>
''', unsafe_allow_html=True)

# ===== CONSTANTES DO MOTOR V31 =====
HOME_ADVANTAGE_MULTIPLIER = 1.10      # Fator casa (+10%)
AWAY_PENALTY_MULTIPLIER = 0.90        # Penalidade fora (-10%)
LEAGUE_AVERAGE_GOALS = 1.35           # Média de gols por liga

PRESSURE_HIGH_THRESHOLD = 6.0         # Chutes no gol para pressão alta
PRESSURE_MED_THRESHOLD = 4.5          # Chutes no gol para pressão média
VIOLENCE_HIGH_THRESHOLD = 12.5        # Faltas para time violento
REF_STRICT_THRESHOLD = 4.5            # Cartões/jogo para árbitro rigoroso

# ===== BLACKLIST V32 =====
BLACKLIST_CORNERS = {
    'Wolves': 2.89, 'Sunderland': 3.61, 'Burnley': 3.78, 'Crystal Palace': 3.78,
    'Elche': 3.06, 'Mallorca': 3.29, 'Osasuna': 3.35, 'Levante': 3.38,
    'Oviedo': 3.82, 'Girona': 3.88, 'Parma': 3.19, 'Cremonese': 3.24,
    'Pisa': 3.35, 'Sassuolo': 3.47, 'Cagliari': 3.55, 'Udinese': 3.74,
    'Empoli': 3.77, 'Strasbourg': 3.39, 'Lorient': 3.44, 'Metz': 3.50,
    'Montpellier': 3.82, 'Heidenheim': 3.55, 'Darmstadt': 3.61
}

BLACKLIST_CARDS = {
    'Arsenal': 1.45, 'Man City': 1.52, 'Liverpool': 1.63, 'Bayern Munich': 1.35,
    'Dortmund': 1.48, 'Inter Milan': 1.55, 'PSG': 1.42, 'Real Madrid': 1.60
}

# ===== MAPEAMENTO DE NOMES =====
NAME_MAPPING = {
    'Man United': 'Manchester United', 'Man Utd': 'Manchester United',
    'Manchester Utd': 'Manchester United', 'Man City': 'Manchester City',
    'Spurs': 'Tottenham', 'Tottenham Hotspur': 'Tottenham',
    'Wolves': 'Wolverhampton', 'Wolverhampton Wanderers': 'Wolverhampton',
    'Paris SG': 'PSG', 'Paris Saint-Germain': 'PSG',
    'Nottm Forest': 'Nottingham Forest', 'Sheffield Utd': 'Sheffield United',
    'Newcastle': 'Newcastle United', 'Brighton': 'Brighton & Hove Albion',
    'West Ham': 'West Ham United', 'Inter': 'Inter Milan', 'Milan': 'AC Milan',
    'Ath Madrid': 'Atletico Madrid', 'Ath Bilbao': 'Athletic Club',
    'Betis': 'Real Betis', 'Sociedad': 'Real Sociedad',
    'Dortmund': 'Borussia Dortmund', 'Leverkusen': 'Bayer Leverkusen',
    'Bayern': 'Bayern Munich', 'Gladbach': 'Borussia Monchengladbach',
    'Frankfurt': 'Eintracht Frankfurt', 'Marseille': 'Olympique Marseille',
    'Lyon': 'Olympique Lyon', 'Monaco': 'AS Monaco', 'Lille': 'LOSC Lille',
    'Leicester': 'Leicester City', 'Leeds': 'Leeds United'
}

# ===== MERCADOS DISPONÍVEIS =====
MERCADOS_DISPONIVEIS = [
    "Selecione...",
    # Escanteios
    "Over 7.5 Cantos Total", "Over 8.5 Cantos Total", "Over 9.5 Cantos Total",
    "Over 10.5 Cantos Total", "Over 11.5 Cantos Total", "Over 12.5 Cantos Total",
    "Over 3.5 Cantos Casa", "Over 4.5 Cantos Casa", "Over 5.5 Cantos Casa",
    "Over 2.5 Cantos Fora", "Over 3.5 Cantos Fora", "Over 4.5 Cantos Fora",
    # Cartões
    "Over 2.5 Cartões Total", "Over 3.5 Cartões Total", "Over 4.5 Cartões Total",
    "Over 5.5 Cartões Total", "Over 1.5 Cartões Casa", "Over 2.5 Cartões Casa",
    "Over 1.5 Cartões Fora", "Over 2.5 Cartões Fora",
    # Gols
    "Over 0.5 Gols", "Over 1.5 Gols", "Over 2.5 Gols", "Over 3.5 Gols",
    "Under 2.5 Gols", "Under 1.5 Gols", "Ambos Marcam (BTTS)",
    # Resultado
    "Vitória Casa", "Vitória Fora", "Empate",
    "Dupla Chance Casa/Empate", "Dupla Chance Fora/Empate"
]

# ==============================================================================
# CLASSES
# ==============================================================================

class ChatMemory:
    """Gerencia memória de curto prazo do chatbot"""
    def __init__(self):
        self.context = {
            'ultimo_time': None,
            'ultimo_jogo': None,
            'historico_analises': []
        }

    def update(self, key: str, value: Any):
        """Atualiza contexto"""
        self.context[key] = value
        if key == 'analise':
            self.context['historico_analises'].append(value)

    def get(self, key: str) -> Any:
        """Recupera do contexto"""
        return self.context.get(key)

    def clear(self):
        """Limpa memória"""
        self.context = {
            'ultimo_time': None,
            'ultimo_jogo': None,
            'historico_analises': []
        }

# ==============================================================================
# FUNÇÕES AUXILIARES
# ==============================================================================

def find_file(filename: str) -> Optional[str]:
    """
    Busca arquivo em múltiplos diretórios
    
    Args:
        filename: Nome do arquivo
        
    Returns:
        Caminho completo ou None se não encontrado
    """
    search_paths = [
        Path('/mnt/project') / filename,
        Path('.') / filename,
        Path('./data') / filename,
        BASE_DIR / filename,
        BASE_DIR / 'data' / filename
    ]
    
    for path in search_paths:
        if path.exists():
            logger.info(f"Arquivo encontrado: {path}")
            return str(path)
    
    logger.warning(f"Arquivo não encontrado: {filename}")
    return None

def normalize_name(name: str, known_teams: List[str]) -> Optional[str]:
    """
    Normaliza nome de time com fuzzy matching
    
    Args:
        name: Nome a normalizar
        known_teams: Lista de times conhecidos
        
    Returns:
        Nome normalizado ou None
    """
    if not name or not known_teams:
        return None
    
    name = str(name).strip()
    
    # Verificar mapeamento direto
    if name in NAME_MAPPING:
        target = NAME_MAPPING[name]
        if target in known_teams:
            return target
        name = target
    
    # Verificar se já está correto
    if name in known_teams:
        return name
    
    # Fuzzy matching
    matches = get_close_matches(name, known_teams, n=1, cutoff=0.6)
    return matches[0] if matches else None

def format_currency(value: float) -> str:
    """Formata valor em reais"""
    return f"R$ {value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def get_prob_emoji(prob: float) -> str:
    """Retorna emoji baseado na probabilidade"""
    if prob >= 80: return "🔥"
    elif prob >= 70: return "✅"
    elif prob >= 60: return "⚠️"
    else: return "🔻"

def get_league_emoji(league: str) -> str:
    """Retorna emoji da liga (corrigido)"""
    emojis = {
        'Premier League': '🏴󠁧󠁢󠁥󠁮󠁧󠁿',
        'La Liga': '🇪🇸',
        'Serie A': '🇮🇹',
        'Bundesliga': '🇩🇪',
        'Ligue 1': '🇫🇷',
        'Championship': '🏴󠁧󠁢󠁥󠁮󠁧󠁿',
        'Bundesliga 2': '🇩🇪',
        'Pro League': '🇧🇪',
        'Super Lig': '🇹🇷',
        'Premiership': '🏴󠁧󠁢󠁳󠁣󠁴󠁿'
    }
    return emojis.get(league, '⚽')

def validar_odd(odd: float) -> Tuple[bool, str]:
    """Valida odd de aposta"""
    if odd < 1.01:
        return False, "❌ Odd muito baixa (mín: 1.01)"
    elif odd > 50.0:
        return False, "❌ Odd muito alta (máx: 50.0)"
    elif odd < 1.10:
        return True, "⚠️ Odd baixa - Pouco valor"
    return True, "✅ OK"

def validar_stake(stake: float, banca: float, max_percent: float = 10.0) -> Tuple[bool, str]:
    """Valida stake de aposta"""
    if stake <= 0:
        return False, "❌ Stake deve ser maior que zero"
    
    percent = (stake / banca * 100) if banca > 0 else 0
    
    if percent > max_percent:
        return False, f"❌ Stake muito alto! ({percent:.1f}% da banca, máx {max_percent}%)"
    elif percent > 5.0:
        return True, f"⚠️ Stake agressivo ({percent:.1f}%)"
    return True, f"✅ Stake seguro ({percent:.1f}%)"

# ==============================================================================
# CARREGAMENTO DE DADOS (REFATORADO)
# ==============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def load_all_data() -> Tuple[Dict, pd.DataFrame, Dict]:
    """
    Carrega todos os dados com tratamento robusto de erros
    
    Returns:
        Tuple[stats_db, calendario, referees]
    """
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
    
    loaded_count = 0
    
    # Carregar ligas
    for league_name, filename in league_files.items():
        filepath = find_file(filename)
        
        if not filepath:
            logger.warning(f"{league_name}: Arquivo não encontrado")
            st.sidebar.warning(f"⚠️ {league_name}: Arquivo não encontrado")
            continue
        
        try:
            df = pd.read_csv(filepath, encoding='utf-8-sig')
            
            # Limpar nomes de colunas
            df.columns = [c.strip() for c in df.columns]
            
            # Verificar colunas essenciais
            required_cols = ['HomeTeam', 'AwayTeam']
            missing_cols = [c for c in required_cols if c not in df.columns]
            
            if missing_cols:
                raise ValueError(f"Colunas faltando: {missing_cols}")
            
            # Processar times
            teams = set(df['HomeTeam'].dropna().unique()) | set(df['AwayTeam'].dropna().unique())
            
            for team in teams:
                if pd.isna(team):
                    continue
                
                h_games = df[df['HomeTeam'] == team]
                a_games = df[df['AwayTeam'] == team]
                
                # Calcular métricas com fallback
                stats_db[team] = {
                    'league': league_name,
                    'corners_home': h_games['HC'].mean() if 'HC' in df.columns else 5.0,
                    'corners_away': a_games['AC'].mean() if 'AC' in df.columns else 4.0,
                    'cards_home': (h_games['HY'].mean() + h_games['HR'].mean() * 2) if 'HY' in df.columns else 1.8,
                    'cards_away': (a_games['AY'].mean() + a_games['AR'].mean() * 2) if 'AY' in df.columns else 2.2,
                    'fouls_home': h_games['HF'].mean() if 'HF' in df.columns else 11.5,
                    'fouls_away': a_games['AF'].mean() if 'AF' in df.columns else 12.5,
                    'goals_f_home': h_games['FTHG'].mean() if 'FTHG' in df.columns else 1.4,
                    'goals_f_away': a_games['FTAG'].mean() if 'FTAG' in df.columns else 1.1,
                    'goals_a_home': h_games['FTAG'].mean() if 'FTAG' in df.columns else 1.0,
                    'goals_a_away': a_games['FTHG'].mean() if 'FTHG' in df.columns else 1.5,
                    'shots_home': h_games['HST'].mean() if 'HST' in df.columns else 4.8,
                    'shots_away': a_games['AST'].mean() if 'AST' in df.columns else 3.8,
                    'games_played': len(h_games) + len(a_games)
                }
            
            loaded_count += 1
            logger.info(f"✅ {league_name}: {len(teams)} times carregados")
            st.sidebar.success(f"✅ {league_name}: {len(teams)} times")
            
        except FileNotFoundError:
            st.sidebar.error(f"❌ {league_name}: Arquivo não encontrado")
            logger.error(f"{league_name}: Arquivo não encontrado")
            
        except pd.errors.EmptyDataError:
            st.sidebar.error(f"❌ {league_name}: Arquivo vazio")
            logger.error(f"{league_name}: Arquivo vazio")
            
        except ValueError as e:
            st.sidebar.error(f"❌ {league_name}: {str(e)}")
            logger.error(f"{league_name}: {str(e)}")
            
        except Exception as e:
            st.sidebar.error(f"❌ {league_name}: Erro inesperado")
            logger.exception(f"{league_name}: {str(e)}")
    
    # Carregar calendário
    cal_path = find_file('calendario_ligas.csv')
    if cal_path:
        try:
            cal = pd.read_csv(cal_path, encoding='utf-8-sig')
            if 'Data' in cal.columns:
                cal['DtObj'] = pd.to_datetime(cal['Data'], dayfirst=True, errors='coerce')
            logger.info(f"✅ Calendário: {len(cal)} jogos")
            st.sidebar.success(f"✅ Calendário: {len(cal)} jogos")
        except Exception as e:
            st.sidebar.warning(f"⚠️ Calendário: {str(e)}")
            logger.error(f"Calendário: {str(e)}")
    
    # Carregar árbitros
    ref_path = find_file('arbitros_5_ligas_2025_2026.csv')
    if ref_path:
        try:
            refs_df = pd.read_csv(ref_path, encoding='utf-8-sig')
            for _, row in refs_df.iterrows():
                avg = row.get('Media_Cartoes_Por_Jogo', 4.0)
                games = row.get('Jogos_Apitados', 1)
                referees[row['Arbitro']] = {
                    'factor': avg / 4.0,
                    'avg_cards': avg,
                    'games': games,
                    'red_rate': row.get('Cartoes_Vermelhos', 0) / max(games, 1)
                }
            logger.info(f"✅ Árbitros: {len(referees)}")
            st.sidebar.success(f"✅ Árbitros: {len(referees)}")
        except Exception as e:
            st.sidebar.warning(f"⚠️ Árbitros: {str(e)}")
            logger.error(f"Árbitros: {str(e)}")
    
    # Resumo final
    if loaded_count > 0:
        st.sidebar.info(f"📊 {len(stats_db)} times carregados de {loaded_count} ligas")
    else:
        st.sidebar.error("❌ Nenhuma liga foi carregada!")
    
    return stats_db, cal, referees

# ==============================================================================
# MOTOR DE CÁLCULO V31 (ESTRUTURA CONSISTENTE)
# ==============================================================================

def calcular_poisson(media: float, linha: float) -> float:
    """
    Calcula probabilidade de OVER usando distribuição Poisson
    
    Args:
        media: Média esperada
        linha: Linha da aposta (ex: 9.5)
        
    Returns:
        Probabilidade em percentual (0-100)
    """
    if media <= 0:
        return 0.0
    
    if SCIPY_AVAILABLE:
        try:
            return (1 - poisson.cdf(int(linha), media)) * 100
        except Exception as e:
            logger.warning(f"Erro no scipy poisson: {e}")
    
    # Fallback manual
    try:
        k = int(linha)
        prob_at_most_k = sum(
            (math.exp(-media) * (media ** i)) / math.factorial(i)
            for i in range(k + 1)
        )
        return (1 - prob_at_most_k) * 100
    except Exception as e:
        logger.error(f"Erro no cálculo Poisson: {e}")
        return 0.0

def calcular_jogo_v31(home_stats: Dict, away_stats: Dict, ref_data: Dict) -> Dict:
    """
    Motor de Cálculo V31 - Causality Engine
    Retorna estrutura CONSISTENTE (chaves planas)
    
    Args:
        home_stats: Estatísticas do mandante
        away_stats: Estatísticas do visitante
        ref_data: Dados do árbitro
        
    Returns:
        Dict com projeções (estrutura plana)
    """
    if not home_stats or not away_stats:
        return {
            'corners_home': 0, 'corners_away': 0, 'corners_total': 0,
            'cards_home': 0, 'cards_away': 0, 'cards_total': 0,
            'goals_home': 0, 'goals_away': 0, 'goals_total': 0
        }
    
    # === ESCANTEIOS ===
    base_corners_h = home_stats.get('corners_home', 5.0)
    base_corners_a = away_stats.get('corners_away', 4.0)
    
    # Pressão ofensiva (baseada em chutes)
    shots_h = home_stats.get('shots_home', 4.5)
    shots_a = away_stats.get('shots_away', 3.5)
    
    press_h = 1.15 if shots_h > PRESSURE_HIGH_THRESHOLD else \
              1.05 if shots_h > PRESSURE_MED_THRESHOLD else 1.0
    press_a = 1.10 if shots_a > PRESSURE_MED_THRESHOLD else 1.0
    
    corners_h = base_corners_h * press_h * HOME_ADVANTAGE_MULTIPLIER
    corners_a = base_corners_a * press_a * AWAY_PENALTY_MULTIPLIER
    corners_total = corners_h + corners_a
    
    # === CARTÕES ===
    fouls_h = home_stats.get('fouls_home', 11.0)
    fouls_a = away_stats.get('fouls_away', 12.0)
    
    # Fator de violência
    viol_h = 1.1 if fouls_h > VIOLENCE_HIGH_THRESHOLD else 1.0
    viol_a = 1.1 if fouls_a > VIOLENCE_HIGH_THRESHOLD else 1.0
    
    # Fator do árbitro
    ref_avg = ref_data.get('avg_cards', 4.0) if ref_data else 4.0
    cards_h_base = home_stats.get('cards_home', 1.8)
    cards_a_base = away_stats.get('cards_away', 2.2)
    
    cards_h = ((cards_h_base + (ref_avg / 2)) / 2) * viol_h
    cards_a = ((cards_a_base + (ref_avg / 2)) / 2) * viol_a
    cards_total = cards_h + cards_a
    
    # === GOLS (xG) ===
    goals_fh = home_stats.get('goals_f_home', 1.4)
    goals_fa = away_stats.get('goals_f_away', 1.1)
    goals_ah = away_stats.get('goals_a_away', 1.5)
    goals_aa = home_stats.get('goals_a_home', 1.0)
    
    xg_h = (goals_fh / LEAGUE_AVERAGE_GOALS) * (goals_ah / LEAGUE_AVERAGE_GOALS) * LEAGUE_AVERAGE_GOALS
    xg_a = (goals_fa / LEAGUE_AVERAGE_GOALS) * (goals_aa / LEAGUE_AVERAGE_GOALS) * LEAGUE_AVERAGE_GOALS
    goals_total = xg_h + xg_a
    
    return {
        'corners_home': corners_h,
        'corners_away': corners_a,
        'corners_total': corners_total,
        'cards_home': cards_h,
        'cards_away': cards_a,
        'cards_total': cards_total,
        'goals_home': xg_h,
        'goals_away': xg_a,
        'goals_total': goals_total
    }

def calcular_probabilidade_mercado(mercado: str, calc: Dict) -> float:
    """
    Calcula probabilidade de um mercado (COM VALIDAÇÃO ROBUSTA)
    
    Args:
        mercado: String do mercado
        calc: Dict de projeções
        
    Returns:
        Probabilidade 0-100 ou 0.0 se inválido
    """
    if not mercado or mercado == "Selecione...":
        return 0.0
    
    if not calc or not isinstance(calc, dict):
        logger.warning("Cálculos inválidos")
        return 0.0
    
    if "Over" in mercado:
        try:
            # Extrair linha
            linha_match = re.findall(r'\d+\.5', mercado)
            if not linha_match:
                logger.warning(f"Linha não encontrada em: {mercado}")
                return 0.0
            
            linha = float(linha_match[0])
            
            # Mapear mercado -> chave
            mapa = {
                'Cantos Casa': 'corners_home',
                'Cantos Fora': 'corners_away',
                'Cantos Total': 'corners_total',
                'Cartões Casa': 'cards_home',
                'Cartões Fora': 'cards_away',
                'Cartões Total': 'cards_total',
                'Gols': 'goals_total'
            }
            
            for chave_merc, chave_calc in mapa.items():
                if chave_merc in mercado:
                    if chave_calc not in calc:
                        logger.error(f"Chave '{chave_calc}' não encontrada em calc")
                        return 0.0
                    
                    media = calc[chave_calc]
                    return calcular_poisson(media, linha)
            
            logger.warning(f"Mercado não mapeado: {mercado}")
            return 0.0
            
        except (IndexError, ValueError) as e:
            logger.error(f"Erro ao processar mercado '{mercado}': {e}")
            return 0.0
    
    return 0.0

def simulate_game_v31(home_stats: Dict, away_stats: Dict, ref_data: Dict, n_sims: int = 3000) -> Dict:
    """Simulador Monte Carlo"""
    calc = calcular_jogo_v31(home_stats, away_stats, ref_data)
    
    return {
        'corners_total': np.random.poisson(calc['corners_total'], n_sims),
        'cards_total': np.random.poisson(calc['cards_total'], n_sims),
        'goals_h': np.random.poisson(calc['goals_home'], n_sims),
        'goals_a': np.random.poisson(calc['goals_away'], n_sims),
        'goals_total': np.random.poisson(calc['goals_total'], n_sims)
    }

# ==============================================================================
# FUNÇÕES DE UI (MODULARIZADAS)
# ==============================================================================

def initialize_session_state():
    """Inicializa estado da sessão"""
    if 'current_ticket' not in st.session_state:
        st.session_state.current_ticket = []
    if 'bet_results' not in st.session_state:
        st.session_state.bet_results = []
    if 'bankroll_history' not in st.session_state:
        st.session_state.bankroll_history = [1000.0]
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'chat_memory' not in st.session_state:
        st.session_state.chat_memory = ChatMemory()

def render_sidebar(stats_db: Dict, cal: pd.DataFrame, refs: Dict):
    """Renderiza sidebar"""
    with st.sidebar:
        st.header("📊 Dashboard")
        
        c1, c2 = st.columns(2)
        c1.metric("Times", len(stats_db))
        c2.metric("Jogos", len(cal) if not cal.empty else 0)
        
        banca = st.session_state.bankroll_history[-1]
        lucro = banca - 1000.0
        st.metric("💰 Banca", format_currency(banca), delta=format_currency(lucro))
        
        if st.session_state.current_ticket:
            st.success(f"🎫 {len(st.session_state.current_ticket)} apostas")
            if st.button("Limpar Bilhete", use_container_width=True):
                st.session_state.current_ticket = []
                st.rerun()
        
        st.markdown("---")
        if st.session_state.current_ticket:
            st.download_button(
                "📥 Exportar JSON",
                json.dumps(st.session_state.current_ticket, indent=2),
                "ticket.json",
                use_container_width=True
            )

def render_header(cal: pd.DataFrame):
    """Renderiza header"""
    col1, col2, col3 = st.columns([1, 5, 2])
    
    with col1:
        st.markdown("# ⚽")
    
    with col2:
        st.title("FutPrevisão V33.1 REFATORADO")
        st.caption("Sistema Profissional | Causality Engine V31")
    
    with col3:
        if not cal.empty:
            hoje = datetime.now().strftime('%d/%m/%Y')
            jogos_hoje = len(cal[cal['Data'] == hoje]) if 'Data' in cal.columns else 0
            st.metric("Jogos Hoje", jogos_hoje)
    
    st.markdown("---")

def processar_jogo_calendario(row: pd.Series, stats_db: Dict) -> Optional[Dict]:
    """
    Processa jogo do calendário (ELIMINA DUPLICAÇÃO)
    
    Args:
        row: Linha do DataFrame
        stats_db: Base de dados de stats
        
    Returns:
        Dict com dados do jogo ou None
    """
    h = normalize_name(row.get('Time_Casa', ''), list(stats_db.keys()))
    a = normalize_name(row.get('Time_Visitante', ''), list(stats_db.keys()))
    
    if not (h and a and h in stats_db and a in stats_db):
        return None
    
    return {
        'home': h,
        'away': a,
        'hora': row.get('Hora', '-'),
        'calc': calcular_jogo_v31(stats_db[h], stats_db[a], {})
    }

# ==============================================================================
# TABS INDIVIDUAIS (MODULARIZADAS)
# ==============================================================================

def render_tab_construtor(stats_db: Dict, cal: pd.DataFrame):
    """TAB 1: Construtor"""
    st.subheader("🛠️ Construtor de Bilhetes")
    
    c1, c2 = st.columns([1, 1])
    
    # Modo Automático
    with c1:
        st.markdown("#### 📅 Seleção Automática")
        if cal.empty:
            st.info("Calendário não disponível")
        else:
            dates = sorted(cal['DtObj'].dt.strftime('%d/%m/%Y').unique())
            if dates:
                data_sel = st.selectbox("Data:", dates)
                jogos_dia = cal[cal['DtObj'].dt.strftime('%d/%m/%Y') == data_sel]
                
                for idx, row in jogos_dia.iterrows():
                    jogo = processar_jogo_calendario(row, stats_db)
                    
                    if jogo:
                        with st.expander(f"⚽ {jogo['home']} vs {jogo['away']} | {jogo['hora']}"):
                            m1, m2, m3 = st.columns(3)
                            m1.metric("Cantos", f"{jogo['calc']['corners_total']:.1f}")
                            m2.metric("Gols", f"{jogo['calc']['goals_total']:.1f}")
                            m3.metric("Cartões", f"{jogo['calc']['cards_total']:.1f}")
                            
                            if st.button("+ Over 9.5 Cantos", key=f"ac_{idx}"):
                                prob = calcular_poisson(jogo['calc']['corners_total'], 9.5)
                                st.session_state.current_ticket.append({
                                    'jogo': f"{jogo['home']} vs {jogo['away']}",
                                    'mercado': 'Over 9.5 Cantos Total',
                                    'market_display': 'Over 9.5 Cantos Total',
                                    'odd': 1.85,
                                    'prob': prob,
                                    'tipo': 'Auto'
                                })
                                st.success("✅ Adicionado!")
                                st.rerun()
    
    # Modo Manual
    with c2:
        st.markdown("#### 📝 Modo Manual")
        all_teams = sorted(list(stats_db.keys()))
        
        tc = st.selectbox("🏠 Casa:", ["Selecione..."] + all_teams, key="m_casa")
        tv = st.selectbox("✈️ Fora:", ["Selecione..."] + all_teams, key="m_fora")
        
        c_mk, c_od = st.columns(2)
        m_mercado = c_mk.selectbox("Mercado:", MERCADOS_DISPONIVEIS)
        m_odd = c_od.number_input("Odd:", 1.01, 100.0, 1.90)
        
        # Preview
        if tc != "Selecione..." and tv != "Selecione..." and m_mercado != "Selecione...":
            calc_m = calcular_jogo_v31(stats_db[tc], stats_db[tv], {})
            prob_est = calcular_probabilidade_mercado(m_mercado, calc_m)
            
            if prob_est > 0:
                st.caption(f"🎲 Probabilidade: **{prob_est:.1f}%** {get_prob_emoji(prob_est)}")
        
        if st.button("➕ Adicionar", use_container_width=True):
            if tc != "Selecione..." and tv != "Selecione..." and m_mercado != "Selecione...":
                calc_m = calcular_jogo_v31(stats_db[tc], stats_db[tv], {})
                prob_est = calcular_probabilidade_mercado(m_mercado, calc_m)
                
                st.session_state.current_ticket.append({
                    'jogo': f"{tc} vs {tv}",
                    'mercado': m_mercado,
                    'market_display': m_mercado,
                    'odd': m_odd,
                    'prob': prob_est,
                    'tipo': 'Manual'
                })
                st.success("✅ Adicionado!")
                st.rerun()
    
    # Visualização do Bilhete
    st.markdown("---")
    if st.session_state.current_ticket:
        st.subheader("📋 Seu Bilhete")
        
        df_tick = pd.DataFrame(st.session_state.current_ticket)
        df_show = df_tick.copy()
        df_show['Prob'] = df_show['prob'].apply(lambda x: f"{x:.1f}%")
        df_show['Emoji'] = df_show['prob'].apply(get_prob_emoji)
        
        st.dataframe(
            df_show[['Emoji', 'jogo', 'mercado', 'odd', 'Prob', 'tipo']],
            use_container_width=True
        )
        
        # Métricas
        total_odd = np.prod([x['odd'] for x in st.session_state.current_ticket])
        prob_real = np.prod([x['prob'] / 100 for x in st.session_state.current_ticket]) * 100
        fair_odd = 100 / prob_real if prob_real > 0 else 0
        ev = (total_odd - fair_odd) / fair_odd * 100 if fair_odd > 0 else 0
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Odd Total", f"{total_odd:.2f}")
        c2.metric("Prob Real", f"{prob_real:.1f}%")
        c3.metric("Fair Odd", f"{fair_odd:.2f}")
        c4.metric("EV", f"{ev:+.1f}%")
        
        # Remover
        if len(st.session_state.current_ticket) > 0:
            idx_remove = st.selectbox(
                "Remover:",
                range(len(st.session_state.current_ticket)),
                format_func=lambda i: f"{st.session_state.current_ticket[i]['jogo']} - {st.session_state.current_ticket[i]['mercado']}"
            )
            
            if st.button("🗑️ Remover"):
                st.session_state.current_ticket.pop(idx_remove)
                st.rerun()

def render_tab_hedges(stats_db: Dict):
    """TAB 2: Hedges"""
    st.header("🛡️ Hedges Super Inteligentes")
    
    bilhete = st.session_state.current_ticket
    
    if not bilhete:
        st.warning("⚠️ Crie um bilhete no Construtor primeiro")
        return
    
    col1, col2 = st.columns(2)
    stake = col1.number_input("Stake Principal (R$)", 10.0, 10000.0, 100.0)
    
    odd_total = np.prod([x['odd'] for x in bilhete])
    col2.metric("Odd do Bilhete", f"{odd_total:.2f}")
    
    retorno_max = stake * odd_total
    
    st.markdown("### 🧮 Calculadora de Hedge")
    
    with st.expander("📊 Hedge Manual", expanded=True):
        st.write("Proteja seu investimento contra perdas")
        
        odd_hedge = st.number_input("Odd da Contra-Aposta:", 2.0, 100.0, 3.5)
        stake_hedge = stake / (odd_hedge - 1)
        
        c1, c2 = st.columns(2)
        c1.metric("Apostar na Zebra", format_currency(stake_hedge))
        c2.metric("Custo Total", format_currency(stake + stake_hedge))
        
        lucro_principal = retorno_max - stake_hedge if retorno_max > stake_hedge else 0
        lucro_hedge = (stake_hedge * odd_hedge) - stake if stake_hedge * odd_hedge > stake else 0
        
        st.markdown("### 💰 Cenários")
        col1, col2 = st.columns(2)
        
        with col1:
            st.success("✅ Principal Bate")
            st.metric("Lucro", format_currency(lucro_principal))
        
        with col2:
            st.info("🔄 Zebra Acontece")
            st.metric("Lucro", format_currency(lucro_hedge))

def render_tab_simulador(stats_db: Dict):
    """TAB 3: Simulador Monte Carlo"""
    st.header("🎲 Simulador Monte Carlo")
    st.caption("3.000 iterações por simulação")
    
    c1, c2 = st.columns(2)
    
    all_teams = sorted(list(stats_db.keys()))
    sim_h = c1.selectbox("Time Casa:", all_teams, key='sh')
    sim_a = c2.selectbox("Time Visitante:", all_teams, key='sa')
    
    if st.button("🚀 Iniciar Simulação", use_container_width=True):
        if sim_h == sim_a:
            st.error("❌ Selecione times diferentes")
            return
        
        with st.spinner("Simulando 3.000 partidas..."):
            res = simulate_game_v31(stats_db[sim_h], stats_db[sim_a], {}, 3000)
            st.success("✅ Simulação concluída!")
        
        # Métricas
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Média Gols", f"{res['goals_total'].mean():.2f}")
        m2.metric("Média Cantos", f"{res['corners_total'].mean():.2f}")
        m3.metric("Média Cartões", f"{res['cards_total'].mean():.2f}")
        m4.metric("Prob Over 2.5", f"{(res['goals_total'] > 2.5).mean()*100:.1f}%")
        
        # Gráfico
        fig = px.histogram(
            res['goals_total'], 
            nbins=15,
            title="Distribuição de Gols (3.000 simulações)",
            labels={'value': 'Gols', 'count': 'Frequência'},
            color_discrete_sequence=['#1e3c72']
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Tabela de probabilidades
        st.markdown("### 📊 Probabilidades por Mercado")
        probs_data = {
            'Mercado': [
                'Over 1.5 Gols', 'Over 2.5 Gols', 'Over 3.5 Gols',
                'Over 8.5 Cantos', 'Over 9.5 Cantos', 'Over 10.5 Cantos',
                'Over 3.5 Cartões', 'Over 4.5 Cartões'
            ],
            'Probabilidade': [
                (res['goals_total'] > 1.5).mean() * 100,
                (res['goals_total'] > 2.5).mean() * 100,
                (res['goals_total'] > 3.5).mean() * 100,
                (res['corners_total'] > 8.5).mean() * 100,
                (res['corners_total'] > 9.5).mean() * 100,
                (res['corners_total'] > 10.5).mean() * 100,
                (res['cards_total'] > 3.5).mean() * 100,
                (res['cards_total'] > 4.5).mean() * 100
            ]
        }
        
        df_probs = pd.DataFrame(probs_data)
        df_probs['Probabilidade'] = df_probs['Probabilidade'].apply(lambda x: f"{x:.1f}%")
        df_probs['Emoji'] = df_probs['Probabilidade'].apply(
            lambda x: get_prob_emoji(float(x.replace('%', '')))
        )
        
        st.dataframe(df_probs[['Emoji', 'Mercado', 'Probabilidade']], use_container_width=True)

def render_tab_metricas():
    """TAB 4: Métricas de Performance"""
    st.header("📊 Métricas de Performance")
    
    if not st.session_state.bet_results:
        st.info("📝 Registre apostas na tab 'Registro' para ver métricas aqui")
        return
    
    df = pd.DataFrame(st.session_state.bet_results)
    
    # KPIs principais
    wins = df[df['ganhou'] == True]
    win_rate = len(wins) / len(df) * 100
    total_investido = df['stake'].sum()
    total_lucro = df['lucro'].sum()
    roi = (total_lucro / total_investido) * 100 if total_investido > 0 else 0
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Win Rate", f"{win_rate:.1f}%", delta=f"+{win_rate-50:.1f}%" if win_rate > 50 else f"{win_rate-50:.1f}%")
    c2.metric("ROI", f"{roi:.1f}%")
    c3.metric("Lucro Total", format_currency(total_lucro))
    c4.metric("Apostas", len(df))
    
    # Gráfico de evolução
    st.markdown("### 📈 Evolução da Banca")
    fig = px.line(
        y=st.session_state.bankroll_history,
        title="Histórico de Banca",
        labels={'y': 'Banca (R$)', 'index': 'Aposta #'},
        color_discrete_sequence=['#1e3c72']
    )
    fig.add_hline(y=1000, line_dash="dash", line_color="red", annotation_text="Banca Inicial")
    st.plotly_chart(fig, use_container_width=True)
    
    # Tabela de apostas
    st.markdown("### 📋 Histórico de Apostas")
    df_show = df.copy()
    df_show['lucro'] = df_show['lucro'].apply(format_currency)
    df_show['stake'] = df_show['stake'].apply(format_currency)
    df_show['ganhou'] = df_show['ganhou'].apply(lambda x: '✅' if x else '❌')
    st.dataframe(df_show, use_container_width=True)

def render_tab_visualizacoes(stats_db: Dict):
    """TAB 5: Visualizações Avançadas"""
    st.header("🎨 Visualizações Avançadas")
    
    tipo = st.selectbox(
        "Selecione o tipo:",
        ["Ranking Cantos", "Ranking Cartões", "Ranking Gols", "Ataque vs Defesa"]
    )
    
    if tipo == "Ranking Cantos":
        data = [
            {'Time': k, 'Cantos/Jogo': v.get('corners_home', 0) + v.get('corners_away', 0) / 2, 'Liga': v.get('league', 'N/A')}
            for k, v in stats_db.items()
        ]
        df = pd.DataFrame(data).sort_values('Cantos/Jogo', ascending=False).head(20)
        
        fig = px.bar(
            df, x='Cantos/Jogo', y='Time', orientation='h',
            color='Liga', title="Top 20 Times - Escanteios",
            labels={'Cantos/Jogo': 'Média de Cantos por Jogo'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif tipo == "Ranking Cartões":
        data = [
            {'Time': k, 'Cartões/Jogo': (v.get('cards_home', 0) + v.get('cards_away', 0)) / 2, 'Liga': v.get('league', 'N/A')}
            for k, v in stats_db.items()
        ]
        df = pd.DataFrame(data).sort_values('Cartões/Jogo', ascending=False).head(20)
        
        fig = px.bar(
            df, x='Cartões/Jogo', y='Time', orientation='h',
            color='Liga', title="Top 20 Times - Cartões",
            labels={'Cartões/Jogo': 'Média de Cartões por Jogo'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif tipo == "Ranking Gols":
        data = [
            {'Time': k, 'Gols/Jogo': (v.get('goals_f_home', 0) + v.get('goals_f_away', 0)) / 2, 'Liga': v.get('league', 'N/A')}
            for k, v in stats_db.items()
        ]
        df = pd.DataFrame(data).sort_values('Gols/Jogo', ascending=False).head(20)
        
        fig = px.bar(
            df, x='Gols/Jogo', y='Time', orientation='h',
            color='Liga', title="Top 20 Times - Gols Marcados",
            labels={'Gols/Jogo': 'Média de Gols Marcados'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    else:  # Ataque vs Defesa
        data = [
            {
                'Time': k,
                'Gols Feitos': (v.get('goals_f_home', 0) + v.get('goals_f_away', 0)) / 2,
                'Gols Sofridos': (v.get('goals_a_home', 0) + v.get('goals_a_away', 0)) / 2,
                'Liga': v.get('league', 'N/A')
            }
            for k, v in stats_db.items()
        ]
        df = pd.DataFrame(data)
        
        fig = px.scatter(
            df, x='Gols Feitos', y='Gols Sofridos',
            color='Liga', hover_name='Time',
            title="Ataque vs Defesa",
            labels={'Gols Feitos': 'Gols Marcados/Jogo', 'Gols Sofridos': 'Gols Sofridos/Jogo'}
        )
        st.plotly_chart(fig, use_container_width=True)

def render_tab_registro():
    """TAB 6: Registro Manual"""
    st.header("📝 Registro de Apostas")
    
    with st.form("form_registro"):
        st.markdown("### 🎫 Nova Aposta")
        
        c1, c2 = st.columns(2)
        descricao = c1.text_input("Descrição:", placeholder="Ex: Arsenal vs Chelsea - Over 9.5 Cantos")
        stake = c2.number_input("Stake (R$):", 10.0, 10000.0, 100.0)
        
        c3, c4 = st.columns(2)
        odd = c3.number_input("Odd:", 1.01, 100.0, 1.90)
        resultado = c4.selectbox("Resultado:", ["Pendente", "Green ✅", "Red ❌", "Void 🔄"])
        
        submitted = st.form_submit_button("💾 Salvar", use_container_width=True)
        
        if submitted:
            if not descricao:
                st.error("❌ Preencha a descrição")
            else:
                # Calcular lucro
                if resultado == "Green ✅":
                    lucro = (stake * odd) - stake
                    ganhou = True
                elif resultado == "Red ❌":
                    lucro = -stake
                    ganhou = False
                else:  # Void
                    lucro = 0
                    ganhou = False
                
                # Salvar
                st.session_state.bet_results.append({
                    'data': datetime.now().strftime('%d/%m/%Y %H:%M'),
                    'descricao': descricao,
                    'stake': stake,
                    'odd': odd,
                    'resultado': resultado,
                    'ganhou': ganhou,
                    'lucro': lucro
                })
                
                # Atualizar banca
                st.session_state.bankroll_history.append(
                    st.session_state.bankroll_history[-1] + lucro
                )
                
                st.success("✅ Aposta registrada com sucesso!")
                st.rerun()
    
    # Mostrar histórico
    if st.session_state.bet_results:
        st.markdown("---")
        st.markdown("### 📊 Histórico")
        
        df = pd.DataFrame(st.session_state.bet_results)
        df_show = df.copy()
        df_show['stake'] = df_show['stake'].apply(format_currency)
        df_show['lucro'] = df_show['lucro'].apply(format_currency)
        
        st.dataframe(df_show, use_container_width=True)
        
        # Botão de limpar
        if st.button("🗑️ Limpar Histórico"):
            st.session_state.bet_results = []
            st.session_state.bankroll_history = [1000.0]
            st.rerun()

def render_tab_scanner(stats_db: Dict, cal: pd.DataFrame):
    """TAB 7: Scanner Inteligente"""
    st.header("🔍 Scanner de Oportunidades")
    
    if cal.empty:
        st.warning("⚠️ Calendário não disponível")
        return
    
    c1, c2 = st.columns(2)
    
    dates = sorted(cal['DtObj'].dt.strftime('%d/%m/%Y').unique()) if 'DtObj' in cal.columns else []
    
    if not dates:
        st.error("❌ Datas não disponíveis no calendário")
        return
    
    data_scan = c1.selectbox("📅 Data para Escanear:", dates)
    prob_min = c2.slider("🎯 Probabilidade Mínima:", 50, 90, 70)
    
    if st.button("🔎 Escanear Jogos", use_container_width=True):
        jogos_dia = cal[cal['DtObj'].dt.strftime('%d/%m/%Y') == data_scan]
        
        st.markdown(f"### 📊 Escaneando {len(jogos_dia)} jogos...")
        
        resultados = []
        
        for _, row in jogos_dia.iterrows():
            jogo = processar_jogo_calendario(row, stats_db)
            
            if jogo:
                calc = jogo['calc']
                
                # Verificar múltiplos mercados
                mercados_teste = [
                    ('Over 9.5 Cantos', calc['corners_total'], 9.5),
                    ('Over 10.5 Cantos', calc['corners_total'], 10.5),
                    ('Over 2.5 Gols', calc['goals_total'], 2.5),
                    ('Over 4.5 Cartões', calc['cards_total'], 4.5)
                ]
                
                for merc_nome, media, linha in mercados_teste:
                    prob = calcular_poisson(media, linha)
                    
                    if prob >= prob_min:
                        resultados.append({
                            'Jogo': f"{jogo['home']} vs {jogo['away']}",
                            'Hora': jogo['hora'],
                            'Mercado': merc_nome,
                            'Prob': prob,
                            'Emoji': get_prob_emoji(prob)
                        })
        
        if resultados:
            st.success(f"✅ {len(resultados)} oportunidades encontradas!")
            
            df_result = pd.DataFrame(resultados).sort_values('Prob', ascending=False)
            df_result['Prob'] = df_result['Prob'].apply(lambda x: f"{x:.1f}%")
            
            st.dataframe(
                df_result[['Emoji', 'Jogo', 'Hora', 'Mercado', 'Prob']],
                use_container_width=True
            )
        else:
            st.warning(f"⚠️ Nenhuma oportunidade com probabilidade ≥ {prob_min}%")

def render_tab_importar(stats_db: Dict):
    """TAB 8: Importar Bilhete"""
    st.header("📋 Importar Bilhete de Texto")
    
    st.markdown("""
    Cole o texto do seu bilhete no formato:
    ```
    Arsenal vs Chelsea
    Liverpool x Manchester United
    Real Madrid vs Barcelona
    ```
    """)
    
    texto = st.text_area("Cole seu bilhete aqui:", height=200)
    
    if st.button("🔍 Analisar", use_container_width=True):
        if not texto:
            st.error("❌ Cole o texto do bilhete")
            return
        
        # Extrair jogos
        lines = texto.split('\n')
        jogos_encontrados = []
        
        for line in lines:
            if ' vs ' in line.lower() or ' x ' in line.lower():
                # Substituir separadores
                line_clean = line.replace(' x ', ' vs ').replace(' X ', ' vs ')
                
                if ' vs ' in line_clean:
                    parts = line_clean.split(' vs ')
                    if len(parts) == 2:
                        jogos_encontrados.append({
                            'home': parts[0].strip(),
                            'away': parts[1].strip()
                        })
        
        if not jogos_encontrados:
            st.warning("⚠️ Nenhum jogo identificado")
            return
        
        st.success(f"✅ {len(jogos_encontrados)} jogos identificados")
        
        # Processar cada jogo
        for jogo in jogos_encontrados:
            h = normalize_name(jogo['home'], list(stats_db.keys()))
            a = normalize_name(jogo['away'], list(stats_db.keys()))
            
            if h and a:
                calc = calcular_jogo_v31(stats_db[h], stats_db[a], {})
                
                with st.expander(f"✅ {h} vs {a}"):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Cantos", f"{calc['corners_total']:.1f}")
                    c2.metric("Gols", f"{calc['goals_total']:.1f}")
                    c3.metric("Cartões", f"{calc['cards_total']:.1f}")
            else:
                st.error(f"❌ Times não encontrados: {jogo['home']} vs {jogo['away']}")

def render_tab_ai_advisor(stats_db: Dict, cal: pd.DataFrame):
    """TAB 9: AI Advisor"""
    st.header("🤖 AI Advisor ULTRA")
    st.caption("Powered by Causality Engine V31")
    
    # Container de chat
    chat_container = st.container()
    
    with chat_container:
        if not st.session_state.chat_history:
            st.info("👋 Olá! Pergunte sobre confrontos, times ou jogos de hoje.")
        
        for msg in st.session_state.chat_history:
            role = msg['role']
            avatar = "👤" if role == 'user' else "🤖"
            st.chat_message(role, avatar=avatar).markdown(msg['content'])
    
    # Input de chat
    prompt = st.chat_input("Digite sua pergunta...")
    
    if prompt:
        # Adicionar mensagem do usuário
        st.session_state.chat_history.append({'role': 'user', 'content': prompt})
        
        # Processar resposta
        with st.spinner("🧠 Analisando..."):
            # Resposta simples por enquanto
            resposta = f"Você perguntou: '{prompt}'\n\nFuncionalidade de AI Advisor em desenvolvimento."
        
        st.session_state.chat_history.append({'role': 'assistant', 'content': resposta})
        st.rerun()

def render_tab_blacklist():
    """TAB 10: Blacklist"""
    st.header("⛔ Blacklist Científica V32")
    st.caption("Times com médias muito baixas - Evite apostar OVER")
    
    tb1, tb2 = st.tabs(["📉 Cantos", "🟨 Cartões"])
    
    with tb1:
        if BLACKLIST_CORNERS:
            df = pd.DataFrame(
                list(BLACKLIST_CORNERS.items()),
                columns=['Time', 'Média Cantos/Jogo']
            ).sort_values('Média Cantos/Jogo')
            
            st.dataframe(df, use_container_width=True)
            st.caption(f"📊 {len(BLACKLIST_CORNERS)} times monitorados")
        else:
            st.info("Lista vazia")
    
    with tb2:
        if BLACKLIST_CARDS:
            df = pd.DataFrame(
                list(BLACKLIST_CARDS.items()),
                columns=['Time', 'Média Cartões/Jogo']
            ).sort_values('Média Cartões/Jogo')
            
            st.dataframe(df, use_container_width=True)
            st.caption(f"📊 {len(BLACKLIST_CARDS)} times monitorados")
        else:
            st.info("Lista vazia")

# ==============================================================================
# MAIN (SIMPLIFICADA)
# ==============================================================================

def main():
    """Função principal simplificada"""
    
    # 1. Carregar dados
    STATS, CAL, REFS = load_all_data()
    
    # 2. Inicializar estado
    initialize_session_state()
    
    # 3. UI
    render_sidebar(STATS, CAL, REFS)
    render_header(CAL)
    
    # 4. Tabs
    tabs = st.tabs([
        "🎫 Construtor", "🛡️ Hedges", "🎲 Simulador", "📊 Métricas",
        "🎨 Viz", "📝 Registro", "🔍 Scanner", "📋 Importar",
        "🤖 AI", "⛔ Blacklist"
    ])
    
    with tabs[0]:
        render_tab_construtor(STATS, CAL)
    
    with tabs[1]:
        render_tab_hedges(STATS)
    
    with tabs[2]:
        render_tab_simulador(STATS)
    
    with tabs[3]:
        render_tab_metricas()
    
    with tabs[4]:
        render_tab_visualizacoes(STATS)
    
    with tabs[5]:
        render_tab_registro()
    
    with tabs[6]:
        render_tab_scanner(STATS, CAL)
    
    with tabs[7]:
        render_tab_importar(STATS)
    
    with tabs[8]:
        render_tab_ai_advisor(STATS, CAL)
    
    with tabs[9]:
        render_tab_blacklist()

if __name__ == "__main__":
    main()
