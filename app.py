"""
FutPrevisao V33.1 GITHUB-SAFE
Versao sem caracteres Unicode problematicos para GitHub

Autor: Diego & Claude AI
Versao: 33.1 GITHUB-SAFE
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
    logger.warning("Scipy nao disponivel. Usando fallback para Poisson.")

# ==============================================================================
# CONFIGURACOES E CONSTANTES
# ==============================================================================

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="FutPrevisao V33.1",
    layout="wide",
    page_icon=":soccer:",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# ESTILIZACAO CSS
# ==============================================================================

st.markdown('''
<style>
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
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
</style>
''', unsafe_allow_html=True)

# ==============================================================================
# CONSTANTES DO MOTOR V31
# ==============================================================================

HOME_ADVANTAGE_MULTIPLIER = 1.10
AWAY_PENALTY_MULTIPLIER = 0.90
LEAGUE_AVERAGE_GOALS = 1.35

PRESSURE_HIGH_THRESHOLD = 6.0
PRESSURE_MED_THRESHOLD = 4.5
VIOLENCE_HIGH_THRESHOLD = 12.5
REF_STRICT_THRESHOLD = 4.5

# ==============================================================================
# BLACKLIST V32
# ==============================================================================

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

# ==============================================================================
# MAPEAMENTO DE NOMES
# ==============================================================================

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

# ==============================================================================
# MERCADOS DISPONIVEIS
# ==============================================================================

MERCADOS_DISPONIVEIS = [
    "Selecione...",
    "Over 7.5 Cantos Total", "Over 8.5 Cantos Total", "Over 9.5 Cantos Total",
    "Over 10.5 Cantos Total", "Over 11.5 Cantos Total", "Over 12.5 Cantos Total",
    "Over 3.5 Cantos Casa", "Over 4.5 Cantos Casa", "Over 5.5 Cantos Casa",
    "Over 2.5 Cantos Fora", "Over 3.5 Cantos Fora", "Over 4.5 Cantos Fora",
    "Over 2.5 Cartoes Total", "Over 3.5 Cartoes Total", "Over 4.5 Cartoes Total",
    "Over 5.5 Cartoes Total", "Over 1.5 Cartoes Casa", "Over 2.5 Cartoes Casa",
    "Over 1.5 Cartoes Fora", "Over 2.5 Cartoes Fora",
    "Over 0.5 Gols", "Over 1.5 Gols", "Over 2.5 Gols", "Over 3.5 Gols",
    "Under 2.5 Gols", "Under 1.5 Gols", "Ambos Marcam (BTTS)",
    "Vitoria Casa", "Vitoria Fora", "Empate",
    "Dupla Chance Casa/Empate", "Dupla Chance Fora/Empate"
]

# ==============================================================================
# CLASSES
# ==============================================================================

class ChatMemory:
    """Gerencia memoria de curto prazo do chatbot"""
    def __init__(self):
        self.context = {
            'ultimo_time': None,
            'ultimo_jogo': None,
            'historico_analises': []
        }

    def update(self, key: str, value: Any):
        self.context[key] = value
        if key == 'analise':
            self.context['historico_analises'].append(value)

    def get(self, key: str) -> Any:
        return self.context.get(key)

    def clear(self):
        self.context = {
            'ultimo_time': None,
            'ultimo_jogo': None,
            'historico_analises': []
        }

# ==============================================================================
# FUNCOES AUXILIARES
# ==============================================================================

def find_file(filename: str) -> Optional[str]:
    """Busca arquivo em multiplos diretorios"""
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
    
    logger.warning(f"Arquivo nao encontrado: {filename}")
    return None

def normalize_name(name: str, known_teams: List[str]) -> Optional[str]:
    """Normaliza nome de time com fuzzy matching"""
    if not name or not known_teams:
        return None
    
    name = str(name).strip()
    
    if name in NAME_MAPPING:
        target = NAME_MAPPING[name]
        if target in known_teams:
            return target
        name = target
    
    if name in known_teams:
        return name
    
    matches = get_close_matches(name, known_teams, n=1, cutoff=0.6)
    return matches[0] if matches else None

def format_currency(value: float) -> str:
    """Formata valor em reais"""
    return f"R$ {value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def get_prob_emoji(prob: float) -> str:
    """Retorna emoji baseado na probabilidade (SAFE)"""
    if prob >= 80: return "[HOT]"
    elif prob >= 70: return "[OK]"
    elif prob >= 60: return "[WARN]"
    else: return "[LOW]"

def get_league_emoji(league: str) -> str:
    """Retorna emoji da liga (SAFE)"""
    return "[BALL]"  # Usar texto ao inves de emoji para GitHub

def validar_odd(odd: float) -> Tuple[bool, str]:
    """Valida odd de aposta"""
    if odd < 1.01:
        return False, "Odd muito baixa (min: 1.01)"
    elif odd > 50.0:
        return False, "Odd muito alta (max: 50.0)"
    elif odd < 1.10:
        return True, "Odd baixa - Pouco valor"
    return True, "OK"

def validar_stake(stake: float, banca: float, max_percent: float = 10.0) -> Tuple[bool, str]:
    """Valida stake de aposta"""
    if stake <= 0:
        return False, "Stake deve ser maior que zero"
    
    percent = (stake / banca * 100) if banca > 0 else 0
    
    if percent > max_percent:
        return False, f"Stake muito alto! ({percent:.1f}% da banca, max {max_percent}%)"
    elif percent > 5.0:
        return True, f"Stake agressivo ({percent:.1f}%)"
    return True, f"Stake seguro ({percent:.1f}%)"

# ==============================================================================
# CARREGAMENTO DE DADOS
# ==============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def load_all_data() -> Tuple[Dict, pd.DataFrame, Dict]:
    """Carrega todos os dados com tratamento robusto de erros"""
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
    
    for league_name, filename in league_files.items():
        filepath = find_file(filename)
        
        if not filepath:
            logger.warning(f"{league_name}: Arquivo nao encontrado")
            st.sidebar.warning(f"[!] {league_name}: Arquivo nao encontrado")
            continue
        
        try:
            df = pd.read_csv(filepath, encoding='utf-8-sig')
            df.columns = [c.strip() for c in df.columns]
            
            required_cols = ['HomeTeam', 'AwayTeam']
            missing_cols = [c for c in required_cols if c not in df.columns]
            
            if missing_cols:
                raise ValueError(f"Colunas faltando: {missing_cols}")
            
            teams = set(df['HomeTeam'].dropna().unique()) | set(df['AwayTeam'].dropna().unique())
            
            for team in teams:
                if pd.isna(team):
                    continue
                
                h_games = df[df['HomeTeam'] == team]
                a_games = df[df['AwayTeam'] == team]
                
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
            logger.info(f"[OK] {league_name}: {len(teams)} times carregados")
            st.sidebar.success(f"[OK] {league_name}: {len(teams)} times")
            
        except FileNotFoundError:
            st.sidebar.error(f"[X] {league_name}: Arquivo nao encontrado")
            logger.error(f"{league_name}: Arquivo nao encontrado")
            
        except pd.errors.EmptyDataError:
            st.sidebar.error(f"[X] {league_name}: Arquivo vazio")
            logger.error(f"{league_name}: Arquivo vazio")
            
        except ValueError as e:
            st.sidebar.error(f"[X] {league_name}: {str(e)}")
            logger.error(f"{league_name}: {str(e)}")
            
        except Exception as e:
            st.sidebar.error(f"[X] {league_name}: Erro inesperado")
            logger.exception(f"{league_name}: {str(e)}")
    
    # Carregar calendario
    cal_path = find_file('calendario_ligas.csv')
    if cal_path:
        try:
            cal = pd.read_csv(cal_path, encoding='utf-8-sig')
            if 'Data' in cal.columns:
                cal['DtObj'] = pd.to_datetime(cal['Data'], dayfirst=True, errors='coerce')
            logger.info(f"[OK] Calendario: {len(cal)} jogos")
            st.sidebar.success(f"[OK] Calendario: {len(cal)} jogos")
        except Exception as e:
            st.sidebar.warning(f"[!] Calendario: {str(e)}")
            logger.error(f"Calendario: {str(e)}")
    
    # Carregar arbitros
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
            logger.info(f"[OK] Arbitros: {len(referees)}")
            st.sidebar.success(f"[OK] Arbitros: {len(referees)}")
        except Exception as e:
            st.sidebar.warning(f"[!] Arbitros: {str(e)}")
            logger.error(f"Arbitros: {str(e)}")
    
    if loaded_count > 0:
        st.sidebar.info(f"[i] {len(stats_db)} times carregados de {loaded_count} ligas")
    else:
        st.sidebar.error("[X] Nenhuma liga foi carregada!")
    
    return stats_db, cal, referees

# ==============================================================================
# MOTOR DE CALCULO V31
# ==============================================================================

def calcular_poisson(media: float, linha: float) -> float:
    """Calcula probabilidade de OVER usando distribuicao Poisson"""
    if media <= 0:
        return 0.0
    
    if SCIPY_AVAILABLE:
        try:
            return (1 - poisson.cdf(int(linha), media)) * 100
        except Exception as e:
            logger.warning(f"Erro no scipy poisson: {e}")
    
    try:
        k = int(linha)
        prob_at_most_k = sum(
            (math.exp(-media) * (media ** i)) / math.factorial(i)
            for i in range(k + 1)
        )
        return (1 - prob_at_most_k) * 100
    except Exception as e:
        logger.error(f"Erro no calculo Poisson: {e}")
        return 0.0

def calcular_jogo_v31(home_stats: Dict, away_stats: Dict, ref_data: Dict) -> Dict:
    """Motor de Calculo V31 - Causality Engine"""
    if not home_stats or not away_stats:
        return {
            'corners_home': 0, 'corners_away': 0, 'corners_total': 0,
            'cards_home': 0, 'cards_away': 0, 'cards_total': 0,
            'goals_home': 0, 'goals_away': 0, 'goals_total': 0
        }
    
    # Escanteios
    base_corners_h = home_stats.get('corners_home', 5.0)
    base_corners_a = away_stats.get('corners_away', 4.0)
    shots_h = home_stats.get('shots_home', 4.5)
    shots_a = away_stats.get('shots_away', 3.5)
    
    press_h = 1.15 if shots_h > PRESSURE_HIGH_THRESHOLD else \
              1.05 if shots_h > PRESSURE_MED_THRESHOLD else 1.0
    press_a = 1.10 if shots_a > PRESSURE_MED_THRESHOLD else 1.0
    
    corners_h = base_corners_h * press_h * HOME_ADVANTAGE_MULTIPLIER
    corners_a = base_corners_a * press_a * AWAY_PENALTY_MULTIPLIER
    corners_total = corners_h + corners_a
    
    # Cartoes
    fouls_h = home_stats.get('fouls_home', 11.0)
    fouls_a = away_stats.get('fouls_away', 12.0)
    
    viol_h = 1.1 if fouls_h > VIOLENCE_HIGH_THRESHOLD else 1.0
    viol_a = 1.1 if fouls_a > VIOLENCE_HIGH_THRESHOLD else 1.0
    
    ref_avg = ref_data.get('avg_cards', 4.0) if ref_data else 4.0
    cards_h_base = home_stats.get('cards_home', 1.8)
    cards_a_base = away_stats.get('cards_away', 2.2)
    
    cards_h = ((cards_h_base + (ref_avg / 2)) / 2) * viol_h
    cards_a = ((cards_a_base + (ref_avg / 2)) / 2) * viol_a
    cards_total = cards_h + cards_a
    
    # Gols (xG)
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
    """Calcula probabilidade de um mercado (COM VALIDACAO ROBUSTA)"""
    if not mercado or mercado == "Selecione...":
        return 0.0
    
    if not calc or not isinstance(calc, dict):
        logger.warning("Calculos invalidos")
        return 0.0
    
    if "Over" in mercado:
        try:
            linha_match = re.findall(r'\d+\.5', mercado)
            if not linha_match:
                logger.warning(f"Linha nao encontrada em: {mercado}")
                return 0.0
            
            linha = float(linha_match[0])
            
            mapa = {
                'Cantos Casa': 'corners_home',
                'Cantos Fora': 'corners_away',
                'Cantos Total': 'corners_total',
                'Cartoes Casa': 'cards_home',
                'Cartoes Fora': 'cards_away',
                'Cartoes Total': 'cards_total',
                'Gols': 'goals_total'
            }
            
            for chave_merc, chave_calc in mapa.items():
                if chave_merc in mercado:
                    if chave_calc not in calc:
                        logger.error(f"Chave '{chave_calc}' nao encontrada em calc")
                        return 0.0
                    
                    media = calc[chave_calc]
                    return calcular_poisson(media, linha)
            
            logger.warning(f"Mercado nao mapeado: {mercado}")
            return 0.0
            
        except (IndexError, ValueError) as e:
            logger.error(f"Erro ao processar mercado '{mercado}': {e}")
            return 0.0
    
    return 0.0

# AVISO: Versao simplificada para evitar problemas no GitHub
# Para versao completa com todas as tabs, use futprevisao_v33_1_REFATORADO.py

def main():
    """Funcao principal"""
    st.title("FutPrevisao V33.1 - GITHUB SAFE")
    st.caption("Versao sem caracteres Unicode problematicos")
    
    STATS, CAL, REFS = load_all_data()
    
    st.write(f"Times carregados: {len(STATS)}")
    st.write(f"Jogos no calendario: {len(CAL)}")
    st.write(f"Arbitros: {len(REFS)}")
    
    st.info("Esta e uma versao simplificada. Para versao completa, use futprevisao_v33_1_REFATORADO.py")

if __name__ == "__main__":
    main()
