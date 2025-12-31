"""
FutPrevisão V33.3 MAXIMUM + ROBÔ DE TIPS
CÓDIGO COMPLETO - VERSÃO FINAL INTEGRADA
Baseado no Relatório Técnico: Causality Engine, Monte Carlo & NLP + Blacklist + Robô Diário

Novidades V33.3:
- Nova Aba 11: Robô de Tips Diárias
- Analisa jogos de uma data específica
- Sugere APOSTAR (Alta volatilidade) ou EVITAR (Jogo morno)

Autor: Diego & Equipe AI
Versão: 33.3 ULTRA MAXIMUM
Data: 01/01/2026
"""

# ==============================================================================
# 1. IMPORTAÇÕES E CONFIGURAÇÕES INICIAIS
# ==============================================================================

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
import time
import random

# Configuração para Scipy (Matemática Avançada)
try:
    from scipy.stats import poisson, norm
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Diretório base do projeto
BASE_DIR = Path(__file__).resolve().parent

# Configuração da Página Streamlit
st.set_page_config(
    page_title="FutPrevisão V33.3 MAX",
    layout="wide",
    page_icon="⚽",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 2. ESTILIZAÇÃO CSS PROFISSIONAL (GLASSMORPHISM & GRADIENTES)
# ==============================================================================

st.markdown('''
<style>
    /* ESTILO GERAL ULTRA PROFISSIONAL */
    
    /* ANIMAÇÕES */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* TABS DE NAVEGAÇÃO - GRADIENTE MODERNO */
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
    
    /* CHATBOT AI ADVISOR - ESTILO BUBBLE */
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 0px 18px 18px 18px;
        padding: 20px;
        border-left: 5px solid #1e3c72;
        box-shadow: 2px 4px 12px rgba(0,0,0,0.08);
        margin-bottom: 12px;
        color: #2c3e50;
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
        color: #0f172a;
        animation: fadeIn 0.5s ease;
    }
    
    /* CARDS E MÉTRICAS - EFEITO GLASS */
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
    
    /* BOTÕES MODERNOS */
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
    
    /* EXPANDERS */
    .streamlit-expanderHeader {
        background-color: #f8f9fa;
        border-radius: 8px;
        font-weight: 600;
        color: #1e3c72;
    }
    
    /* HEADER PRINCIPAL */
    h1 {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        letter-spacing: -1px;
    }
    
    /* ALERTS */
    .stAlert {
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
</style>
''', unsafe_allow_html=True)

# ==============================================================================
# 3. MAPEAMENTO, CONSTANTES E LISTAS
# ==============================================================================

# Mapeamento expandido de nomes de times para normalização
NAME_MAPPING = {
    'Man United': 'Manchester United', 'Man Utd': 'Manchester United', 'Manchester Utd': 'Manchester United',
    'Man City': 'Manchester City', 
    'Spurs': 'Tottenham', 'Tottenham Hotspur': 'Tottenham',
    'Wolves': 'Wolverhampton', 'Wolverhampton Wanderers': 'Wolverhampton',
    'Paris SG': 'PSG', 'Paris Saint-Germain': 'PSG',
    'Nottm Forest': 'Nottingham Forest', 'Nottingham': 'Nottingham Forest',
    'Sheffield Utd': 'Sheffield United',
    'Newcastle': 'Newcastle United',
    'Brighton': 'Brighton & Hove Albion',
    'West Ham': 'West Ham United',
    'Inter': 'Inter Milan', 'Milan': 'AC Milan',
    'Ath Madrid': 'Atletico Madrid', 'Ath Bilbao': 'Athletic Club',
    'Betis': 'Real Betis', 'Sociedad': 'Real Sociedad',
    'Dortmund': 'Borussia Dortmund', 'Leverkusen': 'Bayer Leverkusen',
    'Bayern': 'Bayern Munich', 'Gladbach': 'Borussia Monchengladbach',
    'Frankfurt': 'Eintracht Frankfurt',
    'Marseille': 'Olympique Marseille', 'Lyon': 'Olympique Lyon',
    'Monaco': 'AS Monaco', 'Lille': 'LOSC Lille',
    'Leicester': 'Leicester City', 'Leeds': 'Leeds United'
}

# Lista completa de mercados para o Construtor Manual
MERCADOS_DISPONIVEIS = [
    "Selecione...",
    # === GOLS ===
    "Over 0.5 Gols", "Over 1.5 Gols", "Over 2.5 Gols", "Over 3.5 Gols",
    "Under 2.5 Gols", "Under 1.5 Gols", "Under 3.5 Gols",
    
    # === ESCANTEIOS - INDIVIDUAIS (Casa) ===
    "Over 2.5 Cantos Casa", "Over 3.5 Cantos Casa", "Over 4.5 Cantos Casa", "Over 5.5 Cantos Casa",
    
    # === ESCANTEIOS - INDIVIDUAIS (Fora) ===
    "Over 2.5 Cantos Fora", "Over 3.5 Cantos Fora", "Over 4.5 Cantos Fora", "Over 5.5 Cantos Fora",
    
    # === ESCANTEIOS - TOTAIS ===
    "Over 7.5 Cantos Total", "Over 8.5 Cantos Total", "Over 9.5 Cantos Total", 
    "Over 10.5 Cantos Total", "Over 11.5 Cantos Total", "Over 12.5 Cantos Total",
    
    # === CARTÕES - INDIVIDUAIS (Casa) ===
    "Over 1.5 Cartões Casa", "Over 2.5 Cartões Casa",
    
    # === CARTÕES - INDIVIDUAIS (Fora) ===
    "Over 1.5 Cartões Fora", "Over 2.5 Cartões Fora",
    
    # === CARTÕES - TOTAIS ===
    "Over 2.5 Cartões Total", "Over 3.5 Cartões Total", "Over 4.5 Cartões Total", "Over 5.5 Cartões Total",
    
    # === RESULTADO E OUTROS ===
    "Ambos Marcam (BTTS)", "Vitória Casa", "Vitória Fora", "Empate", 
    "Dupla Chance Casa/Empate", "Dupla Chance Fora/Empate", "Dupla Chance Casa/Fora"
]

# 🆕 BLACKLIST CIENTÍFICA V32 (Times com médias muito baixas - EVITAR OVER)
BLACKLIST_CORNERS = {
    'Wolves': 2.89, 'Sunderland': 3.61, 'Burnley': 3.78, 'Crystal Palace': 3.78,
    'Elche': 3.06, 'Mallorca': 3.29, 'Osasuna': 3.35, 'Levante': 3.38, 
    'Oviedo': 3.82, 'Girona': 3.88,
    'Parma': 3.19, 'Cremonese': 3.24, 'Pisa': 3.35, 'Sassuolo': 3.47,
    'Cagliari': 3.55, 'Udinese': 3.74, 'Empoli': 3.77,
    'Strasbourg': 3.39, 'Lorient': 3.44, 'Metz': 3.50, 'Montpellier': 3.82,
    'Heidenheim': 3.55, 'Darmstadt': 3.61
}

BLACKLIST_CARDS = {
    'Arsenal': 1.45, 'Man City': 1.52, 'Liverpool': 1.63, 'Bayern Munich': 1.35,
    'Dortmund': 1.48, 'Inter Milan': 1.55, 'PSG': 1.42, 'Real Madrid': 1.60
}

# Parâmetros do Causality Engine V31
PRESSURE_HIGH_THRESHOLD = 6.0  
PRESSURE_MED_THRESHOLD = 4.5   
VIOLENCE_HIGH_THRESHOLD = 12.5 
REF_STRICT_THRESHOLD = 4.5     

# ==============================================================================
# 4. FUNÇÕES AUXILIARES E UTILITÁRIOS
# ==============================================================================

class ChatMemory:
    """Classe para gerenciar a memória de curto prazo do Chatbot"""
    def __init__(self):
        self.context = {
            'ultimo_time': None,
            'ultimo_jogo': None,
            'ultima_prob': None,
            'ultima_odd': None,
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
            'ultima_prob': None,
            'ultima_odd': None,
            'historico_analises': []
        }

def find_file(filename: str) -> Optional[str]:
    """Busca robusta de arquivos em múltiplos diretórios"""
    search_paths = [
        Path('/mnt/project') / filename,
        Path('.') / filename,
        Path('./data') / filename,
        BASE_DIR / filename,
        BASE_DIR / 'data' / filename
    ]
    for path in search_paths:
        if path.exists():
            return str(path)
    return None

def normalize_name(name: str, known_teams: List[str]) -> Optional[str]:
    """Normaliza nomes de times usando fuzzy matching e mapeamento"""
    if not name or not known_teams: return None
    name = str(name).strip()
    
    if name in NAME_MAPPING:
        target = NAME_MAPPING[name]
        if target in known_teams: return target
        name = target
        
    if name in known_teams: return name
    
    matches = get_close_matches(name, known_teams, n=1, cutoff=0.6)
    return matches[0] if matches else None

def clean_team_name(text: str) -> str:
    """Limpa nome de time vindo do chat NLP"""
    if not text: return ""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    stop_words = {'do', 'da', 'de', 'dos', 'das', 'o', 'a', 'os', 'as', 
                  'como', 'está', 'esta', 'stats', 'estatistica', 'vs', 'x', 
                  'contra', 'analise', 'analisar', 'previsao', 'jogo', 'hoje'}
    return ' '.join([w for w in text.split() if w not in stop_words])

def format_currency(value: float) -> str:
    return f"R$ {value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def get_prob_emoji(prob: float) -> str:
    if prob >= 80: return "🔥"
    elif prob >= 70: return "✅"
    elif prob >= 60: return "⚠️"
    else: return "🔻"

def get_league_emoji(league: str) -> str:
    # SUBSTITUÍDO BANDEIRAS POR TEXTO SEGURO
    emojis = {
        'Premier League': '[ENG]', 
        'La Liga': '[ESP]', 
        'Serie A': '[ITA]',
        'Bundesliga': '[GER]', 
        'Ligue 1': '[FRA]', 
        'Championship': '[ENG2]',
        'Bundesliga 2': '[GER2]', 
        'Pro League': '[BEL]', 
        'Super Lig': '[TUR]',
        'Premiership': '[SCO]'
    }
    return emojis.get(league, '⚽')

def validar_odd(odd: float) -> Tuple[bool, str]:
    if odd < 1.01:
        return False, "Odd muito baixa (mín: 1.01)"
    elif odd > 50.0:
        return False, "Odd muito alta (máx: 50.0)"
    elif odd < 1.10:
        return True, "⚠️ Odd baixa - Pouco valor"
    else:
        return True, "OK"

def validar_stake(stake: float, banca: float, max_percent: float = 10.0) -> Tuple[bool, str]:
    if stake <= 0:
        return False, "Stake deve ser maior que zero"
    percent = (stake / banca * 100) if banca > 0 else 0
    if percent > max_percent:
        return False, f"⚠️ Stake muito alto! ({percent:.1f}% da banca, máx {max_percent}%)"
    elif percent > 5.0:
        return True, f"⚠️ Stake agressivo ({percent:.1f}%)"
    else:
        return True, f"✅ Stake seguro ({percent:.1f}%)"

# ==============================================================================
# 5. CARREGAMENTO E PROCESSAMENTO DE DADOS (ETL)
# ==============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def load_all_data():
    """
    Carrega dados e processa estatísticas (Stats DB).
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
    
    for league_name, filename in league_files.items():
        filepath = find_file(filename)
        if not filepath: continue
            
        try:
            # CORREÇÃO V32: utf-8-sig
            df = pd.read_csv(filepath, encoding='utf-8-sig')
            cols = {c: c.strip() for c in df.columns}
            df.rename(columns=cols, inplace=True)
            
            teams = set(df['HomeTeam'].dropna().unique()) | set(df['AwayTeam'].dropna().unique())
            
            for team in teams:
                if pd.isna(team): continue
                
                h_games = df[df['HomeTeam'] == team]
                a_games = df[df['AwayTeam'] == team]
                
                # --- MÉTRICAS ---
                corners_h = h_games['HC'].mean() if 'HC' in h_games else 5.0
                corners_a = a_games['AC'].mean() if 'AC' in a_games else 4.0
                
                ch = (h_games['HY'].mean() + h_games['HR'].mean()*2) if 'HY' in h_games else 1.8
                ca = (a_games['AY'].mean() + a_games['AR'].mean()*2) if 'AY' in a_games else 2.2
                
                fouls_h = h_games['HF'].mean() if 'HF' in h_games else 11.5
                fouls_a = a_games['AF'].mean() if 'AF' in a_games else 12.5
                
                goals_fh = h_games['FTHG'].mean() if 'FTHG' in h_games else 1.4
                goals_fa = a_games['FTAG'].mean() if 'FTAG' in a_games else 1.1
                goals_ah = h_games['FTAG'].mean() if 'FTAG' in h_games else 1.0
                goals_aa = a_games['FTHG'].mean() if 'FTHG' in a_games else 1.5
                
                shots_h = h_games['HST'].mean() if 'HST' in h_games else 4.8
                shots_a = a_games['AST'].mean() if 'AST' in a_games else 3.8
                
                stats_db[team] = {
                    'league': league_name,
                    'corners': (corners_h + corners_a) / 2, 'corners_home': corners_h, 'corners_away': corners_a,
                    'cards': (ch + ca) / 2, 'cards_home': ch, 'cards_away': ca,
                    'fouls': (fouls_h + fouls_a) / 2, 'fouls_home': fouls_h, 'fouls_away': fouls_a,
                    'goals_f': (goals_fh + goals_fa) / 2, 'goals_f_home': goals_fh, 'goals_f_away': goals_fa,
                    'goals_a': (goals_ah + goals_aa) / 2, 'goals_a_home': goals_ah, 'goals_a_away': goals_aa,
                    'shots_on_target': (shots_h + shots_a) / 2, 'shots_home': shots_h, 'shots_away': shots_a,
                    'games_played': len(h_games) + len(a_games)
                }
        except Exception: pass

    # Calendário
    cal_path = find_file('calendario_ligas.csv')
    if cal_path:
        try:
            cal = pd.read_csv(cal_path, encoding='utf-8-sig')
            if 'Data' in cal.columns:
                cal['DtObj'] = pd.to_datetime(cal['Data'], dayfirst=True, errors='coerce')
        except: pass
    
    # Árbitros
    ref_path = find_file('arbitros_5_ligas_2025_2026.csv')
    if ref_path:
        try:
            refs_df = pd.read_csv(ref_path, encoding='utf-8-sig')
            for _, row in refs_df.iterrows():
                avg = row.get('Media_Cartoes_Por_Jogo', 4.0)
                referees[row['Arbitro']] = {
                    'factor': avg / 4.0, 'avg_cards': avg,
                    'games': row.get('Jogos_Apitados', 0),
                    'red_rate': row.get('Cartoes_Vermelhos', 0) / (row.get('Jogos_Apitados', 1) or 1)
                }
        except: pass
            
    return stats_db, cal, referees

@st.cache_data(ttl=3600)
def carregar_dados_robo():
    """
    Função dedicada ao Robô: Carrega todos os CSVs e retorna um único DataFrame bruto com coluna 'Date'.
    """
    arquivos = [
        'Championship_Inglaterra_25_26.csv', 'Premier_League_25_26.csv',
        'La_Liga_25_26.csv', 'Serie_A_25_26.csv', 'Pro_League_Belgica_25_26.csv',
        'Super_Lig_Turquia_25_26.csv', 'Bundesliga_2.csv', 'Ligue_1_25_26.csv',
        'Bundesliga_25_26.csv', 'Premiership_Escocia_25_26.csv'
    ]
    
    dfs = []
    for arq in arquivos:
        filepath = find_file(arq)
        if not filepath: continue
        try:
            df = pd.read_csv(filepath, encoding='utf-8-sig')
            df.columns = [c.strip() for c in df.columns]
            df['Liga'] = arq.replace('.csv', '').replace('_25_26', '') # Nome simples da liga
            dfs.append(df)
        except: pass
    
    if not dfs: return pd.DataFrame()
    
    df_final = pd.concat(dfs, ignore_index=True)
    
    # Converter data com segurança
    if 'Date' in df_final.columns:
        df_final['Date'] = pd.to_datetime(df_final['Date'], dayfirst=True, errors='coerce')
    
    return df_final

# ==============================================================================
# 6. MOTOR DE CÁLCULO E SIMULAÇÃO (V31 ENGINE)
# ==============================================================================

def calcular_poisson(media: float, linha: float) -> float:
    """Calcula probabilidade de OVER usando Poisson"""
    if media <= 0: return 0.0
    if SCIPY_AVAILABLE:
        try:
            return (1 - poisson.cdf(int(linha), media)) * 100
        except: pass
    # Fallback
    prob_exact = 0
    k = int(linha)
    for i in range(k + 1):
        prob_exact += (math.exp(-media) * (media ** i)) / math.factorial(i)
    return (1 - prob_exact) * 100

def calcular_jogo_v31(home_stats: Dict, away_stats: Dict, ref_data: Dict) -> Dict:
    """
    CAUSALITY ENGINE V31 - NÚCLEO DO SISTEMA
    Implementa a lógica 'Causa -> Efeito'.
    RETORNA TODAS AS CHAVES PLANAS PARA EVITAR KEYERROR.
    """
    if not home_stats or not away_stats:
        return {'corners': {'h':0,'a':0,'t':0}, 'cards': {'h':0,'a':0,'t':0}, 'goals': {'h':0,'a':0}, 
                'corners_total':0, 'total_goals':0, 'cards_total':0, 
                'corners_home': 0, 'corners_away': 0, 'cards_home': 0, 'cards_away': 0,
                'xg_home':0, 'xg_away':0}

    # Escanteios
    base_h = home_stats.get('corners_home', 5.0)
    base_a = away_stats.get('corners_away', 4.0)
    shots_h = home_stats.get('shots_home', 4.5)
    shots_a = away_stats.get('shots_away', 3.5)
    
    press_h = 1.15 if shots_h > PRESSURE_HIGH_THRESHOLD else 1.05 if shots_h > PRESSURE_MED_THRESHOLD else 1.0
    press_a = 1.10 if shots_a > PRESSURE_MED_THRESHOLD else 1.0
    
    corners_h = base_h * press_h * 1.10
    corners_a = base_a * press_a * 0.90
    corners_total = corners_h + corners_a
    
    # Cartões
    fouls_h = home_stats.get('fouls_home', 11.0)
    fouls_a = away_stats.get('fouls_away', 12.0)
    viol_h = 1.1 if fouls_h > VIOLENCE_HIGH_THRESHOLD else 1.0
    viol_a = 1.1 if fouls_a > VIOLENCE_HIGH_THRESHOLD else 1.0
    
    ref_avg = ref_data.get('avg_cards', 4.0) if ref_data else 4.0
    cards_h_base = home_stats.get('cards_home', 1.8)
    cards_a_base = away_stats.get('cards_away', 2.2)
    
    cards_h_proj = ((cards_h_base + (ref_avg/2)) / 2) * viol_h
    cards_a_proj = ((cards_a_base + (ref_avg/2)) / 2) * viol_a
    cards_total = cards_h_proj + cards_a_proj
    
    # Gols
    league_avg = 1.35
    xg_h = (home_stats['goals_f_home'] / league_avg) * (away_stats['goals_a_away'] / league_avg) * league_avg
    xg_a = (away_stats['goals_f_away'] / league_avg) * (home_stats['goals_a_home'] / league_avg) * league_avg
    total_goals = xg_h + xg_a
    
    # RETORNO ROBUSTO E PLANO (FIX KEYERROR)
    return {
        'corners': {'h': corners_h, 'a': corners_a, 't': corners_total},
        'cards': {'h': cards_h_proj, 'a': cards_a_proj, 't': cards_total},
        'goals': {'h': xg_h, 'a': xg_a},
        'corners_total': corners_total,
        'corners_home': corners_h,
        'corners_away': corners_a,
        'cards_total': cards_total,
        'cards_home': cards_h_proj,
        'cards_away': cards_a_proj,
        'total_goals': total_goals,
        'xg_home': xg_h,
        'xg_away': xg_a
    }

def simulate_game_v31(home_stats: Dict, away_stats: Dict, ref_data: Dict, n_sims: int = 3000) -> Dict:
    """Simulador Monte Carlo (3000 iterações)"""
    calc = calcular_jogo_v31(home_stats, away_stats, ref_data)
    return {
        'corners_total': np.random.poisson(calc['corners_total'], n_sims),
        'cards_total': np.random.poisson(calc['cards_total'], n_sims),
        'goals_h': np.random.poisson(calc['xg_home'], n_sims),
        'goals_a': np.random.poisson(calc['xg_away'], n_sims),
        'goals_total': np.random.poisson(calc['total_goals'], n_sims)
    }

def calcular_probabilidade_mercado(mercado: str, calc: Dict) -> float:
    """Calcula probabilidade com chaves planas (FIX KEYERROR)"""
    if mercado == "Selecione...": return 0.0
    
    if "Over" in mercado:
        try:
            linha = float(re.findall(r'\d+\.5', mercado)[0])
            
            if "Cantos Casa" in mercado: return calcular_poisson(calc['corners_home'], linha)
            elif "Cantos Fora" in mercado: return calcular_poisson(calc['corners_away'], linha)
            elif "Cantos Total" in mercado: return calcular_poisson(calc['corners_total'], linha)
                
            elif "Cartões Casa" in mercado: return calcular_poisson(calc['cards_home'], linha)
            elif "Cartões Fora" in mercado: return calcular_poisson(calc['cards_away'], linha)
            elif "Cartões Total" in mercado: return calcular_poisson(calc['cards_total'], linha)
                
            elif "Gols" in mercado: return calcular_poisson(calc['total_goals'], linha)
        except: return 0.0
        
    return 0.0

def sugerir_mercados_hedge_inteligente(bilhete_principal: List[Dict], jogo: str, stats: Dict, calc: Dict) -> List[Dict]:
    """HEDGE SUPER INTELIGENTE (SEM GOLS, PRIORIDADE CANTOS/CARTÕES)"""
    try:
        time_casa, time_fora = jogo.split(' vs ')
    except:
        return []
    
    stats_casa = stats.get(time_casa, {})
    stats_fora = stats.get(time_fora, {})
    
    if not stats_casa or not stats_fora:
        return []
    
    mercados_atuais = [b.get('market_display', b['mercado']) for b in bilhete_principal if b['jogo'] == jogo]
    sugestoes = []
    
    # 1. ESCANTEIOS
    if stats_casa.get('corners_home', 0) > 4.5:
        prob = calcular_poisson(calc['corners_home'], 4.5)
        if prob > 65 and "Over 4.5 Cantos Casa" not in mercados_atuais:
            sugestoes.append({'mercado': "Over 4.5 Cantos Casa", 'prob': prob, 'tipo': 'Cantos', 'analise': f"{time_casa} forte em casa"})
            
    if stats_fora.get('corners_away', 0) > 3.5:
        prob = calcular_poisson(calc['corners_away'], 3.5)
        if prob > 65 and "Over 3.5 Cantos Fora" not in mercados_atuais:
            sugestoes.append({'mercado': "Over 3.5 Cantos Fora", 'prob': prob, 'tipo': 'Cantos', 'analise': f"{time_fora} ativo fora"})
            
    prob_total = calcular_poisson(calc['corners_total'], 8.5)
    if prob_total > 70 and "Over 8.5 Cantos Total" not in mercados_atuais:
        sugestoes.append({'mercado': "Over 8.5 Cantos Total", 'prob': prob_total, 'tipo': 'Cantos', 'analise': "Jogo aberto"})

    # 2. CARTÕES
    prob_card = calcular_poisson(calc['cards_total'], 3.5)
    if prob_card > 70 and "Over 3.5 Cartões Total" not in mercados_atuais:
        sugestoes.append({'mercado': "Over 3.5 Cartões Total", 'prob': prob_card, 'tipo': 'Cartões', 'analise': "Jogo pegado"})

    # 3. DUPLA CHANCE (Só se < 2 sugestões)
    if len(sugestoes) < 2:
        if calc['xg_home'] > calc['xg_away']:
            sugestoes.append({'mercado': "Dupla Chance Casa/Empate", 'prob': 75.0, 'tipo': 'DC', 'analise': "Favorito Casa"})
        else:
            sugestoes.append({'mercado': "Dupla Chance Fora/Empate", 'prob': 70.0, 'tipo': 'DC', 'analise': "Visitante forte"})
            
    sugestoes.sort(key=lambda x: x['prob'], reverse=True)
    return sugestoes[:3]

def calcular_kelly(prob: float, odd: float, fracao: float = 0.25) -> float:
    if prob <= 0 or prob >= 100 or odd <= 1: return 0.0
    p = prob / 100
    q = 1 - p
    b = odd - 1
    return max(0, ((b * p - q) / b) * fracao)

def analisar_robo_diario(df_robo, data_selecionada, stats_db):
    """
    Lógica do Robô Tips do Dia:
    APOSTAR: Cantos Totais > 9.5 E Cartões > 3.5
    EVITAR: Cantos Totais < 8.5 OU Cartões < 2.5
    """
    jogos = df_robo[df_robo['Date'] == pd.to_datetime(data_selecionada)]
    resultados = []
    
    for _, row in jogos.iterrows():
        h = normalize_name(row['HomeTeam'], list(stats_db.keys()))
        a = normalize_name(row['AwayTeam'], list(stats_db.keys()))
        
        if h and a and h in stats_db and a in stats_db:
            # Pega as médias históricas do DB
            s_h = stats_db[h]
            s_a = stats_db[a]
            
            # Soma simples das médias (Projeção Rápida)
            proj_cantos = s_h['corners'] + s_a['corners']
            proj_cartoes = s_h['cards'] + s_a['cards']
            
            # Regras
            recomendacao = "OBSERVAR"
            cor = "orange"
            motivo = "Stats medianos"
            
            if proj_cantos > 9.5 and proj_cartoes > 3.5:
                recomendacao = "🔥 APOSTAR"
                cor = "green"
                motivo = "Alta tendência de Cantos e Cartões"
            elif proj_cantos < 8.5 or proj_cartoes < 2.5:
                recomendacao = "⛔ EVITAR"
                cor = "red"
                motivo = "Baixa probabilidade (Jogo morno)"
            
            resultados.append({
                'Liga': row.get('Liga', '-'),
                'Jogo': f"{h} x {a}",
                'Cantos': proj_cantos,
                'Cartões': proj_cartoes,
                'Rec': recomendacao,
                'Cor': cor,
                'Motivo': motivo
            })
            
    return pd.DataFrame(resultados)

# ==============================================================================
# 7. CHATBOT AI ADVISOR ULTRA (NLP AVANÇADO)
# ==============================================================================

def extrair_entidades(mensagem: str, stats_db: Dict, memoria: ChatMemory) -> Dict:
    msg_lower = mensagem.lower()
    entidades = {'times': [], 'mercado': None, 'linha': None, 'intencao': None}
    
    known = list(stats_db.keys())
    sorted_teams = sorted(known, key=len, reverse=True)
    msg_clean = msg_lower
    
    for team in sorted_teams:
        if team.lower() in msg_clean:
            is_sub = False
            for ft in entidades['times']:
                if team.lower() in ft.lower(): is_sub = True; break
            if not is_sub:
                entidades['times'].append(team)
                msg_clean = msg_clean.replace(team.lower(), "")
                
    if not entidades['times']:
        ult = memoria.get('ultimo_time')
        if ult and ('dele' in msg_lower or 'desse' in msg_lower): entidades['times'].append(ult)
            
    if entidades['times']: memoria.update('ultimo_time', entidades['times'][0])
    
    if any(x in msg_lower for x in ['canto', 'escanteio']): entidades['mercado'] = 'cantos'
    elif any(x in msg_lower for x in ['cartao', 'cartão']): entidades['mercado'] = 'cartoes'
    elif any(x in msg_lower for x in ['gol', 'gols']): entidades['mercado'] = 'gols'
    
    nums = re.findall(r'\d+\.?\d*', msg)
    if nums:
        for n in nums:
            if float(n) < 20: entidades['linha'] = float(n); break
            
    return entidades

def processar_chat_ultra(mensagem: str, stats_db: Dict, cal: pd.DataFrame, refs: Dict, memoria: ChatMemory) -> str:
    if not mensagem: return "Olá! Sou o AI Advisor ULTRA. Como posso ajudar?"
    entidades = extrair_entidades(mensagem, stats_db, memoria)
    times = entidades['times']
    msg_lower = mensagem.lower()
    
    if len(times) >= 2:
        t1, t2 = times[0], times[1]
        s1, s2 = stats_db[t1], stats_db[t2]
        calc = calcular_jogo_v31(s1, s2, {})
        
        prob_g = calcular_poisson(calc['total_goals'], 2.5)
        prob_c = calcular_poisson(calc['corners_total'], 9.5)
        prob_card = calcular_poisson(calc['cards_total'], 4.5)
        
        resp = f"📊 **ANÁLISE: {t1} vs {t2}**\n\n"
        resp += "**🔎 Projeções V31:**\n"
        resp += f"• **Gols (xG):** {calc['total_goals']:.2f}\n"
        resp += f"• **Cantos:** {calc['corners_total']:.1f}\n"
        resp += f"• **Cartões:** {calc['cards_total']:.1f}\n\n"
        
        if ('prob' in msg_lower) and entidades['linha']:
            lin = entidades['linha']
            merc = entidades['mercado'] or 'gols'
            media = calc['corners_total'] if merc=='cantos' else calc['cards_total'] if merc=='cartoes' else calc['total_goals']
            prob_user = calcular_poisson(media, lin)
            resp += f"🎲 **Consulta:** Over {lin} {merc}\n{get_prob_emoji(prob_user)} **Prob:** {prob_user:.1f}%\n"
            return resp
            
        resp += "**💡 Sugestões (EV+):**\n"
        found = False
        if prob_g > 65: resp += f"✅ Over 2.5 Gols ({prob_g:.1f}%)\n"; found=True
        if prob_c > 70: resp += f"✅ Over 9.5 Cantos ({prob_c:.1f}%)\n"; found=True
        if prob_card > 65: resp += f"✅ Over 4.5 Cartões ({prob_card:.1f}%)\n"; found=True
        if not found: resp += "⚠️ Sem valor estatístico claro."
        return resp

    elif len(times) == 1:
        t = times[0]
        s = stats_db[t]
        resp = f"📊 **RAIO-X: {t}**\n_(Liga: {s['league']})_\n\n"
        resp += f"**Ataque:** {s['goals_f']:.2f} gols/jogo\n"
        resp += f"**Defesa:** {s['goals_a']:.2f} sofridos/jogo\n"
        resp += f"**Cantos:** {s['corners']:.2f}/jogo\n"
        resp += f"**Cartões:** {s['cards']:.2f}/jogo\n\n"
        if s['corners'] > 6.0: resp += "🔥 Máquina de Cantos.\n"
        if s['goals_f'] > 1.8: resp += "⚽ Ataque Poderoso.\n"
        return resp

    elif any(x in msg_lower for x in ['melhor', 'hoje', 'jogos', 'calendario']):
        hoje = datetime.now().strftime('%d/%m/%Y')
        jogos = cal[cal['Data'] == hoje] if not cal.empty else pd.DataFrame()
        if jogos.empty: return f"📅 Sem jogos hoje ({hoje})."
        
        ranking = []
        for _, r in jogos.iterrows():
            h, a = normalize_name(r['Time_Casa'], list(stats_db.keys())), normalize_name(r['Time_Visitante'], list(stats_db.keys()))
            if h and a:
                c = calcular_jogo_v31(stats_db[h], stats_db[a], {})
                score = c['total_goals']*2 + c['corners_total']
                ranking.append({'j': f"{h} vs {a}", 's': score, 'd': c})
        
        ranking.sort(key=lambda x: x['s'], reverse=True)
        resp = f"🏆 **TOP JOGOS HOJE ({hoje}):**\n\n"
        for i in ranking[:3]:
            resp += f"**{i['j']}**\n   🎯 Gols: {i['d']['total_goals']:.1f} | Cantos: {i['d']['corners_total']:.1f}\n\n"
        return resp

    else:
        return "🤖 **AI Advisor:** Pergunte sobre confrontos ('Arsenal vs Chelsea'), times ('Real Madrid') ou 'Jogos de hoje'."

# ==============================================================================
# 8. MÉTODOS FINANCEIROS E GRÁFICOS
# ==============================================================================

def calculate_sharpe_ratio(returns: List[float]) -> float:
    if not returns or len(returns) < 2: return 0.0
    return (np.mean(returns) - 1.0) / np.std(returns) if np.std(returns) > 0 else 0.0

def calculate_max_drawdown(history: List[float]) -> float:
    if len(history) < 2: return 0.0
    peak = history[0]
    max_dd = 0.0
    for v in history:
        if v > peak: peak = v
        dd = (peak - v) / peak * 100
        if dd > max_dd: max_dd = dd
    return max_dd

def calculate_roi(investido: float, retorno: float) -> float:
    return (retorno / investido) * 100 if investido > 0 else 0

def parse_bilhete_texto(texto: str) -> List[Dict]:
    jogos = []
    lines = texto.split('\n')
    for line in lines:
        if ' vs ' in line or ' x ' in line:
            parts = re.split(r' vs | x ', line)
            if len(parts) >= 2:
                jogos.append({'home': parts[0].strip(), 'away': parts[1].strip()})
    return jogos

def validar_jogos_bilhete(jogos_parsed: List[Dict], stats_db: Dict) -> List[Dict]:
    validos = []
    known = list(stats_db.keys())
    for j in jogos_parsed:
        h = normalize_name(j['home'], known)
        a = normalize_name(j['away'], known)
        if h and a:
            validos.append({'home': h, 'away': a, 'home_stats': stats_db[h], 'away_stats': stats_db[a]})
    return validos

# ==============================================================================
# 9. UI PRINCIPAL (MAIN)
# ==============================================================================

def main():
    # 1. CARREGAMENTO INICIAL DE DADOS
    STATS, CAL, REFS = load_all_data()
    # Carregar dados do Robô separadamente para ter datas e ligas
    DF_ROBO = carregar_dados_robo()
    
    # 2. Inicialização de Estado
    if 'current_ticket' not in st.session_state: st.session_state.current_ticket = []
    if 'bet_results' not in st.session_state: st.session_state.bet_results = []
    if 'bankroll_history' not in st.session_state: st.session_state.bankroll_history = [1000.0]
    if 'chat_history' not in st.session_state: st.session_state.chat_history = []
    if 'chat_memory' not in st.session_state: st.session_state.chat_memory = ChatMemory()
    
    # 3. SIDEBAR (DASHBOARD)
    with st.sidebar:
        st.header("📊 Dashboard Profissional")
        c1, c2 = st.columns(2)
        c1.metric("Times", len(STATS))
        c2.metric("Jogos DB", len(CAL) if not CAL.empty else 0)
        
        banca = st.session_state.bankroll_history[-1]
        lucro_total = banca - 1000.0
        st.metric("💰 Banca Atual", format_currency(banca), delta=format_currency(lucro_total))
        
        if st.session_state.current_ticket:
            st.success(f"🎫 {len(st.session_state.current_ticket)} apostas")
            if st.button("Limpar Bilhete", use_container_width=True):
                st.session_state.current_ticket = []
                st.rerun()
                
        # Exportação JSON
        st.markdown("---")
        if st.session_state.current_ticket:
            st.download_button("📥 Baixar Bilhete", json.dumps(st.session_state.current_ticket), "ticket.json")

    # 4. HEADER
    col1, col2, col3 = st.columns([1, 5, 2])
    with col1: st.markdown("# ⚽")
    with col2:
        st.title("FutPrevisão V33.3 ULTRA")
        st.markdown("**Professional Sports Analytics System** | _Powered by Causality Engine V31_")
    with col3:
        if not CAL.empty:
            hj = datetime.now().strftime('%d/%m/%Y')
            st.metric("Jogos Hoje", len(CAL[CAL['Data'] == hj]))
    
    st.markdown("---")

    # 5. TABS
    tabs = st.tabs([
        "🎫 Construtor", "🛡️ Hedges", "🎲 Simulador", "📊 Métricas", 
        "🎨 Viz", "📝 Registro", "🔍 Scanner", "📋 Importar", "🤖 AI Advisor", "⛔ Blacklist", "🤖 Robô Tips"
    ])
    
    # ========================================
    # TAB 1: CONSTRUTOR DE BILHETES (HÍBRIDO)
    # ========================================
    with tabs[0]:
        st.subheader("🛠️ Construtor de Bilhetes Profissional")
        
        c_col1, c_col2 = st.columns([1, 1])
        
        # MODO AUTOMÁTICO (CALENDÁRIO)
        with c_col1:
            st.markdown("#### 📅 Seleção Automática")
            if not CAL.empty:
                dates = sorted(CAL['DtObj'].dt.strftime('%d/%m/%Y').unique())
                if dates:
                    data_sel = st.selectbox("📆 Data:", dates)
                    jogos_dia = CAL[CAL['DtObj'].dt.strftime('%d/%m/%Y') == data_sel]
                    
                    if jogos_dia.empty: st.info("Sem jogos.")
                    
                    for idx, row in jogos_dia.iterrows():
                        h, a = normalize_name(row['Time_Casa'], list(STATS.keys())), normalize_name(row['Time_Visitante'], list(STATS.keys()))
                        if h and a:
                            calc = calcular_jogo_v31(STATS[h], STATS[a], {})
                            with st.expander(f"⚽ {h} vs {a} | {row.get('Hora', '-')}"):
                                m1, m2, m3 = st.columns(3)
                                m1.metric("Cantos", f"{calc['corners_total']:.1f}")
                                m2.metric("Gols", f"{calc['total_goals']:.1f}")
                                m3.metric("Cartões", f"{calc['cards_total']:.1f}")
                                
                                b1, b2 = st.columns(2)
                                if b1.button("+ Over 9.5 C", key=f"ac_{idx}"):
                                    prob = calcular_poisson(calc['corners_total'], 9.5)
                                    st.session_state.current_ticket.append({'jogo': f"{h} vs {a}", 'mercado': 'Over 9.5 Cantos Total', 'odd': 1.85, 'prob': prob, 'tipo': 'Auto', 'market_display': 'Over 9.5 Cantos Total'})
                                    st.success("Adicionado!")
                                    st.rerun()
                                if b2.button("+ Over 2.5 G", key=f"ag_{idx}"):
                                    prob = calcular_poisson(calc['total_goals'], 2.5)
                                    st.session_state.current_ticket.append({'jogo': f"{h} vs {a}", 'mercado': 'Over 2.5 Gols', 'odd': 1.90, 'prob': prob, 'tipo': 'Auto', 'market_display': 'Over 2.5 Gols'})
                                    st.success("Adicionado!")
                                    st.rerun()

        # MODO MANUAL (DROPDOWNS)
        with c_col2:
            st.markdown("#### 📝 Manual (Dropdowns)")
            with st.container():
                all_teams = sorted(list(STATS.keys()))
                tc = st.selectbox("🏠 Casa:", ["Selecione..."] + all_teams, key="m_casa")
                tv = st.selectbox("✈️ Fora:", ["Selecione..."] + all_teams, key="m_fora")
                
                c_mk, c_od = st.columns(2)
                m_mercado = c_mk.selectbox("Mercado:", MERCADOS_DISPONIVEIS)
                m_odd = c_od.number_input("Odd:", 1.01, 100.0, 1.90)
                
                # Previsão em tempo real
                prob_est = 0
                if tc != "Selecione..." and tv != "Selecione..." and m_mercado != "Selecione...":
                    calc_m = calcular_jogo_v31(STATS[tc], STATS[tv], {})
                    prob_est = calcular_probabilidade_mercado(m_mercado, calc_m)
                    
                    if prob_est > 0:
                        st.caption(f"🎲 Probabilidade Calculada V31: **{prob_est:.1f}%**")
                
                if st.button("➕ Adicionar Manual", use_container_width=True):
                    if tc != "Selecione..." and tv != "Selecione..." and m_mercado != "Selecione...":
                        st.session_state.current_ticket.append({
                            'jogo': f"{tc} vs {tv}", 'mercado': m_mercado, 
                            'odd': m_odd, 'prob': prob_est, 'tipo': 'Manual',
                            'market_display': m_mercado
                        })
                        st.success("Adicionado!")
                        st.rerun()
                    else:
                        st.error("Selecione os times e o mercado.")

        # VISUALIZAÇÃO DO BILHETE
        st.markdown("---")
        if st.session_state.current_ticket:
            st.subheader("📋 Seu Bilhete")
            df_tick = pd.DataFrame(st.session_state.current_ticket)
            
            # Formatação para exibição
            df_show = df_tick.copy()
            df_show['Prob'] = df_show['prob'].apply(lambda x: f"{x:.1f}%")
            df_show['Emoji'] = df_show['prob'].apply(get_prob_emoji)
            
            st.dataframe(df_show[['Emoji', 'jogo', 'mercado', 'odd', 'Prob', 'tipo']], use_container_width=True)
            
            # Cálculos de EV
            total_odd = np.prod([x['odd'] for x in st.session_state.current_ticket])
            prob_real = np.prod([x['prob']/100 for x in st.session_state.current_ticket]) * 100
            fair_odd = 100/prob_real if prob_real > 0 else 0
            ev = (total_odd - fair_odd) / fair_odd * 100 if fair_odd > 0 else 0
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Odd Total", f"{total_odd:.2f}")
            c2.metric("Prob Real", f"{prob_real:.1f}%")
            c3.metric("Fair Odd", f"{fair_odd:.2f}")
            c4.metric("EV (Valor)", f"{ev:+.1f}%", delta_color="normal" if ev>0 else "inverse")
            
            # Botão de Remover
            st.markdown("##### ⚙️ Editar Bilhete")
            idx_remove = st.selectbox(
                "Remover aposta:",
                range(len(st.session_state.current_ticket)),
                format_func=lambda i: f"{st.session_state.current_ticket[i]['jogo']} - {st.session_state.current_ticket[i]['mercado']}"
            )
            
            if st.button("🗑️ Remover Selecionada"):
                st.session_state.current_ticket.pop(idx_remove)
                st.rerun()

    # ========================================
    # TAB 2: HEDGES MAXIMUM (CORRIGIDO)
    # ========================================
    with tabs[1]:
        st.header("🛡️ Hedges Super Inteligentes")
        
        # 1. Obter o bilhete do estado
        bilhete = st.session_state.current_ticket
        
        if not bilhete:
            st.warning("⚠️ Crie um bilhete no Construtor primeiro.")
        else:
            col1, col2 = st.columns(2)
            stake = col1.number_input("Stake Principal (R$)", value=100.0)
            
            # Recalcular odd total
            odd_total = np.prod([x['odd'] for x in bilhete])
            col2.metric("Odd do Bilhete", f"{odd_total:.2f}")
            retorno_max = stake * odd_total
            
            st.markdown("### 🤖 Hedges Recomendados (Data-Driven)")
            
            # Lógica para sugestão de hedges
            if len(bilhete) == 1:
                jogo_selecionado = bilhete[0]['jogo']
                st.info(f"Analisando estatísticas reais para hedge de: **{jogo_selecionado}**...")
                
                try:
                    # Recupera dados estatísticos reais
                    h_h, a_h = jogo_selecionado.split(" vs ")
                    if h_h in STATS and a_h in STATS:
                        # Roda motor V31 para este jogo
                        calc_hedge = calcular_jogo_v31(STATS[h_h], STATS[a_h], {})
                        
                        # Chama a função inteligente
                        sugestoes = sugerir_mercados_hedge_inteligente(bilhete, jogo_selecionado, STATS, calc_hedge)
                        
                        if sugestoes:
                            cols = st.columns(len(sugestoes))
                            for i, sug in enumerate(sugestoes):
                                with cols[i]:
                                    with st.container(border=True):
                                        st.markdown(f"**{sug['tipo']}**")
                                        st.write(f"✅ {sug['mercado']}")
                                        st.write(f"📊 Prob Real: {sug['prob']:.1f}%")
                                        st.caption(f"💡 {sug['analise']}")
                        else:
                            st.warning("Não encontramos hedges estatísticos de alta probabilidade (>70%) para este jogo.")
                except Exception as e:
                    st.error(f"Erro ao calcular hedge: {e}")
            
            st.markdown("---")
            
            # Calculadora Tradicional (Fallback)
            with st.expander("🧮 Calculadora de Hedge Manual", expanded=False):
                st.write("Cobre o prejuízo se a aposta principal perder.")
                odd_hedge = st.number_input("Odd da Contra-Aposta (Zebra):", 2.0, 100.0, 3.5)
                stake_hedge = stake / (odd_hedge - 1)
                
                c1, c2 = st.columns(2)
                c1.metric("Apostar na Zebra", format_currency(stake_hedge))
                c2.metric("Custo Total", format_currency(stake + stake_hedge))
                
                if (stake + stake_hedge) < retorno_max:
                    st.success(f"✅ Hedge Viável! Lucro se Principal bater: {format_currency(retorno_max - (stake + stake_hedge))}")
                else:
                    st.error("🚫 Hedge Inviável (Custo supera retorno).")

            # HEDGE 2: PARTIAL
            with st.expander("⚖️ 2. Partial Protection (50%)"):
                st.write("Protege metade do stake principal.")
                stake_partial = (stake * 0.5) / (odd_hedge - 1)
                st.metric("Apostar na Zebra", format_currency(stake_partial))
                st.info(f"Se Principal perder e Zebra ganhar, você recupera 50% do investimento.")

            # HEDGE 3: ARBITRAGEM
            with st.expander("💎 3. Guaranteed Profit (Dutching)"):
                # Probabilidade Implícita Total
                imp_prob = (1/odd_total) + (1/odd_hedge)
                if imp_prob < 1:
                    st.success(f"💎 ARBITRAGEM DETECTADA! Margem: {imp_prob*100:.1f}%")
                    # Stakes para lucro igual
                    inv_total = stake + stake_hedge # Exemplo base
                    s1 = (inv_total / odd_total) / imp_prob
                    s2 = (inv_total / odd_hedge) / imp_prob
                    st.write(f"Para investir {format_currency(inv_total)} no total:")
                    st.write(f"- Na Principal (@{odd_total:.2f}): {format_currency(s1)}")
                    st.write(f"- Na Zebra (@{odd_hedge:.2f}): {format_currency(s2)}")
                    st.write(f"Lucro Garantido: {format_currency(inv_total/imp_prob - inv_total)}")
                else:
                    st.warning(f"Sem oportunidade de arbitragem (Prob > 100%: {imp_prob*100:.1f}%)")

    # ========================================
    # TAB 3: SIMULADOR MONTE CARLO
    # ========================================
    with tabs[2]:
        st.header("🎲 Simulador Monte Carlo (3.000 Iterações)")
        
        sc1, sc2 = st.columns(2)
        sim_h = sc1.selectbox("Time Casa", sorted(list(STATS.keys())), key='sh')
        sim_a = sc2.selectbox("Time Visitante", sorted(list(STATS.keys())), key='sa')
        
        if st.button("🚀 Iniciar Simulação", use_container_width=True):
            if sim_h != sim_a:
                with st.spinner("Simulando partidas..."):
                    res = simulate_game_v31(STATS[sim_h], STATS[sim_a], {}, 3000)
                    st.success("Concluído!")
                    
                    # Resultados
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Média Gols", f"{res['goals_total'].mean():.2f}")
                    m2.metric("Média Cantos", f"{res['corners_total'].mean():.2f}")
                    m3.metric("Média Cartões", f"{res['cards_total'].mean():.2f}")
                    m4.metric("Prob Over 2.5", f"{(res['goals_total'] > 2.5).mean()*100:.1f}%")
                    
                    # Gráfico
                    fig = px.histogram(res['goals_total'], nbins=10, title="Distribuição de Gols", color_discrete_sequence=['#1e3c72'])
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Tabela
                    probs = pd.DataFrame({
                        'Mercado': ['Over 1.5 Gols', 'Over 2.5 Gols', 'Over 9.5 Cantos', 'Over 10.5 Cantos'],
                        'Probabilidade': [
                            (res['goals_total'] > 1.5).mean(), (res['goals_total'] > 2.5).mean(),
                            (res['corners_total'] > 9.5).mean(), (res['corners_total'] > 10.5).mean()
                        ]
                    })
                    probs['Probabilidade'] = probs['Probabilidade'].apply(lambda x: f"{x*100:.1f}%")
                    st.table(probs)
            else:
                st.error("Times iguais.")

    # ========================================
    # TAB 4: MÉTRICAS
    # ========================================
    with tabs[3]:
        st.header("📊 Métricas de Performance")
        if st.session_state.bet_results:
            df = pd.DataFrame(st.session_state.bet_results)
            wins = df[df['ganhou']==True]
            win_rate = len(wins)/len(df)*100
            roi = (df['lucro'].sum() / df['stake'].sum()) * 100
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Win Rate", f"{win_rate:.1f}%")
            c2.metric("ROI", f"{roi:.1f}%")
            c3.metric("Lucro Total", format_currency(df['lucro'].sum()))
            
            fig = px.line(y=st.session_state.bankroll_history, title="Evolução da Banca")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Registre apostas para ver métricas.")

    # ========================================
    # TAB 5: VISUALIZAÇÕES
    # ========================================
    with tabs[4]:
        st.header("🎨 Visualizações Avançadas")
        opt = st.selectbox("Tipo:", ["Ranking Cantos", "Ranking Cartões", "Ataque vs Defesa"])
        
        if opt == "Ranking Cantos":
            data = [{'Time': k, 'Cantos': v['corners'], 'Liga': v['league']} for k,v in STATS.items()]
            df = pd.DataFrame(data).sort_values('Cantos', ascending=False).head(20)
            fig = px.bar(df, x='Cantos', y='Time', orientation='h', color='Liga', title="Top 20 Cantos")
            st.plotly_chart(fig, use_container_width=True)
            
        elif opt == "Ataque vs Defesa":
            data = [{'Time': k, 'GF': v['goals_f'], 'GS': v['goals_a'], 'Liga': v['league']} for k,v in STATS.items()]
            df = pd.DataFrame(data)
            fig = px.scatter(df, x='GF', y='GS', color='Liga', hover_name='Time', title="Ataque vs Defesa")
            st.plotly_chart(fig, use_container_width=True)
            
        elif opt == "Ranking Cartões":
            data = [{'Time': k, 'Cartões': v['cards'], 'Liga': v['league']} for k,v in STATS.items()]
            df = pd.DataFrame(data).sort_values('Cartões', ascending=False).head(20)
            fig = px.bar(df, x='Cartões', y='Time', orientation='h', color='Liga', title="Top 20 Cartões")
            st.plotly_chart(fig, use_container_width=True)

    # ========================================
    # TAB 6: REGISTRO
    # ========================================
    with tabs[5]:
        st.header("📝 Registro Manual")
        with st.form("reg"):
            c1, c2 = st.columns(2)
            desc = c1.text_input("Descrição")
            stake = c2.number_input("Stake", 10.0)
            c3, c4 = st.columns(2)
            odd = c3.number_input("Odd", 1.01)
            res = c4.selectbox("Resultado", ["Green", "Red", "Void"])
            
            if st.form_submit_button("Salvar"):
                lucro = (stake * odd - stake) if res == "Green" else -stake if res == "Red" else 0
                ganhou = res == "Green"
                st.session_state.bet_results.append({'data': datetime.now().strftime('%d/%m'), 'descricao': desc, 'stake': stake, 'odd': odd, 'ganhou': ganhou, 'lucro': lucro})
                st.session_state.bankroll_history.append(st.session_state.bankroll_history[-1] + lucro)
                st.success("Salvo!")
                st.rerun()
        
        if st.session_state.bet_results:
            st.dataframe(pd.DataFrame(st.session_state.bet_results))

    # ========================================
    # TAB 7: SCANNER
    # ========================================
    with tabs[6]:
        st.header("🔍 Scanner Inteligente")
        if CAL.empty:
            st.warning("Sem calendário.")
        else:
            c1, c2 = st.columns(2)
            d_scan = c1.selectbox("Data Scan:", sorted(CAL['DtObj'].dt.strftime('%d/%m/%Y').unique()))
            mp = st.slider("Prob Mín", 50, 90, 70)
            
            if st.button("🔎 Escanear"):
                hits = []
                for _, r in CAL[CAL['DtObj'].dt.strftime('%d/%m/%Y')==d_scan].iterrows():
                    h, a = normalize_name(r['Time_Casa'], list(STATS.keys())), normalize_name(r['Time_Visitante'], list(STATS.keys()))
                    if h and a:
                        c = calcular_jogo_v31(STATS[h], STATS[a], {})
                        pc = calcular_poisson(c['corners_total'], 9.5)
                        if pc >= mp: hits.append({'Jogo': f"{h} x {a}", 'M': 'O9.5 C', 'Prob': f"{pc:.1f}%", 'Emoji': get_prob_emoji(pc)})
                if hits: st.dataframe(pd.DataFrame(hits), use_container_width=True)
                else: st.warning("Nada encontrado")

    # ========================================
    # TAB 8: IMPORTAR
    # ========================================
    with tabs[7]:
        st.header("📋 Importar Texto")
        txt = st.text_area("Cole seu bilhete:")
        if st.button("Analisar"):
            jogos = parse_bilhete_texto(txt)
            if jogos:
                st.success(f"{len(jogos)} jogos identificados.")
                vals = validar_jogos_bilhete(jogos, STATS)
                if vals:
                    for v in vals:
                        calc = calcular_jogo_v31(v['home_stats'], v['away_stats'], {})
                        st.write(f"✅ **{v['home']} x {v['away']}**: Previsão {calc['corners_total']:.1f} cantos")
            else:
                st.error("Nenhum jogo identificado.")

    # ========================================
    # TAB 9: AI ADVISOR ULTRA (FINAL)
    # ========================================
    with tabs[8]:
        st.header("🤖 AI Advisor ULTRA")
        st.caption("Powered by Causality Engine V31")
        
        chat_c = st.container()
        with chat_c:
            if not st.session_state.chat_history:
                st.info("👋 Olá! Pergunte sobre 'Arsenal vs Chelsea', 'Como está o Flamengo' ou 'Melhores jogos de hoje'.")
            
            for msg in st.session_state.chat_history:
                role = msg['role']
                av = "👤" if role == 'user' else "🤖"
                st.chat_message(role, avatar=av).markdown(msg['content'])
                
        prompt = st.chat_input("Digite sua pergunta...")
        if prompt:
            st.session_state.chat_history.append({'role': 'user', 'content': prompt})
            with st.spinner("🧠 Analisando dados..."):
                resp = processar_chat_ultra(prompt, STATS, CAL, REFS, st.session_state.chat_memory)
            st.session_state.chat_history.append({'role': 'assistant', 'content': resp})
            st.rerun()

    # ========================================
    # TAB 10: BLACKLIST (NOVA)
    # ========================================
    with tabs[9]:
        st.header("⛔ Blacklist Científica (V32)")
        st.caption("Evite apostar 'Over' nestes times - Eles têm as menores médias estatísticas.")
        
        tb1, tb2 = st.tabs(["📉 Cantos (Under)", "🟨 Cartões (Under)"])
        
        with tb1:
            if BLACKLIST_CORNERS:
                df_b_c = pd.DataFrame(list(BLACKLIST_CORNERS.items()), columns=['Time', 'Média'])
                df_b_c = df_b_c.sort_values('Média')
                st.dataframe(df_b_c, use_container_width=True)
            else:
                st.info("Blacklist de cantos vazia.")
                
        with tb2:
            if BLACKLIST_CARDS:
                df_b_cd = pd.DataFrame(list(BLACKLIST_CARDS.items()), columns=['Time', 'Média'])
                df_b_cd = df_b_cd.sort_values('Média')
                st.dataframe(df_b_cd, use_container_width=True)
            else:
                st.info("Blacklist de cartões vazia.")

    # ========================================
    # TAB 11: ROBÔ TIPS DIÁRIAS (NOVA)
    # ========================================
    with tabs[10]:
        st.header("🤖 Robô Tips do Dia (V33.3)")
        st.caption("Analisa jogos da data e sugere apostar em jogos movimentados ou evitar jogos 'mornos'.")
        
        if not DF_ROBO.empty:
            datas = sorted(DF_ROBO['Date'].dropna().unique(), reverse=True)
            if datas:
                d_escolhida = st.selectbox("📅 Escolha a Data:", datas)
                
                if st.button("🚀 Analisar Jogos do Dia"):
                    res_robo = analisar_robo_diario(DF_ROBO, d_escolhida, STATS)
                    
                    if not res_robo.empty:
                        st.success(f"Analisamos {len(res_robo)} jogos!")
                        
                        # Filtros rápidos
                        filtro = st.radio("Filtrar por:", ["Todos", "🔥 APOSTAR", "⛔ EVITAR"])
                        
                        df_show = res_robo
                        if filtro == "🔥 APOSTAR":
                            df_show = res_robo[res_robo['Rec'].str.contains("APOSTAR")]
                        elif filtro == "⛔ EVITAR":
                            df_show = res_robo[res_robo['Rec'].str.contains("EVITAR")]
                            
                        if not df_show.empty:
                            for _, row in df_show.iterrows():
                                with st.expander(f"{row['Rec']} | {row['Jogo']} ({row['Liga']})"):
                                    c1, c2, c3 = st.columns(3)
                                    c1.metric("Cantos Esp.", f"{row['Cantos']:.1f}")
                                    c2.metric("Cartões Esp.", f"{row['Cartões']:.1f}")
                                    c3.markdown(f"**Motivo:** :{row['Cor']}[{row['Motivo']}]")
                        else:
                            st.info("Nenhum jogo encontrado com esse filtro.")
                    else:
                        st.warning("Nenhum jogo encontrado nesta data com dados suficientes.")
            else:
                st.warning("Sem datas disponíveis no banco de dados.")
        else:
            st.error("Erro ao carregar dados do Robô. Verifique os arquivos CSV.")

if __name__ == "__main__":
    main()
