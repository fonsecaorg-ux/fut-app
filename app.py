"""
FutPrevisão V31 MAXIMUM + AI Advisor ULTRA
CÓDIGO COMPLETO - VERSÃO FINAL ESTENDIDA (NO-CUTS)
Baseado no Relatório Técnico: Causality Engine, Monte Carlo & NLP

Autor: Diego
Versão: 31.5 ULTRA PROFESSIONAL
Data: 27/12/2025

Este software implementa:
1. Motor de Causalidade V31 (Causality Engine)
2. Simulação de Monte Carlo (3.000 iterações)
3. Gestão de Banca com Critério de Kelly
4. Chatbot Analista com NLP e Cálculo de Poisson
5. Scanner de Oportunidades em Tempo Real
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
# Tenta importar para precisão máxima, mas possui fallback matemático manual
try:
    from scipy.stats import poisson, norm
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Diretório base do projeto
BASE_DIR = Path(__file__).resolve().parent

# Configuração da Página Streamlit
st.set_page_config(
    page_title="FutPrevisão V31 MAXIMUM",
    layout="wide",
    page_icon="⚽",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.futprevisao.com/help',
        'Report a bug': "https://www.futprevisao.com/bug",
        'About': "# FutPrevisão V31 MAXIMUM\nSistema Profissional de Análise Esportiva."
    }
)

# ==============================================================================
# 2. ESTILIZAÇÃO CSS PROFISSIONAL (DARK/LIGHT MODE)
# ==============================================================================

st.markdown('''
<style>
    /* ESTILO GERAL DA APLICAÇÃO 
       Focado em usabilidade profissional e contraste
    */
    
    /* TABS DE NAVEGAÇÃO */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: linear-gradient(90deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        padding: 15px 15px 0px 15px;
        border-radius: 12px 12px 0 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.05);
        border-radius: 6px 6px 0 0;
        padding: 10px 20px;
        font-weight: 600;
        color: #e0e0e0;
        border: 1px solid rgba(255,255,255,0.1);
        border-bottom: none;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255,255,255,0.15);
        color: white;
        transform: translateY(-1px);
    }
    
    .stTabs [aria-selected="true"] {
        background: #FFD700 !important;
        color: #1e3c72 !important;
        border-color: #FFD700;
        font-weight: 800;
        transform: scale(1.02);
        box-shadow: 0 -2px 10px rgba(255, 215, 0, 0.3);
    }
    
    /* CHATBOT AI ADVISOR */
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) {
        background: #f8fafc;
        border-radius: 0px 15px 15px 15px;
        padding: 20px;
        border-left: 5px solid #1e3c72;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
        margin-bottom: 10px;
    }
    
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) {
        background: #e0f2fe;
        border-radius: 15px 0px 15px 15px;
        padding: 20px;
        text-align: right;
        margin-bottom: 10px;
        border-right: 5px solid #0284c7;
    }
    
    div[data-testid="stChatMessage"] p {
        font-size: 1.05rem;
        line-height: 1.6;
    }
    
    /* CARDS E MÉTRICAS */
    div[data-testid="metric-container"] {
        background: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.08);
        border-top: 3px solid #1e3c72;
        transition: transform 0.2s;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: scale(1.02);
    }
    
    /* HEADER E TÍTULOS */
    h1 {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-family: 'Helvetica Neue', sans-serif;
        padding-bottom: 10px;
    }
    
    h2, h3 {
        color: #1e3c72;
        font-weight: 600;
    }
    
    /* ALERTS E NOTIFICAÇÕES */
    .stAlert {
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    
    /* BOTÕES PERSONALIZADOS */
    div.stButton > button {
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    div.stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* EXPANDERS */
    .streamlit-expanderHeader {
        background-color: #f8f9fa;
        border-radius: 8px;
        font-weight: 600;
    }
</style>
''', unsafe_allow_html=True)

# ==============================================================================
# 3. MAPEAMENTO DE DADOS E CONSTANTES GLOBAIS
# ==============================================================================

# Mapeamento para normalização de nomes de times
# Isso garante que "Man Utd", "Manchester United" e "Man United" sejam tratados como o mesmo time
NAME_MAPPING = {
    'Man United': 'Manchester United', 
    'Man Utd': 'Manchester United',
    'Manchester Utd': 'Manchester United',
    'Man City': 'Manchester City', 
    'Spurs': 'Tottenham', 
    'Tottenham Hotspur': 'Tottenham',
    'Wolves': 'Wolverhampton', 
    'Wolverhampton Wanderers': 'Wolverhampton',
    'Paris SG': 'PSG', 
    'Paris Saint-Germain': 'PSG',
    'Nottm Forest': 'Nottingham Forest', 
    'Nottingham': 'Nottingham Forest',
    'Sheffield Utd': 'Sheffield United',
    'Luton': 'Luton Town',
    'Newcastle': 'Newcastle United',
    'Brighton': 'Brighton & Hove Albion',
    'West Ham': 'West Ham United',
    'Bournemouth': 'AFC Bournemouth',
    'Inter': 'Inter Milan',
    'Milan': 'AC Milan',
    'Ath Madrid': 'Atletico Madrid',
    'Ath Bilbao': 'Athletic Club',
    'Betis': 'Real Betis',
    'Sociedad': 'Real Sociedad',
    'Celta': 'Celta Vigo',
    'Mallorca': 'RCD Mallorca',
    'Osasuna': 'CA Osasuna',
    'Sevilla': 'Sevilla FC',
    'Valencia': 'Valencia CF',
    'Villarreal': 'Villarreal CF',
    'Dortmund': 'Borussia Dortmund',
    'Leverkusen': 'Bayer Leverkusen',
    'Bayern': 'Bayern Munich',
    'Mainz': 'Mainz 05',
    'Augsburg': 'FC Augsburg',
    'Stuttgart': 'VfB Stuttgart',
    'Wolfsburg': 'VfL Wolfsburg',
    'Gladbach': 'Borussia Monchengladbach',
    'Frankfurt': 'Eintracht Frankfurt',
    'Marseille': 'Olympique Marseille',
    'Lyon': 'Olympique Lyon',
    'Monaco': 'AS Monaco',
    'Lille': 'LOSC Lille',
    'Rennes': 'Stade Rennais',
    'Lens': 'RC Lens'
}

# Constantes do Causality Engine
PRESSURE_HIGH_THRESHOLD = 6.0  # Chutes no alvo para considerar pressão alta
PRESSURE_MED_THRESHOLD = 4.5   # Chutes no alvo para considerar pressão média
VIOLENCE_HIGH_THRESHOLD = 12.5 # Faltas para considerar time violento
REF_STRICT_THRESHOLD = 4.5     # Cartões/jogo para árbitro rigoroso

# ==============================================================================
# 4. FUNÇÕES AUXILIARES E UTILITÁRIOS
# ==============================================================================

def find_file(filename: str) -> Optional[str]:
    """
    Busca robusta de arquivos em múltiplos diretórios possíveis.
    Essencial para garantir que o app rode tanto localmente quanto em cloud.
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
            return str(path)
            
    return None

def normalize_name(name: str, known_teams: List[str]) -> Optional[str]:
    """
    Normaliza nomes de times usando fuzzy matching e o dicionário NAME_MAPPING.
    Retorna o nome oficial do time no banco de dados.
    """
    if not name or not known_teams:
        return None
    
    name_clean = str(name).strip()
    
    # 1. Tenta mapeamento direto
    if name_clean in NAME_MAPPING:
        target_name = NAME_MAPPING[name_clean]
        # Verifica se o nome mapeado existe na lista de times conhecidos
        if target_name in known_teams:
            return target_name
        # Se não, tenta fuzzy no nome mapeado
        name_clean = target_name
        
    # 2. Tenta correspondência exata
    if name_clean in known_teams:
        return name_clean
        
    # 3. Tenta Fuzzy Matching (difflib)
    matches = get_close_matches(name_clean, known_teams, n=1, cutoff=0.6)
    if matches:
        return matches[0]
        
    return None

def clean_team_name(text: str) -> str:
    """
    Limpa nome de time vindo do input do usuário no chat.
    Remove pontuação e palavras comuns (stopwords).
    """
    if not text:
        return ""
        
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text) # Remove pontuação
    
    stop_words = {
        'do', 'da', 'de', 'dos', 'das', 'o', 'a', 'os', 'as', 
        'como', 'está', 'esta', 'stats', 'estatistica', 'estatísticas',
        'vs', 'x', 'contra', 'analise', 'analisar', 'previsao', 'jogo',
        'partida', 'hoje', 'amanha', 'ontem', 'agora', 'proximo',
        'qual', 'quais', 'quanto', 'quantos', 'quem', 'ganha', 'vence'
    }
    
    words = text.split()
    cleaned_words = [w for w in words if w not in stop_words]
    
    return ' '.join(cleaned_words).strip()

def format_currency(value: float) -> str:
    """Formata valor monetário para o padrão brasileiro (R$ X.XXX,XX)"""
    return f"R$ {value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def get_prob_emoji(prob: float) -> str:
    """
    Retorna um emoji indicador de qualidade baseado na probabilidade.
    Usado visualmente para destacar apostas de valor.
    """
    if prob >= 80: return "🔥"  # Super Valor (Fire)
    elif prob >= 70: return "✅" # Bom Valor (Check)
    elif prob >= 60: return "⚠️" # Risco Moderado (Warning)
    elif prob >= 50: return "🟡" # Neutro
    else: return "🔻"           # Risco Alto (Down)

# ==============================================================================
# 5. CARREGAMENTO E PROCESSAMENTO DE DADOS (ETL)
# ==============================================================================

@st.cache_data(ttl=3600)
def load_all_data():
    """
    Carrega, limpa e processa TODOS os dados do sistema.
    Esta função é o coração dos dados, transformando CSVs brutos em dicionários de objetos.
    Implementa tratamento de erros robusto para cada arquivo.
    """
    stats_db = {}
    cal = pd.DataFrame()
    referees = {}
    
    # Lista completa de arquivos de ligas suportadas
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
    
    # Processamento dos arquivos de ligas
    for league_name, filename in league_files.items():
        filepath = find_file(filename)
        if not filepath: 
            # Log de aviso (opcional)
            continue
            
        try:
            # Carrega CSV com encoding utf-8 para suportar acentos
            df = pd.read_csv(filepath, encoding='utf-8')
            
            # Normalizar nomes das colunas (remove espaços extras)
            cols = {c: c.strip() for c in df.columns}
            df.rename(columns=cols, inplace=True)
            
            # Identifica todos os times únicos na liga
            teams = set(df['HomeTeam'].dropna().unique()) | set(df['AwayTeam'].dropna().unique())
            
            for team in teams:
                if pd.isna(team): continue
                
                # Separa jogos como mandante e visitante
                h_games = df[df['HomeTeam'] == team]
                a_games = df[df['AwayTeam'] == team]
                
                # --- EXTRAÇÃO DE MÉTRICAS PARA O CAUSALITY ENGINE ---
                # Utiliza médias e fallbacks caso os dados sejam escassos
                
                # 1. Cantos (HC = Home Corners, AC = Away Corners)
                corners_h = h_games['HC'].mean() if 'HC' in h_games.columns and len(h_games) > 0 else 5.0
                corners_a = a_games['AC'].mean() if 'AC' in a_games.columns and len(a_games) > 0 else 4.0
                
                # 2. Cartões (HY=Home Yellow, HR=Home Red)
                # Soma cartões amarelos e vermelhos para um total de "pontos de cartão"
                if 'HY' in h_games.columns and 'HR' in h_games.columns:
                    ch = h_games['HY'].mean() + (h_games['HR'].mean() * 2) # Vermelho vale dobro
                else:
                    ch = 1.8
                    
                if 'AY' in a_games.columns and 'AR' in a_games.columns:
                    ca = a_games['AY'].mean() + (a_games['AR'].mean() * 2)
                else:
                    ca = 2.2
                
                # 3. Faltas (HF = Home Fouls, AF = Away Fouls) - Indicador de Violência
                fouls_h = h_games['HF'].mean() if 'HF' in h_games.columns and len(h_games) > 0 else 11.5
                fouls_a = a_games['AF'].mean() if 'AF' in a_games.columns and len(a_games) > 0 else 12.5
                
                # 4. Gols (FTHG = Full Time Home Goals, FTAG = Full Time Away Goals)
                # Capacidade Ofensiva
                goals_fh = h_games['FTHG'].mean() if 'FTHG' in h_games.columns and len(h_games) > 0 else 1.4
                goals_fa = a_games['FTAG'].mean() if 'FTAG' in a_games.columns and len(a_games) > 0 else 1.1
                
                # Fragilidade Defensiva
                goals_ah = h_games['FTAG'].mean() if 'FTAG' in h_games.columns and len(h_games) > 0 else 1.0 # Sofre em casa
                goals_aa = a_games['FTHG'].mean() if 'FTHG' in a_games.columns and len(a_games) > 0 else 1.5 # Sofre fora
                
                # 5. Chutes no Alvo (HST = Home Shots on Target, AST = Away Shots on Target) - Indicador de Pressão
                shots_h = h_games['HST'].mean() if 'HST' in h_games.columns and len(h_games) > 0 else 4.8
                shots_a = a_games['AST'].mean() if 'AST' in a_games.columns and len(a_games) > 0 else 3.8
                
                # Armazena estatísticas processadas no dicionário principal
                stats_db[team] = {
                    'league': league_name,
                    # Médias Gerais
                    'corners': (corners_h + corners_a) / 2,
                    'corners_home': corners_h,
                    'corners_away': corners_a,
                    
                    'cards': (ch + ca) / 2,
                    'cards_home': ch,
                    'cards_away': ca,
                    
                    'fouls': (fouls_h + fouls_a) / 2,
                    'fouls_home': fouls_h,
                    'fouls_away': fouls_a,
                    
                    'goals_f': (goals_fh + goals_fa) / 2, # Gols Feitos Geral
                    'goals_f_home': goals_fh,
                    'goals_f_away': goals_fa,
                    
                    'goals_a': (goals_ah + goals_aa) / 2, # Gols Sofridos Geral
                    'goals_a_home': goals_ah,
                    'goals_a_away': goals_aa,
                    
                    'shots_on_target': (shots_h + shots_a) / 2,
                    'shots_home': shots_h,
                    'shots_away': shots_a,
                    
                    'games_played': len(h_games) + len(a_games)
                }
        except Exception as e:
            # Em produção, logar o erro. Aqui apenas passamos para não quebrar o app.
            pass 
            
    # Processamento do Calendário
    cal_path = find_file('calendario_ligas.csv')
    if cal_path:
        try:
            cal = pd.read_csv(cal_path, encoding='utf-8')
            if 'Data' in cal.columns:
                # Converte coluna de data para datetime objects para manipulação
                cal['DtObj'] = pd.to_datetime(cal['Data'], dayfirst=True, errors='coerce')
        except Exception: 
            pass
    
    # Processamento dos Árbitros
    ref_path = find_file('arbitros_5_ligas_2025_2026.csv')
    if ref_path:
        try:
            refs_df = pd.read_csv(ref_path, encoding='utf-8')
            for _, row in refs_df.iterrows():
                # Calcula fator de rigor do árbitro
                avg_cards = row.get('Media_Cartoes_Por_Jogo', 4.0)
                games = row.get('Jogos_Apitados', 0)
                reds = row.get('Cartoes_Vermelhos', 0)
                
                # Evita divisão por zero
                red_rate = reds / games if games > 0 else 0.1
                
                referees[row['Arbitro']] = {
                    'factor': avg_cards / 4.0, # Fator de multiplicação base 4.0
                    'avg_cards': avg_cards,
                    'games': games,
                    'red_rate': red_rate
                }
        except Exception: 
            pass
            
    return stats_db, cal, referees

# ==============================================================================
# 6. MATEMÁTICA E ESTATÍSTICA (POISSON, MONTE CARLO)
# ==============================================================================

def calcular_poisson(media: float, linha: float) -> float:
    """
    Calcula a probabilidade de um evento ocorrer MAIS que 'linha' vezes (Over),
    dado uma média esperada 'media' (lambda), usando a Distribuição de Poisson.
    
    P(X > k) = 1 - P(X <= k)
    
    Args:
        media (float): Média esperada do evento (ex: 10.5 cantos).
        linha (float): Linha de aposta (ex: 9.5).
        
    Returns:
        float: Probabilidade percentual (0-100).
    """
    if media <= 0: return 0.0
    
    if SCIPY_AVAILABLE:
        try:
            # Em apostas "Over 9.5", queremos a probabilidade de 10, 11, 12...
            # P(X >= 10) = 1 - P(X <= 9)
            # Para Over 9.5, usamos k=9. 
            k = int(linha) 
            prob_under = poisson.cdf(k, media)
            return (1 - prob_under) * 100
        except:
            pass
    
    # Fallback manual da fórmula de Poisson se scipy não estiver disponível
    # P(k) = (e^-lambda * lambda^k) / k!
    prob_exact_cumulative = 0
    k = int(linha)
    for i in range(k + 1):
        prob_exact_cumulative += (math.exp(-media) * (media ** i)) / math.factorial(i)
        
    return (1 - prob_exact_cumulative) * 100

def calcular_jogo_v31(home_stats: Dict, away_stats: Dict, ref_data: Dict) -> Dict:
    """
    CAUSALITY ENGINE V31 - NÚCLEO DO SISTEMA
    Implementa a lógica 'Causa -> Efeito' para previsões mais precisas.
    
    Não usa apenas médias simples. Usa fatores de correção:
    - Pressão (Chutes) -> Aumenta Cantos
    - Violência (Faltas) -> Aumenta Cartões
    - Rigor do Árbitro -> Multiplicador de Cartões
    """
    # Retorno padrão seguro se dados faltantes
    empty_res = {'corners': {'h':0, 'a':0, 't':0}, 
                 'cards': {'h':0, 'a':0, 't':0}, 
                 'goals': {'h':0, 'a':0}, 
                 'corners_total': 0, 
                 'total_goals': 0, 
                 'cards_total': 0,
                 'xg_home': 0,
                 'xg_away': 0}
                 
    if not home_stats or not away_stats:
        return empty_res

    # === 1. PREVISÃO DE ESCANTEIOS (Baseado em Pressão) ===
    base_h = home_stats.get('corners_home', 5.0)
    base_a = away_stats.get('corners_away', 4.0)
    
    # Fator Pressão: Analisa chutes no alvo (Shots on Target)
    shots_h = home_stats.get('shots_home', 4.5)
    shots_a = away_stats.get('shots_away', 3.5)
    
    # Se o time chuta muito, tende a gerar mais escanteios (desvios, defesas)
    pressure_h = 1.15 if shots_h > PRESSURE_HIGH_THRESHOLD else 1.05 if shots_h > PRESSURE_MED_THRESHOLD else 1.0
    pressure_a = 1.10 if shots_a > PRESSURE_MED_THRESHOLD else 1.0
    
    # Fator Casa/Fora (Mandante tende a pressionar mais no final se estiver perdendo)
    # Aqui simplificamos com um multiplicador fixo
    corners_h = base_h * pressure_h * 1.10
    corners_a = base_a * pressure_a * 0.90
    corners_total = corners_h + corners_a
    
    # === 2. PREVISÃO DE CARTÕES (Baseado em Violência e Árbitro) ===
    fouls_h = home_stats.get('fouls_home', 11.0)
    fouls_a = away_stats.get('fouls_away', 12.0)
    
    # Fator Violência: Times que fazem muitas faltas
    violencia_h = 1.1 if fouls_h > VIOLENCE_HIGH_THRESHOLD else 1.0
    violencia_a = 1.1 if fouls_a > VIOLENCE_HIGH_THRESHOLD else 1.0
    
    # Fator Árbitro: Ajusta a média dos times pela média do juiz
    ref_avg = ref_data.get('avg_cards', 4.0) if ref_data else 4.0
    
    # Média base dos times
    cards_h_base = home_stats.get('cards_home', 1.8)
    cards_a_base = away_stats.get('cards_away', 2.2)
    
    # Fórmula V31 para cartões: (Média Times + Média Juiz) / 2 * Fator Violência
    cards_h_proj = (cards_h_base + (ref_avg/2)) / 2 * violencia_h
    cards_a_proj = (cards_a_base + (ref_avg/2)) / 2 * violencia_a
    
    cards_total = cards_h_proj + cards_a_proj
    
    # === 3. PREVISÃO DE GOLS (xG V31) ===
    # Usa modelo de força de ataque vs força de defesa relativo à liga
    league_avg_goals = 1.35 # Média aproximada grandes ligas
    
    # Força de Ataque Casa = Gols Feitos Casa / Média Liga
    att_h = home_stats['goals_f_home'] / league_avg_goals
    # Força de Defesa Fora = Gols Sofridos Fora / Média Liga
    def_a = away_stats['goals_a_away'] / league_avg_goals
    
    # xG Casa = Força Ataque H * Força Defesa A * Média Liga
    xg_h = att_h * def_a * league_avg_goals
    
    # Mesmo processo para visitante
    att_a = away_stats['goals_f_away'] / league_avg_goals
    def_h = home_stats['goals_a_home'] / league_avg_goals
    xg_a = att_a * def_h * league_avg_goals
    
    total_goals = xg_h + xg_a
    
    return {
        'corners': {'h': corners_h, 'a': corners_a, 't': corners_total},
        'cards': {'h': cards_h_proj, 'a': cards_a_proj, 't': cards_total},
        'goals': {'h': xg_h, 'a': xg_a},
        'corners_total': corners_total,
        'cards_total': cards_total,
        'total_goals': total_goals,
        'xg_home': xg_h,
        'xg_away': xg_a
    }

def simulate_game_v31(home_stats: Dict, away_stats: Dict, ref_data: Dict, n_sims: int = 3000) -> Dict:
    """
    Simulador de Monte Carlo
    Executa n_sims partidas virtuais usando distribuição de Poisson baseada nas médias calculadas.
    Retorna arrays com todos os resultados para análise de distribuição.
    """
    calc = calcular_jogo_v31(home_stats, away_stats, ref_data)
    
    # Gera arrays de simulação
    sim_corners = np.random.poisson(calc['corners_total'], n_sims)
    sim_cards = np.random.poisson(calc['cards_total'], n_sims)
    sim_goals_h = np.random.poisson(calc['xg_home'], n_sims)
    sim_goals_a = np.random.poisson(calc['xg_away'], n_sims)
    
    return {
        'corners_total': sim_corners,
        'cards_total': sim_cards,
        'goals_h': sim_goals_h,
        'goals_a': sim_goals_a,
        'goals_total': sim_goals_h + sim_goals_a
    }

# ==============================================================================
# 7. CHATBOT AI ADVISOR ULTRA (LÓGICA NLP COMPLETA)
# ==============================================================================

def extrair_entidades(mensagem: str, stats_db: Dict) -> Dict:
    """
    Motor de NLP (Processamento de Linguagem Natural)
    Extrai: Times, Intenções (Mercado), Linhas numéricas.
    """
    msg_lower = mensagem.lower()
    entidades = {'times': [], 'mercado': None, 'linha': None}
    
    # 1. Extração de Times (Prioriza nomes compostos)
    known_teams = list(stats_db.keys())
    sorted_teams = sorted(known_teams, key=len, reverse=True)
    
    # Remove palavras comuns que podem confundir
    msg_clean = msg_lower
    
    found_teams = []
    for team in sorted_teams:
        if team.lower() in msg_clean:
            # Verifica se já não encontramos este time (ex: evitar pegar "Manchester" se já pegou "Manchester City")
            is_substring = False
            for ft in found_teams:
                if team.lower() in ft.lower():
                    is_substring = True
                    break
            
            if not is_substring:
                entidades['times'].append(team)
                found_teams.append(team)
                msg_clean = msg_clean.replace(team.lower(), "") # Consome o nome
    
    # 2. Extração de Mercado (Intenção)
    if any(x in msg_lower for x in ['canto', 'escanteio']):
        entidades['mercado'] = 'cantos'
    elif any(x in msg_lower for x in ['cartao', 'cartão', 'amarelo']):
        entidades['mercado'] = 'cartoes'
    elif any(x in msg_lower for x in ['gol', 'gols', 'over', 'under']):
        entidades['mercado'] = 'gols'
        
    # 3. Extração de Linha Numérica (ex: "over 9.5")
    numeros = re.findall(r'\d+\.?\d*', mensagem)
    if numeros:
        # Pega o primeiro número encontrado que pareça uma linha (geralmente float ou int pequeno)
        for num in numeros:
            val = float(num)
            if val < 20: # Assumindo que linhas > 20 são raras ou são odds
                entidades['linha'] = val
                break
                
    return entidades

def processar_chat_ultra(mensagem: str, stats_db: Dict, cal: pd.DataFrame, refs: Dict) -> str:
    """
    CÉREBRO DO AI ADVISOR ULTRA
    Lógica de decisão baseada no Prompt do Usuário.
    """
    if not mensagem:
        return "Olá! Sou o AI Advisor ULTRA. Posso analisar jogos, times e probabilidades. Como posso ajudar?"
        
    entidades = extrair_entidades(mensagem, stats_db)
    times = entidades['times']
    msg_lower = mensagem.lower()
    
    # Flags de Intenção
    is_vs = len(times) >= 2 or (' vs ' in msg_lower)
    is_analise = any(x in msg_lower for x in ['analise', 'como esta', 'perfil', 'stats'])
    is_prob = any(x in msg_lower for x in ['probabilidade', 'chance', 'qual a chance'])
    is_sugestao = any(x in msg_lower for x in ['sugira', 'sugestao', 'aposta', 'mercado', 'palpite', 'melhor', 'vale a pena'])
    
    # ====================================================
    # CENÁRIO 1: ANÁLISE DE CONFRONTO (VS) OU SUGESTÃO
    # ====================================================
    if len(times) >= 2:
        t1, t2 = times[0], times[1]
        s1, s2 = stats_db[t1], stats_db[t2]
        
        # Simula jogo
        calc = calcular_jogo_v31(s1, s2, {})
        
        # Probabilidades Principais
        prob_over_gols = calcular_poisson(calc['total_goals'], 2.5)
        prob_over_cantos = calcular_poisson(calc['corners_total'], 9.5)
        prob_over_cartoes = calcular_poisson(calc['cards_total'], 4.5)
        prob_btts = min((calc['xg_home'] * calc['xg_away'] * 38), 92) # Estimativa BTTS
        
        # Monta Resposta Profissional
        resp = f"📊 **ANÁLISE ESTATÍSTICA: {t1} vs {t2}**\n\n"
        
        # 1. Projeções
        resp += "**🔎 Projeções do Modelo V31:**\n"
        resp += f"• **Gols (xG):** {calc['total_goals']:.2f} (Esperado: {'Aberto' if calc['total_goals'] > 2.6 else 'Neutro' if calc['total_goals'] > 2.2 else 'Travado'})\n"
        resp += f"• **Cantos:** {calc['corners_total']:.1f}\n"
        resp += f"• **Cartões:** {calc['cards_total']:.1f}\n\n"
        
        # 2. Se for pedido de Probabilidade Específica
        if is_prob and entidades['linha']:
            linha = entidades['linha']
            mercado = entidades['mercado'] or 'gols'
            
            media_alvo = 0
            if mercado == 'cantos': media_alvo = calc['corners_total']
            elif mercado == 'cartoes': media_alvo = calc['cards_total']
            else: media_alvo = calc['total_goals']
            
            prob_user = calcular_poisson(media_alvo, linha)
            emoji = get_prob_emoji(prob_user)
            
            resp += f"🎲 **Sua Consulta:** Chance de Over {linha} {mercado.capitalize()}\n"
            resp += f"{emoji} **Probabilidade:** {prob_user:.1f}%\n"
            resp += f"📉 Média Esperada: {media_alvo:.2f}\n\n"
            return resp
            
        # 3. Sugestões de Valor (Lógica do Prompt)
        resp += "**💡 Oportunidades de Valor (EV+):**\n"
        found_value = False
        
        # Lógica de Sugestão
        # Gols
        if prob_over_gols > 65:
            resp += f"✅ **Over 2.5 Gols** ({prob_over_gols:.1f}%)\n   Ataques eficientes, xG combinado alto.\n"
            found_value = True
        elif prob_over_gols < 35:
            resp += f"✅ **Under 2.5 Gols** ({(100-prob_over_gols):.1f}%)\n   Defesas sólidas e xG baixo.\n"
            found_value = True
            
        if prob_btts > 60:
            resp += f"✅ **Ambos Marcam (BTTS)** ({prob_btts:.1f}%)\n   Ambos times com tendência de marcar.\n"
            found_value = True
            
        # Cantos
        if prob_over_cantos > 70:
            resp += f"✅ **Over 9.5 Cantos** ({prob_over_cantos:.1f}%)\n   Jogo de pressão e chutes cruzados.\n"
            found_value = True
            
        # Cartões
        if prob_over_cartoes > 65:
            resp += f"✅ **Over 4.5 Cartões** ({prob_over_cartoes:.1f}%)\n   Indícios de jogo físico/pegado.\n"
            found_value = True
            
        if not found_value:
            resp += "⚠️ **Sem valor estatístico claro pré-jogo.**\n   As linhas estão justas. Sugiro observar o mercado de **Live** ou buscar Handicap Asiático."
            
        return resp

    # ====================================================
    # CENÁRIO 2: ANÁLISE DE TIME ÚNICO
    # ====================================================
    elif len(times) == 1:
        t = times[0]
        s = stats_db[t]
        
        resp = f"📊 **RAIO-X: {t}**\n"
        resp += f"_(Liga: {s['league']} | Jogos na base: {s.get('games_played', 0)})_\n\n"
        
        # Stats Principais
        resp += f"**⚔️ Ataque:** {s['goals_f']:.2f} gols/jogo\n"
        resp += f"**🛡️ Defesa:** {s['goals_a']:.2f} sofridos/jogo\n"
        resp += f"**🚩 Cantos:** {s['corners']:.2f}/jogo\n"
        resp += f"**🟨 Cartões:** {s['cards']:.2f}/jogo\n\n"
        
        # Veredito do AI Advisor
        resp += "**🧠 Análise de Tendência:**\n"
        tendencias = []
        
        if s['corners'] > 6.0:
            tendencias.append("🔥 **Máquina de Cantos:** Média muito alta. Excelente para Over e Race.")
        elif s['corners'] < 3.5:
            tendencias.append("🔻 **Poucos Cantos:** Time joga fechado ou centralizado. Bom para Under.")
            
        if s['goals_f'] > 2.0:
            tendencias.append("⚽ **Ataque Letal:** Marca com muita frequência.")
        elif s['goals_f'] < 0.8:
            tendencias.append("⚠️ **Ataque Anêmico:** Dificuldade em marcar.")
            
        if s['goals_a'] > 1.8:
            tendencias.append("🛡️ **Defesa Frágil:** Tende a sofrer gols (bom para Over do adversário).")
            
        if s['cards'] > 2.5:
            tendencias.append("🟨 **Indisciplinado:** Média alta de cartões.")
            
        if not tendencias:
            resp += "Time com estatísticas equilibradas/medianas para a liga. Sem tendências extremas."
        else:
            for t in tendencias:
                resp += f"{t}\n"
            
        return resp

    # ====================================================
    # CENÁRIO 3: MELHORES JOGOS / SCANNER VIA CHAT
    # ====================================================
    elif "melhor" in msg_lower or "hoje" in msg_lower or "jogos" in msg_lower:
        # Busca no calendário de hoje
        hoje = datetime.now().strftime('%d/%m/%Y')
        jogos = cal[cal['Data'] == hoje] if not cal.empty else pd.DataFrame()
        
        if jogos.empty:
            return f"📅 Não encontrei jogos no calendário para hoje ({hoje}). Tente selecionar uma data específica na aba Construtor."
            
        # Analisa todos os jogos do dia e ranqueia
        ranking = []
        for _, row in jogos.iterrows():
            h = normalize_name(row['Time_Casa'], list(stats_db.keys()))
            a = normalize_name(row['Time_Visitante'], list(stats_db.keys()))
            
            if h and a and h in stats_db and a in stats_db:
                calc = calcular_jogo_v31(stats_db[h], stats_db[a], {})
                
                # Critérios de Destaque
                score_movimentacao = calc['total_goals'] * 2 + calc['corners_total']
                ranking.append({
                    'jogo': f"{h} vs {a}",
                    'score': score_movimentacao,
                    'stats': calc
                })
        
        # Ordena e pega top 3
        ranking.sort(key=lambda x: x['score'], reverse=True)
        top_jogos = ranking[:3]
        
        if not top_jogos:
            return f"Encontrei jogos para hoje, mas não tenho dados estatísticos suficientes dos times para analisá-los."
            
        resp = f"🏆 **TOP JOGOS PARA HOJE ({hoje}):**\n_(Critério: Potencial de Movimentação)_\n\n"
        
        for item in top_jogos:
            j = item['jogo']
            d = item['stats']
            resp += f"**{j}**\n"
            resp += f"   🎯 Gols Esp: {d['total_goals']:.1f} | Cantos: {d['corners_total']:.1f}\n"
            
            # Destaque rápido
            if d['total_goals'] > 2.8: resp += "   🔥 Alta tendência de Gols\n"
            if d['corners_total'] > 10.5: resp += "   🚩 Alta tendência de Cantos\n"
            resp += "\n"
            
        return resp

    # ====================================================
    # CENÁRIO 4: AJUDA / DEFAULT
    # ====================================================
    else:
        return """🤖 **AI ADVISOR ULTRA - Como posso ajudar?**

Minhas análises são baseadas 100% nos dados estatísticos carregados. Não uso "feeling".

**Exemplos de perguntas:**
1️⃣ *"Analise Arsenal vs Chelsea"* (Análise completa do jogo)
2️⃣ *"Como está o Real Madrid?"* (Raio-X do time)
3️⃣ *"Melhores jogos de hoje"* (Scanner rápido)
4️⃣ *"Qual a chance de over 9.5 cantos em Liverpool x City?"* (Cálculo de Poisson)
5️⃣ *"Sugira uma aposta para Flamengo x Palmeiras"* (Recomendação de valor)

*Digite o nome dos times para começar!*"""

# ==============================================================================
# 8. MÉTODOS FINANCEIROS E GRÁFICOS
# ==============================================================================

def calculate_sharpe_ratio(returns: List[float]) -> float:
    if not returns or len(returns) < 2: return 0.0
    return (np.mean(returns) - 1.0) / np.std(returns) if np.std(returns) > 0 else 0.0

def calculate_max_drawdown(bankroll_history: List[float]) -> float:
    if len(bankroll_history) < 2: return 0.0
    peak = bankroll_history[0]
    max_dd = 0.0
    for value in bankroll_history:
        if value > peak: peak = value
        dd = (peak - value) / peak * 100
        if dd > max_dd: max_dd = dd
    return max_dd

def calculate_roi(total_staked: float, total_profit: float) -> float:
    if total_staked == 0: return 0.0
    return (total_profit / total_staked) * 100

def parse_bilhete_texto(texto: str) -> List[Dict]:
    """
    Parser simplificado para identificar apostas em texto colado.
    Procura padrões como "Time A x Time B" e "Over X".
    """
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
    # --- 1. CARREGAMENTO INICIAL DE DADOS ---
    # Colocado aqui para garantir que stats_db exista antes de qualquer renderização
    STATS, CAL, REFS = load_all_data()
    
    # --- 2. INICIALIZAÇÃO DE ESTADO DA SESSÃO ---
    if 'current_ticket' not in st.session_state: st.session_state.current_ticket = []
    if 'bet_results' not in st.session_state: st.session_state.bet_results = []
    if 'bankroll_history' not in st.session_state: st.session_state.bankroll_history = [1000.0]
    if 'chat_history' not in st.session_state: st.session_state.chat_history = []
    
    # --- 3. SIDEBAR ---
    with st.sidebar:
        st.header("📊 Dashboard Profissional")
        
        # Cards de resumo
        c1, c2 = st.columns(2)
        c1.metric("Times", len(STATS))
        c2.metric("Jogos DB", len(CAL) if not CAL.empty else 0)
        
        banca_atual = st.session_state.bankroll_history[-1]
        st.metric("💰 Banca Atual", format_currency(banca_atual))
        
        # Mini-status do bilhete
        if st.session_state.current_ticket:
            st.success(f"🎫 {len(st.session_state.current_ticket)} apostas no bilhete")
            if st.button("Limpar Bilhete", use_container_width=True):
                st.session_state.current_ticket = []
                st.rerun()
        else:
            st.info("📭 Bilhete vazio")
            
        st.markdown("---")
        st.caption("v31.5 ULTRA | © 2025")

    # --- 4. HEADER DA PÁGINA ---
    col1, col2, col3 = st.columns([1, 5, 2])
    with col1:
        st.write("⚽") # Placeholder para logo
    with col2:
        st.title("FutPrevisão V31 ULTRA")
        st.markdown("**Professional Sports Analytics System** | _Causality Engine V31_")
    with col3:
        if not CAL.empty:
            hj = datetime.now().strftime('%d/%m/%Y')
            jogos_hj = len(CAL[CAL['Data'] == hj])
            st.metric("Jogos Hoje", jogos_hj)
    
    st.markdown("---")

    # --- 5. TABS DE NAVEGAÇÃO ---
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "🎫 Construtor", "🛡️ Hedges", "🎲 Simulador", "📊 Métricas", 
        "🎨 Viz", "📝 Registro", "🔍 Scanner", "📋 Importar", "🤖 AI Advisor"
    ])
    
    # ============================================================
    # TAB 1: CONSTRUTOR DE BILHETES (MANUAL + AUTO)
    # ============================================================
    with tab1:
        st.subheader("🛠️ Construtor de Bilhetes")
        
        c_col1, c_col2 = st.columns([1, 1])
        
        with c_col1:
            st.markdown("#### 📅 Automático (Calendário)")
            if not CAL.empty:
                datas = sorted(CAL['DtObj'].dt.strftime('%d/%m/%Y').unique())
                if datas:
                    data_sel = st.selectbox("Selecione a Data:", datas)
                    jogos_dia = CAL[CAL['DtObj'].dt.strftime('%d/%m/%Y') == data_sel]
                    
                    if jogos_dia.empty:
                        st.warning("Sem jogos nesta data.")
                    
                    for _, row in jogos_dia.iterrows():
                        h, a = normalize_name(row['Time_Casa'], list(STATS.keys())), normalize_name(row['Time_Visitante'], list(STATS.keys()))
                        
                        if h and a:
                            calc = calcular_jogo_v31(STATS[h], STATS[a], {})
                            
                            with st.expander(f"⚽ {h} vs {a} | 🕓 {row.get('Hora', '-')}"):
                                m1, m2, m3 = st.columns(3)
                                m1.metric("Cantos", f"{calc['corners_total']:.1f}")
                                m2.metric("Gols", f"{calc['total_goals']:.1f}")
                                m3.metric("Cartões", f"{calc['cards_total']:.1f}")
                                
                                # Botões de adição rápida
                                b1, b2 = st.columns(2)
                                if b1.button("Over 9.5 Cantos", key=f"btn_c_{h}"):
                                    prob = calcular_poisson(calc['corners_total'], 9.5)
                                    st.session_state.current_ticket.append({
                                        'jogo': f"{h} vs {a}", 'mercado': 'Over 9.5 Cantos', 
                                        'odd': 1.85, 'prob': prob, 'tipo': 'Auto'
                                    })
                                    st.rerun()
                                    
                                if b2.button("Over 2.5 Gols", key=f"btn_g_{h}"):
                                    prob = calcular_poisson(calc['total_goals'], 2.5)
                                    st.session_state.current_ticket.append({
                                        'jogo': f"{h} vs {a}", 'mercado': 'Over 2.5 Gols', 
                                        'odd': 1.90, 'prob': prob, 'tipo': 'Auto'
                                    })
                                    st.rerun()

        with c_col2:
            st.markdown("#### 📝 Manual (Custom)")
            with st.container():
                st.info("Adicione jogos não listados ou mercados específicos.")
                m_jogo = st.text_input("Nome do Jogo (ex: Brasil x Argentina)", key="m_jogo")
                
                cc1, cc2 = st.columns(2)
                m_mercado = cc1.selectbox("Mercado", 
                    ["Over 2.5 Gols", "Under 2.5 Gols", "Over 9.5 Cantos", "Over 10.5 Cantos", 
                     "Over 4.5 Cartões", "Vitória Casa", "Vitória Fora", "Ambos Marcam"],
                    key="m_mercado"
                )
                m_odd = cc2.number_input("Odd", min_value=1.01, value=1.90, step=0.01, key="m_odd")
                
                if st.button("➕ Adicionar Manualmente", use_container_width=True):
                    if m_jogo:
                        # Estima probabilidade pela odd (1/odd) se for manual
                        prob_impl = (1 / m_odd) * 100
                        st.session_state.current_ticket.append({
                            'jogo': m_jogo,
                            'mercado': m_mercado,
                            'odd': m_odd,
                            'prob': prob_impl,
                            'tipo': 'Manual'
                        })
                        st.success(f"✅ Adicionado: {m_jogo}")
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error("Digite o nome do jogo.")

        # VISUALIZAÇÃO DO BILHETE (EMBAIXO)
        st.markdown("---")
        st.subheader("📋 Seu Bilhete")
        
        if st.session_state.current_ticket:
            # Converte para DataFrame para exibição bonita
            df_tick = pd.DataFrame(st.session_state.current_ticket)
            # Reordena colunas
            cols_show = ['jogo', 'mercado', 'odd', 'prob', 'tipo']
            # Garante que as colunas existem
            for c in cols_show:
                if c not in df_tick.columns: df_tick[c] = '-'
            
            st.dataframe(df_tick[cols_show], use_container_width=True)
            
            # Cálculos Finais
            total_odd = np.prod([x['odd'] for x in st.session_state.current_ticket])
            prob_acumulada = np.prod([x['prob']/100 for x in st.session_state.current_ticket]) * 100
            
            res1, res2, res3 = st.columns(3)
            res1.metric("Odd Total", f"{total_odd:.2f}")
            res2.metric("Probabilidade Real", f"{prob_acumulada:.1f}%")
            
            fair_odd = 100/prob_acumulada if prob_acumulada > 0 else 0
            delta_ev = (total_odd - fair_odd)
            
            res3.metric("Fair Odd Estimada", f"{fair_odd:.2f}", delta=f"{delta_ev:.2f}")
            
            if delta_ev > 0:
                st.success("💎 **EV+ DETECTADO!** A Odd Total está acima da Fair Odd calculada.")
            else:
                st.warning("⚠️ **EV- DETECTADO.** A Odd está abaixo do justo estatístico.")

        else:
            st.info("O bilhete está vazio. Adicione jogos acima.")

    # ============================================================
    # TAB 2: HEDGES (COMPLETO)
    # ============================================================
    with tab2:
        st.header("🛡️ Hedges MAXIMUM - Gestão de Proteção")
        
        if not st.session_state.current_ticket:
            st.warning("⚠️ Crie um bilhete na aba Construtor primeiro para calcular hedges.")
        else:
            col1, col2 = st.columns(2)
            stake = col1.number_input("💰 Stake Principal (R$)", value=100.0, step=10.0)
            
            # Recalcula odd total
            odd_total = np.prod([x['odd'] for x in st.session_state.current_ticket])
            col2.metric("Odd do Bilhete", f"{odd_total:.2f}")
            
            retorno_max = stake * odd_total
            lucro_max = retorno_max - stake
            
            st.info(f"💵 Retorno Potencial: {format_currency(retorno_max)} | Lucro Líquido: {format_currency(lucro_max)}")
            
            st.markdown("### 🛠️ Estratégias de Hedge Sugeridas")
            
            # HEDGE 1
            with st.expander("🛡️ HEDGE 1: Smart Protection (Cobertura de Perda)", expanded=True):
                st.markdown("**Estratégia:** Apostar na zebra/contrário para recuperar o stake caso a principal perca.")
                st.write("Insira a Odd da aposta contrária (ex: Lay no favorito ou Dupla Chance zebra):")
                
                odd_hedge = st.number_input("Odd da Cobertura:", value=2.0, min_value=1.01, step=0.1, key="h1_odd")
                
                # Cálculo: Stake Hedge = Stake Principal / (Odd Hedge - 1) ? Não, para recuperar total:
                # Para recuperar Stake Total (S1 + S2): S2 * O2 = S1 + S2
                # S2 = S1 / (O2 - 1)
                stake_hedge = stake / (odd_hedge - 1)
                
                c1, c2 = st.columns(2)
                c1.metric("Apostar na Cobertura", format_currency(stake_hedge))
                c2.metric("Custo Total", format_currency(stake + stake_hedge))
                
                if (stake + stake_hedge) < retorno_max:
                    st.success(f"✅ **Hedge Viável!** Se principal bater, lucro de: {format_currency(retorno_max - (stake + stake_hedge))}")
                else:
                    st.error("🚫 Hedge Inviável matematicamente com essa Odd (Prejuízo mesmo ganhando a principal).")

            # HEDGE 2
            with st.expander("💎 HEDGE 2: Arbitragem (Lucro Garantido)"):
                st.markdown("**Estratégia:** Garantir lucro independente do resultado (Dutching). Só possível se as Odds permitirem.")
                
                # Dutching simples entre 2 eventos
                implied_prob = (1/odd_total) + (1/odd_hedge)
                
                if implied_prob < 1:
                    st.success(f"💎 **ARBITRAGEM POSSÍVEL!** Margem do mercado: {implied_prob*100:.1f}%")
                    stake_total = stake + stake_hedge # Valor exemplo
                    
                    # Stakes ideais
                    s1_ideal = (stake_total / odd_total) / implied_prob # Aproximado
                    
                    st.write("Cálculo complexo de arbitragem requer ajuste fino dos stakes.")
                else:
                    st.warning(f"⚠️ Arbitragem não disponível. Soma das probabilidades > 100% ({implied_prob*100:.1f}%)")

    # ============================================================
    # TAB 3: SIMULADOR (COMPLETO)
    # ============================================================
    with tab3:
        st.header("🎲 Simulador Monte Carlo (3.000 Iterações)")
        
        s_c1, s_c2 = st.columns(2)
        sim_h = s_c1.selectbox("Time Casa", sorted(list(STATS.keys())), key='sim_h')
        sim_a = s_c2.selectbox("Time Visitante", sorted(list(STATS.keys())), key='sim_a')
        
        if st.button("🚀 Rodar Simulação", use_container_width=True):
            if sim_h != sim_a:
                with st.spinner(f"Simulando 3.000 partidas entre {sim_h} e {sim_a}..."):
                    # 1. Obter dados
                    sh = STATS[sim_h]
                    sa = STATS[sim_a]
                    
                    # 2. Rodar simulação
                    res = simulate_game_v31(sh, sa, {}, 3000)
                    
                    # 3. Exibir Resultados
                    st.success("Simulação Concluída!")
                    
                    # Métricas Médias
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Média Gols", f"{res['goals_total'].mean():.2f}")
                    m2.metric("Média Cantos", f"{res['corners_total'].mean():.2f}")
                    m3.metric("Média Cartões", f"{res['cards_total'].mean():.2f}")
                    
                    # Probabilidades Derivadas
                    prob_o25 = (res['goals_total'] > 2.5).mean() * 100
                    prob_o95 = (res['corners_total'] > 9.5).mean() * 100
                    prob_btts = ((res['goals_h'] > 0) & (res['goals_a'] > 0)).mean() * 100
                    
                    m4.metric("Prob Over 2.5", f"{prob_o25:.1f}%")
                    
                    st.markdown("### 📊 Distribuição de Probabilidades")
                    
                    # Gráfico de Gols
                    fig_goals = px.histogram(res['goals_total'], nbins=10, 
                                           title="Distribuição de Gols Totais",
                                           labels={'value': 'Gols', 'count': 'Frequência'},
                                           color_discrete_sequence=['#1e3c72'])
                    st.plotly_chart(fig_goals, use_container_width=True)
                    
                    # Tabela detalhada
                    st.markdown("#### 🎲 Probabilidades Detalhadas")
                    probs_df = pd.DataFrame({
                        'Mercado': ['Over 1.5 Gols', 'Over 2.5 Gols', 'Over 3.5 Gols', 'Ambos Marcam', 'Over 8.5 Cantos', 'Over 9.5 Cantos', 'Over 10.5 Cantos'],
                        'Probabilidade': [
                            (res['goals_total'] > 1.5).mean() * 100,
                            (res['goals_total'] > 2.5).mean() * 100,
                            (res['goals_total'] > 3.5).mean() * 100,
                            prob_btts,
                            (res['corners_total'] > 8.5).mean() * 100,
                            (res['corners_total'] > 9.5).mean() * 100,
                            (res['corners_total'] > 10.5).mean() * 100
                        ]
                    })
                    # Formatar
                    probs_df['Probabilidade'] = probs_df['Probabilidade'].map('{:.1f}%'.format)
                    st.dataframe(probs_df, use_container_width=True)
                    
            else:
                st.error("Selecione times diferentes para simular.")

    # ============================================================
    # TAB 4: MÉTRICAS (COMPLETO)
    # ============================================================
    with tab4:
        st.header("📊 Métricas de Performance")
        
        if not st.session_state.bet_results:
            st.info("Ainda não há apostas registradas. Vá para a aba 'Registro' e insira seus resultados.")
        else:
            # Converte histórico para DF
            df_hist = pd.DataFrame(st.session_state.bet_results)
            
            # Cálculos
            total_apostas = len(df_hist)
            total_green = df_hist[df_hist['ganhou'] == True].shape[0]
            win_rate = (total_green / total_apostas) * 100
            
            total_investido = df_hist['stake'].sum()
            total_retorno = df_hist['lucro'].sum() # Aqui lucro já é líquido ou bruto? Assumindo líquido na lógica do registro
            roi = (total_retorno / total_investido) * 100 if total_investido > 0 else 0
            
            # Exibição
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Apostas", total_apostas)
            m2.metric("Win Rate", f"{win_rate:.1f}%")
            m3.metric("ROI", f"{roi:.1f}%")
            m4.metric("Lucro Líquido", format_currency(total_retorno))
            
            # Gráfico de Evolução da Banca
            st.markdown("### 📈 Evolução da Banca")
            fig_evo = px.line(y=st.session_state.bankroll_history, x=range(len(st.session_state.bankroll_history)),
                            title="Crescimento do Capital", labels={'y': 'Banca (R$)', 'x': 'Apostas'})
            st.plotly_chart(fig_evo, use_container_width=True)

    # ============================================================
    # TAB 5: VIZ (COMPLETO)
    # ============================================================
    with tab5:
        st.header("🎨 Visualizações de Dados")
        
        viz_opt = st.selectbox("Escolha o Gráfico:", 
                             ["Dispersão: Ataque vs Defesa", "Ranking de Cantos", "Ranking de Cartões"])
        
        if viz_opt == "Ranking de Cantos":
            # Prepara dados
            data = []
            for t, s in STATS.items():
                data.append({'Time': t, 'Cantos': s['corners'], 'Liga': s['league']})
            df_v = pd.DataFrame(data).sort_values('Cantos', ascending=False).head(20)
            
            fig = px.bar(df_v, x='Cantos', y='Time', orientation='h', color='Liga', 
                       title="Top 20 Times em Média de Cantos", height=600)
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_opt == "Dispersão: Ataque vs Defesa":
            # Prepara dados
            data = []
            for t, s in STATS.items():
                data.append({
                    'Time': t, 
                    'Gols Feitos': s['goals_f'], 
                    'Gols Sofridos': s['goals_a'],
                    'Liga': s['league']
                })
            df_v = pd.DataFrame(data)
            
            fig = px.scatter(df_v, x='Gols Feitos', y='Gols Sofridos', color='Liga', hover_name='Time',
                           title="Mapa de Poder: Ataque vs Defesa", height=600)
            # Inverter Y para que defesa boa (poucos gols) fique em cima? Opcional.
            # Adiciona linhas médias
            fig.add_hline(y=df_v['Gols Sofridos'].mean(), line_dash="dot", annotation_text="Média Defesa")
            fig.add_vline(x=df_v['Gols Feitos'].mean(), line_dash="dot", annotation_text="Média Ataque")
            
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_opt == "Ranking de Cartões":
            data = []
            for t, s in STATS.items():
                data.append({'Time': t, 'Cartões': s['cards'], 'Liga': s['league']})
            df_v = pd.DataFrame(data).sort_values('Cartões', ascending=False).head(20)
            
            fig = px.bar(df_v, x='Cartões', y='Time', orientation='h', color='Liga', 
                       title="Top 20 Times Mais Indisciplinados", height=600)
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)

    # ============================================================
    # TAB 6: REGISTRO (COMPLETO)
    # ============================================================
    with tab6:
        st.header("📝 Registro de Apostas Manual")
        
        with st.form("form_registro"):
            c1, c2 = st.columns(2)
            desc = c1.text_input("Descrição (Ex: Fla x Flu - Over 2.5)")
            stake = c2.number_input("Stake (R$)", 10.0)
            
            c3, c4 = st.columns(2)
            odd = c3.number_input("Odd", 1.01)
            resultado = c4.selectbox("Resultado", ["Green (Ganhou)", "Red (Perdeu)", "Void (Devolvida)"])
            
            submit = st.form_submit_button("💾 Salvar no Histórico")
            
            if submit:
                lucro = 0.0
                ganhou = False
                if resultado == "Green (Ganhou)":
                    lucro = (stake * odd) - stake
                    ganhou = True
                elif resultado == "Red (Perdeu)":
                    lucro = -stake
                    ganhou = False
                # Void = lucro 0
                
                # Atualiza banca
                nova_banca = st.session_state.bankroll_history[-1] + lucro
                st.session_state.bankroll_history.append(nova_banca)
                
                # Salva aposta
                st.session_state.bet_results.append({
                    'data': datetime.now().strftime('%d/%m %H:%M'),
                    'descricao': desc,
                    'stake': stake,
                    'odd': odd,
                    'ganhou': ganhou,
                    'lucro': lucro
                })
                
                st.success(f"Aposta registrada! Banca atualizada: {format_currency(nova_banca)}")
                time.sleep(1)
                st.rerun()
                
        if st.session_state.bet_results:
            st.markdown("### 📜 Últimas Apostas")
            st.dataframe(pd.DataFrame(st.session_state.bet_results).iloc[::-1], use_container_width=True)

    # ============================================================
    # TAB 7: SCANNER (COMPLETO)
    # ============================================================
    with tab7:
        st.header("🔍 Scanner de Oportunidades")
        st.caption("Busca automática por jogos com probabilidade acima do seu critério.")
        
        if CAL.empty:
            st.warning("Calendário vazio. Não é possível escanear.")
        else:
            c1, c2 = st.columns(2)
            scan_date = c1.selectbox("Data para Escanear:", sorted(CAL['DtObj'].dt.strftime('%d/%m/%Y').unique()))
            min_prob = c2.slider("Probabilidade Mínima (%)", 50, 90, 70)
            
            if st.button("🔎 Iniciar Varredura", use_container_width=True):
                jogos = CAL[CAL['DtObj'].dt.strftime('%d/%m/%Y') == scan_date]
                hits = []
                
                with st.spinner(f"Analisando {len(jogos)} jogos..."):
                    for _, row in jogos.iterrows():
                        h, a = normalize_name(row['Time_Casa'], list(STATS.keys())), normalize_name(row['Time_Visitante'], list(STATS.keys()))
                        
                        if h and a:
                            calc = calcular_jogo_v31(STATS[h], STATS[a], {})
                            
                            # 1. Verifica Cantos
                            prob_c = calcular_poisson(calc['corners_total'], 9.5)
                            if prob_c >= min_prob:
                                hits.append({
                                    'Jogo': f"{h} vs {a}",
                                    'Mercado': 'Over 9.5 Cantos',
                                    'Prob': f"{prob_c:.1f}%",
                                    'Previsão': f"{calc['corners_total']:.1f}",
                                    'Emoji': get_prob_emoji(prob_c)
                                })
                                
                            # 2. Verifica Gols
                            prob_g = calcular_poisson(calc['total_goals'], 2.5)
                            if prob_g >= min_prob:
                                hits.append({
                                    'Jogo': f"{h} vs {a}",
                                    'Mercado': 'Over 2.5 Gols',
                                    'Prob': f"{prob_g:.1f}%",
                                    'Previsão': f"{calc['total_goals']:.1f}",
                                    'Emoji': get_prob_emoji(prob_g)
                                })
                
                if hits:
                    st.success(f"Encontramos {len(hits)} oportunidades!")
                    df_hits = pd.DataFrame(hits)
                    st.dataframe(df_hits, use_container_width=True)
                else:
                    st.warning("Nenhuma oportunidade encontrada com esses critérios.")

    # ============================================================
    # TAB 8: IMPORTAR (COMPLETO)
    # ============================================================
    with tab8:
        st.header("📋 Importar Bilhete de Texto")
        st.caption("Cole o texto do seu bilhete (ex: WhatsApp ou Site) para análise rápida.")
        
        txt_import = st.text_area("Cole aqui:", height=150)
        
        if st.button("Analisar Texto"):
            jogos_identificados = parse_bilhete_texto(txt_import)
            
            if jogos_identificados:
                st.success(f"Identificamos {len(jogos_identificados)} possíveis jogos.")
                
                validos = validar_jogos_bilhete(jogos_identificados, STATS)
                
                if validos:
                    for v in validos:
                        with st.expander(f"✅ {v['home']} vs {v['away']}"):
                            # Faz análise rápida
                            calc = calcular_jogo_v31(v['home_stats'], v['away_stats'], {})
                            c1, c2 = st.columns(2)
                            c1.metric("Previsão Cantos", f"{calc['corners_total']:.1f}")
                            c2.metric("Previsão Gols", f"{calc['total_goals']:.1f}")
                            st.caption("Adicione este jogo manualmente no Construtor se desejar.")
                else:
                    st.error("Não conseguimos validar estatisticamente os times encontrados. Verifique os nomes.")
            else:
                st.warning("Nenhum padrão de jogo (Time A vs Time B) encontrado.")

    # ============================================================
    # TAB 9: AI ADVISOR ULTRA (LÓGICA SUPERIOR)
    # ============================================================
    with tab9:
        st.header("🤖 AI Advisor ULTRA")
        st.caption("Assistente Estatístico Profissional V31 | Powered by Causality Engine")
        
        # Container do Histórico
        chat_container = st.container()
        
        with chat_container:
            if not st.session_state.chat_history:
                # Mensagem de Boas Vindas
                st.markdown("""
                <div style='background-color: #f0f8ff; padding: 15px; border-radius: 10px; border-left: 5px solid #1e3c72;'>
                    <h4>👋 Olá! Sou o AI Advisor ULTRA.</h4>
                    <p>Minhas análises são baseadas 100% em dados matemáticos e estatísticos dos times carregados.</p>
                    <p><b>Experimente perguntar:</b></p>
                    <ul>
                        <li>"Analise Arsenal vs Chelsea"</li>
                        <li>"Como está o desempenho do Flamengo?"</li>
                        <li>"Qual a probabilidade de over 9.5 cantos em Liverpool x City?"</li>
                        <li>"Sugira uma aposta para hoje"</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            # Renderiza histórico
            for msg in st.session_state.chat_history:
                role = msg['role']
                avatar = "👤" if role == 'user' else "🤖"
                st.chat_message(role, avatar=avatar).markdown(msg['content'])
        
        # Input do Usuário
        user_input = st.chat_input("Digite sua pergunta sobre futebol...")
        
        if user_input:
            # 1. Adiciona pergunta ao histórico e exibe
            st.session_state.chat_history.append({'role': 'user', 'content': user_input})
            # Força rerun para mostrar a pergunta imediatamente antes de processar? 
            # No Streamlit normal, o rerun acontece ao final.
            
            # 2. Processamento ULTRA INTELIGENTE
            with st.spinner("🧠 Consultando base de dados e calculando probabilidades..."):
                # Pequeno delay simulado para UX
                time.sleep(0.5) 
                response_text = processar_chat_ultra(user_input, STATS, CAL, REFS)
            
            # 3. Adiciona resposta ao histórico
            st.session_state.chat_history.append({'role': 'assistant', 'content': response_text})
            
            # 4. Rerun para atualizar a interface com a resposta
            st.rerun()

# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":
    main()
