"""
FutPrevisão V13.2 (Refatorado)
Sistema de análise de apostas esportivas com V12.0 Causality Engine
Autor: Refatorado por Claude
Data: 17/12/2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import math
import difflib
from datetime import datetime
from typing import Dict, Tuple, Optional, List

# ==============================================================================
# 0. CONFIGURAÇÃO & CONSTANTES
# ==============================================================================

st.set_page_config(
    page_title="FutPrevisão V13.2 (Refatorado)", 
    layout="wide", 
    page_icon="⚽"
)

# CSS Customizado
st.markdown("""
<style>
    .stProgress > div > div > div > div { 
        background-color: #4CAF50; 
    }
    .metric-card { 
        background: #ffffff; 
        border: 2px solid #e0e0e0; 
        padding: 15px; 
        border-radius: 8px; 
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .opportunity-high { 
        border-left: 5px solid #28a745;
        background: #d4edda;
    }
    .opportunity-medium { 
        border-left: 5px solid #ffc107;
        background: #fff3cd;
    }
    .stat-positive { color: #28a745; font-weight: bold; }
    .stat-neutral { color: #ffc107; font-weight: bold; }
    .stat-negative { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Configuração de OCR (Opcional)
HAS_OCR = False
try:
    import pytesseract
    TESSERACT_PATHS = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        r"C:\Users\Kaiqu\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
    ]
    for path in TESSERACT_PATHS:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            HAS_OCR = True
            break
except ImportError:
    pass

# ==============================================================================
# 1. CONFIGURAÇÕES & MAPEAMENTOS
# ==============================================================================

# Mapeamento de arquivos por liga
LEAGUE_FILES = {
    "Premier League": {
        "csv": "Premier League 25.26.csv",
        "txt": "Prewmier League.txt",
        "txt_cards": "Cartoes Premier League - Inglaterra.txt"
    },
    "La Liga": {
        "csv": "La Liga 25.26.csv",
        "txt": "Escanteios Espanha.txt",
        "txt_cards": "Cartoes La Liga - Espanha.txt"
    },
    "Serie A": {
        "csv": "Serie A 25.26.csv",
        "txt": "Escanteios Italia.txt",
        "txt_cards": "Cartoes Serie A - Italia.txt"
    },
    "Bundesliga": {
        "csv": "Bundesliga 25.26.csv",
        "txt": "Escanteios Alemanha.txt",
        "txt_cards": "Cartoes Bundesliga - Alemanha.txt"
    },
    "Ligue 1": {
        "csv": "Ligue 1 25.26.csv",
        "txt": "Escanteios França.txt",
        "txt_cards": "Cartoes Ligue 1 - França.txt"
    },
    "Championship": {
        "csv": "Championship Inglaterra 25.26.csv",
        "txt": "Championship Escanteios Inglaterra.txt",
        "txt_cards": None
    },
    "Bundesliga 2": {
        "csv": "Bundesliga 2.csv",
        "txt": "Bundesliga 2.txt",
        "txt_cards": None
    },
    "Pro League (BEL)": {
        "csv": "Pro League Belgica 25.26.csv",
        "txt": "Pro League Belgica.txt",
        "txt_cards": None
    },
    "Süper Lig (TUR)": {
        "csv": "Super Lig Turquia 25.26.csv",
        "txt": "Super Lig Turquia.txt",
        "txt_cards": None
    },
    "Premiership (SCO)": {
        "csv": "Premiership Escocia 25.26.csv",
        "txt": "Premiership Escocia.txt",
        "txt_cards": None
    }
}

# Mapeamento de nomes de times (normalização)
NAME_MAPPING = {
    "Man City": "Man City",
    "Man Utd": "Man United",
    "Sheffield Wed": "Sheffield Wednesday",
    "Forest": "Nott'm Forest",
    "Wolves": "Wolverhampton",
    "Spurs": "Tottenham",
    "Atl. Madrid": "Atl Madrid",
    "Athletic Club": "Ath Bilbao",
    "PSG": "Paris SG",
    "St Etienne": "Saint-Etienne",
    "Fenerbahce": "Fenerbahçe",
    "Galatasaray": "Galatasaray",
    "Rangers": "Rangers",
    "Celtic": "Celtic"
}

# Thresholds do sistema V12.0 Causality Engine
THRESHOLDS = {
    'fouls_violent': 12.5,      # Acima = time violento (remove penalidade)
    'goals_pressure': 1.8,      # Acima = time ofensivo (boost escanteios)
    'prob_elite': 75,           # Probabilidade mínima para "elite"
    'prob_elite_cards': 70,     # Probabilidade mínima para cartões elite
    'hist_validation': 70       # % histórico mínimo para validação
}

# Multiplicadores V12.0
MULTIPLIERS = {
    'home_corners': 1.15,       # Boost mandante escanteios
    'away_corners': 0.90,       # Penalidade visitante escanteios
    'pressure_boost': 1.10,     # Boost pressão ofensiva
    'violence_safe': 0.85,      # Penalidade time disciplinado
    'violence_active': 1.0,     # Sem penalidade (time violento)
    'goals_balance': 1.3        # Balanceamento xG
}

# ==============================================================================
# 2. SISTEMA DE ÁRBITROS
# ==============================================================================

@st.cache_data(ttl=3600)
def load_referees_unified() -> Dict[str, float]:
    """
    Carrega banco de dados unificado de árbitros.
    
    Prioridade:
    1. arbitros_5_ligas_2025_2026.csv (62 árbitros secundários)
    2. arbitros.csv (árbitros principais)
    
    Returns:
        Dict[str, float]: {nome_arbitro: fator_cartoes}
    """
    refs = {}
    
    # Arquivo 1: 5 ligas secundárias
    file_secondary = "arbitros_5_ligas_2025_2026.csv"
    if os.path.exists(file_secondary):
        try:
            df = pd.read_csv(file_secondary)
            for _, row in df.iterrows():
                nome = str(row['Arbitro']).strip()
                fator = float(row['Media_Cartoes_Por_Jogo']) / 4.0  # Normaliza para fator
                refs[nome] = fator
        except Exception as e:
            st.warning(f"⚠️ Erro ao carregar {file_secondary}: {e}")
    
    # Arquivo 2: Árbitros principais (não sobrescreve)
    file_main = "arbitros.csv"
    if os.path.exists(file_main):
        try:
            df = pd.read_csv(file_main)
            for _, row in df.iterrows():
                nome = str(row['Nome']).strip()
                if nome not in refs:  # Só adiciona se não existir
                    refs[nome] = float(row['Fator'])
        except Exception as e:
            st.warning(f"⚠️ Erro ao carregar {file_main}: {e}")
    
    return refs


@st.cache_data(ttl=3600)
def get_ref_factor(name: Optional[str]) -> float:
    """
    Retorna fator de cartões do árbitro (normalizado).
    
    Args:
        name: Nome do árbitro
        
    Returns:
        float: Fator de cartões (0.8-1.3, default=1.0 neutro)
    """
    if not name or str(name).lower() in ["nan", "none", "neutro", "desconhecido", ""]:
        return 1.0
    
    refs_db = load_referees_unified()
    
    # Busca exata
    if name in refs_db:
        return refs_db[name]
    
    # Busca fuzzy (similaridade 70%)
    match = difflib.get_close_matches(name, refs_db.keys(), n=1, cutoff=0.7)
    if match:
        return refs_db[match[0]]
    
    # Fallback conservador
    return 0.90  # Árbitro desconhecido = leniente


# ==============================================================================
# 3. LEITOR DE CALENDÁRIO
# ==============================================================================

@st.cache_data(ttl=600)
def load_calendar_file() -> Tuple[pd.DataFrame, str]:
    """
    Carrega arquivo calendario_ligas.csv com validação robusta.
    
    Returns:
        Tuple[pd.DataFrame, str]: (dataframe, mensagem_status)
    """
    filename = "calendario_ligas.csv"
    
    if not os.path.exists(filename):
        return pd.DataFrame(), f"❌ Arquivo '{filename}' não encontrado"
    
    try:
        # Tenta UTF-8 primeiro, depois latin1
        try:
            df = pd.read_csv(filename, dtype=str)
        except UnicodeDecodeError:
            df = pd.read_csv(filename, dtype=str, encoding='latin1')
        
        # Normaliza nomes de colunas
        df.columns = [c.strip().replace(' ', '_') for c in df.columns]
        
        # Valida colunas obrigatórias
        required_cols = ['Data', 'Liga', 'Time_Casa', 'Time_Visitante', 'Hora']
        missing_cols = set(required_cols) - set(df.columns)
        
        if missing_cols:
            return pd.DataFrame(), f"❌ Colunas faltando: {missing_cols}"
        
        # Converte data para datetime
        df['DtObj'] = pd.to_datetime(df['Data'], format="%d/%m/%Y", errors='coerce')
        
        # Remove linhas com data inválida
        invalid_dates = df['DtObj'].isna().sum()
        if invalid_dates > 0:
            st.warning(f"⚠️ {invalid_dates} jogos com datas inválidas foram removidos")
        
        df = df.dropna(subset=['DtObj'])
        
        # Ordena por data e hora
        df = df.sort_values(['DtObj', 'Hora'], ascending=[True, True])
        
        # Adiciona aliases para compatibilidade
        if 'Time_Casa' in df.columns:
            df['Mandante'] = df['Time_Casa']
        if 'Time_Visitante' in df.columns:
            df['Visitante'] = df['Time_Visitante']
        
        return df, f"✅ {len(df)} jogos carregados com sucesso"
        
    except Exception as e:
        return pd.DataFrame(), f"❌ Erro ao processar: {str(e)}"


# ==============================================================================
# 4. MOTOR DE ESTATÍSTICAS
# ==============================================================================

@st.cache_data(ttl=3600)
def learn_stats() -> Dict[str, Dict[str, float]]:
    """
    Carrega estatísticas de todos os times de todos os CSVs.
    
    Extrai:
    - Escanteios (HC/AC)
    - Cartões (HY/AY)
    - Faltas (HF/AF) - para V12.0 Causality
    - Gols (FTHG/FTAG) - para xG
    
    Returns:
        Dict: {time: {corners, cards, fouls, goals_f, goals_a, league}}
    """
    db = {}
    
    for liga_key, files in LEAGUE_FILES.items():
        csv_file = files['csv']
        
        if not os.path.exists(csv_file):
            continue
        
        try:
            # Carrega CSV
            try:
                df = pd.read_csv(csv_file, encoding='latin1')
            except:
                df = pd.read_csv(csv_file)
            
            cols = df.columns
            
            # Identifica colunas de times
            home_col = next((c for c in cols if c in ['HomeTeam', 'Mandante']), None)
            away_col = next((c for c in cols if c in ['AwayTeam', 'Visitante']), None)
            
            if not home_col or not away_col:
                continue
            
            # Verifica disponibilidade de dados
            has_corners = 'HC' in cols and 'AC' in cols
            has_cards = 'HY' in cols and 'AY' in cols
            has_fouls = 'HF' in cols and 'AF' in cols
            has_goals = 'FTHG' in cols and 'FTAG' in cols
            
            # Extrai times únicos
            teams = set(df[home_col].dropna().unique()).union(
                set(df[away_col].dropna().unique())
            )
            
            # Processa cada time
            for team in teams:
                # Jogos em casa
                home_games = df[df[home_col] == team].copy()
                # Jogos fora
                away_games = df[df[away_col] == team].copy()
                
                total_games = len(home_games) + len(away_games)
                
                # Mínimo 3 jogos para estatística válida
                if total_games < 3:
                    continue
                
                # ESCANTEIOS
                if has_corners:
                    corners_made = home_games['HC'].sum() + away_games['AC'].sum()
                    corners_avg = corners_made / total_games
                else:
                    corners_avg = 5.0  # Fallback
                
                # CARTÕES
                if has_cards:
                    cards_yellow = home_games['HY'].sum() + away_games['AY'].sum()
                    cards_avg = cards_yellow / total_games
                else:
                    cards_avg = 2.0  # Fallback
                
                # FALTAS (V12.0)
                if has_fouls:
                    fouls_made = home_games['HF'].sum() + away_games['AF'].sum()
                    fouls_avg = fouls_made / total_games
                else:
                    fouls_avg = 11.0  # Fallback (média geral)
                
                # GOLS (xG)
                if has_goals:
                    try:
                        goals_for = (
                            home_games['FTHG'].astype(float).sum() + 
                            away_games['FTAG'].astype(float).sum()
                        ) / total_games
                        
                        goals_against = (
                            home_games['FTAG'].astype(float).sum() + 
                            away_games['FTHG'].astype(float).sum()
                        ) / total_games
                    except:
                        goals_for, goals_against = 1.2, 1.2
                else:
                    goals_for, goals_against = 1.2, 1.2
                
                # Salva estatísticas
                db[team] = {
                    'corners': round(corners_avg, 2),
                    'cards': round(cards_avg, 2),
                    'fouls': round(fouls_avg, 2),
                    'goals_f': round(goals_for, 2),
                    'goals_a': round(goals_against, 2),
                    'league': liga_key,
                    'games': total_games
                }
        
        except Exception as e:
            st.warning(f"⚠️ Erro ao processar {liga_key}: {e}")
            continue
    
    return db


# ==============================================================================
# 5. LOADER DE HISTÓRICO (Adam Choi)
# ==============================================================================

class HistoryLoader:
    """
    Carrega dados históricos de escanteios e cartões (Adam Choi Format).
    """
    
    def __init__(self):
        self.corn = {}   # Escanteios
        self.card = {}   # Cartões
        self.load()
    
    def load(self):
        """Carrega todos os arquivos .txt de histórico."""
        for league, files in LEAGUE_FILES.items():
            # Escanteios
            if files['txt'] and os.path.exists(files['txt']):
                try:
                    with open(files['txt'], 'r', encoding='utf-8') as f:
                        raw = f.read().strip()
                        # Remove lixo antes do JSON
                        if '{' in raw:
                            raw = raw[raw.find('{'):]
                        self.corn[league] = json.loads(raw)
                except Exception as e:
                    pass  # Silencioso
            
            # Cartões
            if files['txt_cards'] and os.path.exists(files['txt_cards']):
                try:
                    with open(files['txt_cards'], 'r', encoding='utf-8') as f:
                        raw = f.read().strip()
                        if '{' in raw:
                            raw = raw[raw.find('{'):]
                        self.card[league] = json.loads(raw)
                except Exception as e:
                    pass  # Silencioso
    
    def get_global(self, team: str, market: str, key: str) -> Optional[Tuple[int, int, float]]:
        """
        Busca histórico global (todas as ligas).
        
        Args:
            team: Nome do time
            market: 'corners' ou 'cards'
            key: 'homeTeamOver35', 'awayCardsOver15', etc.
            
        Returns:
            Tuple[int, int, float] ou None: (acertos, total, %)
        """
        source = self.corn if market == 'corners' else self.card
        
        for league in source:
            result = self._find_in_league(source[league], team, key)
            if result:
                return result
        
        return None
    
    def _find_in_league(self, data: dict, team: str, key: str) -> Optional[Tuple[int, int, float]]:
        """Busca time específico no JSON."""
        if not data:
            return None
        
        # Extrai lista de times
        teams = [t['teamName'] for t in data.get('teams', [])]
        
        # Aplica mapeamento manual primeiro
        target = NAME_MAPPING.get(team, team)
        
        # Busca fuzzy (60% similaridade)
        match = difflib.get_close_matches(target, teams, n=1, cutoff=0.6)
        
        if match:
            for t in data['teams']:
                if t['teamName'] == match[0]:
                    hist_data = t.get(key)
                    if hist_data and len(hist_data) >= 3:
                        return (hist_data[0], hist_data[1], hist_data[2])
        
        return None


# ==============================================================================
# 6. MOTOR DE CÁLCULO V12.0 (CAUSALITY ENGINE)
# ==============================================================================

def poisson_prob(k: int, lam: float) -> float:
    """Calcula P(X = k) para distribuição de Poisson."""
    return (math.exp(-lam) * (lam ** k)) / math.factorial(int(k))


def prob_over(lam: float, line: float) -> float:
    """
    Calcula probabilidade de Over X.5 usando Poisson.
    
    Args:
        lam: Lambda (expectativa)
        line: Linha (ex: 3.5, 9.5)
        
    Returns:
        float: Probabilidade em % (0-100)
    """
    try:
        # CDF = P(X <= line)
        cdf = sum(poisson_prob(i, lam) for i in range(int(line) + 1))
        # P(X > line) = 1 - CDF
        return max(0.0, min(100.0, (1 - cdf) * 100))
    except:
        return 0.0


def normalize_team(name: str, stats_db: dict) -> Optional[str]:
    """
    Normaliza nome do time usando fuzzy matching.
    
    Args:
        name: Nome do time (pode estar errado)
        stats_db: Base de dados de estatísticas
        
    Returns:
        str ou None: Nome normalizado ou None se não encontrado
    """
    if name in stats_db:
        return name
    
    # Busca fuzzy 60%
    match = difflib.get_close_matches(name, stats_db.keys(), n=1, cutoff=0.6)
    return match[0] if match else None


def calcular_jogo(home_raw: str, away_raw: str, ref_name: Optional[str], 
                  stats_db: dict) -> Optional[Dict]:
    """
    Motor de cálculo V12.0 - Causality Engine.
    
    Aplica:
    - Boost de pressão ofensiva (gols > 1.8)
    - Boost de violência (faltas > 12.5)
    - Fator árbitro
    - Mando de campo
    
    Args:
        home_raw: Time mandante (nome bruto)
        away_raw: Time visitante (nome bruto)
        ref_name: Nome do árbitro (opcional)
        stats_db: Base de estatísticas
        
    Returns:
        Dict ou None: {corners, cards, goals} ou None se times não encontrados
    """
    # Normaliza nomes
    home = normalize_team(home_raw, stats_db)
    away = normalize_team(away_raw, stats_db)
    
    # Fallback seguro para times não encontrados
    default_home = {
        'corners': 5.0, 'cards': 2.0, 'fouls': 11.0, 
        'goals_f': 1.2, 'goals_a': 1.2
    }
    default_away = {
        'corners': 4.0, 'cards': 2.0, 'fouls': 11.0, 
        'goals_f': 1.0, 'goals_a': 1.5
    }
    
    s_home = stats_db.get(home, default_home)
    s_away = stats_db.get(away, default_away)
    
    # Fator árbitro
    ref_factor = get_ref_factor(ref_name)
    
    # === ESCANTEIOS (V12.0) ===
    # Boost de pressão: times que marcam >1.8 gols ganham +10%
    pressure_home = (MULTIPLIERS['pressure_boost'] 
                     if s_home['goals_f'] > THRESHOLDS['goals_pressure'] 
                     else 1.0)
    
    corners_home = (s_home['corners'] * 
                    MULTIPLIERS['home_corners'] * 
                    pressure_home)
    
    corners_away = s_away['corners'] * MULTIPLIERS['away_corners']
    
    # === CARTÕES (V12.0) ===
    # Violência: times que fazem >12.5 faltas REMOVEM penalidade
    violence_home = (MULTIPLIERS['violence_active'] 
                     if s_home['fouls'] > THRESHOLDS['fouls_violent'] 
                     else MULTIPLIERS['violence_safe'])
    
    violence_away = (MULTIPLIERS['violence_active'] 
                     if s_away['fouls'] > THRESHOLDS['fouls_violent'] 
                     else MULTIPLIERS['violence_safe'])
    
    cards_home = s_home['cards'] * violence_home * ref_factor
    cards_away = s_away['cards'] * violence_away * ref_factor
    
    # === GOLS (xG) ===
    # xG casa = ataque_casa × defesa_visitante / balance
    xg_home = (s_home['goals_f'] * s_away['goals_a']) / MULTIPLIERS['goals_balance']
    xg_away = (s_away['goals_f'] * s_home['goals_a']) / MULTIPLIERS['goals_balance']
    
    return {
        'corners': {
            'h': round(corners_home, 2),
            'a': round(corners_away, 2),
            't': round(corners_home + corners_away, 2)
        },
        'cards': {
            'h': round(cards_home, 2),
            'a': round(cards_away, 2),
            't': round(cards_home + cards_away, 2)
        },
        'goals': {
            'h': round(xg_home, 2),
            'a': round(xg_away, 2)
        },
        'metadata': {
            'ref_factor': round(ref_factor, 2),
            'pressure_home': pressure_home > 1.0,
            'violence_home': violence_home == 1.0,
            'violence_away': violence_away == 1.0
        }
    }


# ==============================================================================
# 7. FUNÇÕES DE UTILIDADE
# ==============================================================================

def fmt_hist(data: Optional[Tuple]) -> str:
    """Formata dados históricos: (hits, total, %)."""
    return f"{data[1]}/{data[0]}" if data else "N/A"


def check_elite(prob: float, hist_data: Optional[Tuple], is_card: bool = False) -> bool:
    """
    Verifica se aposta é 'elite' (dupla validação).
    
    Critérios:
    - Probabilidade Poisson >= 75% (70% para cartões)
    - Histórico >= 70% OU probabilidade >= 80%
    """
    cutoff = THRESHOLDS['prob_elite_cards'] if is_card else THRESHOLDS['prob_elite']
    
    if hist_data:
        # Dupla validação: Poisson + Histórico
        return (float(hist_data[2]) >= THRESHOLDS['hist_validation'] and 
                prob >= cutoff)
    
    # Sem histórico: requer 80%+
    return prob >= (cutoff + 5)


def get_color(prob: float) -> str:
    """Retorna cor baseada em probabilidade."""
    if prob >= 70:
        return "green"
    elif prob >= 50:
        return "orange"
    else:
        return "red"


def get_emoji_violence(is_violent: bool) -> str:
    """Emoji para indicar violência do time."""
    return "🔥" if is_violent else "🛡️"


# ==============================================================================
# 8. COMPONENTES DE UI
# ==============================================================================

def render_stat_card(label: str, prob: float, hist_data: Optional[Tuple], 
                     is_elite: bool = False):
    """Renderiza card de estatística individual."""
    color = get_color(prob)
    elite_badge = " ⭐" if is_elite else ""
    hist_str = fmt_hist(hist_data)
    
    st.markdown(f"""
    <div class="metric-card {'opportunity-high' if is_elite else ''}">
        <strong>{label}</strong>{elite_badge}<br>
        <span class="stat-{color}">{prob:.0f}%</span> 
        <small>(Histórico: {hist_str})</small>
    </div>
    """, unsafe_allow_html=True)


def render_match_analysis(home: str, away: str, liga: str, hora: str, 
                          match_data: dict, hist_loader: HistoryLoader):
    """
    Renderiza análise completa de um jogo.
    
    Args:
        home: Time mandante
        away: Time visitante
        liga: Nome da liga
        hora: Horário
        match_data: Dados calculados pelo motor V12.0
        hist_loader: Instância do HistoryLoader
    """
    m = match_data
    meta = m.get('metadata', {})
    
    with st.expander(f"⏰ {hora} | {liga} | {home} x {away}", expanded=False):
        # Metadados V12.0
        col_meta1, col_meta2, col_meta3 = st.columns(3)
        with col_meta1:
            st.metric("xG Mandante", f"{m['goals']['h']:.2f}⚽")
        with col_meta2:
            st.metric("xG Visitante", f"{m['goals']['a']:.2f}⚽")
        with col_meta3:
            st.metric("Árbitro", f"{meta.get('ref_factor', 1.0):.2f}👨‍⚖️")
        
        st.markdown("---")
        
        col_corners, col_cards = st.columns(2)
        
        # === ESCANTEIOS ===
        with col_corners:
            st.markdown("### 🚩 Escanteios")
            
            # Mandante
            st.markdown(f"**🏠 {home}** {get_emoji_violence(meta.get('pressure_home', False))}")
            
            p_h35 = prob_over(m['corners']['h'], 3.5)
            h_h35 = hist_loader.get_global(home, 'corners', 'homeTeamOver35')
            elite_h35 = check_elite(p_h35, h_h35, False)
            render_stat_card(f"+3.5 Escanteios", p_h35, h_h35, elite_h35)
            
            p_h45 = prob_over(m['corners']['h'], 4.5)
            h_h45 = hist_loader.get_global(home, 'corners', 'homeTeamOver45')
            elite_h45 = check_elite(p_h45, h_h45, False)
            render_stat_card(f"+4.5 Escanteios", p_h45, h_h45, elite_h45)
            
            st.markdown("---")
            
            # Visitante
            st.markdown(f"**✈️ {away}**")
            
            p_a35 = prob_over(m['corners']['a'], 3.5)
            h_a35 = hist_loader.get_global(away, 'corners', 'awayTeamOver35')
            elite_a35 = check_elite(p_a35, h_a35, False)
            render_stat_card(f"+3.5 Escanteios", p_a35, h_a35, elite_a35)
            
            p_a45 = prob_over(m['corners']['a'], 4.5)
            h_a45 = hist_loader.get_global(away, 'corners', 'awayTeamOver45')
            elite_a45 = check_elite(p_a45, h_a45, False)
            render_stat_card(f"+4.5 Escanteios", p_a45, h_a45, elite_a45)
        
        # === CARTÕES ===
        with col_cards:
            st.markdown("### 🟨 Cartões")
            
            # Mandante
            st.markdown(f"**🏠 {home}** {get_emoji_violence(meta.get('violence_home', False))}")
            
            p_hk15 = prob_over(m['cards']['h'], 1.5)
            h_hk15 = hist_loader.get_global(home, 'cards', 'homeCardsOver15')
            elite_hk15 = check_elite(p_hk15, h_hk15, True)
            render_stat_card(f"+1.5 Cartões", p_hk15, h_hk15, elite_hk15)
            
            p_hk25 = prob_over(m['cards']['h'], 2.5)
            h_hk25 = hist_loader.get_global(home, 'cards', 'homeCardsOver25')
            elite_hk25 = check_elite(p_hk25, h_hk25, True)
            render_stat_card(f"+2.5 Cartões", p_hk25, h_hk25, elite_hk25)
            
            st.markdown("---")
            
            # Visitante
            st.markdown(f"**✈️ {away}** {get_emoji_violence(meta.get('violence_away', False))}")
            
            p_ak15 = prob_over(m['cards']['a'], 1.5)
            h_ak15 = hist_loader.get_global(away, 'cards', 'awayCardsOver15')
            elite_ak15 = check_elite(p_ak15, h_ak15, True)
            render_stat_card(f"+1.5 Cartões", p_ak15, h_ak15, elite_ak15)
            
            p_ak25 = prob_over(m['cards']['a'], 2.5)
            h_ak25 = hist_loader.get_global(away, 'cards', 'awayCardsOver25')
            elite_ak25 = check_elite(p_ak25, h_ak25, True)
            render_stat_card(f"+2.5 Cartões", p_ak25, h_ak25, elite_ak25)


# ==============================================================================
# 9. DASHBOARD PRINCIPAL
# ==============================================================================

def render_dashboard():
    """Renderiza dashboard principal do app."""
    
    # Inicialização
    stats_db = learn_stats()
    hist_loader = HistoryLoader()
    team_list = sorted(list(stats_db.keys())) if stats_db else ["Carregando..."]
    
    # Header
    st.title("⚽ FutPrevisão V13.2 (Refatorado)")
    st.caption("Sistema V12.0 Causality Engine | Desenvolvido por Diego")
    
    # Metrics de sistema
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Times Cadastrados", len(stats_db))
    with col_m2:
        st.metric("Árbitros no DB", len(load_referees_unified()))
    with col_m3:
        st.metric("Ligas Ativas", len(LEAGUE_FILES))
    with col_m4:
        st.metric("OCR Disponível", "✅" if HAS_OCR else "❌")
    
    st.markdown("---")
    
    # === TABS PRINCIPAIS ===
    tab_scan, tab_sim, tab_info = st.tabs([
        "🔥 Scanner de Jogos", 
        "🔮 Simulador Manual",
        "ℹ️ Sistema V12.0"
    ])
    
    # --- TAB 1: SCANNER DE JOGOS ---
    with tab_scan:
        st.subheader("📡 Rastreador de Oportunidades")
        
        df_cal, status = load_calendar_file()
        
        if df_cal.empty:
            st.error(status)
            st.info("💡 Certifique-se de que o arquivo 'calendario_ligas.csv' está no diretório.")
        else:
            st.success(status)
            
            # Seletor de data
            dates = df_cal['Data'].unique()
            hoje = datetime.now().strftime("%d/%m/%Y")
            idx_hoje = list(dates).index(hoje) if hoje in dates else 0
            
            selected_date = st.selectbox("📅 Selecione a Data:", dates, index=idx_hoje)
            jogos_hoje = df_cal[df_cal['Data'] == selected_date]
            
            st.info(f"🎯 {len(jogos_hoje)} jogos agendados para {selected_date}")
            
            # Botão de scan
            if st.button("🔍 Rastrear Oportunidades Elite", type="primary"):
                with st.spinner("🔎 Analisando jogos..."):
                    opportunities = []
                    
                    for _, row in jogos_hoje.iterrows():
                        tc = row['Mandante']
                        tv = row['Visitante']
                        
                        m = calcular_jogo(tc, tv, None, stats_db)
                        if not m:
                            continue
                        
                        # Verifica oportunidades elite
                        msgs = []
                        
                        # Escanteios casa
                        p_h35 = prob_over(m['corners']['h'], 3.5)
                        h_h35 = hist_loader.get_global(tc, 'corners', 'homeTeamOver35')
                        if check_elite(p_h35, h_h35, False):
                            msgs.append(f"🚩 {tc} +3.5 Escanteios ({p_h35:.0f}%)")
                        
                        # Escanteios fora
                        p_a35 = prob_over(m['corners']['a'], 3.5)
                        h_a35 = hist_loader.get_global(tv, 'corners', 'awayTeamOver35')
                        if check_elite(p_a35, h_a35, False):
                            msgs.append(f"🚩 {tv} +3.5 Escanteios ({p_a35:.0f}%)")
                        
                        # Cartões casa
                        p_hk15 = prob_over(m['cards']['h'], 1.5)
                        h_hk15 = hist_loader.get_global(tc, 'cards', 'homeCardsOver15')
                        if check_elite(p_hk15, h_hk15, True):
                            msgs.append(f"🟨 {tc} +1.5 Cartões ({p_hk15:.0f}%)")
                        
                        # Cartões fora
                        p_ak15 = prob_over(m['cards']['a'], 1.5)
                        h_ak15 = hist_loader.get_global(tv, 'cards', 'awayCardsOver15')
                        if check_elite(p_ak15, h_ak15, True):
                            msgs.append(f"🟨 {tv} +1.5 Cartões ({p_ak15:.0f}%)")
                        
                        if msgs:
                            opportunities.append({
                                'liga': row['Liga'],
                                'hora': row['Hora'],
                                'mandante': tc,
                                'visitante': tv,
                                'mensagens': msgs
                            })
                    
                    # Exibe resultados
                    if opportunities:
                        st.success(f"🎯 {len(opportunities)} oportunidades encontradas!")
                        
                        cols = st.columns(3)
                        for idx, opp in enumerate(opportunities):
                            with cols[idx % 3]:
                                st.markdown(f"""
                                <div class="metric-card opportunity-high">
                                    <strong>{opp['liga']}</strong><br>
                                    ⏰ {opp['hora']}<br>
                                    <strong>{opp['mandante']} x {opp['visitante']}</strong><br>
                                    <br>
                                    {'<br>'.join(opp['mensagens'])}
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        st.warning("⚠️ Nenhuma oportunidade elite encontrada para esta data.")
            
            st.markdown("---")
            st.subheader("📋 Todos os Jogos")
            
            # Lista todos os jogos
            for _, row in jogos_hoje.iterrows():
                tc = row['Mandante']
                tv = row['Visitante']
                
                m = calcular_jogo(tc, tv, None, stats_db)
                if m:
                    render_match_analysis(tc, tv, row['Liga'], row['Hora'], m, hist_loader)
    
    # --- TAB 2: SIMULADOR MANUAL ---
    with tab_sim:
        st.subheader("🔮 Simulador de Confrontos")
        st.caption("Escolha times de qualquer liga para análise customizada")
        
        col_h, col_a, col_r = st.columns(3)
        
        with col_h:
            home = st.selectbox("🏠 Mandante", team_list, index=0)
        
        with col_a:
            away = st.selectbox("✈️ Visitante", team_list, 
                               index=1 if len(team_list) > 1 else 0)
        
        with col_r:
            ref_list = ["Neutro (1.0)"] + sorted(load_referees_unified().keys())
            ref = st.selectbox("👨‍⚖️ Árbitro", ref_list)
        
        if st.button("⚡ Simular Confronto", type="primary"):
            ref_name = None if ref == "Neutro (1.0)" else ref
            
            with st.spinner("Calculando..."):
                m = calcular_jogo(home, away, ref_name, stats_db)
                
                if m:
                    st.success("✅ Análise concluída!")
                    st.markdown("---")
                    
                    # Exibe análise completa
                    render_match_analysis(home, away, "Simulação Manual", "N/A", m, hist_loader)
                else:
                    st.error("❌ Erro ao calcular. Verifique se os times existem no banco de dados.")
    
    # --- TAB 3: INFO DO SISTEMA ---
    with tab_info:
        st.subheader("ℹ️ Sistema V12.0 - Causality Engine")
        
        st.markdown("""
        ### 🧠 Como Funciona
        
        O **V12.0 Causality Engine** não se baseia apenas em estatísticas históricas.
        Ele identifica **CAUSAS** (faltas, chutes no gol) que levam aos **EFEITOS** 
        (cartões, escanteios).
        
        #### 📊 Fórmulas Principais
        
        **Escanteios:**
        ```
        Mandante = Base × 1.15 × Boost_Pressão
        Visitante = Base × 0.90
        
        Boost_Pressão = 1.10 se gols_marcados > 1.8 
        ```
        
        **Cartões:**
        ```
        Cartões = Base × Fator_Violência × Fator_Árbitro
        
        Fator_Violência:
        - 0.85 (penalidade) se faltas ≤ 12.5/jogo 🛡️
        - 1.0 (sem penalidade) se faltas > 12.5/jogo 🔥
        ```
        
        **xG (Expected Goals):**
        ```
        xG_Mandante = Ataque_Mandante × Defesa_Visitante / 1.3
        ```
        
        #### 🎯 Critérios Elite
        
        Uma aposta é considerada **ELITE** quando:
        1. Probabilidade Poisson ≥ 75% (escanteios) ou 70% (cartões)
        2. **E** Histórico ≥ 70%
        3. **OU** Probabilidade ≥ 80% (dispensa histórico)
        
        #### 🔧 Configurações Atuais
        
        | Parâmetro | Valor |
        |-----------|-------|
        | Threshold Violência | {THRESHOLDS['fouls_violent']} faltas/jogo |
        | Threshold Pressão | {THRESHOLDS['goals_pressure']} gols/jogo |
        | Boost Mandante | {MULTIPLIERS['home_corners']}x |
        | Penalidade Visitante | {MULTIPLIERS['away_corners']}x |
        | Boost Pressão | {MULTIPLIERS['pressure_boost']}x |
        
        #### 📚 Fontes de Dados
        
        - **CSVs**: Football-Data.co.uk (histórico completo)
        - **Adam Choi**: Validação histórica de linhas
        - **Árbitros**: Base de 62+ árbitros (5 ligas secundárias + principais)
        
        #### 🔮 Versões
        
        - **V11.1**: Card Safe Mode (penalidade 15%)
        - **V11.2**: Balanced Mode (threshold 65%)
        - **V12.0**: Causality Engine (faltas + chutes)
        - **V13.0**: Motor de Copas (histórico global)
        - **V13.2**: Refatoração completa ✨
        """)
        
        st.markdown("---")
        st.info("💡 **Dica**: Sempre combine Poisson + Histórico para apostas mais seguras!")


# ==============================================================================
# 10. PONTO DE ENTRADA
# ==============================================================================

if __name__ == "__main__":
    render_dashboard()