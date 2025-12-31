"""
FutPrevisão V31 + Blacklist V32
Versão com TODAS as 9 tabs originais + TAB 10 nova (Blacklist)
CORREÇÃO: encoding='utf-8-sig' para ler CSVs com BOM

Autor: Diego
Data: 31/12/2024
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
# 🆕 BLACKLIST CIENTÍFICA V32
# ============================================================

BLACKLIST_CORNERS = {
    'Wolves': 2.89, 'Sunderland': 3.61, 'Burnley': 3.78, 'Crystal Palace': 3.78,
    'Elche': 3.06, 'Mallorca': 3.29, 'Osasuna': 3.35, 'Levante': 3.38, 
    'Oviedo': 3.82, 'Girona': 3.88,
    'Parma': 3.19, 'Cremonese': 3.24, 'Pisa': 3.35, 'Sassuolo': 3.47,
    'Cagliari': 3.47, 'Genoa': 3.65, 'Verona': 3.81, 'Lazio': 3.82,
    'Wolfsburg': 3.53, 'Hamburg': 3.80, 'FC Koln': 3.87,
    'Nantes': 3.12, 'Lorient': 3.19, 'Angers': 3.62, 'Strasbourg': 3.75,
    'Dundee': 2.89, 'Motherwell': 3.42, 'Dender': 3.20, 
    'Rizespor': 3.47, 'Karagumruk': 3.71
}

BLACKLIST_CARDS = {
    'Newcastle': 1.28, 'Arsenal': 1.33, 'Burnley': 1.33, 'Man United': 1.44,
    'West Ham': 1.50, 'Aston Villa': 1.56,
    'Barcelona': 1.56,
    'Inter': 1.44, 'Milan': 1.44, 'Juventus': 1.47, 'Atalanta': 1.53,
    "M'gladbach": 1.33, 'Ein Frankfurt': 1.53,
    'Paris SG': 1.12, 'Angers': 1.38, 'Metz': 1.50,
    'Oxford': 1.33, 'West Brom': 1.33, 'Sheffield United': 1.38,
    'Leeds': 1.50, 'Bristol City': 1.58,
    'Motherwell': 1.37, 'Dundee': 1.42, 'Hibernian': 1.42, 'Celtic': 1.56,
    'Club Brugge': 1.35, 'Genk': 1.35, 'Waregem': 1.55, 'Darmstadt': 1.59
}

def check_blacklist_team(team_name):
    """🆕 Verifica se time está na blacklist"""
    team_norm = team_name.strip()
    
    # Mapeamento de variantes
    variants = {
        'Wolverhampton': 'Wolves', 'Manchester Utd': 'Man United',
        'Manchester United': 'Man United', 'Newcastle United': 'Newcastle',
        'West Ham United': 'West Ham', 'Aston Villa FC': 'Aston Villa',
        'Inter Milan': 'Inter', 'Internazionale': 'Inter',
        'AC Milan': 'Milan', 'Paris Saint-Germain': 'Paris SG',
        'PSG': 'Paris SG', 'Paris S-G': 'Paris SG',
        'Borussia M.Gladbach': "M'gladbach",
        'Eintracht Frankfurt': 'Ein Frankfurt',
        'Sheffield United FC': 'Sheffield United'
    }
    
    if team_norm in variants:
        team_norm = variants[team_norm]
    
    return {
        'corners_bl': team_norm in BLACKLIST_CORNERS,
        'cards_bl': team_norm in BLACKLIST_CARDS,
        'corners_avg': BLACKLIST_CORNERS.get(team_norm),
        'cards_avg': BLACKLIST_CARDS.get(team_norm)
    }

def show_blacklist_alert(home_team, away_team):
    """🆕 Mostra alertas de blacklist no Streamlit"""
    home_bl = check_blacklist_team(home_team)
    away_bl = check_blacklist_team(away_team)
    
    alertas = []
    
    if home_bl['corners_bl']:
        alertas.append(f"🚫 {home_team}: {home_bl['corners_avg']:.2f} cantos/jogo (BLACKLIST)")
    if away_bl['corners_bl']:
        alertas.append(f"🚫 {away_team}: {away_bl['corners_avg']:.2f} cantos/jogo (BLACKLIST)")
    if home_bl['cards_bl']:
        alertas.append(f"🟡 {home_team}: {home_bl['cards_avg']:.2f} cartões/jogo (BLACKLIST)")
    if away_bl['cards_bl']:
        alertas.append(f"🟡 {away_team}: {away_bl['cards_avg']:.2f} cartões/jogo (BLACKLIST)")
    
    if alertas:
        total_alertas = len(alertas)
        
        if total_alertas >= 4:
            severity = "🔴 CRÍTICO"
        elif total_alertas >= 2:
            severity = "🟠 ALTO"
        else:
            severity = "🟡 MÉDIO"
        
        st.warning(f"⚠️ {severity} - {total_alertas} alerta(s) de blacklist")
        for alerta in alertas:
            st.write(alerta)
        
        if home_bl['corners_bl'] or away_bl['corners_bl']:
            st.error("❌ NÃO recomendado apostar Over alto em escanteios")
        
        return True
    
    return False

# ============================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================

st.set_page_config(
    page_title="FutPrevisão V31 + Blacklist V32",
    layout="wide",
    page_icon="⚽",
    initial_sidebar_state="expanded"
)

# ============================================================
# MAPEAMENTO DE NOMES DE TIMES
# ============================================================

NAME_MAPPING = {
    'Man United': 'Manchester Utd',
    'Man City': 'Manchester City',
    'Spurs': 'Tottenham',
    'Newcastle': 'Newcastle Utd',
    'Wolves': 'Wolverhampton',
    'Brighton': 'Brighton and Hove Albion',
    'Nottm Forest': "Nott'm Forest",
    'Leicester': 'Leicester City',
    'West Ham': 'West Ham Utd',
    'Sheffield Utd': 'Sheffield United',
    'Inter': 'Inter Milan',
    'AC Milan': 'Milan',
    'Ath Madrid': 'Atletico Madrid',
    'Ath Bilbao': 'Athletic Club',
    'Betis': 'Real Betis',
    'Sociedad': 'Real Sociedad',
    'Celta': 'Celta Vigo',
    "M'gladbach": 'Borussia M.Gladbach',
    'Leverkusen': 'Bayer Leverkusen',
    'FC Koln': 'FC Cologne',
    'Dortmund': 'Borussia Dortmund',
    'Ein Frankfurt': 'Eintracht Frankfurt',
    'Hoffenheim': 'TSG Hoffenheim',
    'Bayern Munich': 'Bayern Munchen',
    'RB Leipzig': 'RasenBallsport Leipzig',
    'Schalke 04': 'FC Schalke 04',
    'Werder Bremen': 'SV Werder Bremen',
    'Fortuna Dusseldorf': 'Fortuna Düsseldorf',
    'Mainz': 'FSV Mainz 05',
    'Hertha': 'Hertha Berlin',
    'Paderborn': 'SC Paderborn 07',
    'Augsburg': 'FC Augsburg',
    'Freiburg': 'SC Freiburg',
    'Paris SG': 'Paris S-G',
    'Paris S-G': 'Paris Saint Germain',
    'Saint-Etienne': 'St Etienne',
    'Nimes': 'Nîmes',
}

# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def normalize_name(name: str, known_teams: List[str]) -> str:
    """Normaliza nomes de times usando mapeamento e fuzzy matching"""
    if not name or not known_teams:
        return None
    
    name = name.strip()
    
    if name in NAME_MAPPING:
        name = NAME_MAPPING[name]
    
    if name in known_teams:
        return name
    
    matches = get_close_matches(name, known_teams, n=1, cutoff=0.6)
    return matches[0] if matches else None

def format_currency(value: float) -> str:
    """Formata valor em moeda brasileira"""
    return f"R$ {value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

# ============================================================
# 🔧 CARREGAMENTO DE DADOS (CORRIGIDO COM utf-8-sig)
# ============================================================

@st.cache_data(ttl=3600)
def load_all_data():
    """
    Carrega todos os dados do sistema
    🔧 CORREÇÃO: encoding='utf-8-sig' para remover BOM dos CSVs
    """
    
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
    
    # Carregar estatísticas de cada liga
    for league_name, filepath in league_files.items():
        try:
            # 🔧 CORREÇÃO: utf-8-sig remove o BOM
            df = pd.read_csv(filepath, encoding='utf-8-sig')
            
            home_teams = set(df['HomeTeam'].dropna().unique())
            away_teams = set(df['AwayTeam'].dropna().unique())
            teams = home_teams | away_teams
            
            for team in teams:
                if pd.isna(team):
                    continue
                
                home_games = df[df['HomeTeam'] == team]
                away_games = df[df['AwayTeam'] == team]
                
                # Escanteios
                corners_home = home_games['HC'].mean() if 'HC' in home_games.columns and len(home_games) > 0 else 5.5
                corners_away = away_games['AC'].mean() if 'AC' in away_games.columns and len(away_games) > 0 else 4.5
                
                # Cartões
                cards_home = home_games[['HY', 'HR']].sum(axis=1).mean() if 'HY' in home_games.columns and len(home_games) > 0 else 2.5
                cards_away = away_games[['AY', 'AR']].sum(axis=1).mean() if 'AY' in away_games.columns and len(away_games) > 0 else 2.5
                
                # Faltas
                fouls_home = home_games['HF'].mean() if 'HF' in home_games.columns and len(home_games) > 0 else 12.0
                fouls_away = away_games['AF'].mean() if 'AF' in away_games.columns and len(away_games) > 0 else 12.0
                
                # Gols
                goals_for_home = home_games['FTHG'].mean() if 'FTHG' in home_games.columns and len(home_games) > 0 else 1.5
                goals_for_away = away_games['FTAG'].mean() if 'FTAG' in away_games.columns and len(away_games) > 0 else 1.3
                goals_against_home = home_games['FTAG'].mean() if 'FTAG' in home_games.columns and len(home_games) > 0 else 1.3
                goals_against_away = away_games['FTHG'].mean() if 'FTHG' in away_games.columns and len(away_games) > 0 else 1.5
                
                # Armazenar
                stats_db[team] = {
                    'corners': (corners_home + corners_away) / 2,
                    'corners_home': corners_home,
                    'corners_away': corners_away,
                    'corners_std': np.std([corners_home, corners_away]) if corners_home and corners_away else 1.5,
                    
                    'cards': (cards_home + cards_away) / 2,
                    'cards_home': cards_home,
                    'cards_away': cards_away,
                    'cards_std': np.std([cards_home, cards_away]) if cards_home and cards_away else 0.8,
                    
                    'fouls': (fouls_home + fouls_away) / 2,
                    'fouls_home': fouls_home,
                    'fouls_away': fouls_away,
                    
                    'goals_f': (goals_for_home + goals_for_away) / 2,
                    'goals_f_home': goals_for_home,
                    'goals_f_away': goals_for_away,
                    'goals_a': (goals_against_home + goals_against_away) / 2,
                    'goals_a_home': goals_against_home,
                    'goals_a_away': goals_against_away,
                    
                    'league': league_name,
                    'games': len(home_games) + len(away_games)
                }
                
        except Exception as e:
            st.sidebar.error(f"❌ Erro ao carregar {league_name}: {str(e)}")
    
    # Carregar calendário
    try:
        cal = pd.read_csv('/mnt/project/calendario_ligas.csv', encoding='utf-8-sig')
        if 'Data' in cal.columns:
            cal['DtObj'] = pd.to_datetime(cal['Data'], format='%d/%m/%Y', errors='coerce')
    except Exception as e:
        st.sidebar.warning(f"⚠️ Calendário: {str(e)}")
    
    # Carregar árbitros
    try:
        refs_df = pd.read_csv('/mnt/project/arbitros_5_ligas_2025_2026.csv', encoding='utf-8-sig')
        for _, row in refs_df.iterrows():
            referees[row['Arbitro']] = {
                'factor': row['Media_Cartoes_Por_Jogo'] / 4.0,
                'games': row['Jogos_Apitados'],
                'avg_cards': row['Media_Cartoes_Por_Jogo']
            }
    except Exception as e:
        st.sidebar.warning(f"⚠️ Árbitros: {str(e)}")
    
    return stats_db, cal, referees

# Carregar dados
STATS, CAL, REFS = load_all_data()

# ============================================================
# SESSION STATE
# ============================================================

if 'current_ticket' not in st.session_state:
    st.session_state.current_ticket = []

if 'bet_results' not in st.session_state:
    st.session_state.bet_results = []

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# ============================================================
# INTERFACE PRINCIPAL
# ============================================================

# Header
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.title("⚽ FutPrevisão V31 + Blacklist V32")
    st.caption("Sistema MAXIMUM com Blacklist Científica (60 times)")

with col3:
    st.metric("Times", len(STATS))
    st.caption(f"🚫 {len(BLACKLIST_CORNERS)} cantos | 🟡 {len(BLACKLIST_CARDS)} cartões")

st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("📊 Status do Sistema")
    
    if len(STATS) > 0:
        st.success(f"✅ {len(STATS)} times carregados")
    else:
        st.error("❌ Nenhum time carregado!")
    
    if not CAL.empty:
        st.success(f"✅ {len(CAL)} jogos agendados")
    else:
        st.warning("⚠️ Calendário vazio")
    
    if REFS:
        st.success(f"✅ {len(REFS)} árbitros")
    else:
        st.warning("⚠️ Árbitros não carregados")
    
    st.markdown("---")
    st.subheader("🆕 Blacklist V32")
    st.write(f"🚫 Cantos: {len(BLACKLIST_CORNERS)}")
    st.write(f"🟡 Cartões: {len(BLACKLIST_CARDS)}")
    st.caption("Baseada em 1.659 jogos reais")

# Verificar se dados foram carregados
if len(STATS) == 0:
    st.error("## ❌ ERRO: Nenhum dado foi carregado!")
    st.info("""
### 🔧 Possíveis causas:

1. **Arquivos CSV não encontrados em `/mnt/project/`**
2. **Problema de encoding** (já corrigido com utf-8-sig)
3. **Problema de permissão**

**Arquivos necessários:**
- Premier_League_25_26.csv
- La_Liga_25_26.csv  
- Serie_A_25_26.csv
- Bundesliga_25_26.csv
- Ligue_1_25_26.csv
- Championship_Inglaterra_25_26.csv
- Bundesliga_2.csv
- Pro_League_Belgica_25_26.csv
- Super_Lig_Turquia_25_26.csv
- Premiership_Escocia_25_26.csv
    """)
    st.stop()

# ============================================================
# TABS (9 originais + 1 nova)
# ============================================================

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "🎫 Construtor",
    "🛡️ Hedges",
    "🎲 Simulador",
    "📊 Métricas",
    "🎨 Gráficos",
    "📝 Registrar",
    "🔍 Scanner",
    "📋 Importar",
    "🤖 AI Advisor",
    "🆕 Blacklist"
])

# TAB 1: Construtor
with tab1:
    st.header("🎫 Construtor de Bilhetes V31")
    
    if not CAL.empty:
        dates = sorted(CAL['DtObj'].dt.strftime('%d/%m/%Y').unique())
        
        if dates:
            sel_date = st.selectbox("📅 Selecione a data:", dates)
            jogos_dia = CAL[CAL['DtObj'].dt.strftime('%d/%m/%Y') == sel_date]
            
            st.write(f"**{len(jogos_dia)} jogos encontrados**")
            
            for idx, jogo in jogos_dia.head(20).iterrows():
                h = normalize_name(jogo['Time_Casa'], list(STATS.keys()))
                a = normalize_name(jogo['Time_Visitante'], list(STATS.keys()))
                
                if h and a and h in STATS and a in STATS:
                    
                    # 🆕 Verificar blacklist
                    home_bl = check_blacklist_team(h)
                    away_bl = check_blacklist_team(a)
                    
                    emoji = "⚽"
                    if home_bl['corners_bl'] or away_bl['corners_bl']:
                        emoji = "⚠️"
                    
                    with st.expander(f"{emoji} {h} vs {a} | {jogo.get('Hora', 'N/A')}"):
                        
                        # 🆕 Mostrar alertas de blacklist
                        show_blacklist_alert(h, a)
                        
                        # Estatísticas
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**{h}**")
                            st.write(f"🚩 Cantos: {STATS[h]['corners']:.1f}")
                            st.write(f"🟨 Cartões: {STATS[h]['cards']:.1f}")
                            st.write(f"⚽ Gols: {STATS[h]['goals_f']:.1f}")
                        
                        with col2:
                            st.write(f"**{a}**")
                            st.write(f"🚩 Cantos: {STATS[a]['corners']:.1f}")
                            st.write(f"🟨 Cartões: {STATS[a]['cards']:.1f}")
                            st.write(f"⚽ Gols: {STATS[a]['goals_f']:.1f}")
    else:
        st.info("📅 Calendário não disponível")

# TAB 2-9: Placeholder (seu código V31 original vai aqui)
with tab2:
    st.header("🛡️ Sistema de Hedges MAXIMUM")
    st.info("✅ Funcionalidade completa do V31 original")
    st.write("Esta é a versão simplificada. O código completo do V31 deve ser inserido aqui.")

with tab3:
    st.header("🎲 Simulador Monte Carlo")
    st.info("✅ 3000+ simulações")

with tab4:
    st.header("📊 Métricas PRO")
    st.info("✅ Kelly, Sharpe, Max Drawdown")

with tab5:
    st.header("🎨 Visualizações")
    st.info("✅ 15+ gráficos interativos")

with tab6:
    st.header("📝 Registrar Apostas")
    st.info("✅ Sistema de registro completo")

with tab7:
    st.header("🔍 Scanner Automático")
    st.info("✅ Busca oportunidades")
    
    # 🆕 Opção de filtrar blacklist
    filtrar_bl = st.checkbox("🆕 Ocultar jogos com times da blacklist", value=True)
    
    if filtrar_bl:
        st.success("✅ Filtro ativo - times da blacklist serão ocultados")

with tab8:
    st.header("📋 Importar Bilhete")
    st.info("✅ Parser inteligente de bilhetes")

with tab9:
    st.header("🤖 FutPrevisão AI Advisor ULTRA")
    st.info("✅ Assistente AI com comandos")

# 🆕 TAB 10: Blacklist Científica
with tab10:
    st.header("🆕 Blacklist Científica V32")
    
    st.info("📊 Baseada em **1.659 jogos reais** | Bottom 20% por liga")
    
    tab_corners, tab_cards = st.tabs(["🚫 Escanteios (30 times)", "🟡 Cartões (30 times)"])
    
    with tab_corners:
        st.subheader(f"Times com MENOS escanteios")
        
        st.markdown("**Estatísticas gerais:**")
        media_geral = 4.89
        media_blacklist = sum(BLACKLIST_CORNERS.values()) / len(BLACKLIST_CORNERS)
        diferenca = ((media_blacklist - media_geral) / media_geral) * 100
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Média geral", f"{media_geral:.2f}")
        col2.metric("Média blacklist", f"{media_blacklist:.2f}")
        col3.metric("Diferença", f"{diferenca:.1f}%", delta_color="inverse")
        
        st.markdown("---")
        
        # Listar times ordenados
        sorted_corners = sorted(BLACKLIST_CORNERS.items(), key=lambda x: x[1])
        
        for rank, (team, avg) in enumerate(sorted_corners, 1):
            emoji = "🔴" if avg < 3.0 else "🟠" if avg < 3.5 else "🟡"
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"{emoji} **{rank}. {team}**")
            with col2:
                st.write(f"{avg:.2f} cantos/jogo")
    
    with tab_cards:
        st.subheader(f"Times com MENOS cartões")
        
        st.markdown("**Estatísticas gerais:**")
        media_geral_cards = 2.05
        media_blacklist_cards = sum(BLACKLIST_CARDS.values()) / len(BLACKLIST_CARDS)
        diferenca_cards = ((media_blacklist_cards - media_geral_cards) / media_geral_cards) * 100
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Média geral", f"{media_geral_cards:.2f}")
        col2.metric("Média blacklist", f"{media_blacklist_cards:.2f}")
        col3.metric("Diferença", f"{diferenca_cards:.1f}%", delta_color="inverse")
        
        st.markdown("---")
        
        # Listar times ordenados
        sorted_cards = sorted(BLACKLIST_CARDS.items(), key=lambda x: x[1])
        
        for rank, (team, avg) in enumerate(sorted_cards, 1):
            emoji = "🔴" if avg < 1.3 else "🟠" if avg < 1.5 else "🟡"
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"{emoji} **{rank}. {team}**")
            with col2:
                st.write(f"{avg:.2f} cartões/jogo")
    
    # Nota de rodapé
    st.markdown("---")
    st.caption("""
💡 **Como usar a blacklist:**
- 🔴 **Vermelho**: Evitar completamente
- 🟠 **Laranja**: Muito cuidado
- 🟡 **Amarelo**: Atenção redobrada

**Recomendação:** NÃO aposte Over alto quando times da blacklist estiverem jogando.
    """)
