"""
FutPrevisão V32 - CORRIGIDO
Problema: encoding UTF-8 BOM nos CSVs
Solução: usar encoding='utf-8-sig'
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from difflib import get_close_matches

# ============================================================
# BLACKLIST
# ============================================================

BLACKLIST_CORNERS = {
    'Wolves': 2.89, 'Sunderland': 3.61, 'Burnley': 3.78, 'Crystal Palace': 3.78,
    'Elche': 3.06, 'Mallorca': 3.29, 'Osasuna': 3.35, 'Levante': 3.38, 
    'Oviedo': 3.82, 'Girona': 3.88, 'Parma': 3.19, 'Cremonese': 3.24, 
    'Pisa': 3.35, 'Sassuolo': 3.47, 'Cagliari': 3.47, 'Genoa': 3.65, 
    'Verona': 3.81, 'Lazio': 3.82, 'Wolfsburg': 3.53, 'Hamburg': 3.80, 
    'FC Koln': 3.87, 'Nantes': 3.12, 'Lorient': 3.19, 'Angers': 3.62, 
    'Strasbourg': 3.75, 'Dundee': 2.89, 'Motherwell': 3.42, 'Dender': 3.20, 
    'Rizespor': 3.47, 'Karagumruk': 3.71
}

BLACKLIST_CARDS = {
    'Newcastle': 1.28, 'Arsenal': 1.33, 'Burnley': 1.33, 'Man United': 1.44,
    'West Ham': 1.50, 'Aston Villa': 1.56, 'Barcelona': 1.56, 'Inter': 1.44, 
    'Milan': 1.44, 'Juventus': 1.47, 'Atalanta': 1.53, "M'gladbach": 1.33, 
    'Ein Frankfurt': 1.53, 'Paris SG': 1.12, 'Angers': 1.38, 'Metz': 1.50,
    'Oxford': 1.33, 'West Brom': 1.33, 'Sheffield United': 1.38, 'Leeds': 1.50, 
    'Bristol City': 1.58, 'Motherwell': 1.37, 'Dundee': 1.42, 'Hibernian': 1.42, 
    'Celtic': 1.56, 'Club Brugge': 1.35, 'Genk': 1.35, 'Waregem': 1.55, 
    'Darmstadt': 1.59
}

def check_blacklist(team):
    team_norm = team.strip()
    variants = {
        'Wolverhampton': 'Wolves', 'Manchester Utd': 'Man United',
        'Newcastle United': 'Newcastle', 'West Ham United': 'West Ham',
        'Inter Milan': 'Inter', 'AC Milan': 'Milan', 'PSG': 'Paris SG'
    }
    if team_norm in variants:
        team_norm = variants[team_norm]
    
    return {
        'corners_bl': team_norm in BLACKLIST_CORNERS,
        'cards_bl': team_norm in BLACKLIST_CARDS,
        'corners_avg': BLACKLIST_CORNERS.get(team_norm),
        'cards_avg': BLACKLIST_CARDS.get(team_norm)
    }

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(page_title="FutPrevisão V32", layout="wide", page_icon="⚽")

st.markdown('''
<style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 15px;
        border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        background: white;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        color: #667eea;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
    }
</style>
''', unsafe_allow_html=True)

# ============================================================
# CARREGAMENTO COM ENCODING CORRETO
# ============================================================

@st.cache_data(ttl=3600)
def load_all_data():
    """🔧 CORRIGIDO: usa encoding='utf-8-sig' para BOM"""
    
    stats_db = {}
    
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
    
    loaded = []
    failed = []
    
    for league_name, filepath in league_files.items():
        try:
            # 🔧 CORREÇÃO: encoding='utf-8-sig' remove o BOM
            df = pd.read_csv(filepath, encoding='utf-8-sig')
            
            teams = set(df['HomeTeam'].dropna().unique()) | set(df['AwayTeam'].dropna().unique())
            
            for team in teams:
                if pd.isna(team):
                    continue
                
                h_games = df[df['HomeTeam'] == team]
                a_games = df[df['AwayTeam'] == team]
                
                corners_h = h_games['HC'].mean() if 'HC' in h_games.columns and len(h_games) > 0 else 5.5
                corners_a = a_games['AC'].mean() if 'AC' in a_games.columns and len(a_games) > 0 else 4.5
                
                cards_h = h_games[['HY', 'HR']].sum(axis=1).mean() if 'HY' in h_games.columns and len(h_games) > 0 else 2.5
                cards_a = a_games[['AY', 'AR']].sum(axis=1).mean() if 'AY' in a_games.columns and len(a_games) > 0 else 2.5
                
                fouls_h = h_games['HF'].mean() if 'HF' in h_games.columns and len(h_games) > 0 else 12.0
                fouls_a = a_games['AF'].mean() if 'AF' in a_games.columns and len(a_games) > 0 else 12.0
                
                goals_fh = h_games['FTHG'].mean() if 'FTHG' in h_games.columns and len(h_games) > 0 else 1.5
                goals_fa = a_games['FTAG'].mean() if 'FTAG' in a_games.columns and len(a_games) > 0 else 1.3
                
                shots_h = h_games['HST'].mean() if 'HST' in h_games.columns and len(h_games) > 0 else 4.5
                shots_a = a_games['AST'].mean() if 'AST' in a_games.columns and len(a_games) > 0 else 4.0
                
                stats_db[team] = {
                    'corners': (corners_h + corners_a) / 2,
                    'corners_home': corners_h,
                    'corners_away': corners_a,
                    'cards': (cards_h + cards_a) / 2,
                    'cards_home': cards_h,
                    'cards_away': cards_a,
                    'fouls': (fouls_h + fouls_a) / 2,
                    'goals_f': (goals_fh + goals_fa) / 2,
                    'shots': (shots_h + shots_a) / 2,
                    'league': league_name,
                    'games': len(h_games) + len(a_games)
                }
            
            loaded.append(league_name)
        
        except Exception as e:
            failed.append(f"{league_name}: {str(e)}")
    
    # Calendário
    cal = pd.DataFrame()
    try:
        cal = pd.read_csv('/mnt/project/calendario_ligas.csv', encoding='utf-8-sig')
        cal['DtObj'] = pd.to_datetime(cal['Data'], format='%d/%m/%Y', errors='coerce')
        loaded.append("Calendário")
    except Exception as e:
        failed.append(f"Calendário: {str(e)}")
    
    # Árbitros
    refs = {}
    try:
        refs_df = pd.read_csv('/mnt/project/arbitros_5_ligas_2025_2026.csv', encoding='utf-8-sig')
        for _, row in refs_df.iterrows():
            refs[row['Arbitro']] = {
                'factor': row['Media_Cartoes_Por_Jogo'] / 4.0,
                'games': row['Jogos_Apitados']
            }
        loaded.append("Árbitros")
    except Exception as e:
        failed.append(f"Árbitros: {str(e)}")
    
    return stats_db, cal, refs, loaded, failed

# Carregar
STATS, CAL, REFS, LOADED, FAILED = load_all_data()

# ============================================================
# INTERFACE
# ============================================================

def main():
    
    # Header
    st.title("⚽ FutPrevisão V32 - Blacklist Científica")
    
    # Status do carregamento
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if len(STATS) > 0:
            st.success(f"✅ {len(STATS)} times carregados")
        else:
            st.error("❌ Nenhum time carregado")
    
    with col2:
        if len(CAL) > 0:
            st.success(f"✅ {len(CAL)} jogos no calendário")
        else:
            st.warning("⚠️ Calendário vazio")
    
    with col3:
        st.info(f"🚫 Blacklist: {len(BLACKLIST_CORNERS)} cantos + {len(BLACKLIST_CARDS)} cartões")
    
    # Mostrar o que foi carregado
    with st.expander("📊 Status do Carregamento", expanded=len(STATS)==0):
        if LOADED:
            st.success("**✅ Carregados com sucesso:**")
            for item in LOADED:
                st.write(f"• {item}")
        
        if FAILED:
            st.error("**❌ Falhas:**")
            for item in FAILED:
                st.write(f"• {item}")
        
        if len(STATS) == 0:
            st.error("""
### 🔧 NENHUM DADO FOI CARREGADO!

**Possíveis causas:**
1. Arquivos CSV não encontrados em `/mnt/project/`
2. Problema de permissão
3. Arquivos corrompidos

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
            return
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["🎯 Análise", "🆕 Blacklist", "📊 Estatísticas"])
    
    with tab1:
        st.header("🎯 Análise de Jogos")
        
        if not CAL.empty:
            dates = sorted(CAL['DtObj'].dt.strftime('%d/%m/%Y').unique())
            if dates:
                sel_date = st.selectbox("📅 Data:", dates)
                
                jogos_dia = CAL[CAL['DtObj'].dt.strftime('%d/%m/%Y') == sel_date]
                
                st.write(f"**{len(jogos_dia)} jogos**")
                
                for idx, jogo in jogos_dia.head(15).iterrows():
                    
                    h = jogo['Time_Casa']
                    a = jogo['Time_Visitante']
                    
                    if h in STATS and a in STATS:
                        
                        home_bl = check_blacklist(h)
                        away_bl = check_blacklist(a)
                        
                        emoji = "⚽"
                        if home_bl['corners_bl'] or away_bl['corners_bl']:
                            emoji = "⚠️"
                        
                        with st.expander(f"{emoji} {h} vs {a} | {jogo.get('Hora', 'N/A')}"):
                            
                            # Alertas
                            if home_bl['corners_bl']:
                                st.warning(f"🚫 {h}: {home_bl['corners_avg']:.2f} cantos (BLACKLIST)")
                            if away_bl['corners_bl']:
                                st.warning(f"🚫 {a}: {away_bl['corners_avg']:.2f} cantos (BLACKLIST)")
                            if home_bl['cards_bl']:
                                st.info(f"🟡 {h}: {home_bl['cards_avg']:.2f} cartões (BLACKLIST)")
                            if away_bl['cards_bl']:
                                st.info(f"🟡 {a}: {away_bl['cards_avg']:.2f} cartões (BLACKLIST)")
                            
                            # Stats
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
    
    with tab2:
        st.header("🆕 Blacklist Científica")
        
        st.info("📊 Baseada em 1.659 jogos reais")
        
        tab_c, tab_y = st.tabs(["🚫 Cantos", "🟡 Cartões"])
        
        with tab_c:
            st.subheader(f"Times com MENOS cantos ({len(BLACKLIST_CORNERS)})")
            
            sorted_c = sorted(BLACKLIST_CORNERS.items(), key=lambda x: x[1])
            
            for team, avg in sorted_c:
                emoji = "🔴" if avg < 3.0 else "🟠" if avg < 3.5 else "🟡"
                st.write(f"{emoji} **{team}**: {avg:.2f} cantos/jogo")
        
        with tab_y:
            st.subheader(f"Times com MENOS cartões ({len(BLACKLIST_CARDS)})")
            
            sorted_y = sorted(BLACKLIST_CARDS.items(), key=lambda x: x[1])
            
            for team, avg in sorted_y:
                emoji = "🔴" if avg < 1.3 else "🟠" if avg < 1.5 else "🟡"
                st.write(f"{emoji} **{team}**: {avg:.2f} cartões/jogo")
    
    with tab3:
        st.header("📊 Estatísticas Gerais")
        
        if len(STATS) > 0:
            
            # Top 10 Cantos
            st.subheader("🔥 Top 10 - Mais Cantos")
            top_corners = sorted(STATS.items(), key=lambda x: x[1]['corners'], reverse=True)[:10]
            
            for i, (team, stats) in enumerate(top_corners, 1):
                st.write(f"{i}. **{team}**: {stats['corners']:.2f} cantos/jogo ({stats['league']})")
            
            st.markdown("---")
            
            # Top 10 Cartões
            st.subheader("🟨 Top 10 - Mais Cartões")
            top_cards = sorted(STATS.items(), key=lambda x: x[1]['cards'], reverse=True)[:10]
            
            for i, (team, stats) in enumerate(top_cards, 1):
                st.write(f"{i}. **{team}**: {stats['cards']:.2f} cartões/jogo ({stats['league']})")

if __name__ == "__main__":
    main()
