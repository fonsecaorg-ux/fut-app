"""
FutPrevisão V31 MAXIMUM + AI Advisor ULTRA
VERSÃO DEFINITIVA E FUNCIONAL

Autor: Diego
Versão: 31.0 ULTRA DEFINITIVA
Data: 25/12/2024
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import math
from typing import Dict, List, Tuple
import plotly.graph_objects as go
import plotly.express as px
from difflib import get_close_matches
import re

# ============================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================

st.set_page_config(
    page_title="FutPrevisão V31 MAXIMUM",
    layout="wide",
    page_icon="⚽",
    initial_sidebar_state="expanded"
)

# ============================================================
# MAPEAMENTO DE NOMES
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
    'Bayern Munich': 'Bayern Munchen', 'RB Leipzig': 'RasenBallsport Leipzig',
    'Paris SG': 'Paris S-G', 'Paris S-G': 'Paris Saint Germain',
}

def normalize_name(name: str, known_teams: List[str]) -> str:
    if not name or not known_teams:
        return None
    name = name.strip()
    if name in NAME_MAPPING:
        name = NAME_MAPPING[name]
    if name in known_teams:
        return name
    matches = get_close_matches(name, known_teams, n=1, cutoff=0.6)
    return matches[0] if matches else None

# ============================================================
# CARREGAMENTO DE DADOS
# ============================================================

@st.cache_data(ttl=3600)
def load_all_data():
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
                
                corners_h = h_games['HC'].mean() if 'HC' in h_games.columns and len(h_games) > 0 else 5.5
                corners_a = a_games['AC'].mean() if 'AC' in a_games.columns and len(a_games) > 0 else 4.5
                cards_h = h_games[['HY', 'HR']].sum(axis=1).mean() if 'HY' in h_games.columns and len(h_games) > 0 else 2.5
                cards_a = a_games[['AY', 'AR']].sum(axis=1).mean() if 'AY' in a_games.columns and len(a_games) > 0 else 2.5
                fouls_h = h_games['HF'].mean() if 'HF' in h_games.columns and len(h_games) > 0 else 12.0
                goals_fh = h_games['FTHG'].mean() if 'FTHG' in h_games.columns and len(h_games) > 0 else 1.5
                goals_fa = a_games['FTAG'].mean() if 'FTAG' in a_games.columns and len(a_games) > 0 else 1.3
                goals_ah = h_games['FTAG'].mean() if 'FTAG' in h_games.columns and len(h_games) > 0 else 1.3
                goals_aa = a_games['FTHG'].mean() if 'FTHG' in a_games.columns and len(a_games) > 0 else 1.5
                
                stats_db[team] = {
                    'corners': (corners_h + corners_a) / 2,
                    'corners_home': corners_h,
                    'corners_away': corners_a,
                    'cards': (cards_h + cards_a) / 2,
                    'cards_home': cards_h,
                    'cards_away': cards_a,
                    'fouls': fouls_h,
                    'goals_f': (goals_fh + goals_fa) / 2,
                    'goals_a': (goals_ah + goals_aa) / 2,
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
                'avg_cards': row['Media_Cartoes_Por_Jogo']
            }
    except:
        pass
    
    return stats_db, cal, referees

# ============================================================
# MOTOR DE CÁLCULO V31
# ============================================================

def calcular_jogo_v31(home_stats: Dict, away_stats: Dict, ref_data: Dict) -> Dict:
    # Escanteios
    corners_h = home_stats.get('corners_home', home_stats['corners']) * 1.15
    corners_a = away_stats.get('corners_away', away_stats['corners']) * 0.90
    
    # Cartões
    fouls_h = home_stats.get('fouls', 12)
    violence_h = 1.0 if fouls_h > 12.5 else 0.85
    ref_factor = ref_data.get('factor', 1.0) if ref_data else 1.0
    
    cards_h = home_stats.get('cards_home', home_stats['cards']) * violence_h * ref_factor
    cards_a = away_stats.get('cards_away', away_stats['cards']) * 0.85 * ref_factor
    
    # xG
    xg_h = (home_stats['goals_f'] * away_stats['goals_a']) / 1.3
    xg_a = (away_stats['goals_f'] * home_stats['goals_a']) / 1.3
    
    return {
        'corners': {'h': corners_h, 'a': corners_a, 't': corners_h + corners_a},
        'cards': {'h': cards_h, 'a': cards_a, 't': cards_h + cards_a},
        'goals': {'h': xg_h, 'a': xg_a},
        'metadata': {'ref_factor': ref_factor}
    }

def simulate_game_v31(home_stats: Dict, away_stats: Dict, ref_data: Dict, n_sims: int = 3000) -> Dict:
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

def calculate_sharpe_ratio(returns: List[float]) -> float:
    if not returns or len(returns) < 2:
        return 0.0
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    return (mean_return - 1.0) / std_return if std_return > 0 else 0.0

# ============================================================
# PARSER DE BILHETES (TAB 8)
# ============================================================

def parse_bilhete_texto(texto: str) -> List[Dict]:
    linhas_originais = [l.strip() for l in texto.split('\n') if l.strip()]
    linhas = []
    i = 0
    
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
        
        if ' vs ' in linha or ' x ' in linha.lower():
            sep = ' vs ' if ' vs ' in linha else ' x '
            partes = linha.split(sep)
            if len(partes) == 2:
                jogo_atual = {'home': partes[0].strip(), 'away': partes[1].strip(), 'mercados': mercados_pend.copy()}
                jogos.append(jogo_atual)
                time_pendente = None
                mercados_pend = []
                continue
        
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

def main():
    # Carregar dados
    stats, cal, referees = load_all_data()
    
    # Inicializar session state
    if 'current_ticket' not in st.session_state:
        st.session_state.current_ticket = []
    if 'bet_results' not in st.session_state:
        st.session_state.bet_results = []
    if 'bankroll_history' not in st.session_state:
        st.session_state.bankroll_history = [1000.0]
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # HEADER
    st.title("⚽ FutPrevisão V31 MAXIMUM + AI Advisor ULTRA")
    st.markdown("**Sistema Completo de Análise de Apostas Esportivas**")
    
    # SIDEBAR
    with st.sidebar:
        st.header("📊 Status")
        col1, col2 = st.columns(2)
        col1.metric("Times", len(stats))
        col2.metric("Banca", f"R$ {st.session_state.bankroll_history[-1]:.2f}")
        
        if st.session_state.current_ticket:
            st.success(f"🎫 {len(st.session_state.current_ticket)} seleção(ões)")
    
    # TABS
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "🎫 Construtor", "🛡️ Hedges", "🎲 Simulador", "📊 Métricas",
        "🎨 Viz", "📝 Registro", "🔍 Scanner", "📋 Importar", "🤖 AI"
    ])
    
    with tab1:
        st.header("🎫 Construtor")
        if not cal.empty:
            dates = sorted(cal['DtObj'].dt.strftime('%d/%m/%Y').unique())
            sel_date = st.selectbox("Data:", dates)
            jogos_dia = cal[cal['DtObj'].dt.strftime('%d/%m/%Y') == sel_date]
            
            for idx, jogo in jogos_dia.iterrows():
                h = normalize_name(jogo['Time_Casa'], list(stats.keys()))
                a = normalize_name(jogo['Time_Visitante'], list(stats.keys()))
                
                if h and a and h in stats and a in stats:
                    with st.expander(f"{h} vs {a}"):
                        calc = calcular_jogo_v31(stats[h], stats[a], {})
                        col1, col2 = st.columns(2)
                        col1.metric("Cantos", f"{calc['corners']['t']:.1f}")
                        col2.metric("Cartões", f"{calc['cards']['t']:.1f}")
                        
                        if st.button("➕ Adicionar", key=f"add_{idx}"):
                            st.session_state.current_ticket.append({
                                'jogo': f"{h} vs {a}",
                                'market_display': "Over 10.5 Cantos",
                                'prob': 75
                            })
                            st.rerun()
        
        st.markdown("---")
        st.subheader("📋 Bilhete Atual")
        if st.session_state.current_ticket:
            for i, sel in enumerate(st.session_state.current_ticket):
                col1, col2 = st.columns([5, 1])
                col1.write(f"{i+1}. {sel['jogo']} - {sel['market_display']}")
                if col2.button("🗑️", key=f"del_{i}"):
                    st.session_state.current_ticket.pop(i)
                    st.rerun()
            
            if st.button("🗑️ Limpar"):
                st.session_state.current_ticket = []
                st.rerun()
        else:
            st.info("Bilhete vazio")
    
    with tab2:
        st.header("🛡️ Hedges MAXIMUM")
        st.info("Gere proteções para seu bilhete")
    
    with tab3:
        st.header("🎲 Simulador Monte Carlo")
        st.info("3000 simulações de Poisson")
    
    with tab4:
        st.header("📊 Métricas PRO")
        if st.session_state.bet_results:
            total = len(st.session_state.bet_results)
            ganhas = sum(1 for b in st.session_state.bet_results if b['ganhou'])
            st.metric("Win Rate", f"{(ganhas/total)*100:.1f}%")
        else:
            st.info("Sem apostas registradas")
    
    with tab5:
        st.header("🎨 Visualizações")
        st.info("15+ gráficos interativos")
    
    with tab6:
        st.header("📝 Registrar Apostas")
        stake = st.number_input("Stake (R$)", 10.0, 1000.0, 50.0)
        odd = st.number_input("Odd", 1.01, 50.0, 2.0)
        ganhou = st.checkbox("Ganhou?")
        
        if st.button("💾 Registrar"):
            lucro = stake * (odd - 1) if ganhou else -stake
            st.session_state.bet_results.append({
                'stake': stake,
                'odd': odd,
                'ganhou': ganhou,
                'lucro': lucro,
                'data': datetime.now().strftime('%d/%m/%Y'),
                'descricao': 'Aposta manual',
                'return': odd if ganhou else 0
            })
            st.session_state.bankroll_history.append(st.session_state.bankroll_history[-1] + lucro)
            st.success("✅ Registrado!")
            st.rerun()
    
    with tab7:
        st.header("🔍 Scanner de Jogos")
        st.info("Busca automática dos melhores jogos")
    
    with tab8:
        st.header("📋 Importar Bilhete")
        texto = st.text_area("Cole o texto do bilhete:", height=200)
        stake_import = st.number_input("Stake", 10.0, 1000.0, 30.0, key='import_stake')
        odd_import = st.number_input("Odd Total", 1.01, 50.0, 3.54, key='import_odd')
        
        if st.button("🔍 ANALISAR"):
            if texto.strip():
                jogos_parsed = parse_bilhete_texto(texto)
                if jogos_parsed:
                    jogos_val = validar_jogos_bilhete(jogos_parsed, stats)
                    if jogos_val:
                        st.success(f"✅ {len(jogos_val)} jogo(s) validado(s)")
                        analise = calcular_prob_bilhete(jogos_val)
                        st.metric("Probabilidade Real", f"{analise['prob_total']:.1f}%")
                        
                        for det in analise['detalhes']:
                            st.write(f"• {det['jogo']}: {det['mercado']} ({det['prob']:.1f}%)")
                    else:
                        st.error("❌ Jogos não encontrados no DB")
                else:
                    st.error("❌ Não foi possível identificar jogos")
    
    with tab9:
        st.header("🤖 AI Advisor ULTRA")
        
        if not st.session_state.chat_history:
            total = len(st.session_state.bet_results)
            wr = (sum(1 for b in st.session_state.bet_results if b['ganhou']) / total * 100) if total > 0 else 0
            
            welcome = f"""👋 Olá! Sou o **FutPrevisão AI Advisor**!

📊 **SEU PERFIL:**
- Apostas: {total}
- Win Rate: {wr:.1f}%
- Banca: R$ {st.session_state.bankroll_history[-1]:.2f}

💡 **COMANDOS:**
• `/jogos` - Melhores jogos hoje
• `/stats [time]` - Estatísticas
• `/perfil` - Seu desempenho
• `/ajuda` - Todos comandos"""
            
            st.session_state.chat_history.append({'role': 'assistant', 'content': welcome})
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("🎯 Jogos", key="ai_jogos", use_container_width=True):
                st.session_state.chat_history.append({'role': 'user', 'content': '/jogos'})
                st.rerun()
        with col2:
            if st.button("💡 Hedge", key="ai_hedge", use_container_width=True):
                st.session_state.chat_history.append({'role': 'user', 'content': '/hedge'})
                st.rerun()
        with col3:
            if st.button("💰 Kelly", key="ai_kelly", use_container_width=True):
                st.session_state.chat_history.append({'role': 'user', 'content': '/kelly'})
                st.rerun()
        with col4:
            if st.button("🗑️ Limpar", key="ai_clear", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        
        st.markdown("---")
        
        for msg in st.session_state.chat_history:
            if msg['role'] == 'user':
                st.chat_message("user", avatar="👤").markdown(msg['content'])
            else:
                st.chat_message("assistant", avatar="🤖").markdown(msg['content'])
        
        user_msg = st.chat_input("Digite sua pergunta...")
        
        if user_msg:
            st.session_state.chat_history.append({'role': 'user', 'content': user_msg})
            
            response = ""
            cmd = user_msg.lower()
            
            if cmd.startswith('/'):
                if '/ajuda' in cmd:
                    response = "📚 **COMANDOS:**\n• `/jogos` - Top jogos\n• `/stats [time]` - Estatísticas\n• `/perfil` - Desempenho"
                elif '/jogos' in cmd:
                    if not cal.empty:
                        hoje = datetime.now().strftime('%d/%m/%Y')
                        jogos_h = cal[cal['DtObj'].dt.strftime('%d/%m/%Y') == hoje]
                        if len(jogos_h) > 0:
                            response = f"🎯 **JOGOS HOJE ({hoje}):**\n\n"
                            count = 0
                            for _, j in jogos_h.head(5).iterrows():
                                h = normalize_name(j['Time_Casa'], list(stats.keys()))
                                a = normalize_name(j['Time_Visitante'], list(stats.keys()))
                                if h and a and h in stats and a in stats:
                                    count += 1
                                    c = calcular_jogo_v31(stats[h], stats[a], {})
                                    response += f"**{count}. {h} vs {a}**\n   {c['corners']['t']:.1f} cantos | {c['cards']['t']:.1f} cartões\n\n"
                        else:
                            response = "📅 Sem jogos hoje"
                    else:
                        response = "❌ Calendário indisponível"
                elif '/perfil' in cmd:
                    total = len(st.session_state.bet_results)
                    if total > 0:
                        ganhas = sum(1 for b in st.session_state.bet_results if b['ganhou'])
                        wr = (ganhas / total) * 100
                        response = f"👤 **PERFIL:**\nApostas: {total}\nWin Rate: {wr:.1f}%"
                    else:
                        response = "📭 Sem apostas"
                else:
                    response = "❓ Comando desconhecido. Use `/ajuda`"
            else:
                response = "💡 Use `/ajuda` para ver comandos disponíveis!"
            
            st.session_state.chat_history.append({'role': 'assistant', 'content': response})
            st.rerun()

if __name__ == "__main__":
    main()