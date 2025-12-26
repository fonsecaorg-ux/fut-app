import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import poisson
import json
import hmac
import os
from datetime import datetime, timedelta
import uuid
from difflib import get_close_matches
from typing import Dict, List, Optional
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# 0. CONFIGURAÇÃO
# ==============================================================================
st.set_page_config(
    page_title="FutPrevisão V31 Pro",
    layout="wide",
    page_icon="⚽",
    initial_sidebar_state="expanded"
)

# CSS
st.markdown("""
<style>
    .stApp { background: linear-gradient(to bottom, #f8f9fa 0%, #ffffff 100%); }
    
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
    
    /* Cards */
    .metric-card { 
        background: white; 
        padding: 20px; 
        border-radius: 10px; 
        box-shadow: 0 2px 8px rgba(0,0,0,0.1); 
        border-left: 4px solid #667eea;
    }
    
    /* Bilhete */
    .ticket-item {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 3px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. AUTENTICAÇÃO
# ==============================================================================
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    def password_entered():
        if "passwords" in st.secrets:
            user = st.session_state["username"]
            password = st.session_state["password"]
            
            if user in st.secrets["passwords"] and hmac.compare_digest(password, st.secrets["passwords"][user]):
                st.session_state["password_correct"] = True
                del st.session_state["password"]
                del st.session_state["username"]
            else:
                st.error("😕 Usuário ou senha incorretos")
        else:
            st.session_state["password_correct"] = True  # Modo dev

    if st.session_state["password_correct"]:
        return True

    st.markdown("### 🔒 FutPrevisão V31 - Login")
    st.text_input("Usuário", key="username")
    st.text_input("Senha", type="password", key="password")
    st.button("Entrar", on_click=password_entered)
    return False

if not check_password():
    st.stop()

# ==============================================================================
# 2. CARREGAMENTO DE DADOS
# ==============================================================================

# Configuração GitHub (caso arquivos não estejam localmente)
GITHUB_USER = "seu_usuario"
GITHUB_REPO = "seu_repo"
GITHUB_BRANCH = "main"
GITHUB_BASE_URL = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{GITHUB_BRANCH}/"

LEAGUE_FILES = {
    'Premier League': ['Premier_League_25_26.csv'],
    'La Liga': ['La_Liga_25_26.csv'],
    'Serie A': ['Serie_A_25_26.csv'],
    'Bundesliga': ['Bundesliga_25_26.csv'],
    'Ligue 1': ['Ligue_1_25_26.csv'],
    'Championship': ['Championship_Inglaterra_25_26.csv'],
    'Bundesliga 2': ['Bundesliga_2.csv'],
    'Pro League': ['Pro_League_Belgica_25_26.csv'],
    'Super Lig': ['Super_Lig_Turquia_25_26.csv'],
    'Premiership': ['Premiership_Escocia_25_26.csv']
}

NAME_MAPPING = {
    'Man United': 'Manchester Utd', 'Man City': 'Manchester City',
    'Spurs': 'Tottenham', 'Newcastle': 'Newcastle Utd',
    'Wolves': 'Wolverhampton', 'Nottm Forest': "Nott'm Forest",
    'Inter': 'Inter Milan', 'Ath Madrid': 'Atletico Madrid'
}

def normalize_name(name: str, known_teams: List[str]) -> Optional[str]:
    if not name: return None
    name = name.strip()
    if name in NAME_MAPPING: name = NAME_MAPPING[name]
    if name in known_teams: return name
    matches = get_close_matches(name, known_teams, n=1, cutoff=0.6)
    return matches[0] if matches else None

@st.cache_data(ttl=3600)
def load_all_data():
    stats_db = {}
    possible_paths = ['.', './data', 'data', os.getcwd()]
    
    def try_read_csv(filename):
        # Local
        for p in possible_paths:
            fpath = os.path.join(p, filename)
            if os.path.exists(fpath):
                try: return pd.read_csv(fpath, encoding='utf-8')
                except: 
                    try: return pd.read_csv(fpath, encoding='latin1')
                    except: pass
        
        # GitHub fallback
        try:
            url = f"{GITHUB_BASE_URL}{filename}"
            return pd.read_csv(url)
        except:
            return pd.DataFrame()
    
    # Carregar stats
    for league_name, filenames in LEAGUE_FILES.items():
        df = pd.DataFrame()
        for fname in filenames:
            df = try_read_csv(fname)
            if not df.empty: break
        
        if df.empty: continue
        
        teams = set(df['HomeTeam'].dropna().unique()) | set(df['AwayTeam'].dropna().unique())
        
        for team in teams:
            h_games = df[df['HomeTeam'] == team]
            a_games = df[df['AwayTeam'] == team]
            
            def safe_mean(df_h, df_a, col_h, col_a, default):
                val_h = df_h[col_h].mean() if col_h in df_h.columns and not df_h.empty else default
                val_a = df_a[col_a].mean() if col_a in df_a.columns and not df_a.empty else default
                return (val_h + val_a) / 2
            
            stats_db[team] = {
                'corners': safe_mean(h_games, a_games, 'HC', 'AC', 5.0),
                'cards': safe_mean(h_games, a_games, 'HY', 'AY', 2.0),
                'fouls': safe_mean(h_games, a_games, 'HF', 'AF', 11.0),
                'goals_f': safe_mean(h_games, a_games, 'FTHG', 'FTAG', 1.3),
                'goals_a': safe_mean(h_games, a_games, 'FTAG', 'FTHG', 1.3),
                'shots_on_target': safe_mean(h_games, a_games, 'HST', 'AST', 4.5),
                'league': league_name,
                'games': len(h_games) + len(a_games)
            }
    
    # Calendário
    cal = try_read_csv('calendario_ligas.csv')
    if not cal.empty and 'Data' in cal.columns:
        cal['DtObj'] = pd.to_datetime(cal['Data'], format='%d/%m/%Y', errors='coerce')
    
    # Árbitros
    refs = {}
    ref_df = try_read_csv('arbitros_5_ligas_2025_2026.csv')
    if not ref_df.empty:
        for _, row in ref_df.iterrows():
            refs[row['Arbitro']] = {
                'factor': row.get('Media_Cartoes_Por_Jogo', 4.0) / 4.0,
                'avg_cards': row.get('Media_Cartoes_Por_Jogo', 4.0)
            }
    
    return stats_db, cal, refs

# Carregar dados globais
STATS, CAL, REFS = load_all_data()

# ==============================================================================
# 3. MOTOR DE CÁLCULO V31
# ==============================================================================
def calcular_jogo_v31(home_stats, away_stats, ref_data):
    """Motor V31 - Causality Engine"""
    
    # Pressão ofensiva
    shots_h = home_stats.get('shots_on_target', 4.5)
    pressure = 1.15 if shots_h > 5.5 else 1.0
    
    # Cantos
    corners_h = home_stats['corners'] * 1.15 * pressure
    corners_a = away_stats['corners'] * 0.90
    corners_t = corners_h + corners_a
    
    # Violência
    fouls_h = home_stats.get('fouls', 11.0)
    fouls_a = away_stats.get('fouls', 11.0)
    violence = 1.10 if (fouls_h + fouls_a) > 24 else 0.95
    
    ref_factor = ref_data.get('factor', 1.0) if ref_data else 1.0
    
    cards_h = home_stats['cards'] * violence * ref_factor
    cards_a = away_stats['cards'] * violence * ref_factor
    cards_t = cards_h + cards_a
    
    # xG
    xg_h = (home_stats['goals_f'] * away_stats['goals_a']) / 1.3
    xg_a = (away_stats['goals_f'] * home_stats['goals_a']) / 1.3
    
    return {
        'corners': {'h': corners_h, 'a': corners_a, 't': corners_t},
        'cards': {'h': cards_h, 'a': cards_a, 't': cards_t},
        'goals': {'h': xg_h, 'a': xg_a},
        'metadata': {'pressure': pressure, 'violence': violence, 'ref_factor': ref_factor}
    }

# ==============================================================================
# 4. SESSION STATE
# ==============================================================================
if 'current_ticket' not in st.session_state:
    st.session_state.current_ticket = []

if 'bankroll_history' not in st.session_state:
    st.session_state.bankroll_history = [1000.0]

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# ==============================================================================
# 5. FUNÇÕES AUXILIARES
# ==============================================================================
def format_currency(value):
    return f"R$ {value:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')

def calculate_kelly(prob, odd, bankroll):
    """Kelly Criterion"""
    if prob >= 1.0 or odd <= 1.0:
        return {'stake': 0, 'percentage': 0, 'recommendation': 'Inválido'}
    
    q = 1 - prob
    kelly_frac = (prob * odd - 1) / (odd - 1)
    
    if kelly_frac <= 0:
        return {'stake': 0, 'percentage': 0, 'recommendation': 'Sem Value'}
    
    kelly_pct = min(kelly_frac, 0.15) * 100  # Limite 15%
    stake = bankroll * (kelly_pct / 100)
    
    if kelly_pct < 1:
        rec = "Stake Muito Baixo"
    elif kelly_pct < 3:
        rec = "Stake Conservador"
    elif kelly_pct < 8:
        rec = "Stake Moderado"
    else:
        rec = "Stake Agressivo"
    
    return {
        'stake': stake,
        'percentage': kelly_pct,
        'recommendation': rec
    }

# ==============================================================================
# 6. PÁGINAS DO APP
# ==============================================================================

def render_scanner():
    """Tab 1: Scanner de Jogos"""
    st.header("🔍 Scanner Inteligente de Jogos")
    
    if CAL.empty:
        st.warning("📅 Calendário não carregado")
        return
    
    # Filtros
    col1, col2 = st.columns(2)
    
    with col1:
        date_filter = st.date_input(
            "📅 Data",
            value=datetime.now(),
            format="DD/MM/YYYY"
        )
    
    with col2:
        market_filter = st.selectbox(
            "🎯 Mercado",
            ["Todos", "Gols (Over 2.5)", "Cantos (Over 10.5)", "Cartões (Over 4.5)"]
        )
    
    # Buscar jogos
    date_str = date_filter.strftime('%d/%m/%Y')
    jogos_dia = CAL[CAL['DtObj'].dt.strftime('%d/%m/%Y') == date_str]
    
    if len(jogos_dia) == 0:
        st.info(f"📭 Sem jogos para {date_str}")
        return
    
    st.subheader(f"⚽ Jogos Encontrados: {len(jogos_dia)}")
    
    recommendations = []
    
    for _, jogo in jogos_dia.iterrows():
        h = normalize_name(jogo['Time_Casa'], list(STATS.keys()))
        a = normalize_name(jogo['Time_Visitante'], list(STATS.keys()))
        
        if not h or not a or h not in STATS or a not in STATS:
            continue
        
        calc = calcular_jogo_v31(STATS[h], STATS[a], {})
        
        # Score baseado no mercado
        if market_filter == "Gols (Over 2.5)":
            total_gols = calc['goals']['h'] + calc['goals']['a']
            if total_gols > 2.5:
                score = int(total_gols * 30)
                prob = min(int((total_gols - 2.5) * 30 + 65), 85)
                info = f"Over 2.5 Gols ({prob}%)"
            else:
                continue
                
        elif market_filter == "Cantos (Over 10.5)":
            if calc['corners']['t'] > 10.5:
                score = int(calc['corners']['t'] * 8)
                prob = min(int((calc['corners']['t'] - 10.5) * 12 + 70), 85)
                info = f"Over 10.5 Cantos ({prob}%)"
            else:
                continue
                
        elif market_filter == "Cartões (Over 4.5)":
            if calc['cards']['t'] > 4.5:
                score = int(calc['cards']['t'] * 15)
                prob = min(int((calc['cards']['t'] - 4.5) * 18 + 68), 82)
                info = f"Over 4.5 Cartões ({prob}%)"
            else:
                continue
        else:
            # Todos
            score = int((calc['corners']['t'] * 6) + (calc['cards']['t'] * 10))
            info = f"Cantos: {calc['corners']['t']:.1f} | Cartões: {calc['cards']['t']:.1f}"
            prob = 70
        
        recommendations.append({
            'jogo': f"{h} vs {a}",
            'hora': jogo.get('Hora', 'N/A'),
            'liga': STATS[h]['league'],
            'score': score,
            'info': info,
            'prob': prob,
            'calc': calc
        })
    
    if not recommendations:
        st.warning("⚠️ Nenhum jogo atende aos critérios selecionados")
        return
    
    # Ordenar por score
    recommendations.sort(key=lambda x: x['score'], reverse=True)
    
    # Exibir
    for i, rec in enumerate(recommendations[:10], 1):
        emoji = "🔥" if i == 1 else "✅" if i <= 3 else "📊"
        
        with st.container():
            col_jogo, col_dados, col_acao = st.columns([3, 2, 1])
            
            with col_jogo:
                st.markdown(f"### {emoji} {i}. {rec['jogo']}")
                st.caption(f"🕐 {rec['hora']} | 🏆 {rec['liga']}")
            
            with col_dados:
                st.metric("Projeção", rec['info'])
                st.caption(f"Confiança: {rec['prob']}%")
            
            with col_acao:
                if st.button("➕ Add Bilhete", key=f"add_{i}"):
                    # Adicionar ao bilhete
                    selection = {
                        'game': rec['jogo'],
                        'market': rec['info'].split('(')[0].strip(),
                        'prob': rec['prob'],
                        'market_display': rec['info']
                    }
                    st.session_state.current_ticket.append(selection)
                    st.success(f"✅ Adicionado: {rec['jogo']}")
                    st.rerun()
            
            st.divider()

def render_construtor():
    """Tab 2: Construtor de Bilhetes"""
    st.header("📋 Construtor de Bilhetes")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🎯 Bilhete Atual")
        
        if not st.session_state.current_ticket:
            st.info("📭 Bilhete vazio. Use o Scanner para adicionar jogos!")
        else:
            # Calcular odds
            prob_total = 1.0
            for sel in st.session_state.current_ticket:
                prob_total *= (sel['prob'] / 100)
            
            odd_total = 1 / prob_total if prob_total > 0 else 1.0
            prob_total_pct = prob_total * 100
            
            # Exibir seleções
            for i, sel in enumerate(st.session_state.current_ticket):
                with st.container():
                    col_info, col_del = st.columns([4, 1])
                    
                    with col_info:
                        st.markdown(f"""
                        <div class="ticket-item">
                            <b>{sel['game']}</b><br>
                            {sel['market_display']}<br>
                            <small>Prob: {sel['prob']}%</small>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_del:
                        if st.button("🗑️", key=f"del_{i}"):
                            st.session_state.current_ticket.pop(i)
                            st.rerun()
            
            st.divider()
            
            # Resumo
            st.subheader("📊 Resumo do Bilhete")
            
            col_a, col_b = st.columns(2)
            col_a.metric("Seleções", len(st.session_state.current_ticket))
            col_a.metric("Probabilidade Total", f"{prob_total_pct:.1f}%")
            
            col_b.metric("Odd Estimada", f"@{odd_total:.2f}")
            
            # Value bet
            if odd_total > 0:
                value = prob_total_pct / 100 * odd_total
                col_b.metric(
                    "Value",
                    f"{value:.2f}",
                    delta="Value Bet ✅" if value > 1.0 else "Sem Value ❌"
                )
            
            st.divider()
            
            # Limpar
            if st.button("🗑️ Limpar Bilhete", use_container_width=True):
                st.session_state.current_ticket = []
                st.rerun()
    
    with col2:
        st.subheader("💰 Gestão de Stake")
        
        bankroll = st.session_state.bankroll_history[-1]
        st.metric("💼 Banca Atual", format_currency(bankroll))
        
        if st.session_state.current_ticket:
            prob_total = 1.0
            for sel in st.session_state.current_ticket:
                prob_total *= (sel['prob'] / 100)
            
            odd_total = 1 / prob_total
            
            kelly_result = calculate_kelly(prob_total, odd_total, bankroll)
            
            st.divider()
            st.markdown("### 📈 Kelly Criterion")
            
            st.metric("Stake Recomendado", format_currency(kelly_result['stake']))
            st.metric("% da Banca", f"{kelly_result['percentage']:.2f}%")
            
            st.info(f"💡 {kelly_result['recommendation']}")

def render_rankings():
    """Tab 3: Rankings"""
    st.header("🏆 Rankings & Estatísticas")
    
    metric = st.selectbox(
        "📊 Métrica",
        ["Cantos", "Cartões", "Gols Feitos", "Gols Sofridos", "Faltas"]
    )
    
    # Mapear métrica
    metric_map = {
        'Cantos': 'corners',
        'Cartões': 'cards',
        'Gols Feitos': 'goals_f',
        'Gols Sofridos': 'goals_a',
        'Faltas': 'fouls'
    }
    
    metric_key = metric_map[metric]
    
    # Criar ranking
    ranking = []
    for team, stats in STATS.items():
        ranking.append({
            'Time': team,
            'Liga': stats['league'],
            'Valor': stats[metric_key],
            'Jogos': stats['games']
        })
    
    # Ordenar
    ranking.sort(key=lambda x: x['Valor'], reverse=True)
    
    # Exibir Top 20
    st.subheader(f"🥇 Top 20 - {metric}")
    
    for i, item in enumerate(ranking[:20], 1):
        col1, col2, col3, col4 = st.columns([1, 3, 2, 1])
        
        col1.markdown(f"**#{i}**")
        col2.markdown(f"**{item['Time']}**")
        col3.caption(item['Liga'])
        col4.metric("", f"{item['Valor']:.2f}")
        
        if i < 20:
            st.divider()

def render_gestao_banca():
    """Tab 4: Gestão de Banca"""
    st.header("💼 Gestão Profissional de Banca")
    
    # Config
    config_file = "config_banca.json"
    if os.path.exists(config_file):
        with open(config_file, "r") as f:
            config = json.load(f)
    else:
        config = {"banca_inicial": 1000.0}
    
    # Histórico
    data_file = "historico_bilhetes.json"
    if os.path.exists(data_file):
        with open(data_file, "r") as f:
            tickets = json.load(f)
    else:
        tickets = []
    
    # KPIs
    lucro_total = sum(t.get('Lucro', 0) for t in tickets)
    banca_atual = config['banca_inicial'] + lucro_total
    
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("💰 Banca Atual", format_currency(banca_atual))
    k2.metric("📈 Lucro/Prejuízo", format_currency(lucro_total), delta=f"{(lucro_total/config['banca_inicial']*100):.1f}%")
    k3.metric("📝 Bilhetes", len(tickets))
    
    if tickets:
        wins = len([t for t in tickets if "Green" in t.get('Resultado', '')])
        win_rate = (wins / len(tickets)) * 100
        k4.metric("🎯 Win Rate", f"{win_rate:.1f}%")
    
    st.divider()
    
    # Tabs
    tab1, tab2 = st.tabs(["➕ Novo Bilhete", "📜 Histórico"])
    
    with tab1:
        with st.form("new_ticket_form"):
            st.subheader("Registrar Novo Bilhete")
            
            c1, c2 = st.columns(2)
            stake = c1.number_input("Stake (R$)", value=10.0, step=5.0)
            odd = c2.number_input("Odd Total", value=2.00, step=0.01)
            
            resultado = st.selectbox(
                "Resultado",
                ["Green ✅", "Red ❌", "Green (Cashout) 💰", "Reembolso 🔄"]
            )
            
            descricao = st.text_area(
                "Descrição/Jogos",
                placeholder="Ex: Arsenal x Liverpool - Over 2.5 Gols..."
            )
            
            if st.form_submit_button("💾 Salvar Bilhete"):
                # Calcular lucro
                if "Green ✅" in resultado:
                    lucro = (stake * odd) - stake
                elif "Green (Cashout)" in resultado:
                    lucro = (stake * odd * 0.85) - stake  # 85% do valor
                elif "Red" in resultado:
                    lucro = -stake
                else:
                    lucro = 0
                
                # Salvar
                ticket = {
                    'id': str(uuid.uuid4())[:8],
                    'Data': datetime.now().strftime("%d/%m/%Y %H:%M"),
                    'Descricao': descricao,
                    'Stake': stake,
                    'Odd': odd,
                    'Resultado': resultado,
                    'Lucro': lucro
                }
                
                tickets.insert(0, ticket)
                
                with open(data_file, "w") as f:
                    json.dump(tickets, f)
                
                st.success("✅ Bilhete salvo com sucesso!")
                st.rerun()
    
    with tab2:
        if not tickets:
            st.info("📭 Nenhum bilhete registrado ainda")
        else:
            for ticket in tickets:
                lucro = ticket.get('Lucro', 0)
                
                if lucro > 0:
                    color = "#d4edda"
                    emoji = "💚"
                elif lucro < 0:
                    color = "#f8d7da"
                    emoji = "❌"
                else:
                    color = "#fff3cd"
                    emoji = "🔄"
                
                st.markdown(f"""
                <div style="background-color: {color}; padding: 15px; border-radius: 10px; margin-bottom: 10px;">
                    <div style="display: flex; justify-content: space-between;">
                        <div>
                            <b>{emoji} {ticket['Data']}</b><br>
                            {ticket['Descricao']}<br>
                            <small>Stake: {format_currency(ticket['Stake'])} @ {ticket['Odd']:.2f}</small>
                        </div>
                        <div style="text-align: right;">
                            <h3>{format_currency(lucro)}</h3>
                            <small>{ticket['Resultado']}</small>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

def render_chat():
    """Tab 5: Chatbot IA"""
    st.header("🤖 SuperBot V31 - Assistente Inteligente")
    
    # Inicializar chat
    if not st.session_state.chat_history:
        welcome = f"""👋 **Olá! Sou o SuperBot V31!**

📅 Hoje é **{datetime.now().strftime('%d/%m/%Y')}**

💬 **Posso ajudar com:**
• Estatísticas de times
• Análise de confrontos
• Jogos de hoje/amanhã
• Rankings e comparações

🎯 **Exemplos:**
• "Como está o Liverpool?"
• "Arsenal vs Chelsea"
• "Jogos de hoje"
• "Top 10 cantos"

Digite sua pergunta abaixo! 👇"""
        
        st.session_state.chat_history.append({
            'role': 'assistant',
            'content': welcome
        })
    
    # Botões rápidos
    col1, col2, col3, col4 = st.columns(4)
    
    if col1.button("⚽ Jogos Hoje", use_container_width=True):
        st.session_state.chat_history.append({'role': 'user', 'content': 'Quais os jogos de hoje?'})
        st.rerun()
    
    if col2.button("🏆 Rankings", use_container_width=True):
        st.session_state.chat_history.append({'role': 'user', 'content': 'Top 10 times em cantos'})
        st.rerun()
    
    if col3.button("📊 Análise", use_container_width=True):
        st.session_state.chat_history.append({'role': 'user', 'content': 'Me ajuda a analisar um jogo'})
        st.rerun()
    
    if col4.button("🗑️ Limpar", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
    
    st.divider()
    
    # Exibir histórico
    for msg in st.session_state.chat_history:
        if msg['role'] == 'user':
            st.chat_message("user", avatar="👤").markdown(msg['content'])
        else:
            st.chat_message("assistant", avatar="🤖").markdown(msg['content'])
    
    # Input
    user_input = st.chat_input("Digite sua pergunta...")
    
    if user_input:
        st.session_state.chat_history.append({'role': 'user', 'content': user_input})
        
        cmd = user_input.lower()
        response = ""
        
        # Processar pergunta
        try:
            # ESTATÍSTICAS DE TIME
            if 'como está' in cmd or 'como esta' in cmd or 'como ta' in cmd:
                # Buscar time
                team = None
                for t in STATS.keys():
                    if t.lower() in cmd:
                        team = t
                        break
                
                if team:
                    s = STATS[team]
                    response = f"""**📊 ANÁLISE - {team}**

🏟️ **Liga:** {s['league']}
📈 **Jogos Analisados:** {s['games']}

⚽ **ATAQUE:**
• Gols: {s['goals_f']:.2f}/jogo
{'🔥 Muito ofensivo!' if s['goals_f'] > 2.0 else '📊 Médio' if s['goals_f'] > 1.5 else '⚠️ Fraco'}

🛡️ **DEFESA:**
• Gols sofridos: {s['goals_a']:.2f}/jogo
{'✅ Sólida!' if s['goals_a'] < 1.0 else '📊 Média' if s['goals_a'] < 1.5 else '⚠️ Vazada'}

🔶 **CANTOS:** {s['corners']:.1f}/jogo
🟨 **CARTÕES:** {s['cards']:.1f}/jogo
⚠️ **FALTAS:** {s['fouls']:.1f}/jogo"""
                else:
                    response = "⚠️ Não identifiquei o time. Tente: 'Como está o Liverpool?'"
            
            # ANÁLISE DE JOGO
            elif ' vs ' in cmd or ' x ' in cmd:
                separator = ' vs ' if ' vs ' in cmd else ' x '
                parts = cmd.split(separator)
                
                if len(parts) == 2:
                    t1 = normalize_name(parts[0].strip(), list(STATS.keys()))
                    t2 = normalize_name(parts[1].strip(), list(STATS.keys()))
                    
                    if t1 and t2:
                        calc = calcular_jogo_v31(STATS[t1], STATS[t2], {})
                        
                        response = f"""**⚔️ ANÁLISE: {t1} vs {t2}**

⚽ **EXPECTED GOALS (xG):**
• {t1}: {calc['goals']['h']:.2f}
• {t2}: {calc['goals']['a']:.2f}

🔶 **CANTOS ESPERADOS:**
• Total: {calc['corners']['t']:.1f}
• {t1}: {calc['corners']['h']:.1f}
• {t2}: {calc['corners']['a']:.1f}

🟨 **CARTÕES ESPERADOS:**
• Total: {calc['cards']['t']:.1f}
• {t1}: {calc['cards']['h']:.1f}
• {t2}: {calc['cards']['a']:.1f}

💡 **RECOMENDAÇÕES:**"""
                        
                        # Adicionar recos
                        total_gols = calc['goals']['h'] + calc['goals']['a']
                        if total_gols > 2.5:
                            prob = min(int((total_gols - 2.5) * 30 + 65), 85)
                            response += f"\n✅ Over 2.5 Gols ({prob}%)"
                        
                        if calc['corners']['t'] > 10.5:
                            prob = min(int((calc['corners']['t'] - 10.5) * 12 + 70), 85)
                            response += f"\n✅ Over 10.5 Cantos ({prob}%)"
                        
                        if calc['cards']['t'] > 4.5:
                            prob = min(int((calc['cards']['t'] - 4.5) * 18 + 68), 82)
                            response += f"\n✅ Over 4.5 Cartões ({prob}%)"
                    else:
                        response = "❌ Times não encontrados. Verifique os nomes."
                else:
                    response = "❌ Use o formato: 'Time A vs Time B'"
            
            # JOGOS DE HOJE
            elif 'jogo' in cmd and ('hoje' in cmd or 'agora' in cmd):
                if CAL.empty:
                    response = "📅 Calendário não disponível"
                else:
                    hoje = datetime.now().strftime('%d/%m/%Y')
                    jogos = CAL[CAL['DtObj'].dt.strftime('%d/%m/%Y') == hoje]
                    
                    if len(jogos) > 0:
                        response = f"⚽ **JOGOS HOJE ({hoje}):**\n\n"
                        
                        for i, (_, jogo) in enumerate(jogos.head(8).iterrows(), 1):
                            response += f"{i}. **{jogo['Time_Casa']} vs {jogo['Time_Visitante']}**\n"
                            response += f"   🕐 {jogo.get('Hora', 'N/A')}\n\n"
                    else:
                        response = f"📭 Nenhum jogo encontrado para hoje ({hoje})"
            
            # TOP RANKINGS
            elif 'top' in cmd or 'ranking' in cmd:
                if 'canto' in cmd:
                    metric = 'corners'
                    label = 'Cantos'
                elif 'cartao' in cmd or 'cartão' in cmd:
                    metric = 'cards'
                    label = 'Cartões'
                elif 'gol' in cmd:
                    metric = 'goals_f'
                    label = 'Gols'
                else:
                    metric = 'corners'
                    label = 'Cantos'
                
                ranking = sorted(
                    [(team, stats[metric]) for team, stats in STATS.items()],
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
                
                response = f"🏆 **TOP 10 - {label}:**\n\n"
                
                for i, (team, value) in enumerate(ranking, 1):
                    emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "📊"
                    response += f"{emoji} {i}. **{team}** - {value:.2f}/jogo\n"
            
            # FALLBACK
            else:
                response = """🤔 Não entendi completamente...

💡 **Tente:**
• "Como está o Arsenal"
• "Liverpool vs Chelsea"
• "Jogos de hoje"
• "Top 10 cantos"

Ou seja mais específico!"""
        
        except Exception as e:
            response = f"⚠️ Erro ao processar: {str(e)}"
        
        st.session_state.chat_history.append({'role': 'assistant', 'content': response})
        st.rerun()

# ==============================================================================
# 7. MAIN APP
# ==============================================================================
def main():
    # Sidebar
    st.sidebar.title("💎 FutPrevisão V31")
    st.sidebar.caption("Sistema Profissional de Análise")
    
    # Menu
    page = st.sidebar.radio(
        "📍 Navegação",
        [
            "🔍 Scanner",
            "📋 Construtor",
            "🏆 Rankings",
            "💼 Gestão de Banca",
            "🤖 Chatbot IA"
        ]
    )
    
    st.sidebar.divider()
    
    # Stats
    st.sidebar.metric("📚 Times na Base", len(STATS))
    st.sidebar.metric("🏟️ Ligas", len(LEAGUE_FILES))
    
    if not CAL.empty:
        st.sidebar.metric("📅 Jogos no Calendário", len(CAL))
    
    st.sidebar.divider()
    st.sidebar.info("⚡ Motor V31 Ativo")
    
    # Renderizar página
    if page == "🔍 Scanner":
        render_scanner()
    elif page == "📋 Construtor":
        render_construtor()
    elif page == "🏆 Rankings":
        render_rankings()
    elif page == "💼 Gestão de Banca":
        render_gestao_banca()
    elif page == "🤖 Chatbot IA":
        render_chat()

if __name__ == "__main__":
    main()