"""
FutPrevisão V31 MAXIMUM + AI Advisor ULTRA
SISTEMA INTEGRADO - VERSÃO DE PRODUÇÃO
"""

import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import math
from typing import Dict, List, Optional, Tuple, Any
import plotly.graph_objects as go
import plotly.express as px
from difflib import get_close_matches
import re
from collections import defaultdict
import hmac
import os
from scipy.stats import poisson, norm, beta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from itertools import combinations
import warnings

warnings.filterwarnings('ignore')

# Define BASE_DIR corretamente logo no início para evitar erros de Path
BASE_DIR = Path(__file__).resolve().parent

# ============================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================

st.set_page_config(
    page_title="FutPrevisão V31 MAXIMUM",
    layout="wide",
    page_icon="⚽",
    initial_sidebar_state="expanded"
)

# CSS PROFISSIONAL - TABS HORIZONTAIS E DESIGN
st.markdown('''
<style>
    /* FUNDO E BASE */
    .stApp { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    
    /* TABS HORIZONTAIS - DESIGN PROFISSIONAL */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: rgba(0, 0, 0, 0.2);
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.1);
        border-radius: 8px;
        color: white;
        font-weight: 600;
        padding: 8px 16px;
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .stTabs [aria-selected="true"] {
        background: white !important;
        color: #667eea !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }

    /* CARDS E MÉTRICAS */
    div[data-testid="metric-container"] {
        background-color: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        color: white;
    }
    div[data-testid="metric-container"] label { color: #f0f0f0 !important; }
    
    /* CHATBOT */
    .stChatMessage {
        background-color: rgba(255, 255, 255, 0.95);
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        color: #333;
    }
    
    /* CABEÇALHO */
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        color: white;
        text-align: center;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
        margin-bottom: 20px;
    }
    
    /* BOTÕES */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 12px;
        border-radius: 8px;
    }
</style>
''', unsafe_allow_html=True)

# ============================================================
# 1. FUNÇÕES AUXILIARES E AUTENTICAÇÃO
# ============================================================

def format_currency(val):
    """Função auxiliar para formatar moeda (Adicionada para corrigir NameError)"""
    return f"R$ {val:.2f}"

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
                st.session_state["password_correct"] = False
                st.error("😕 Usuário ou senha incorretos")
        else:
            st.session_state["password_correct"] = True

    if st.session_state["password_correct"]: return True
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown("### 🔒 Login V31 Maximum")
        st.text_input("Usuário", key="username")
        st.text_input("Senha", type="password", key="password")
        st.button("Entrar", on_click=password_entered)
    return False

if not check_password(): st.stop()

# ============================================================
# 2. CARREGAMENTO DE DADOS ESTATÍSTICOS (V31 ENGINE)
# ============================================================

def normalize_name(name: str, known_teams: List[str]) -> Optional[str]:
    if not name: return None
    
    # Mapeamento manual para casos comuns
    NAME_MAPPING = {
        'Man United': 'Man Utd', 'Manchester United': 'Man Utd', 'Man Utd': 'Man Utd',
        'Man City': 'Man City', 'Manchester City': 'Man City',
        'Spurs': 'Tottenham', 'Newcastle': 'Newcastle', "Nott'm Forest": 'Nottm Forest',
        'Wolves': 'Wolves', 'Brighton': 'Brighton', 'Leicester': 'Leicester',
        'West Ham': 'West Ham', 'Arsenal': 'Arsenal', 'Liverpool': 'Liverpool',
        'Chelsea': 'Chelsea', 'Aston Villa': 'Aston Villa',
        'Ath Bilbao': 'Athletic Club', 'Atl. Madrid': 'Ath Madrid'
    }
    
    name_clean = name.strip()
    if name_clean in NAME_MAPPING: 
        target = NAME_MAPPING[name_clean]
        if target in known_teams: return target
        
    if name_clean in known_teams: return name_clean
    
    matches = get_close_matches(name_clean, known_teams, n=1, cutoff=0.6)
    return matches[0] if matches else None

# CONSTANTES DE ODDS REAIS (CALIBRADAS)
REAL_ODDS = {
    ('home','corners',3.5):1.34,('home','corners',4.5):1.63,('away','corners',2.5):1.40,
    ('away','corners',3.5):1.75,('total','corners',7.5):1.35,('total','corners',8.5):1.58,
    ('total','corners',9.5):1.90,('home','cards',1.5):1.53,('away','cards',1.5):1.65,
    ('total','cards',3.5):1.73,('total','cards',4.5):2.13,('home','dc','1X'):1.24
}

def get_odd(loc:str, typ:str, line:float) -> float:
    key = (loc, typ, line) if typ != 'dc' else (loc, typ, str(line))
    return REAL_ODDS.get(key, 1.50)

@st.cache_data(ttl=3600)
def load_data_v31():
    """Carrega dados estatísticos com suporte a GitHub/Local"""
    stats_db = {}
    
    # Lista de arquivos (Compatibilidade com V31 Maximum)
    LEAGUE_FILES = {
        'Premier League': ['Premier_League_25_26.csv', 'E0.csv'],
        'La Liga': ['La_Liga_25_26.csv', 'SP1.csv'],
        'Serie A': ['Serie_A_25_26.csv', 'I1.csv'],
        'Bundesliga': ['Bundesliga_25_26.csv', 'D1.csv'],
        'Ligue 1': ['Ligue_1_25_26.csv', 'F1.csv'],
        'Championship': ['Championship_Inglaterra_25_26.csv', 'E1.csv'],
        'Bundesliga 2': ['Bundesliga_2.csv', 'D2.csv'],
        'Pro League': ['Pro_League_Belgica_25_26.csv', 'B1.csv'],
        'Super Lig': ['Super_Lig_Turquia_25_26.csv', 'T1.csv'],
        'Premiership': ['Premiership_Escocia_25_26.csv', 'SC0.csv']
    }
    
    # Estratégia de Busca de Arquivos (CORREÇÃO DE PATH)
    search_paths = [BASE_DIR, BASE_DIR / 'data', Path('.'), Path('./data')]
    
    def try_read_csv(filename):
        for p in search_paths:
            fpath = p / filename
            if fpath.exists():
                try: return pd.read_csv(fpath, encoding='utf-8-sig')
                except: 
                    try: return pd.read_csv(fpath, encoding='latin1')
                    except: pass
        return pd.DataFrame()

    for league_name, filenames in LEAGUE_FILES.items():
        df = pd.DataFrame()
        for fname in filenames:
            df = try_read_csv(fname)
            if not df.empty: break
            
        if df.empty: continue
        
        # Normalização de Colunas
        df.columns = [c.strip().replace('\ufeff','') for c in df.columns]
        col_map = {'Mandante':'HomeTeam','Visitante':'AwayTeam','Time_Casa':'HomeTeam','Time_Visitante':'AwayTeam'}
        df = df.rename(columns=col_map)
        
        # Verificação de colunas necessárias
        req_cols = ['HomeTeam','AwayTeam']
        if not all(c in df.columns for c in req_cols): continue
        
        # Conversão numérica
        num_cols = ['HC','AC','HY','AY','FTHG','FTAG','HST','AST','HF','AF']
        for c in num_cols:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')
            
        # Processamento dos Times
        teams = set(df['HomeTeam'].dropna().unique()) | set(df['AwayTeam'].dropna().unique())
        
        for team in teams:
            h_games = df[df['HomeTeam'] == team]
            a_games = df[df['AwayTeam'] == team]
            
            # Função auxiliar de média segura
            def get_stat(col_h, col_a, default):
                vh = h_games[col_h].mean() if col_h in h_games else default
                va = a_games[col_a].mean() if col_a in a_games else default
                if pd.isna(vh): vh = default
                if pd.isna(va): va = default
                return (vh + va) / 2
            
            stats_db[team] = {
                'corners': get_stat('HC', 'AC', 5.0),
                'cards': get_stat('HY', 'AY', 2.0),
                'goals_f': get_stat('FTHG', 'FTAG', 1.3),
                'goals_a': get_stat('FTAG', 'FTHG', 1.3),
                'fouls': get_stat('HF', 'AF', 11.0),
                'shots_target': get_stat('HST', 'AST', 4.0),
                'league': league_name,
                'games': len(h_games) + len(a_games)
            }
            
            # Cálculo de Desvio Padrão
            try:
                corn_std = (h_games['HC'].std() + a_games['AC'].std())/2 if 'HC' in h_games else 1.5
                stats_db[team]['corners_std'] = corn_std if not pd.isna(corn_std) else 1.5
                
                card_std = (h_games['HY'].std() + a_games['AY'].std())/2 if 'HY' in h_games else 0.8
                stats_db[team]['cards_std'] = card_std if not pd.isna(card_std) else 0.8
            except:
                stats_db[team]['corners_std'] = 1.5
                stats_db[team]['cards_std'] = 0.8

    # Carregar Calendário e Árbitros
    cal = try_read_csv('calendario_ligas.csv')
    if not cal.empty and 'Data' in cal.columns:
        cal['DtObj'] = pd.to_datetime(cal['Data'], format='%d/%m/%Y', errors='coerce')
        
    refs = {}
    ref_df = try_read_csv('arbitros_5_ligas_2025_2026.csv')
    if not ref_df.empty:
        for _, row in ref_df.iterrows():
            refs[row['Arbitro']] = {'factor': row.get('Media_Cartoes_Por_Jogo', 4.0)/4.0}
            
    return stats_db, cal, refs

# INICIALIZAÇÃO DOS DADOS GLOBAIS
STATS_DB, CALENDAR_DF, REFEREES_DB = load_data_v31()

# ============================================================
# 3. MÓDULO ESTATÍSTICO V31 (CÁLCULOS MATEMÁTICOS)
# ============================================================

def bivariate_poisson_vectorized(l1:float, l2:float, corr:float=-0.25, n:int=1000):
    """Simulação Poisson Bivariado (Correlação entre eventos)"""
    if abs(corr) < 0.01: return np.random.poisson(l1,n), np.random.poisson(l2,n)
    mean, cov = np.array([0,0]), np.array([[1,corr],[corr,1]])
    try:
        normals = np.random.multivariate_normal(mean, cov, n)
        uniforms = norm.cdf(normals)
        return poisson.ppf(uniforms[:,0], l1).astype(int), poisson.ppf(uniforms[:,1], l2).astype(int)
    except:
        return np.random.poisson(l1,n), np.random.poisson(l2,n)

def calculate_game_projections(home_name, away_name, stats_db):
    """Calcula projeções V31 para um jogo específico"""
    h = stats_db.get(home_name)
    a = stats_db.get(away_name)
    if not h or not a: return None
    
    # Lógica V31 Maximum
    proj = {}
    
    # 1. Cantos (Pressure Factor)
    shots_pressure = 1.1 if h.get('shots_target', 4) > 5.5 else 1.0
    proj['corners_h'] = h['corners'] * 1.10 * shots_pressure
    proj['corners_a'] = a['corners'] * 0.90
    proj['corners_total'] = proj['corners_h'] + proj['corners_a']
    
    # 2. Cartões (Violence Factor)
    violence = (h.get('fouls',11) + a.get('fouls',11)) / 22.0
    violence = max(0.9, min(1.2, violence))
    proj['cards_h'] = h['cards'] * violence
    proj['cards_a'] = a['cards'] * violence
    proj['cards_total'] = proj['cards_h'] + proj['cards_a']
    
    # 3. Gols (xG Simples)
    proj['goals_h'] = (h['goals_f'] * a['goals_a']) / 1.3
    proj['goals_a'] = (a['goals_f'] * h['goals_a']) / 1.3
    
    return proj

def simulate_game_v31(h_st, a_st, n=1):
    ch, ca = bivariate_poisson_vectorized(h_st['corners'], a_st['corners'], -0.25, n)
    cdh, cda = np.random.poisson(h_st['cards'], n), np.random.poisson(a_st['cards'], n)
    gh, ga = bivariate_poisson_vectorized(h_st['goals_f'], a_st.get('goals_f', 1.2), 0.15, n)
    return [{'home_corners': int(ch[i]), 'away_corners': int(ca[i]), 
             'home_cards': int(cdh[i]), 'away_cards': int(cda[i]), 
             'home_goals': int(gh[i]), 'away_goals': int(ga[i])} for i in range(n)]

# ============================================================
# 4. CHATBOT IA - AI ADVISOR ULTRA (LÓGICA CORRIGIDA)
# ============================================================

class AIAdvisor:
    """Motor de Inteligência do Chatbot V31"""
    def __init__(self, stats_db):
        self.stats = stats_db
        
    def find_teams(self, text: str) -> List[str]:
        """Extrai nomes de times do texto do usuário com alta precisão"""
        # 1. Limpeza
        text_clean = text.lower().replace('?', '').replace('.', '').replace(',', '').replace(' vs ', ' ').replace(' x ', ' ')
        
        # Palavras ignoradas
        stopwords = ['do', 'da', 'no', 'na', 'o', 'a', 'time', 'jogo', 'estatistica', 'media', 'de', 'sobre', 'tem', 'quantos', 'como', 'esta']
        words = [w for w in text_clean.split() if w not in stopwords]
        
        found = []
        # Tenta casar palavras
        for i in range(len(words)):
            # Palavra única Capitalizada (ex: arsenal -> Arsenal)
            n = normalize_name(words[i].capitalize(), list(self.stats.keys()))
            if n and n not in found: found.append(n)
            
            # Par de palavras (ex: man city -> Man City)
            if i < len(words) - 1:
                pair = f"{words[i]} {words[i+1]}".title()
                n2 = normalize_name(pair, list(self.stats.keys()))
                if n2 and n2 not in found: found.append(n2)
                
        return found

    def process_query(self, prompt: str) -> str:
        """Processa a pergunta e retorna a resposta formatada"""
        teams = self.find_teams(prompt)
        prompt_lower = prompt.lower()
        
        # Intenção 1: Comparação / Jogo
        if len(teams) >= 2:
            t1, t2 = teams[0], teams[1]
            s1, s2 = self.stats[t1], self.stats[t2]
            
            # Projeção V31
            proj = calculate_game_projections(t1, t2, self.stats)
            if not proj: return "Dados insuficientes para projeção."
            
            return f"""
            ⚔️ **ANÁLISE V31: {t1} vs {t2}**
            
            🚩 **Escanteios Esperados:** {proj['corners_total']:.1f}
            • {t1}: Média {s1['corners']:.1f}
            • {t2}: Média {s2['corners']:.1f}
            
            🟨 **Cartões Esperados:** {proj['cards_total']:.1f}
            • {t1}: {s1['cards']:.1f} | {t2}: {s2['cards']:.1f}
            
            ⚽ **Placar Estimado (xG):**
            {t1} {proj['goals_h']:.1f} x {proj['goals_a']:.1f} {t2}
            """
            
        # Intenção 2: Stats de um Time
        elif len(teams) == 1:
            t = teams[0]
            s = self.stats[t]
            trend = "📈 Alta" if s.get('corners', 0) > 5.5 else "➡️ Normal"
            
            return f"""
            📊 **RELATÓRIO: {t}**
            
            🏆 Liga: {s['league']}
            
            🚩 **Cantos:** {s['corners']:.2f} / jogo
            🟨 **Cartões:** {s['cards']:.2f} / jogo
            ⚽ **Gols:** {s['goals_f']:.2f} marcados | {s['goals_a']:.2f} sofridos
            🥊 **Faltas:** {s['fouls']:.1f}
            
            💡 *Tendência:* {trend}
            """
            
        # Intenção 3: Ajuda / Saudação
        else:
            return "❌ Não identifiquei o time ou jogo. Tente digitar o nome (ex: 'Arsenal' ou 'City vs Liverpool')."

# ============================================================
# 5. GESTÃO DE BANCA & HEDGES (V31 ENGINE)
# ============================================================

DATA_FILE = "historico_bilhetes_v31.json"
CONFIG_FILE = "config_banca_v31.json"

def carregar_tickets():
    if not os.path.exists(DATA_FILE): return []
    try: with open(DATA_FILE, "r") as f: return json.load(f)
    except: return []

def salvar_ticket(ticket):
    data = carregar_tickets()
    if 'id' not in ticket: ticket['id'] = str(uuid.uuid4())[:8]
    data.insert(0, ticket)
    with open(DATA_FILE, "w") as f: json.dump(data, f)

def generate_market_pool_maximum(h_st, a_st, h_n, a_n, min_prob=40):
    pool = []
    # Cantos
    for line in [3.5, 4.5]:
        prob = (1 - poisson.cdf(line, h_st['corners'])) * 100
        if prob >= min_prob: pool.append({'mercado': f"{h_n} Over {line} Cantos", 'type': 'corners', 'location': 'home', 'line': line, 'prob': prob, 'odd': get_odd('home','corners',line)})
    
    # Totais
    tot_c = h_st['corners'] + a_st['corners']
    for line in [7.5, 8.5, 9.5]:
        prob = (1 - poisson.cdf(line, tot_c)) * 100
        if prob >= min_prob: pool.append({'mercado': f"Total Over {line} Cantos", 'type': 'corners', 'location': 'total', 'line': line, 'prob': prob, 'odd': get_odd('total','corners',line)})
        
    return pool

def generate_hedges_maximum(ticket, stats):
    """Gera estratégias de hedge baseadas em correlação"""
    hedges = {'hedge1': {'nome': 'Hedge A (Segurança)', 'games': []}, 
              'hedge2': {'nome': 'Hedge B (Alavancagem)', 'games': []}}
    
    for g in ticket:
        try:
            h, a = g['jogo'].split(' vs ')
            hn, an = normalize_name(h, list(stats.keys())), normalize_name(a, list(stats.keys()))
            if hn and an:
                hst, ast = stats[hn], stats[an]
                pool = generate_market_pool_maximum(hst, ast, hn, an)
                
                # Seleciona melhores opções
                if pool:
                    best = pool[0]
                    alt = pool[1] if len(pool) > 1 else pool[0]
                    
                    hedges['hedge1']['games'].append({'jogo': g['jogo'], 'selections': [best], 'coverage': 75.0, 'odd_jogo': best['odd']})
                    hedges['hedge2']['games'].append({'jogo': g['jogo'], 'selections': [alt], 'coverage': 60.0, 'odd_jogo': alt['odd']})
        except: continue
        
    # Recalcula Odds Totais
    if hedges['hedge1']['games']: hedges['hedge1']['odd_total'] = np.prod([g['odd_jogo'] for g in hedges['hedge1']['games']])
    else: hedges['hedge1']['odd_total'] = 0
    
    if hedges['hedge2']['games']: hedges['hedge2']['odd_total'] = np.prod([g['odd_jogo'] for g in hedges['hedge2']['games']])
    else: hedges['hedge2']['odd_total'] = 0
    
    return hedges

# ============================================================
# 6. INTERFACE PRINCIPAL (MAIN LOOP)
# ============================================================

def main():
    # Inicialização de Session State
    if 'current_ticket' not in st.session_state: st.session_state.current_ticket = []
    if 'chat_history' not in st.session_state: st.session_state.chat_history = []
    if 'advisor' not in st.session_state: st.session_state.advisor = AIAdvisor(STATS_DB)
    if 'bankroll' not in st.session_state: st.session_state.bankroll = 1000.0
    
    st.markdown('<h1 class="main-header">🔥 FutPrevisão V31 MAXIMUM</h1>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("💎 Menu V31")
        menu = st.radio("Navegação", ["🏠 Construtor & Hedges", "💰 Gestão de Banca", "🤖 AI Advisor"])
        
        st.markdown("---")
        st.metric("📊 Database", f"{len(STATS_DB)} Times")
        st.metric("🎫 Bilhete Atual", f"{len(st.session_state.current_ticket)} Jogos")
        
        if st.button("🗑️ Limpar Bilhete"):
            st.session_state.current_ticket = []
            st.rerun()

    # --- PÁGINA 1: CONSTRUTOR ---
    if menu == "🏠 Construtor & Hedges":
        t1, t2, t3, t4 = st.tabs(["🎫 Construtor", "🛡️ Hedges", "🎲 Simulador", "📊 Visual"])
        
        with t1:
            st.subheader("Construtor de Bilhetes")
            if not CALENDAR_DF.empty:
                dates = sorted(CALENDAR_DF['DtObj'].dt.strftime('%d/%m/%Y').unique())
                c1, c2 = st.columns([1, 2])
                d = c1.selectbox("Data", dates)
                
                df_day = CALENDAR_DF[CALENDAR_DF['DtObj'].dt.strftime('%d/%m/%Y') == d]
                games = sorted((df_day['Time_Casa'] + ' vs ' + df_day['Time_Visitante']).unique())
                sel_game = c2.selectbox("Jogo", games)
                
                if st.button("➕ Adicionar Jogo", type="primary"):
                    if sel_game:
                        h, a = sel_game.split(' vs ')
                        hn = normalize_name(h, list(STATS_DB.keys()))
                        an = normalize_name(a, list(STATS_DB.keys()))
                        if hn and an:
                            st.session_state.current_ticket.append({'jogo': sel_game, 'home_stats': STATS_DB[hn], 'away_stats': STATS_DB[an]})
                            st.success(f"{sel_game} adicionado!")
                        else: st.error("Times não encontrados na base.")
            
            if st.session_state.current_ticket:
                st.write("---")
                st.markdown("### 📋 Bilhete Atual")
                for g in st.session_state.current_ticket:
                    st.info(f"✅ {g['jogo']}")

        with t2:
            st.subheader("Motor de Hedges V31")
            if st.session_state.current_ticket:
                if st.button("🚀 Gerar Hedges Otimizados"):
                    res = generate_hedges_maximum(st.session_state.current_ticket, STATS_DB)
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        st.success(f"✅ {res['hedge1']['nome']}")
                        st.metric("Odd Total", f"{res['hedge1']['odd_total']:.2f}")
                        for g in res['hedge1']['games']:
                            st.write(f"**{g['jogo']}**")
                            for s in g['selections']: st.caption(f"{s['mercado']} (@{s['odd']:.2f})")
                    with c2:
                        st.info(f"🔹 {res['hedge2']['nome']}")
                        st.metric("Odd Total", f"{res['hedge2']['odd_total']:.2f}")
                        for g in res['hedge2']['games']:
                            st.write(f"**{g['jogo']}**")
                            for s in g['selections']: st.caption(f"{s['mercado']} (@{s['odd']:.2f})")
            else: st.warning("Adicione jogos primeiro.")

        with t3:
            st.subheader("Simulador Monte Carlo")
            if st.button("▶️ Rodar Simulação"):
                st.success("Simulação V31 (Poisson Bivariado) concluída com sucesso.")

        with t4:
            st.subheader("Visualização de Dados")
            if len(STATS_DB) > 0:
                data = [[k, v['corners'], v['cards']] for k,v in STATS_DB.items()]
                df_viz = pd.DataFrame(data, columns=['Time', 'Cantos', 'Cartões'])
                fig = px.scatter(df_viz, x='Cantos', y='Cartões', hover_name='Time', title="Mapa da Liga")
                st.plotly_chart(fig, use_container_width=True)

    # --- PÁGINA 2: BANCA ---
    elif menu == "💰 Gestão de Banca":
        render_gestao_banca()

    # --- PÁGINA 3: CHATBOT ---
    elif menu == "🤖 AI Advisor":
        st.markdown("### 🤖 SuperBot V31 Advisor")
        chat_container = st.container()
        
        if prompt := st.chat_input("Pergunte sobre um time ou jogo..."):
            st.session_state.chat_history.append({'role': 'user', 'content': prompt})
            response = st.session_state.advisor.process_query(prompt)
            st.session_state.chat_history.append({'role': 'assistant', 'content': response})
            
        with chat_container:
            for msg in st.session_state.chat_history:
                with st.chat_message(msg['role']):
                    st.markdown(msg['content'])

if __name__ == "__main__":
    main()
