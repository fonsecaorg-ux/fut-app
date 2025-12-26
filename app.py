import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import poisson, norm, beta
import json
import hmac
import os
from datetime import datetime, timedelta
import uuid
import math
import re
from difflib import get_close_matches
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# 0. CONFIGURAÇÃO E CSS
# ==============================================================================
st.set_page_config(
    page_title="FutPrevisão V31 Pro",
    layout="wide",
    page_icon="⚽",
    initial_sidebar_state="expanded"
)

# CSS PROFISSIONAL UNIFICADO
st.markdown("""
<style>
    /* Estilo Geral */
    .stApp { background: linear-gradient(to bottom, #f8f9fa 0%, #ffffff 100%); }
    
    /* Chatbot Styles */
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border-radius: 15px !important;
        padding: 15px !important;
        margin: 10px 0 !important;
    }
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) p {
        color: white !important;
    }
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) {
        background: #2d3748 !important;
        border-radius: 15px !important;
        padding: 15px !important;
        margin: 10px 0 !important;
    }
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) p {
        color: white !important;
    }
    
    /* Gestão de Banca Styles */
    .ticket-header-win { background-color: #d4edda; padding: 10px; border-radius: 5px; border-left: 5px solid #28a745; }
    .ticket-header-loss { background-color: #f8d7da; padding: 10px; border-radius: 5px; border-left: 5px solid #dc3545; }
    .ticket-header-cashout { background-color: #fff3cd; padding: 10px; border-radius: 5px; border-left: 5px solid #ffc107; }
    .metric-card { background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. AUTENTICAÇÃO (LOGIN)
# ==============================================================================
def check_password():
    """Retorna True se o usuário tiver a senha correta."""
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    def password_entered():
        if "passwords" in st.secrets:
            user = st.session_state["username"]
            password = st.session_state["password"]
            
            if user in st.secrets["passwords"] and \
               hmac.compare_digest(password, st.secrets["passwords"][user]):
                st.session_state["password_correct"] = True
                del st.session_state["password"]
                del st.session_state["username"]
            else:
                st.session_state["password_correct"] = False
                st.error("😕 Usuário ou senha incorretos")
        else:
            # Se não houver secrets configurado, libera acesso (modo dev)
            st.session_state["password_correct"] = True

    if st.session_state["password_correct"]:
        return True

    st.markdown("### 🔒 Acesso Restrito - FutPrevisão V31")
    st.text_input("Usuário", key="username")
    st.text_input("Senha", type="password", key="password")
    st.button("Entrar", on_click=password_entered)
    return False

if not check_password():
    st.stop()

# ==============================================================================
# 2. CARREGAMENTO DE DADOS (V31 ENGINE - GITHUB FIX)
# ==============================================================================

# IMPORTANTE: Configure aqui o seu usuário e repositório do GitHub se os arquivos não carregarem localmente
GITHUB_USER = "seu_usuario_github" # Ex: "joaosilva"
GITHUB_REPO = "seu_repo_futprevisao" # Ex: "fut-app"
GITHUB_BRANCH = "main" # Geralmente é 'main' ou 'master'
GITHUB_BASE_URL = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{GITHUB_BRANCH}/"

# Definição de Ligas e Arquivos
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

NAME_MAPPING = {
    'Man United': 'Manchester Utd', 'Manchester United': 'Manchester Utd', 'Man Utd': 'Manchester Utd',
    'Man City': 'Manchester City', 'Spurs': 'Tottenham', 'Newcastle': 'Newcastle Utd',
    'Wolves': 'Wolverhampton', 'Brighton': 'Brighton and Hove Albion',
    'Nottm Forest': "Nott'm Forest", 'Leicester': 'Leicester City',
    'West Ham': 'West Ham Utd', 'Sheffield Utd': 'Sheffield United',
    'Inter': 'Inter Milan', 'AC Milan': 'Milan',
    'Ath Madrid': 'Atletico Madrid', 'Ath Bilbao': 'Athletic Club',
    'Betis': 'Real Betis', 'Sociedad': 'Real Sociedad'
}

def normalize_name(name: str, known_teams: List[str]) -> Optional[str]:
    """Normaliza nomes com fuzzy matching"""
    if not name: return None
    name = name.strip()
    if name in NAME_MAPPING:
        name = NAME_MAPPING[name]
    if name in known_teams:
        return name
    matches = get_close_matches(name, known_teams, n=1, cutoff=0.6)
    return matches[0] if matches else None

@st.cache_data(ttl=3600)
def load_all_data_v31():
    """
    Carrega dados com estratégia Híbrida: Local -> GitHub Raw.
    """
    stats_db = {}
    
    # 1. Identificar Diretório de Dados Local (Compatível com GitHub Clone)
    possible_paths = ['.', './data', 'data', '/mount/src/fut-app', os.getcwd()]
    
    def try_read_csv(filename):
        # Tentativa 1: Local
        for p in possible_paths:
            fpath = os.path.join(p, filename)
            if os.path.exists(fpath):
                try: return pd.read_csv(fpath, encoding='utf-8')
                except: 
                    try: return pd.read_csv(fpath, encoding='latin1')
                    except: pass
        
        # Tentativa 2: GitHub Raw (Fallback)
        try:
            url = f"{GITHUB_BASE_URL}{filename}"
            return pd.read_csv(url, encoding='utf-8')
        except:
            return pd.DataFrame() # Falhou

    # 2. Carregar Stats
    for league_name, filenames in LEAGUE_FILES.items():
        df = pd.DataFrame()
        for fname in filenames:
            df = try_read_csv(fname)
            if not df.empty: break
        
        if df.empty: continue
        
        # Padronizar colunas
        cols_map = {'HomeTeam': 'HomeTeam', 'AwayTeam': 'AwayTeam', 
                   'FTHG': 'FTHG', 'FTAG': 'FTAG', 
                   'HC': 'HC', 'AC': 'AC', 'HY': 'HY', 'AY': 'AY',
                   'HST': 'HST', 'AST': 'AST', 'HF': 'HF', 'AF': 'AF'}
        
        teams = set(df['HomeTeam'].dropna().unique()) | set(df['AwayTeam'].dropna().unique())
        
        for team in teams:
            h_games = df[df['HomeTeam'] == team]
            a_games = df[df['AwayTeam'] == team]
            
            # Cálculo de Médias (com fallback seguro)
            def get_mean(df_h, df_a, col_h, col_a, default):
                val_h = df_h[col_h].mean() if col_h in df_h.columns and not df_h.empty else default
                val_a = df_a[col_a].mean() if col_a in df_a.columns and not df_a.empty else default
                return (val_h + val_a) / 2
                
            corners = get_mean(h_games, a_games, 'HC', 'AC', 5.0)
            cards = get_mean(h_games, a_games, 'HY', 'AY', 2.0)
            fouls = get_mean(h_games, a_games, 'HF', 'AF', 11.0)
            goals_f = get_mean(h_games, a_games, 'FTHG', 'FTAG', 1.3)
            goals_a = get_mean(h_games, a_games, 'FTAG', 'FTHG', 1.3)
            shots = get_mean(h_games, a_games, 'HST', 'AST', 4.5)
            
            stats_db[team] = {
                'corners': corners,
                'cards': cards,
                'fouls': fouls,
                'goals_f': goals_f,
                'goals_a': goals_a,
                'shots_on_target': shots,
                'league': league_name,
                'games': len(h_games) + len(a_games)
            }

    # 3. Carregar Calendário
    cal = try_read_csv('calendario_ligas.csv')
    if not cal.empty and 'Data' in cal.columns:
        cal['DtObj'] = pd.to_datetime(cal['Data'], format='%d/%m/%Y', errors='coerce')
        
    # 4. Carregar Árbitros
    refs = {}
    ref_df = try_read_csv('arbitros_5_ligas_2025_2026.csv')
    if not ref_df.empty:
        for _, row in ref_df.iterrows():
            refs[row['Arbitro']] = {
                'factor': row.get('Media_Cartoes_Por_Jogo', 4.0) / 4.0,
                'avg_cards': row.get('Media_Cartoes_Por_Jogo', 4.0),
                'red_rate': 0.1 # Default
            }
        
    return stats_db, cal, refs

# Carregamento Global
STATS_DB, CALENDAR_DF, REFEREES_DB = load_all_data_v31()

# ==============================================================================
# 3. MOTOR MATEMÁTICO V31 (CAUSALITY ENGINE)
# ==============================================================================
def calcular_jogo_v31(home_stats: Dict, away_stats: Dict, ref_data: Dict) -> Dict:
    """Motor V31: Calcula projeções baseadas em stats + contexto"""
    
    # 1. Fatores de Pressão (Baseado em Chutes no Alvo)
    shots_h = home_stats.get('shots_on_target', 4.5)
    pressure_h = 1.15 if shots_h > 5.5 else 1.0
    
    # 2. Cantos (Ajuste por Pressão)
    corners_h = home_stats['corners'] * 1.15 * pressure_h
    corners_a = away_stats['corners'] * 0.90
    corners_total = corners_h + corners_a
    
    # 3. Violência e Árbitro
    fouls_h = home_stats.get('fouls', 11.0)
    fouls_a = away_stats.get('fouls', 11.0)
    violence_factor = 1.10 if (fouls_h + fouls_a) > 24 else 0.95
    
    ref_factor = ref_data.get('factor', 1.0) if ref_data else 1.0
    
    cards_h = home_stats['cards'] * violence_factor * ref_factor
    cards_a = away_stats['cards'] * violence_factor * ref_factor
    cards_total = cards_h + cards_a
    
    # 4. Gols (xG Simples)
    xg_h = (home_stats['goals_f'] * away_stats['goals_a']) / 1.3
    xg_a = (away_stats['goals_f'] * home_stats['goals_a']) / 1.3
    
    return {
        'corners': {'h': corners_h, 'a': corners_a, 't': corners_total},
        'cards': {'h': cards_h, 'a': cards_a, 't': cards_total},
        'goals': {'h': xg_h, 'a': xg_a},
        'meta': {'pressure': pressure_h, 'violence': violence_factor}
    }

# ==============================================================================
# 4. SUPERBOT V2.0 (INTELIGÊNCIA CONVERSACIONAL)
# ==============================================================================
class SuperIntentDetector:
    def __init__(self):
        self.patterns = {
            'stats_time': ['como está', 'como esta', 'estatística', 'dados do', 'média de', 'fale sobre'],
            'jogos_hoje': ['jogos hoje', 'partidas hoje', 'hoje', 'agenda'],
            'analise_jogo': ['vs', ' x ', 'analisa', 'analise', 'contra'],
            'ranking_cantos': ['mais cantos', 'top cantos', 'ranking cantos'],
            'saudacao': ['oi', 'olá', 'ola', 'bom dia', 'boa noite']
        }

    def detect(self, text: str) -> str:
        text_lower = text.lower()
        if ' vs ' in text_lower or ' x ' in text_lower:
            return 'analise_jogo'
        for intent, patterns in self.patterns.items():
            if any(p in text_lower for p in patterns):
                return intent
        return 'stats_time' # Default fallback inteligente

class SuperKnowledgeBase:
    def __init__(self, stats, cal, refs):
        self.stats = stats
        self.cal = cal
        self.refs = refs

    def get_team_stats(self, team):
        norm = normalize_name(team, list(self.stats.keys()))
        return self.stats.get(norm) if norm else None

    def get_games_date(self, date_str):
        if self.cal.empty: return []
        try:
            mask = self.cal['DtObj'].dt.strftime('%d/%m/%Y') == date_str
            return self.cal[mask].to_dict('records')
        except: return []

class SuperResponder:
    def __init__(self, kb: SuperKnowledgeBase):
        self.kb = kb

    def team_stats(self, team_name):
        s = self.kb.get_team_stats(team_name)
        if not s: return f"❌ Time '{team_name}' não encontrado na base."
        
        return f"""
        📊 **Estatísticas: {team_name}** ({s['league']})
        
        🚩 **Cantos:** {s['corners']:.1f} / jogo
        🟨 **Cartões:** {s['cards']:.1f} / jogo
        ⚽ **Gols:** {s['goals_f']:.1f} Pró | {s['goals_a']:.1f} Contra
        🥊 **Faltas:** {s['fouls']:.1f} / jogo
        
        *Baseado em {s['games']} jogos analisados.*
        """

    def analise_jogo(self, t1, t2):
        s1 = self.kb.get_team_stats(t1)
        s2 = self.kb.get_team_stats(t2)
        if not s1 or not s2: return "❌ Um dos times não foi encontrado."
        
        calc = calcular_jogo_v31(s1, s2, {})
        
        return f"""
        ⚔️ **ANÁLISE V31: {t1} vs {t2}**
        
        🚩 **Escanteios Esperados:** {calc['corners']['t']:.1f}
           • {t1}: {calc['corners']['h']:.1f}
           • {t2}: {calc['corners']['a']:.1f}
           
        🟨 **Cartões Esperados:** {calc['cards']['t']:.1f}
           • Tensão do Jogo: {"🔥 Alta" if calc['meta']['violence'] > 1 else "Normal"}
           
        ⚽ **xG (Gols Esperados):**
           • {t1}: {calc['goals']['h']:.2f}
           • {t2}: {calc['goals']['a']:.2f}
        """

# ==============================================================================
# 5. GESTÃO DE BANCA (MÓDULO FINANCEIRO)
# ==============================================================================
DATA_FILE = "historico_bilhetes_v5.json"
CONFIG_FILE = "config_banca.json"

def carregar_tickets():
    if not os.path.exists(DATA_FILE): return []
    try:
        with open(DATA_FILE, "r") as f: return json.load(f)
    except: return []

def salvar_ticket(ticket):
    data = carregar_tickets()
    if 'id' not in ticket: ticket['id'] = str(uuid.uuid4())[:8]
    data.insert(0, ticket) # Adiciona no início
    with open(DATA_FILE, "w") as f: json.dump(data, f)

def render_gestao_banca():
    st.header("📊 Gestão Profissional de Banca")
    
    # Config
    if not os.path.exists(CONFIG_FILE):
        config = {"banca_inicial": 1000.0}
    else:
        with open(CONFIG_FILE, "r") as f: config = json.load(f)
        
    tickets = carregar_tickets()
    lucro_total = sum(t['Lucro'] for t in tickets)
    banca_atual = config['banca_inicial'] + lucro_total
    
    # KPIs
    k1, k2, k3 = st.columns(3)
    k1.metric("🏦 Banca Atual", f"R$ {banca_atual:.2f}", f"{lucro_total:.2f}")
    k2.metric("📝 Bilhetes", len(tickets))
    win_rate = len([t for t in tickets if "Green" in t['Resultado']]) / len(tickets) * 100 if tickets else 0
    k3.metric("🎯 Win Rate", f"{win_rate:.1f}%")
    
    st.divider()
    
    t1, t2 = st.tabs(["➕ Novo Bilhete", "📜 Histórico"])
    
    with t1:
        with st.form("new_ticket"):
            c1, c2 = st.columns(2)
            stake = c1.number_input("Stake (R$)", value=10.0, step=5.0)
            odd = c2.number_input("Odd Total", value=2.00, step=0.01)
            res = st.selectbox("Resultado", ["Green ✅", "Red ❌", "Reembolso 🔄", "Green (Cashout)"])
            
            desc = st.text_area("Descrição / Jogos", placeholder="Ex: Flamengo x Vasco - Over 2.5")
            
            if st.form_submit_button("💾 Registrar"):
                lucro = (stake * odd - stake) if "Green ✅" in res else (-stake if "Red" in res else 0)
                ticket = {
                    "Data": datetime.now().strftime("%d/%m/%Y"),
                    "Descricao": desc,
                    "Stake": stake,
                    "Odd": odd,
                    "Resultado": res,
                    "Lucro": lucro
                }
                salvar_ticket(ticket)
                st.success("Bilhete Salvo!")
                st.rerun()
                
    with t2:
        for t in tickets:
            color = "#d4edda" if t['Lucro'] > 0 else ("#f8d7da" if t['Lucro'] < 0 else "#fff3cd")
            with st.container():
                st.markdown(f"""
                <div style="background-color: {color}; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                    <b>{t['Data']}</b> | {t['Descricao']}<br>
                    Stake: R${t['Stake']} @ {t['Odd']} | <b>Retorno: R$ {t['Lucro']:+.2f}</b>
                </div>
                """, unsafe_allow_html=True)

# ==============================================================================
# 6. APP PRINCIPAL (MAIN LOOP)
# ==============================================================================
def main():
    # Inicialização do Bot
    if 'kb' not in st.session_state:
        st.session_state.kb = SuperKnowledgeBase(STATS_DB, CALENDAR_DF, REFEREES_DB)
        st.session_state.bot = SuperResponder(st.session_state.kb)
        st.session_state.intent = SuperIntentDetector()
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = [{
            'role': 'assistant', 
            'content': f"🤖 **Olá! Sou o SuperBot V31.**\nTenho dados de {len(STATS_DB)} times.\n\nExperimente:\n- 'Como está o Arsenal?'\n- 'Liverpool vs City'\n- 'Jogos de hoje'"
        }]

    # Menu Lateral
    st.sidebar.title("💎 FutPrevisão V31")
    page = st.sidebar.radio("Navegação", ["🤖 SuperBot IA", "📊 Gestão de Banca"])
    
    st.sidebar.divider()
    st.sidebar.info(f"📚 Database: {len(STATS_DB)} times")
    
    if page == "🤖 SuperBot IA":
        st.title("🤖 SuperBot V31 - Analista Virtual")
        
        # Render Chat
        for msg in st.session_state.chat_history:
            st.chat_message(msg['role']).markdown(msg['content'])
            
        # Input
        if prompt := st.chat_input("Pergunte sobre times, jogos ou stats..."):
            st.session_state.chat_history.append({'role': 'user', 'content': prompt})
            st.chat_message("user").markdown(prompt)
            
            # Processamento
            intent = st.session_state.intent.detect(prompt)
            resp = "Desculpe, não entendi."
            
            try:
                if intent == 'stats_time':
                    # Extração simples de entidade
                    words = prompt.split()
                    team = None
                    for w in words:
                        if len(w) > 3:
                            team = normalize_name(w, list(STATS_DB.keys()))
                            if team: break
                    if not team: # Tenta frase inteira
                        team = normalize_name(prompt, list(STATS_DB.keys()))
                        
                    if team: resp = st.session_state.bot.team_stats(team)
                    else: resp = "❌ Não identifiquei o time. Tente digitar o nome corretamente."
                    
                elif intent == 'analise_jogo':
                    if ' vs ' in prompt: parts = prompt.split(' vs ')
                    elif ' x ' in prompt: parts = prompt.split(' x ')
                    else: parts = []
                    
                    if len(parts) == 2:
                        t1 = normalize_name(parts[0], list(STATS_DB.keys()))
                        t2 = normalize_name(parts[1], list(STATS_DB.keys()))
                        if t1 and t2: resp = st.session_state.bot.analise_jogo(t1, t2)
                        else: resp = "❌ Nomes de times ambíguos ou não encontrados."
                    else:
                        resp = "❌ Use o formato 'Time A vs Time B'."
                        
                elif intent == 'jogos_hoje':
                    if CALENDAR_DF.empty:
                        resp = "📅 Calendário não carregado."
                    else:
                        # Pega hoje (simulado ou real)
                        today = datetime.now().strftime('%d/%m/%Y')
                        games = st.session_state.kb.get_games_date(today)
                        if games:
                            resp = f"📅 **Jogos de Hoje ({today}):**\n\n"
                            for g in games:
                                resp += f"• {g['Time_Casa']} x {g['Time_Visitante']} ({g.get('Hora', '??:??')})\n"
                        else:
                            resp = f"📅 Nenhum jogo encontrado para hoje ({today})."
            
            except Exception as e:
                resp = f"⚠️ Erro ao processar: {str(e)}"
                
            st.session_state.chat_history.append({'role': 'assistant', 'content': resp})
            st.chat_message("assistant").markdown(resp)
            
    elif page == "📊 Gestão de Banca":
        render_gestao_banca()

if __name__ == "__main__":
    main()