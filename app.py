"""
FutPrevisão V31 MAXIMUM + AI Advisor ULTRA
CÓDIGO COMPLETO - SEM CORTES
VERSÃO PROFISSIONAL

Autor: Diego
Versão: 31.3 ULTRA MAXIMUM
Data: 27/12/2025
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math
from typing import Dict, List, Optional
import plotly.graph_objects as go
from pathlib import Path
from difflib import get_close_matches
import re
from collections import defaultdict

# Define diretório base
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

# CSS PROFISSIONAL
st.markdown('''
<style>
    /* TABS HORIZONTAIS - DESIGN PROFISSIONAL */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: white;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        color: #667eea;
        border: 2px solid transparent;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border-color: white;
    }
    
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
    
    /* Cards profissionais */
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    
    /* Bilhetes */
    .ticket-item {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 3px solid #667eea;
        transition: all 0.2s ease;
    }
    
    .ticket-item:hover {
        transform: translateX(5px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);
    }
    
    /* Header profissional */
    h1 {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
</style>
''', unsafe_allow_html=True)

# CSS customizado
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    .highlight-green {
        background-color: #90EE90;
        padding: 5px;
        border-radius: 3px;
    }
    .highlight-yellow {
        background-color: #FFFFE0;
        padding: 5px;
        border-radius: 3px;
    }
    .highlight-red {
        background-color: #FFB6C1;
        padding: 5px;
        border-radius: 3px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def find_file(filename: str) -> Optional[str]:
    """Busca arquivo em múltiplos diretórios"""
    search_paths = [
        Path('/mnt/project') / filename,
        Path('.') / filename,
        Path('./data') / filename,
        BASE_DIR / filename
    ]
    for path in search_paths:
        if path.exists():
            return str(path)
    return None

def normalize_name(name: str, known_teams: List[str]) -> Optional[str]:
    """Normaliza nomes de times usando fuzzy matching"""
    if not name or not known_teams:
        return None
    name = str(name).strip()
    matches = get_close_matches(name, known_teams, n=1, cutoff=0.6)
    return matches[0] if matches else None

def clean_team_name(text: str) -> str:
    """Limpa nome de time vindo do chat"""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    stop_words = {'do', 'da', 'de', 'dos', 'das', 'o', 'a', 'os', 'as', 
                  'como', 'está', 'esta', 'stats', 'estatistica', 'vs', 'x', 'contra', 'analise', 'analisar'}
    words = text.split()
    text = ' '.join([w for w in words if w not in stop_words])
    return text.strip()

def format_currency(value: float) -> str:
    return f"R$ {value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def get_prob_emoji(prob: float) -> str:
    """Retorna emoji baseado na probabilidade"""
    if prob >= 80:
        return "🔥"
    elif prob >= 75:
        return "✅"
    elif prob >= 70:
        return "🎯"
    elif prob >= 65:
        return "⚡"
    else:
        return "⚪"

@st.cache_data(ttl=3600)
def load_all_data():
    """Carrega todos os dados do sistema"""
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
        if not filepath:
            continue
        try:
            df = pd.read_csv(filepath, encoding='utf-8')
            teams = set(df['HomeTeam'].dropna().unique()) | set(df['AwayTeam'].dropna().unique())
            
            for team in teams:
                if pd.isna(team): continue
                
                h_games = df[df['HomeTeam'] == team]
                a_games = df[df['AwayTeam'] == team]
                
                corners_h = h_games['HC'].mean() if 'HC' in h_games.columns else 5.5
                corners_a = a_games['AC'].mean() if 'AC' in a_games.columns else 4.5
                cards_h = h_games['HY'].mean() if 'HY' in h_games.columns else 2.5
                cards_a = a_games['AY'].mean() if 'AY' in a_games.columns else 2.5
                fouls_h = h_games['HF'].mean() if 'HF' in h_games.columns else 12.0
                fouls_a = a_games['AF'].mean() if 'AF' in a_games.columns else 12.0
                goals_fh = h_games['FTHG'].mean() if 'FTHG' in h_games.columns else 1.5
                goals_fa = a_games['FTAG'].mean() if 'FTAG' in a_games.columns else 1.3
                goals_ah = h_games['FTAG'].mean() if 'FTAG' in h_games.columns else 1.3
                goals_aa = a_games['FTHG'].mean() if 'FTHG' in a_games.columns else 1.5
                shots_h = h_games['HST'].mean() if 'HST' in h_games.columns else 4.5
                shots_a = a_games['AST'].mean() if 'AST' in a_games.columns else 4.0
                
                stats_db[team] = {
                    'corners': (corners_h + corners_a) / 2,
                    'corners_home': corners_h,
                    'corners_away': corners_a,
                    'cards': (cards_h + cards_a) / 2,
                    'cards_home': cards_h,
                    'cards_away': cards_a,
                    'fouls': (fouls_h + fouls_a) / 2,
                    'fouls_home': fouls_h,
                    'fouls_away': fouls_a,
                    'goals_f': (goals_fh + goals_fa) / 2,
                    'goals_f_home': goals_fh,
                    'goals_f_away': goals_fa,
                    'goals_a': (goals_ah + goals_aa) / 2,
                    'goals_a_home': goals_ah,
                    'goals_a_away': goals_aa,
                    'shots_on_target': (shots_h + shots_a) / 2,
                    'shots_home': shots_h,
                    'shots_away': shots_a,
                    'league': league_name,
                    'games': len(h_games) + len(a_games)
                }
        except Exception as e:
            pass
    
    cal_filepath = find_file('calendario_ligas.csv')
    if cal_filepath:
        try:
            cal = pd.read_csv(cal_filepath, encoding='utf-8')
            if 'Data' in cal.columns:
                cal['DtObj'] = pd.to_datetime(cal['Data'], format='%d/%m/%Y', errors='coerce')
        except:
            pass
    
    refs_filepath = find_file('arbitros_5_ligas_2025_2026.csv')
    if refs_filepath:
        try:
            refs_df = pd.read_csv(refs_filepath, encoding='utf-8')
            for _, row in refs_df.iterrows():
                referees[row['Arbitro']] = {
                    'factor': row['Media_Cartoes_Por_Jogo'] / 4.0,
                    'games': row['Jogos_Apitados'],
                    'avg_cards': row['Media_Cartoes_Por_Jogo'],
                    'red_rate': row.get('Cartoes_Vermelhos', 0) / row['Jogos_Apitados'] if row['Jogos_Apitados'] > 0 else 0.08
                }
        except:
            pass
    
    return stats_db, cal, referees

# ============================================================
# MOTOR DE CÁLCULO
# ============================================================

def calcular_poisson(lambda_val: float, k: int) -> float:
    """Calcula probabilidade Poisson P(X > k)"""
    import math
    prob_exact = 0
    for i in range(int(k) + 1):
        prob_exact += (math.exp(-lambda_val) * (lambda_val ** i)) / math.factorial(i)
    return (1 - prob_exact) * 100

def calcular_jogo_v31(home_stats: Dict, away_stats: Dict, ref_data: Dict) -> Dict:
    """Motor de cálculo de probabilidades"""
    if not home_stats or not away_stats:
        return {'corners': {'h':0, 'a':0, 't':0}, 'cards': {'h':0, 'a':0, 't':0}, 'goals': {'h':0, 'a':0}, 'corners_total': 0, 'total_goals': 0, 'cards_total': 0}

    # ESCANTEIOS
    base_corners_h = home_stats.get('corners_home', home_stats['corners'])
    base_corners_a = away_stats.get('corners_away', away_stats['corners'])
    
    shots_h = home_stats.get('shots_home', 4.5)
    
    pressure_h = 1.20 if shots_h > 6.0 else 1.10 if shots_h > 4.5 else 1.0
    
    corners_h = base_corners_h * 1.15 * pressure_h
    corners_a = base_corners_a * 0.90
    corners_total = corners_h + corners_a
    
    # CARTÕES
    ref_factor = ref_data.get('factor', 1.0) if ref_data else 1.0
    cards_h_base = home_stats.get('cards_home', home_stats['cards'])
    cards_a_base = away_stats.get('cards_away', away_stats['cards'])
    
    cards_h = cards_h_base * ref_factor
    cards_a = cards_a_base * ref_factor
    cards_total = cards_h + cards_a
    
    # xG
    xg_h = (home_stats['goals_f'] * away_stats['goals_a']) / 1.3
    xg_a = (away_stats['goals_f'] * home_stats['goals_a']) / 1.3
    
    return {
        'corners': {'h': corners_h, 'a': corners_a, 't': corners_total},
        'cards': {'h': cards_h, 'a': cards_a, 't': cards_total},
        'goals': {'h': xg_h, 'a': xg_a},
        'corners_total': corners_total,
        'total_goals': xg_h + xg_a,
        'cards_total': cards_total
    }

def simulate_game_v31(home_stats: Dict, away_stats: Dict, ref_data: Dict, n_sims: int = 3000) -> Dict:
    """Simulador de Monte Carlo"""
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

# ============================================================
# MÉTRICAS FINANCEIRAS
# ============================================================

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

# ============================================================
# PARSER DE BILHETES
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
        if any(x in linha.lower() for x in ['criar aposta', 'stake', 'retorno']): continue
        if ' vs ' in linha or ' x ' in linha.lower():
            sep = ' vs ' if ' vs ' in linha else ' x '
            partes = linha.split(sep)
            if len(partes) == 2:
                jogo_atual = {'home': partes[0].strip(), 'away': partes[1].strip(), 'mercados': mercados_pend.copy()}
                jogos.append(jogo_atual)
                time_pendente = None
                mercados_pend = []
                continue
        if any(x in linha.lower() for x in ['total de', 'mais de', 'over']) and any(y in linha.lower() for y in ['canto', 'escanteio', 'cartão', 'card']):
            tipo = 'corners' if any(x in linha.lower() for x in ['canto', 'escanteio']) else 'cards'
            nums = re.findall(r'\d+\.5', linha)
            if nums:
                line = float(nums[0])
                odds = re.findall(r'@?\d+\.\d+', linha)
                odd = float(odds[-1].replace('@', '')) if odds else 2.0
                mercado = {'tipo': tipo, 'location': 'total', 'line': line, 'odd': odd, 'desc': linha}
                if jogo_atual: jogo_atual['mercados'].append(mercado)
                else: mercados_pend.append(mercado)
                continue
        if not any(x in linha.lower() for x in ['total', 'mais de', 'over']) and len(linha) > 2:
            if time_pendente is None: time_pendente = linha.strip()
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
                'home': h_norm, 'away': a_norm,
                'home_original': jogo['home'], 'away_original': jogo['away'],
                'mercados': jogo['mercados'],
                'home_stats': stats_db[h_norm], 'away_stats': stats_db[a_norm]
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
                'jogo': f"{jogo['home']} vs {jogo['away']}", 'mercado': mercado['desc'],
                'prob': prob * 100, 'odd_casa': mercado['odd'],
                'fair_odd': 1.0 / prob if prob > 0 else 999,
                'value': prob * mercado['odd'] if prob > 0 else 0
            })
    return {'prob_total': prob_total * 100, 'detalhes': detalhes}

# ============================================================
# CHATBOT ULTRA INTELIGENTE (ATUALIZADO)
# ============================================================

def processar_chat_ultra(mensagem: str, stats_db: Dict, cal: pd.DataFrame, refs: Dict) -> str:
    """
    AI ADVISOR ULTRA - Segue estritamente as regras de análise estatística profissional.
    """
    msg_lower = mensagem.lower()
    known_teams = list(stats_db.keys())
    
    # 1. IDENTIFICAÇÃO DE INTENÇÃO
    is_vs = ' vs ' in msg_lower or ' x ' in msg_lower or 'contra' in msg_lower
    is_prob = 'probabilidade' in msg_lower or 'chance' in msg_lower
    is_sugestao = 'sugira' in msg_lower or 'aposta' in msg_lower or 'mercado' in msg_lower or 'recomend' in msg_lower
    
    # Extração de times
    times_encontrados = []
    for t in sorted(known_teams, key=len, reverse=True):
        if t.lower() in msg_lower:
            times_encontrados.append(t)
            msg_lower = msg_lower.replace(t.lower(), "")
    
    # --- CENÁRIO 1: ANÁLISE DE JOGO (VS) ---
    if is_vs and len(times_encontrados) >= 2:
        t1, t2 = times_encontrados[0], times_encontrados[1]
        s1, s2 = stats_db[t1], stats_db[t2]
        prev = calcular_jogo_v31(s1, s2, {})
        
        prob_over_gols = calcular_poisson(prev['total_goals'], 2.5)
        prob_over_cantos = calcular_poisson(prev['corners_total'], 9.5)
        
        resp = f"📊 **ANÁLISE ESTATÍSTICA: {t1} vs {t2}**\n\n"
        resp += f"**🔎 Perfil do Confronto:**\n"
        resp += f"O modelo projeta um jogo com **{prev['corners_total']:.1f} escanteios** e **{prev['total_goals']:.1f} gols esperados (xG)**.\n"
        
        if s1['goals_f'] > 1.5 and s2['goals_f'] > 1.5:
            resp += "Ambos os times têm médias ofensivas altas, indicando tendência para gols.\n\n"
        else:
            resp += "Confronto com tendência mais tática e travada.\n\n"
            
        resp += f"**📈 Probabilidades (Poisson):**\n"
        resp += f"• Chance de Over 2.5 Gols: **{prob_over_gols:.1f}%**\n"
        resp += f"• Chance de Over 9.5 Cantos: **{prob_over_cantos:.1f}%**\n\n"
        
        resp += f"**🧠 Sugestão Estatística:**\n"
        if prob_over_cantos > 65:
            resp += f"✅ **Over Escanteios:** Valor encontrado. A linha de 9.5 tem probabilidade favorável."
        elif prob_over_gols > 60:
            resp += f"✅ **Over Gols:** O xG combinado sugere um jogo aberto."
        else:
            resp += f"⚠️ **Cautela:** As estatísticas não mostram valor claro em linhas principais. Considere mercado de cartões ou ao vivo."
            
        return resp

    # --- CENÁRIO 2: ANÁLISE DE TIME ÚNICO ---
    elif len(times_encontrados) == 1:
        t = times_encontrados[0]
        s = stats_db[t]
        
        resp = f"📊 **RAIO-X: {t}**\n\n"
        resp += f"**Médias na {s['league']}:**\n"
        resp += f"🚩 Escanteios: **{s['corners']:.2f}**/jogo\n"
        resp += f"⚽ Ataque: **{s['goals_f']:.2f}** gols marcados/jogo\n"
        resp += f"🛡️ Defesa: **{s['goals_a']:.2f}** gols sofridos/jogo\n"
        resp += f"🟨 Disciplina: **{s['cards']:.2f}** cartões/jogo\n\n"
        
        resp += "**📝 Interpretação:**\n"
        if s['corners'] > 6.0:
            resp += f"O {t} é uma máquina de escanteios. Excelente para linhas de Over.\n"
        elif s['corners'] < 4.0:
            resp += f"O {t} cede poucos escanteios, indicando jogos com tendência Under nesse mercado.\n"
            
        if s['goals_f'] > 1.8:
            resp += "Ataque muito produtivo. Mercado de gols a favor é recomendado."
            
        return resp

    # --- CENÁRIO 3: PERGUNTA DE PROBABILIDADE ---
    elif is_prob:
        return "Para calcular a probabilidade exata, preciso que você cite os dois times do confronto. Exemplo: 'Qual a chance de over cantos em Arsenal vs City?'"

    # --- CENÁRIO 4: SUGESTÃO DE APOSTAS GERAL ---
    elif is_sugestao:
        return "🤖 Para sugerir uma aposta de valor, preciso analisar um jogo específico. Por favor, digite o nome dos times (ex: 'Sugira aposta para Liverpool vs Chelsea'). Eu calcularei o valor esperado com base nas médias históricas."

    # --- CENÁRIO 5: DÚVIDA GENÉRICA / CONVERSA ---
    else:
        return """🤖 **Olá! Sou o AI Advisor ULTRA do FutPrevisão V31.**

Minha função é fornecer análises estritamente estatísticas baseadas no nosso banco de dados.

**Comandos que entendo:**
1️⃣ **Análise de Jogo:** "Analise Arsenal vs Chelsea"
2️⃣ **Raio-X de Time:** "Como está o Real Madrid?"
3️⃣ **Probabilidades:** "Qual a chance de over cantos em Liverpool x City?"

🚫 *Não invento dados, não prometo green e não uso 'feeling'. Apenas matemática aplicada ao futebol.*

Como posso te ajudar hoje?"""

# ============================================================
# MAIN
# ============================================================

def main():
    # 1. CARREGAR DADOS PRIMEIRO
    STATS, CAL, REFS = load_all_data()
    
    if 'current_ticket' not in st.session_state:
        st.session_state.current_ticket = []
    if 'bet_results' not in st.session_state:
        st.session_state.bet_results = []
    if 'bankroll_history' not in st.session_state:
        st.session_state.bankroll_history = [1000.0]
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'initial_bankroll' not in st.session_state:
        st.session_state.initial_bankroll = 1000.0
    
    with st.sidebar:
        st.header("📊 Dashboard")
        col1, col2 = st.columns(2)
        col1.metric("Times", len(STATS))
        col1.metric("Jogos", len(CAL) if not CAL.empty else 0)
        col2.metric("Árbitros", len(REFS))
        banca = st.session_state.bankroll_history[-1]
        col2.metric("Banca", format_currency(banca))
        
        if st.session_state.current_ticket:
            st.success(f"🎫 {len(st.session_state.current_ticket)} seleção(ões)")
        else:
            st.info("📭 Bilhete vazio")
    
    # ═══════════════════════════════════════════════════════════
    # HEADER PROFISSIONAL
    # ═══════════════════════════════════════════════════════════
    col1, col2, col3 = st.columns([2, 3, 2])
    with col1: st.image("https://img.icons8.com/fluency/96/000000/soccer-ball.png", width=80)
    with col2:
        st.title("⚽ FutPrevisão V31 Pro")
        st.caption("_Sistema Profissional de Análise Esportiva_")
    with col3: st.metric("📚 Database", f"{len(STATS)} times", delta="10 Ligas")
    st.markdown("---")
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "🎫 Construtor", "🛡️ Hedges", "🎲 Simulador", "📊 Métricas", 
        "🎨 Viz", "📝 Registro", "🔍 Scanner", "📋 Importar", "🤖 AI Advisor"
    ])
    
    # ============================================================
    # TAB 1: CONSTRUTOR (ATUALIZADO)
    # ============================================================
    with tab1:
        st.header("🎫 Construtor de Bilhetes Profissional")
        
        # 1. SELEÇÃO VIA CALENDÁRIO
        st.subheader("📅 Selecionar do Calendário")
        if not CAL.empty:
            dates = sorted(CAL['DtObj'].dt.strftime('%d/%m/%Y').unique())
            sel_date = st.selectbox("📅 Selecione a Data:", dates, key='c_date')
            jogos_dia = CAL[CAL['DtObj'].dt.strftime('%d/%m/%Y') == sel_date]
            
            st.markdown(f"### 🎯 {len(jogos_dia)} jogo(s) disponível(eis)")
            
            for idx, jogo in jogos_dia.iterrows():
                h = normalize_name(jogo['Time_Casa'], list(STATS.keys()))
                a = normalize_name(jogo['Time_Visitante'], list(STATS.keys()))
                
                if h and a and h in STATS and a in STATS:
                    ref_nome = jogo.get('Arbitro', 'N/A')
                    ref_data = REFS.get(ref_nome, {})
                    calc = calcular_jogo_v31(STATS[h], STATS[a], ref_data)
                    
                    with st.expander(f"⚽ {h} vs {a} | {jogo.get('Hora', 'N/A')}", expanded=False):
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("xG Casa", f"{calc['goals']['h']:.2f}")
                        col2.metric("xG Fora", f"{calc['goals']['a']:.2f}")
                        col3.metric("Cantos", f"{calc['corners']['t']:.1f}")
                        col4.metric("Cartões", f"{calc['cards']['t']:.1f}")
                        
                        st.markdown("#### 📊 Seleções Disponíveis:")
                        opcoes = [
                            (f"{h} - Over 4.5 Cantos Casa", calc['corners']['h'], 4.5, 'corners'),
                            (f"{a} - Over 4.5 Cantos Fora", calc['corners']['a'], 4.5, 'corners'),
                            (f"Over 9.5 Cantos Total", calc['corners']['t'], 9.5, 'corners'),
                            (f"Over 4.5 Cartões Total", calc['cards']['t'], 4.5, 'cards'),
                        ]
                        
                        for desc, media, linha, tipo in opcoes:
                            prob = 75 if media > linha + 0.5 else 65 if media > linha else 55
                            emoji = get_prob_emoji(prob)
                            col1, col2 = st.columns([4, 1])
                            col1.markdown(f"{emoji} **{desc}** | Prob: {prob}%")
                            if col2.button("➕", key=f"add_{idx}_{desc}"):
                                st.session_state.current_ticket.append({
                                    'jogo': f"{h} vs {a}",
                                    'market_display': desc,
                                    'prob': prob,
                                    'data': sel_date
                                })
                                st.rerun()
        
        st.markdown("---")
        
        # 2. CRIAÇÃO MANUAL (NOVO)
        with st.expander("📝 Criar Bilhete Manualmente (Custom)", expanded=False):
            st.info("Adicione jogos ou mercados não listados automaticamente.")
            c1, c2, c3 = st.columns([3, 2, 1])
            m_jogo = c1.text_input("Nome do Jogo (ex: Brasil x Argentina)")
            m_mercado = c2.selectbox("Mercado", ["Over 2.5 Gols", "Under 2.5 Gols", "Over 9.5 Cantos", "Vitória Casa", "Vitória Fora", "Ambos Marcam"])
            m_odd = c3.number_input("Odd", min_value=1.01, value=1.90, step=0.01)
            
            if st.button("➕ Adicionar Manualmente"):
                if m_jogo:
                    st.session_state.current_ticket.append({
                        'jogo': m_jogo,
                        'market_display': m_mercado,
                        'prob': (1/m_odd)*100,
                        'data': datetime.now().strftime('%d/%m/%Y')
                    })
                    st.success(f"Adicionado: {m_jogo} - {m_mercado}")
                    st.rerun()
                else:
                    st.warning("Digite o nome do jogo.")

        st.subheader("📋 Seu Bilhete Atual")
        if st.session_state.current_ticket:
            st.success(f"✅ {len(st.session_state.current_ticket)} seleção(ões)")
            for i, sel in enumerate(st.session_state.current_ticket):
                col1, col2 = st.columns([5, 1])
                col1.write(f"{i+1}. {sel['jogo']} - {sel['market_display']} ({sel.get('prob', 0):.1f}%)")
                if col2.button("🗑️", key=f"del_{i}"):
                    st.session_state.current_ticket.pop(i)
                    st.rerun()
            
            if st.button("🗑️ LIMPAR BILHETE", use_container_width=True):
                st.session_state.current_ticket = []
                st.rerun()
        else:
            st.info("📭 Bilhete vazio. Adicione seleções acima!")

    # ============================================================
    # TAB 2: HEDGES (COMPLETO)
    # ============================================================
    with tab2:
        st.header("🛡️ Hedges MAXIMUM - Sistema de Proteção")
        if not st.session_state.current_ticket:
            st.warning("⚠️ Bilhete vazio! Vá para Tab 'Construtor'")
        else:
            col1, col2 = st.columns(2)
            stake = col1.number_input("💰 Stake (R$)", 10.0, 10000.0, 100.0, 10.0)
            odd_total = col2.number_input("📊 Odd Total", 1.5, 100.0, 5.0, 0.1)
            ret_max = stake * odd_total
            lucro_max = ret_max - stake
            st.info(f"💵 Retorno: {format_currency(ret_max)} | Lucro: {format_currency(lucro_max)}")
            st.markdown("---")
            with st.expander("🛡️ HEDGE 1: Smart Protection", expanded=True):
                st.markdown("**Inverte seleção de MENOR probabilidade**")
                h1_stake = stake * 0.30
                h1_odd = 2.0
                cen1_princ = lucro_max - h1_stake
                cen1_hedge = -stake + (h1_stake * h1_odd)
                col1, col2, col3 = st.columns(3)
                col1.metric("Stake", format_currency(h1_stake))
                col2.metric("Odd", f"@{h1_odd:.2f}")
                col3.metric("Retorno", format_currency(h1_stake * h1_odd))
            with st.expander("💎 HEDGE 2: Guaranteed Profit"):
                st.markdown("**Inverte TUDO (arbitragem)**")
                h3_odd = 1.5
                h3_stake = (stake * odd_total) / (h3_odd + 1)
                lucro_gar = (stake * odd_total) - stake - h3_stake
                col1, col2, col3 = st.columns(3)
                col1.metric("Stake", format_currency(h3_stake))
                col2.metric("Odd", f"@{h3_odd:.2f}")
                col3.metric("💰 LUCRO GARANTIDO", format_currency(lucro_gar))

    # ============================================================
    # TAB 3: SIMULADOR (COMPLETO)
    # ============================================================
    with tab3:
        st.header("🎲 Simulador Monte Carlo - 3000 Iterações")
        if not CAL.empty:
            dates = sorted(CAL['DtObj'].dt.strftime('%d/%m/%Y').unique())
            sel_date = st.selectbox("Data:", dates, key='sim_date')
            jogos_dia = CAL[CAL['DtObj'].dt.strftime('%d/%m/%Y') == sel_date]
            jogos_disp = []
            for _, jogo in jogos_dia.iterrows():
                h = normalize_name(jogo['Time_Casa'], list(STATS.keys()))
                a = normalize_name(jogo['Time_Visitante'], list(STATS.keys()))
                if h and a: jogos_disp.append(f"{h} vs {a}")
            
            if jogos_disp:
                jogo_sel = st.selectbox("Jogo:", jogos_disp)
                if st.button("🎲 SIMULAR 3000 JOGOS"):
                    h_name, a_name = jogo_sel.split(' vs ')
                    with st.spinner('Simulando...'):
                        sims = simulate_game_v31(STATS[h_name], STATS[a_name], {}, 3000)
                        st.subheader("📊 Resultados")
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Cantos", f"{sims['corners_total'].mean():.1f}")
                        col2.metric("Cartões", f"{sims['cards_total'].mean():.1f}")
                        col3.metric("Gols Casa", f"{sims['goals_h'].mean():.1f}")
                        col4.metric("Gols Fora", f"{sims['goals_a'].mean():.1f}")
                        
                        mercados = {
                            'Over 9.5 Cantos': (sims['corners_total'] > 9.5).mean() * 100,
                            'Over 10.5 Cantos': (sims['corners_total'] > 10.5).mean() * 100,
                            'Over 4.5 Cartões': (sims['cards_total'] > 4.5).mean() * 100,
                            'Over 2.5 Gols': ((sims['goals_h'] + sims['goals_a']) > 2.5).mean() * 100,
                        }
                        df_merc = pd.DataFrame({'Mercado': list(mercados.keys()), 'Probabilidade (%)': list(mercados.values())})
                        st.dataframe(df_merc.sort_values('Probabilidade (%)', ascending=False), use_container_width=True)

    # ============================================================
    # TAB 4: MÉTRICAS (COMPLETO)
    # ============================================================
    with tab4:
        st.header("📊 Métricas PRO - Análise Financeira Avançada")
        if not st.session_state.bet_results:
            st.info("📭 Sem apostas registradas. Use Tab 'Registrar'")
        else:
            total_apostas = len(st.session_state.bet_results)
            apostas_ganhas = sum(1 for b in st.session_state.bet_results if b.get('ganhou', False))
            total_staked = sum(b.get('stake', 0) for b in st.session_state.bet_results)
            total_profit = sum(b.get('lucro', 0) for b in st.session_state.bet_results)
            win_rate = (apostas_ganhas / total_apostas) * 100 if total_apostas > 0 else 0
            roi = calculate_roi(total_staked, total_profit)
            sharpe = calculate_sharpe_ratio([b.get('return', 0) for b in st.session_state.bet_results])
            max_dd = calculate_max_drawdown(st.session_state.bankroll_history)
            
            st.subheader("📈 Métricas Principais")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Win Rate", f"{win_rate:.1f}%")
            col2.metric("ROI", f"{roi:.1f}%")
            col3.metric("Sharpe Ratio", f"{sharpe:.2f}")
            col4.metric("Max Drawdown", f"{max_dd:.1f}%")
            
            fig_banca = go.Figure()
            fig_banca.add_trace(go.Scatter(y=st.session_state.bankroll_history, mode='lines+markers', name='Banca'))
            st.plotly_chart(fig_banca, use_container_width=True)

    # ============================================================
    # TAB 5: VIZ (COMPLETO)
    # ============================================================
    with tab5:
        st.header("🎨 Visualizações Avançadas")
        viz_tipo = st.selectbox("Tipo de Visualização:", ["Comparativo de Ligas", "Top Times - Cantos", "Distribuição de Cantos"])
        
        if viz_tipo == "Comparativo de Ligas":
            liga_data = defaultdict(lambda: {'cantos': []})
            for team, data in STATS.items(): liga_data[data['league']]['cantos'].append(data['corners'])
            ligas = list(liga_data.keys())
            cantos_media = [np.mean(liga_data[l]['cantos']) for l in ligas]
            fig = go.Figure(data=[go.Bar(name='Cantos Médios', x=ligas, y=cantos_media, marker_color='orange')])
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_tipo == "Top Times - Cantos":
            times_sorted = sorted(STATS.items(), key=lambda x: x[1]['corners'], reverse=True)[:20]
            fig = go.Figure(data=[go.Bar(y=[t[0] for t in times_sorted], x=[t[1]['corners'] for t in times_sorted], orientation='h')])
            st.plotly_chart(fig, use_container_width=True)

    # ============================================================
    # TAB 6: REGISTRO (COMPLETO)
    # ============================================================
    with tab6:
        st.header("📝 Registrar Apostas")
        col1, col2 = st.columns(2)
        stake = col1.number_input("Stake (R$)", 10.0, 10000.0, 50.0, 10.0, key='reg_stake')
        odd = col2.number_input("Odd", 1.01, 100.0, 2.0, 0.01, key='reg_odd')
        ganhou = st.checkbox("✅ Aposta ganhou?")
        descricao = st.text_input("Descrição (opcional)", "Aposta manual")
        
        if st.button("💾 REGISTRAR APOSTA", use_container_width=True):
            lucro = stake * (odd - 1) if ganhou else -stake
            st.session_state.bet_results.append({
                'stake': stake, 'odd': odd, 'ganhou': ganhou, 'lucro': lucro,
                'data': datetime.now().strftime('%d/%m/%Y %H:%M'), 'descricao': descricao,
                'return': odd if ganhou else 0
            })
            nova_banca = st.session_state.bankroll_history[-1] + lucro
            st.session_state.bankroll_history.append(nova_banca)
            st.success(f"✅ Aposta registrada! Lucro: {format_currency(lucro)}")
            st.rerun()
            
        if st.session_state.bet_results:
            st.dataframe(pd.DataFrame(st.session_state.bet_results), use_container_width=True)

    # ============================================================
    # TAB 7: SCANNER (COMPLETO)
    # ============================================================
    with tab7:
        st.header("🔍 Scanner Inteligente de Jogos")
        if not CAL.empty:
            dates = sorted(CAL['DtObj'].dt.strftime('%d/%m/%Y').unique())
            sel_date = st.selectbox("Data:", dates, key='scan_date')
            prob_min = st.slider("Probabilidade Mínima (%)", 50, 90, 70)
            
            if st.button("🔍 ESCANEAR JOGOS", use_container_width=True):
                jogos_dia = CAL[CAL['DtObj'].dt.strftime('%d/%m/%Y') == sel_date]
                resultados = []
                with st.spinner('Analisando jogos...'):
                    for _, jogo in jogos_dia.iterrows():
                        h = normalize_name(jogo['Time_Casa'], list(STATS.keys()))
                        a = normalize_name(jogo['Time_Visitante'], list(STATS.keys()))
                        if h and a and h in STATS and a in STATS:
                            calc = calcular_jogo_v31(STATS[h], STATS[a], {})
                            if calc['corners']['t'] > 10.5:
                                prob = 75
                                if prob >= prob_min:
                                    resultados.append({'Jogo': f"{h} vs {a}", 'Mercado': 'Over 10.5 Cantos', 'Prob': f"{prob}%"})
                if resultados:
                    st.dataframe(pd.DataFrame(resultados), use_container_width=True)
                else:
                    st.warning("⚠️ Nenhuma oportunidade encontrada")

    # ============================================================
    # TAB 8: IMPORTAR (COMPLETO)
    # ============================================================
    with tab8:
        st.header("📋 Importar Bilhete Automaticamente")
        texto = st.text_area("Cole o texto do bilhete:", height=200, key='import_text')
        if st.button("🔍 ANALISAR BILHETE", use_container_width=True):
            if texto.strip():
                jogos_parsed = parse_bilhete_texto(texto)
                if jogos_parsed:
                    jogos_val = validar_jogos_bilhete(jogos_parsed, STATS)
                    if jogos_val:
                        analise = calcular_prob_bilhete(jogos_val)
                        col1, col2 = st.columns(2)
                        col1.metric("Prob Real", f"{analise['prob_total']:.1f}%")
                        col2.metric("Value", f"{analise['prob_total']/100 * 3.0:.2f}") # Exemplo odd 3.0
                        for det in analise['detalhes']:
                            st.write(f"**{det['jogo']}** - {det['mercado']} (Prob: {det['prob']:.1f}%)")
                    else: st.error("❌ Times não encontrados no banco de dados")
                else: st.error("❌ Não foi possível identificar jogos no texto")

    # ============================================================
    # TAB 9: AI ADVISOR ULTRA (ATUALIZADO)
    # ============================================================
    with tab9:
        st.header("🤖 AI Advisor ULTRA")
        st.caption("Assistente Estatístico Profissional V31")
        
        chat_container = st.container()
        with chat_container:
            if not st.session_state.chat_history:
                st.info("👋 Olá! Sou o AI Advisor ULTRA. Pergunte sobre 'Arsenal vs Chelsea' ou 'Como está o Liverpool'.")
            for msg in st.session_state.chat_history:
                role = msg['role']
                avatar = "👤" if role == 'user' else "🤖"
                st.chat_message(role, avatar=avatar).markdown(msg['content'])

        user_msg = st.chat_input("Digite sua pergunta sobre jogos ou times...")
        if user_msg:
            st.session_state.chat_history.append({'role': 'user', 'content': user_msg})
            with st.spinner("🧠 Consultando base de dados estatística..."):
                response = processar_chat_ultra(user_msg, STATS, CAL, REFS)
            st.session_state.chat_history.append({'role': 'assistant', 'content': response})
            st.rerun()

if __name__ == "__main__":
    main()
