import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import poisson
import json
import hmac

# ==============================================================================
# 0. CONFIGURAÇÃO DA PÁGINA E LOGIN
# ==============================================================================
st.set_page_config(
    page_title="FutPrevisão Pro",
    layout="wide",
    page_icon="⚽"
)

def check_password():
    """Retorna True se o usuário logar corretamente."""
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    def password_entered():
        """Verifica se o usuário e senha batem com o cadastrado nos Secrets."""
        if "passwords" in st.secrets:
            user = st.session_state["username"]
            password = st.session_state["password"]
            
            # Validação segura
            if user in st.secrets["passwords"] and \
               hmac.compare_digest(password, st.secrets["passwords"][user]):
                st.session_state["password_correct"] = True
                del st.session_state["password"]
                del st.session_state["username"]
            else:
                st.session_state["password_correct"] = False
                st.error("😕 Usuário ou senha incorretos")
        else:
            st.error("Erro de configuração: Senhas não encontradas no servidor.")

    if st.session_state["password_correct"]:
        return True

    # Tela de Login
    st.markdown("### 🔒 Acesso Restrito - FutPrevisão Pro")
    st.text_input("Usuário", key="username")
    st.text_input("Senha", type="password", key="password")
    st.button("Entrar", on_click=password_entered)
    return False

if not check_password():
    st.stop()  # Para a execução se não estiver logado

# ==============================================================================
# 1. CARREGAMENTO DE DADOS (COM BACKUP DO ROBÔ)
# ==============================================================================
# Dados de Backup (Caso o CSV falhe)
BACKUP_TEAMS = {
    "Arsenal": {"corners": 6.82, "cards": 1.00, "fouls": 10.45, "goals_f": 2.3, "goals_a": 0.8},
    "Man City": {"corners": 7.45, "cards": 1.50, "fouls": 9.20, "goals_f": 2.7, "goals_a": 0.8},
    "Liverpool": {"corners": 6.18, "cards": 1.91, "fouls": 11.64, "goals_f": 2.5, "goals_a": 0.9},
    # ... (O sistema usa o CSV atualizado pelo robô, isso é só segurança)
}

def safe_float(value):
    try:
        return float(str(value).replace(',', '.'))
    except:
        return 0.0

@st.cache_data(ttl=3600) # Cache de 1 hora
def load_data():
    # 1. Carrega Times
    try:
        df = pd.read_csv("dados_times.csv") # Nome corrigido para o do robô
        teams_dict = {}
        for _, row in df.iterrows():
            teams_dict[row['Time']] = {
                'corners': safe_float(row['Escanteios']),
                'cards': safe_float(row['CartoesAmarelos']), 
                'fouls': safe_float(row['Faltas']),
                'goals_f': safe_float(row['GolsFeitos']),
                'goals_a': safe_float(row['GolsSofridos'])
            }
    except Exception as e:
        st.warning(f"Usando base de backup (Erro no CSV: {e})")
        teams_dict = BACKUP_TEAMS

    # 2. Carrega Árbitros
    try:
        df_ref = pd.read_csv("arbitros.csv")
        referees = dict(zip(df_ref['Nome'], df_ref['Fator']))
    except:
        referees = {"Genérico (Médio)": 1.0, "Rigoroso": 1.2, "Leniente": 0.8}
        
    return teams_dict, referees

# Carrega os dados
teams_data, referees_data = load_data()

# ==============================================================================
# 2. BARRA LATERAL (COM STATUS DO ROBÔ)
# ==============================================================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/53/53283.png", width=80)
st.sidebar.title("FutPrevisão Pro v2.6")
st.sidebar.markdown("---")

# --- NOVO: STATUS DO ROBÔ ---
def carregar_metadados():
    try:
        with open("metadados.json", "r", encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

meta = carregar_metadados()

if meta:
    st.sidebar.caption("🤖 **Status do Robô:**")
    st.sidebar.text(f"Verificado em:\n{meta['ultima_verificacao']}")
    
    if meta['times_alterados'] > 0:
        st.sidebar.success(f"⚡ Atualizado! {meta['times_alterados']} times mudaram.")
    else:
        st.sidebar.info("✔ Base verificada e estável.")
        
    st.sidebar.caption(f"📡 Fontes: {meta.get('fontes', 'Adamchoi & FBref')}")
else:
    st.sidebar.warning("⚠ Aguardando execução do robô...")
# -----------------------------

st.sidebar.markdown("---")
st.sidebar.header("Configuração do Jogo")

# Seletores
team_list = sorted(list(teams_data.keys()))
home_team = st.sidebar.selectbox("Mandante (Casa)", team_list, index=0)
away_team = st.sidebar.selectbox("Visitante (Fora)", team_list, index=1)

referee_list = sorted(list(referees_data.keys())) + ["Outro"]
referee_name = st.sidebar.selectbox("Árbitro", referee_list)

if referee_name == "Outro":
    referee_factor = st.sidebar.slider("Fator Rigor Manual", 0.8, 1.4, 1.0, 0.1)
else:
    referee_factor = referees_data[referee_name]
    st.sidebar.metric("Rigor do Árbitro", referee_factor)

champions_mode = st.sidebar.checkbox("Modo Jogo Estudado (Champions)", value=False)

# ==============================================================================
# 3. LÓGICA MATEMÁTICA (INTOCADA)
# ==============================================================================
def calculate_metrics(home, away, ref_factor, is_champions):
    h_data = teams_data[home]
    a_data = teams_data[away]
    
    # --- ESCANTEIOS ---
    # Mandante +10%, Visitante -15%, Champions -15% geral
    corn_h = h_data['corners'] * 1.10
    corn_a = a_data['corners'] * 0.85
    
    total_corners = corn_h + corn_a
    if is_champions:
        total_corners *= 0.85
        
    # --- CARTÕES (TENSION FACTOR) ---
    # Tensão = Média de faltas dos dois times / 24
    tension = (h_data['fouls'] + a_data['fouls']) / 24.0
    # Cap de segurança (0.85 a 1.30)
    tension = max(0.85, min(tension, 1.30))
    
    base_cards = h_data['cards'] + a_data['cards']
    total_cards = base_cards * tension * ref_factor
    
    # --- GOLS (POISSON) ---
    # Ataque Casa x Defesa Fora
    # Assumindo média da liga ~1.3 gols
    league_avg = 1.3
    
    att_h = h_data['goals_f'] / league_avg
    def_a = a_data['goals_a'] / league_avg
    exp_h = att_h * def_a * league_avg
    
    att_a = a_data['goals_f'] / league_avg
    def_h = a_data['goals_a'] / league_avg # Defesa casa
    exp_a = att_a * def_h * league_avg
    
    return {
        'corners': round(total_corners, 2),
        'cards': round(total_cards, 2),
        'goals_home': exp_h,
        'goals_away': exp_a,
        'tension': tension
    }

# ==============================================================================
# 4. INTERFACE PRINCIPAL
# ==============================================================================
st.title(f"📊 {home_team} x {away_team}")
st.markdown("### Análise Probabilística de Futebol")

if st.sidebar.button("Calcular Previsão 🚀", type="primary"):
    
    # Cálculos
    metrics = calculate_metrics(home_team, away_team, referee_factor, champions_mode)
    
    # POISSON (Probabilidades de Gols)
    p_h = [poisson.pmf(i, metrics['goals_home']) for i in range(6)]
    p_a = [poisson.pmf(i, metrics['goals_away']) for i in range(6)]
    
    prob_home_win = sum([p_h[i]*p_a[j] for i in range(6) for j in range(6) if i > j])
    prob_draw = sum([p_h[i]*p_a[j] for i in range(6) for j in range(6) if i == j])
    prob_away_win = sum([p_h[i]*p_a[j] for i in range(6) for j in range(6) if i < j])
    
    prob_btts = 1 - (poisson.pmf(0, metrics['goals_home']) + poisson.pmf(0, metrics['goals_away']) - poisson.pmf(0, metrics['goals_home'])*poisson.pmf(0, metrics['goals_away']))

    # --- EXIBIÇÃO ---
    
    # 1. Placar Resumo
    c1, c2, c3 = st.columns(3)
    c1.metric("Escanteios Esperados", f"{metrics['corners']}")
    c2.metric("Cartões Esperados", f"{metrics['cards']}")
    c3.metric("Tensão do Jogo", f"{metrics['tension']:.2f}", delta_color="inverse")
    
    st.divider()
    
    # 2. Abas de Detalhes
    tab1, tab2 = st.tabs(["🎯 Mercados Principais", "📈 Probabilidades Exatas"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🚩 Escanteios")
            st.info(f"Linha Sugerida: **Over {int(metrics['corners'])-0.5}**")
            st.progress(min(metrics['corners']/15, 1.0))
            
        with col2:
            st.subheader("🟨 Cartões")
            st.warning(f"Linha Sugerida: **Over {int(metrics['cards'])-0.5}**")
            st.progress(min(metrics['cards']/8, 1.0))
            
        st.subheader("⚽ Gols (Match Odds)")
        cols = st.columns(3)
        cols[0].metric(f"Vitória {home_team}", f"{prob_home_win*100:.1f}%")
        cols[1].metric("Empate", f"{prob_draw*100:.1f}%")
        cols[2].metric(f"Vitória {away_team}", f"{prob_away_win*100:.1f}%")
        
        st.success(f"Probabilidade BTTS (Ambas Marcam): **{prob_btts*100:.1f}%**")

    with tab2:
        st.write("Matriz de Probabilidades (Poisson):")
        
        data_matrix = [[p_h[i]*p_a[j]*100 for j in range(5)] for i in range(5)]
        
        fig = go.Figure(data=go.Heatmap(
            z=data_matrix,
            x=[f"{away_team} {j}" for j in range(5)],
            y=[f"{home_team} {i}" for i in range(5)],
            colorscale='Viridis',
            texttemplate="%{z:.1f}%"
        ))
        fig.update_layout(title="Mapa de Calor do Placar Exato")
        st.plotly_chart(fig, use_container_width=True)

    # Rodapé
    st.caption("Sistema desenvolvido para fins acadêmicos. Não recomendado para apostas reais.")
