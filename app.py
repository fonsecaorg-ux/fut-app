import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import uuid
from datetime import datetime
from scipy.stats import poisson

# ==============================================================================
# 0. CONFIGURAÇÃO INICIAL
# ==============================================================================
st.set_page_config(page_title="FutPrevisão Pro", layout="wide", page_icon="⚽")

# Tentativa de importar Plotly
try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# ==============================================================================
# 1. SEGURANÇA E LOGIN
# ==============================================================================
USERS = {
    "diego": "@Casa612"
}

def check_login():
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    if st.session_state["logged_in"]:
        return True

    st.markdown("### 🔒 Acesso FutPrevisão Pro")
    
    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        username = st.text_input("Usuário")
        password = st.text_input("Senha", type="password")
        
        if st.button("Entrar"):
            if username in USERS and password == USERS[username]:
                st.session_state["logged_in"] = True
                st.rerun()
            else:
                st.error("❌ Usuário ou senha incorretos")
    
    return False

if not check_login():
    st.stop()

# ==============================================================================
# 2. CARREGAMENTO DE DADOS (CSV)
# ==============================================================================
BACKUP_TEAMS = {
    "Arsenal": {"corners": 6.82, "cards": 1.00, "fouls": 10.45, "goals_f": 2.3, "goals_a": 0.8},
    "Man City": {"corners": 7.45, "cards": 1.50, "fouls": 9.20, "goals_f": 2.7, "goals_a": 0.8},
}

def safe_float(value):
    try: return float(str(value).replace(',', '.'))
    except: return 0.0

@st.cache_data(ttl=3600)
def load_data():
    try:
        df = pd.read_csv("dados_times.csv")
        teams_dict = {}
        for _, row in df.iterrows():
            teams_dict[row['Time']] = {
                'corners': safe_float(row['Escanteios']),
                'cards': safe_float(row['CartoesAmarelos']), 
                'fouls': safe_float(row['Faltas']),
                'goals_f': safe_float(row['GolsFeitos']),
                'goals_a': safe_float(row['GolsSofridos'])
            }
    except:
        teams_dict = BACKUP_TEAMS
    
    return teams_dict

teams_data = load_data()
team_list_raw = sorted(list(teams_data.keys()))
team_list_with_empty = [""] + team_list_raw

# Lista de Mercados para o Registro
MERCADOS_LISTA = ["Selecione o mercado..."]
for i in [2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5]:
    MERCADOS_LISTA.append(f"Escanteios Mais de {i}")
for i in [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]:
    MERCADOS_LISTA.append(f"Cartões Mais de {i}")
MERCADOS_LISTA.extend(["Gols Mais de 0.5", "Gols Mais de 1.5", "Gols Mais de 2.5", "Ambas Marcam", "Vitória Casa", "Vitória Fora"])

# ==============================================================================
# 3. MOTOR DE CÁLCULO (POISSON / SIMULAÇÃO)
# ==============================================================================
def prob_over(exp, line):
    return poisson.sf(int(line), exp) * 100

def calcular_previsao(home, away, context_h, context_a):
    h_data = teams_data.get(home, BACKUP_TEAMS["Arsenal"])
    a_data = teams_data.get(away, BACKUP_TEAMS["Man City"])
    
    # Fatores de Contexto (1.0 = Neutro, >1 = Motivado, <1 = Crise)
    f_h = context_h
    f_a = context_a
    
    # 1. Escanteios (Modelo Contextual Simplificado)
    # Casa tem boost natural (1.10), Fora tem penalidade (0.85)
    corn_h = (h_data['corners'] * 1.10) * f_h
    corn_a = (a_data['corners'] * 0.85) * f_a
    total_corners = corn_h + corn_a
    
    # 2. Cartões (Modelo de Tensão)
    # Tensão baseada na média de faltas do jogo
    avg_fouls = (h_data['fouls'] + a_data['fouls']) / 2
    tension = avg_fouls / 12.0 # Normalizador
    card_h = h_data['cards'] * tension
    card_a = a_data['cards'] * tension
    total_cards = card_h + card_a
    
    # 3. Gols (Modelo Poisson Ofensivo/Defensivo)
    avg_league_goals = 1.3
    exp_goals_h = ((h_data['goals_f'] * f_h) / avg_league_goals) * (a_data['goals_a'] / avg_league_goals) * avg_league_goals
    exp_goals_a = ((a_data['goals_f'] * f_a) / avg_league_goals) * (h_data['goals_a'] / avg_league_goals) * avg_league_goals
    
    return {
        "corners": {"total": total_corners, "h": corn_h, "a": corn_a},
        "cards": {"total": total_cards, "h": card_h, "a": card_a},
        "goals": {"h": exp_goals_h, "a": exp_goals_a}
    }

def get_color(prob):
    if prob >= 70: return "green"
    if prob >= 50: return "orange"
    return "red"

# ==============================================================================
# 4. ESTILOS CSS
# ==============================================================================
st.markdown("""
<style>
    .bet-card-green { background-color: #d4edda; border-left: 5px solid #28a745; padding: 15px; border-radius: 5px; margin-bottom: 10px; }
    .bet-card-red { background-color: #f8d7da; border-left: 5px solid #dc3545; padding: 15px; border-radius: 5px; margin-bottom: 10px; }
    .bet-card-cashout { background-color: #fff3cd; border-left: 5px solid #ffc107; padding: 15px; border-radius: 5px; margin-bottom: 10px; }
    
    .pred-card { background: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); margin-bottom: 10px; border: 1px solid #e0e0e0; }
    .pred-header { font-weight: bold; font-size: 16px; margin-bottom: 10px; color: #333; }
    .prob-val { font-weight: bold; font-size: 18px; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 5. FUNÇÕES DE GESTÃO (JSON)
# ==============================================================================
DATA_FILE = "historico_bilhetes_v6.json"
CONFIG_FILE = "config_banca.json"

def carregar_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f: return json.load(f)
    return {"banca_inicial": 1000.0, "stop_loss": 50.0}

def salvar_config(cfg):
    with open(CONFIG_FILE, "w") as f: json.dump(cfg, f)

def carregar_tickets():
    if not os.path.exists(DATA_FILE): return []
    try:
        with open(DATA_FILE, "r") as f: dados = json.load(f)
        return sorted(dados, key=lambda x: datetime.strptime(x['Data'], "%d/%m/%Y"), reverse=True)
    except: return []

def salvar_ticket(ticket_data):
    if os.path.exists(DATA_FILE):
        try: with open(DATA_FILE, "r") as f: dados = json.load(f)
        except: dados = []
    else: dados = []
    if "id" not in ticket_data: ticket_data["id"] = str(uuid.uuid4())[:8].upper()
    dados.append(ticket_data)
    with open(DATA_FILE, "w") as f: json.dump(dados, f, indent=2)

def excluir_ticket(id_ticket):
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f: dados = json.load(f)
        novos_dados = [t for t in dados if t.get("id") != id_ticket]
        with open(DATA_FILE, "w") as f: json.dump(novos_dados, f, indent=2)
        return True
    return False

# ==============================================================================
# 6. RENDERIZAÇÃO DO DASHBOARD
# ==============================================================================
def render_dashboard():
    try:
        st.title("📊 Painel de Controle")
        
        # --- Config Sidebar ---
        cfg = carregar_config()
        with st.sidebar.expander("⚙️ Configurações", expanded=False):
            nova_banca = st.number_input("Banca Inicial", value=cfg["banca_inicial"], step=50.0)
            novo_stop = st.number_input("Stop Loss", value=cfg["stop_loss"], step=10.0)
            if st.button("Salvar Config"):
                salvar_config({"banca_inicial": nova_banca, "stop_loss": novo_stop})
                st.success("Salvo!")
                st.rerun()

        # --- KPIs ---
        tickets = carregar_tickets()
        lucro_total = sum(t["Lucro"] for t in tickets)
        banca_atual = cfg["banca_inicial"] + lucro_total
        
        hoje = datetime.now().strftime("%d/%m/%Y")
        prejuizo_hoje = sum(t["Lucro"] for t in tickets if t["Data"] == hoje and t["Lucro"] < 0)
        perda_atual = abs(prejuizo_hoje)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Banca Atual", f"R$ {banca_atual:,.2f}", f"{lucro_total:,.2f}")
        c2.metric("Lucro Total", f"R$ {lucro_total:,.2f}")
        c3.metric("Stop Loss Hoje", f"R$ {perda_atual:.2f} / {cfg['stop_loss']:.2f}")

        if perda_atual >= cfg['stop_loss']:
            st.error("🛑 STOP LOSS ATINGIDO! PARE DE APOSTAR HOJE.")

        st.divider()

        # --- ABAS (AQUI ESTÁ A NOVIDADE) ---
        tab_sim, tab_new, tab_hist, tab_stats = st.tabs(["🔮 Simulação IA", "➕ Novo Bilhete", "📜 Histórico", "📈 Gráficos"])

        # === ABA 0: SIMULAÇÃO / PREVISÃO (RESTAURADA) ===
        with tab_sim:
            st.markdown("### 🤖 Previsão de Partidas (Baseado no CSV)")
            
            c_h, c_a = st.columns(2)
            home_team = c_h.selectbox("Mandante", team_list_raw, index=0)
            away_team = c_a.selectbox("Visitante", team_list_raw, index=1 if len(team_list_raw) > 1 else 0)
            
            with st.expander("Ajustes de Contexto (Opcional)"):
                cc1, cc2 = st.columns(2)
                ctx_h = cc1.select_slider("Momento Mandante", options=[0.8, 1.0, 1.2], value=1.0, format_func=lambda x: "Crise" if x<1 else "Neutro" if x==1 else "Motivado")
                ctx_a = cc2.select_slider("Momento Visitante", options=[0.8, 1.0, 1.2], value=1.0, format_func=lambda x: "Crise" if x<1 else "Neutro" if x==1 else "Motivado")

            if st.button("🔮 Calcular Probabilidades", type="primary"):
                m = calcular_previsao(home_team, away_team, ctx_h, ctx_a)
                
                # --- RESULTADOS DA SIMULAÇÃO ---
                col_esc, col_gol, col_car = st.columns(3)
                
                # 1. Escanteios
                with col_esc:
                    st.info(f"🚩 **Escanteios** (Exp: {m['corners']['total']:.2f})")
                    for line in [8.5, 9.5, 10.5, 11.5]:
                        prob = prob_over(m['corners']['total'], line)
                        cor = get_color(prob)
                        st.markdown(f"**+{line}**: :{cor}[**{prob:.1f}%**]")
                    
                    st.markdown("---")
                    st.caption(f"🏠 {home_team}: {m['corners']['h']:.2f} | ✈️ {away_team}: {m['corners']['a']:.2f}")

                # 2. Gols
                with col_gol:
                    total_gols = m['goals']['h'] + m['goals']['a']
                    st.success(f"⚽ **Gols** (Exp: {total_gols:.2f})")
                    
                    # BTTS
                    btts_prob = (1 - poisson.pmf(0, m['goals']['h'])) * (1 - poisson.pmf(0, m['goals']['a'])) * 100
                    st.markdown(f"**Ambas Marcam**: :{get_color(btts_prob)}[**{btts_prob:.1f}%**]")
                    
                    for line in [1.5, 2.5, 3.5]:
                        prob = prob_over(total_gols, line)
                        cor = get_color(prob)
                        st.markdown(f"**+{line}**: :{cor}[**{prob:.1f}%**]")

                # 3. Cartões
                with col_car:
                    st.warning(f"🟨 **Cartões** (Exp: {m['cards']['total']:.2f})")
                    for line in [3.5, 4.5, 5.5]:
                        prob = prob_over(m['cards']['total'], line)
                        cor = get_color(prob)
                        st.markdown(f"**+{line}**: :{cor}[**{prob:.1f}%**]")
                    st.markdown("---")
                    st.caption("Baseado na intensidade de faltas")

        # === ABA 1: NOVO BILHETE (MANTIDA) ===
        with tab_new:
            st.markdown("### Registrar Aposta Real")
            
            if 'num_jogos' not in st.session_state: st.session_state['num_jogos'] = 1
            
            c_d, c_s, c_o = st.columns(3)
            data_t = c_d.date_input("Data", datetime.now())
            stake = c_s.number_input("Stake (R$)", min_value=0.0, value=10.0)
            odd = c_o.number_input("Odd Total", min_value=1.0, value=1.50)
            
            resultado = st.selectbox("Resultado", ["Green ✅", "Green (Cashout) 💰", "Red ❌", "Reembolso 🔄"])
            
            lucro = 0.0
            if "Green ✅" in resultado: lucro = (stake * odd) - stake
            elif "Red" in resultado: lucro = -stake
            elif "Cashout" in resultado:
                val_cash = st.number_input("Valor Cashout", min_value=0.0)
                lucro = val_cash - stake
            
            st.info(f"Lucro Previsto: R$ {lucro:.2f}")
            st.divider()

            jogos_data = []
            for j in range(st.session_state['num_jogos']):
                st.markdown(f"**Jogo {j+1}**")
                col_m, col_v = st.columns(2)
                m = col_m.selectbox(f"Mandante {j}", team_list_with_empty, key=f"m_{j}")
                v = col_v.selectbox(f"Visitante {j}", team_list_with_empty, key=f"v_{j}")
                sel_mercado = st.selectbox(f"Mercado {j}", MERCADOS_LISTA, key=f"merc_{j}")
                
                jogos_data.append({
                    "Nome": f"{m} x {v}",
                    "Selecoes": [{"Mercado": sel_mercado, "Status": "Registrado"}]
                })
                st.markdown("---")

            cb1, cb2 = st.columns(2)
            if cb1.button("➕ Adicionar Jogo"):
                st.session_state['num_jogos'] += 1
                st.rerun()
            if cb2.button("🔄 Resetar"):
                st.session_state['num_jogos'] = 1
                st.rerun()

            if st.button("💾 SALVAR BILHETE", type="primary", use_container_width=True):
                novo = {
                    "Data": data_t.strftime("%d/%m/%Y"),
                    "Resultado": resultado,
                    "Stake": stake,
                    "Odd": odd,
                    "Lucro": lucro,
                    "Jogos": jogos_data,
                    "id": str(uuid.uuid4())[:8]
                }
                salvar_ticket(novo)
                st.success("Bilhete Salvo!")
                st.rerun()

        # === ABA 2: HISTÓRICO (MANTIDA) ===
        with tab_hist:
            if not tickets:
                st.info("Nenhum bilhete.")
            else:
                for t in tickets:
                    css_class = "bet-card-cashout"
                    if "Green" in t["Resultado"] and "Cashout" not in t["Resultado"]: css_class = "bet-card-green"
                    if "Red" in t["Resultado"]: css_class = "bet-card-red"
                    
                    st.markdown(f"""
                    <div class="{css_class}">
                        <strong>{t['Data']}</strong> | Stake: R$ {t['Stake']} | Lucro: <strong>R$ {t['Lucro']:.2f}</strong><br>
                        Resultado: {t['Resultado']}
                    </div>
                    """, unsafe_allow_html=True)
                    with st.expander("Detalhes"):
                        for j in t['Jogos']: st.write(f"⚽ {j['Nome']} - {j['Selecoes'][0]['Mercado']}")
                        if st.button("Excluir", key=f"del_{t['id']}"):
                            excluir_ticket(t['id'])
                            st.rerun()

        # === ABA 3: GRÁFICOS (MANTIDA) ===
        with tab_stats:
            if HAS_PLOTLY and tickets:
                df = pd.DataFrame(tickets)
                try:
                    df['Data_Dt'] = pd.to_datetime(df['Data'], format="%d/%m/%Y")
                    df = df.sort_values('Data_Dt')
                    df['Acumulado'] = df['Lucro'].cumsum()
                    fig = px.line(df, x='Data_Dt', y='Acumulado', title="Evolução da Banca", markers=True)
                    st.plotly_chart(fig, use_container_width=True)
                except: st.error("Erro ao gerar gráfico.")
            else:
                st.info("Sem dados para gráficos.")

    except Exception as e:
        st.error(f"Erro no Dashboard: {str(e)}")

if __name__ == "__main__":
    render_dashboard()
