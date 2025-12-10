import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import uuid
from datetime import datetime

# ==============================================================================
# 0. CONFIGURAÇÃO INICIAL (DEVE SER A PRIMEIRA LINHA)
# ==============================================================================
st.set_page_config(page_title="FutPrevisão Pro", layout="wide", page_icon="⚽")

# Tentativa de importar Plotly (com aviso caso falhe)
try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    st.error("⚠️ Biblioteca 'plotly' não encontrada. O Dashboard ficará sem gráficos.")

# ==============================================================================
# 1. SEGURANÇA E LOGIN
# ==============================================================================
# Credenciais
USERS = {
    "diego": "@Casa612"
}

def check_login():
    """Sistema de Login Simplificado e Robusto"""
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

# SE NÃO ESTIVER LOGADO, PARA AQUI.
if not check_login():
    st.stop()

# ==============================================================================
# 2. CARREGAMENTO DE DADOS
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
    # 1. Tenta carregar dados_times.csv
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
    except Exception as e:
        # Se falhar, usa backup silencioso para não travar
        teams_dict = BACKUP_TEAMS
    
    # 2. Tenta carregar arbitros.csv
    try:
        df_ref = pd.read_csv("arbitros.csv")
        referees = dict(zip(df_ref['Nome'], df_ref['Fator']))
    except:
        referees = {}
        
    referees['Estilo: Normal (Padrão)'] = 1.00
    referees['Estilo: Rigoroso'] = 1.25
    referees['Estilo: Conservador'] = 0.80
        
    return teams_dict, referees

# Carrega os dados com mensagem de status
with st.spinner("Carregando base de dados..."):
    teams_data, referees_data = load_data()
    team_list_raw = sorted(list(teams_data.keys()))
    team_list_with_empty = [""] + team_list_raw

# Configura lista de mercados
MERCADOS_LISTA = ["Selecione o mercado..."]
for i in [2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5]:
    MERCADOS_LISTA.append(f"Escanteios Mais de {i}")
for i in [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]:
    MERCADOS_LISTA.append(f"Cartões Mais de {i}")
MERCADOS_LISTA.extend(["Gols Mais de 0.5", "Gols Mais de 1.5", "Gols Mais de 2.5", "Ambas Marcam", "Vitória Casa", "Vitória Fora"])

# ==============================================================================
# 3. ESTILOS CSS
# ==============================================================================
st.markdown("""
<style>
    .bet-card-green { background-color: #d4edda; border-left: 5px solid #28a745; padding: 15px; border-radius: 5px; margin-bottom: 10px; }
    .bet-card-red { background-color: #f8d7da; border-left: 5px solid #dc3545; padding: 15px; border-radius: 5px; margin-bottom: 10px; }
    .bet-card-cashout { background-color: #fff3cd; border-left: 5px solid #ffc107; padding: 15px; border-radius: 5px; margin-bottom: 10px; }
    .kpi-box { background: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 4. FUNÇÕES DE GESTÃO (ARQUIVOS JSON)
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
    except:
        return []

def salvar_ticket(ticket_data):
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r") as f: dados = json.load(f)
        except: dados = []
    else: dados = []
    
    if "id" not in ticket_data:
        ticket_data["id"] = str(uuid.uuid4())[:8].upper()
        
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
# 5. RENDERIZAÇÃO DO DASHBOARD (BLOCO PRINCIPAL)
# ==============================================================================
def render_dashboard():
    try:
        st.title("📊 Gestão de Banca")
        
        # --- Sidebar Config ---
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
        
        # Stop Loss Check
        hoje = datetime.now().strftime("%d/%m/%Y")
        prejuizo_hoje = sum(t["Lucro"] for t in tickets if t["Data"] == hoje and t["Lucro"] < 0)
        perda_atual = abs(prejuizo_hoje)
        
        # Top KPIs
        c1, c2, c3 = st.columns(3)
        c1.metric("Banca Atual", f"R$ {banca_atual:,.2f}", f"{lucro_total:,.2f}")
        c2.metric("Lucro Total", f"R$ {lucro_total:,.2f}")
        c3.metric("Stop Loss Hoje", f"R$ {perda_atual:.2f} / {cfg['stop_loss']:.2f}")

        if perda_atual >= cfg['stop_loss']:
            st.error("🛑 STOP LOSS ATINGIDO! PARE DE APOSTAR HOJE.")

        st.divider()

        # --- ABAS ---
        tab_new, tab_hist, tab_stats = st.tabs(["➕ Novo Bilhete", "📜 Histórico", "📈 Gráficos"])

        # === ABA 1: NOVO BILHETE ===
        with tab_new:
            st.markdown("### Registrar Aposta")
            
            # Inicializa Num Jogos
            if 'num_jogos' not in st.session_state: st.session_state['num_jogos'] = 1
            
            # Dados Gerais
            c_d, c_s, c_o = st.columns(3)
            data_t = c_d.date_input("Data", datetime.now())
            stake = c_s.number_input("Stake (R$)", min_value=0.0, value=10.0)
            odd = c_o.number_input("Odd Total", min_value=1.0, value=1.50)
            
            res_options = ["Green ✅", "Green (Cashout) 💰", "Red ❌", "Reembolso 🔄"]
            resultado = st.selectbox("Resultado", res_options)
            
            # Cálculo Lucro
            lucro = 0.0
            if "Green ✅" in resultado: lucro = (stake * odd) - stake
            elif "Red" in resultado: lucro = -stake
            elif "Cashout" in resultado:
                val_cash = st.number_input("Valor Cashout", min_value=0.0)
                lucro = val_cash - stake
            
            st.info(f"Lucro Previsto: R$ {lucro:.2f}")
            st.divider()

            # Loop de Jogos
            jogos_data = []
            for j in range(st.session_state['num_jogos']):
                st.markdown(f"**Jogo {j+1}**")
                col_m, col_v = st.columns(2)
                m = col_m.selectbox(f"Mandante {j}", team_list_with_empty, key=f"m_{j}")
                v = col_v.selectbox(f"Visitante {j}", team_list_with_empty, key=f"v_{j}")
                
                # Seleção Simples (1 por jogo para simplificar UX se quiser, ou manter complexo)
                # Mantendo simplificado para evitar tela branca por excesso de widgets
                sel_mercado = st.selectbox(f"Mercado {j}", MERCADOS_LISTA, key=f"merc_{j}")
                
                jogos_data.append({
                    "Nome": f"{m} x {v}",
                    "Selecoes": [{"Mercado": sel_mercado, "Status": "Registrado"}]
                })
                st.markdown("---")

            # Botões Controle
            cb1, cb2 = st.columns(2)
            if cb1.button("➕ Adicionar Jogo"):
                st.session_state['num_jogos'] += 1
                st.rerun()
            if cb2.button("🔄 Resetar"):
                st.session_state['num_jogos'] = 1
                st.rerun()

            # Salvar
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

        # === ABA 2: HISTÓRICO ===
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
                        for j in t['Jogos']:
                            st.write(f"⚽ {j['Nome']}")
                        if st.button("Excluir", key=f"del_{t['id']}"):
                            excluir_ticket(t['id'])
                            st.rerun()

        # === ABA 3: GRÁFICOS ===
        with tab_stats:
            if HAS_PLOTLY and tickets:
                df = pd.DataFrame(tickets)
                # Conversão de data segura
                try:
                    df['Data_Dt'] = pd.to_datetime(df['Data'], format="%d/%m/%Y")
                    df = df.sort_values('Data_Dt')
                    df['Acumulado'] = df['Lucro'].cumsum()
                    
                    fig = px.line(df, x='Data_Dt', y='Acumulado', title="Evolução da Banca", markers=True)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Erro ao gerar gráfico: {e}")
            else:
                st.info("Gráficos indisponíveis (sem dados ou Plotly ausente).")

    except Exception as e:
        # CAPTURA QUALQUER ERRO E MOSTRA NA TELA EM VEZ DE FICAR BRANCO
        st.error(f"Ocorreu um erro crítico no Dashboard: {str(e)}")
        st.write("Tente recarregar a página.")

# Chamada Principal
if __name__ == "__main__":
    render_dashboard()
