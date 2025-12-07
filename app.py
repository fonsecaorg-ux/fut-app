

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import poisson
import json
import hmac
import os
from datetime import datetime

# ==============================================================================
# 0. CONFIGURAÇÃO E LOGIN
# ==============================================================================
st.set_page_config(page_title="FutPrevisão Pro", layout="wide", page_icon="⚽")

def check_password():
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
            st.error("Erro: Senhas não configuradas.")

    if st.session_state["password_correct"]: return True
    st.markdown("### 🔒 Acesso Restrito - FutPrevisão Pro")
    st.text_input("Usuário", key="username")
    st.text_input("Senha", type="password", key="password")
    st.button("Entrar", on_click=password_entered)
    return False

if not check_password(): st.stop()

# ==============================================================================
# 1. CARREGAMENTO DE DADOS (DADOS_TIMES.CSV - INTACTO)
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
    
    try:
        df_ref = pd.read_csv("arbitros.csv")
        referees = dict(zip(df_ref['Nome'], df_ref['Fator']))
    except:
        referees = {}
        
    referees[' Estilo: Rigoroso (+ Cartões)'] = 1.25
    referees[' Estilo: Normal (Padrão)'] = 1.00
    referees[' Estilo: Conservador (- Cartões)'] = 0.80
        
    return teams_dict, referees

teams_data, referees_data = load_data()
team_list_raw = sorted(list(teams_data.keys()))
team_list_with_empty = [""] + team_list_raw

# LISTA MESTRA DE MERCADOS
MERCADOS_LISTA = ["Selecione o mercado..."]
for i in [2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5]:
    MERCADOS_LISTA.append(f"Escanteios Mais de {i}")
for i in [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]:
    MERCADOS_LISTA.append(f"Cartões Mais de {i}")
for i in [0.5, 1.5, 2.5, 3.5, 4.5]:
    MERCADOS_LISTA.append(f"Gols Mais de {i}")
MERCADOS_LISTA.extend(["Ambas Marcam", "Vitória (ML) Casa", "Vitória (ML) Fora", "Empate", "Dupla Chance"])

# ==============================================================================
# 2. FUNÇÕES DE GESTÃO (SALVAR/CARREGAR/CONFIG)
# ==============================================================================
DATA_FILE = "diario_v4.json"
CONFIG_FILE = "config_banca.json"

def carregar_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f: return json.load(f)
    return {"banca_inicial": 1000.0, "stop_loss": 50.0}

def salvar_config(cfg):
    with open(CONFIG_FILE, "w") as f: json.dump(cfg, f)

def carregar_historico():
    if not os.path.exists(DATA_FILE): return pd.DataFrame()
    with open(DATA_FILE, "r") as f: dados = json.load(f)
    return pd.DataFrame(dados)

def salvar_ticket(ticket_data):
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f: dados = json.load(f)
    else: dados = []
    dados.append(ticket_data)
    with open(DATA_FILE, "w") as f: json.dump(dados, f)

def excluir_ticket(index):
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f: dados = json.load(f)
        if 0 <= index < len(dados):
            dados.pop(index)
            with open(DATA_FILE, "w") as f: json.dump(dados, f)
            return True
    return False

# ==============================================================================
# 3. DASHBOARD PROFISSIONAL
# ==============================================================================
def render_dashboard():
    st.title("📊 Gestão Profissional de Banca")
    
    # --- CONFIGURAÇÃO RÁPIDA (SIDEBAR) ---
    cfg = carregar_config()
    with st.sidebar.expander("⚙️ Configurações da Banca", expanded=False):
        nova_banca = st.number_input("Banca Inicial (R$)", value=cfg["banca_inicial"])
        novo_stop = st.number_input("Stop Loss Diário (R$)", value=cfg["stop_loss"])
        if st.button("Salvar Config"):
            salvar_config({"banca_inicial": nova_banca, "stop_loss": novo_stop})
            st.rerun()
    
    # --- CÁLCULOS GERAIS ---
    df = carregar_historico()
    lucro_total = df["Lucro"].sum() if not df.empty else 0.0
    banca_atual = cfg["banca_inicial"] + lucro_total
    
    # --- SEMÁFORO STOP LOSS (PROTEÇÃO) ---
    hoje = datetime.now().strftime("%d/%m/%Y")
    if not df.empty:
        df_hoje = df[df["Data"] == hoje]
        prejuizo_hoje = df_hoje[df_hoje["Lucro"] < 0]["Lucro"].sum() # Valor negativo
        perda_atual = abs(prejuizo_hoje)
    else:
        perda_atual = 0.0
        
    pct_perda = min(perda_atual / cfg["stop_loss"], 1.0)
    
    col_stop, col_banca = st.columns([3, 2])
    with col_stop:
        st.markdown(f"**🛡️ Monitor de Stop Loss (Hoje)**: Perdeu R$ {perda_atual:.2f} / Limite R$ {cfg['stop_loss']:.2f}")
        cor_barra = "green" if pct_perda < 0.5 else "orange" if pct_perda < 0.9 else "red"
        st.progress(pct_perda)
        if pct_perda >= 1.0:
            st.error("⛔ STOP LOSS ATINGIDO! PARE DE APOSTAR HOJE PARA PROTEGER SUA BANCA.")
            
    with col_banca:
        delta_banca = banca_atual - cfg["banca_inicial"]
        st.metric("Banca Atual", f"R$ {banca_atual:.2f}", delta=f"{delta_banca:.2f}")

    st.divider()

    # --- ABAS DE GESTÃO ---
    tab_add, tab_analise, tab_manage = st.tabs(["➕ Novo Bilhete", "📈 Análise Pro", "📚 Editar/Excluir"])

    # --- ABA 1: NOVO BILHETE (COM TRAVAS) ---
    with tab_add:
        with st.form("form_pro"):
            c1, c2, c3 = st.columns(3)
            with c1: 
                data_bilhete = st.date_input("Data", datetime.now())
                resultado_bilhete = st.selectbox("Resultado", ["Green ✅", "Green (Cashout) 💰", "Red ❌", "Reembolso 🔄"])
            with c2: 
                stake = st.number_input("Stake / Valor Apostado (R$)", min_value=0.0, step=1.0)
                odd = st.number_input("Odd / Cotação Total", min_value=1.00, step=0.01)
            with c3:
                # Cálculo automático de retorno
                retorno_calc = stake * odd
                lucro_previsto = retorno_calc - stake
                st.metric("Retorno Potencial", f"R$ {retorno_calc:.2f}")
                
                # Se for Cashout ou Red, o usuário ajusta o valor real retornado
                valor_retornado = st.number_input("Valor Retornado (Preencher se Cashout)", min_value=0.0, value=retorno_calc if "Green ✅" in resultado_bilhete else 0.0)

            # Lógica do Lucro Final
            if "Green ✅" in resultado_bilhete: lucro_final = (stake * odd) - stake
            elif "Red" in resultado_bilhete: lucro_final = -stake
            elif "Reembolso" in resultado_bilhete: lucro_final = 0.0
            else: lucro_final = valor_retornado - stake # Cashout

            st.divider()
            st.markdown("#### 📝 Detalhes da Múltipla")
            qtd_jogos = st.slider("Qtd. Jogos", 1, 5, 1)
            selecoes = []
            
            for i in range(qtd_jogos):
                c_h, c_x, c_a = st.columns([3, 0.2, 3])
                with c_h: mandante = st.selectbox(f"Casa {i+1}", team_list_with_empty, key=f"h{i}")
                with c_x: st.write("x")
                with c_a: visitante = st.selectbox(f"Fora {i+1}", team_list_with_empty, key=f"a{i}")
                
                c_m1, c_m2 = st.columns([2, 3])
                with c_m1: alvo = st.selectbox(f"Alvo {i+1}", ["🟢 Mandante", "🔴 Visitante", "⚪ Geral"], key=f"al{i}")
                with c_m2: mercado = st.selectbox(f"Mercado {i+1}", MERCADOS_LISTA, key=f"me{i}")
                
                nome_jogo = f"{mandante} x {visitante}" if mandante and visitante else f"Jogo {i+1}"
                selecoes.append({"Jogo": nome_jogo, "Alvo": alvo, "Mercado": mercado})
                st.markdown("---")

            # --- TRAVA DE DISCIPLINA (CHECKLIST) ---
            st.markdown("#### 🛡️ Checklist de Disciplina (Obrigatório)")
            chk1 = st.checkbox("Analisei as escalações e o 'Must Win'?")
            chk2 = st.checkbox(f"A Stake (R$ {stake}) está dentro da minha gestão?")
            chk3 = st.checkbox("Estou calmo e sem tentar recuperar Reds passados?")
            
            pode_salvar = chk1 and chk2 and chk3
            
            submit = st.form_submit_button("💾 REGISTRAR APOSTA", disabled=not pode_salvar)
            if not pode_salvar: st.caption("⚠️ Marque todas as opções do checklist para liberar o botão.")
            
            if submit:
                novo_ticket = {
                    "Data": data_bilhete.strftime("%d/%m/%Y"),
                    "Resultado": resultado_bilhete,
                    "Stake": stake,
                    "Odd": odd,
                    "Lucro": lucro_final,
                    "Selecoes": selecoes,
                    "Resumo": f"{len(selecoes)} jogos"
                }
                salvar_ticket(novo_ticket)
                st.success("Bilhete Registrado!")
                st.rerun()

    # --- ABA 2: ANÁLISE PRO ---
    with tab_analise:
        if df.empty:
            st.info("Sem dados.")
        else:
            # ROI e Winrate
            total_stake = df["Stake"].sum()
            roi = (lucro_total / total_stake) * 100 if total_stake > 0 else 0
            greens = len(df[df["Resultado"].str.contains("Green")])
            win_rate = (greens / len(df)) * 100
            
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Lucro Líquido", f"R$ {lucro_total:.2f}")
            k2.metric("ROI (Retorno)", f"{roi:.1f}%")
            k3.metric("Win Rate", f"{win_rate:.1f}%")
            k4.metric("Stake Total", f"R$ {total_stake:.2f}")
            
            st.subheader("📈 Curva de Crescimento da Banca")
            df['Lucro_Acumulado'] = df['Lucro'].cumsum() + cfg["banca_inicial"]
            # Adiciona ponto inicial
            df_chart = pd.concat([pd.DataFrame({'Lucro_Acumulado': [cfg["banca_inicial"]]}), df[['Lucro_Acumulado']]]).reset_index(drop=True)
            
            st.line_chart(df_chart)

    # --- ABA 3: GERENCIAR (EDITAR/EXCLUIR) ---
    with tab_manage:
        st.subheader("📚 Histórico Completo")
        if df.empty:
            st.info("Histórico vazio.")
        else:
            # Exibe tabela com Index para referência
            st.dataframe(df[["Data", "Stake", "Odd", "Lucro", "Resultado"]])
            
            c_del1, c_del2 = st.columns([3, 1])
            with c_del1:
                id_to_delete = st.number_input("Digite o ID (Índice) da linha para excluir:", min_value=0, max_value=len(df)-1, step=1)
            with c_del2:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("🗑️ Excluir Bilhete"):
                    if excluir_ticket(id_to_delete):
                        st.success(f"Bilhete {id_to_delete} excluído!")
                        st.rerun()
                    else:
                        st.error("Erro ao excluir.")

# ==============================================================================
# 4. NAVEGAÇÃO
# ==============================================================================
st.sidebar.markdown("---")
pagina = st.sidebar.radio("Navegação", ["🏠 Previsões IA", "📊 Gestão de Banca"])

if pagina == "📊 Gestão de Banca":
    render_dashboard()
    st.stop()

# ==============================================================================
# 5. PREVISÕES (CÓDIGO ORIGINAL - MANTIDO)
# ==============================================================================
st.sidebar.markdown("---")
st.sidebar.title("FutPrevisão Pro v4.0")

def carregar_metadados():
    try:
        with open("metadados.json", "r", encoding='utf-8') as f: return json.load(f)
    except: return None

meta = carregar_metadados()
if meta:
    st.sidebar.caption("🤖 Status do Robô:")
    st.sidebar.text(f"{meta['ultima_verificacao']}")
    if meta['times_alterados'] == 0 and 'log' not in meta: st.sidebar.info("✔ Base estável.")
else: st.sidebar.warning("⚠ Aguardando dados...")

st.sidebar.markdown("---")
st.sidebar.header("Configuração da Partida")

home_team = st.sidebar.selectbox("Mandante", team_list_raw, index=0)
away_team = st.sidebar.selectbox("Visitante", team_list_raw, index=1)

st.sidebar.caption("🧠 Contexto")
context_options = {
    "⚪ Neutro": 1.0, "🔥 Must Win (Z4)": 1.15, "🏆 Must Win (Título)": 1.15,
    "❄️ Desmobilizado": 0.85, "💪 Super Favorito": 1.25, "🚑 Crise": 0.80
}
ctx_h = st.sidebar.selectbox(f"Momento: {home_team}", list(context_options.keys()), index=0)
ctx_a = st.sidebar.selectbox(f"Momento: {away_team}", list(context_options.keys()), index=0)
f_h = context_options[ctx_h]
f_a = context_options[ctx_a]

st.sidebar.markdown("---")
referee_list = sorted(list(referees_data.keys()))
ref_name = st.sidebar.selectbox("Árbitro", referee_list)
ref_factor = referees_data[ref_name]
st.sidebar.metric("Rigor", ref_factor)

champions_mode = st.sidebar.checkbox("Modo Champions (-15%)", value=False)

def calculate_metrics(home, away, ref_factor, is_champions, fact_h, fact_a):
    h_data = teams_data[home]
    a_data = teams_data[away]
    
    corn_h = (h_data['corners'] * 1.10) * fact_h
    corn_a = (a_data['corners'] * 0.85) * fact_a
    if is_champions: corn_h *= 0.85; corn_a *= 0.85
    total_corners = corn_h + corn_a
        
    tension_boost = 1.10 if fact_h > 1.0 or fact_a > 1.0 else 1.0
    tension = ((h_data['fouls'] + a_data['fouls']) / 24.0) * tension_boost
    tension = max(0.85, min(tension, 1.40))
    
    card_h = h_data['cards'] * tension * ref_factor
    card_a = a_data['cards'] * tension * ref_factor
    total_cards = card_h + card_a
    
    avg_l = 1.3
    exp_h = ((h_data['goals_f'] * fact_h)/avg_l) * (a_data['goals_a']/avg_l) * avg_l
    exp_a = ((a_data['goals_f'] * fact_a)/avg_l) * (h_data['goals_a']/avg_l) * avg_l
    
    return {'total_corners': total_corners, 'ind_corn_h': corn_h, 'ind_corn_a': corn_a,
            'total_cards': total_cards, 'ind_card_h': card_h, 'ind_card_a': card_a,
            'goals_h': exp_h, 'goals_a': exp_a, 'tension': tension}

def prob_over(exp, line): return poisson.sf(int(line), exp) * 100

st.title("⚽ FutPrevisão Pro")
tab_analise, tab_scanner = st.tabs(["📊 Análise do Jogo", "🔍 Scanner"])

with tab_analise:
    st.markdown(f"### {home_team} x {away_team}")
    if st.sidebar.button("Gerar Previsões 🚀", type="primary"):
        m = calculate_metrics(home_team, away_team, ref_factor, champions_mode, f_h, f_a)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Escanteios", f"{m['total_corners']:.2f}")
        c2.metric("Cartões", f"{m['total_cards']:.2f}")
        c3.metric("Tensão", f"{m['tension']:.2f}")
        st.divider()

        st.subheader("🚩 Escanteios")
        cols = st.columns(6)
        for i, line in enumerate([7.5, 8.5, 9.5, 10.5, 11.5, 12.5]):
            p = prob_over(m['total_corners'], line)
            c = "green" if p >= 70 else "orange" if p >= 50 else "red"
            cols[i].markdown(f"**+{line}**\n:{c}[**{p:.1f}%**]")
        
        c_h, c_m, c_a = st.columns([1,1,1])
        with c_h:
            st.write(f"🏠 {home_team}: **{m['ind_corn_h']:.2f}**")
            for l in [3.5, 4.5, 5.5]: st.write(f"+{l}: **{prob_over(m['ind_corn_h'], l):.1f}%**")
        with c_a:
            st.write(f"✈️ {away_team}: **{m['ind_corn_a']:.2f}**")
            for l in [3.5, 4.5, 5.5]: st.write(f"+{l}: **{prob_over(m['ind_corn_a'], l):.1f}%**")
        with c_m:
            fig = go.Figure(data=[go.Bar(x=[home_team, away_team], y=[m['ind_corn_h'], m['ind_corn_a']], marker_color=['blue', 'red'])])
            fig.update_layout(height=150, margin=dict(l=5, r=5, t=5, b=5))
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("🟨 Cartões")
        c_h, c_m, c_a = st.columns([1,1,1])
        with c_h:
            st.write(f"🏠 {home_team}: **{m['ind_card_h']:.2f}**")
            for l in [1.5, 2.5, 3.5]: st.markdown(f"+{l}: **{prob_over(m['ind_card_h'], l):.1f}%**")
        with c_a:
            st.write(f"✈️ {away_team}: **{m['ind_card_a']:.2f}**")
            for l in [1.5, 2.5, 3.5]: st.markdown(f"+{l}: **{prob_over(m['ind_card_a'], l):.1f}%**")
        
        st.divider()
        st.subheader("⚽ Gols")
        ph = [poisson.pmf(i, m['goals_h']) for i in range(6)]
        pa = [poisson.pmf(i, m['goals_a']) for i in range(6)]
        prob_h = sum([ph[i]*pa[j] for i in range(6) for j in range(6) if i > j]) * 100
        prob_d = sum([ph[i]*pa[j] for i in range(6) for j in range(6) if i == j]) * 100
        prob_a = sum([ph[i]*pa[j] for i in range(6) for j in range(6) if i < j]) * 100
        c1, c2, c3 = st.columns(3)
        c1.metric(home_team, f"{prob_h:.1f}%")
        c2.metric("Empate", f"{prob_d:.1f}%")
        c3.metric(away_team, f"{prob_a:.1f}%")

with tab_scanner:
    st.subheader("🕵️‍♂️ Scanner")
    df_rank = pd.DataFrame.from_dict(teams_data, orient='index')
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Cantos**")
        st.dataframe(df_rank.sort_values('corners', ascending=False)[['corners']].head(10))
    with c2:
        st.markdown("**Cartões**")
        st.dataframe(df_rank.sort_values('cards', ascending=False)[['cards']].head(10))