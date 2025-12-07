

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
import uuid

# ==============================================================================
# 0. CONFIGURAÇÃO E LOGIN
# ==============================================================================
st.set_page_config(page_title="FutPrevisão Pro", layout="wide", page_icon="⚽")

# CSS PERSONALIZADO PARA ESTILO "BETTING APP"
st.markdown("""
<style>
    .ticket-header-win { background-color: #d4edda; padding: 10px; border-radius: 5px; border-left: 5px solid #28a745; }
    .ticket-header-loss { background-color: #f8d7da; padding: 10px; border-radius: 5px; border-left: 5px solid #dc3545; }
    .ticket-header-cashout { background-color: #fff3cd; padding: 10px; border-radius: 5px; border-left: 5px solid #ffc107; }
    .selection-row { font-size: 14px; margin-bottom: 5px; padding-bottom: 5px; border-bottom: 1px solid #eee; }
    .selection-game { font-weight: bold; color: #333; }
    .selection-market { color: #555; }
    .big-stat { font-size: 20px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

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
DATA_FILE = "historico_bilhetes_v5.json" # Novo arquivo para nova estrutura
CONFIG_FILE = "config_banca.json"

def carregar_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f: return json.load(f)
    return {"banca_inicial": 1000.0, "stop_loss": 50.0}

def salvar_config(cfg):
    with open(CONFIG_FILE, "w") as f: json.dump(cfg, f)

def carregar_tickets():
    if not os.path.exists(DATA_FILE): return []
    with open(DATA_FILE, "r") as f: dados = json.load(f)
    # Ordenar do mais recente para o mais antigo
    return sorted(dados, key=lambda x: datetime.strptime(x['Data'], "%d/%m/%Y"), reverse=True)

def salvar_ticket(ticket_data):
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f: dados = json.load(f)
    else: dados = []
    
    # Adiciona ID único se não tiver
    if "id" not in ticket_data:
        ticket_data["id"] = str(uuid.uuid4())[:8]
        
    dados.append(ticket_data)
    with open(DATA_FILE, "w") as f: json.dump(dados, f)

def excluir_ticket(id_ticket):
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f: dados = json.load(f)
        
        novos_dados = [t for t in dados if t.get("id") != id_ticket]
        
        with open(DATA_FILE, "w") as f: json.dump(novos_dados, f)
        return True
    return False

# ==============================================================================
# 3. DASHBOARD PROFISSIONAL
# ==============================================================================
def render_dashboard():
    st.title("📊 Gestão Profissional de Banca")
    
    # --- CONFIGURAÇÃO RÁPIDA ---
    cfg = carregar_config()
    with st.sidebar.expander("⚙️ Configurações da Banca", expanded=False):
        nova_banca = st.number_input("Banca Inicial (R$)", value=cfg["banca_inicial"])
        novo_stop = st.number_input("Stop Loss Diário (R$)", value=cfg["stop_loss"])
        if st.button("Salvar Config"):
            salvar_config({"banca_inicial": nova_banca, "stop_loss": novo_stop})
            st.rerun()
    
    # --- CÁLCULOS GERAIS ---
    tickets = carregar_tickets()
    lucro_total = sum(t["Lucro"] for t in tickets)
    banca_atual = cfg["banca_inicial"] + lucro_total
    
    # --- MONITOR DE STOP LOSS ---
    hoje = datetime.now().strftime("%d/%m/%Y")
    prejuizo_hoje = sum(t["Lucro"] for t in tickets if t["Data"] == hoje and t["Lucro"] < 0)
    perda_atual = abs(prejuizo_hoje)
    pct_perda = min(perda_atual / cfg["stop_loss"], 1.0) if cfg["stop_loss"] > 0 else 0
    
    col_stop, col_banca = st.columns([3, 2])
    with col_stop:
        st.markdown(f"**🛡️ Stop Loss Diário**: Perdeu R$ {perda_atual:.2f} de R$ {cfg['stop_loss']:.2f}")
        cor_barra = "green" if pct_perda < 0.5 else "orange" if pct_perda < 0.9 else "red"
        st.progress(pct_perda)
        if pct_perda >= 1.0: st.error("⛔ STOP LOSS ATINGIDO! PARE HOJE.")
            
    with col_banca:
        delta_banca = banca_atual - cfg["banca_inicial"]
        st.metric("Banca Atual", f"R$ {banca_atual:.2f}", delta=f"{delta_banca:.2f}")

    st.divider()

    # --- ABAS ---
    tab_add, tab_history, tab_analise = st.tabs(["➕ Novo Bilhete", "📜 Histórico de Bilhetes", "📈 Análise"])

    # ==============================================================================
    # ABA 1: NOVO BILHETE (INPUT)
    # ==============================================================================
    with tab_add:
        with st.form("form_pro"):
            st.subheader("💰 Novo Bilhete")
            c1, c2 = st.columns(2)
            with c1: 
                data_bilhete = st.date_input("Data", datetime.now())
                resultado_bilhete = st.selectbox("Resultado", ["Green ✅", "Green (Cashout) 💰", "Red ❌", "Reembolso 🔄"])
            with c2: 
                stake = st.number_input("Stake (R$)", min_value=0.0, step=5.0)
                odd = st.number_input("Odd Total", min_value=1.00, step=0.01)

            # Lógica Financeira
            valor_retornado_manual = 0.0
            if "Cashout" in resultado_bilhete:
                valor_retornado_manual = st.number_input("Valor Saque (Cashout)", min_value=0.0)

            if "Green ✅" in resultado_bilhete: lucro_final = (stake * odd) - stake
            elif "Red" in resultado_bilhete: lucro_final = -stake
            elif "Reembolso" in resultado_bilhete: lucro_final = 0.0
            else: lucro_final = valor_retornado_manual - stake

            st.divider()
            
            # CONSTRUTOR DE SELEÇÕES
            st.markdown("#### 📝 Seleções")
            qtd_jogos = st.slider("Qtd. Jogos", 1, 8, 1)
            selecoes = []
            
            for i in range(qtd_jogos):
                st.markdown(f"**Jogo {i+1}**")
                c_h, c_x, c_a = st.columns([3, 0.2, 3])
                with c_h: mandante = st.selectbox(f"Casa", team_list_with_empty, key=f"h{i}")
                with c_x: st.write("x")
                with c_a: visitante = st.selectbox(f"Fora", team_list_with_empty, key=f"a{i}")
                
                c_m1, c_m2 = st.columns([2, 3])
                with c_m1: alvo = st.selectbox(f"Alvo", ["🟢 Mandante", "🔴 Visitante", "⚪ Geral"], key=f"al{i}")
                with c_m2: mercado = st.selectbox(f"Mercado", MERCADOS_LISTA, key=f"me{i}")
                
                # Checkbox simples para status individual (para visualização futura)
                status_sel = "Green" if "Green" in resultado_bilhete else "Red" # Padrão segue o bilhete
                
                nome_jogo = f"{mandante} x {visitante}" if mandante and visitante else f"Jogo {i+1}"
                
                # Definir ícone baseado no mercado
                icon = "⚽"
                if "Escanteios" in mercado: icon = "🚩"
                elif "Cartões" in mercado: icon = "🟨"
                
                selecoes.append({
                    "Jogo": nome_jogo,
                    "Alvo": alvo,
                    "Mercado": mercado,
                    "Icon": icon
                })
                st.markdown("---")

            # TRAVA
            chk1 = st.checkbox("Checklist: Análise feita e gestão respeitada?")
            submit = st.form_submit_button("💾 SALVAR BILHETE", disabled=not chk1)
            
            if submit:
                if stake <= 0:
                    st.error("Stake inválida!")
                else:
                    novo_ticket = {
                        "Data": data_bilhete.strftime("%d/%m/%Y"),
                        "Resultado": resultado_bilhete,
                        "Stake": stake,
                        "Odd": odd,
                        "Lucro": lucro_final,
                        "Selecoes": selecoes
                    }
                    salvar_ticket(novo_ticket)
                    st.success("Bilhete registrado!")
                    st.rerun()

    # ==============================================================================
    # ABA 2: HISTÓRICO DE BILHETES (VISUAL ESTILO BETANO)
    # ==============================================================================
    with tab_history:
        st.subheader("📜 Meus Bilhetes")
        
        if not tickets:
            st.info("Nenhum bilhete registrado.")
        
        for ticket in tickets:
            # Definir Cores e Estilos baseado no resultado
            res = ticket["Resultado"]
            if "Green" in res:
                border_color = "#28a745" # Verde
                bg_color = "#e8f5e9" # Verde claro
                icon_status = "✅ GANHO"
                text_color = "green"
            elif "Red" in res:
                border_color = "#dc3545" # Vermelho
                bg_color = "#f8d7da" # Vermelho claro
                icon_status = "❌ PERDIDO"
                text_color = "red"
            else:
                border_color = "#ffc107" # Amarelo
                bg_color = "#fff3cd"
                icon_status = "🔄 REEMBOLSO"
                text_color = "orange"
            
            # CARD DO BILHETE (CONTAINER)
            with st.container(border=True):
                # CABEÇALHO DO BILHETE
                col_head1, col_head2, col_head3, col_head4 = st.columns([2, 1, 1, 1])
                
                with col_head1:
                    st.caption(f"Bilhete #{ticket.get('id', '---')}")
                    st.markdown(f"**{ticket['Data']}**")
                    st.markdown(f"<span style='color:{text_color}; font-weight:bold'>{icon_status}</span>", unsafe_allow_html=True)
                
                with col_head2:
                    st.caption("Stake")
                    st.markdown(f"**R$ {ticket['Stake']:.2f}**")
                    
                with col_head3:
                    st.caption("Odd Total")
                    st.markdown(f"**{ticket['Odd']:.2f}**")
                    
                with col_head4:
                    st.caption("Retorno")
                    val = ticket['Lucro'] + ticket['Stake'] if ticket['Lucro'] > 0 else 0
                    if "Reembolso" in res: val = ticket['Stake']
                    st.markdown(f"**R$ {val:.2f}**")

                # CORPO DO BILHETE (EXPANDER)
                with st.expander("Ver Seleções"):
                    for sel in ticket["Selecoes"]:
                        # Layout da Linha da Seleção
                        c_sel_icon, c_sel_det = st.columns([0.5, 5])
                        with c_sel_icon:
                            st.markdown(f"### {sel.get('Icon', '⚽')}")
                        with c_sel_det:
                            st.markdown(f"**{sel['Jogo']}**")
                            # Formatação da aposta: Alvo + Mercado
                            alvo_limpo = sel['Alvo'].replace("🟢 ", "").replace("🔴 ", "").replace("⚪ ", "")
                            st.caption(f"{alvo_limpo} — {sel['Mercado']}")
                        st.divider()
                    
                    # Botão de Excluir (Discreto dentro do expander)
                    if st.button("🗑️ Excluir Bilhete", key=f"del_{ticket.get('id')}"):
                        if excluir_ticket(ticket.get('id')):
                            st.success("Excluído!")
                            st.rerun()

    # ==============================================================================
    # ABA 3: ANÁLISE GRÁFICA
    # ==============================================================================
    with tab_analise:
        if not tickets:
            st.info("Sem dados.")
        else:
            total_stake = sum(t["Stake"] for t in tickets)
            roi = (lucro_total / total_stake) * 100 if total_stake > 0 else 0
            greens = len([t for t in tickets if "Green" in t["Resultado"]])
            win_rate = (greens / len(tickets)) * 100
            
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Lucro Líquido", f"R$ {lucro_total:.2f}", delta_color="normal")
            k2.metric("ROI", f"{roi:.1f}%")
            k3.metric("Win Rate", f"{win_rate:.1f}%")
            k4.metric("Total Apostado", f"R$ {total_stake:.2f}")
            
            # Gráfico de Evolução
            df_hist = pd.DataFrame(tickets)
            # Reverter para ordem cronológica para o gráfico
            df_chart = df_hist.iloc[::-1].copy()
            df_chart['Lucro_Acumulado'] = df_chart['Lucro'].cumsum() + cfg["banca_inicial"]
            
            st.subheader("Curva de Banca")
            st.line_chart(df_chart['Lucro_Acumulado'])

# ==============================================================================
# 4. NAVEGAÇÃO E PREVISÕES (CÓDIGO ORIGINAL)
# ==============================================================================
st.sidebar.markdown("---")
pagina = st.sidebar.radio("Menu", ["🏠 Previsões IA", "📊 Gestão de Banca"])

if pagina == "📊 Gestão de Banca":
    render_dashboard()
    st.stop()

# --- PREVISÕES ---
st.sidebar.markdown("---")
st.sidebar.title("FutPrevisão Pro v5.0")

def carregar_metadados():
    try:
        with open("metadados.json", "r", encoding='utf-8') as f: return json.load(f)
    except: return None

meta = carregar_metadados()
if meta:
    st.sidebar.caption("🤖 Status do Robô:")
    st.sidebar.text(f"{meta['ultima_verificacao']}")
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
