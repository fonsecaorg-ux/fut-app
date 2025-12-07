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
# 1. CARREGAMENTO DE DADOS E LISTAS
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

# --- AJUSTE: LISTA DE TIMES COM CAMPO VAZIO NO INÍCIO ---
team_list_raw = sorted(list(teams_data.keys()))
team_list_with_empty = [""] + team_list_raw # Adiciona opção vazia para não travar em Alaves

# --- LISTA MESTRA DE MERCADOS ---
MERCADOS_LISTA = ["Selecione o mercado..."]
# Escanteios
for i in [2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5]:
    MERCADOS_LISTA.append(f"Escanteios Mais de {i}")
# Cartões
for i in [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]:
    MERCADOS_LISTA.append(f"Cartões Mais de {i}")
# Gols
for i in [0.5, 1.5, 2.5, 3.5, 4.5]:
    MERCADOS_LISTA.append(f"Gols Mais de {i}")
# Outros
MERCADOS_LISTA.append("Ambas Marcam")
MERCADOS_LISTA.append("Vitória (ML) Casa")
MERCADOS_LISTA.append("Vitória (ML) Fora")
MERCADOS_LISTA.append("Empate")

# ==============================================================================
# 2. FUNÇÕES DO DASHBOARD
# ==============================================================================
DATA_FILE = "diario_apostas_v3.json"

def carregar_historico():
    if not os.path.exists(DATA_FILE):
        return pd.DataFrame()
    else:
        with open(DATA_FILE, "r") as f:
            dados = json.load(f)
        return pd.DataFrame(dados)

def salvar_ticket(ticket_data):
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            dados = json.load(f)
    else:
        dados = []
    
    dados.append(ticket_data)
    with open(DATA_FILE, "w") as f:
        json.dump(dados, f)

def render_dashboard():
    st.title("📊 Dashboard & Diário de Apostas")
    
    # --- CONSTRUTOR DE BILHETE ---
    with st.expander("➕ Novo Bilhete (Construtor Completo)", expanded=True):
        
        with st.form("form_bet_builder"):
            # 1. Cabeçalho Financeiro
            st.markdown("#### 💰 Resumo do Bilhete")
            c1, c2, c3 = st.columns(3)
            with c1: 
                data_bilhete = st.date_input("Data", datetime.now())
            with c2: 
                resultado_bilhete = st.selectbox("Resultado Final", ["Green ✅", "Green (Cashout) 💰", "Red ❌"])
            with c3: 
                lucro_bilhete = st.number_input("Lucro/Prejuízo Total (R$)", min_value=-10000.0, max_value=10000.0, step=1.0, help="Use valor negativo para Red")
            
            st.divider()
            
            # 2. Configuração dos Jogos
            st.markdown("#### 📝 Montar Bilhete")
            qtd_jogos = st.slider("Quantos jogos neste bilhete?", 1, 5, 1)
            
            selecoes_final = []
            
            # LOOP PARA CRIAR OS BLOCOS DE JOGO
            for i in range(qtd_jogos):
                st.markdown(f"**JOGO {i+1}**")
                
                # A. Seleção do Confronto (Agora com lista vazia inicial)
                col_home, col_x, col_away = st.columns([3, 0.5, 3])
                with col_home:
                    mandante = st.selectbox(f"🏠 Mandante {i+1}", team_list_with_empty, key=f"home_{i}")
                with col_x:
                    st.markdown("<div style='text-align: center; padding-top: 30px;'>x</div>", unsafe_allow_html=True)
                with col_away:
                    visitante = st.selectbox(f"✈️ Visitante {i+1}", team_list_with_empty, key=f"away_{i}")

                # B. Seleção das Apostas (Genérico para não travar)
                # Linha 1 (Obrigatória)
                c_alvo1, c_mercado1 = st.columns([2, 3])
                with c_alvo1:
                    # FIX: Usamos nomes estáticos para não depender da seleção acima (evita lag)
                    opcoes_alvo = ["🟢 Mandante (Casa)", "🔴 Visitante (Fora)", "⚪ Total do Jogo"]
                    alvo1 = st.selectbox(f"Aposta 1 - Alvo", opcoes_alvo, key=f"alvo1_{i}")
                with c_mercado1:
                    mercado1 = st.selectbox(f"Aposta 1 - Mercado", MERCADOS_LISTA, key=f"merc1_{i}")
                
                # Se o usuário selecionou times, salvamos os nomes reais. Se não, salvamos "Indefinido"
                nome_jogo = f"{mandante} x {visitante}" if mandante and visitante else f"Jogo {i+1}"
                
                # Lógica para salvar o nome real do time na aposta (Pós-processamento)
                nome_alvo_real = alvo1
                if mandante and "Mandante" in alvo1: nome_alvo_real = f"🟢 {mandante} (Casa)"
                if visitante and "Visitante" in alvo1: nome_alvo_real = f"🔴 {visitante} (Fora)"

                selecoes_final.append({"Jogo": nome_jogo, "Alvo": nome_alvo_real, "Mercado": mercado1})

                # Linha 2 (Opcional - Checkbox para ativar)
                usar_segunda = st.checkbox(f"Adicionar 2ª seleção para este jogo? (Criar Aposta)", key=f"check_{i}")
                if usar_segunda:
                    c_alvo2, c_mercado2 = st.columns([2, 3])
                    with c_alvo2:
                        alvo2 = st.selectbox(f"Aposta 2 - Alvo", opcoes_alvo, key=f"alvo2_{i}")
                    with c_mercado2:
                        mercado2 = st.selectbox(f"Aposta 2 - Mercado", MERCADOS_LISTA, key=f"merc2_{i}")
                    
                    # Lógica para salvar nome real
                    nome_alvo_real_2 = alvo2
                    if mandante and "Mandante" in alvo2: nome_alvo_real_2 = f"🟢 {mandante} (Casa)"
                    if visitante and "Visitante" in alvo2: nome_alvo_real_2 = f"🔴 {visitante} (Fora)"
                    
                    selecoes_final.append({"Jogo": nome_jogo, "Alvo": nome_alvo_real_2, "Mercado": mercado2})
                
                st.markdown("---") # Linha separadora entre jogos

            # Botão Salvar
            submitted = st.form_submit_button("💾 Salvar Bilhete no Diário")
            
            if submitted:
                # Validação básica
                if lucro_bilhete == 0 and "Green" in resultado_bilhete:
                    st.warning("⚠️ Atenção: Você marcou Green mas o lucro está R$ 0.00.")
                
                # Ajuste de sinal financeiro
                if "Red" in resultado_bilhete and lucro_bilhete > 0:
                    lucro_bilhete = lucro_bilhete * -1
                
                novo_ticket = {
                    "Data": data_bilhete.strftime("%d/%m/%Y"),
                    "Resultado": resultado_bilhete,
                    "Lucro": lucro_bilhete,
                    "Qtd_Jogos": qtd_jogos,
                    "Selecoes": selecoes_final,
                    "Resumo": f"{len(selecoes_final)} seleções"
                }
                salvar_ticket(novo_ticket)
                st.success("✅ Bilhete registrado com sucesso!")
                st.rerun()

    # --- ANÁLISE DOS DADOS (GRÁFICOS) ---
    df = carregar_historico()
    
    if df.empty:
        st.info("👈 Use o formulário acima para registrar seu primeiro bilhete.")
        return

    st.divider()
    
    # 1. KPIs Gerais
    lucro_total = df["Lucro"].sum()
    greens = len(df[df["Resultado"].str.contains("Green")])
    total = len(df)
    win_rate = (greens / total) * 100 if total > 0 else 0

    k1, k2, k3 = st.columns(3)
    k1.metric("Banca (Lucro Líquido)", f"R$ {lucro_total:.2f}", delta=f"{lucro_total:.2f}")
    k2.metric("Assertividade", f"{win_rate:.1f}%")
    k3.metric("Total de Bilhetes", total)

    # 2. Gráficos Inteligentes
    st.subheader("🔍 Raio-X da Performance")
    
    # Processar dados para gráficos (Explodir seleções)
    lista_analise = []
    for _, row in df.iterrows():
        status_simples = "Green" if "Green" in row["Resultado"] else "Red"
        for sel in row["Selecoes"]:
            tipo = "Casa" if "🟢" in sel["Alvo"] else "Fora" if "🔴" in sel["Alvo"] else "Geral"
            # Tenta extrair o nome do mercado (ex: "Escanteios")
            try: nome_mercado = sel["Mercado"].split()[0]
            except: nome_mercado = "Outros"
            
            lista_analise.append({
                "Local": tipo,
                "Mercado": nome_mercado,
                "Status": status_simples
            })
            
    if lista_analise:
        df_an = pd.DataFrame(lista_analise)
        
        g1, g2 = st.columns(2)
        with g1:
            st.markdown("**Onde você acerta mais? (Casa vs Fora)**")
            fig1 = px.histogram(df_an, x="Local", color="Status", barmode="group",
                                color_discrete_map={"Green": "#00CC96", "Red": "#EF553B"})
            st.plotly_chart(fig1, use_container_width=True)
            
        with g2:
            st.markdown("**Qual seu melhor mercado?**")
            fig2 = px.histogram(df_an, x="Mercado", color="Status", barmode="group",
                                color_discrete_map={"Green": "#00CC96", "Red": "#EF553B"})
            st.plotly_chart(fig2, use_container_width=True)

    # 3. Histórico Recente
    st.subheader("📜 Histórico de Bilhetes")
    st.dataframe(df[["Data", "Resultado", "Lucro", "Resumo"]].iloc[::-1], use_container_width=True)


# ==============================================================================
# 3. MENU DE NAVEGAÇÃO
# ==============================================================================
st.sidebar.markdown("---")
st.sidebar.header("Navegação")
pagina = st.sidebar.radio("Ir para:", ["🏠 Previsões do Jogo", "📊 Dashboard & Diário"])

if pagina == "📊 Dashboard & Diário":
    render_dashboard()
    st.stop() # PARA TUDO AQUI PARA MOSTRAR SÓ O DASHBOARD

# ==============================================================================
# 4. PREVISÕES (CÓDIGO ORIGINAL)
# ==============================================================================
st.sidebar.markdown("---")
st.sidebar.title("FutPrevisão Pro v3.2")

def carregar_metadados():
    try:
        with open("metadados.json", "r", encoding='utf-8') as f:
            return json.load(f)
    except: return None

meta = carregar_metadados()
if meta:
    st.sidebar.caption("🤖 **Status do Robô:**")
    st.sidebar.text(f"{meta['ultima_verificacao']}")
    if meta['times_alterados'] == 0 and 'log' not in meta:
        st.sidebar.info("✔ Base verificada e estável.")
else:
    st.sidebar.warning("⚠ Aguardando robô...")

st.sidebar.markdown("---")
st.sidebar.header("Configuração da Partida")

# Usa a lista já carregada lá em cima
home_team = st.sidebar.selectbox("Mandante", team_list_raw, index=0)
away_team = st.sidebar.selectbox("Visitante", team_list_raw, index=1)

st.sidebar.markdown("---")
st.sidebar.caption("🧠 **Contexto**")
context_options = {
    "⚪ Neutro (Meio de Tabela": 1.0,
    "🔥 Must Win (Z4)": 1.15,
    "🏆 Must Win (Título/Libertadores)": 1.15,
    "❄️ Desmobilizado (Rebaixado)": 0.85,
    "💪 Super Favorito": 1.25,
    "🚑 Crise": 0.80
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

# CÁLCULOS
def calculate_metrics(home, away, ref_factor, is_champions, fact_h, fact_a):
    h_data = teams_data[home]
    a_data = teams_data[away]
    
    # Escanteios
    corn_h = (h_data['corners'] * 1.10) * fact_h
    corn_a = (a_data['corners'] * 0.85) * fact_a
    if is_champions: 
        corn_h *= 0.85
        corn_a *= 0.85
    total_corners = corn_h + corn_a
        
    # Cartões
    tension_boost = 1.10 if fact_h > 1.0 or fact_a > 1.0 else 1.0
    tension = ((h_data['fouls'] + a_data['fouls']) / 24.0) * tension_boost
    tension = max(0.85, min(tension, 1.40))
    
    card_h = h_data['cards'] * tension * ref_factor
    card_a = a_data['cards'] * tension * ref_factor
    total_cards = card_h + card_a
    
    # Gols
    avg_l = 1.3
    exp_h = ((h_data['goals_f'] * fact_h)/avg_l) * (a_data['goals_a']/avg_l) * avg_l
    exp_a = ((a_data['goals_f'] * fact_a)/avg_l) * (h_data['goals_a']/avg_l) * avg_l
    
    return {'total_corners': total_corners, 'ind_corn_h': corn_h, 'ind_corn_a': corn_a,
            'total_cards': total_cards, 'ind_card_h': card_h, 'ind_card_a': card_a,
            'goals_h': exp_h, 'goals_a': exp_a, 'tension': tension}

def prob_over(exp, line): return poisson.sf(int(line), exp) * 100

# INTERFACE PRINCIPAL
st.title("⚽ FutPrevisão Pro")

tab_analise, tab_scanner = st.tabs(["📊 Análise do Jogo", "🔍 Scanner de Oportunidades"])

# --- ABA 1: ANÁLISE DO JOGO ---
with tab_analise:
    st.markdown(f"### {home_team} x {away_team}")
    
    if st.sidebar.button("Gerar Previsões 🚀", type="primary"):
        m = calculate_metrics(home_team, away_team, ref_factor, champions_mode, f_h, f_a)
        
        # Cabeçalho
        c1, c2, c3 = st.columns(3)
        c1.metric("Escanteios", f"{m['total_corners']:.2f}")
        c2.metric("Cartões", f"{m['total_cards']:.2f}")
        c3.metric("Tensão", f"{m['tension']:.2f}")
        st.divider()

        # Escanteios
        st.subheader("🚩 Escanteios")
        cols = st.columns(6)
        for i, line in enumerate([7.5, 8.5, 9.5, 10.5, 11.5, 12.5]):
            p = prob_over(m['total_corners'], line)
            c = "green" if p >= 70 else "orange" if p >= 50 else "red"
            cols[i].markdown(f"**+{line}**\n:{c}[**{p:.1f}%**]")

        col_h, col_m, col_a = st.columns([1, 1, 1])
        with col_h:
            st.markdown(f"**🏠 {home_team}**")
            st.write(f"M: **{m['ind_corn_h']:.2f}**")
            for l in [2.5, 3.5, 4.5, 5.5]: st.write(f"+{l}: **{prob_over(m['ind_corn_h'], l):.1f}%**")
        with col_a:
            st.markdown(f"**✈️ {away_team}**")
            st.write(f"M: **{m['ind_corn_a']:.2f}**")
            for l in [2.5, 3.5, 4.5, 5.5]: st.write(f"+{l}: **{prob_over(m['ind_corn_a'], l):.1f}%**")
        with col_m:
            fig = go.Figure(data=[go.Bar(x=[home_team, away_team], y=[m['ind_corn_h'], m['ind_corn_a']], marker_color=['blue', 'red'])])
            fig.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("🟨 Cartões")
        ch, cm, ca = st.columns([1, 1, 1])
        with ch:
            st.markdown(f"**🏠 {home_team}**")
            st.write(f"M: **{m['ind_card_h']:.2f}**")
            for l in [1.5, 2.5, 3.5]: st.markdown(f"+{l}: **{prob_over(m['ind_card_h'], l):.1f}%**")
        with ca:
            st.markdown(f"**✈️ {away_team}**")
            st.write(f"M: **{m['ind_card_a']:.2f}**")
            for l in [1.5, 2.5, 3.5]: st.markdown(f"+{l}: **{prob_over(m['ind_card_a'], l):.1f}%**")
        with cm:
            figc = go.Figure(data=[go.Bar(x=[home_team, away_team], y=[m['ind_card_h'], m['ind_card_a']], marker_color=['gold', 'gold'])])
            figc.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(figc, use_container_width=True)

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
        
        with st.expander("Placar Exato"):
            matrix = [[ph[i]*pa[j]*100 for j in range(5)] for i in range(5)]
            fig2 = go.Figure(data=go.Heatmap(z=matrix, x=[f"{away_team} {j}" for j in range(5)], y=[f"{home_team} {i}" for i in range(5)], colorscale='Viridis', texttemplate="%{z:.1f}%"))
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("👈 Configure o jogo na barra lateral e clique em 'Gerar Previsões'.")

# --- ABA 2: SCANNER ---
with tab_scanner:
    st.subheader("🕵️‍♂️ Scanner de Oportunidades")
    st.markdown("Os melhores times da base de dados para cada mercado.")
    
    df_rank = pd.DataFrame.from_dict(teams_data, orient='index')
    
    col_s1, col_s2, col_s3 = st.columns(3)
    
    with col_s1:
        st.markdown("### 🚩 Máquinas de Cantos")
        top_cantos = df_rank.sort_values(by='corners', ascending=False).head(10)
        st.dataframe(
            top_cantos[['corners']],
            column_config={"corners": st.column_config.ProgressColumn("Média", format="%.2f", min_value=0, max_value=12)},
            use_container_width=True
        )

    with col_s2:
        st.markdown("### 🟨 Mais Violentos")
        top_cards = df_rank.sort_values(by='cards', ascending=False).head(10)
        st.dataframe(
            top_cards[['cards', 'fouls']],
            column_config={
                "cards": st.column_config.NumberColumn("Cartões", format="%.2f"),
                "fouls": st.column_config.NumberColumn("Faltas", format="%.1f")
            },
            use_container_width=True
        )

    with col_s3:
        st.markdown("### ⚽ Melhores Ataques")
        top_gols = df_rank.sort_values(by='goals_f', ascending=False).head(10)
        st.dataframe(
            top_gols[['goals_f']],
            column_config={"goals_f": st.column_config.ProgressColumn("Gols Feitos", format="%.2f", min_value=0, max_value=4)},
            use_container_width=True
        )
        
    st.success("💡 Dica: Verifique se algum destes times joga hoje!")