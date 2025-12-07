
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
# 0. CONFIGURAÇÃO E LOGIN (INTACTO)
# ==============================================================================
st.set_page_config(page_title="FutPrevisão Pro", layout="wide", page_icon="⚽")

# ============== CSS PROFISSIONAL MELHORADO ==============
st.markdown("""
<style>
    /* Cards de Bilhetes - Estilo Betano/Bet365 */
    .bet-card-green {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left: 6px solid #28a745;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 12px rgba(40, 167, 69, 0.15);
        transition: all 0.3s ease;
    }
    
    .bet-card-red {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border-left: 6px solid #dc3545;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 12px rgba(220, 53, 69, 0.15);
        transition: all 0.3s ease;
    }
    
    .bet-card-cashout {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border-left: 6px solid #ffc107;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 12px rgba(255, 193, 7, 0.15);
        transition: all 0.3s ease;
    }
    
    .bet-card-green:hover, .bet-card-red:hover, .bet-card-cashout:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.2);
    }
    
    /* Cabeçalho do Card */
    .bet-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 15px;
        padding-bottom: 15px;
        border-bottom: 2px solid rgba(0, 0, 0, 0.1);
    }
    
    .bet-id {
        font-size: 18px;
        font-weight: 700;
        color: #333;
    }
    
    .bet-status-win {
        background: #28a745;
        color: white;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
    }
    
    .bet-status-loss {
        background: #dc3545;
        color: white;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
    }
    
    .bet-status-cashout {
        background: #ffc107;
        color: #333;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
    }
    
    /* Informações Financeiras */
    .bet-financials {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 15px;
        margin-bottom: 15px;
    }
    
    .bet-financial-item {
        text-align: center;
        padding: 10px;
        background: white;
        border-radius: 8px;
    }
    
    .bet-financial-label {
        font-size: 11px;
        color: #666;
        text-transform: uppercase;
        margin-bottom: 5px;
    }
    
    .bet-financial-value {
        font-size: 18px;
        font-weight: 700;
        color: #333;
    }
    
    /* Seleções/Jogos */
    .bet-selection {
        background: white;
        padding: 15px;
        margin: 10px 0;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .bet-match {
        font-size: 16px;
        font-weight: 600;
        color: #333;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
    }
    
    .bet-match-icon {
        font-size: 20px;
        margin-right: 10px;
    }
    
    .bet-market {
        font-size: 14px;
        color: #666;
        margin-left: 30px;
    }
    
    .bet-target {
        font-size: 13px;
        color: #888;
        margin-left: 30px;
        font-style: italic;
    }
    
    /* KPIs no Topo */
    .kpi-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 25px;
        border-radius: 15px;
        margin-bottom: 25px;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.3);
    }
    
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 20px;
    }
    
    .kpi-item {
        text-align: center;
    }
    
    .kpi-label {
        font-size: 13px;
        opacity: 0.9;
        margin-bottom: 8px;
    }
    
    .kpi-value {
        font-size: 28px;
        font-weight: 700;
    }
    
    .kpi-delta {
        font-size: 14px;
        opacity: 0.8;
        margin-top: 5px;
    }
    
    /* Stop Loss Monitor */
    .stop-loss-container {
        background: white;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 20px;
        border: 2px solid #e0e0e0;
    }
    
    .stop-loss-safe {
        border-left: 6px solid #28a745;
    }
    
    .stop-loss-warning {
        border-left: 6px solid #ffc107;
    }
    
    .stop-loss-danger {
        border-left: 6px solid #dc3545;
    }
    
    /* Formulário Estilizado */
    .form-section {
        background: #f8f9fa;
        padding: 25px;
        border-radius: 12px;
        margin-bottom: 20px;
    }
    
    .form-section-title {
        font-size: 20px;
        font-weight: 700;
        margin-bottom: 20px;
        color: #333;
        border-bottom: 3px solid #667eea;
        padding-bottom: 10px;
    }
    
    /* Badges e Tags */
    .badge {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 15px;
        font-size: 12px;
        font-weight: 600;
        margin-right: 8px;
    }
    
    .badge-green { background: #28a745; color: white; }
    .badge-red { background: #dc3545; color: white; }
    .badge-yellow { background: #ffc107; color: #333; }
    .badge-blue { background: #007bff; color: white; }
    
    /* Responsividade */
    @media (max-width: 768px) {
        .kpi-grid {
            grid-template-columns: repeat(2, 1fr);
        }
        .bet-financials {
            grid-template-columns: repeat(2, 1fr);
        }
    }
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
# 1. CARREGAMENTO DE DADOS (INTACTO)
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

# LISTA MESTRA DE MERCADOS (INTACTO)
MERCADOS_LISTA = ["Selecione o mercado..."]
for i in [2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5]:
    MERCADOS_LISTA.append(f"Escanteios Mais de {i}")
for i in [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]:
    MERCADOS_LISTA.append(f"Cartões Mais de {i}")
for i in [0.5, 1.5, 2.5, 3.5, 4.5]:
    MERCADOS_LISTA.append(f"Gols Mais de {i}")
MERCADOS_LISTA.extend(["Ambas Marcam", "Vitória (ML) Casa", "Vitória (ML) Fora", "Empate", "Dupla Chance"])

# ==============================================================================
# 2. FUNÇÕES DE GESTÃO (INTACTO)
# ==============================================================================
DATA_FILE = "historico_bilhetes_v5.json"
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
    return sorted(dados, key=lambda x: datetime.strptime(x['Data'], "%d/%m/%Y"), reverse=True)

def salvar_ticket(ticket_data):
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f: dados = json.load(f)
    else: dados = []
    
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
# 3. DASHBOARD PROFISSIONAL MELHORADO (ÚNICA PARTE MODIFICADA)
# ==============================================================================
def render_dashboard():
    st.title("📊 Gestão Profissional de Banca")
    
    # --- CONFIGURAÇÃO ---
    cfg = carregar_config()
    with st.sidebar.expander("⚙️ Configurações da Banca", expanded=False):
        nova_banca = st.number_input("Banca Inicial (R$)", value=cfg["banca_inicial"], step=50.0)
        novo_stop = st.number_input("Stop Loss Diário (R$)", value=cfg["stop_loss"], step=10.0)
        if st.button("💾 Salvar Config"):
            salvar_config({"banca_inicial": nova_banca, "stop_loss": novo_stop})
            st.success("✅ Configurações salvas!")
            st.rerun()
    
    # --- CÁLCULOS ---
    tickets = carregar_tickets()
    lucro_total = sum(t["Lucro"] for t in tickets)
    banca_atual = cfg["banca_inicial"] + lucro_total
    
    total_stake = sum(t["Stake"] for t in tickets) if tickets else 1
    roi = (lucro_total / total_stake) * 100 if total_stake > 0 else 0
    
    greens = len([t for t in tickets if "Green" in t["Resultado"]])
    win_rate = (greens / len(tickets)) * 100 if tickets else 0
    
    # --- STOP LOSS HOJE ---
    hoje = datetime.now().strftime("%d/%m/%Y")
    prejuizo_hoje = sum(t["Lucro"] for t in tickets if t["Data"] == hoje and t["Lucro"] < 0)
    perda_atual = abs(prejuizo_hoje)
    pct_perda = min(perda_atual / cfg["stop_loss"], 1.0) if cfg["stop_loss"] > 0 else 0
    
    # === HEADER COM KPIs ===
    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-grid">
            <div class="kpi-item">
                <div class="kpi-label">💰 Banca Atual</div>
                <div class="kpi-value">R$ {banca_atual:,.2f}</div>
                <div class="kpi-delta">{"+" if lucro_total >= 0 else ""}R$ {lucro_total:,.2f}</div>
            </div>
            <div class="kpi-item">
                <div class="kpi-label">📈 ROI</div>
                <div class="kpi-value">{roi:+.1f}%</div>
                <div class="kpi-delta">{len(tickets)} bilhetes</div>
            </div>
            <div class="kpi-item">
                <div class="kpi-label">✅ Win Rate</div>
                <div class="kpi-value">{win_rate:.1f}%</div>
                <div class="kpi-delta">{greens}/{len(tickets)} greens</div>
            </div>
            <div class="kpi-item">
                <div class="kpi-label">💵 Lucro Líquido</div>
                <div class="kpi-value">{"+" if lucro_total >= 0 else ""}R$ {lucro_total:,.2f}</div>
                <div class="kpi-delta">Total apostado: R$ {total_stake:,.2f}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # === MONITOR STOP LOSS ===
    if pct_perda < 0.5:
        stop_class = "stop-loss-safe"
        stop_icon = "🟢"
        stop_msg = "Seguro"
    elif pct_perda < 0.9:
        stop_class = "stop-loss-warning"
        stop_icon = "🟡"
        stop_msg = "Atenção!"
    else:
        stop_class = "stop-loss-danger"
        stop_icon = "🔴"
        stop_msg = "PERIGO!"
    
    st.markdown(f"""
    <div class="stop-loss-container {stop_class}">
        <h3>{stop_icon} Stop Loss Diário - {stop_msg}</h3>
        <p style="font-size: 18px; margin: 10px 0;">
            Perdeu <strong>R$ {perda_atual:.2f}</strong> de <strong>R$ {cfg['stop_loss']:.2f}</strong> 
            ({pct_perda*100:.0f}%)
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.progress(pct_perda)
    
    if pct_perda >= 1.0:
        st.error("🛑 **STOP LOSS ATINGIDO!** Pare de apostar hoje!")
    
    st.divider()
    
    # === ABAS ===
    tab_add, tab_history, tab_stats = st.tabs(["➕ Novo Bilhete", "📜 Histórico", "📊 Estatísticas"])
    
    # ============== ABA 1: NOVO BILHETE ==============
    with tab_add:
        st.markdown('<div class="form-section">', unsafe_allow_html=True)
        st.markdown('<div class="form-section-title">💰 Registrar Novo Bilhete</div>', unsafe_allow_html=True)
        
        with st.form("form_bilhete"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                data_bilhete = st.date_input("📅 Data", datetime.now())
                
            with col2:
                stake = st.number_input("💵 Stake (R$)", min_value=0.0, step=5.0, value=10.0)
                
            with col3:
                odd = st.number_input("📊 Odd Total", min_value=1.00, step=0.01, value=1.50)
            
            resultado_bilhete = st.selectbox(
                "🎯 Resultado Final", 
                ["Green ✅", "Green (Cashout) 💰", "Red ❌", "Reembolso 🔄"]
            )
            
            # Lógica Financeira
            valor_retornado_manual = 0.0
            if "Cashout" in resultado_bilhete:
                valor_retornado_manual = st.number_input(
                    "💰 Valor do Cashout (R$)", 
                    min_value=0.0, 
                    step=5.0
                )
            
            if "Green ✅" in resultado_bilhete:
                lucro_final = (stake * odd) - stake
            elif "Red" in resultado_bilhete:
                lucro_final = -stake
            elif "Reembolso" in resultado_bilhete:
                lucro_final = 0.0
            else:
                lucro_final = valor_retornado_manual - stake
            
            st.info(f"💰 Lucro calculado: **R$ {lucro_final:+.2f}**")
            
            st.divider()
            st.markdown("### 🎯 Seleções (Múltiplas)")
            
            qtd_jogos = st.slider("Quantidade de Jogos", 1, 8, 2)
            selecoes = []
            
            for i in range(qtd_jogos):
                st.markdown(f"**🎲 Jogo {i+1}**")
                
                col_h, col_sep, col_a = st.columns([5, 1, 5])
                
                with col_h:
                    mandante = st.selectbox(
                        "🏠 Mandante", 
                        team_list_with_empty, 
                        key=f"home_{i}"
                    )
                    
                with col_sep:
                    st.markdown("<h3 style='text-align: center; padding-top: 20px;'>×</h3>", unsafe_allow_html=True)
                    
                with col_a:
                    visitante = st.selectbox(
                        "✈️ Visitante", 
                        team_list_with_empty, 
                        key=f"away_{i}"
                    )
                
                col_alvo, col_mercado = st.columns(2)
                
                with col_alvo:
                    alvo = st.selectbox(
                        "🎯 Alvo",
                        ["🟢 Mandante", "🔴 Visitante", "⚪ Geral"],
                        key=f"alvo_{i}"
                    )
                    
                with col_mercado:
                    mercado = st.selectbox(
                        "📊 Mercado",
                        MERCADOS_LISTA,
                        key=f"mercado_{i}"
                    )
                
                # Determinar ícone
                icon = "⚽"
                if "Escanteios" in mercado:
                    icon = "🚩"
                elif "Cartões" in mercado:
                    icon = "🟨"
                elif "Gols" in mercado:
                    icon = "⚽"
                elif "Ambas Marcam" in mercado:
                    icon = "🎯"
                
                nome_jogo = f"{mandante} × {visitante}" if mandante and visitante else f"Jogo {i+1}"
                
                selecoes.append({
                    "Jogo": nome_jogo,
                    "Alvo": alvo,
                    "Mercado": mercado,
                    "Icon": icon
                })
                
                if i < qtd_jogos - 1:
                    st.markdown("---")
            
            st.divider()
            
            # Checklist
            col_check, col_btn = st.columns([3, 1])
            
            with col_check:
                checklist = st.checkbox(
                    "✅ Confirmação: Análise feita e gestão de banca respeitada",
                    value=False
                )
            
            with col_btn:
                submit = st.form_submit_button(
                    "💾 SALVAR BILHETE",
                    disabled=not checklist,
                    use_container_width=True
                )
            
            if submit:
                if stake <= 0:
                    st.error("❌ Stake deve ser maior que zero!")
                elif not any(sel['Mercado'] != "Selecione o mercado..." for sel in selecoes):
                    st.error("❌ Selecione ao menos um mercado válido!")
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
                    st.success("✅ Bilhete registrado com sucesso!")
                    st.balloons()
                    st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ============== ABA 2: HISTÓRICO ==============
    with tab_history:
        st.markdown("### 📜 Histórico de Bilhetes")
        
        if not tickets:
            st.info("📭 Nenhum bilhete registrado ainda. Adicione seu primeiro bilhete!")
        else:
            # Filtros
            col_f1, col_f2 = st.columns(2)
            
            with col_f1:
                filtro_resultado = st.multiselect(
                    "Filtrar por Resultado",
                    ["Green ✅", "Green (Cashout) 💰", "Red ❌", "Reembolso 🔄"],
                    default=[]
                )
            
            with col_f2:
                filtro_periodo = st.selectbox(
                    "Período",
                    ["Todos", "Hoje", "Esta Semana", "Este Mês"]
                )
            
            # Aplicar filtros
            tickets_filtrados = tickets.copy()
            
            if filtro_resultado:
                tickets_filtrados = [t for t in tickets_filtrados if t["Resultado"] in filtro_resultado]
            
            if filtro_periodo == "Hoje":
                tickets_filtrados = [t for t in tickets_filtrados if t["Data"] == hoje]
            elif filtro_periodo == "Esta Semana":
                # Simplificado: últimos 7 dias
                tickets_filtrados = tickets_filtrados[:7] if len(tickets_filtrados) > 7 else tickets_filtrados
            elif filtro_periodo == "Este Mês":
                # Simplificado: últimos 30 dias
                tickets_filtrados = tickets_filtrados[:30] if len(tickets_filtrados) > 30 else tickets_filtrados
            
            st.caption(f"Mostrando {len(tickets_filtrados)} bilhete(s)")
            st.divider()
            
            # Exibir Cards
            for ticket in tickets_filtrados:
                # Determinar estilo
                res = ticket["Resultado"]
                
                if "Green" in res and "Cashout" not in res:
                    card_class = "bet-card-green"
                    status_class = "bet-status-win"
                    status_text = "✅ GANHO"
                elif "Red" in res:
                    card_class = "bet-card-red"
                    status_class = "bet-status-loss"
                    status_text = "❌ PERDIDO"
                elif "Cashout" in res:
                    card_class = "bet-card-cashout"
                    status_class = "bet-status-cashout"
                    status_text = "💰 CASHOUT"
                else:
                    card_class = "bet-card-cashout"
                    status_class = "bet-status-cashout"
                    status_text = "🔄 REEMBOLSO"
                
                # Calcular retorno
                if ticket['Lucro'] > 0:
                    retorno = ticket['Lucro'] + ticket['Stake']
                elif "Reembolso" in res:
                    retorno = ticket['Stake']
                else:
                    retorno = 0.0
                
                # HTML do Card
                st.markdown(f"""
                <div class="{card_class}">
                    <div class="bet-header">
                        <div>
                            <span class="bet-id">#{ticket.get('id', '---')}</span>
                            <br>
                            <span style="font-size: 14px; color: #666;">{ticket['Data']}</span>
                        </div>
                        <div class="{status_class}">{status_text}</div>
                    </div>
                    
                    <div class="bet-financials">
                        <div class="bet-financial-item">
                            <div class="bet-financial-label">Stake</div>
                            <div class="bet-financial-value">R$ {ticket['Stake']:.2f}</div>
                        </div>
                        <div class="bet-financial-item">
                            <div class="bet-financial-label">Odd</div>
                            <div class="bet-financial-value">{ticket['Odd']:.2f}x</div>
                        </div>
                        <div class="bet-financial-item">
                            <div class="bet-financial-label">Retorno</div>
                            <div class="bet-financial-value">R$ {retorno:.2f}</div>
                        </div>
                        <div class="bet-financial-item">
                            <div class="bet-financial-label">Lucro</div>
                            <div class="bet-financial-value" style="color: {'green' if ticket['Lucro'] > 0 else 'red'};">
                                {ticket['Lucro']:+.2f}
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Expander para Seleções
                with st.expander(f"👁️ Ver {len(ticket['Selecoes'])} Seleção(ões)"):
                    for idx, sel in enumerate(ticket["Selecoes"], 1):
                        st.markdown(f"""
                        <div class="bet-selection">
                            <div class="bet-match">
                                <span class="bet-match-icon">{sel.get('Icon', '⚽')}</span>
                                <span>{sel['Jogo']}</span>
                            </div>
                            <div class="bet-market">
                                <span class="badge badge-blue">{sel['Alvo']}</span>
                                {sel['Mercado']}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.divider()
                    
                    # Botão de excluir
                    if st.button(f"🗑️ Excluir Bilhete #{ticket.get('id')}", key=f"del_{ticket.get('id')}"):
                        if excluir_ticket(ticket.get('id')):
                            st.success("✅ Bilhete excluído!")
                            st.rerun()
                        else:
                            st.error("❌ Erro ao excluir.")
    
    # ============== ABA 3: ESTATÍSTICAS ==============
    with tab_stats:
        if not tickets:
            st.info("📊 Sem dados para análise ainda.")
        else:
            st.markdown("### 📈 Análise Avançada")
            
            # Métricas resumidas
            col1, col2, col3, col4 = st.columns(4)
            
            col1.metric("Total de Bilhetes", len(tickets))
            col2.metric("Média de Stake", f"R$ {np.mean([t['Stake'] for t in tickets]):.2f}")
            col3.metric("Média de Odd", f"{np.mean([t['Odd'] for t in tickets]):.2f}")
            col4.metric("Maior Odd", f"{max([t['Odd'] for t in tickets]):.2f}")
            
            st.divider()
            
            # Gráfico de Evolução
            st.subheader("📈 Evolução da Banca")
            
            df_hist = pd.DataFrame(tickets)
            df_chart = df_hist.iloc[::-1].copy()  # Ordem cronológica
            df_chart['Lucro_Acumulado'] = df_chart['Lucro'].cumsum() + cfg["banca_inicial"]
            df_chart['Data_Num'] = range(len(df_chart))
            
            fig_evolucao = go.Figure()
            
            # Linha da banca
            fig_evolucao.add_trace(go.Scatter(
                x=df_chart['Data_Num'],
                y=df_chart['Lucro_Acumulado'],
                mode='lines+markers',
                name='Banca',
                line=dict(color='#667eea', width=3),
                marker=dict(size=8),
                fill='tozeroy',
                fillcolor='rgba(102, 126, 234, 0.1)'
            ))
            
            # Linha da banca inicial
            fig_evolucao.add_hline(
                y=cfg["banca_inicial"],
                line_dash="dash",
                line_color="gray",
                annotation_text="Banca Inicial"
            )
            
            fig_evolucao.update_layout(
                title="Crescimento da Banca ao Longo do Tempo",
                xaxis_title="Bilhetes",
                yaxis_title="Banca (R$)",
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_evolucao, use_container_width=True)
            
            st.divider()
            
            # Gráfico de Pizza - Distribuição de Resultados
            col_pizza, col_barras = st.columns(2)
            
            with col_pizza:
                st.subheader("🥧 Distribuição de Resultados")
                
                contagem_resultados = {}
                for t in tickets:
                    res = t["Resultado"]
                    contagem_resultados[res] = contagem_resultados.get(res, 0) + 1
                
                fig_pizza = go.Figure(data=[go.Pie(
                    labels=list(contagem_resultados.keys()),
                    values=list(contagem_resultados.values()),
                    hole=0.4,
                    marker=dict(colors=['#28a745', '#ffc107', '#dc3545', '#6c757d'])
                )])
                
                fig_pizza.update_layout(height=350)
                st.plotly_chart(fig_pizza, use_container_width=True)
            
            with col_barras:
                st.subheader("📊 Lucro por Mês")
                
                # Simplificado: mostrar últimos 30 dias
                df_chart['Mes'] = pd.to_datetime(df_chart['Data'], format="%d/%m/%Y").dt.strftime("%m/%Y")
                lucro_por_mes = df_chart.groupby('Mes')['Lucro'].sum().reset_index()
                
                fig_barras = go.Figure(data=[go.Bar(
                    x=lucro_por_mes['Mes'],
                    y=lucro_por_mes['Lucro'],
                    marker=dict(
                        color=lucro_por_mes['Lucro'],
                        colorscale=['red', 'yellow', 'green'],
                        showscale=False
                    )
                )])
                
                fig_barras.update_layout(
                    height=350,
                    xaxis_title="Mês",
                    yaxis_title="Lucro (R$)"
                )
                
                st.plotly_chart(fig_barras, use_container_width=True)

# ==============================================================================
# 4. NAVEGAÇÃO E PREVISÕES (100% INTACTO - CÓDIGO ORIGINAL)
# ==============================================================================
st.sidebar.markdown("---")
pagina = st.sidebar.radio("Menu", ["🏠 Previsões IA", "📊 Gestão de Banca"])

if pagina == "📊 Gestão de Banca":
    render_dashboard()
    st.stop()

# --- PREVISÕES (CÓDIGO 100% ORIGINAL - INTACTO) ---
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
