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
    
    /* Jogo Individual */
    .game-container {
        background: white;
        border-left: 4px solid #007bff;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    }
    
    .game-header {
        font-size: 16px;
        font-weight: 700;
        color: #333;
        margin-bottom: 10px;
        padding-bottom: 10px;
        border-bottom: 2px solid #f0f0f0;
    }
    
    .game-status-green {
        background: #28a745;
        color: white;
        padding: 4px 12px;
        border-radius: 15px;
        font-size: 12px;
        font-weight: 600;
    }
    
    .game-status-red {
        background: #dc3545;
        color: white;
        padding: 4px 12px;
        border-radius: 15px;
        font-size: 12px;
        font-weight: 600;
    }
    
    /* Seleção Individual */
    .selection-item {
        background: #f8f9fa;
        padding: 12px;
        margin: 8px 0;
        border-radius: 8px;
        border-left: 3px solid #6c757d;
    }
    
    .selection-correct {
        border-left-color: #28a745;
        background: #e8f5e9;
    }
    
    .selection-wrong {
        border-left-color: #dc3545;
        background: #ffebee;
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
    
    .stop-loss-safe { border-left: 6px solid #28a745; }
    .stop-loss-warning { border-left: 6px solid #ffc107; }
    .stop-loss-danger { border-left: 6px solid #dc3545; }
    
    /* Formulário */
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
    
    /* Badges */
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
# 2. FUNÇÕES DE GESTÃO (INTACTO)
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
    with open(DATA_FILE, "r") as f: dados = json.load(f)
    return sorted(dados, key=lambda x: datetime.strptime(x['Data'], "%d/%m/%Y"), reverse=True)

def salvar_ticket(ticket_data):
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f: dados = json.load(f)
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
# 3. DASHBOARD COM ESTRUTURA DE JOGOS E SELEÇÕES
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
    
    # --- STOP LOSS ---
    hoje = datetime.now().strftime("%d/%m/%Y")
    prejuizo_hoje = sum(t["Lucro"] for t in tickets if t["Data"] == hoje and t["Lucro"] < 0)
    perda_atual = abs(prejuizo_hoje)
    pct_perda = min(perda_atual / cfg["stop_loss"], 1.0) if cfg["stop_loss"] > 0 else 0
    
    # === HEADER KPIs ===
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
    
    # === STOP LOSS ===
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
        
        # Inicializar session_state
        if 'num_jogos' not in st.session_state:
            st.session_state['num_jogos'] = 1
        
        for j in range(st.session_state['num_jogos']):
            key = f'num_selecoes_jogo_{j}'
            if key not in st.session_state:
                st.session_state[key] = 1
        
        # Inputs principais
        col1, col2, col3 = st.columns(3)
        
        with col1:
            data_bilhete = st.date_input("📅 Data", datetime.now(), key="data_bilhete")
            
        with col2:
            stake = st.number_input("💵 Stake (R$)", min_value=0.0, step=5.0, value=10.0, key="stake_bilhete")
            
        with col3:
            odd = st.number_input("📊 Odd Total", min_value=1.00, step=0.01, value=1.50, key="odd_bilhete")
        
        resultado_bilhete = st.selectbox(
            "🎯 Resultado Final", 
            ["Green ✅", "Green (Cashout) 💰", "Red ❌", "Reembolso 🔄"],
            key="resultado_bilhete"
        )
        
        valor_retornado_manual = 0.0
        if "Cashout" in resultado_bilhete:
            valor_retornado_manual = st.number_input(
                "💰 Valor do Cashout (R$)", 
                min_value=0.0, 
                step=5.0,
                key="cashout_valor"
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
        st.markdown("### 🎲 JOGOS")
        
        jogos_data = []
        
        for jogo_idx in range(st.session_state['num_jogos']):
            st.markdown(f"#### 🎲 JOGO {jogo_idx + 1}")
            
            with st.container(border=True):
                col_h, col_sep, col_a = st.columns([5, 1, 5])
                
                with col_h:
                    mandante = st.selectbox(
                        "🏠 Mandante", 
                        team_list_with_empty, 
                        key=f"home_{jogo_idx}"
                    )
                    
                with col_sep:
                    st.markdown("<h3 style='text-align: center; padding-top: 20px;'>×</h3>", unsafe_allow_html=True)
                    
                with col_a:
                    visitante = st.selectbox(
                        "✈️ Visitante", 
                        team_list_with_empty, 
                        key=f"away_{jogo_idx}"
                    )
                
                st.markdown("**📋 Seleções deste jogo:**")
                
                selecoes_jogo = []
                num_selecoes_key = f'num_selecoes_jogo_{jogo_idx}'
                num_selecoes = st.session_state[num_selecoes_key]
                
                for sel_idx in range(num_selecoes):
                    st.markdown(f"*Seleção {sel_idx + 1}:*")
                    
                    col_alvo, col_mercado, col_status = st.columns([2, 3, 1])
                    
                    with col_alvo:
                        alvo = st.selectbox(
                            "🎯 Alvo",
                            ["🟢 Mandante", "🔴 Visitante", "⚪ Geral"],
                            key=f"alvo_j{jogo_idx}_s{sel_idx}"
                        )
                        
                    with col_mercado:
                        mercado = st.selectbox(
                            "📊 Mercado",
                            MERCADOS_LISTA,
                            key=f"mercado_j{jogo_idx}_s{sel_idx}"
                        )
                    
                    with col_status:
                        st.write("")
                        st.write("")
                        acertou = st.checkbox(
                            "✅ Acertou",
                            value=True,
                            key=f"status_j{jogo_idx}_s{sel_idx}"
                        )
                    
                    icon = "⚽"
                    if "Escanteios" in mercado:
                        icon = "🚩"
                    elif "Cartões" in mercado:
                        icon = "🟨"
                    elif "Gols" in mercado:
                        icon = "⚽"
                    elif "Ambas Marcam" in mercado:
                        icon = "🎯"
                    
                    selecoes_jogo.append({
                        "Alvo": alvo,
                        "Mercado": mercado,
                        "Icon": icon,
                        "Status": "Acertou" if acertou else "Errou"
                    })
                    
                    if sel_idx < num_selecoes - 1:
                        st.markdown("---")
                
                todas_acertaram = all(sel["Status"] == "Acertou" for sel in selecoes_jogo)
                status_jogo = "Green" if todas_acertaram else "Red"
                
                nome_jogo = f"{mandante} × {visitante}" if mandante and visitante else f"Jogo {jogo_idx + 1}"
                
                jogos_data.append({
                    "Mandante": mandante,
                    "Visitante": visitante,
                    "Nome": nome_jogo,
                    "Status_Jogo": status_jogo,
                    "Selecoes": selecoes_jogo
                })
        
        st.divider()
        
        # Botões de controle
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        
        with col_btn1:
            if st.button("➕ Adicionar Jogo", disabled=(st.session_state['num_jogos'] >= 5)):
                if st.session_state['num_jogos'] < 5:
                    st.session_state['num_jogos'] += 1
                    st.rerun()
        
        with col_btn2:
            ultimo_jogo = st.session_state['num_jogos'] - 1
            key_ultimo = f'num_selecoes_jogo_{ultimo_jogo}'
            num_sel_ultimo = st.session_state[key_ultimo]
            
            if st.button("➕ Adicionar Seleção", disabled=(num_sel_ultimo >= 3)):
                if num_sel_ultimo < 3:
                    st.session_state[key_ultimo] += 1
                    st.rerun()
        
        with col_btn3:
            if st.button("🔄 Resetar"):
                st.session_state['num_jogos'] = 1
                for j in range(10):
                    if f'num_selecoes_jogo_{j}' in st.session_state:
                        del st.session_state[f'num_selecoes_jogo_{j}']
                st.rerun()
        
        st.caption(f"Jogos: {st.session_state['num_jogos']}/5 | Último jogo: {st.session_state[f'num_selecoes_jogo_{st.session_state['num_jogos']-1}']}/3 seleções")
        
        st.divider()
        
        # Botão de Salvar
        col_check, col_btn_save = st.columns([3, 1])
        
        with col_check:
            checklist = st.checkbox(
                "✅ Confirmação: Análise feita e gestão de banca respeitada",
                value=False,
                key="checklist_confirmacao"
            )
        
        with col_btn_save:
            if st.button("💾 SALVAR", type="primary", disabled=not checklist, use_container_width=True):
                if stake <= 0:
                    st.error("❌ Stake deve ser maior que zero!")
                elif not any(j['Mandante'] and j['Visitante'] for j in jogos_data):
                    st.error("❌ Preencha ao menos um jogo completo!")
                elif not any(sel['Mercado'] != "Selecione o mercado..." for j in jogos_data for sel in j['Selecoes']):
                    st.error("❌ Selecione ao menos um mercado válido!")
                else:
                    novo_ticket = {
                        "Data": data_bilhete.strftime("%d/%m/%Y"),
                        "Resultado": resultado_bilhete,
                        "Stake": stake,
                        "Odd": odd,
                        "Lucro": lucro_final,
                        "Jogos": jogos_data
                    }
                    salvar_ticket(novo_ticket)
                    st.success("✅ Bilhete registrado!")
                    st.balloons()
                    
                    st.session_state['num_jogos'] = 1
                    for j in range(10):
                        if f'num_selecoes_jogo_{j}' in st.session_state:
                            del st.session_state[f'num_selecoes_jogo_{j}']
                    
                    st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ============== ABA 2: HISTÓRICO ==============
    with tab_history:
        st.markdown("### 📜 Histórico de Bilhetes")
        
        if not tickets:
            st.info("📭 Nenhum bilhete registrado ainda.")
        else:
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
            
            tickets_filtrados = tickets.copy()
            
            if filtro_resultado:
                tickets_filtrados = [t for t in tickets_filtrados if t["Resultado"] in filtro_resultado]
            
            if filtro_periodo == "Hoje":
                tickets_filtrados = [t for t in tickets_filtrados if t["Data"] == hoje]
            elif filtro_periodo == "Esta Semana":
                tickets_filtrados = tickets_filtrados[:7] if len(tickets_filtrados) > 7 else tickets_filtrados
            elif filtro_periodo == "Este Mês":
                tickets_filtrados = tickets_filtrados[:30] if len(tickets_filtrados) > 30 else tickets_filtrados
            
            st.caption(f"Mostrando {len(tickets_filtrados)} bilhete(s)")
            st.divider()
            
            for ticket in tickets_filtrados:
                res = ticket["Resultado"]
                
                if "Green" in res and "Cashout" not in res:
                    status_text = "✅ GANHO"
                elif "Red" in res:
                    status_text = "❌ PERDIDO"
                elif "Cashout" in res:
                    status_text = "💰 CASHOUT"
                else:
                    status_text = "🔄 REEMBOLSO"
                
                if ticket['Lucro'] > 0:
                    retorno = ticket['Lucro'] + ticket['Stake']
                elif "Reembolso" in res:
                    retorno = ticket['Stake']
                else:
                    retorno = 0.0
                
                with st.container(border=True):
                    col_h1, col_h2 = st.columns([3, 1])
                    
                    with col_h1:
                        st.markdown(f"### {status_text} - #{ticket.get('id', '---')}")
                        st.caption(f"📅 {ticket['Data']}")
                    
                    with col_h2:
                        st.metric("Lucro", f"R$ {ticket['Lucro']:+.2f}")
                    
                    col_f1, col_f2, col_f3 = st.columns(3)
                    col_f1.metric("Stake", f"R$ {ticket['Stake']:.2f}")
                    col_f2.metric("Odd", f"{ticket['Odd']:.2f}x")
                    col_f3.metric("Retorno", f"R$ {retorno:.2f}")
                    
                    st.divider()
                    
                    with st.expander(f"👁️ Ver {len(ticket.get('Jogos', []))} Jogo(s)"):
                        for jogo_idx, jogo in enumerate(ticket.get('Jogos', []), 1):
                            status_jogo = jogo.get('Status_Jogo', 'Red')
                            status_icon = "✅" if status_jogo == "Green" else "❌"
                            
                            st.markdown(f"**{status_icon} JOGO {jogo_idx}: {jogo.get('Nome', '')}**")
                            
                            for sel in jogo.get('Selecoes', []):
                                status_sel = sel.get('Status', 'Errou')
                                sel_icon = "✅" if status_sel == "Acertou" else "❌"
                                
                                st.markdown(f"{sel_icon} {sel.get('Icon', '⚽')} {sel.get('Alvo', '')} - {sel.get('Mercado', '')}")
                            
                            if jogo_idx < len(ticket.get('Jogos', [])):
                                st.markdown("---")
                        
                        st.divider()
                        
                        if st.button(f"🗑️ Excluir #{ticket.get('id')}", key=f"del_{ticket.get('id')}"):
                            if excluir_ticket(ticket.get('id')):
                                st.success("✅ Excluído!")
                                st.rerun()
    
    # ============== ABA 3: ESTATÍSTICAS ==============
    with tab_stats:
        if not tickets:
            st.info("📊 Sem dados.")
        else:
            st.markdown("### 📈 Análise Avançada")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Bilhetes", len(tickets))
            col2.metric("Média Stake", f"R$ {np.mean([t['Stake'] for t in tickets]):.2f}")
            col3.metric("Média Odd", f"{np.mean([t['Odd'] for t in tickets]):.2f}")
            col4.metric("Maior Odd", f"{max([t['Odd'] for t in tickets]):.2f}")
            
            st.divider()
            st.subheader("📈 Evolução da Banca")
            
            df_hist = pd.DataFrame(tickets)
            df_chart = df_hist.iloc[::-1].copy()
            df_chart['Lucro_Acumulado'] = df_chart['Lucro'].cumsum() + cfg["banca_inicial"]
            df_chart['Data_Num'] = range(len(df_chart))
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_chart['Data_Num'],
                y=df_chart['Lucro_Acumulado'],
                mode='lines+markers',
                name='Banca',
                line=dict(color='#667eea', width=3),
                fill='tozeroy',
                fillcolor='rgba(102, 126, 234, 0.1)'
            ))
            
            fig.add_hline(
                y=cfg["banca_inicial"],
                line_dash="dash",
                line_color="gray",
                annotation_text="Banca Inicial"
            )
            
            fig.update_layout(xaxis_title="Bilhetes", yaxis_title="Banca (R$)", height=400)
            st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# 4. NAVEGAÇÃO E PREVISÕES (INTACTO)
# ==============================================================================
st.sidebar.markdown("---")
pagina = st.sidebar.radio("Menu", ["🏠 Previsões IA", "📊 Gestão de Banca"])

if pagina == "📊 Gestão de Banca":
    render_dashboard()
    st.stop()

# CÓDIGO DE PREVISÕES (INTACTO)
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
