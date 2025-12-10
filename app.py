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

# CREDENCIAIS CONFIGURADAS DIRETAMENTE
USERS = {
    "diego": "@Casa612"
}

def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    def password_entered():
        user = st.session_state["username"]
        password = st.session_state["password"]
        
        # Verifica direto no dicionário USERS
        if user in USERS and password == USERS[user]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False
            st.error("😕 Usuário ou senha incorretos")

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
    referees[' Estilo: Normal (Padrão)'] = 1
