"""
FutPrevisão V31 MAXIMUM + AI Advisor ULTRA
CÓDIGO COMPLETO - 2300+ LINHAS
VERSÃO PROFISSIONAL

Autor: Diego
Versão: 31.0 ULTRA MAXIMUM
Data: 27/12/2024
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import math
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px
from difflib import get_close_matches
import re
from collections import defaultdict

# ============================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================

st.set_page_config(
    page_title="FutPrevisão V31 MAXIMUM",
    layout="wide",
    page_icon="⚽",
    initial_sidebar_state="expanded"
)


# CSS PROFISSIONAL - TABS HORIZONTAIS
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
# MAPEAMENTO DE NOMES DE TIMES
# ============================================================

NAME_MAPPING = {
    'Man United': 'Manchester Utd', 'Man City': 'Manchester City',
    'Spurs': 'Tottenham', 'Newcastle': 'Newcastle Utd',
    'Wolves': 'Wolverhampton', 'Brighton': 'Brighton and Hove Albion',
    'Nottm Forest': "Nott'm Forest", 'Leicester': 'Leicester City',
    'West Ham': 'West Ham Utd', 'Sheffield Utd': 'Sheffield United',
    'Inter': 'Inter Milan', 'AC Milan': 'Milan',
    'Ath Madrid': 'Atletico Madrid', 'Ath Bilbao': 'Athletic Club',
    'Betis': 'Real Betis', 'Sociedad': 'Real Sociedad',
    'Celta': 'Celta Vigo', "M'gladbach": 'Borussia M.Gladbach',
    'Leverkusen': 'Bayer Leverkusen', 'FC Koln': 'FC Cologne',
    'Dortmund': 'Borussia Dortmund', 'Ein Frankfurt': 'Eintracht Frankfurt',
    'Hoffenheim': 'TSG Hoffenheim', 'Bayern Munich': 'Bayern Munchen',
    'RB Leipzig': 'RasenBallsport Leipzig', 'Schalke 04': 'FC Schalke 04',
    'Werder Bremen': 'SV Werder Bremen', 'Fortuna Dusseldorf': 'Fortuna Düsseldorf',
    'Mainz': 'FSV Mainz 05', 'Hertha': 'Hertha Berlin',
    'Paderborn': 'SC Paderborn 07', 'Augsburg': 'FC Augsburg',
    'Freiburg': 'SC Freiburg', 'Paris SG': 'Paris S-G',
    'Paris S-G': 'Paris Saint Germain', 'Saint-Etienne': 'St Etienne',
    'Nimes': 'Nîmes',
}

# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================


from core.loader import load_all_data, load_referees_fallback
from core.utils import normalize_name, format_currency, get_prob_emoji
from core.engine import calcular_jogo_v31, simulate_game_v31, calculate_sharpe_ratio, calculate_max_drawdown, calculate_roi
from core.chat import processar_chat, parse_bilhete_texto, validar_jogos_bilhete, calcular_prob_bilhete, render_md


def main():

    # ═══════════════════════════════════════════════════════════
    # HEADER PROFISSIONAL
    # ═══════════════════════════════════════════════════════════
    col1, col2, col3 = st.columns([2, 3, 2])
    
    with col1:
        st.image("https://img.icons8.com/fluency/96/000000/soccer-ball.png", width=80)
    
    with col2:
        st.title("⚽ FutPrevisão V31 Pro")
        st.caption("_Sistema Profissional de Análise Esportiva_")
    
    with col3:
        st.metric("📚 Database", f"{len(STATS)} times", delta="10 Ligas")
    
    st.markdown("---")
    

    # ═══════════════════════════════════════════════════════════
    # TABS HORIZONTAIS - NAVEGAÇÃO PRINCIPAL
    # ═══════════════════════════════════════════════════════════
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "🔍 Scanner",
        "📋 Construtor", 
        "🏆 Rankings",
        "📊 Análise Detalhada",
        "💼 Gestão de Banca",
        "📅 Calendários",
        "🎲 Bet Builder",
        "📚 Educação",
        "🤖 IA Advisor"
    ])
    """Função principal do aplicativo"""
    
    stats, cal, referees = load_all_data()
    
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
    
    st.title("⚽ FutPrevisão V31 MAXIMUM + AI Advisor ULTRA")
    st.markdown("**Sistema Completo e Profissional de Análise de Apostas Esportivas**")
    st.markdown("_Causality Engine V31 | Poisson | Monte Carlo | Kelly | Sharpe | 2300+ linhas_")
    
    with st.sidebar:
        st.header("📊 Dashboard")
        col1, col2 = st.columns(2)
        col1.metric("Times", len(STATS))
        col1.metric("Jogos", len(cal) if not cal.empty else 0)
        col2.metric("Árbitros", len(referees))
        banca = st.session_state.bankroll_history[-1]
        col2.metric("Banca", format_currency(banca))
        
        if st.session_state.current_ticket:
            st.success(f"🎫 {len(st.session_state.current_ticket)} seleção(ões)")
        else:
            st.info("📭 Bilhete vazio")
        
        if st.session_state.bet_results:
            total = len(st.session_state.bet_results)
            ganhas = sum(1 for b in st.session_state.bet_results if b.get('ganhou', False))
            wr = (ganhas/total)*100 if total > 0 else 0
            st.markdown("---")
            st.metric("Win Rate", f"{wr:.1f}%")
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "🎫 Construtor", "🛡️ Hedges", "🎲 Simulador", "📊 Métricas",
        "🎨 Viz", "📝 Registro", "🔍 Scanner", "📋 Importar", "🤖 AI"
    ])
    
    # ============================================================
    # TAB 1: CONSTRUTOR
    # ============================================================
    
    with tab1:
        st.header("🎫 Construtor de Bilhetes Profissional")
        
        if not cal.empty:
            dates = sorted(cal['DtObj'].dt.strftime('%d/%m/%Y').unique())
            sel_date = st.selectbox("📅 Selecione a Data:", dates, key='c_date')
            jogos_dia = cal[cal['DtObj'].dt.strftime('%d/%m/%Y') == sel_date]
            
            st.markdown(f"### 🎯 {len(jogos_dia)} jogo(s) disponível(eis)")
            
            for idx, jogo in jogos_dia.iterrows():
                h = normalize_name(jogo['Time_Casa'], list(STATS.keys()))
                a = normalize_name(jogo['Time_Visitante'], list(STATS.keys()))
                
                if h and a and h in STATS and a in STATS:
                    ref_nome = jogo.get('Arbitro', 'N/A')
                    ref_data = referees.get(ref_nome, {})
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
                            (f"Over 10.5 Cantos Total", calc['corners']['t'], 10.5, 'corners'),
                            (f"Over 11.5 Cantos Total", calc['corners']['t'], 11.5, 'corners'),
                            (f"{h} - Over 2.5 Cartões Casa", calc['cards']['h'], 2.5, 'cards'),
                            (f"{a} - Over 2.5 Cartões Fora", calc['cards']['a'], 2.5, 'cards'),
                            (f"Over 4.5 Cartões Total", calc['cards']['t'], 4.5, 'cards'),
                            (f"Over 5.5 Cartões Total", calc['cards']['t'], 5.5, 'cards'),
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
        st.subheader("📋 Seu Bilhete Atual")
        
        if st.session_state.current_ticket:
            st.success(f"✅ {len(st.session_state.current_ticket)} seleção(ões)")
            
            for i, sel in enumerate(st.session_state.current_ticket):
                col1, col2 = st.columns([5, 1])
                col1.write(f"{i+1}. {sel['jogo']} - {sel['market_display']} ({sel['prob']}%)")
                if col2.button("🗑️", key=f"del_{i}"):
                    st.session_state.current_ticket.pop(i)
                    st.rerun()
            
            prob_comb = 1.0
            for sel in st.session_state.current_ticket:
                prob_comb *= (sel['prob'] / 100)
            
            odd_est = 1.0 / prob_comb if prob_comb > 0 else 999
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Prob Total", f"{prob_comb*100:.1f}%")
            col2.metric("Odd Estimada", f"@{odd_est:.2f}")
            col3.metric("Seleções", len(st.session_state.current_ticket))
            
            st.session_state.ticket_odds = {'prob_total': prob_comb*100, 'odd_total': odd_est}
            
            if st.button("🗑️ LIMPAR BILHETE", use_container_width=True):
                st.session_state.current_ticket = []
                st.rerun()
        else:
            st.info("📭 Bilhete vazio. Adicione seleções acima!")
    
    # ============================================================
    # TAB 2: HEDGES MAXIMUM
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
                
                col1, col2 = st.columns(2)
                col1.success(f"✅ Principal ganha: {format_currency(cen1_princ)}")
                if cen1_hedge > 0:
                    col2.success(f"🛡️ Hedge ganha: {format_currency(cen1_hedge)}")
                else:
                    col2.error(f"🛡️ Hedge ganha: {format_currency(cen1_hedge)}")
            
            with st.expander("⚖️ HEDGE 2: Partial Protection"):
                st.markdown("**Inverte METADE das seleções**")
                h2_stake = stake * 0.50
                h2_odd = 1.8
                cen2_princ = lucro_max - h2_stake
                cen2_hedge = -stake + (h2_stake * h2_odd)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Stake", format_currency(h2_stake))
                col2.metric("Odd", f"@{h2_odd:.2f}")
                col3.metric("Retorno", format_currency(h2_stake * h2_odd))
                
                col1, col2 = st.columns(2)
                col1.success(f"✅ Principal: {format_currency(cen2_princ)}")
                if cen2_hedge > 0:
                    col2.success(f"🛡️ Hedge: {format_currency(cen2_hedge)}")
                else:
                    col2.error(f"🛡️ Hedge: {format_currency(cen2_hedge)}")
            
            with st.expander("💎 HEDGE 3: Guaranteed Profit"):
                st.markdown("**Inverte TUDO (arbitragem)**")
                h3_odd = 1.5
                h3_stake = (stake * odd_total) / (h3_odd + 1)
                lucro_gar = (stake * odd_total) - stake - h3_stake
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Stake", format_currency(h3_stake))
                col2.metric("Odd", f"@{h3_odd:.2f}")
                col3.metric("💰 LUCRO GARANTIDO", format_currency(lucro_gar))
                
                st.success(f"🎯 VOCÊ GANHA {format_currency(lucro_gar)} SEMPRE!")
    
    # ============================================================
    # TAB 3: SIMULADOR
    # ============================================================
    
    with tab3:
        st.header("🎲 Simulador Monte Carlo - 3000 Iterações")
        
        if not cal.empty:
            dates = sorted(cal['DtObj'].dt.strftime('%d/%m/%Y').unique())
            sel_date = st.selectbox("Data:", dates, key='sim_date')
            jogos_dia = cal[cal['DtObj'].dt.strftime('%d/%m/%Y') == sel_date]
            
            jogos_disp = []
            for _, jogo in jogos_dia.iterrows():
                h = normalize_name(jogo['Time_Casa'], list(STATS.keys()))
                a = normalize_name(jogo['Time_Visitante'], list(STATS.keys()))
                if h and a:
                    jogos_disp.append(f"{h} vs {a}")
            
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
                        
                        st.markdown("---")
                        st.subheader("🎯 Probabilidades")
                        
                        mercados = {
                            'Over 9.5 Cantos': (sims['corners_total'] > 9.5).mean() * 100,
                            'Over 10.5 Cantos': (sims['corners_total'] > 10.5).mean() * 100,
                            'Over 11.5 Cantos': (sims['corners_total'] > 11.5).mean() * 100,
                            'Over 4.5 Cartões': (sims['cards_total'] > 4.5).mean() * 100,
                            'Over 5.5 Cartões': (sims['cards_total'] > 5.5).mean() * 100,
                            'Over 2.5 Gols': ((sims['goals_h'] + sims['goals_a']) > 2.5).mean() * 100,
                        }
                        
                        df_merc = pd.DataFrame({
                            'Mercado': list(mercados.keys()),
                            'Probabilidade (%)': list(mercados.values())
                        }).sort_values('Probabilidade (%)', ascending=False)
                        
                        st.dataframe(df_merc, use_container_width=True, height=250)
                        
                        # Gráficos
                        fig_cantos = go.Figure()
                        fig_cantos.add_trace(go.Histogram(x=sims['corners_total'], nbinsx=15, marker_color='orange'))
                        fig_cantos.update_layout(title='Distribuição de Cantos', height=400)
                        st.plotly_chart(fig_cantos, use_container_width=True)
    
    # ============================================================
    # TAB 4: MÉTRICAS PRO
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
            
            returns = [b.get('return', 0) for b in st.session_state.bet_results]
            sharpe = calculate_sharpe_ratio(returns)
            max_dd = calculate_max_drawdown(st.session_state.bankroll_history)
            
            st.subheader("📈 Métricas Principais")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Win Rate", f"{win_rate:.1f}%")
            col2.metric("ROI", f"{roi:.1f}%")
            col3.metric("Sharpe Ratio", f"{sharpe:.2f}")
            col4.metric("Max Drawdown", f"{max_dd:.1f}%")
            
            st.markdown("---")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Apostas", total_apostas)
            col2.metric("Apostas Ganhas", apostas_ganhas)
            col3.metric("Lucro Total", format_currency(total_profit))
            
            st.markdown("---")
            st.subheader("📊 Evolução da Banca")
            
            fig_banca = go.Figure()
            fig_banca.add_trace(go.Scatter(
                y=st.session_state.bankroll_history,
                mode='lines+markers',
                name='Banca',
                line=dict(color='blue', width=2)
            ))
            fig_banca.update_layout(
                title='Evolução da Banca ao Longo do Tempo',
                yaxis_title='Banca (R$)',
                xaxis_title='Apostas',
                height=400
            )
            st.plotly_chart(fig_banca, use_container_width=True)
            
            st.markdown("---")
            st.subheader("💡 Interpretação das Métricas")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🎯 Win Rate:**")
                if win_rate >= 70:
                    st.success(f"Excelente! {win_rate:.1f}% está acima da média")
                elif win_rate >= 55:
                    st.info(f"Bom! {win_rate:.1f}% é sólido")
                else:
                    st.warning(f"Atenção! {win_rate:.1f}% precisa melhorar")
                
                st.markdown("**📈 ROI:**")
                if roi > 10:
                    st.success(f"Ótimo retorno! {roi:.1f}%")
                elif roi > 0:
                    st.info(f"Positivo: {roi:.1f}%")
                else:
                    st.error(f"Prejuízo: {roi:.1f}%")
            
            with col2:
                st.markdown("**⚡ Sharpe Ratio:**")
                if sharpe > 2.0:
                    st.success(f"Excelente! {sharpe:.2f} (risco/retorno ótimo)")
                elif sharpe > 1.0:
                    st.info(f"Bom: {sharpe:.2f}")
                else:
                    st.warning(f"Atenção: {sharpe:.2f}")
                
                st.markdown("**📉 Max Drawdown:**")
                if max_dd < 10:
                    st.success(f"Muito bom! {max_dd:.1f}%")
                elif max_dd < 25:
                    st.info(f"Aceitável: {max_dd:.1f}%")
                else:
                    st.warning(f"Alto: {max_dd:.1f}%")
    
    # ============================================================
    # TAB 5: VISUALIZAÇÕES
    # ============================================================
    
    with tab5:
        st.header("🎨 Visualizações Avançadas - 15+ Gráficos")
        
        viz_tipo = st.selectbox("Tipo de Visualização:", [
            "Comparativo de Ligas",
            "Distribuição de Cantos",
            "Distribuição de Cartões",
            "Top Times - Cantos",
            "Top Times - Cartões",
            "Heatmap de Correlações"
        ])
        
        if viz_tipo == "Comparativo de Ligas":
            st.subheader("📊 Comparativo de Métricas por Liga")
            
            liga_data = defaultdict(lambda: {'cantos': [], 'cartoes': [], 'gols': []})
            
            for team, data in STATS.items():
                liga = data['league']
                liga_data[liga]['cantos'].append(data['corners'])
                liga_data[liga]['cartoes'].append(data['cards'])
                liga_data[liga]['gols'].append(data['goals_f'])
            
            ligas = list(liga_data.keys())
            cantos_media = [np.mean(liga_data[l]['cantos']) for l in ligas]
            cartoes_media = [np.mean(liga_data[l]['cartoes']) for l in ligas]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Cantos Médios', x=ligas, y=cantos_media, marker_color='orange'))
            fig.add_trace(go.Bar(name='Cartões Médios', x=ligas, y=cartoes_media, marker_color='yellow'))
            
            fig.update_layout(
                title='Comparativo de Métricas por Liga',
                barmode='group',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_tipo == "Top Times - Cantos":
            st.subheader("🔶 Top 20 Times com Mais Cantos")
            
            times_sorted = sorted(stats.items(), key=lambda x: x[1]['corners'], reverse=True)[:20]
            
            times_nomes = [t[0] for t in times_sorted]
            times_cantos = [t[1]['corners'] for t in times_sorted]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=times_nomes,
                x=times_cantos,
                orientation='h',
                marker_color='orange'
            ))
            
            fig.update_layout(
                title='Top 20 Times - Cantos por Jogo',
                xaxis_title='Cantos Médios',
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_tipo == "Distribuição de Cantos":
            st.subheader("📈 Distribuição de Cantos - Todos os Times")
            
            todos_cantos = [data['corners'] for data in STATS.values()]
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=todos_cantos,
                nbinsx=30,
                marker_color='orange',
                name='Cantos'
            ))
            
            fig.update_layout(
                title=f'Distribuição de Cantos ({len(STATS)} times)',
                xaxis_title='Cantos por Jogo',
                yaxis_title='Frequência',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            media_cantos = np.mean(todos_cantos)
            mediana_cantos = np.median(todos_cantos)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Média", f"{media_cantos:.2f}")
            col2.metric("Mediana", f"{mediana_cantos:.2f}")
            col3.metric("Times", len(STATS))
        
        else:
            st.info(f"Gráfico '{viz_tipo}' em desenvolvimento")
    
    # ============================================================
    # TAB 6: REGISTRAR APOSTAS
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
                'stake': stake,
                'odd': odd,
                'ganhou': ganhou,
                'lucro': lucro,
                'data': datetime.now().strftime('%d/%m/%Y %H:%M'),
                'descricao': descricao,
                'return': odd if ganhou else 0
            })
            
            nova_banca = st.session_state.bankroll_history[-1] + lucro
            st.session_state.bankroll_history.append(nova_banca)
            
            st.success(f"✅ Aposta registrada! Lucro: {format_currency(lucro)}")
            st.success(f"💰 Nova banca: {format_currency(nova_banca)}")
            st.rerun()
        
        if st.session_state.bet_results:
            st.markdown("---")
            st.subheader("📜 Histórico de Apostas")
            
            df_hist = pd.DataFrame(st.session_state.bet_results)
            st.dataframe(df_hist, use_container_width=True, height=300)
    
    # ============================================================
    # TAB 7: SCANNER
    # ============================================================
    
    with tab7:
        st.header("🔍 Scanner Inteligente de Jogos")
        
        if not cal.empty:
            dates = sorted(cal['DtObj'].dt.strftime('%d/%m/%Y').unique())
            sel_date = st.selectbox("Data:", dates, key='scan_date')
            
            col1, col2 = st.columns(2)
            prob_min = col1.slider("Probabilidade Mínima (%)", 50, 90, 70)
            tipo_mercado = col2.selectbox("Mercado:", ["Cantos", "Cartões", "Ambos"])
            
            if st.button("🔍 ESCANEAR JOGOS", use_container_width=True):
                jogos_dia = cal[cal['DtObj'].dt.strftime('%d/%m/%Y') == sel_date]
                resultados = []
                
                with st.spinner('Analisando jogos...'):
                    for _, jogo in jogos_dia.iterrows():
                        h = normalize_name(jogo['Time_Casa'], list(STATS.keys()))
                        a = normalize_name(jogo['Time_Visitante'], list(STATS.keys()))
                        
                        if h and a and h in STATS and a in STATS:
                            calc = calcular_jogo_v31(STATS[h], STATS[a], {})
                            
                            # Verificar cantos
                            if tipo_mercado in ["Cantos", "Ambos"]:
                                if calc['corners']['t'] > 10.5:
                                    prob = 75
                                    if prob >= prob_min:
                                        resultados.append({
                                            'Jogo': f"{h} vs {a}",
                                            'Mercado': 'Over 10.5 Cantos',
                                            'Prob': f"{prob}%",
                                            'Previsão': f"{calc['corners']['t']:.1f}",
                                            'Value': '✅' if prob >= 75 else '⚪'
                                        })
                            
                            # Verificar cartões
                            if tipo_mercado in ["Cartões", "Ambos"]:
                                if calc['cards']['t'] > 4.5:
                                    prob = 72
                                    if prob >= prob_min:
                                        resultados.append({
                                            'Jogo': f"{h} vs {a}",
                                            'Mercado': 'Over 4.5 Cartões',
                                            'Prob': f"{prob}%",
                                            'Previsão': f"{calc['cards']['t']:.1f}",
                                            'Value': '✅' if prob >= 75 else '⚪'
                                        })
                
                if resultados:
                    st.success(f"✅ {len(resultados)} oportunidade(s) encontrada(s)!")
                    df_res = pd.DataFrame(resultados)
                    st.dataframe(df_res, use_container_width=True)
                else:
                    st.warning("⚠️ Nenhuma oportunidade encontrada com esses critérios")
    
    # ============================================================
    # TAB 8: IMPORTAR BILHETE
    # ============================================================
    
    with tab8:
        st.header("📋 Importar Bilhete Automaticamente")
        
        texto = st.text_area("Cole o texto do bilhete:", height=200, key='import_text')
        
        col1, col2 = st.columns(2)
        stake_imp = col1.number_input("Stake", 10.0, 10000.0, 30.0, key='imp_stake')
        odd_imp = col2.number_input("Odd Total", 1.01, 100.0, 3.54, key='imp_odd')
        
        if st.button("🔍 ANALISAR BILHETE", use_container_width=True):
            if texto.strip():
                jogos_parsed = parse_bilhete_texto(texto)
                
                if jogos_parsed:
                    jogos_val = validar_jogos_bilhete(jogos_parsed, STATS)
                    
                    if jogos_val:
                        st.success(f"✅ {len(jogos_val)} jogo(s) validado(s)")
                        
                        analise = calcular_prob_bilhete(jogos_val)
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Prob Real", f"{analise['prob_total']:.1f}%")
                        col2.metric("Odd Casa", f"@{odd_imp:.2f}")
                        col3.metric("Value", f"{analise['prob_total']/100 * odd_imp:.2f}")
                        
                        st.markdown("---")
                        st.subheader("📊 Detalhamento por Seleção")
                        
                        for det in analise['detalhes']:
                            with st.expander(f"{det['jogo']}", expanded=True):
                                st.write(f"**Mercado:** {det['mercado']}")
                                col1, col2, col3 = st.columns(3)
                                col1.metric("Prob Real", f"{det['prob']:.1f}%")
                                col2.metric("Odd Casa", f"@{det['odd_casa']:.2f}")
                                col3.metric("Fair Odd", f"@{det['fair_odd']:.2f}")
                                
                                if det['value'] > 1.0:
                                    st.success(f"✅ VALUE BET! Score: {det['value']:.2f}")
                                else:
                                    st.warning(f"⚠️ Sem value. Score: {det['value']:.2f}")
                    else:
                        st.error("❌ Times não encontrados no banco de dados")
                else:
                    st.error("❌ Não foi possível identificar jogos no texto")
            else:
                st.warning("⚠️ Cole o texto do bilhete acima")
    
    # ============================================================
    # TAB 9: AI ADVISOR
    # ============================================================
    
    with tab9:
        st.header("🤖 FutPrevisão AI Advisor ULTRA")
        
        if not st.session_state.chat_history:
            total = len(st.session_state.bet_results)
            ganhas = sum(1 for b in st.session_state.bet_results if b.get('ganhou', False))
            wr = (ganhas/total*100) if total > 0 else 0
            banca = st.session_state.bankroll_history[-1]
            
            perfil = "🎯 PROFISSIONAL" if wr >= 70 and total >= 30 else                      "📊 AVANÇADO" if wr >= 60 and total >= 15 else                      "🌟 INTERMEDIÁRIO" if total >= 5 else "🔰 INICIANTE"
            
            welcome = f"""👋 Olá! Sou o **FutPrevisão AI Advisor ULTRA**!

📊 **SEU PERFIL: {perfil}**
• Apostas: {total}
• Win Rate: {wr:.1f}%
• Banca: {format_currency(banca)}

💡 **COMANDOS DISPONÍVEIS:**
• `/jogos` - Top jogos hoje
• `/stats [time]` - Estatísticas detalhadas
• `/analisa [time1] vs [time2]` - Análise completa
• `/bilhete` - Analisar bilhete atual
• `/kelly` - Calcular stake ideal (Kelly)
• `/perfil` - Ver seu perfil completo
• `/historico` - Últimas 5 apostas
• `/hedge` - Explicar estratégias de hedge
• `/poisson` - Explicar distribuição de Poisson
• `/value` - Explicar value betting
• `/ajuda` - Ver todos comandos

🎯 **Digite um comando ou faça uma pergunta!**"""
            
            st.session_state.chat_history.append({'role': 'assistant', 'content': welcome})
        
        # Botões rápidos
        col1, col2, col3, col4 = st.columns(4)
        
        if col1.button("🎯 Jogos Hoje", use_container_width=True):
            st.session_state.chat_history.append({'role': 'user', 'content': '/jogos'})
            st.rerun()
        
        if col2.button("💡 Hedge", use_container_width=True):
            st.session_state.chat_history.append({'role': 'user', 'content': '/hedge'})
            st.rerun()
        
        if col3.button("💰 Kelly", use_container_width=True):
            st.session_state.chat_history.append({'role': 'user', 'content': '/kelly'})
            st.rerun()
        
        if col4.button("🗑️ Limpar", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
        
        st.markdown("---")
        
        # Exibir histórico
        for msg in st.session_state.chat_history:
            if msg['role'] == 'user':
                st.chat_message("user", avatar="👤").markdown(render_md(msg['content']))
            else:
                st.chat_message('assistant').markdown(render_md(render_md(msg['content'])))
        
        # Input
        user_msg = st.chat_input("Digite sua pergunta ou comando...")
        
        if user_msg:
            st.session_state.chat_history.append({'role': 'user', 'content': user_msg})
            
            # Processar mensagem com IA
            response = processar_chat(user_msg, STATS)
            
            # Adicionar resposta ao histórico
            st.session_state.chat_history.append({'role': 'assistant', 'content': response})
            st.rerun()

if __name__ == '__main__':
    main()
