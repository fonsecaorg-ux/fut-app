import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import uuid
import math
import difflib
import random
import re
from pathlib import Path
from datetime import datetime
from PIL import Image

# ==============================================================================
# 0. CONFIGURAÇÃO & SETUP
# ==============================================================================
st.set_page_config(page_title="FutPrevisão Pro V8.0 (Architect)", layout="wide", page_icon="⚽")

# Configuração OCR (Opcional, não trava se falhar)
HAS_OCR = False
try:
    import pytesseract
    path_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(path_tesseract):
        pytesseract.pytesseract.tesseract_cmd = path_tesseract
        HAS_OCR = True
    else:
        alts = [r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe", r"C:\Users\Kaiqu\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"]
        for p in alts:
            if os.path.exists(p):
                pytesseract.pytesseract.tesseract_cmd = p
                HAS_OCR = True; break
except: pass

# ==============================================================================
# 1. MAPEAMENTO DE ARQUIVOS (As 10 Ligas)
# ==============================================================================
LEAGUE_FILES = {
    "Premier League": {"csv": "Premier League 25.26.csv", "txt": "Prewmier League.txt", "txt_cards": "Cartoes Premier League - Inglaterra.txt"},
    "La Liga": {"csv": "La Liga 25.26.csv", "txt": "Escanteios Espanha.txt", "txt_cards": "Cartoes La Liga - Espanha.txt"},
    "Serie A": {"csv": "Serie A 25.26.csv", "txt": "Escanteios Italia.txt", "txt_cards": "Cartoes Serie A - Italia.txt"},
    "Bundesliga": {"csv": "Bundesliga 25.26.csv", "txt": "Escanteios Alemanha.txt", "txt_cards": "Cartoes Bundesliga - Alemanha.txt"},
    "Ligue 1": {"csv": "Ligue 1 25.26.csv", "txt": "Escanteios França.txt", "txt_cards": "Cartoes Ligue 1 - França.txt"},
    "Championship": {"csv": "Championship Inglaterra 25.26.csv", "txt": "Championship Escanteios Inglaterra.txt", "txt_cards": None},
    "Bundesliga 2": {"csv": "Bundesliga 2.csv", "txt": "Bundesliga 2.txt", "txt_cards": None},
    "Pro League (BEL)": {"csv": "Pro League Belgica 25.26.csv", "txt": "Pro League Belgica.txt", "txt_cards": None},
    "Süper Lig (TUR)": {"csv": "Super Lig Turquia 25.26.csv", "txt": "Super Lig Turquia.txt", "txt_cards": None},
    "Premiership (SCO)": {"csv": "Premiership Escocia 25.26.csv", "txt": "Premiership Escocia.txt", "txt_cards": None}
}

# ==============================================================================
# 2. MOTOR DE ÁRBITROS
# ==============================================================================
@st.cache_data(ttl=3600)
def load_referees():
    """Carrega o CSV de árbitros e cria um dicionário {Nome: Fator}"""
    refs = {}
    if os.path.exists("arbitros_5_ligas_2025_2026.csv"):
        try:
            df = pd.read_csv("arbitros_5_ligas_2025_2026.csv")
            # Normaliza nomes para garantir match
            for _, row in df.iterrows():
                nome = str(row['Arbitro']).strip()
                media = float(row['Media_Cartoes_Por_Jogo'])
                refs[nome] = media
        except: pass
    return refs

referees_db = load_referees()

def get_referee_factor(ref_name):
    """Calcula o multiplicador de cartões baseado no árbitro"""
    if not ref_name: return 1.0
    
    # Tenta match exato
    if ref_name in referees_db:
        avg = referees_db[ref_name]
        return avg / 4.0 # 4.0 é a media base mundial. Se ele tem 5.0, fator é 1.25 (+25%)
    
    # Tenta fuzzy match (se o nome no calendario for 'M. Oliver' e no csv 'Michael Oliver')
    matches = difflib.get_close_matches(ref_name, referees_db.keys(), n=1, cutoff=0.7)
    if matches:
        avg = referees_db[matches[0]]
        return avg / 4.0
        
    return 1.0 # Neutro

# ==============================================================================
# 3. APRENDIZADO DE MÁQUINA (CSVs -> Médias Matemáticas)
# ==============================================================================
@st.cache_data(ttl=3600)
def learn_stats():
    """Lê todos os CSVs e aprende a média de Cantos e Cartões de cada time"""
    db = {}
    for liga, files in LEAGUE_FILES.items():
        f = files["csv"]
        if os.path.exists(f):
            try:
                try: df = pd.read_csv(f, encoding='latin1')
                except: df = pd.read_csv(f)
                
                # Identifica colunas
                cols = df.columns
                c_home = 'HomeTeam' if 'HomeTeam' in cols else 'Mandante'
                c_away = 'AwayTeam' if 'AwayTeam' in cols else 'Visitante'
                
                # Verifica se tem dados
                has_corn = 'HC' in cols and 'AC' in cols
                has_card = 'HY' in cols and 'AY' in cols # Usando Amarelos como proxy de cartões
                
                if c_home in cols:
                    teams = set(df[c_home].unique()).union(set(df[c_away].unique()))
                    for t in teams:
                        # Filtra jogos do time
                        h_games = df[df[c_home] == t]
                        a_games = df[df[c_away] == t]
                        
                        count = len(h_games) + len(a_games)
                        if count < 3: continue # Ignora times com poucos jogos
                        
                        # Soma Cantos
                        corn_sum = 0
                        if has_corn:
                            corn_sum += h_games['HC'].sum() + a_games['AC'].sum()
                        else: corn_sum = count * 5.0 # Fallback
                        
                        # Soma Cartões
                        card_sum = 0
                        if has_card:
                            card_sum += h_games['HY'].sum() + a_games['AY'].sum()
                        else: card_sum = count * 2.0 # Fallback
                        
                        db[t] = {
                            'corners': corn_sum / count,
                            'cards': card_sum / count,
                            'league': liga
                        }
            except: pass
    return db

stats_db = learn_stats()
team_list = sorted(list(stats_db.keys()))

# ==============================================================================
# 4. CARREGADOR DE HISTÓRICO (JSON/TXT)
# ==============================================================================
class HistoryLoader:
    def __init__(self):
        self.corners = {}
        self.cards = {}
        self.load_all()

    def load_all(self):
        for liga, files in LEAGUE_FILES.items():
            # Cantos
            if files['txt'] and os.path.exists(files['txt']):
                try: 
                    with open(files['txt'], 'r', encoding='utf-8') as f:
                        raw = f.read().strip()
                        if '{' in raw: raw = raw[raw.find('{'):]
                        self.corners[liga] = json.loads(raw)
                except: pass
            # Cartões
            if files['txt_cards'] and os.path.exists(files['txt_cards']):
                try: 
                    with open(files['txt_cards'], 'r', encoding='utf-8') as f:
                        raw = f.read().strip()
                        if '{' in raw: raw = raw[raw.find('{'):]
                        self.cards[liga] = json.loads(raw)
                except: pass

    def get_data(self, team, liga, market, key):
        """
        Retorna (TotalJogos, JogosBateu, Porcentagem)
        Ex: (10, 7, 70.0)
        """
        source = self.corners if market == 'corners' else self.cards
        
        # Tenta achar o time na liga correta
        target_src = source.get(liga)
        if not target_src: 
            # Fallback: Procura em todas as ligas
            for l in source:
                res = self._search_in_league(source[l], team, key)
                if res: return res
            return None
        
        return self._search_in_league(target_src, team, key)

    def _search_in_league(self, json_data, team_name, key):
        # Mapeamento Fuzzy para nomes de times
        avail_teams = [t['teamName'] for t in json_data.get('teams', [])]
        
        # Tenta exato
        match = None
        if team_name in avail_teams: match = team_name
        else:
            # Tenta aproximação
            close = difflib.get_close_matches(team_name, avail_teams, n=1, cutoff=0.6)
            if close: match = close[0]
            
        if match:
            for t in json_data['teams']:
                if t['teamName'] == match:
                    # O JSON do Adam Choi geralmente é: [Total, Bateu, %, Streak]
                    data = t.get(key)
                    if data and isinstance(data, list) and len(data) >= 3:
                        return data[0], data[1], data[2] # (Total, Hits, %)
        return None

history = HistoryLoader()

# ==============================================================================
# 5. MATEMÁTICA E CÁLCULO
# ==============================================================================
def poisson_prob(k, lam):
    return (math.exp(-lam) * (lam ** k)) / math.factorial(k)

def prob_over_line(lam, line):
    # Probabilidade de ser MAIOR que a linha (Ex: > 3.5 é 4, 5, 6...)
    # 1 - (Prob(0) + Prob(1) + ... + Prob(int(line)))
    cdf = sum(poisson_prob(i, lam) for i in range(int(line) + 1))
    return max(0.0, (1 - cdf) * 100)

def normalize_name(name):
    if name in stats_db: return name
    matches = difflib.get_close_matches(name, stats_db.keys(), n=1, cutoff=0.6)
    return matches[0] if matches else None

def calcular_previsao(home, away, referee=None):
    h = normalize_name(home)
    a = normalize_name(away)
    
    if not h or not a: return None
    
    s_h = stats_db.get(h, {'corners': 5.0, 'cards': 2.0})
    s_a = stats_db.get(a, {'corners': 4.0, 'cards': 2.0})
    
    # Fator Árbitro
    ref_boost = get_referee_factor(referee)
    
    # Matemática (Lambda Esperado)
    exp_corn_h = s_h['corners'] * 1.10 # Leve vantagem casa
    exp_corn_a = s_a['corners'] * 0.90 # Leve desvantagem fora
    
    exp_card_h = s_h['cards'] * ref_boost
    exp_card_a = s_a['cards'] * ref_boost
    
    return {
        'corners': {'h': exp_corn_h, 'a': exp_corn_a, 't': exp_corn_h + exp_corn_a},
        'cards': {'h': exp_card_h, 'a': exp_card_a, 't': exp_card_h + exp_card_a}
    }

def get_color(prob):
    if prob >= 70: return "green"
    if prob >= 50: return "orange"
    return "red"

# ==============================================================================
# 6. INTERFACE (DASHBOARD V8.0)
# ==============================================================================
def render_dashboard():
    st.title("🛡️ FutPrevisão Pro V8.0 (Architect)")
    
    st.markdown("""
    <style>
        .stat-box { background: #f8f9fa; padding: 10px; border-radius: 5px; border: 1px solid #dee2e6; margin-bottom: 5px; }
        .green { color: #28a745; font-weight: bold; }
        .orange { color: #fd7e14; font-weight: bold; }
        .red { color: #dc3545; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

    tab_scan, tab_sim = st.tabs(["🔍 Scanner", "🔮 Simulação Detalhada"])

    # --- SIMULAÇÃO (O PEDIDO PRINCIPAL) ---
    with tab_sim:
        c1, c2, c3 = st.columns(3)
        # Seletores
        teams = team_list if team_list else ["Carregando..."]
        idx_h = teams.index("Arsenal") if "Arsenal" in teams else 0
        idx_a = teams.index("Man City") if "Man City" in teams else 1
        
        home = c1.selectbox("Mandante", teams, index=idx_h)
        away = c2.selectbox("Visitante", teams, index=idx_a)
        
        # Tenta pegar liga do time
        liga_estimada = stats_db.get(home, {}).get('league', 'Premier League')
        
        # Arbitros
        all_refs = sorted(list(referees_db.keys()))
        referee = c3.selectbox("Árbitro", ["Média/Neutro"] + all_refs)
        
        if st.button("📊 Analisar Confronto V8"):
            ref_clean = referee if referee != "Média/Neutro" else None
            m = calcular_previsao(home, away, ref_clean)
            
            if m:
                st.divider()
                
                # --- COLUNA ESCANTEIOS ---
                k1, k2 = st.columns(2)
                with k1:
                    st.subheader(f"🚩 Escanteios")
                    st.info(f"Total Esperado: {m['corners']['t']:.2f}")
                    
                    # === MANDANTE CANTOS ===
                    st.markdown(f"**🏠 {home} (Cantos)**")
                    
                    # Linha 3.5
                    p35 = prob_over_line(m['corners']['h'], 3.5)
                    h35 = history.get_data(home, liga_estimada, 'corners', 'homeTeamOver35')
                    hist_txt = f"{h35[2]}% ({h35[1]}/{h35[0]})" if h35 else "S/ Dados"
                    st.markdown(f"Over 3.5: :{get_color(p35)}[{p35:.0f}%] | Hist: **{hist_txt}**")
                    
                    # Linha 4.5
                    p45 = prob_over_line(m['corners']['h'], 4.5)
                    h45 = history.get_data(home, liga_estimada, 'corners', 'homeTeamOver45')
                    hist_txt = f"{h45[2]}% ({h45[1]}/{h45[0]})" if h45 else "S/ Dados"
                    st.markdown(f"Over 4.5: :{get_color(p45)}[{p45:.0f}%] | Hist: **{hist_txt}**")
                    
                    st.markdown("---")
                    
                    # === VISITANTE CANTOS ===
                    st.markdown(f"**✈️ {away} (Cantos)**")
                    
                    # Linha 3.5
                    p35 = prob_over_line(m['corners']['a'], 3.5)
                    h35 = history.get_data(away, liga_estimada, 'corners', 'awayTeamOver35')
                    hist_txt = f"{h35[2]}% ({h35[1]}/{h35[0]})" if h35 else "S/ Dados"
                    st.markdown(f"Over 3.5: :{get_color(p35)}[{p35:.0f}%] | Hist: **{hist_txt}**")
                    
                    # Linha 4.5
                    p45 = prob_over_line(m['corners']['a'], 4.5)
                    h45 = history.get_data(away, liga_estimada, 'corners', 'awayTeamOver45')
                    hist_txt = f"{h45[2]}% ({h45[1]}/{h45[0]})" if h45 else "S/ Dados"
                    st.markdown(f"Over 4.5: :{get_color(p45)}[{p45:.0f}%] | Hist: **{hist_txt}**")

                # --- COLUNA CARTÕES ---
                with k2:
                    st.subheader(f"🟨 Cartões")
                    st.warning(f"Total Esperado: {m['cards']['t']:.2f} (Juiz: {get_referee_factor(ref_clean):.2f}x)")
                    
                    # === MANDANTE CARTÕES ===
                    st.markdown(f"**🏠 {home} (Cartões)**")
                    
                    # Linha 1.5
                    p15 = prob_over_line(m['cards']['h'], 1.5)
                    # Nota: O JSON do Adam Choi nem sempre tem 'homeCardsOver15' separado por time em todas ligas
                    # Mas tentamos buscar
                    h15 = history.get_data(home, liga_estimada, 'cards', 'homeCardsOver15')
                    hist_txt = f"{h15[2]}% ({h15[1]}/{h15[0]})" if h15 else "S/ Dados"
                    st.markdown(f"Over 1.5: :{get_color(p15)}[{p15:.0f}%] | Hist: **{hist_txt}**")
                    
                    # Linha 2.5
                    p25 = prob_over_line(m['cards']['h'], 2.5)
                    h25 = history.get_data(home, liga_estimada, 'cards', 'homeCardsOver25')
                    hist_txt = f"{h25[2]}% ({h25[1]}/{h25[0]})" if h25 else "S/ Dados"
                    st.markdown(f"Over 2.5: :{get_color(p25)}[{p25:.0f}%] | Hist: **{hist_txt}**")
                    
                    st.markdown("---")
                    
                    # === VISITANTE CARTÕES ===
                    st.markdown(f"**✈️ {away} (Cartões)**")
                    
                    # Linha 1.5
                    p15 = prob_over_line(m['cards']['a'], 1.5)
                    h15 = history.get_data(away, liga_estimada, 'cards', 'awayCardsOver15')
                    hist_txt = f"{h15[2]}% ({h15[1]}/{h15[0]})" if h15 else "S/ Dados"
                    st.markdown(f"Over 1.5: :{get_color(p15)}[{p15:.0f}%] | Hist: **{hist_txt}**")
                    
                    # Linha 2.5
                    p25 = prob_over_line(m['cards']['a'], 2.5)
                    h25 = history.get_data(away, liga_estimada, 'cards', 'awayCardsOver25')
                    hist_txt = f"{h25[2]}% ({h25[1]}/{h25[0]})" if h25 else "S/ Dados"
                    st.markdown(f"Over 2.5: :{get_color(p25)}[{p25:.0f}%] | Hist: **{hist_txt}**")

    # --- SCANNER (LÊ CALENDÁRIOS) ---
    with tab_scan:
        # Carrega calendários unificados
        all_games = []
        for l_name, f_data in LEAGUE_FILES.items():
            if os.path.exists(f_data['csv']):
                try:
                    try: df = pd.read_csv(f_data['csv'], encoding='latin1', dtype=str)
                    except: df = pd.read_csv(f_data['csv'], dtype=str)
                    
                    # Normaliza colunas
                    cols = df.columns
                    d_col = 'Date' if 'Date' in cols else 'Data'
                    h_col = 'HomeTeam' if 'HomeTeam' in cols else 'Mandante'
                    a_col = 'AwayTeam' if 'AwayTeam' in cols else 'Visitante'
                    r_col = 'Referee' if 'Referee' in cols else ('Arbitro' if 'Arbitro' in cols else None)
                    
                    if d_col in cols and h_col in cols:
                        temp = df[[d_col, h_col, a_col]].copy()
                        temp.columns = ['Data', 'Mandante', 'Visitante']
                        temp['Liga'] = l_name
                        if r_col: temp['Arbitro'] = df[r_col]
                        else: temp['Arbitro'] = None
                        all_games.append(temp)
                except: pass
        
        if all_games:
            full_df = pd.concat(all_games, ignore_index=True)
            full_df['Data'] = full_df['Data'].str.strip()
            
            dias = sorted(full_df['Data'].unique())
            hoje = datetime.now().strftime("%d/%m/%Y")
            idx = dias.index(hoje) if hoje in dias else 0
            
            dia_sel = st.selectbox("Selecione Data:", dias, index=idx)
            jogos = full_df[full_df['Data'] == dia_sel]
            
            st.write(f"Encontrados {len(jogos)} jogos.")
            
            if st.button("Escanear Oportunidades V8"):
                for i, row in jogos.iterrows():
                    h, a, l, r = row['Mandante'], row['Visitante'], row['Liga'], row['Arbitro']
                    m = calcular_previsao(h, a, r)
                    
                    if m:
                        # Regra de Ouro: Matemática > 60% E Histórico > 60%
                        # Cantos Casa
                        ph35 = prob_over_line(m['corners']['h'], 3.5)
                        hh35 = history.get_data(h, l, 'corners', 'homeTeamOver35')
                        if ph35 > 60 and hh35 and float(hh35[2]) > 60:
                            st.success(f"🚩 **{h} +3.5 Cantos** | Math: {ph35:.0f}% | Hist: {hh35[2]}% ({hh35[1]}/{hh35[0]}) | Liga: {l}")
                            
                        # Cantos Fora
                        pa35 = prob_over_line(m['corners']['a'], 3.5)
                        ha35 = history.get_data(a, l, 'corners', 'awayTeamOver35')
                        if pa35 > 60 and ha35 and float(ha35[2]) > 60:
                            st.success(f"🚩 **{a} +3.5 Cantos** | Math: {pa35:.0f}% | Hist: {ha35[2]}% ({ha35[1]}/{ha35[0]}) | Liga: {l}")

                        # Cartões (Se tiver ref forte)
                        rf = get_referee_factor(r)
                        if rf > 1.1: # Juiz rigoroso
                             st.warning(f"🟨 **{h} x {a}** | Juiz Rigoroso ({r}). Atenção em Over Cartões.")

if __name__ == "__main__":
    render_dashboard()
