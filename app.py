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
# 0. CONFIGURAÇÃO
# ==============================================================================
st.set_page_config(page_title="FutPrevisão V8.1 (Refined)", layout="wide", page_icon="⚽")

# Configuração OCR (Silenciosa)
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
# 1. MAPEAMENTO DE ARQUIVOS (10 Ligas)
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
# 2. DOUTOR DAS DATAS (CORREÇÃO DE BUG)
# ==============================================================================
def fix_date_format(date_str):
    """Tenta padronizar qualquer data para DD/MM/YYYY"""
    if pd.isna(date_str): return None
    s = str(date_str).strip()
    
    # Formatos comuns encontrados no Football-Data
    formats = [
        "%d/%m/%Y",   # 25/12/2025
        "%d/%m/%y",   # 25/12/25
        "%Y-%m-%d",   # 2025-12-25
        "%d-%m-%Y"    # 25-12-2025
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(s, fmt).strftime("%d/%m/%Y")
        except: continue
    
    return s # Retorna original se falhar (para debug)

# ==============================================================================
# 3. FUSÃO DE ÁRBITROS
# ==============================================================================
@st.cache_data(ttl=3600)
def load_referees_unified():
    """Lê os dois arquivos e cria um mega dicionário de árbitros"""
    refs = {}
    
    # 1. Lê o arquivo simples (arbitros.csv)
    if os.path.exists("arbitros.csv"):
        try:
            df1 = pd.read_csv("arbitros.csv")
            for _, r in df1.iterrows():
                if 'Nome' in r and 'Fator' in r:
                    refs[str(r['Nome']).strip()] = float(r['Fator'])
        except: pass
        
    # 2. Lê o arquivo detalhado (arbitros_5_ligas...) e sobrescreve/adiciona
    if os.path.exists("arbitros_5_ligas_2025_2026.csv"):
        try:
            df2 = pd.read_csv("arbitros_5_ligas_2025_2026.csv")
            for _, r in df2.iterrows():
                nome = str(r['Arbitro']).strip()
                media = float(r['Media_Cartoes_Por_Jogo'])
                # Converte Média para Fator (Base 4.0 cartões/jogo)
                # Ex: Juiz com média 5.0 -> Fator 1.25
                refs[nome] = media / 4.0
        except: pass
        
    return refs

referees_db = load_referees_unified()

def get_referee_factor(ref_name):
    if not ref_name: return 1.0
    # Match Exato
    if ref_name in referees_db: return referees_db[ref_name]
    # Match Aproximado
    match = difflib.get_close_matches(ref_name, referees_db.keys(), n=1, cutoff=0.7)
    if match: return referees_db[match[0]]
    return 1.0 # Neutro

# ==============================================================================
# 4. LEITOR DE CALENDÁRIO UNIFICADO
# ==============================================================================
@st.cache_data(ttl=600)
def load_unified_calendar():
    all_games = []
    
    for l_name, f_data in LEAGUE_FILES.items():
        csv_path = f_data['csv']
        if os.path.exists(csv_path):
            try:
                # Tenta ler com diferentes encodings
                try: df = pd.read_csv(csv_path, encoding='latin1', dtype=str)
                except: df = pd.read_csv(csv_path, dtype=str)
                
                cols = [c.strip() for c in df.columns]
                df.columns = cols
                
                # Mapeia colunas
                d_col = next((c for c in cols if c in ['Date', 'Data']), None)
                h_col = next((c for c in cols if c in ['HomeTeam', 'Mandante']), None)
                a_col = next((c for c in cols if c in ['AwayTeam', 'Visitante']), None)
                r_col = next((c for c in cols if c in ['Referee', 'Arbitro']), None)
                
                if d_col and h_col and a_col:
                    temp = pd.DataFrame()
                    # Aplica o Doutor das Datas
                    temp['Data'] = df[d_col].apply(fix_date_format)
                    temp['Mandante'] = df[h_col]
                    temp['Visitante'] = df[a_col]
                    temp['Liga'] = l_name
                    temp['Arbitro'] = df[r_col] if r_col else None
                    
                    # Remove datas inválidas
                    temp = temp.dropna(subset=['Data'])
                    all_games.append(temp)
            except Exception as e:
                print(f"Erro lendo {l_name}: {e}")

    if all_games:
        return pd.concat(all_games, ignore_index=True)
    return pd.DataFrame()

# ==============================================================================
# 5. APRENDIZADO E HISTÓRICO
# ==============================================================================
@st.cache_data(ttl=3600)
def learn_stats_from_csvs():
    db = {}
    for liga, files in LEAGUE_FILES.items():
        f = files["csv"]
        if os.path.exists(f):
            try:
                try: df = pd.read_csv(f, encoding='latin1')
                except: df = pd.read_csv(f)
                
                cols = df.columns
                h_c = 'HomeTeam' if 'HomeTeam' in cols else 'Mandante'
                a_c = 'AwayTeam' if 'AwayTeam' in cols else 'Visitante'
                
                # Detecta colunas de dados
                has_corn = 'HC' in cols and 'AC' in cols
                has_card = 'HY' in cols and 'AY' in cols 
                
                if h_c in cols:
                    teams = set(df[h_c].unique()).union(set(df[a_c].unique()))
                    for t in teams:
                        hg = df[df[h_c] == t]
                        ag = df[df[a_c] == t]
                        n = len(hg) + len(ag)
                        if n < 3: continue
                        
                        c_avg = 5.0
                        if has_corn: c_avg = (hg['HC'].sum() + ag['AC'].sum()) / n
                        
                        k_avg = 2.0
                        if has_card: k_avg = (hg['HY'].sum() + ag['AY'].sum()) / n
                        
                        db[t] = {'corners': c_avg, 'cards': k_avg, 'league': liga}
            except: pass
    return db

stats_db = learn_stats_from_csvs()
team_list = sorted(list(stats_db.keys()))

class HistoryLoader:
    def __init__(self):
        self.corners = {}
        self.cards = {}
        self.load_all()

    def load_all(self):
        for liga, files in LEAGUE_FILES.items():
            if files['txt'] and os.path.exists(files['txt']):
                try: 
                    with open(files['txt'], 'r', encoding='utf-8') as f:
                        raw = f.read().strip()
                        if '{' in raw: raw = raw[raw.find('{'):]
                        self.corners[liga] = json.loads(raw)
                except: pass
            if files['txt_cards'] and os.path.exists(files['txt_cards']):
                try: 
                    with open(files['txt_cards'], 'r', encoding='utf-8') as f:
                        raw = f.read().strip()
                        if '{' in raw: raw = raw[raw.find('{'):]
                        self.cards[liga] = json.loads(raw)
                except: pass

    def get_data(self, team, liga, market, key):
        src = self.corners if market == 'corners' else self.cards
        
        # Busca Prioritária (Liga Correta)
        if liga in src:
            res = self._find(src[liga], team, key)
            if res: return res
            
        # Busca Global
        for l in src:
            res = self._find(src[l], team, key)
            if res: return res
        return None

    def _find(self, json_data, team, key):
        avail = [t['teamName'] for t in json_data.get('teams', [])]
        match = difflib.get_close_matches(team, avail, n=1, cutoff=0.6)
        if match:
            for t in json_data['teams']:
                if t['teamName'] == match[0]:
                    d = t.get(key)
                    if d and len(d) >= 3: return d[0], d[1], d[2]
        return None

history = HistoryLoader()

# ==============================================================================
# 6. LÓGICA DE PREVISÃO
# ==============================================================================
def poisson_prob(k, lam): return (math.exp(-lam) * (lam ** k)) / math.factorial(k)

def prob_over_line(lam, line):
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
    
    rf = get_referee_factor(referee)
    
    # Modelo Matemático
    # Casa tem boost de 10% em cantos, Fora tem penalidade de 10%
    c_h = s_h['corners'] * 1.10
    c_a = s_a['corners'] * 0.90
    
    # Cartões dependem do Juiz
    k_h = s_h['cards'] * rf
    k_a = s_a['cards'] * rf
    
    return {
        'corners': {'h': c_h, 'a': c_a, 't': c_h + c_a},
        'cards': {'h': k_h, 'a': k_a, 't': k_h + k_a}
    }

def get_color(prob):
    if prob >= 70: return "green"
    if prob >= 50: return "orange"
    return "red"

# ==============================================================================
# 7. DASHBOARD V8.1
# ==============================================================================
def render_dashboard():
    st.title("⚽ FutPrevisão V8.1 (Refined)")
    
    tab_scan, tab_sim = st.tabs(["🔍 Scanner (Corrigido)", "🔮 Simulação"])
    
    # --- SCANNER ---
    with tab_scan:
        df = load_unified_calendar()
        
        if df.empty:
            st.error("⚠️ Erro Crítico: Nenhum calendário carregado. Verifique os arquivos CSV.")
        else:
            # Filtro de datas para Selectbox
            dias_disponiveis = sorted(df['Data'].unique())
            
            # Tenta selecionar hoje
            hoje = datetime.now().strftime("%d/%m/%Y")
            idx = 0
            if hoje in dias_disponiveis:
                idx = dias_disponiveis.index(hoje)
            
            dia_sel = st.selectbox("Escolha a Data do Jogo:", dias_disponiveis, index=idx)
            
            jogos = df[df['Data'] == dia_sel]
            st.info(f"{len(jogos)} jogos encontrados para {dia_sel}.")
            
            if st.button("Rastrear Oportunidades"):
                for _, row in jogos.iterrows():
                    h, a, l, r = row['Mandante'], row['Visitante'], row['Liga'], row['Arbitro']
                    
                    m = calcular_previsao(h, a, r)
                    if m:
                        with st.expander(f"{l} | {h} x {a}"):
                            c1, c2 = st.columns(2)
                            
                            # Analise Cantos
                            ph35 = prob_over_line(m['corners']['h'], 3.5)
                            hh35 = history.get_data(h, l, 'corners', 'homeTeamOver35')
                            
                            pa35 = prob_over_line(m['corners']['a'], 3.5)
                            ha35 = history.get_data(a, l, 'corners', 'awayTeamOver35')
                            
                            with c1:
                                st.write("🚩 **Cantos**")
                                if ph35 > 60:
                                    htxt = f"{hh35[2]}%" if hh35 else "?"
                                    st.write(f"🏠 {h} +3.5: :{get_color(ph35)}[{ph35:.0f}%] (Hist: {htxt})")
                                if pa35 > 60:
                                    htxt = f"{ha35[2]}%" if ha35 else "?"
                                    st.write(f"✈️ {a} +3.5: :{get_color(pa35)}[{pa35:.0f}%] (Hist: {htxt})")
                                    
                            # Analise Cartões
                            with c2:
                                st.write("🟨 **Cartões**")
                                rf = get_referee_factor(r)
                                st.caption(f"Juiz: {r} (Fator: {rf:.2f}x)")
                                
                                pk15h = prob_over_line(m['cards']['h'], 1.5)
                                if pk15h > 50: st.write(f"🏠 {h} +1.5: {pk15h:.0f}%")
                                
                                pk15a = prob_over_line(m['cards']['a'], 1.5)
                                if pk15a > 50: st.write(f"✈️ {a} +1.5: {pk15a:.0f}%")

    # --- SIMULAÇÃO ---
    with tab_sim:
        st.subheader("Simulador Avançado")
        c1, c2, c3 = st.columns(3)
        
        tl = team_list if team_list else ["Carregando..."]
        home = c1.selectbox("Mandante", tl, index=0)
        away = c2.selectbox("Visitante", tl, index=1 if len(tl)>1 else 0)
        
        # Lista de Árbitros Fundida
        ref_list = sorted(list(referees_db.keys()))
        referee = c3.selectbox("Árbitro", ["Neutro"] + ref_list)
        
        if st.button("Analisar"):
            ref_n = referee if referee != "Neutro" else None
            m = calcular_previsao(home, away, ref_n)
            liga_est = stats_db.get(home, {}).get('league', 'Premier League')
            
            if m:
                st.divider()
                k1, k2 = st.columns(2)
                
                # ESCANTEIOS
                with k1:
                    st.info(f"🚩 Cantos Totais: {m['corners']['t']:.2f}")
                    
                    # Casa
                    p35 = prob_over_line(m['corners']['h'], 3.5)
                    h35 = history.get_data(home, liga_est, 'corners', 'homeTeamOver35')
                    htxt = f"{h35[2]}% ({h35[1]}/{h35[0]})" if h35 else "N/A"
                    st.write(f"🏠 {home} +3.5: :{get_color(p35)}[{p35:.0f}%] | Hist: **{htxt}**")
                    
                    p45 = prob_over_line(m['corners']['h'], 4.5)
                    h45 = history.get_data(home, liga_est, 'corners', 'homeTeamOver45')
                    htxt = f"{h45[2]}% ({h45[1]}/{h45[0]})" if h45 else "N/A"
                    st.write(f"🏠 {home} +4.5: :{get_color(p45)}[{p45:.0f}%] | Hist: **{htxt}**")
                    
                    st.markdown("---")
                    
                    # Fora
                    p35a = prob_over_line(m['corners']['a'], 3.5)
                    h35a = history.get_data(away, liga_est, 'corners', 'awayTeamOver35')
                    htxt = f"{h35a[2]}% ({h35a[1]}/{h35a[0]})" if h35a else "N/A"
                    st.write(f"✈️ {away} +3.5: :{get_color(p35a)}[{p35a:.0f}%] | Hist: **{htxt}**")

                # CARTÕES
                with k2:
                    st.warning(f"🟨 Cartões Totais: {m['cards']['t']:.2f}")
                    
                    # Casa
                    p15 = prob_over_line(m['cards']['h'], 1.5)
                    h15 = history.get_data(home, liga_est, 'cards', 'homeCardsOver15')
                    htxt = f"{h15[2]}% ({h15[1]}/{h15[0]})" if h15 else "N/A"
                    st.write(f"🏠 {home} +1.5: :{get_color(p15)}[{p15:.0f}%] | Hist: **{htxt}**")
                    
                    # Fora
                    p15a = prob_over_line(m['cards']['a'], 1.5)
                    h15a = history.get_data(away, liga_est, 'cards', 'awayCardsOver15')
                    htxt = f"{h15a[2]}% ({h15a[1]}/{h15a[0]})" if h15a else "N/A"
                    st.write(f"✈️ {away} +1.5: :{get_color(p15a)}[{p15a:.0f}%] | Hist: **{htxt}**")

if __name__ == "__main__":
    render_dashboard()
