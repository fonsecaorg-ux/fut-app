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

# ==============================================================================
# 0. CONFIGURAÇÃO
# ==============================================================================
st.set_page_config(page_title="FutPrevisão V8.2 (Lines Fix)", layout="wide", page_icon="⚽")

# ==============================================================================
# 1. ARQUIVOS & MAPAS
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
# 2. SISTEMA DE ÁRBITROS (FUSÃO DUPLA)
# ==============================================================================
@st.cache_data(ttl=3600)
def load_referees_unified():
    refs = {}
    # Arquivo 1 (Simples)
    if os.path.exists("arbitros.csv"):
        try:
            df1 = pd.read_csv("arbitros.csv")
            for _, r in df1.iterrows():
                refs[str(r['Nome']).strip()] = float(r['Fator'])
        except: pass
    # Arquivo 2 (Detalhado)
    if os.path.exists("arbitros_5_ligas_2025_2026.csv"):
        try:
            df2 = pd.read_csv("arbitros_5_ligas_2025_2026.csv")
            for _, r in df2.iterrows():
                nome = str(r['Arbitro']).strip()
                # Converte Média para Fator (Base 4.0)
                media = float(r['Media_Cartoes_Por_Jogo'])
                refs[nome] = media / 4.0
        except: pass
    return refs

referees_db = load_referees_unified()

def get_ref_factor(name):
    if not name or name == "Neutro": return 1.0
    if name in referees_db: return referees_db[name]
    # Fuzzy
    match = difflib.get_close_matches(name, referees_db.keys(), n=1, cutoff=0.7)
    if match: return referees_db[match[0]]
    return 1.0

# ==============================================================================
# 3. DATAS & CALENDÁRIO
# ==============================================================================
def fix_date(d):
    try:
        s = str(d).strip()
        for fmt in ["%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d", "%d-%m-%Y"]:
            try: return datetime.strptime(s, fmt).strftime("%d/%m/%Y")
            except: continue
        return s
    except: return None

@st.cache_data(ttl=600)
def load_unified_calendar():
    all_games = []
    for l_name, f_data in LEAGUE_FILES.items():
        if os.path.exists(f_data['csv']):
            try:
                try: df = pd.read_csv(f_data['csv'], encoding='latin1', dtype=str)
                except: df = pd.read_csv(f_data['csv'], dtype=str)
                
                cols = df.columns
                d_col = next((c for c in cols if c in ['Date', 'Data']), None)
                h_col = next((c for c in cols if c in ['HomeTeam', 'Mandante']), None)
                a_col = next((c for c in cols if c in ['AwayTeam', 'Visitante']), None)
                r_col = next((c for c in cols if c in ['Referee', 'Arbitro']), None)
                
                if d_col and h_col and a_col:
                    temp = pd.DataFrame()
                    temp['Data'] = df[d_col].apply(fix_date)
                    temp['Mandante'] = df[h_col]
                    temp['Visitante'] = df[a_col]
                    temp['Liga'] = l_name
                    temp['Arbitro'] = df[r_col] if r_col else "Desconhecido"
                    # Identifica se é jogo futuro (sem gols marcados)
                    # Verifica se tem FTHG e se está vazio
                    res_col = next((c for c in cols if c in ['FTHG', 'HG', 'GolsMandante']), None)
                    if res_col:
                        temp['Status'] = df[res_col].apply(lambda x: 'Finalizado' if pd.notna(x) and str(x).strip() != '' else 'Futuro')
                    else:
                        temp['Status'] = 'Futuro' # Assume futuro se não tiver coluna de gols
                        
                    all_games.append(temp.dropna(subset=['Data']))
            except: pass
            
    if all_games:
        final = pd.concat(all_games, ignore_index=True)
        # Ordena datas (converte para datetime para sortear, depois volta string)
        final['DtObj'] = pd.to_datetime(final['Data'], format="%d/%m/%Y", errors='coerce')
        final = final.sort_values('DtObj', ascending=False) # Mais recentes primeiro
        return final
    return pd.DataFrame()

# ==============================================================================
# 4. LEARNING & HISTÓRICO
# ==============================================================================
@st.cache_data(ttl=3600)
def learn_stats():
    db = {}
    for liga, files in LEAGUE_FILES.items():
        f = files['csv']
        if os.path.exists(f):
            try:
                try: df = pd.read_csv(f, encoding='latin1')
                except: df = pd.read_csv(f)
                
                cols = df.columns
                h_c = 'HomeTeam' if 'HomeTeam' in cols else 'Mandante'
                a_c = 'AwayTeam' if 'AwayTeam' in cols else 'Visitante'
                has_corn = 'HC' in cols and 'AC' in cols
                has_card = 'HY' in cols and 'AY' in cols
                
                if h_c in cols:
                    teams = set(df[h_c].dropna().unique()).union(set(df[a_c].dropna().unique()))
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

stats_db = learn_stats()
team_list = sorted(list(stats_db.keys()))

class HistoryLoader:
    def __init__(self):
        self.corn = {}
        self.card = {}
        self.load()
    def load(self):
        for l, f in LEAGUE_FILES.items():
            if f['txt'] and os.path.exists(f['txt']):
                try: self.corn[l] = json.load(open(f['txt'], encoding='utf-8'))
                except: pass
            if f['txt_cards'] and os.path.exists(f['txt_cards']):
                try: self.card[l] = json.load(open(f['txt_cards'], encoding='utf-8'))
                except: pass
    
    def get(self, team, liga, mkt, key):
        src = self.corn if mkt == 'corners' else self.card
        # Tenta liga
        if liga in src:
            res = self._search(src[liga], team, key)
            if res: return res
        # Tenta global
        for l in src:
            res = self._search(src[l], team, key)
            if res: return res
        return None

    def _search(self, data, team, key):
        teams = [t['teamName'] for t in data.get('teams', [])]
        match = difflib.get_close_matches(team, teams, n=1, cutoff=0.6)
        if match:
            for t in data['teams']:
                if t['teamName'] == match[0]:
                    d = t.get(key)
                    if d and len(d) >= 3: return d[0], d[1], d[2]
        return None

hist = HistoryLoader()

# ==============================================================================
# 5. MATEMÁTICA V8.2 (LINHAS ESPECÍFICAS)
# ==============================================================================
def poisson_prob(k, lam): return (math.exp(-lam) * (lam ** k)) / math.factorial(int(k))
def prob_over(lam, line):
    cdf = sum(poisson_prob(i, lam) for i in range(int(line) + 1))
    return max(0.0, (1 - cdf) * 100)

def normalize(name):
    if name in stats_db: return name
    m = difflib.get_close_matches(name, stats_db.keys(), n=1, cutoff=0.6)
    return m[0] if m else None

def calcular(h_name, a_name, ref_name):
    h = normalize(h_name)
    a = normalize(a_name)
    if not h or not a: return None
    
    s_h = stats_db.get(h, {'corners': 5.0, 'cards': 2.0})
    s_a = stats_db.get(a, {'corners': 4.0, 'cards': 2.0})
    
    rf = get_ref_factor(ref_name)
    
    # Modelagem
    c_h_exp = s_h['corners'] * 1.15 # Mandante tem mais volume
    c_a_exp = s_a['corners'] * 0.90 # Visitante recua
    
    k_h_exp = s_h['cards'] * rf
    k_a_exp = s_a['cards'] * rf
    
    return {
        'corners': {'h': c_h_exp, 'a': c_a_exp, 't': c_h_exp + c_a_exp},
        'cards': {'h': k_h_exp, 'a': k_a_exp, 't': k_h_exp + k_a_exp}
    }

def color(p): return "green" if p >= 65 else ("orange" if p >= 50 else "red")

# ==============================================================================
# 6. DASHBOARD
# ==============================================================================
def render_dashboard():
    st.title("🛡️ FutPrevisão V8.2 (Scanner de Linhas)")
    
    tab_scan, tab_sim = st.tabs(["🔍 Scanner", "🔮 Simulação"])
    
    # --- SCANNER CORRIGIDO ---
    with tab_scan:
        df = load_unified_calendar()
        if df.empty:
            st.warning("⚠️ CSVs não carregados ou vazios.")
        else:
            # Filtro Data
            dates = df['Data'].unique()
            hoje = datetime.now().strftime("%d/%m/%Y")
            idx = 0
            # Tenta achar hoje, senão pega o primeiro (mais recente devido ao sort)
            if hoje in list(dates): idx = list(dates).index(hoje)
            
            sel_date = st.selectbox("Data do Jogo", dates, index=idx)
            
            # Filtra Jogos
            jogos = df[df['Data'] == sel_date]
            st.info(f"{len(jogos)} jogos listados para {sel_date}.")
            
            if st.button("🔎 Escanear Linhas (3.5/4.5 & 1.5/2.5)"):
                for _, row in jogos.iterrows():
                    h, a, l, r = row['Mandante'], row['Visitante'], row['Liga'], row['Arbitro']
                    m = calcular(h, a, r)
                    
                    if m:
                        with st.expander(f"{l} | {h} x {a} | Juiz: {r}"):
                            c1, c2 = st.columns(2)
                            
                            # --- CANTO ESQUERDO: ESCANTEIOS ---
                            with c1:
                                st.markdown("### 🚩 Escanteios")
                                
                                # CASA
                                ph35 = prob_over(m['corners']['h'], 3.5)
                                ph45 = prob_over(m['corners']['h'], 4.5)
                                hh35 = hist.get(h, l, 'corners', 'homeTeamOver35')
                                
                                st.markdown(f"**🏠 {h}**")
                                st.markdown(f"Over 3.5: :{color(ph35)}[{ph35:.0f}%] | Hist: {hh35[2] if hh35 else 'N/A'}%")
                                st.markdown(f"Over 4.5: :{color(ph45)}[{ph45:.0f}%]")
                                
                                st.divider()
                                
                                # FORA
                                pa35 = prob_over(m['corners']['a'], 3.5)
                                pa45 = prob_over(m['corners']['a'], 4.5)
                                ha35 = hist.get(a, l, 'corners', 'awayTeamOver35')
                                
                                st.markdown(f"**✈️ {a}**")
                                st.markdown(f"Over 3.5: :{color(pa35)}[{pa35:.0f}%] | Hist: {ha35[2] if ha35 else 'N/A'}%")
                                st.markdown(f"Over 4.5: :{color(pa45)}[{pa45:.0f}%]")

                            # --- CANTO DIREITO: CARTÕES ---
                            with c2:
                                st.markdown("### 🟨 Cartões")
                                rf = get_ref_factor(r)
                                st.caption(f"Rigidez do Juiz: {rf:.2f}x")
                                
                                # CASA
                                kh15 = prob_over(m['cards']['h'], 1.5)
                                kh25 = prob_over(m['cards']['h'], 2.5)
                                hist_k_h = hist.get(h, l, 'cards', 'homeCardsOver15')
                                
                                st.markdown(f"**🏠 {h}**")
                                st.markdown(f"Over 1.5: :{color(kh15)}[{kh15:.0f}%] | Hist: {hist_k_h[2] if hist_k_h else 'N/A'}%")
                                st.markdown(f"Over 2.5: :{color(kh25)}[{kh25:.0f}%]")
                                
                                st.divider()
                                
                                # FORA
                                ka15 = prob_over(m['cards']['a'], 1.5)
                                ka25 = prob_over(m['cards']['a'], 2.5)
                                hist_k_a = hist.get(a, l, 'cards', 'awayCardsOver15')
                                
                                st.markdown(f"**✈️ {a}**")
                                st.markdown(f"Over 1.5: :{color(ka15)}[{ka15:.0f}%] | Hist: {hist_k_a[2] if hist_k_a else 'N/A'}%")
                                st.markdown(f"Over 2.5: :{color(ka25)}[{ka25:.0f}%]")

    # --- SIMULAÇÃO ---
    with tab_sim:
        st.subheader("Simulação Manual")
        c1, c2, c3 = st.columns(3)
        tl = team_list if team_list else ["Carregando..."]
        
        home = c1.selectbox("Mandante", tl, index=0)
        away = c2.selectbox("Visitante", tl, index=1 if len(tl)>1 else 0)
        
        ref_keys = sorted(list(referees_db.keys()))
        ref = c3.selectbox("Árbitro", ["Neutro"] + ref_keys)
        
        if st.button("Processar Dados"):
            rn = ref if ref != "Neutro" else None
            m = calcular(home, away, rn)
            l_est = stats_db.get(home, {}).get('league', 'Premier League')
            
            if m:
                st.write("---")
                k1, k2 = st.columns(2)
                
                # ESCANTEIOS
                with k1:
                    st.info("🚩 Escanteios (Linhas Individuais)")
                    
                    # CASA
                    p35 = prob_over(m['corners']['h'], 3.5)
                    p45 = prob_over(m['corners']['h'], 4.5)
                    h35 = hist.get(home, l_est, 'corners', 'homeTeamOver35')
                    
                    st.write(f"**🏠 {home}**")
                    st.write(f"+3.5: :{color(p35)}[{p35:.0f}%] (Hist: {h35[2] if h35 else 'N/A'}%)")
                    st.write(f"+4.5: :{color(p45)}[{p45:.0f}%]")
                    
                    st.write("---")
                    
                    # FORA
                    pa35 = prob_over(m['corners']['a'], 3.5)
                    pa45 = prob_over(m['corners']['a'], 4.5)
                    ha35 = hist.get(away, l_est, 'corners', 'awayTeamOver35')
                    
                    st.write(f"**✈️ {away}**")
                    st.write(f"+3.5: :{color(pa35)}[{pa35:.0f}%] (Hist: {ha35[2] if ha35 else 'N/A'}%)")
                    st.write(f"+4.5: :{color(pa45)}[{pa45:.0f}%]")
                
                # CARTÕES
                with k2:
                    st.warning("🟨 Cartões (Linhas Individuais)")
                    
                    # CASA
                    pk15 = prob_over(m['cards']['h'], 1.5)
                    pk25 = prob_over(m['cards']['h'], 2.5)
                    hk15 = hist.get(home, l_est, 'cards', 'homeCardsOver15')
                    
                    st.write(f"**🏠 {home}**")
                    st.write(f"+1.5: :{color(pk15)}[{pk15:.0f}%] (Hist: {hk15[2] if hk15 else 'N/A'}%)")
                    st.write(f"+2.5: :{color(pk25)}[{pk25:.0f}%]")
                    
                    st.write("---")
                    
                    # FORA
                    pka15 = prob_over(m['cards']['a'], 1.5)
                    pka25 = prob_over(m['cards']['a'], 2.5)
                    hka15 = hist.get(away, l_est, 'cards', 'awayCardsOver15')
                    
                    st.write(f"**✈️ {away}**")
                    st.write(f"+1.5: :{color(pka15)}[{pka15:.0f}%] (Hist: {hka15[2] if hka15 else 'N/A'}%)")
                    st.write(f"+2.5: :{color(pka25)}[{pka25:.0f}%]")

if __name__ == "__main__":
    render_dashboard()
