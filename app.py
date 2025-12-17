import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import math
import difflib
import re
from datetime import datetime
from PIL import Image

# ==============================================================================
# 0. CONFIGURAÇÃO
# ==============================================================================
st.set_page_config(page_title="FutPrevisão V13.1 (Fix)", layout="wide", page_icon="🏆")

st.markdown("""
<style>
    .stProgress > div > div > div > div { background-color: #4CAF50; }
    .scan-card { background: #f0f8ff; border: 1px solid #bce8f1; padding: 10px; border-radius: 5px; margin-bottom: 8px; }
    .scan-high { border-left: 5px solid #28a745; }
</style>
""", unsafe_allow_html=True)

# Tenta OCR (Opcional)
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
# 1. MAPEAMENTO DE ARQUIVOS
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

# Mapeamento Estendido para Ligas Menores
NAME_MAPPING = {
    "Man City": "Man City", "Man Utd": "Man United", "Sheffield Wed": "Sheffield Wednesday",
    "Forest": "Nott'm Forest", "Wolves": "Wolverhampton", "Spurs": "Tottenham",
    "Atl. Madrid": "Atl Madrid", "Athletic Club": "Ath Bilbao",
    "PSG": "Paris SG", "St Etienne": "Saint-Etienne",
    "Fenerbahce": "Fenerbahçe", "Galatasaray": "Galatasaray",
    "Rangers": "Rangers", "Celtic": "Celtic"
}

# ==============================================================================
# 2. SISTEMA DE ÁRBITROS
# ==============================================================================
@st.cache_data(ttl=3600)
def load_referees_unified():
    refs = {}
    if os.path.exists("arbitros_5_ligas_2025_2026.csv"):
        try:
            df = pd.read_csv("arbitros_5_ligas_2025_2026.csv")
            for _, r in df.iterrows(): refs[str(r['Arbitro']).strip()] = float(r['Media_Cartoes_Por_Jogo']) / 4.0
        except: pass
    if os.path.exists("arbitros.csv"): 
        try:
            df = pd.read_csv("arbitros.csv")
            for _, r in df.iterrows():
                nome = str(r['Nome']).strip()
                if nome not in refs: refs[nome] = float(r['Fator'])
        except: pass
    return refs

referees_db = load_referees_unified()

def get_ref_factor(name):
    if not name or str(name).lower() in ["nan", "none", "neutro", "desconhecido"]: return 1.0
    if name in referees_db: return referees_db[name]
    match = difflib.get_close_matches(name, referees_db.keys(), n=1, cutoff=0.7)
    return referees_db[match[0]] if match else 1.0

# ==============================================================================
# 3. LEITOR DE CALENDÁRIO
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
def load_calendar_file():
    f = "calendario_ligas.csv"
    if not os.path.exists(f): return pd.DataFrame(), "Arquivo não encontrado"
    try:
        try: df = pd.read_csv(f, dtype=str)
        except: df = pd.read_csv(f, dtype=str, encoding='latin1')
        df.columns = [c.strip().replace(' ', '_') for c in df.columns]
        req = ['Data', 'Liga', 'Time_Casa', 'Time_Visitante']
        if not set(req).issubset(df.columns): return pd.DataFrame(), "Colunas incorretas"
        df['DtObj'] = pd.to_datetime(df['Data'], format="%d/%m/%Y", errors='coerce')
        df = df.dropna(subset=['DtObj']).sort_values(['DtObj', 'Hora'])
        return df, "Sucesso"
    except Exception as e: return pd.DataFrame(), str(e)

# ==============================================================================
# 4. ESTATÍSTICAS & HISTÓRICO
# ==============================================================================
@st.cache_data(ttl=3600)
def learn_stats():
    db = {}
    for liga_key, files in LEAGUE_FILES.items():
        if os.path.exists(files['csv']):
            try:
                try: df = pd.read_csv(files['csv'], encoding='latin1')
                except: df = pd.read_csv(files['csv'])
                cols = df.columns
                h_c = next((c for c in cols if c in ['HomeTeam', 'Mandante']), None)
                a_c = next((c for c in cols if c in ['AwayTeam', 'Visitante']), None)
                has_corn = 'HC' in cols and 'AC' in cols
                has_card = 'HY' in cols and 'AY' in cols
                
                # Faltas e Gols (V12)
                has_foul = 'HF' in cols and 'AF' in cols
                
                if h_c:
                    teams = set(df[h_c].dropna().unique()).union(set(df[a_c].dropna().unique()))
                    for t in teams:
                        hg = df[(df[h_c] == t) & (df['HC'].notna() if has_corn else True)]
                        ag = df[(df[a_c] == t) & (df['AC'].notna() if has_corn else True)]
                        n = len(hg) + len(ag)
                        if n < 3: continue
                        
                        c = ((hg['HC'].sum() + ag['AC'].sum()) / n) if has_corn else 5.0
                        k = ((hg['HY'].sum() + ag['AY'].sum()) / n) if has_card else 2.0
                        f = ((hg['HF'].sum() + ag['AF'].sum()) / n) if has_foul else 11.0
                        
                        try:
                            g_f = (hg['FTHG'].astype(float).sum() + ag['FTAG'].astype(float).sum()) / n
                            g_a = (hg['FTAG'].astype(float).sum() + ag['FTHG'].astype(float).sum()) / n
                        except: g_f, g_a = 1.2, 1.2
                        
                        db[t] = {'corners': c, 'cards': k, 'fouls': f, 'goals_f': g_f, 'goals_a': g_a, 'league': liga_key}
            except: pass
    return db

stats_db = learn_stats()
team_list_all = sorted(list(stats_db.keys()))

class HistoryLoader:
    def __init__(self):
        self.corn = {}
        self.card = {}
        self.load()
    def load(self):
        for l, f in LEAGUE_FILES.items():
            # Carregamento seguro (Try/Except por arquivo)
            if f['txt'] and os.path.exists(f['txt']):
                try: 
                    with open(f['txt'], 'r', encoding='utf-8') as file:
                        raw = file.read().strip()
                        if '{' in raw: raw = raw[raw.find('{'):]
                        self.corn[l] = json.load(file)
                except: pass
            if f['txt_cards'] and os.path.exists(f['txt_cards']):
                try: 
                    with open(f['txt_cards'], 'r', encoding='utf-8') as file:
                        raw = file.read().strip()
                        if '{' in raw: raw = raw[raw.find('{'):]
                        self.card[l] = json.load(file)
                except: pass
    
    def get_global(self, team, mkt, key):
        src = self.corn if mkt == 'corners' else self.card
        for l in src:
            res = self._find(src[l], team, key)
            if res: return res
        return None

    def _find(self, data, team, key):
        if not data: return None
        teams = [t['teamName'] for t in data.get('teams', [])]
        
        # Mapeamento Manual primeiro
        target = team
        if team in NAME_MAPPING: target = NAME_MAPPING[team]
        
        match = difflib.get_close_matches(target, teams, n=1, cutoff=0.6)
        if match:
            for t in data['teams']:
                if t['teamName'] == match[0]:
                    d = t.get(key)
                    if d and len(d) >= 3: return d[0], d[1], d[2]
        return None

hist = HistoryLoader()

# ==============================================================================
# 5. MATEMÁTICA CAUSAL (V13)
# ==============================================================================
def poisson_prob(k, lam): return (math.exp(-lam) * (lam ** k)) / math.factorial(int(k))
def prob_over(lam, line):
    try:
        cdf = sum(poisson_prob(i, lam) for i in range(int(line) + 1))
        return max(0.0, (1 - cdf) * 100)
    except: return 0.0

def normalize_team(name):
    if name in stats_db: return name
    m = difflib.get_close_matches(name, stats_db.keys(), n=1, cutoff=0.6)
    return m[0] if m else None

def calcular_jogo(home_raw, away_raw, ref_name):
    h = normalize_team(home_raw)
    a = normalize_team(away_raw)
    
    # Fallback seguro para times não encontrados
    s_h = stats_db.get(h, {'corners': 5.0, 'cards': 2.0, 'fouls': 11.0, 'goals_f': 1.2, 'goals_a': 1.2})
    s_a = stats_db.get(a, {'corners': 4.0, 'cards': 2.0, 'fouls': 11.0, 'goals_f': 1.0, 'goals_a': 1.5})
    
    rf = get_ref_factor(ref_name)
    
    # Lógica V13
    pressao_h = 1.10 if s_h['goals_f'] > 1.8 else 1.0
    c_h_exp = s_h['corners'] * 1.15 * pressao_h
    c_a_exp = s_a['corners'] * 0.90
    
    violencia_h = 1.0 if s_h['fouls'] > 12.5 else 0.85
    violencia_a = 1.0 if s_a['fouls'] > 12.5 else 0.85
    
    k_h_exp = s_h['cards'] * violencia_h * rf
    k_a_exp = s_a['cards'] * violencia_a * rf
    
    g_h_exp = s_h['goals_f'] * s_a['goals_a'] / 1.3
    g_a_exp = s_a['goals_f'] * s_h['goals_a'] / 1.3
    
    return {
        'corners': {'h': c_h_exp, 'a': c_a_exp, 't': c_h_exp + c_a_exp},
        'cards': {'h': k_h_exp, 'a': k_a_exp, 't': k_h_exp + k_a_exp},
        'goals': {'h': g_h_exp, 'a': g_a_exp}
    }

def fmt_hist(data): return f"{data[1]}/{data[0]}" if data else "N/A"
def check_elite(prob, hist_data, is_card=False):
    cutoff = 75 if not is_card else 70 
    if hist_data: return float(hist_data[2]) >= 70 and prob >= cutoff
    return prob >= (cutoff + 5)
def get_color(prob): return "green" if prob >= 70 else ("orange" if prob >= 50 else "red")

# ==============================================================================
# 6. UI
# ==============================================================================
def render_match_row(t_casa, t_visitante, liga_nome, hora):
    with st.expander(f"⏰ {hora} | {liga_nome} | {t_casa} x {t_visitante}"):
        # Garante que não quebre se o nome for estranho
        try:
            m = calcular_jogo(t_casa, t_visitante, None)
            if m:
                c1, c2 = st.columns(2)
                
                with c1:
                    st.markdown("##### 🚩 Escanteios")
                    # Casa
                    ph35 = prob_over(m['corners']['h'], 3.5)
                    hh35 = hist.get_global(t_casa, 'corners', 'homeTeamOver35')
                    st.write(f"🏠 {t_casa} +3.5: :{get_color(ph35)}[{ph35:.0f}%] ({fmt_hist(hh35)})")
                    
                    ph45 = prob_over(m['corners']['h'], 4.5)
                    hh45 = hist.get_global(t_casa, 'corners', 'homeTeamOver45')
                    st.write(f"🏠 {t_casa} +4.5: :{get_color(ph45)}[{ph45:.0f}%] ({fmt_hist(hh45)})")
                    
                    st.markdown("---")
                    
                    # Fora
                    pa35 = prob_over(m['corners']['a'], 3.5)
                    ha35 = hist.get_global(t_visitante, 'corners', 'awayTeamOver35')
                    st.write(f"✈️ {t_visitante} +3.5: :{get_color(pa35)}[{pa35:.0f}%] ({fmt_hist(ha35)})")
                    
                    pa45 = prob_over(m['corners']['a'], 4.5)
                    ha45 = hist.get_global(t_visitante, 'corners', 'awayTeamOver45')
                    st.write(f"✈️ {t_visitante} +4.5: :{get_color(pa45)}[{pa45:.0f}%] ({fmt_hist(ha45)})")

                with c2:
                    st.markdown("##### 🟨 Cartões")
                    # Casa
                    kh15 = prob_over(m['cards']['h'], 1.5)
                    hk15 = hist.get_global(t_casa, 'cards', 'homeCardsOver15')
                    st.write(f"🏠 {t_casa} +1.5: :{get_color(kh15)}[{kh15:.0f}%] ({fmt_hist(hk15)})")

                    kh25 = prob_over(m['cards']['h'], 2.5)
                    hk25 = hist.get_global(t_casa, 'cards', 'homeCardsOver25')
                    st.write(f"🏠 {t_casa} +2.5: :{get_color(kh25)}[{kh25:.0f}%] ({fmt_hist(hk25)})")

                    st.markdown("---")

                    # Fora
                    ka15 = prob_over(m['cards']['a'], 1.5)
                    hka15 = hist.get_global(t_visitante, 'cards', 'awayCardsOver15')
                    st.write(f"✈️ {t_visitante} +1.5: :{get_color(ka15)}[{ka15:.0f}%] ({fmt_hist(hka15)})")

                    ka25 = prob_over(m['cards']['a'], 2.5)
                    hka25 = hist.get_global(t_visitante, 'cards', 'awayCardsOver25')
                    st.write(f"✈️ {t_visitante} +2.5: :{get_color(ka25)}[{ka25:.0f}%] ({fmt_hist(hka25)})")
        except:
            st.error("Erro ao processar este jogo.")

def render_dashboard():
    st.title("🏆 FutPrevisão V13.1 (Fix)")
    
    tab_scan, tab_sim = st.tabs(["🔥 Partidas do Dia", "🔮 Simulação Manual / Copas"])
    
    # --- ABA 1: SCANNER ---
    with tab_scan:
        df, status = load_calendar_file()
        if df.empty:
            st.error(f"Erro no Calendário: {status}")
        else:
            dates = df['Data'].unique()
            hoje = datetime.now().strftime("%d/%m/%Y")
            idx = list(dates).index(hoje) if hoje in dates else 0
            
            sel_date = st.selectbox("Data:", dates, index=idx)
            jogos_dia = df[df['Data'] == sel_date]
            
            st.markdown("---")
            if st.button("📡 Rastrear Oportunidades"):
                found = False
                st.info("Buscando Ouro...")
                cols = st.columns(3)
                idx = 0
                for _, row in jogos_dia.iterrows():
                    tc, tv = row['Mandante'], row['Visitante']
                    m = calcular_jogo(tc, tv, None)
                    
                    if m:
                        msg = ""
                        # Verifica histórico GLOBAL (Motor Copa)
                        hh35 = hist.get_global(tc, 'corners', 'homeTeamOver35')
                        ha35 = hist.get_global(tv, 'corners', 'awayTeamOver35')
                        
                        if check_elite(prob_over(m['corners']['h'], 3.5), hh35, False):
                            msg += f"🚩 {tc} +3.5 C\n"
                        if check_elite(prob_over(m['corners']['a'], 3.5), ha35, False):
                            msg += f"🚩 {tv} +3.5 C\n"
                        
                        if msg:
                            found = True
                            with cols[idx % 3]:
                                st.success(f"**{row['Liga']}**\n\n**{tc} x {tv}**\n\n{msg}")
                            idx += 1
                if not found: st.warning("Sem oportunidades > 75% hoje.")
            
            st.write(f"Jogos ({len(jogos_dia)}):")
            for _, row in jogos_dia.iterrows():
                render_match_row(row['Mandante'], row['Visitante'], row['Liga'], row['Hora'])

    # --- ABA 2: SIMULAÇÃO MANUAL (GLOBAL) ---
    with tab_sim:
        st.subheader("Simulador Universal (Ligas & Copas)")
        c1, c2, c3 = st.columns(3)
        tl = team_list_all if team_list_all else ["Carregando..."]
        
        home = c1.selectbox("Mandante", tl, index=0)
        away = c2.selectbox("Visitante", tl, index=1 if len(tl)>1 else 0)
        
        ref_keys = sorted(list(referees_db.keys()))
        ref = c3.selectbox("Árbitro", ["Neutro"] + ref_keys)
        
        if st.button("Simular Confronto"):
            rn = ref if ref != "Neutro" else None
            m = calcular_jogo(home, away, rn)
            if m:
                st.write("---")
                
                st.metric("⚽ Expectativa de Gols (xG)", f"{m['goals']['h']:.2f} x {m['goals']['a']:.2f}")
                
                k1, k2 = st.columns(2)
                
                # ESCANTEIOS
                with k1:
                    st.info("🚩 Escanteios (Linhas Individuais)")
                    # Casa
                    p35 = prob_over(m['corners']['h'], 3.5)
                    h35 = hist.get_global(home, 'corners', 'homeTeamOver35')
                    st.write(f"🏠 {home} +3.5: :{get_color(p35)}[{p35:.0f}%] ({fmt_hist(h35)})")
                    
                    p45 = prob_over(m['corners']['h'], 4.5)
                    h45 = hist.get_global(home, 'corners', 'homeTeamOver45')
                    st.write(f"🏠 {home} +4.5: :{get_color(p45)}[{p45:.0f}%] ({fmt_hist(h45)})")
                    
                    st.markdown("---")
                    
                    # Fora
                    p35a = prob_over(m['corners']['a'], 3.5)
                    h35a = hist.get_global(away, 'corners', 'awayTeamOver35')
                    st.write(f"✈️ {away} +3.5: :{get_color(p35a)}[{p35a:.0f}%] ({fmt_hist(h35a)})")
                    
                    p45a = prob_over(m['corners']['a'], 4.5)
                    h45a = hist.get_global(away, 'corners', 'awayTeamOver45')
                    st.write(f"✈️ {away} +4.5: :{get_color(p45a)}[{p45a:.0f}%] ({fmt_hist(h45a)})")

                # CARTÕES
                with k2:
                    st.warning("🟨 Cartões (Linhas Individuais)")
                    # Casa
                    p15 = prob_over(m['cards']['h'], 1.5)
                    h15 = hist.get_global(home, 'cards', 'homeCardsOver15')
                    st.write(f"🏠 {home} +1.5: :{get_color(p15)}[{p15:.0f}%] ({fmt_hist(h15)})")
                    
                    p25 = prob_over(m['cards']['h'], 2.5)
                    h25 = hist.get_global(home, 'cards', 'homeCardsOver25')
                    st.write(f"🏠 {home} +2.5: :{get_color(p25)}[{p25:.0f}%] ({fmt_hist(h25)})")
                    
                    st.markdown("---")
                    
                    # Fora
                    p15a = prob_over(m['cards']['a'], 1.5)
                    h15a = hist.get_global(away, 'cards', 'awayCardsOver15')
                    st.write(f"✈️ {away} +1.5: :{get_color(p15a)}[{p15a:.0f}%] ({fmt_hist(h15a)})")
                    
                    p25a = prob_over(m['cards']['a'], 2.5)
                    h25a = hist.get_global(away, 'cards', 'awayCardsOver25')
                    st.write(f"✈️ {away} +2.5: :{get_color(p25a)}[{p25a:.0f}%] ({fmt_hist(h25a)})")

if __name__ == "__main__":
    render_dashboard()