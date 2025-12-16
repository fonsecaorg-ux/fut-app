import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import math
import difflib
from datetime import datetime

# ==============================================================================
# 0. CONFIGURAÇÃO
# ==============================================================================
st.set_page_config(page_title="FutPrevisão V11.1 (Card Safe)", layout="wide", page_icon="⚽")

# CSS: Barras verdes agora indicam alta confiança real
st.markdown("""
<style>
    .stProgress > div > div > div > div { background-color: #4CAF50; }
</style>
""", unsafe_allow_html=True)

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

LIGA_MAP = {
    "Trendyol Süper Lig": "Süper Lig (TUR)", "Scottish Premiership": "Premiership (SCO)",
    "Pro League": "Pro League (BEL)", "Bundesliga 2": "Bundesliga 2",
    "Championship": "Championship", "Premier League": "Premier League",
    "La Liga": "La Liga", "Serie A": "Serie A", "Bundesliga": "Bundesliga", "Ligue 1": "Ligue 1"
}

# ==============================================================================
# 2. SISTEMA DE ÁRBITROS (COM PENALIDADE NA DÚVIDA)
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
    # AJUSTE V11.1: Se não sabemos o juiz, assumimos que ele dá POUCO cartão (0.9) para ser seguro.
    if not name or str(name).lower() in ["nan", "none", "neutro", "desconhecido"]: return 0.90
    if name in referees_db: return referees_db[name]
    match = difflib.get_close_matches(name, referees_db.keys(), n=1, cutoff=0.7)
    return referees_db[match[0]] if match else 0.90 # Fallback seguro

# ==============================================================================
# 3. LEITOR DO CALENDÁRIO
# ==============================================================================
@st.cache_data(ttl=600)
def load_calendar_file():
    f = "calendario_ligas.csv"
    if not os.path.exists(f): return pd.DataFrame(), f"Arquivo '{f}' não encontrado."
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
# 4. ESTATÍSTICAS
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
                
                if h_c:
                    teams = set(df[h_c].dropna().unique()).union(set(df[a_c].dropna().unique()))
                    for t in teams:
                        hg = df[(df[h_c] == t) & (df['HC'].notna() if has_corn else True)]
                        ag = df[(df[a_c] == t) & (df['AC'].notna() if has_corn else True)]
                        n = len(hg) + len(ag)
                        if n < 3: continue
                        c = ((hg['HC'].sum() + ag['AC'].sum()) / n) if has_corn else 5.0
                        k = ((hg['HY'].sum() + ag['AY'].sum()) / n) if has_card else 2.0
                        db[t] = {'corners': c, 'cards': k, 'league': liga_key}
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
            if f['txt'] and os.path.exists(f['txt']):
                try: self.corn[l] = json.load(open(f['txt'], encoding='utf-8'))
                except: pass
            if f['txt_cards'] and os.path.exists(f['txt_cards']):
                try: self.card[l] = json.load(open(f['txt_cards'], encoding='utf-8'))
                except: pass
    def get(self, team, liga_key, mkt, key):
        src = self.corn if mkt == 'corners' else self.card
        if liga_key in src:
            res = self._find(src[liga_key], team, key)
            if res: return res
        for l in src:
            res = self._find(src[l], team, key)
            if res: return res
        return None
    def _find(self, data, team, key):
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
# 5. MATEMÁTICA (MODIFICADA V11.1)
# ==============================================================================
def poisson_prob(k, lam): return (math.exp(-lam) * (lam ** k)) / math.factorial(int(k))
def prob_over(lam, line):
    cdf = sum(poisson_prob(i, lam) for i in range(int(line) + 1))
    return max(0.0, (1 - cdf) * 100)

def normalize_team(name):
    if name in stats_db: return name
    m = difflib.get_close_matches(name, stats_db.keys(), n=1, cutoff=0.6)
    return m[0] if m else None

def calcular_jogo(home_raw, away_raw, ref_name):
    h = normalize_team(home_raw)
    a = normalize_team(away_raw)
    s_h = stats_db.get(h, {'corners': 5.0, 'cards': 2.0})
    s_a = stats_db.get(a, {'corners': 4.5, 'cards': 2.0})
    rf = get_ref_factor(ref_name)
    
    # AJUSTE 1: Escanteios mantêm a lógica agressiva
    c_h_exp = s_h['corners'] * 1.15
    c_a_exp = s_a['corners'] * 0.90
    
    # AJUSTE 2: Cartões recebem "Nerf" (Redutor de Segurança) de 15%
    # Multiplicamos por 0.85 para ser pessimista.
    # Só vai dar Green se o time for MUITO faltoso e o juiz MUITO rigoroso.
    k_h_exp = (s_h['cards'] * 0.85) * rf
    k_a_exp = (s_a['cards'] * 0.85) * rf
    
    return {'corners': {'h': c_h_exp, 'a': c_a_exp}, 'cards': {'h': k_h_exp, 'a': k_a_exp}}

def fmt_hist(data): return f"{data[1]}/{data[0]}" if data else "N/A"

def check_elite(prob, hist_data, is_card=False):
    # AJUSTE 3: Régua mais alta para Cartões
    cutoff = 78 if is_card else 75 # Escanteios 75%, Cartões 78%
    if hist_data: return float(hist_data[2]) >= 70 and prob >= cutoff
    return prob >= (cutoff + 5) # Se não tem historico, exige ainda mais da matemática

def get_bar_color(prob): return "green" if prob >= 70 else ("orange" if prob >= 50 else "red")

# ==============================================================================
# 6. UI
# ==============================================================================
def render_match_row(t_casa, t_visitante, liga_nome, hora, liga_key):
    with st.expander(f"⏰ {hora} | {liga_nome} | {t_casa} x {t_visitante}"):
        m = calcular_jogo(t_casa, t_visitante, None)
        if m:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("##### 🚩 Escanteios")
                ph35 = prob_over(m['corners']['h'], 3.5)
                hh35 = hist.get(t_casa, liga_key, 'corners', 'homeTeamOver35')
                st.write(f"🏠 {t_casa} +3.5: **{ph35:.0f}%** ({fmt_hist(hh35)})")
                st.progress(int(ph35)/100)
                
                pa35 = prob_over(m['corners']['a'], 3.5)
                ha35 = hist.get(t_visitante, liga_key, 'corners', 'awayTeamOver35')
                st.write(f"✈️ {t_visitante} +3.5: **{pa35:.0f}%** ({fmt_hist(ha35)})")
                st.progress(int(pa35)/100)

            with c2:
                st.markdown("##### 🟨 Cartões (Modo Seguro)")
                kh15 = prob_over(m['cards']['h'], 1.5)
                hk15 = hist.get(t_casa, liga_key, 'cards', 'homeCardsOver15')
                st.write(f"🏠 {t_casa} +1.5: **{kh15:.0f}%** ({fmt_hist(hk15)})")
                st.progress(int(kh15)/100)
                
                ka15 = prob_over(m['cards']['a'], 1.5)
                hka15 = hist.get(t_visitante, liga_key, 'cards', 'awayCardsOver15')
                st.write(f"✈️ {t_visitante} +1.5: **{ka15:.0f}%** ({fmt_hist(hka15)})")
                st.progress(int(ka15)/100)

def render_dashboard():
    st.title("🛡️ FutPrevisão V11.1 (Card Safe Mode)")
    
    tab_day, tab_sim = st.tabs(["📅 Partidas do Dia", "🔮 Simulação Manual"])
    
    with tab_day:
        df, status = load_calendar_file()
        if df.empty:
            st.error(f"Erro no Calendário: {status}")
        else:
            dates = df['Data'].unique()
            hoje = datetime.now().strftime("%d/%m/%Y")
            idx = list(dates).index(hoje) if hoje in dates else 0
            
            c_d, c_f = st.columns([1, 2])
            sel_date = c_d.selectbox("Data:", dates, index=idx)
            jogos_dia = df[df['Data'] == sel_date]
            
            st.markdown("---")
            if st.button("📡 Rastrear Oportunidades (Filtro Rígido)"):
                found = False
                st.info("Filtrando jogos com alta segurança...")
                
                cols = st.columns(3)
                idx = 0
                for _, row in jogos_dia.iterrows():
                    tc, tv, lig = row['Time_Casa'], row['Time_Visitante'], row['Liga']
                    lk = LIGA_MAP.get(lig, "Premier League")
                    m = calcular_jogo(tc, tv, None)
                    
                    if m:
                        msg = ""
                        # Cantos (IsCard=False)
                        if check_elite(prob_over(m['corners']['h'], 3.5), hist.get(tc, lk, 'corners', 'homeTeamOver35'), False):
                            msg += f"🚩 {tc} +3.5 Cantos\n"
                        if check_elite(prob_over(m['corners']['a'], 3.5), hist.get(tv, lk, 'corners', 'awayTeamOver35'), False):
                            msg += f"🚩 {tv} +3.5 Cantos\n"
                        
                        # Cartões (IsCard=True -> Mais rigoroso)
                        if check_elite(prob_over(m['cards']['h'], 1.5), hist.get(tc, lk, 'cards', 'homeCardsOver15'), True):
                            msg += f"🟨 {tc} +1.5 Cartões\n"
                        if check_elite(prob_over(m['cards']['a'], 1.5), hist.get(tv, lk, 'cards', 'awayCardsOver15'), True):
                            msg += f"🟨 {tv} +1.5 Cartões\n"
                        
                        if msg:
                            found = True
                            with cols[idx % 3]:
                                st.success(f"**{lig}**\n\n{msg}")
                            idx += 1
                if not found: st.warning("Nenhuma aposta segura encontrada hoje.")
                st.markdown("---")

            ligas = jogos_dia['Liga'].unique()
            sel_l = st.multiselect("Filtrar Ligas:", ligas, default=ligas)
            jogos_f = jogos_dia[jogos_dia['Liga'].isin(sel_l)]
            
            st.write(f"Lista de Jogos ({len(jogos_f)}):")
            for _, row in jogos_f.iterrows():
                render_match_row(row['Time_Casa'], row['Time_Visitante'], row['Liga'], row['Hora'], LIGA_MAP.get(row['Liga'], "Premier League"))

    with tab_sim:
        st.subheader("Simulador Avançado")
        c1, c2, c3 = st.columns(3)
        tl = team_list_all if team_list_all else ["Carregando..."]
        home = c1.selectbox("Mandante", tl, index=0)
        away = c2.selectbox("Visitante", tl, index=1 if len(tl)>1 else 0)
        ref_keys = sorted(list(referees_db.keys()))
        ref = c3.selectbox("Árbitro", ["Neutro"] + ref_keys)
        
        if st.button("Simular"):
            rn = ref if ref != "Neutro" else None
            m = calcular_jogo(home, away, rn)
            if m:
                st.write("---")
                k1, k2 = st.columns(2)
                with k1:
                    st.info(f"🚩 Escanteios (Exp: {m['corners']['h']:.1f} x {m['corners']['a']:.1f})")
                    p = prob_over(m['corners']['h'], 3.5)
                    st.write(f"🏠 {home} +3.5: **{p:.0f}%**"); st.progress(int(p)/100)
                with k2:
                    st.warning(f"🟨 Cartões (Exp: {m['cards']['h']:.1f} x {m['cards']['a']:.1f})")
                    p = prob_over(m['cards']['h'], 1.5)
                    st.write(f"🏠 {home} +1.5: **{p:.0f}%**"); st.progress(int(p)/100)
                    st.caption("Nota: Probabilidade reduzida em 15% por segurança.")

if __name__ == "__main__":
    render_dashboard()
