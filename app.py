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
st.set_page_config(page_title="FutPrevisão V12.0 (Causality)", layout="wide", page_icon="⚽")

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
    if not name or str(name).lower() in ["nan", "none", "neutro", "desconhecido"]: return 0.90
    if name in referees_db: return referees_db[name]
    match = difflib.get_close_matches(name, referees_db.keys(), n=1, cutoff=0.7)
    return referees_db[match[0]] if match else 0.90

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
# 4. CÉREBRO ESTATÍSTICO (V12 - AGORA COM FALTAS E CHUTES)
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
                
                # Checa se temos as colunas "Mágicas" (Faltas e Chutes no Gol)
                has_corn = 'HC' in cols and 'AC' in cols
                has_card = 'HY' in cols and 'AY' in cols
                has_foul = 'HF' in cols and 'AF' in cols # Faltas
                has_shot = 'HST' in cols and 'AST' in cols # Chutes no Gol
                
                if h_c:
                    teams = set(df[h_c].dropna().unique()).union(set(df[a_c].dropna().unique()))
                    for t in teams:
                        # Pega apenas jogos com dados preenchidos
                        hg = df[(df[h_c] == t) & (df['HC'].notna() if has_corn else True)]
                        ag = df[(df[a_c] == t) & (df['AC'].notna() if has_corn else True)]
                        n = len(hg) + len(ag)
                        if n < 3: continue
                        
                        # Médias Básicas
                        c = ((hg['HC'].sum() + ag['AC'].sum()) / n) if has_corn else 5.0
                        k = ((hg['HY'].sum() + ag['AY'].sum()) / n) if has_card else 2.0
                        
                        # Médias Avançadas (V12)
                        f = 0.0
                        s = 0.0
                        
                        if has_foul:
                            f = (hg['HF'].sum() + ag['AF'].sum()) / n
                        if has_shot:
                            s = (hg['HST'].sum() + ag['AST'].sum()) / n
                            
                        db[t] = {
                            'corners': c, 
                            'cards': k, 
                            'fouls': f,  # Nova Metrica
                            'shots': s,  # Nova Metrica
                            'league': liga_key
                        }
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
# 5. MATEMÁTICA CAUSAL (V12)
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
    
    # Defaults
    s_h = stats_db.get(h, {'corners': 5.0, 'cards': 2.0, 'fouls': 10.0, 'shots': 4.0})
    s_a = stats_db.get(a, {'corners': 4.5, 'cards': 2.0, 'fouls': 10.0, 'shots': 4.0})
    rf = get_ref_factor(ref_name)
    
    # --- ESCANTEIOS (Com Boost de Pressão) ---
    # Se time chuta muito no gol (> 5.5), ganha boost de 10%
    pressao_h = 1.10 if s_h.get('shots', 0) > 5.5 else 1.0
    pressao_a = 1.10 if s_a.get('shots', 0) > 5.5 else 1.0
    
    c_h_exp = s_h['corners'] * 1.15 * pressao_h
    c_a_exp = s_a['corners'] * 0.90 * pressao_a
    
    # --- CARTÕES (Com Analise de Violência) ---
    # Padrão: Penalidade de 15% (0.85)
    # Mas se o time faz muitas faltas (> 12.5), removemos a penalidade (1.0)
    violencia_h = 1.0 if s_h.get('fouls', 0) > 12.5 else 0.85
    violencia_a = 1.0 if s_a.get('fouls', 0) > 12.5 else 0.85
    
    k_h_exp = s_h['cards'] * violencia_h * rf
    k_a_exp = s_a['cards'] * violencia_a * rf
    
    return {
        'corners': {'h': c_h_exp, 'a': c_a_exp}, 
        'cards': {'h': k_h_exp, 'a': k_a_exp},
        'meta': {'vh': violencia_h == 1.0, 'va': violencia_a == 1.0} # Meta dados para UI
    }

def fmt_hist(data): return f"{data[1]}/{data[0]}" if data else "N/A"

def check_elite(prob, hist_data, is_card=False):
    math_cut = 65 if is_card else 75 
    if hist_data: return float(hist_data[2]) >= 70 and prob >= math_cut
    return prob >= (math_cut + 5)

# ==============================================================================
# 6. UI (COM INDICADORES V12)
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
                color = "green" if ph35 >= 70 else "orange" if ph35 >= 50 else "red"
                st.write(f"🏠 {t_casa} +3.5: :{color}[**{ph35:.0f}%**] ({fmt_hist(hh35)})")
                st.progress(int(ph35)/100)
                
                pa35 = prob_over(m['corners']['a'], 3.5)
                ha35 = hist.get(t_visitante, liga_key, 'corners', 'awayTeamOver35')
                color = "green" if pa35 >= 70 else "orange" if pa35 >= 50 else "red"
                st.write(f"✈️ {t_visitante} +3.5: :{color}[**{pa35:.0f}%**] ({fmt_hist(ha35)})")
                st.progress(int(pa35)/100)

            with c2:
                st.markdown("##### 🟨 Cartões")
                # Indicador visual de time violento
                icon_h = "🔥" if m['meta']['vh'] else "🛡️"
                icon_a = "🔥" if m['meta']['va'] else "🛡️"
                
                kh15 = prob_over(m['cards']['h'], 1.5)
                hk15 = hist.get(t_casa, liga_key, 'cards', 'homeCardsOver15')
                color = "green" if kh15 >= 65 else "orange" if kh15 >= 50 else "red"
                st.write(f"🏠 {t_casa} +1.5: :{color}[**{kh15:.0f}%**] ({fmt_hist(hk15)}) {icon_h}")
                st.progress(int(kh15)/100)
                
                ka15 = prob_over(m['cards']['a'], 1.5)
                hka15 = hist.get(t_visitante, liga_key, 'cards', 'awayCardsOver15')
                color = "green" if ka15 >= 65 else "orange" if ka15 >= 50 else "red"
                st.write(f"✈️ {t_visitante} +1.5: :{color}[**{ka15:.0f}%**] ({fmt_hist(hka15)}) {icon_a}")
                st.progress(int(ka15)/100)
                
                if m['meta']['vh'] or m['meta']['va']:
                    st.caption("🔥 = Time muito faltoso (Penalidade removida)")
                else:
                    st.caption("🛡️ = Modo Seguro (-15%) ativado")

def render_dashboard():
    st.title("🛡️ FutPrevisão V12.0 (Causality Engine)")
    
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
            if st.button("📡 Rastrear Oportunidades"):
                found = False
                st.info("Filtrando com Inteligência Causal...")
                
                cols = st.columns(3)
                idx = 0
                for _, row in jogos_dia.iterrows():
                    tc, tv, lig = row['Time_Casa'], row['Time_Visitante'], row['Liga']
                    lk = LIGA_MAP.get(lig, "Premier League")
                    m = calcular_jogo(tc, tv, None)
                    
                    if m:
                        msg = ""
                        # Cantos
                        if check_elite(prob_over(m['corners']['h'], 3.5), hist.get(tc, lk, 'corners', 'homeTeamOver35'), False):
                            msg += f"🚩 {tc} +3.5 Cantos\n"
                        if check_elite(prob_over(m['corners']['a'], 3.5), hist.get(tv, lk, 'corners', 'awayTeamOver35'), False):
                            msg += f"🚩 {tv} +3.5 Cantos\n"
                        
                        # Cartões
                        if check_elite(prob_over(m['cards']['h'], 1.5), hist.get(tc, lk, 'cards', 'homeCardsOver15'), True):
                            icon = "🔥" if m['meta']['vh'] else ""
                            msg += f"🟨 {tc} +1.5 Cartões {icon}\n"
                        if check_elite(prob_over(m['cards']['a'], 1.5), hist.get(tv, lk, 'cards', 'awayCardsOver15'), True):
                            icon = "🔥" if m['meta']['va'] else ""
                            msg += f"🟨 {tv} +1.5 Cartões {icon}\n"
                        
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
                    color = "green" if p >= 65 else "orange"
                    st.write(f"🏠 {home} +1.5: :{color}[**{p:.0f}%**]"); st.progress(int(p)/100)
                    
                    if m['meta']['vh']: st.error("🔥 Time Violento detectado (Modo Seguro Desligado)")
                    else: st.success("🛡️ Modo Seguro Ativado (-15%)")

if __name__ == "__main__":
    render_dashboard()