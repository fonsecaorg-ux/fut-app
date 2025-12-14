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
st.set_page_config(page_title="FutPrevisão V10.0 (Master)", layout="wide", page_icon="⚽")

# ==============================================================================
# 1. MAPEAMENTO DE ARQUIVOS DE ESTATÍSTICA (O Cérebro)
# ==============================================================================
# Estes arquivos são usados APENAS para calcular médias e histórico.
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

# Mapeia nomes do SEU calendario_ligas.csv para as chaves acima
LIGA_MAP = {
    "Trendyol Süper Lig": "Süper Lig (TUR)",
    "Scottish Premiership": "Premiership (SCO)",
    "Pro League": "Pro League (BEL)",
    "Bundesliga 2": "Bundesliga 2",
    "Championship": "Championship",
    "Premier League": "Premier League",
    "La Liga": "La Liga",
    "Serie A": "Serie A",
    "Bundesliga": "Bundesliga",
    "Ligue 1": "Ligue 1"
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
            for _, r in df.iterrows():
                refs[str(r['Arbitro']).strip()] = float(r['Media_Cartoes_Por_Jogo']) / 4.0
        except: pass
    if os.path.exists("arbitros.csv"): # Fallback/Complemento
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
    # Match Exato
    if name in referees_db: return referees_db[name]
    # Fuzzy Match
    match = difflib.get_close_matches(name, referees_db.keys(), n=1, cutoff=0.7)
    return referees_db[match[0]] if match else 1.0

# ==============================================================================
# 3. LEITOR DO CALENDÁRIO (Fonte: calendario_ligas.csv)
# ==============================================================================
@st.cache_data(ttl=600)
def load_calendar_file():
    f = "calendario_ligas.csv"
    if not os.path.exists(f): return pd.DataFrame()
    
    try:
        df = pd.read_csv(f, dtype=str)
        # Limpa espaços
        df.columns = [c.strip() for c in df.columns]
        
        # Garante colunas essenciais
        req = ['Data', 'Liga', 'Time_Casa', 'Time_Visitante']
        if not set(req).issubset(df.columns): return pd.DataFrame()
        
        # Padroniza Data para ordenação
        df['DtObj'] = pd.to_datetime(df['Data'], format="%d/%m/%Y", errors='coerce')
        df = df.sort_values('DtObj') # Cronológico
        
        return df
    except: return pd.DataFrame()

# ==============================================================================
# 4. CÉREBRO ESTATÍSTICO (Learning + History)
# ==============================================================================
@st.cache_data(ttl=3600)
def learn_stats():
    """Lê os CSVs de cada liga para aprender médias de times"""
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
                        # Filtra apenas jogos COM dados (passado)
                        hg = df[(df[h_c] == t) & (df['HC'].notna() if has_corn else True)]
                        ag = df[(df[a_c] == t) & (df['AC'].notna() if has_corn else True)]
                        n = len(hg) + len(ag)
                        
                        if n < 3: continue # Ignora times sem histórico suficiente
                        
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
        # Tenta na liga especifica
        if liga_key in src:
            res = self._find(src[liga_key], team, key)
            if res: return res
        # Tenta global
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
# 5. MATEMÁTICA (POISSON)
# ==============================================================================
def poisson_prob(k, lam): return (math.exp(-lam) * (lam ** k)) / math.factorial(int(k))
def prob_over(lam, line):
    cdf = sum(poisson_prob(i, lam) for i in range(int(line) + 1))
    return max(0.0, (1 - cdf) * 100)

def normalize_team(name):
    # Tenta achar o nome do time no banco de dados estatístico
    if name in stats_db: return name
    m = difflib.get_close_matches(name, stats_db.keys(), n=1, cutoff=0.6)
    return m[0] if m else None

def calcular_jogo(home_raw, away_raw, ref_name):
    h = normalize_team(home_raw)
    a = normalize_team(away_raw)
    
    # Se não achar o time nas estatísticas, usa média genérica (Coringa)
    s_h = stats_db.get(h, {'corners': 5.0, 'cards': 2.0})
    s_a = stats_db.get(a, {'corners': 4.5, 'cards': 2.0})
    
    rf = get_ref_factor(ref_name)
    
    # Modelo Matemático
    # Casa: Média + 15% (Pressão)
    # Fora: Média - 10% (Retranca)
    c_h_exp = s_h['corners'] * 1.15
    c_a_exp = s_a['corners'] * 0.90
    
    k_h_exp = s_h['cards'] * rf
    k_a_exp = s_a['cards'] * rf
    
    return {
        'corners': {'h': c_h_exp, 'a': c_a_exp},
        'cards': {'h': k_h_exp, 'a': k_a_exp}
    }

def fmt_hist(data):
    if not data: return "N/A"
    return f"{data[1]}/{data[0]}" # Hits/Total

def check_elite(prob, hist_data):
    # Lógica > 75%
    if hist_data:
        return float(hist_data[2]) >= 70
    return prob >= 75

def color(p): return "green" if p >= 70 else ("orange" if p >= 50 else "red")

# ==============================================================================
# 6. DASHBOARD
# ==============================================================================
def render_dashboard():
    st.title("📆 FutPrevisão V10.0 (Master Calendar)")
    
    tab_scan, tab_sim = st.tabs(["🔥 Scanner do Dia", "🔮 Simulação Manual"])
    
    # --- ABA SCANNER ---
    with tab_scan:
        df = load_calendar_file()
        
        if df.empty:
            st.error("❌ Arquivo 'calendario_ligas.csv' não encontrado ou inválido.")
        else:
            dates = df['Data'].unique()
            # Tenta selecionar hoje ou o primeiro dia disponível
            hoje = datetime.now().strftime("%d/%m/%Y")
            idx = list(dates).index(hoje) if hoje in dates else 0
            
            sel_date = st.selectbox("📅 Selecione a Data:", dates, index=idx)
            
            # Filtra jogos do dia
            jogos = df[df['Data'] == sel_date]
            st.info(f"{len(jogos)} jogos encontrados na sua lista.")
            
            if st.button("🚀 Escanear Jogos"):
                found_any = False
                
                # Barra de progresso
                bar = st.progress(0)
                
                for i, (_, row) in enumerate(jogos.iterrows()):
                    # Lê dados da linha
                    t_casa = row['Time_Casa']
                    t_visitante = row['Time_Visitante']
                    liga_nome = row['Liga']
                    
                    # Mapeia liga para chave interna
                    liga_key = LIGA_MAP.get(liga_nome, "Premier League") # Default se falhar
                    
                    # Calcula
                    m = calcular_jogo(t_casa, t_visitante, None) # Scanner do CSV não tem arbitro, usa neutro
                    
                    if m:
                        ops = []
                        # --- ESCANTEIOS ---
                        # Casa
                        ph35 = prob_over(m['corners']['h'], 3.5)
                        hh35 = hist.get(t_casa, liga_key, 'corners', 'homeTeamOver35')
                        if check_elite(ph35, hh35):
                            ops.append(f"🚩 **{t_casa}** Over 3.5 Cantos | 📊 {ph35:.0f}% | 📜 {fmt_hist(hh35)}")
                        
                        # Fora
                        pa35 = prob_over(m['corners']['a'], 3.5)
                        ha35 = hist.get(t_visitante, liga_key, 'corners', 'awayTeamOver35')
                        if check_elite(pa35, ha35):
                            ops.append(f"🚩 **{t_visitante}** Over 3.5 Cantos | 📊 {pa35:.0f}% | 📜 {fmt_hist(ha35)}")
                            
                        # --- CARTÕES ---
                        # Casa
                        kh15 = prob_over(m['cards']['h'], 1.5)
                        hk15 = hist.get(t_casa, liga_key, 'cards', 'homeCardsOver15')
                        if check_elite(kh15, hk15):
                            ops.append(f"🟨 **{t_casa}** Over 1.5 Cartões | 📊 {kh15:.0f}% | 📜 {fmt_hist(hk15)}")
                            
                        # Fora
                        ka15 = prob_over(m['cards']['a'], 1.5)
                        hka15 = hist.get(t_visitante, liga_key, 'cards', 'awayCardsOver15')
                        if check_elite(ka15, hka15):
                            ops.append(f"🟨 **{t_visitante}** Over 1.5 Cartões | 📊 {ka15:.0f}% | 📜 {fmt_hist(hka15)}")

                        # Exibe Card
                        if ops:
                            found_any = True
                            with st.expander(f"{liga_nome} | {t_casa} x {t_visitante}"):
                                for op in ops: st.markdown(op)
                    
                    bar.progress((i+1)/len(jogos))
                
                if not found_any:
                    st.warning("Nenhuma oportunidade 'Elite' (>70%) encontrada hoje.")

    # --- ABA SIMULAÇÃO ---
    with tab_sim:
        st.subheader("Simulador Manual")
        c1, c2, c3 = st.columns(3)
        
        tl = team_list_all if team_list_all else ["Carregando..."]
        home = c1.selectbox("Mandante", tl, index=0)
        away = c2.selectbox("Visitante", tl, index=1 if len(tl)>1 else 0)
        
        # Arbitros do arquivo unificado
        ref_keys = sorted(list(referees_db.keys()))
        ref = c3.selectbox("Árbitro", ["Neutro"] + ref_keys)
        
        if st.button("Analisar Confronto"):
            rn = ref if ref != "Neutro" else None
            m = calcular_jogo(home, away, rn)
            
            # Tenta descobrir a liga do time para buscar histórico correto
            l_est = stats_db.get(home, {}).get('league', 'Premier League')
            
            if m:
                st.write("---")
                k1, k2 = st.columns(2)
                
                # ESCANTEIOS
                with k1:
                    st.info("🚩 Escanteios")
                    # Casa
                    p35 = prob_over(m['corners']['h'], 3.5)
                    h35 = hist.get(home, l_est, 'corners', 'homeTeamOver35')
                    st.write(f"🏠 {home} +3.5: :{color(p35)}[{p35:.0f}%] ({fmt_hist(h35)})")
                    
                    p45 = prob_over(m['corners']['h'], 4.5)
                    h45 = hist.get(home, l_est, 'corners', 'homeTeamOver45')
                    st.write(f"🏠 {home} +4.5: :{color(p45)}[{p45:.0f}%] ({fmt_hist(h45)})")
                    
                    st.markdown("---")
                    # Fora
                    pa35 = prob_over(m['corners']['a'], 3.5)
                    ha35 = hist.get(away, l_est, 'corners', 'awayTeamOver35')
                    st.write(f"✈️ {away} +3.5: :{color(pa35)}[{pa35:.0f}%] ({fmt_hist(ha35)})")
                    
                    pa45 = prob_over(m['corners']['a'], 4.5)
                    ha45 = hist.get(away, l_est, 'corners', 'awayTeamOver45')
                    st.write(f"✈️ {away} +4.5: :{color(pa45)}[{pa45:.0f}%] ({fmt_hist(ha45)})")

                # CARTÕES
                with k2:
                    st.warning("🟨 Cartões")
                    # Casa
                    pk15 = prob_over(m['cards']['h'], 1.5)
                    hk15 = hist.get(home, l_est, 'cards', 'homeCardsOver15')
                    st.write(f"🏠 {home} +1.5: :{color(pk15)}[{pk15:.0f}%] ({fmt_hist(hk15)})")
                    
                    pk25 = prob_over(m['cards']['h'], 2.5)
                    hk25 = hist.get(home, l_est, 'cards', 'homeCardsOver25')
                    st.write(f"🏠 {home} +2.5: :{color(pk25)}[{pk25:.0f}%] ({fmt_hist(hk25)})")
                    
                    st.markdown("---")
                    # Fora
                    pka15 = prob_over(m['cards']['a'], 1.5)
                    hka15 = hist.get(away, l_est, 'cards', 'awayCardsOver15')
                    st.write(f"✈️ {away} +1.5: :{color(pka15)}[{pka15:.0f}%] ({fmt_hist(hka15)})")
                    
                    pka25 = prob_over(m['cards']['a'], 2.5)
                    hka25 = hist.get(away, l_est, 'cards', 'awayCardsOver25')
                    st.write(f"✈️ {away} +2.5: :{color(pka25)}[{pka25:.0f}%] ({fmt_hist(hka25)})")

if __name__ == "__main__":
    render_dashboard()
