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
st.set_page_config(page_title="FutPrevisão V8.4 (Future Fix)", layout="wide", page_icon="⚽")

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
# 2. SISTEMA DE ÁRBITROS
# ==============================================================================
@st.cache_data(ttl=3600)
def load_referees_unified():
    refs = {}
    # Arquivo Simples
    if os.path.exists("arbitros.csv"):
        try:
            df1 = pd.read_csv("arbitros.csv")
            for _, r in df1.iterrows(): refs[str(r['Nome']).strip()] = float(r['Fator'])
        except: pass
    # Arquivo Detalhado
    if os.path.exists("arbitros_5_ligas_2025_2026.csv"):
        try:
            df2 = pd.read_csv("arbitros_5_ligas_2025_2026.csv")
            for _, r in df2.iterrows():
                refs[str(r['Arbitro']).strip()] = float(r['Media_Cartoes_Por_Jogo']) / 4.0
        except: pass
    return refs

referees_db = load_referees_unified()

def get_ref_factor(name):
    if not name or name == "Neutro" or name == "Desconhecido": return 1.0
    if name in referees_db: return referees_db[name]
    match = difflib.get_close_matches(name, referees_db.keys(), n=1, cutoff=0.7)
    return referees_db[match[0]] if match else 1.0

# ==============================================================================
# 3. DATAS & CALENDÁRIO (CORRIGIDO PARA FUTURO)
# ==============================================================================
def fix_date(d):
    try:
        s = str(d).strip()
        # Tenta formatos comuns
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
                    
                    # FIX: NÃO DESCARTAR JOGOS SEM RESULTADO
                    # Apenas removemos se não tiver Data ou Times
                    temp = temp.dropna(subset=['Data', 'Mandante', 'Visitante'])
                    all_games.append(temp)
            except: pass
    
    if all_games:
        final = pd.concat(all_games, ignore_index=True)
        # Ordenação inteligente (Datas como objeto para ordenar, mas exibe string)
        final['DtObj'] = pd.to_datetime(final['Data'], format="%d/%m/%Y", errors='coerce')
        # Ordena descrescente (Mais longe no futuro -> Mais antigo)
        final = final.sort_values('DtObj', ascending=False)
        return final
    return pd.DataFrame()

# ==============================================================================
# 4. LEARNING & HISTÓRICO
# ==============================================================================
@st.cache_data(ttl=3600)
def learn_stats():
    # Aqui mantemos rigoroso: só aprende com quem tem dados (jogos passados)
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
                
                # Check se tem colunas de dados
                has_corn = 'HC' in cols and 'AC' in cols
                has_card = 'HY' in cols and 'AY' in cols
                
                if h_c in cols:
                    # Pega times unicos
                    teams = set(df[h_c].dropna().unique()).union(set(df[a_c].dropna().unique()))
                    for t in teams:
                        # Filtra apenas jogos que JÁ ACONTECERAM (têm dados)
                        hg = df[(df[h_c] == t) & (df['HC'].notna() if has_corn else True)]
                        ag = df[(df[a_c] == t) & (df['AC'].notna() if has_corn else True)]
                        
                        n = len(hg) + len(ag)
                        if n < 3: continue # Precisa de min 3 jogos para ter media
                        
                        c_avg = ((hg['HC'].sum() + ag['AC'].sum()) / n) if has_corn else 5.0
                        k_avg = ((hg['HY'].sum() + ag['AY'].sum()) / n) if has_card else 2.0
                        
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
        if liga in src:
            res = self._search(src[liga], team, key)
            if res: return res
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
# 5. MATEMÁTICA
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
    
    c_h_exp = s_h['corners'] * 1.15
    c_a_exp = s_a['corners'] * 0.90
    k_h_exp = s_h['cards'] * rf
    k_a_exp = s_a['cards'] * rf
    
    return {
        'corners': {'h': c_h_exp, 'a': c_a_exp, 't': c_h_exp + c_a_exp},
        'cards': {'h': k_h_exp, 'a': k_a_exp, 't': k_h_exp + k_a_exp}
    }

def fmt_hist(data):
    if not data: return "N/A"
    return f"{data[1]}/{data[0]}"

def check_green(prob, hist_data):
    math_ok = prob >= 75
    hist_ok = False
    if hist_data:
        if float(hist_data[2]) >= 70: hist_ok = True
    else:
        if prob >= 80: hist_ok = True
    return math_ok or hist_ok

# ==============================================================================
# 6. DASHBOARD
# ==============================================================================
def render_dashboard():
    st.title("🎯 FutPrevisão V8.4 (Calendário Futuro)")
    
    tab_scan, tab_sim = st.tabs(["🔥 Scanner Calendário", "🔮 Simulação Manual"])
    
    with tab_scan:
        df = load_unified_calendar()
        if df.empty:
            st.warning("⚠️ Calendários vazios.")
        else:
            dates = df['Data'].unique()
            # Tenta pegar a data de hoje para facilitar
            hoje = datetime.now().strftime("%d/%m/%Y")
            try:
                idx = list(dates).index(hoje)
            except:
                idx = 0 # Se hoje não tem jogo, pega a primeira da lista
            
            sel_date = st.selectbox("📅 Selecione a Data:", dates, index=idx)
            jogos = df[df['Data'] == sel_date]
            
            st.info(f"{len(jogos)} jogos encontrados para {sel_date}.")
            
            if st.button("🔎 Buscar Oportunidades (>75%)"):
                found = False
                progress = st.progress(0)
                
                for i, (_, row) in enumerate(jogos.iterrows()):
                    h, a, l, r = row['Mandante'], row['Visitante'], row['Liga'], row['Arbitro']
                    m = calcular(h, a, r)
                    
                    if m:
                        ops = []
                        # LOGICA DE FILTRO (Mantida)
                        
                        # CASA CANTOS
                        ph35 = prob_over(m['corners']['h'], 3.5)
                        hh35 = hist.get(h, l, 'corners', 'homeTeamOver35')
                        if check_green(ph35, hh35):
                            ops.append(f"🚩 **{h}** Over 3.5 Cantos | Math: {ph35:.0f}% | Hist: {fmt_hist(hh35)}")

                        # FORA CANTOS
                        pa35 = prob_over(m['corners']['a'], 3.5)
                        ha35 = hist.get(a, l, 'corners', 'awayTeamOver35')
                        if check_green(pa35, ha35):
                            ops.append(f"🚩 **{a}** Over 3.5 Cantos | Math: {pa35:.0f}% | Hist: {fmt_hist(ha35)}")
                            
                        # CASA CARTÕES
                        kh15 = prob_over(m['cards']['h'], 1.5)
                        hk15 = hist.get(h, l, 'cards', 'homeCardsOver15')
                        if check_green(kh15, hk15):
                            ops.append(f"🟨 **{h}** Over 1.5 Cartões | Math: {kh15:.0f}% | Hist: {fmt_hist(hk15)}")

                        # FORA CARTÕES
                        ka15 = prob_over(m['cards']['a'], 1.5)
                        hka15 = hist.get(a, l, 'cards', 'awayCardsOver15')
                        if check_green(ka15, hka15):
                            ops.append(f"🟨 **{a}** Over 1.5 Cartões | Math: {ka15:.0f}% | Hist: {fmt_hist(hka15)}")

                        if ops:
                            found = True
                            with st.expander(f"💎 {l} | {h} x {a}"):
                                for op in ops: st.markdown(op)
                    
                    progress.progress((i+1)/len(jogos))
                
                if not found:
                    st.warning("Nenhuma oportunidade 'Elite' (>75%) encontrada nestes jogos.")

    with tab_sim:
        st.subheader("Simulador Manual")
        c1, c2, c3 = st.columns(3)
        tl = team_list if team_list else ["Carregando..."]
        home = c1.selectbox("Mandante", tl, index=0)
        away = c2.selectbox("Visitante", tl, index=1 if len(tl)>1 else 0)
        ref_keys = sorted(list(referees_db.keys()))
        ref = c3.selectbox("Árbitro", ["Neutro"] + ref_keys)
        
        if st.button("Processar"):
            rn = ref if ref != "Neutro" else None
            m = calcular(home, away, rn)
            l_est = stats_db.get(home, {}).get('league', 'Premier League')
            
            if m:
                st.write("---")
                k1, k2 = st.columns(2)
                with k1:
                    st.info("🚩 Escanteios")
                    p35 = prob_over(m['corners']['h'], 3.5)
                    h35 = hist.get(home, l_est, 'corners', 'homeTeamOver35')
                    st.write(f"**🏠 {home}** +3.5: **{p35:.0f}%** ({fmt_hist(h35)})")
                    
                    p45 = prob_over(m['corners']['h'], 4.5)
                    h45 = hist.get(home, l_est, 'corners', 'homeTeamOver45')
                    st.write(f"**🏠 {home}** +4.5: **{p45:.0f}%** ({fmt_hist(h45)})")
                    st.markdown("---")
                    pa35 = prob_over(m['corners']['a'], 3.5)
                    ha35 = hist.get(away, l_est, 'corners', 'awayTeamOver35')
                    st.write(f"**✈️ {away}** +3.5: **{pa35:.0f}%** ({fmt_hist(ha35)})")
                    
                    pa45 = prob_over(m['corners']['a'], 4.5)
                    ha45 = hist.get(away, l_est, 'corners', 'awayTeamOver45')
                    st.write(f"**✈️ {away}** +4.5: **{pa45:.0f}%** ({fmt_hist(ha45)})")
                
                with k2:
                    st.warning("🟨 Cartões")
                    pk15 = prob_over(m['cards']['h'], 1.5)
                    hk15 = hist.get(home, l_est, 'cards', 'homeCardsOver15')
                    st.write(f"**🏠 {home}** +1.5: **{pk15:.0f}%** ({fmt_hist(hk15)})")
                    
                    pk25 = prob_over(m['cards']['h'], 2.5)
                    hk25 = hist.get(home, l_est, 'cards', 'homeCardsOver25')
                    st.write(f"**🏠 {home}** +2.5: **{pk25:.0f}%** ({fmt_hist(hk25)})")
                    st.markdown("---")
                    pka15 = prob_over(m['cards']['a'], 1.5)
                    hka15 = hist.get(away, l_est, 'cards', 'awayCardsOver15')
                    st.write(f"**✈️ {away}** +1.5: **{pka15:.0f}%** ({fmt_hist(hka15)})")

                    pka25 = prob_over(m['cards']['a'], 2.5)
                    hka25 = hist.get(away, l_est, 'cards', 'awayCardsOver25')
                    st.write(f"**✈️ {away}** +2.5: **{pka25:.0f}%** ({fmt_hist(hka25)})")

if __name__ == "__main__":
    render_dashboard()
