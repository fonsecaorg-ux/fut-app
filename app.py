import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import math
import difflib
import re
from datetime import datetime

# ==============================================================================
# 0. CONFIGURAÇÃO & LEITOR PDF
# ==============================================================================
st.set_page_config(page_title="FutPrevisão V9.1 (Blindada)", layout="wide", page_icon="⚽")

HAS_PDF = False
PDF_ERROR = ""
try:
    import pdfplumber
    HAS_PDF = True
except ImportError:
    PDF_ERROR = "Biblioteca 'pdfplumber' não encontrada. Instale usando: pip install pdfplumber"
except Exception as e:
    PDF_ERROR = f"Erro ao carregar biblioteca: {e}"

# ==============================================================================
# 1. MAPEAMENTO DE ARQUIVOS
# ==============================================================================
LEAGUE_FILES = {
    "Premier League": {"csv": "Premier League 25.26.csv", "pdf": "Calendario Premier League 25.26.pdf", "txt": "Prewmier League.txt", "txt_cards": "Cartoes Premier League - Inglaterra.txt"},
    "La Liga": {"csv": "La Liga 25.26.csv", "pdf": "Calendario La Liga 25.26.pdf", "txt": "Escanteios Espanha.txt", "txt_cards": "Cartoes La Liga - Espanha.txt"},
    "Serie A": {"csv": "Serie A 25.26.csv", "pdf": "Calendario Serie A.pdf", "txt": "Escanteios Italia.txt", "txt_cards": "Cartoes Serie A - Italia.txt"},
    "Bundesliga": {"csv": "Bundesliga 25.26.csv", "pdf": "Calendario Bundesliga 25.26.pdf", "txt": "Escanteios Alemanha.txt", "txt_cards": "Cartoes Bundesliga - Alemanha.txt"},
    "Ligue 1": {"csv": "Ligue 1 25.26.csv", "pdf": "Calendario Ligue 1 25.26.pdf", "txt": "Escanteios França.txt", "txt_cards": "Cartoes Ligue 1 - França.txt"},
    "Championship": {"csv": "Championship Inglaterra 25.26.csv", "pdf": "Calendario Championship.pdf", "txt": "Championship Escanteios Inglaterra.txt", "txt_cards": None},
    "Bundesliga 2": {"csv": "Bundesliga 2.csv", "pdf": "Calendario Bundesliga2.pdf", "txt": "Bundesliga 2.txt", "txt_cards": None},
    "Pro League (BEL)": {"csv": "Pro League Belgica 25.26.csv", "pdf": "Calendario Pro League.pdf", "txt": "Pro League Belgica.txt", "txt_cards": None},
    "Süper Lig (TUR)": {"csv": "Super Lig Turquia 25.26.csv", "pdf": "Calendario Super Lig.pdf", "txt": "Super Lig Turquia.txt", "txt_cards": None},
    "Premiership (SCO)": {"csv": "Premiership Escocia 25.26.csv", "pdf": "Calendari Premiership.pdf", "txt": "Premiership Escocia.txt", "txt_cards": None}
}

# ==============================================================================
# 2. SISTEMA DE ÁRBITROS
# ==============================================================================
@st.cache_data(ttl=3600)
def load_referees_unified():
    refs = {}
    # Tenta ler CSV 1
    if os.path.exists("arbitros.csv"):
        try:
            df = pd.read_csv("arbitros.csv")
            for _, r in df.iterrows(): refs[str(r.get('Nome', '')).strip()] = float(r.get('Fator', 1.0))
        except: pass
    # Tenta ler CSV 2
    if os.path.exists("arbitros_5_ligas_2025_2026.csv"):
        try:
            df = pd.read_csv("arbitros_5_ligas_2025_2026.csv")
            for _, r in df.iterrows():
                try:
                    nome = str(r.get('Arbitro', '')).strip()
                    med = float(r.get('Media_Cartoes_Por_Jogo', 4.0))
                    refs[nome] = med / 4.0
                except: pass
        except: pass
    return refs

referees_db = load_referees_unified()

def get_ref_factor(name):
    if not name or name in ["Neutro", "Desconhecido", "None"]: return 1.0
    if name in referees_db: return referees_db[name]
    m = difflib.get_close_matches(name, referees_db.keys(), n=1, cutoff=0.7)
    return referees_db[m[0]] if m else 1.0

# ==============================================================================
# 3. LEITOR INTELIGENTE (PDF + CSV BLINDADO)
# ==============================================================================
def fix_date(d):
    try:
        s = str(d).strip()
        for fmt in ["%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d", "%d-%m-%Y"]:
            try: return datetime.strptime(s, fmt).strftime("%d/%m/%Y")
            except: continue
        return s
    except: return None

def extract_games_from_pdf_safe(pdf_path, league_name):
    """Extrai jogos do PDF com tratamento de erro agressivo"""
    games = []
    log = []
    
    if not HAS_PDF: return [], "PDF Driver ausente"
    if not os.path.exists(pdf_path): return [], "Arquivo não encontrado"
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                try:
                    text = page.extract_text()
                    if not text: continue
                    
                    # Limpeza bruta para lidar com formatação estranha dos seus PDFs
                    # Remove aspas e vírgulas excessivas que vi nos seus arquivos
                    lines = text.split('\n')
                    
                    for line in lines:
                        # Procura padrão de data DD/MM/YY
                        match = re.search(r'(\d{2}/\d{2}/\d{2,4})', line)
                        if match:
                            dt_str = match.group(1)
                            clean_dt = fix_date(dt_str)
                            
                            # Remove a data da linha para sobrar só os times
                            rest = line.replace(match.group(0), "").replace('"', '').replace(',', ' ').strip()
                            
                            # Tenta achar horário (ex: 17:00) para separar
                            time_match = re.search(r'(\d{2}:\d{2})', rest)
                            if time_match:
                                rest = rest.replace(time_match.group(0), "")
                            
                            # O que sobra devem ser os times. Remove lixo.
                            words = [w for w in rest.split() if len(w) > 2 and w.lower() not in ['adiado', 'game']]
                            
                            if len(words) >= 2:
                                # Heurística: Divide a lista de palavras na metade
                                mid = len(words) // 2
                                home = " ".join(words[:mid])
                                away = " ".join(words[mid:])
                                
                                games.append({
                                    'Data': clean_dt,
                                    'Mandante': home,
                                    'Visitante': away,
                                    'Liga': league_name,
                                    'Arbitro': "Desconhecido",
                                    'Origem': 'PDF'
                                })
                except: continue # Se uma página falhar, pula para a próxima
    except Exception as e:
        return [], str(e)
        
    return games, "Sucesso"

@st.cache_data(ttl=600)
def load_unified_calendar():
    all_games = []
    logs = []
    
    for l_name, f_data in LEAGUE_FILES.items():
        # 1. Leitura CSV (Dados Passados/Oficiais)
        if os.path.exists(f_data['csv']):
            try:
                try: df = pd.read_csv(f_data['csv'], encoding='latin1', dtype=str)
                except: df = pd.read_csv(f_data['csv'], dtype=str)
                
                cols = df.columns
                d_c = next((c for c in cols if c in ['Date', 'Data']), None)
                h_c = next((c for c in cols if c in ['HomeTeam', 'Mandante']), None)
                a_c = next((c for c in cols if c in ['AwayTeam', 'Visitante']), None)
                r_c = next((c for c in cols if c in ['Referee', 'Arbitro']), None)
                
                if d_c and h_c and a_c:
                    temp = pd.DataFrame()
                    temp['Data'] = df[d_c].apply(fix_date)
                    temp['Mandante'] = df[h_c]
                    temp['Visitante'] = df[a_c]
                    temp['Liga'] = l_name
                    temp['Arbitro'] = df[r_c] if r_c else "Desconhecido"
                    temp['Origem'] = 'CSV'
                    all_games.append(temp.dropna(subset=['Data']))
            except Exception as e:
                logs.append(f"Erro CSV {l_name}: {e}")
        
        # 2. Leitura PDF (Futuro)
        if f_data.get('pdf'):
            g, status = extract_games_from_pdf_safe(f_data['pdf'], l_name)
            if g:
                all_games.append(pd.DataFrame(g))
            else:
                logs.append(f"PDF {l_name}: {status}")

    final_df = pd.DataFrame()
    if all_games:
        final_df = pd.concat(all_games, ignore_index=True)
        # Remove duplicatas
        final_df = final_df.drop_duplicates(subset=['Data', 'Mandante'], keep='first')
        
        # Ordena datas
        final_df['DtObj'] = pd.to_datetime(final_df['Data'], format="%d/%m/%Y", errors='coerce')
        final_df = final_df.sort_values('DtObj', ascending=False)
        
    return final_df, logs

# ==============================================================================
# 4. LEARNING & HISTÓRICO
# ==============================================================================
@st.cache_data(ttl=3600)
def learn_stats():
    db = {}
    for liga, files in LEAGUE_FILES.items():
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
                        # Só aprende com jogos que TÊM estatísticas (passado)
                        hg = df[(df[h_c] == t) & (df['HC'].notna() if has_corn else True)]
                        ag = df[(df[a_c] == t) & (df['AC'].notna() if has_corn else True)]
                        n = len(hg) + len(ag)
                        if n < 3: continue
                        
                        c = ((hg['HC'].sum() + ag['AC'].sum()) / n) if has_corn else 5.0
                        k = ((hg['HY'].sum() + ag['AY'].sum()) / n) if has_card else 2.0
                        db[t] = {'corners': c, 'cards': k, 'league': liga}
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
    
    c_h = s_h['corners'] * 1.15
    c_a = s_a['corners'] * 0.90
    k_h = s_h['cards'] * rf
    k_a = s_a['cards'] * rf
    
    return {
        'corners': {'h': c_h, 'a': c_a, 't': c_h + c_a},
        'cards': {'h': k_h, 'a': k_a, 't': k_h + k_a}
    }

def fmt_hist(data):
    if not data: return "N/A"
    return f"{data[1]}/{data[0]}"

def check_green(prob, hist_data):
    math_ok = prob >= 75
    hist_ok = False
    if hist_data and float(hist_data[2]) >= 70: hist_ok = True
    if not hist_data and prob >= 80: hist_ok = True
    return math_ok or hist_ok

def color(p): return "green" if p >= 70 else ("orange" if p >= 50 else "red")

# ==============================================================================
# 6. DASHBOARD
# ==============================================================================
def render_dashboard():
    st.title("🛡️ FutPrevisão V9.1 (Safe Mode)")
    
    if PDF_ERROR:
        st.warning(f"Aviso PDF: {PDF_ERROR}")
        st.caption("O app continuará funcionando com os dados CSV.")
    
    tab_scan, tab_sim = st.tabs(["🔥 Scanner Data", "🔮 Simulação"])
    
    with tab_scan:
        df, logs = load_unified_calendar()
        
        # Área de Debug (Expansível)
        with st.expander("Status dos Arquivos (Debug)"):
            if not logs: st.success("Todos os arquivos carregados perfeitamente!")
            for l in logs: st.text(f"⚠️ {l}")
            
        if df.empty:
            st.error("Nenhum jogo carregado. Verifique os arquivos na pasta.")
        else:
            dates = df['Data'].unique()
            # Tenta selecionar hoje
            hoje = datetime.now().strftime("%d/%m/%Y")
            idx = list(dates).index(hoje) if hoje in list(dates) else 0
            
            sel_date = st.selectbox("📅 Data do Jogo:", dates, index=idx)
            jogos = df[df['Data'] == sel_date]
            
            st.info(f"{len(jogos)} jogos encontrados em {sel_date}")
            
            if st.button("🔎 Buscar Oportunidades"):
                found = False
                for _, row in jogos.iterrows():
                    h, a, l, r = row['Mandante'], row['Visitante'], row['Liga'], row['Arbitro']
                    m = calcular(h, a, r)
                    
                    if m:
                        ops = []
                        # LOGICA PADRÃO (Elite 75%)
                        # Cantos
                        ph35 = prob_over(m['corners']['h'], 3.5)
                        hh35 = hist.get(h, l, 'corners', 'homeTeamOver35')
                        if check_green(ph35, hh35): ops.append(f"🚩 **{h}** Over 3.5 Cantos | {ph35:.0f}% | {fmt_hist(hh35)}")
                        
                        pa35 = prob_over(m['corners']['a'], 3.5)
                        ha35 = hist.get(a, l, 'corners', 'awayTeamOver35')
                        if check_green(pa35, ha35): ops.append(f"🚩 **{a}** Over 3.5 Cantos | {pa35:.0f}% | {fmt_hist(ha35)}")
                        
                        # Cartões
                        kh15 = prob_over(m['cards']['h'], 1.5)
                        hk15 = hist.get(h, l, 'cards', 'homeCardsOver15')
                        if check_green(kh15, hk15): ops.append(f"🟨 **{h}** Over 1.5 Cartões | {kh15:.0f}% | {fmt_hist(hk15)}")
                        
                        ka15 = prob_over(m['cards']['a'], 1.5)
                        hka15 = hist.get(a, l, 'cards', 'awayCardsOver15')
                        if check_green(ka15, hka15): ops.append(f"🟨 **{a}** Over 1.5 Cartões | {ka15:.0f}% | {fmt_hist(hka15)}")

                        if ops:
                            found = True
                            with st.expander(f"{l} | {h} x {a} | Juiz: {r}"):
                                for op in ops: st.markdown(op)
                
                if not found: st.warning("Sem oportunidades >75% hoje.")

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
                    st.write(f"🏠 {home} +3.5: :{color(p35)}[{p35:.0f}%] ({fmt_hist(h35)})")
                    p45 = prob_over(m['corners']['h'], 4.5)
                    h45 = hist.get(home, l_est, 'corners', 'homeTeamOver45')
                    st.write(f"🏠 {home} +4.5: :{color(p45)}[{p45:.0f}%] ({fmt_hist(h45)})")
                    st.markdown("---")
                    pa35 = prob_over(m['corners']['a'], 3.5)
                    ha35 = hist.get(away, l_est, 'corners', 'awayTeamOver35')
                    st.write(f"✈️ {away} +3.5: :{color(pa35)}[{pa35:.0f}%] ({fmt_hist(ha35)})")
                    pa45 = prob_over(m['corners']['a'], 4.5)
                    ha45 = hist.get(away, l_est, 'corners', 'awayTeamOver45')
                    st.write(f"✈️ {away} +4.5: :{color(pa45)}[{pa45:.0f}%] ({fmt_hist(ha45)})")
                with k2:
                    st.warning("🟨 Cartões")
                    pk15 = prob_over(m['cards']['h'], 1.5)
                    hk15 = hist.get(home, l_est, 'cards', 'homeCardsOver15')
                    st.write(f"🏠 {home} +1.5: :{color(pk15)}[{pk15:.0f}%] ({fmt_hist(hk15)})")
                    pk25 = prob_over(m['cards']['h'], 2.5)
                    hk25 = hist.get(home, l_est, 'cards', 'homeCardsOver25')
                    st.write(f"🏠 {home} +2.5: :{color(pk25)}[{pk25:.0f}%] ({fmt_hist(hk25)})")
                    st.markdown("---")
                    pka15 = prob_over(m['cards']['a'], 1.5)
                    hka15 = hist.get(away, l_est, 'cards', 'awayCardsOver15')
                    st.write(f"✈️ {away} +1.5: :{color(pka15)}[{pka15:.0f}%] ({fmt_hist(hka15)})")
                    pka25 = prob_over(m['cards']['a'], 2.5)
                    hka25 = hist.get(away, l_est, 'cards', 'awayCardsOver25')
                    st.write(f"✈️ {away} +2.5: :{color(pka25)}[{pka25:.0f}%] ({fmt_hist(hka25)})")

if __name__ == "__main__":
    render_dashboard()
