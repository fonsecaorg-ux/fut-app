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
# 0. CONFIGURAÇÃO & OCR
# ==============================================================================
st.set_page_config(page_title="FutPrevisão Pro V7.0 (Global Master)", layout="wide", page_icon="⚽")

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
# 1. MAPEAMENTO DE ARQUIVOS (O CORAÇÃO DO SISTEMA)
# ==============================================================================
# Mapeia o Nome da Liga -> (Arquivo CSV Calendário, Arquivo TXT Stats)
LEAGUE_FILES = {
    # --- AS 5 GRANDES ---
    "Premier League": {"csv": "Premier League 25.26.csv", "txt": "Prewmier League.txt", "txt_cards": "Cartoes Premier League - Inglaterra.txt"},
    "La Liga": {"csv": "La Liga 25.26.csv", "txt": "Escanteios Espanha.txt", "txt_cards": "Cartoes La Liga - Espanha.txt"},
    "Serie A": {"csv": "Serie A 25.26.csv", "txt": "Escanteios Italia.txt", "txt_cards": "Cartoes Serie A - Italia.txt"},
    "Bundesliga": {"csv": "Bundesliga 25.26.csv", "txt": "Escanteios Alemanha.txt", "txt_cards": "Cartoes Bundesliga - Alemanha.txt"},
    "Ligue 1": {"csv": "Ligue 1 25.26.csv", "txt": "Escanteios França.txt", "txt_cards": "Cartoes Ligue 1 - França.txt"},
    
    # --- NOVAS LIGAS ---
    "Championship": {"csv": "Championship Inglaterra 25.26.csv", "txt": "Championship Escanteios Inglaterra.txt", "txt_cards": None},
    "Bundesliga 2": {"csv": "Bundesliga 2.csv", "txt": "Bundesliga 2.txt", "txt_cards": None},
    "Pro League (BEL)": {"csv": "Pro League Belgica 25.26.csv", "txt": "Pro League Belgica.txt", "txt_cards": None},
    "Süper Lig (TUR)": {"csv": "Super Lig Turquia 25.26.csv", "txt": "Super Lig Turquia.txt", "txt_cards": None},
    "Premiership (SCO)": {"csv": "Premiership Escocia 25.26.csv", "txt": "Premiership Escocia.txt", "txt_cards": None}
}

# ==============================================================================
# 2. MOTOR DE APRENDIZAGEM (LÊ CSVs -> CALCULA MÉDIAS)
# ==============================================================================
GENERIC_STATS = {"corners": 5.0, "cards": 2.0, "fouls": 11.0}

@st.cache_data(ttl=3600)
def learn_stats_from_csvs():
    """
    Lê todos os CSVs configurados, identifica colunas de estatísticas (HC, AC, HY, AY)
    e calcula a média histórica de cada time para usar na previsão.
    """
    learned_db = {}
    
    for liga, files in LEAGUE_FILES.items():
        csv_file = files["csv"]
        if os.path.exists(csv_file):
            try:
                # Tenta ler (football-data costuma ser latin1)
                try: df = pd.read_csv(csv_file, encoding='latin1')
                except: df = pd.read_csv(csv_file)
                
                # Normaliza colunas
                cols = df.columns.tolist()
                
                # Identifica colunas chaves
                # Home/Away Team
                col_h = 'HomeTeam' if 'HomeTeam' in cols else ('Mandante' if 'Mandante' in cols else None)
                col_a = 'AwayTeam' if 'AwayTeam' in cols else ('Visitante' if 'Visitante' in cols else None)
                
                # Corners (HC = Home Corners, AC = Away Corners)
                has_corners = 'HC' in cols and 'AC' in cols
                
                # Cards (HY = Home Yellow, AY = Away Yellow)
                has_cards = 'HY' in cols and 'AY' in cols
                
                if col_h and col_a:
                    teams = set(df[col_h].dropna().unique()).union(set(df[col_a].dropna().unique()))
                    
                    for t in teams:
                        stats = {'cnt': 0, 'sum_corn': 0, 'sum_card': 0}
                        
                        # Jogos em Casa
                        home_games = df[df[col_h] == t]
                        stats['cnt'] += len(home_games)
                        if has_corners: stats['sum_corn'] += home_games['HC'].sum()
                        if has_cards: stats['sum_card'] += home_games['HY'].sum() # Simplificando para amarelos
                        
                        # Jogos Fora
                        away_games = df[df[col_a] == t]
                        stats['cnt'] += len(away_games)
                        if has_corners: stats['sum_corn'] += away_games['AC'].sum()
                        if has_cards: stats['sum_card'] += away_games['AY'].sum()
                        
                        # Calcula Médias
                        if stats['cnt'] > 0:
                            avg_corn = (stats['sum_corn'] / stats['cnt']) if has_corners else 5.0
                            avg_card = (stats['sum_card'] / stats['cnt']) if has_cards else 2.0
                            
                            # Refinamento: Se a média for 0 (dados faltantes), usa genérico
                            if avg_corn == 0: avg_corn = 5.0
                            if avg_card == 0: avg_card = 2.0
                            
                            learned_db[t] = {
                                'corners': avg_corn,
                                'cards': avg_card,
                                'fouls': 11.5 # Faltas raramente tem nos CSVs básicos, mantemos genérico
                            }
            except: pass
            
    return learned_db

# Carrega a base de conhecimento
teams_data = learn_stats_from_csvs()
team_list = sorted(list(teams_data.keys()))

# Carrega Árbitros
@st.cache_data(ttl=3600)
def load_referees():
    if os.path.exists("arbitros_5_ligas_2025_2026.csv"):
        try:
            df = pd.read_csv("arbitros_5_ligas_2025_2026.csv")
            return dict(zip(df['Arbitro'], df['Media_Cartoes_Por_Jogo'])) # Usa média como fator base
        except: return {}
    return {}

referees_data = load_referees()

# ==============================================================================
# 3. CARREGAMENTO DE CALENDÁRIO UNIFICADO
# ==============================================================================
def load_unified_calendar():
    all_games = []
    
    for liga, files in LEAGUE_FILES.items():
        f = files["csv"]
        if os.path.exists(f):
            try:
                try: df = pd.read_csv(f, encoding='latin1', dtype=str)
                except: df = pd.read_csv(f, dtype=str)
                
                # Padroniza nomes das colunas para o Scanner
                rename_map = {}
                if 'Date' in df.columns: rename_map['Date'] = 'Data'
                if 'HomeTeam' in df.columns: rename_map['HomeTeam'] = 'Mandante'
                if 'AwayTeam' in df.columns: rename_map['AwayTeam'] = 'Visitante'
                if 'Referee' in df.columns: rename_map['Referee'] = 'Arbitro'
                
                df = df.rename(columns=rename_map)
                
                # Adiciona coluna da Liga se não tiver
                if 'Liga' not in df.columns:
                    df['Liga'] = liga
                
                # Seleciona apenas colunas essenciais e limpa dados vazios
                cols_req = ['Data', 'Mandante', 'Visitante', 'Liga']
                if set(cols_req).issubset(df.columns):
                    temp_df = df[cols_req].copy()
                    if 'Arbitro' in df.columns: temp_df['Arbitro'] = df['Arbitro']
                    else: temp_df['Arbitro'] = "Desconhecido"
                    
                    all_games.append(temp_df)
            except: pass
            
    if all_games:
        full_df = pd.concat(all_games, ignore_index=True)
        full_df['Data'] = full_df['Data'].str.strip()
        return full_df
    return pd.DataFrame()

# ==============================================================================
# 4. HISTÓRICO (ADAM CHOI LOADER OTIMIZADO)
# ==============================================================================
class AdamChoiLoader:
    def __init__(self):
        self.data_corners = {}
        self.data_cards = {}
        self.load_all()

    def load_all(self):
        for liga, files in LEAGUE_FILES.items():
            # Carrega Cantos
            if files["txt"] and os.path.exists(files["txt"]):
                try:
                    with open(files["txt"], 'r', encoding='utf-8') as f:
                        raw = f.read().strip()
                        if '{' in raw: raw = raw[raw.find('{'):]
                        self.data_corners[liga] = json.loads(raw)
                except: pass
            
            # Carrega Cartões (Se existir)
            if files.get("txt_cards") and os.path.exists(files["txt_cards"]):
                try:
                    with open(files["txt_cards"], 'r', encoding='utf-8') as f:
                        raw = f.read().strip()
                        if '{' in raw: raw = raw[raw.find('{'):]
                        self.data_cards[liga] = json.loads(raw)
                except: pass

    def find_match(self, t, avail):
        # Mapeamentos manuais para casos difíceis
        MAP_MANUAL = {
            "Man United": "Man Utd", "Manchester United": "Man Utd",
            "Nott'm Forest": "Forest", "Nottingham Forest": "Forest",
            "Sheffield United": "Sheff Utd", "Sheffield Utd": "Sheff Utd",
            "Wolverhampton": "Wolves"
        }
        t_clean = t.strip()
        if t_clean in MAP_MANUAL: t_clean = MAP_MANUAL[t_clean]
        
        matches = difflib.get_close_matches(t_clean, avail, n=1, cutoff=0.6)
        return matches[0] if matches else None

    def get_hist(self, team, liga, m_type, key):
        src = self.data_corners if m_type == 'corners' else self.data_cards
        
        # Busca na liga específica primeiro
        if liga in src:
            avail = [x['teamName'] for x in src[liga].get('teams', [])]
            found = self.find_match(team, avail)
            if found:
                for x in src[liga]['teams']:
                    if x['teamName'] == found:
                        s = x.get(key)
                        if s and len(s) >= 3: return s[0], s[1], s[2]
        
        # Se não achar, busca globalmente (Fallback)
        for l in src:
            avail = [x['teamName'] for x in src[l].get('teams', [])]
            found = self.find_match(team, avail)
            if found:
                for x in src[l]['teams']:
                    if x['teamName'] == found:
                        s = x.get(key)
                        if s and len(s) >= 3: return s[0], s[1], s[2]
        return None

history_loader = AdamChoiLoader()

# ==============================================================================
# 5. MATEMÁTICA & PREVISÃO
# ==============================================================================
def poisson_pmf(k, mu):
    try: return (math.exp(-mu) * (mu ** k)) / math.factorial(int(k))
    except: return 0.0

def poisson_sf(k, mu):
    try:
        cdf = 0
        for i in range(int(k) + 1): cdf += poisson_pmf(i, mu)
        return 1 - cdf
    except: return 0.0

def prob_over(exp, line): return poisson_sf(int(line), exp) * 100

def get_color(prob):
    if prob >= 70: return "green"
    if prob >= 50: return "orange"
    return "red"

def normalize_team_name(name):
    if name in teams_data: return name
    matches = difflib.get_close_matches(name, teams_data.keys(), n=1, cutoff=0.6)
    return matches[0] if matches else None

def calcular_previsao(home, away, referee=None):
    h_key = normalize_team_name(home)
    a_key = normalize_team_name(away)
    
    h_data = teams_data.get(h_key, GENERIC_STATS)
    a_data = teams_data.get(a_key, GENERIC_STATS)
    
    # Fator do Árbitro
    ref_factor = 1.0
    if referee and referee in referees_data:
        # Se o juiz dá mais de 4.5 cartões, aumenta a tensão
        try:
            r_avg = float(referees_data[referee])
            ref_factor = r_avg / 4.0 # Normaliza em base 4.0
        except: pass

    # Cálculos
    corn_t = (h_data['corners'] + a_data['corners']) * 1.05 # Fator casa
    card_t = (h_data['cards'] + a_data['cards']) * ref_factor
    
    return {
        'corners': {'t': corn_t, 'h': h_data['corners'], 'a': a_data['corners']},
        'cards': {'t': card_t, 'h': h_data['cards'], 'a': a_data['cards']}
    }

def gerar_multiplas(oportunidades):
    if not oportunidades: return []
    random.shuffle(oportunidades)
    bilhetes = []
    # Tenta agrupar por liga diferente se possivel, ou simplesmente pares
    for i in range(0, len(oportunidades), 2):
        if i+1 < len(oportunidades):
            bilhetes.append([oportunidades[i], oportunidades[i+1]])
        if len(bilhetes) >= 6: break
    return bilhetes

# ==============================================================================
# 6. GESTÃO DE BILHETES & OCR
# ==============================================================================
DATA_FILE = "tickets_v7.json"
def get_tickets():
    if not os.path.exists(DATA_FILE): return []
    try: return json.load(open(DATA_FILE))
    except: return []
def save_ticket(t):
    ts = get_tickets()
    ts.append(t)
    with open(DATA_FILE, 'w') as f: json.dump(ts, f)
def del_ticket(id):
    ts = [t for t in get_tickets() if t['id'] != id]
    with open(DATA_FILE, 'w') as f: json.dump(ts, f)

# Leitor OCR Simplificado para V7
def ler_bilhete(img):
    if not HAS_OCR: return datetime.now(), 10.0, 2.0, "OCR Off"
    try:
        txt = pytesseract.image_to_string(img).lower()
        stake = 10.0
        odd = 2.0
        # Busca simples
        nums = re.findall(r"\d+\.\d+", txt)
        if nums: 
            vals = [float(n) for n in nums]
            odd = max(vals) if max(vals) < 100 else 2.0
            stake = min(vals) if min(vals) > 1 else 10.0
        return datetime.now(), stake, odd, txt
    except: return datetime.now(), 10.0, 2.0, "Erro Leitura"

# ==============================================================================
# 7. DASHBOARD
# ==============================================================================
USERS = {"diego": "@Casa612"}
def check_login():
    if "logged_in" not in st.session_state: st.session_state["logged_in"] = False
    if st.session_state["logged_in"]: return True
    st.markdown("### 🔒 FutPrevisão V7.0")
    c1, c2 = st.columns([1,2])
    with c1:
        if st.button("Entrar (Dev)"):
            st.session_state["logged_in"] = True
            st.rerun()
    return False

if not check_login(): st.stop()

def render_dashboard():
    st.title("🌍 FutPrevisão Pro V7.0 (Master)")
    
    st.markdown("""
    <style>
        .scan-card { background: #e3f2fd; border-left: 5px solid #2196f3; padding: 10px; margin-bottom: 8px; border-radius: 5px; }
        .conf-high { border-left: 5px solid #4caf50; background: #e8f5e9; }
        .multi-card { background: #fff3e0; border: 1px solid #ffe0b2; padding: 10px; border-radius: 8px; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🔍 Scanner (10 Ligas)", "🔮 Simulação", "🎫 Bilhetes"])
    
    # --- SCANNER ---
    with tab1:
        df = load_unified_calendar()
        if df.empty:
            st.warning("Nenhum calendário carregado. Verifique os arquivos CSV.")
        else:
            dias = sorted(df['Data'].unique())
            hoje = datetime.now().strftime("%d/%m/%Y")
            idx = dias.index(hoje) if hoje in dias else 0
            
            c_d, c_b = st.columns([3,1])
            dia_sel = c_d.selectbox("Selecione a Data:", dias, index=idx)
            
            jogos = df[df['Data'] == dia_sel]
            st.info(f"📅 {dia_sel}: {len(jogos)} jogos encontrados em {len(jogos['Liga'].unique())} ligas.")
            
            if c_b.button("🚀 ESCANEAR"):
                res = []
                bar = st.progress(0)
                total = len(jogos)
                
                for i, (_, row) in enumerate(jogos.iterrows()):
                    try:
                        h, a, l = row['Mandante'], row['Visitante'], row['Liga']
                        ref = row['Arbitro'] if 'Arbitro' in row else None
                        
                        # Previsão
                        calc = calcular_previsao(h, a, ref)
                        
                        # Histórico
                        h_corn = history_loader.get_hist(h, l, 'corners', 'homeTeamOver35')
                        a_corn = history_loader.get_hist(a, l, 'corners', 'awayTeamOver35')
                        
                        # Lógica de Filtro (Mais de 65% chance matemática + Histórico sólido)
                        # 1. Escanteios
                        prob_h = prob_over(calc['corners']['h'], 3.5)
                        if prob_h > 65 and (not h_corn or float(h_corn[2]) > 60):
                            res.append({"J": f"{h} x {a}", "A": f"🏠 {h} +3.5 Cantos", "L": l, "P": f"{prob_h:.0f}%"})
                            
                        prob_a = prob_over(calc['corners']['a'], 3.5)
                        if prob_a > 65 and (not a_corn or float(a_corn[2]) > 60):
                            res.append({"J": f"{h} x {a}", "A": f"✈️ {a} +3.5 Cantos", "L": l, "P": f"{prob_a:.0f}%"})
                            
                        # 2. Cartões (Se tiver dados de juiz, melhor ainda)
                        prob_card = prob_over(calc['cards']['t'], 3.5)
                        if prob_card > 70:
                             res.append({"J": f"{h} x {a}", "A": f"🟨 Jogo +3.5 Cartões", "L": l, "P": f"{prob_card:.0f}%"})

                    except: pass
                    bar.progress((i+1)/total)
                
                st.session_state['scan_res'] = res
            
            # Resultados Scanner
            if 'scan_res' in st.session_state and st.session_state['scan_res']:
                st.write("---")
                # Botão Múltiplas
                if st.button("Gerar Bilhetes Prontos"):
                    st.session_state['mult'] = gerar_multiplas(st.session_state['scan_res'])
                
                if 'mult' in st.session_state:
                    cols = st.columns(3)
                    for i, m in enumerate(st.session_state['mult']):
                        with cols[i%3]:
                            st.markdown(f"<div class='multi-card'><b>Bilhete {i+1}</b><br>1. {m[0]['A']}<br>2. {m[1]['A']}</div>", unsafe_allow_html=True)
                
                st.write("### Lista de Oportunidades")
                for r in st.session_state['scan_res']:
                    st.markdown(f"""
                    <div class='scan-card conf-high'>
                        <small>{r['L']}</small><br>
                        <b>{r['J']}</b><br>
                        👉 {r['A']} | Math: {r['P']}
                    </div>
                    """, unsafe_allow_html=True)

    # --- SIMULAÇÃO ---
    with tab2:
        st.subheader("Simulador Manual")
        tl = team_list if team_list else ["Carregando..."]
        c1, c2 = st.columns(2)
        h = c1.selectbox("Mandante", tl, index=0)
        a = c2.selectbox("Visitante", tl, index=1 if len(tl)>1 else 0)
        
        # Tenta achar árbitro se tiver
        arb_list = sorted(list(referees_data.keys()))
        arb = st.selectbox("Árbitro (Opcional)", ["Neutro"] + arb_list)
        
        if st.button("Analisar Confronto"):
            ref_name = arb if arb != "Neutro" else None
            m = calcular_previsao(h, a, ref_name)
            
            st.divider()
            k1, k2 = st.columns(2)
            
            with k1:
                st.info(f"🚩 Escanteios (Exp: {m['corners']['t']:.2f})")
                
                # Casa
                p35 = prob_over(m['corners']['h'], 3.5)
                p45 = prob_over(m['corners']['h'], 4.5)
                st.write(f"**🏠 {h}**")
                st.write(f"+3.5: :{get_color(p35)}[{p35:.0f}%]")
                st.write(f"+4.5: :{get_color(p45)}[{p45:.0f}%]")
                
                st.markdown("---")
                # Fora
                p35a = prob_over(m['corners']['a'], 3.5)
                st.write(f"**✈️ {a}**")
                st.write(f"+3.5: :{get_color(p35a)}[{p35a:.0f}%]")

            with k2:
                st.warning(f"🟨 Cartões (Exp: {m['cards']['t']:.2f})")
                p35c = prob_over(m['cards']['t'], 3.5)
                p45c = prob_over(m['cards']['t'], 4.5)
                st.metric("Prob Jogo +3.5", f"{p35c:.0f}%")
                st.metric("Prob Jogo +4.5", f"{p45c:.0f}%")
                if ref_name: st.caption(f"Ajustado pelo árbitro: {ref_name}")

    # --- BILHETES ---
    with tab3:
        st.subheader("Gestão de Banca")
        
        # OCR Upload
        img_file = st.file_uploader("Ler Print (OCR)", type=['png', 'jpg'])
        if img_file and HAS_OCR:
            if st.button("Ler Dados"):
                _, sk, od, txt = ler_bilhete(Image.open(img_file))
                st.session_state['new_tk'] = {'stake': sk, 'odd': od, 'desc': txt[:50]+"..."}
                st.success("Dados lidos!")
        
        # Form
        default = st.session_state.get('new_tk', {'stake': 10.0, 'odd': 2.0, 'desc': ''})
        
        c1, c2, c3 = st.columns(3)
        dt = c1.date_input("Data")
        sk = c2.number_input("Stake", value=default['stake'])
        od = c3.number_input("Odd", value=default['odd'])
        desc = st.text_input("Descrição", value=default['desc'])
        res = st.selectbox("Status", ["Pendente", "Green", "Red"])
        
        if st.button("Salvar"):
            lucro = 0
            if res == "Green": lucro = (sk * od) - sk
            elif res == "Red": lucro = -sk
            
            save_ticket({"id": str(uuid.uuid4())[:5], "data": str(dt), "desc": desc, "lucro": lucro, "res": res})
            st.success("Registrado!")
            
        # Lista
        ts = get_tickets()
        if ts:
            tot = sum(t['lucro'] for t in ts)
            st.metric("Lucro Total", f"R$ {tot:.2f}")
            for t in reversed(ts):
                color = "green" if t['lucro'] > 0 else ("red" if t['lucro'] < 0 else "grey")
                st.markdown(f":{color}[{t['data']} | {t['desc']} | R$ {t['lucro']:.2f}]")
                if st.button("X", key=t['id']):
                    del_ticket(t['id']); st.rerun()

if __name__ == "__main__":
    render_dashboard()
