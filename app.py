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

# Tenta importar OCR. Se falhar, o app continua funcionando sem essa função.
try:
    import pytesseract
    # CONFIGURAÇÃO DO TESSERACT (IMPORTANTE: Se der erro, verifique este caminho no seu PC)
    # Se você instalou no caminho padrão, geralmente é este:
    path_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(path_tesseract):
        pytesseract.pytesseract.tesseract_cmd = path_tesseract
    HAS_OCR = True
except ImportError:
    HAS_OCR = False

# ==============================================================================
# 0. CONFIGURAÇÃO INICIAL
# ==============================================================================
st.set_page_config(page_title="FutPrevisão Pro V5.1 (OCR)", layout="wide", page_icon="⚽")

# ==============================================================================
# 1. FUNÇÕES DE UTILIDADE E MATEMÁTICA
# ==============================================================================
def safe_float(value):
    try:
        if isinstance(value, (int, float)): return float(value)
        return float(str(value).replace('R$', '').replace(',', '.').strip())
    except: return 0.0

def poisson_pmf(k, mu):
    try: return (math.exp(-mu) * (mu ** k)) / math.factorial(int(k))
    except: return 0.0

def poisson_sf(k, mu):
    try:
        cdf = 0
        for i in range(int(k) + 1): cdf += poisson_pmf(i, mu)
        return 1 - cdf
    except: return 0.0

def prob_over(exp, line):
    return poisson_sf(int(line), exp) * 100

def get_color(prob):
    if prob >= 70: return "green"
    if prob >= 50: return "orange"
    return "red"

# ==============================================================================
# 2. FUNÇÃO DE LEITURA DE BILHETE (SUPERBET / GERAL)
# ==============================================================================
def ler_bilhete(imagem):
    """
    Tenta extrair Stake, Odd e Data de uma imagem de aposta.
    """
    try:
        text = pytesseract.image_to_string(imagem)
        # Limpeza básica
        lines = text.split('\n')
        
        data_detectada = datetime.now()
        stake_detectada = 0.0
        odd_detectada = 0.0
        
        # Lógica de Regex para encontrar padrões
        for line in lines:
            line = line.lower()
            
            # Tenta achar Odd (padrão comum: @ 1.50 ou Odd: 2.00)
            if "@" in line or "odd" in line or "cota" in line:
                numeros = re.findall(r"\d+\.\d+", line)
                if numeros:
                    # Pega o último número da linha que pareça uma odd (> 1.0)
                    for n in numeros:
                        if 1.01 < float(n) < 100.0: odd_detectada = float(n)

            # Tenta achar valor aposta (R$ 10,00 ou 10.00)
            if "aposta" in line or "valor" in line or "stake" in line or "total" in line:
                # Remove R$ e troca virgula por ponto
                clean_line = line.replace("r$", "").replace(",", ".")
                numeros = re.findall(r"\d+\.\d+", clean_line)
                if numeros:
                    v = float(numeros[0])
                    if v > 0: stake_detectada = v
                    
        return data_detectada, stake_detectada, odd_detectada, text
    except Exception as e:
        return datetime.now(), 0.0, 0.0, f"Erro OCR: {str(e)}"

# ==============================================================================
# 3. DADOS & CORINGA
# ==============================================================================
GENERIC_STATS = {"corners": 5.0, "cards": 2.2, "fouls": 11.5, "goals_f": 1.2, "goals_a": 1.2}
INTERNAL_DB = {
    "Arsenal": {"corners": 7.5, "cards": 1.5, "fouls": 9.5},
    "Man City": {"corners": 8.2, "cards": 1.4, "fouls": 8.5},
    "Liverpool": {"corners": 7.2, "cards": 1.8, "fouls": 10.5},
    "Real Madrid": {"corners": 6.8, "cards": 1.9, "fouls": 10.0},
    "Barcelona": {"corners": 6.5, "cards": 2.0, "fouls": 10.2},
}

@st.cache_data(ttl=3600)
def load_data_master():
    final_db = INTERNAL_DB.copy()
    if os.path.exists("dados_times.csv"):
        try:
            try: df = pd.read_csv("dados_times.csv", encoding='utf-8')
            except: df = pd.read_csv("dados_times.csv", encoding='latin1', sep=';')
            df.columns = [c.strip() for c in df.columns]
            for _, row in df.iterrows():
                if 'Time' in row and pd.notna(row['Time']):
                    t_name = str(row['Time']).strip()
                    final_db[t_name] = {
                        'corners': safe_float(row.get('Escanteios', 5.0)),
                        'cards': safe_float(row.get('CartoesAmarelos', 2.0)), 
                        'fouls': safe_float(row.get('Faltas', 12.0))
                    }
        except: pass
    return final_db

teams_data = load_data_master()
team_list = sorted(list(teams_data.keys()))

@st.cache_data(ttl=3600)
def load_referees():
    try:
        df = pd.read_csv("arbitros.csv")
        return dict(zip(df['Nome'], df['Fator']))
    except: return {}

referees_data = load_referees()

# ==============================================================================
# 4. HISTÓRICO & MAPEAMENTO (Mantido da V5.0)
# ==============================================================================
NAME_MAPPING = {
    "Man City": "Man City", "Manchester City": "Man City", "Man Utd": "Man United", 
    "Atl. Madrid": "Atl Madrid", "Athletic Club": "Athletic Club", "Bayern": "Bayern Munich", 
    "PSG": "Paris SG", "Inter": "Inter"
}
FILES_CONFIG = {
    "Premier League": {"corners": "Escanteios Preimier League - codigo fonte.txt", "cards": "Cartoes Premier League - Inglaterra.txt"},
    "La Liga": {"corners": "Escanteios Espanha.txt", "cards": "Cartoes La Liga - Espanha.txt"},
    "Serie A": {"corners": "Escanteios Italia.txt", "cards": "Cartoes Serie A - Italia.txt"},
    "Bundesliga": {"corners": "Escanteios Alemanha.txt", "cards": "Cartoes Bundesliga - Alemanha.txt"},
    "Ligue 1": {"corners": "Escanteios França.txt", "cards": "Cartoes Ligue 1 - França.txt"}
}

class AdamChoiLoader:
    def __init__(self):
        self.data_corners = {}
        self.data_cards = {}
        self.load_all_files()
    def load_all_files(self):
        pasta = Path(__file__).parent
        for liga, files in FILES_CONFIG.items():
            try:
                p = pasta / files["corners"]
                if p.exists():
                    with open(p, 'r', encoding='utf-8') as f:
                        raw = f.read().strip(); 
                        if '{' in raw: raw = raw[raw.find('{'):]
                        self.data_corners[liga] = json.loads(raw)
            except: pass
            try:
                p = pasta / files["cards"]
                if p.exists():
                    with open(p, 'r', encoding='utf-8') as f:
                        raw = f.read().strip(); 
                        if '{' in raw: raw = raw[raw.find('{'):]
                        self.data_cards[liga] = json.loads(raw)
            except: pass
    def find_best_match(self, target_name, available_names):
        if not target_name: return None
        target = target_name.strip()
        if target in NAME_MAPPING: target = NAME_MAPPING[target]
        target_lower = target.lower()
        for name in available_names:
            if name.lower() == target_lower: return name
        matches = difflib.get_close_matches(target, available_names, n=1, cutoff=0.6)
        return matches[0] if matches else None
    def get_history_global(self, team, market_type, key):
        try:
            source = self.data_corners if market_type == 'corners' else self.data_cards
            for league in source:
                avail = [t['teamName'] for t in source[league].get('teams', [])]
                matched = self.find_best_match(team, avail)
                if matched:
                    for t in source[league]['teams']:
                        if t['teamName'] == matched:
                            stats = t.get(key)
                            if stats and isinstance(stats, list) and len(stats) >= 3:
                                return stats[0], stats[1], stats[2], league
            return None
        except: return None

history_loader = AdamChoiLoader()

# ==============================================================================
# 5. PREVISÃO & MÚLTIPLAS
# ==============================================================================
def normalize_team_name_for_math(name):
    if name in teams_data: return name
    if name in NAME_MAPPING:
        mapped = NAME_MAPPING[name]
        if mapped in teams_data: return mapped
    matches = difflib.get_close_matches(name, teams_data.keys(), n=1, cutoff=0.6)
    return matches[0] if matches else None

def calcular_previsao(home, away, f_h=1.0, f_a=1.0, ref_factor=1.0):
    h_key = normalize_team_name_for_math(home)
    a_key = normalize_team_name_for_math(away)
    h_data = teams_data.get(h_key, GENERIC_STATS)
    a_data = teams_data.get(a_key, GENERIC_STATS)
    
    corn_h = (h_data['corners'] * 1.10) * f_h
    corn_a = (a_data['corners'] * 0.85) * f_a
    total_corners = corn_h + corn_a
    
    avg_fouls = (h_data['fouls'] + a_data['fouls']) / 2
    tension = avg_fouls / 12.0
    if f_h > 1.05 or f_a > 1.05: tension *= 1.15
        
    card_h = h_data['cards'] * tension * ref_factor
    card_a = a_data['cards'] * tension * ref_factor
    
    return {"corners": {"t": total_corners, "h": corn_h, "a": corn_a}, "cards": {"t": card_h+card_a, "h": card_h, "a": card_a}}

def gerar_multiplas(oportunidades):
    if not oportunidades: return []
    random.shuffle(oportunidades)
    cantos = [o for o in oportunidades if "Canto" in o['Aposta']]
    cartoes = [o for o in oportunidades if "Cartão" in o['Aposta']]
    bilhetes = []
    for _ in range(6):
        selecao = []
        if cantos and cartoes:
            c = random.choice(cantos)
            k = random.choice(cartoes)
            if c['Jogo'] != k['Jogo']: selecao = [c, k]
        if not selecao and len(oportunidades) >= 2:
            s = random.sample(oportunidades, 2)
            if s[0]['Jogo'] != s[1]['Jogo']: selecao = s
        if selecao: bilhetes.append(selecao)
    return bilhetes

# ==============================================================================
# 6. GESTÃO DE BILHETES & LOGIN
# ==============================================================================
USERS = {"diego": "@Casa612"}
def check_login():
    if "logged_in" not in st.session_state: st.session_state["logged_in"] = False
    if st.session_state["logged_in"]: return True
    st.markdown("### 🔒 Acesso V5.1")
    c1, c2 = st.columns([1,2])
    with c1:
        u = st.text_input("User")
        p = st.text_input("Pass", type="password")
        if st.button("Log"):
            if u in USERS and p == USERS[u]:
                st.session_state["logged_in"] = True
                st.rerun()
    return False

if not check_login(): st.stop()

DATA_FILE = "tickets.json"
CONFIG_FILE = "cfg.json"

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

# ==============================================================================
# 7. RENDERIZAÇÃO
# ==============================================================================
def render_dashboard():
    st.title("📊 FutPrevisão V5.1 (OCR)")
    
    st.markdown("""
    <style>
        .scan-card { background: #f0f8ff; border: 1px solid #bce8f1; padding: 10px; border-radius: 5px; margin-bottom: 8px; }
        .scan-high { border-left: 5px solid #28a745; }
        .scan-med { border-left: 5px solid #ffc107; }
        .multi-card { background: #fff3cd; border: 1px solid #ffeeba; padding: 10px; border-radius: 8px; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)
    
    tab_scan, tab_sim, tab_tik = st.tabs(["🔍 Scanner", "🔮 Simulação", "🎫 Bilhetes (Auto)"])

    # --- SCANNER ---
    with tab_scan:
        st.subheader("Scanner de Oportunidades")
        all_games = []
        for f in ["premier_league.csv", "la_liga.csv", "serie_a.csv", "bundesliga.csv", "ligue_1.csv"]:
            if os.path.exists(f):
                try: 
                    df = pd.read_csv(f, dtype=str)
                    df.columns = [c.strip() for c in df.columns]
                    all_games.append(df)
                except: pass
        if not all_games: st.warning("⚠️ Calendário não encontrado.")
        else:
            full = pd.concat(all_games, ignore_index=True)
            full['Data'] = full['Data'].str.strip()
            dias = sorted(full['Data'].unique())
            hoje = datetime.now().strftime("%d/%m/%Y")
            idx = dias.index(hoje) if hoje in dias else 0
            dia = st.selectbox("Data", dias, index=idx)
            jogos = full[full['Data'] == dia]
            st.info(f"{len(jogos)} jogos encontrados.")
            if st.button("ESCANEAR"):
                res = []
                bar = st.progress(0)
                for i, (_, row) in enumerate(jogos.iterrows()):
                    try:
                        h, a = row['Mandante'], row['Visitante']
                        l = row.get('Liga', 'Premier League')
                        calc = calcular_previsao(h, a)
                        # Filtros Simplificados
                        pm = prob_over(calc['corners']['h'], 3.5)
                        hh = history_loader.get_history_global(h, 'corners', 'homeTeamOver35')
                        if hh and pm > 60 and float(hh[2]) > 70:
                            res.append({"Jogo": f"{h} x {a}", "Aposta": f"🏠 {h} +3.5 Cantos", "Conf": "Alta", "R": f"{hh[2]}%"})
                        pm = prob_over(calc['corners']['a'], 3.5)
                        ha = history_loader.get_history_global(a, 'corners', 'awayTeamOver35')
                        if ha and pm > 60 and float(ha[2]) > 70:
                            res.append({"Jogo": f"{h} x {a}", "Aposta": f"✈️ {a} +3.5 Cantos", "Conf": "Alta", "R": f"{ha[2]}%"})
                        pm = prob_over(calc['cards']['h'], 1.5)
                        hh = history_loader.get_history_global(h, 'cards', 'homeCardsOver15')
                        if hh and pm > 55 and float(hh[2]) > 65:
                            res.append({"Jogo": f"{h} x {a}", "Aposta": f"🏠 {h} +1.5 Cartões", "Conf": "Média", "R": f"{hh[2]}%"})
                        pm = prob_over(calc['cards']['a'], 1.5)
                        ha = history_loader.get_history_global(a, 'cards', 'awayCardsOver15')
                        if ha and pm > 55 and float(ha[2]) > 65:
                            res.append({"Jogo": f"{h} x {a}", "Aposta": f"✈️ {a} +1.5 Cartões", "Conf": "Média", "R": f"{ha[2]}%"})
                    except: pass
                    bar.progress((i+1)/len(jogos))
                st.session_state['scan_res'] = res
            
            if 'scan_res' in st.session_state and st.session_state['scan_res']:
                if st.button("🔄 Gerar Múltiplas"):
                    st.session_state['mult'] = gerar_multiplas(st.session_state['scan_res'])
                if 'mult' in st.session_state:
                    c1, c2, c3 = st.columns(3)
                    for i, m in enumerate(st.session_state['mult']):
                        with [c1,c2,c3][i%3]:
                            st.markdown(f"<div class='multi-card'><b>Bilhete {i+1}</b><br>1. {m[0]['Aposta']}<br>2. {m[1]['Aposta']}</div>", unsafe_allow_html=True)
                st.write("---")
                for r in st.session_state['scan_res']:
                    cls = "scan-high" if r['Conf'] == "Alta" else "scan-med"
                    st.markdown(f"<div class='scan-card {cls}'><b>{r['Jogo']}</b><br>{r['Aposta']} | Hist: {r['R']}</div>", unsafe_allow_html=True)

    # --- SIMULAÇÃO ---
    with tab_sim:
        st.subheader("Simulação Manual Completa")
        tl = team_list if team_list else ["Time A", "Time B"]
        c1, c2 = st.columns(2)
        home = c1.selectbox("Mandante", tl, index=0)
        away = c2.selectbox("Visitante", tl, index=1 if len(tl)>1 else 0)
        c3, c4 = st.columns(2)
        liga = c3.selectbox("Liga", list(FILES_CONFIG.keys()))
        arb = c4.selectbox("Árbitro", sorted(list(referees_data.keys())) or ["Genérico"])
        
        if st.button("Analisar Jogo"):
            m = calcular_previsao(home, away, ref_factor=referees_data.get(arb, 1.0))
            st.divider()
            k1, k2 = st.columns(2)
            with k1:
                st.info(f"🚩 Escanteios (Total: {m['corners']['t']:.2f})")
                st.markdown(f"**🏠 {home}**")
                for line in ['3.5', '4.5']:
                    prob = prob_over(m['corners']['h'], float(line))
                    hist = history_loader.get_history_global(home, 'corners', f'homeTeamOver{line.replace(".","")}')
                    htxt = f"({hist[2]}% - {hist[1]}/{hist[0]})" if hist else "(Sem Hist)"
                    st.write(f"+{line}: :{get_color(prob)}[{prob:.0f}%] {htxt}")
                st.markdown("---")
                st.markdown(f"**✈️ {away}**")
                for line in ['3.5', '4.5']:
                    prob = prob_over(m['corners']['a'], float(line))
                    hist = history_loader.get_history_global(away, 'corners', f'awayTeamOver{line.replace(".","")}')
                    htxt = f"({hist[2]}% - {hist[1]}/{hist[0]})" if hist else "(Sem Hist)"
                    st.write(f"+{line}: :{get_color(prob)}[{prob:.0f}%] {htxt}")
            with k2:
                st.warning(f"🟨 Cartões (Total: {m['cards']['t']:.2f})")
                st.markdown(f"**🏠 {home}**")
                for line in ['1.5', '2.5']:
                    prob = prob_over(m['cards']['h'], float(line))
                    hist = history_loader.get_history_global(home, 'cards', f'homeCardsOver{line.replace(".","")}')
                    htxt = f"({hist[2]}% - {hist[1]}/{hist[0]})" if hist else "(Sem Hist)"
                    st.write(f"+{line}: :{get_color(prob)}[{prob:.0f}%] {htxt}")
                st.markdown("---")
                st.markdown(f"**✈️ {away}**")
                for line in ['1.5', '2.5']:
                    prob = prob_over(m['cards']['a'], float(line))
                    hist = history_loader.get_history_global(away, 'cards', f'awayCardsOver{line.replace(".","")}')
                    htxt = f"({hist[2]}% - {hist[1]}/{hist[0]})" if hist else "(Sem Hist)"
                    st.write(f"+{line}: :{get_color(prob)}[{prob:.0f}%] {htxt}")

    # --- BILHETES (COM UPLOAD DE PRINT) ---
    with tab_tik:
        st.subheader("Meus Bilhetes")
        
        # --- BLOCO DE OCR ---
        st.markdown("#### 📷 Importar do Print (Superbet)")
        uploaded_file = st.file_uploader("Faça upload do Print do Bilhete", type=["png", "jpg", "jpeg"])
        
        ocr_data = None
        ocr_stake = 10.0
        ocr_odd = 2.0
        ocr_text_debug = ""
        
        if uploaded_file and HAS_OCR:
            img = Image.open(uploaded_file)
            d, s, o, txt = ler_bilhete(img)
            ocr_stake = s if s > 0 else 10.0
            ocr_odd = o if o > 1.0 else 2.0
            ocr_data = d
            ocr_text_debug = txt
            st.success("Bilhete lido! Confira os valores abaixo.")
            with st.expander("Ver texto lido (Debug)"):
                st.text(ocr_text_debug)
        elif uploaded_file and not HAS_OCR:
            st.warning("⚠️ Tesseract OCR não encontrado. Instale para usar leitura automática.")

        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        dt = c1.date_input("Data", value=ocr_data if ocr_data else datetime.now())
        sk = c2.number_input("Stake", value=ocr_stake)
        od = c3.number_input("Odd", value=ocr_odd)
        res = st.selectbox("Resultado", ["Green ✅", "Red ❌"])
        
        if st.button("Salvar Bilhete"):
            p = (sk*od - sk) if "Green" in res else -sk
            save_ticket({"id": str(uuid.uuid4())[:5], "data": str(dt), "lucro": p, "res": res})
            st.success("Salvo!")
            st.rerun()
            
        ts = get_tickets()
        if ts:
            total = sum([t['lucro'] for t in ts])
            st.metric("Lucro Total", f"R$ {total:.2f}")
            df = pd.DataFrame(ts)
            st.dataframe(df)
            if st.button("Limpar Histórico"):
                if os.path.exists(DATA_FILE): os.remove(DATA_FILE)
                st.rerun()

if __name__ == "__main__":
    render_dashboard()