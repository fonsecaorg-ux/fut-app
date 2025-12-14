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
# 0. CONFIGURAÇÃO INICIAL
# ==============================================================================
st.set_page_config(page_title="FutPrevisão Pro V5.3 (Safe)", layout="wide", page_icon="⚽")

# ==============================================================================
# 1. CONFIGURAÇÃO DE OCR (BLINDADA)
# ==============================================================================
HAS_OCR = False
OCR_ERROR_MSG = ""

try:
    import pytesseract
    # Caminho que você confirmou
    path_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    
    if os.path.exists(path_tesseract):
        pytesseract.pytesseract.tesseract_cmd = path_tesseract
        HAS_OCR = True
    else:
        # Tenta procurar em outros lugares comuns caso tenha instalado diferente
        paths_alternativos = [
            r"C:\Users\Kaiqu\AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
        ]
        for p in paths_alternativos:
            if os.path.exists(p):
                pytesseract.pytesseract.tesseract_cmd = p
                HAS_OCR = True
                path_tesseract = p
                break
        
        if not HAS_OCR:
            OCR_ERROR_MSG = f"Arquivo não encontrado em: {path_tesseract}"
            
except ImportError:
    OCR_ERROR_MSG = "Biblioteca 'pytesseract' não instalada. (pip install pytesseract)"
except Exception as e:
    OCR_ERROR_MSG = f"Erro genérico OCR: {str(e)}"

# ==============================================================================
# 2. FUNÇÕES MATEMÁTICAS
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

def prob_over(exp, line): return poisson_sf(int(line), exp) * 100

def get_color(prob):
    if prob >= 70: return "green"
    if prob >= 50: return "orange"
    return "red"

# ==============================================================================
# 3. DADOS & CORINGA
# ==============================================================================
GENERIC_STATS = {"corners": 5.0, "cards": 2.2, "fouls": 11.5}
INTERNAL_DB = {
    "Arsenal": {"corners": 7.5, "cards": 1.5, "fouls": 9.5},
    "Man City": {"corners": 8.2, "cards": 1.4, "fouls": 8.5},
    "Liverpool": {"corners": 7.2, "cards": 1.8, "fouls": 10.5},
    "Real Madrid": {"corners": 6.8, "cards": 1.9, "fouls": 10.0},
    "Barcelona": {"corners": 6.5, "cards": 2.0, "fouls": 10.2},
    "Flamengo": {}, "Palmeiras": {}, "São Paulo": {}, "Corinthians": {}
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

# ==============================================================================
# 4. OCR LÓGICA
# ==============================================================================
def extrair_times_e_linhas(texto, lista_times):
    texto_low = texto.lower()
    times_encontrados = []
    
    # Busca Times
    lista_ordenada = sorted(lista_times, key=len, reverse=True)
    for time in lista_ordenada:
        if len(time) > 3 and time.lower() in texto_low:
            if not any(time in t_exist for t_exist in times_encontrados):
                times_encontrados.append(time)
    
    times_encontrados = times_encontrados[:2]
    
    # Busca Linhas
    linhas_encontradas = []
    keywords = ["escanteios", "cantos", "corners", "cartões", "cards", "gols", "over", "under", "mais de"]
    
    lines = texto.split('\n')
    for line in lines:
        line_low = line.lower()
        if any(k in line_low for k in keywords):
            clean = re.sub(r'[^a-zA-Z0-9\.\s]', '', line).strip()
            if len(clean) > 5:
                linhas_encontradas.append(clean)

    return times_encontrados, linhas_encontradas

def ler_bilhete_completo(imagem, lista_times):
    if not HAS_OCR: return datetime.now(), 0.0, 0.0, "", "OCR Não configurado"
    
    try:
        text = pytesseract.image_to_string(imagem)
        data_det = datetime.now()
        stake_det = 0.0
        odd_det = 0.0
        
        lines = text.split('\n')
        for line in lines:
            line_low = line.lower()
            if "@" in line_low or "odd" in line_low or "cota" in line_low:
                nums = re.findall(r"\d+\.\d+", line)
                if nums:
                    for n in nums:
                        if 1.01 < float(n) < 100.0: odd_det = float(n)
            
            if any(x in line_low for x in ["aposta", "valor", "stake", "total"]):
                clean = line_low.replace("r$", "").replace(",", ".")
                nums = re.findall(r"\d+\.\d+", clean)
                if nums:
                    v = float(nums[0])
                    if v > 0: stake_det = v

        times, linhas = extrair_times_e_linhas(text, lista_times)
        desc = ""
        if times: desc += " x ".join(times)
        if linhas:
            if desc: desc += " | "
            desc += ", ".join(linhas)
            
        return data_det, stake_det, odd_det, desc, text
    except Exception as e:
        return datetime.now(), 0.0, 0.0, "", f"Erro: {str(e)}"

# ==============================================================================
# 5. SISTEMA PRINCIPAL (SCANNER/SIMULAÇÃO/GESTÃO)
# ==============================================================================
NAME_MAPPING = {"Man City": "Man City", "Man Utd": "Man United"}
class AdamChoiLoader:
    def __init__(self): self.data = {}
    def get_history_global(self, t, m, k): return None
history_loader = AdamChoiLoader()

# Carregamento Robusto do Histórico
FILES_CONFIG = {
    "Premier League": {"corners": "Escanteios Preimier League - codigo fonte.txt", "cards": "Cartoes Premier League - Inglaterra.txt"},
    "La Liga": {"corners": "Escanteios Espanha.txt", "cards": "Cartoes La Liga - Espanha.txt"},
    "Serie A": {"corners": "Escanteios Italia.txt", "cards": "Cartoes Serie A - Italia.txt"},
    "Bundesliga": {"corners": "Escanteios Alemanha.txt", "cards": "Cartoes Bundesliga - Alemanha.txt"},
    "Ligue 1": {"corners": "Escanteios França.txt", "cards": "Cartoes Ligue 1 - França.txt"}
}

class FullLoader(AdamChoiLoader):
    def __init__(self):
        self.data_corners = {}
        self.data_cards = {}
        self.load_files()
    def load_files(self):
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

history_loader = FullLoader()

# ==============================================================================
# 6. GESTÃO DE TICKETS
# ==============================================================================
DATA_FILE = "tickets_v53.json"
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
def normalize_team_name_for_math(name):
    if name in teams_data: return name
    if name in NAME_MAPPING:
        mapped = NAME_MAPPING[name]
        if mapped in teams_data: return mapped
    matches = difflib.get_close_matches(name, teams_data.keys(), n=1, cutoff=0.6)
    return matches[0] if matches else None

def calcular_previsao(home, away):
    h_key = normalize_team_name_for_math(home)
    a_key = normalize_team_name_for_math(away)
    h_data = teams_data.get(h_key, GENERIC_STATS)
    a_data = teams_data.get(a_key, GENERIC_STATS)
    
    corn_h = (h_data['corners'] * 1.10)
    corn_a = (a_data['corners'] * 0.85)
    total_corners = corn_h + corn_a
    
    avg_fouls = (h_data['fouls'] + a_data['fouls']) / 2
    tension = avg_fouls / 12.0
    
    card_h = h_data['cards'] * tension
    card_a = a_data['cards'] * tension
    
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

# LOGIN
USERS = {"diego": "@Casa612"}
def check_login():
    if "logged_in" not in st.session_state: st.session_state["logged_in"] = False
    if st.session_state["logged_in"]: return True
    st.markdown("### 🔒 Acesso V5.3")
    c1, c2 = st.columns([1,2])
    with c1:
        if st.button("Entrar (Dev Mode)"):
            st.session_state["logged_in"] = True
            st.rerun()
    return False

if not check_login(): st.stop()

def render_dashboard():
    st.title("📊 FutPrevisão V5.3 (Safe Mode)")
    
    st.markdown("""
    <style>
        .bet-card-green { border-left: 5px solid #28a745; background: #f0fff4; padding: 10px; margin-bottom: 5px; }
        .bet-card-red { border-left: 5px solid #dc3545; background: #fff5f5; padding: 10px; margin-bottom: 5px; }
        .scan-card { background: #f0f8ff; border: 1px solid #bce8f1; padding: 10px; border-radius: 5px; margin-bottom: 8px; }
        .scan-high { border-left: 5px solid #28a745; }
        .scan-med { border-left: 5px solid #ffc107; }
        .multi-card { background: #fff3cd; border: 1px solid #ffeeba; padding: 10px; border-radius: 8px; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

    tab_scan, tab_sim, tab_tik = st.tabs(["🔍 Scanner", "🔮 Simulação", "🎫 Bilhetes"])

    with tab_scan:
        st.subheader("Scanner")
        # Load Calendars
        all_games = []
        for f in ["premier_league.csv", "la_liga.csv", "serie_a.csv", "bundesliga.csv", "ligue_1.csv"]:
            if os.path.exists(f):
                try: 
                    df = pd.read_csv(f, dtype=str)
                    df.columns = [c.strip() for c in df.columns]
                    all_games.append(df)
                except: pass
        
        if not all_games:
            st.warning("⚠️ Calendário não encontrado.")
        else:
            full = pd.concat(all_games, ignore_index=True)
            full['Data'] = full['Data'].str.strip()
            dias = sorted(full['Data'].unique())
            
            hoje = datetime.now().strftime("%d/%m/%Y")
            idx = dias.index(hoje) if hoje in dias else 0
            dia = st.selectbox("Data", dias, index=idx)
            jogos = full[full['Data'] == dia]
            st.info(f"{len(jogos)} jogos.")
            
            if st.button("ESCANEAR"):
                res = []
                bar = st.progress(0)
                for i, (_, row) in enumerate(jogos.iterrows()):
                    try:
                        h, a = row['Mandante'], row['Visitante']
                        calc = calcular_previsao(h, a)
                        # Filtros (Resumido para segurança)
                        hh = history_loader.get_history_global(h, 'corners', 'homeTeamOver35')
                        if hh and prob_over(calc['corners']['h'], 3.5) > 60 and float(hh[2]) > 70:
                            res.append({"Jogo": f"{h} x {a}", "Aposta": f"🏠 {h} +3.5 Cantos", "Conf": "Alta", "R": f"{hh[2]}%"})
                        ha = history_loader.get_history_global(a, 'corners', 'awayTeamOver35')
                        if ha and prob_over(calc['corners']['a'], 3.5) > 60 and float(ha[2]) > 70:
                            res.append({"Jogo": f"{h} x {a}", "Aposta": f"✈️ {a} +3.5 Cantos", "Conf": "Alta", "R": f"{ha[2]}%"})
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

    with tab_sim:
        st.subheader("Simulação Manual")
        tl = team_list if team_list else ["Time A", "Time B"]
        c1, c2 = st.columns(2)
        home = c1.selectbox("Mandante", tl, index=0)
        away = c2.selectbox("Visitante", tl, index=1 if len(tl)>1 else 0)
        
        if st.button("Analisar"):
            m = calcular_previsao(home, away)
            st.divider()
            k1, k2 = st.columns(2)
            with k1:
                st.info(f"🚩 Escanteios (Total: {m['corners']['t']:.2f})")
                p = prob_over(m['corners']['h'], 3.5)
                h = history_loader.get_history_global(home, 'corners', 'homeTeamOver35')
                htxt = h[2] if h else "N/A"
                st.write(f"🏠 {home} +3.5: :{get_color(p)}[{p:.0f}%] (Hist: {htxt}%)")
            with k2:
                st.warning(f"🟨 Cartões (Total: {m['cards']['t']:.2f})")
                p = prob_over(m['cards']['a'], 1.5)
                h = history_loader.get_history_global(away, 'cards', 'awayCardsOver15')
                htxt = h[2] if h else "N/A"
                st.write(f"✈️ {away} +1.5: :{get_color(p)}[{p:.0f}%] (Hist: {htxt}%)")

    with tab_tik:
        st.subheader("Importar Bilhete")
        if HAS_OCR:
            st.success(f"✅ OCR Ativado: {path_tesseract}")
            uploaded_file = st.file_uploader("Subir Print", type=["png", "jpg", "jpeg"])
            
            if 'ocr_done' not in st.session_state:
                st.session_state['ocr_data'] = datetime.now()
                st.session_state['ocr_stake'] = 10.0
                st.session_state['ocr_odd'] = 2.0
                st.session_state['ocr_desc'] = ""

            if uploaded_file:
                if st.button("🔍 Ler Imagem"):
                    try:
                        img = Image.open(uploaded_file)
                        d, s, o, desc, raw = ler_bilhete_completo(img, team_list)
                        st.session_state['ocr_data'] = d
                        st.session_state['ocr_stake'] = s if s > 0 else 10.0
                        st.session_state['ocr_odd'] = o if o > 1.0 else 2.0
                        st.session_state['ocr_desc'] = desc
                        st.success("Leitura Concluída!")
                    except Exception as e:
                        st.error(f"Erro na leitura: {e}")
        else:
            st.warning(f"⚠️ OCR Indisponível. Motivo: {OCR_ERROR_MSG}")

        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        dt = c1.date_input("Data", value=st.session_state.get('ocr_data', datetime.now()))
        sk = c2.number_input("Stake", value=st.session_state.get('ocr_stake', 10.0))
        od = c3.number_input("Odd", value=st.session_state.get('ocr_odd', 2.0))
        desc_final = st.text_area("Descrição", value=st.session_state.get('ocr_desc', ""))
        res = st.selectbox("Resultado", ["Aguardando ⏳", "Green ✅", "Red ❌"])
        
        if st.button("Salvar"):
            lucro = (sk * od - sk) if "Green" in res else (-sk if "Red" in res else 0.0)
            save_ticket({"id": str(uuid.uuid4())[:6], "data": str(dt), "stake": sk, "odd": od, "desc": desc_final, "res": res, "lucro": lucro})
            st.success("Salvo!")
            
        ts = get_tickets()
        if ts:
            total = sum([t['lucro'] for t in ts])
            st.metric("Lucro Global", f"R$ {total:.2f}")
            for t in reversed(ts):
                cls = "bet-card-green" if "Green" in t['res'] else "bet-card-red"
                st.markdown(f"<div class='{cls}'>{t['desc']} | R$ {t['lucro']:.2f}</div>", unsafe_allow_html=True)
                if st.button("Excluir", key=t['id']):
                    del_ticket(t['id'])
                    st.rerun()

if __name__ == "__main__":
    render_dashboard()
