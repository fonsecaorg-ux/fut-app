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

# Configuração OCR
try:
    import pytesseract
    # AJUSTE O CAMINHO SE NECESSÁRIO
    path_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(path_tesseract):
        pytesseract.pytesseract.tesseract_cmd = path_tesseract
    HAS_OCR = True
except ImportError:
    HAS_OCR = False

# ==============================================================================
# 0. CONFIGURAÇÃO
# ==============================================================================
st.set_page_config(page_title="FutPrevisão Pro V5.2 (OCR Inteligente)", layout="wide", page_icon="⚽")

# ==============================================================================
# 1. FUNÇÕES MATEMÁTICAS
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
# 2. DADOS (Carregamento Mestre)
# ==============================================================================
GENERIC_STATS = {"corners": 5.0, "cards": 2.2, "fouls": 11.5}
INTERNAL_DB = {
    "Arsenal": {"corners": 7.5, "cards": 1.5, "fouls": 9.5},
    "Man City": {"corners": 8.2, "cards": 1.4, "fouls": 8.5},
    "Liverpool": {"corners": 7.2, "cards": 1.8, "fouls": 10.5},
    "Real Madrid": {"corners": 6.8, "cards": 1.9, "fouls": 10.0},
    "Barcelona": {"corners": 6.5, "cards": 2.0, "fouls": 10.2},
    # Adicione mais times genéricos para o OCR achar se o CSV falhar
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
# 3. OCR INTELIGENTE (TIMES + LINHAS)
# ==============================================================================
def extrair_times_e_linhas(texto, lista_times):
    texto_low = texto.lower()
    times_encontrados = []
    
    # 1. Busca Times (Ordena por tamanho para pegar "Manchester City" antes de "City")
    lista_ordenada = sorted(lista_times, key=len, reverse=True)
    for time in lista_ordenada:
        if len(time) > 3 and time.lower() in texto_low:
            # Evita duplicatas parciais
            if not any(time in t_exist for t_exist in times_encontrados):
                times_encontrados.append(time)
    
    # Pega apenas os 2 primeiros (Mandante x Visitante)
    times_encontrados = times_encontrados[:2]
    
    # 2. Busca Linhas/Mercados (Keywords)
    linhas_encontradas = []
    keywords = [
        "escanteios", "cantos", "corners", 
        "cartões", "cards", "amarelos", 
        "gols", "goals", "ambas marcam",
        "over", "under", "mais de", "menos de",
        "finalização", "chutes"
    ]
    
    # Tenta extrair frases curtas que contenham números e keywords
    # Ex: "Mais de 9.5 escanteios"
    lines = texto.split('\n')
    for line in lines:
        line_low = line.lower()
        if any(k in line_low for k in keywords):
            # Limpa caracteres estranhos do OCR
            clean = re.sub(r'[^a-zA-Z0-9\.\s]', '', line).strip()
            if len(clean) > 5:
                linhas_encontradas.append(clean)

    return times_encontrados, linhas_encontradas

def ler_bilhete_completo(imagem, lista_times):
    try:
        text = pytesseract.image_to_string(imagem)
        
        # Dados Financeiros
        data_det = datetime.now()
        stake_det = 0.0
        odd_det = 0.0
        
        lines = text.split('\n')
        for line in lines:
            line_low = line.lower()
            # Odd
            if "@" in line_low or "odd" in line_low or "cota" in line_low:
                nums = re.findall(r"\d+\.\d+", line)
                if nums:
                    for n in nums:
                        if 1.01 < float(n) < 100.0: odd_det = float(n)
            # Stake
            if any(x in line_low for x in ["aposta", "valor", "stake", "total"]):
                clean = line_low.replace("r$", "").replace(",", ".")
                nums = re.findall(r"\d+\.\d+", clean)
                if nums:
                    v = float(nums[0])
                    if v > 0: stake_det = v

        # Dados de Jogo
        times, linhas = extrair_times_e_linhas(text, lista_times)
        
        # Monta Descrição
        desc = ""
        if times:
            desc += " x ".join(times)
        if linhas:
            if desc: desc += " | "
            desc += ", ".join(linhas)
            
        return data_det, stake_det, odd_det, desc, text
        
    except Exception as e:
        return datetime.now(), 0.0, 0.0, "", f"Erro: {str(e)}"

# ==============================================================================
# 4. SISTEMA PRINCIPAL (Mantido da V5.1 mas simplificado)
# ==============================================================================
# ... (Funções de histórico, cálculo e previsão idênticas à V5.1 - Omitidas aqui para focar no OCR) ...
# Vou reincluir o código base necessário para rodar

NAME_MAPPING = {"Man City": "Man City", "Man Utd": "Man United"} # Simplificado
history_loader = None # Placeholder se não carregar a classe completa, mas vamos assumir que está lá

# Recriando a classe Loader minima
class AdamChoiLoader:
    def __init__(self): self.data = {}
    def get_history_global(self, t, m, k): return None # Placeholder para não quebrar
history_loader = AdamChoiLoader()

# Funções de Gestão de Tickets com Campo Descrição
DATA_FILE = "tickets_v52.json"
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
# 5. RENDERIZAÇÃO
# ==============================================================================
USERS = {"diego": "@Casa612"}
def check_login():
    if "logged_in" not in st.session_state: st.session_state["logged_in"] = False
    if st.session_state["logged_in"]: return True
    st.markdown("### 🔒 Acesso V5.2")
    c1, c2 = st.columns([1,2])
    with c1:
        if st.button("Entrar (Dev Mode)"): # Atalho para teste
            st.session_state["logged_in"] = True
            st.rerun()
    return False

if not check_login(): st.stop()

def render_dashboard():
    st.title("📊 FutPrevisão V5.2 (OCR Completo)")
    
    st.markdown("""
    <style>
        .bet-card-green { border-left: 5px solid #28a745; background: #f0fff4; padding: 10px; margin-bottom: 5px; }
        .bet-card-red { border-left: 5px solid #dc3545; background: #fff5f5; padding: 10px; margin-bottom: 5px; }
    </style>
    """, unsafe_allow_html=True)

    tab_ocr, tab_hist = st.tabs(["📷 Registrar Bilhete", "📜 Histórico"])

    with tab_ocr:
        st.subheader("Importar Bilhete (Superbet/Bet365)")
        uploaded_file = st.file_uploader("Subir Print", type=["png", "jpg", "jpeg"])
        
        # Variáveis de Estado
        if 'ocr_done' not in st.session_state:
            st.session_state['ocr_data'] = datetime.now()
            st.session_state['ocr_stake'] = 10.0
            st.session_state['ocr_odd'] = 2.0
            st.session_state['ocr_desc'] = ""

        if uploaded_file and HAS_OCR:
            if st.button("🔍 Ler Imagem"):
                img = Image.open(uploaded_file)
                d, s, o, desc, raw = ler_bilhete_completo(img, team_list)
                st.session_state['ocr_data'] = d
                st.session_state['ocr_stake'] = s if s > 0 else 10.0
                st.session_state['ocr_odd'] = o if o > 1.0 else 2.0
                st.session_state['ocr_desc'] = desc
                st.success("Leitura Concluída!")
                with st.expander("Ver Texto Bruto"):
                    st.text(raw)
        elif uploaded_file and not HAS_OCR:
            st.error("Tesseract não instalado no servidor.")

        st.markdown("---")
        st.write("### Conferir e Salvar")
        
        c1, c2, c3 = st.columns(3)
        dt = c1.date_input("Data", value=st.session_state['ocr_data'])
        sk = c2.number_input("Stake (R$)", value=st.session_state['ocr_stake'])
        od = c3.number_input("Odd Total", value=st.session_state['ocr_odd'])
        
        desc_final = st.text_area("Descrição (Jogos/Linhas)", value=st.session_state['ocr_desc'], help="O robô tenta preencher isso automaticamente.")
        
        res = st.selectbox("Resultado", ["Aguardando ⏳", "Green ✅", "Red ❌", "Cashout 💰"])
        
        if st.button("💾 Salvar no Histórico", type="primary"):
            lucro = 0.0
            if "Green" in res: lucro = (sk * od) - sk
            elif "Red" in res: lucro = -sk
            
            t = {
                "id": str(uuid.uuid4())[:6],
                "data": str(dt),
                "stake": sk,
                "odd": od,
                "desc": desc_final,
                "res": res,
                "lucro": lucro
            }
            save_ticket(t)
            st.success("Bilhete Registrado!")
            
    with tab_hist:
        ts = get_tickets()
        if ts:
            total = sum([t['lucro'] for t in ts])
            st.metric("Lucro Global", f"R$ {total:.2f}")
            
            for t in reversed(ts):
                cls = "bet-card-green" if "Green" in t['res'] else ("bet-card-red" if "Red" in t['res'] else "")
                st.markdown(f"""
                <div class="{cls}">
                    <strong>{t['data']}</strong> | {t['desc']}<br>
                    Stake: {t['stake']} | Odd: {t['odd']} | <strong>{t['res']}</strong> (R$ {t['lucro']:.2f})
                </div>
                """, unsafe_allow_html=True)
                if st.button("Excluir", key=t['id']):
                    del_ticket(t['id'])
                    st.rerun()

if __name__ == "__main__":
    render_dashboard()
