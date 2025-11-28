import pandas as pd
import requests
import io
import time
import json
from datetime import datetime
import pytz
from difflib import get_close_matches

# ==============================================================================
# 1. CONFIGURAÇÕES
# ==============================================================================
# Nome exato do arquivo no GitHub
NOME_ARQUIVO_CSV = "dados_times.csv" 

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# Fontes de Dados
URL_ESCANTEIOS_ADAMCHOI = "https://www.adamchoi.co.uk/corners/detailed"
URLS_CARTOES_FBREF = [
    "https://fbref.com/en/comps/9/misc/Premier-League-Stats", 
    "https://fbref.com/en/comps/24/misc/Serie-A-Stats",       
    "https://fbref.com/en/comps/12/misc/La-Liga-Stats",       
    "https://fbref.com/en/comps/11/misc/Serie-A-Stats",       
    "https://fbref.com/en/comps/13/misc/Ligue-1-Stats",       
    "https://fbref.com/en/comps/20/misc/Bundesliga-Stats"     
]

# Mapa de Correção de Nomes
MAPA_MANUAL = {
    "Nott'm Forest": "Nottm Forest",
    "Atlético Mineiro": "Atletico-MG",
    "Athletic Club": "Ath Bilbao",
    "Manchester Utd": "Man Utd",
    "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves",
    "Brighton & Hove Albion": "Brighton",
    "Bayer Leverkusen": "Bayer 04 Leverkusen",
    "Inter Milan": "Inter",
    "Milan": "AC Milan",
    "Paris S-G": "Paris SG",
    "Betis": "Real Betis"
}

# ==============================================================================
# 2. FUNÇÕES AUXILIARES
# ==============================================================================
def limpar_string(texto):
    """Remove espaços invisíveis e padroniza."""
    return str(texto).strip()

def encontrar_nome_csv(nome_site, lista_nomes_csv):
    """Encontra o time correspondente no CSV (Exato, Mapa ou Fuzzy)."""
    nome_site_clean = limpar_string(nome_site)
    
    # 1. Mapa Manual
    if nome_site_clean in MAPA_MANUAL:
        target = MAPA_MANUAL[nome_site_clean]
        match = next((x for x in lista_nomes_csv if limpar_string(x).lower() == target.lower()), None)
        if match: return match

    # 2. Busca Exata
    for nome_csv in lista_nomes_csv:
        if limpar_string(nome_csv).lower() == nome_site_clean.lower():
            return nome_csv

    # 3. Fuzzy Match (Semelhança)
    lista_clean = [limpar_string(x) for x in lista_nomes_csv]
    matches = get_close_matches(nome_site_clean, lista_clean, n=1, cutoff=0.65)
    
    if matches:
        match_clean = matches[0]
        original = next((x for x in lista_nomes_csv if limpar_string(x) == match_clean), None)
        return original
    return None

# ==============================================================================
# 3. MÓDULOS DE EXTRAÇÃO
# ==============================================================================
def get_adamchoi_corners():
    print("\n--- 🚩 Adamchoi (Escanteios) ---")
    medias_finais = {}
    try:
        resp = requests.get(URL_ESCANTEIOS_ADAMCHOI, headers=HEADERS)
        if resp.status_code != 200: return {}
        
        # Tenta ler tabelas (com fallback de engine)
        try:
            dfs = pd.read_html(io.StringIO(resp.text))
        except:
            dfs = pd.read_html(io.StringIO(resp.text), flavor='bs4')
            
        if not dfs: return {}
        df = dfs[0]
        
        dados_times = {}
        # Itera sobre os jogos para calcular média real
        for index, row in df.iterrows():
            try:
                time_casa = limpar_string(row[1])
                placar = str(row[2])
                time_fora = limpar_string(row[3])
                
                if "-" in placar:
                    c_casa, c_fora = map(int, placar.split("-"))
                    
                    if time_casa not in dados_times: dados_times[time_casa] = []
                    dados_times[time_casa].append(c_casa)
                    
                    if time_fora not in dados_times: dados_times[time_fora] = []
                    dados_times[time_fora].append(c_fora)
            except: continue

        for t, lista in dados_times.items():
            if lista: medias_finais[t] = round(sum(lista) / len(lista), 2)
            
        return medias_finais
    except Exception as e:
        print(f"Erro Adamchoi: {e}")
        return {}

def processar_fbref(df_atual, times_csv):
    print("\n--- 🟨 FBref (Cartões/Faltas) ---")
    alteracoes = 0
    for url in URLS_CARTOES_FBREF:
        try:
            resp = requests.get(url, headers=HEADERS)
            if resp.status_code != 200: continue
            
            try:
                dfs = pd.read_html(io.StringIO(resp.text))
            except:
                dfs = pd.read_html(io.StringIO(resp.text), flavor='bs4')
                
            df_site = pd.DataFrame()
            # Encontra a tabela certa
            for t in dfs:
                if isinstance(t.columns, pd.MultiIndex):
                    t.columns = [' '.join(col).strip() for col in t.columns.values]
                if 'Squad' in t.columns and 'Performance CrdY' in t.columns:
                    df_site = t
                    break
            
            if df_site.empty: continue

            for index, row in df_site.iterrows():
                nome_match = encontrar_nome_csv(row['Squad'], times_csv)
                if nome_match:
                    jogos = float(row.get('90s', 1))
                    if jogos > 0:
                        n_ama = round(float(row['Performance CrdY']) / jogos, 2)
                        n_ver = round(float(row['Performance CrdR']) / jogos, 2)
                        n_fal = round(float(row['Performance Fls']) / jogos, 2)
                        
                        idx = df_atual.index[df_atual['Time'] == nome_match].tolist()[0]
                        
                        # Atualiza apenas se houver diferença numérica
                        if abs(float(df_atual.at[idx, 'Faltas']) - n_fal) > 0.01:
                            df_atual.at[idx, 'CartoesAmarelos'] = n_ama
                            df_atual.at[idx, 'CartoesVermelhos'] = n_ver
                            df_atual.at[idx, 'Faltas'] = n_fal
                            print(f"✅ {nome_match}: Atualizado.")
                            alteracoes += 1
            time.sleep(2)
        except Exception as e: print(f"Erro URL: {e}")
    return alteracoes

# ==============================================================================
# 4. EXECUÇÃO PRINCIPAL
# ==============================================================================
def main():
    print(f"🚀 Iniciando robô para: {NOME_ARQUIVO_CSV}")
    
    # 1. Carregar CSV
    try:
        df_atual = pd.read_csv(NOME_ARQUIVO_CSV)
        times_csv = df_atual['Time'].astype(str).tolist()
    except Exception as e:
        print(f"❌ Erro crítico: Não achei o arquivo {NOME_ARQUIVO_CSV}. {e}")
        return

    total = 0

    # 2. Atualizar Escanteios
    dict_cantos = get_adamchoi_corners()
    if dict_cantos:
        for nome, media in dict_cantos.items():
            match = encontrar_nome_csv(nome, times_csv)
            if match:
                idx = df_atual.index[df_atual['Time'] == match].tolist()[0]
                if abs(float(df_atual.at[idx, 'Escanteios']) - media) > 0.1:
                    df_atual.at[idx, 'Escanteios'] = media
                    total += 1
    
    # 3. Atualizar Cartões
    total += processar_fbref(df_atual, times_csv)

    # 4. GERAR METADADOS (O Bilhete de Data)
    # Aqui o robô apenas ESCREVE o JSON. Não há código visual (Streamlit).
    try:
        fuso_brasil = pytz.timezone('America/Sao_Paulo')
        data_hora = datetime.now(fuso_brasil).strftime("%d/%m/%Y às %H:%M")
        
        metadados = {
            "ultima_verificacao": data_hora,
            "status": "Sucesso",
            "fontes": "Adamchoi & FBref",
            "times_alterados": total
        }

        with open("metadados.json", "w", encoding='utf-8') as f:
            json.dump(metadados, f, ensure_ascii=False, indent=4)
            
        print(f"🕒 Data atualizada: {data_hora}")
    except Exception as e:
        print(f"⚠️ Erro ao salvar metadados: {e}")

    # 5. Salvar CSV (Se houve mudança)
    if total > 0:
        df_atual.to_csv(NOME_ARQUIVO_CSV, index=False)
        print(f"💾 SUCESSO! {total} times atualizados.")
    else:
        print("🤷 Sem mudanças numéricas.")

if __name__ == "__main__":
    main()
