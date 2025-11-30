import pandas as pd
import requests
import io
import time
import json
from datetime import datetime
import pytz
from difflib import get_close_matches

# ==============================================================================
# 1. CONFIGURAÇÕES GERAIS E PROTEÇÃO DE DADOS
# ==============================================================================
# Nome do arquivo principal (não mudar!)
NOME_ARQUIVO_CSV = "dados_times.csv" 

# Headers para simular um navegador e evitar bloqueios
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

# Mapa de Correção de Nomes para o Fuzzy Match
MAPA_MANUAL = {
    "Nott'm Forest": "Nottm Forest", "Atlético Mineiro": "Atletico-MG", "Athletic Club": "Ath Bilbao",
    "Manchester Utd": "Man Utd", "Tottenham Hotspur": "Tottenham", "West Ham United": "West Ham", 
    "Wolverhampton Wanderers": "Wolves", "Brighton & Hove Albion": "Brighton", 
    "Bayer Leverkusen": "Bayer 04 Leverkusen", "Inter Milan": "Inter", "Milan": "AC Milan",
    "Paris S-G": "Paris SG", "Betis": "Real Betis"
}

# ==============================================================================
# 2. FUNÇÕES AUXILIARES
# ==============================================================================
def limpar_string(texto):
    return str(texto).strip()

def encontrar_nome_csv(nome_site, lista_nomes_csv):
    """Encontra o time no CSV, priorizando o mapa manual e o fuzzy match."""
    nome_site_clean = limpar_string(nome_site)
    
    if nome_site_clean in MAPA_MANUAL:
        target = MAPA_MANUAL[nome_site_clean]
        match = next((x for x in lista_nomes_csv if limpar_string(x).lower() == target.lower()), None)
        if match: return match

    for nome_csv in lista_nomes_csv:
        if limpar_string(nome_csv).lower() == nome_site_clean.lower():
            return nome_csv

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
    """
    MÓDULO DESATIVADO TEMPORARIAMENTE. 
    Retorna dados vazios para proteger o CSV da contaminação histórica.
    """
    return {}, 0 

def processar_fbref(df_atual, times_csv):
    """Processa a atualização de Cartões e Faltas (Módulo Seguro)."""
    alteracoes = 0
    for url in URLS_CARTOES_FBREF:
        try:
            resp = requests.get(url, headers=HEADERS)
            if resp.status_code != 200: continue
            
            try: dfs = pd.read_html(io.StringIO(resp.text))
            except: dfs = pd.read_html(io.StringIO(resp.text), flavor='bs4')
                
            df_site = pd.DataFrame()
            for t in dfs:
                if isinstance(t.columns, pd.MultiIndex):
                    t.columns = [' '.join(col).strip() for col in t.columns.values]
                if 'Squad' in t.columns and 'Performance CrdY' in t.columns:
                    df_site = t; break
            
            if df_site.empty: continue
            for index, row in df_site.iterrows():
                nome_match = encontrar_nome_csv(row['Squad'], times_csv)
                if nome_match:
                    jogos = float(row.get('90s', 1))
                    if jogos > 0:
                        n_ama = round(float(row['Performance CrdY']) / jogos, 2); n_ver = round(float(row['Performance CrdR']) / jogos, 2); n_fal = round(float(row['Performance Fls']) / jogos, 2)
                        idx = df_atual.index[df_atual['Time'] == nome_match].tolist()[0]
                        if abs(float(df_atual.at[idx, 'Faltas']) - n_fal) > 0.01:
                            df_atual.at[idx, 'CartoesAmarelos'] = n_ama; df_atual.at[idx, 'CartoesVermelhos'] = n_ver; df_atual.at[idx, 'Faltas'] = n_fal; alteracoes += 1
            time.sleep(2)
        except Exception as e: print(f"Erro URL: {e}")
    return alteracoes

# ==============================================================================
# 4. EXECUÇÃO PRINCIPAL
# ==============================================================================
def main():
    print(f"🚀 Iniciando robô para: {NOME_ARQUIVO_CSV}")
    
    try:
        df_atual = pd.read_csv(NOME_ARQUIVO_CSV)
        times_csv = df_atual['Time'].astype(str).tolist()
    except Exception as e:
        print(f"❌ Erro crítico: Não achei o arquivo {NOME_ARQUIVO_CSV}. {e}")
        return

    total = 0

    # 1. ATUALIZAR ESCANTEIOS (Ignorado: Módulo desativado na função acima)
    dict_cantos, jogos_lidos = get_adamchoi_corners() # Retorna {}, 0
    
    # 2. ATUALIZAR CARTÕES/FALTAS (Módulo FBref ativo)
    total += processar_fbref(df_atual, times_csv)

    # 3. GERAR METADADOS (O Bilhete Falante)
    try:
        fuso_brasil = pytz.timezone('America/Sao_Paulo')
        data_hora = datetime.now(fuso_brasil).strftime("%d/%m/%Y às %H:%M")
        
        # Mensagem de Log customizada
        msg = f"Controle Manual Ativo. Cartões/Faltas atualizados."
        if total > 0: msg += f" Salvos {total} times."
        else: msg += " Nenhuma alteração estatística relevante."

        metadados = {
            "ultima_verificacao": data_hora,
            "status": "Controle Manual Ativo", 
            "log": msg, 
            "fontes": "FBref (Ativo) / Adamchoi (Protegido)",
            "times_alterados": total
        }

        with open("metadados.json", "w", encoding='utf-8') as f:
            json.dump(metadados, f, ensure_ascii=False, indent=4)
        print(f"🕒 Metadados salvos. Status: {metadados['status']}")
    except Exception as e:
        print(f"⚠️ Erro ao salvar JSON: {e}")

    # 4. Salvar CSV (Commit)
    if total > 0:
        df_atual.to_csv(NOME_ARQUIVO_CSV, index=False)
        print(f"💾 SUCESSO! {total} atualizações salvas no CSV.")
    else:
        print("🤷 Nenhuma atualização numérica necessária.")

if __name__ == "__main__":
    main()
