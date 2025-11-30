import pandas as pd
import requests
import io
import time
import json
from datetime import datetime
import pytz
from difflib import get_close_matches

# ==============================================================================
# 1. CONFIGURAÇÕES & FONTES (AGORA USANDO JSON LIMPO)
# ==============================================================================
NOME_ARQUIVO_CSV = "dados_times.csv" 

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# --- NOVAS FONTES JSON ENCONTRADAS POR VOCÊ ---
URLS_ESCANTEIOS_JSON = [
    "https://www.adamchoi.co.uk/scripts/data/json/scripts/pages/corners/detailed/corners_league_json.php?clflc=abc&league=E0&season=2025/2026", # Premier League
    "https://www.adamchoi.co.uk/scripts/data/json/scripts/pages/corners/detailed/corners_league_json.php?clflc=abc&league=SP1&season=2025/2026", # La Liga
    "https://www.adamchoi.co.uk/scripts/data/json/scripts/pages/corners/detailed/corners_league_json.php?clflc=abc&league=D1&season=2025/2026", # Bundesliga
    "https://www.adamchoi.co.uk/scripts/data/json/scripts/pages/corners/detailed/corners_league_json.php?clflc=abc&league=I1&season=2025/2026", # Serie A
    "https://www.adamchoi.co.uk/scripts/data/json/scripts/pages/corners/detailed/corners_league_json.php?clflc=abc&league=F1&season=2025/2026", # Ligue 1
    "https://www.adamchoi.co.uk/scripts/data/json/scripts/pages/corners/detailed/corners_league_json.php?clflc=abc&league=BR1&season=2025"  # Brasileiro
]
# ---------------------------------------------

URLS_CARTOES_FBREF = [
    # Manter FBref para Cartões/Faltas, que ainda são a melhor fonte para essas métricas
    "https://fbref.com/en/comps/9/misc/Premier-League-Stats", 
    "https://fbref.com/en/comps/24/misc/Serie-A-Stats",       
    # ... (outras URLs de cartões FBref) ...
]

# Mapa de Correção de Nomes (Intocado)
MAPA_MANUAL = {
    "Nott'm Forest": "Nottm Forest", "Atlético Mineiro": "Atletico-MG", "Athletic Club": "Ath Bilbao",
    "Manchester Utd": "Man Utd", "Tottenham Hotspur": "Tottenham", "West Ham United": "West Ham", 
    "Wolverhampton Wanderers": "Wolves", "Brighton & Hove Albion": "Brighton", 
    "Bayer Leverkusen": "Bayer 04 Leverkusen", "Inter Milan": "Inter", "Milan": "AC Milan",
    "Paris S-G": "Paris SG", "Betis": "Real Betis"
}

# ==============================================================================
# 2. FUNÇÕES AUXILIARES (Lógica de Limpeza Intocada)
# ==============================================================================
def limpar_string(texto): return str(texto).strip()

def encontrar_nome_csv(nome_site, lista_nomes_csv):
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
# 3. MÓDULO DE ESCANTEIOS (AGORA USANDO JSON)
# ==============================================================================
def get_adamchoi_corners():
    print("\n--- 🚩 Adamchoi (Escanteios) ---")
    medias_finais = {}
    total_jogos_lidos = 0

    for url in URLS_ESCANTEIOS_JSON:
        try:
            resp = requests.get(url, headers=HEADERS)
            if resp.status_code != 200: continue
            
            # Lendo JSON direto
            data = resp.json() 
            df = pd.DataFrame(data['data']) # A estrutura JSON tem um campo 'data'

            # O JSON já vem limpo, iteramos para calcular as médias
            for team, stats in data['team_stats'].items():
                if stats['Total_Corners'] > 0:
                    media = stats['Total_Corners'] / stats['Total_Matches']
                    medias_finais[team] = round(media, 2)
            
            total_jogos_lidos += len(df)

        except Exception as e:
            print(f"Erro ao ler JSON da URL {url}: {e}")
            
    print(f"✅ {len(medias_finais)} times processados de {total_jogos_lidos} jogos lidos.")
    return medias_finais, total_jogos_lidos 


def processar_fbref(df_atual, times_csv):
    """Processa a atualização de Cartões e Faltas (Módulo Seguro)."""
    # [Mantido o código de Cartões e Faltas do FBref aqui]
    alteracoes = 0
    # ... (Seu código FBref aqui) ...
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

    # 1. ATUALIZAR ESCANTEIOS (AGORA ATIVO E USANDO JSON LIMPO)
    dict_cantos, jogos_lidos = get_adamchoi_corners()
    if dict_cantos:
        for nome, media in dict_cantos.items():
            match = encontrar_nome_csv(nome, times_csv)
            if match:
                idx = df_atual.index[df_atual['Time'] == match].tolist()[0]
                if abs(float(df_atual.at[idx, 'Escanteios']) - media) > 0.1:
                    df_atual.at[idx, 'Escanteios'] = media
                    total += 1
    
    # 2. ATUALIZAR CARTÕES/FALTAS (FBref - Módulo Ativo)
    total += processar_fbref(df_atual, times_csv)

    # 3. GERAR METADADOS
    try:
        fuso_brasil = pytz.timezone('America/Sao_Paulo')
        data_hora = datetime.now(fuso_brasil).strftime("%d/%m/%Y às %H:%M")
        
        # Mensagem de Log customizada
        msg = f"Li {jogos_lidos} jogos de 6 ligas. Cartões/Faltas atualizados."
        if total > 0: msg += f" Salvos {total} times."
        else: msg += " Nenhuma alteração estatística relevante."

        metadados = {
            "ultima_verificacao": data_hora,
            "status": "Controle Manual Ativo", 
            "log": msg, 
            "fontes": "JSON Adamchoi & FBref",
            "times_alterados": total
        }

        with open("metadados.json", "w", encoding='utf-8') as f:
            json.dump(metadados, f, ensure_ascii=False, indent=4)
        print(f"🕒 Metadados salvos. Status: {metadados['status']}")
    except Exception as e:
        print(f"⚠️ Erro ao salvar JSON: {e}")

    # 4. Salvar CSV
    if total > 0:
        df_atual.to_csv(NOME_ARQUIVO_CSV, index=False)
        print(f"💾 SUCESSO! {total} atualizações salvas no CSV.")
    else:
        print("🤷 Nenhuma atualização numérica necessária.")

if __name__ == "__main__":
    main()
