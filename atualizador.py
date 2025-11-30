import pandas as pd
import requests
import json
from datetime import datetime
import pytz
from difflib import get_close_matches
import io

# ==============================================================================
# 1. CONFIGURAÇÕES
# ==============================================================================
NOME_ARQUIVO_CSV = "LISTA_COMPLETA_120_TIMES.csv"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# AS NOVAS URLS QUE VOCÊ DESCOBRIU (APIs JSON)
URLS_API = {
    "Premier League": "https://www.adamchoi.co.uk/scripts/data/json/scripts/pages/corners/detailed/corners_league_json.php?clflc=abc&league=E0&season=2025/2026",
    "La Liga": "https://www.adamchoi.co.uk/scripts/data/json/scripts/pages/corners/detailed/corners_league_json.php?clflc=abc&league=SP1&season=2025/2026",
    "Bundesliga": "https://www.adamchoi.co.uk/scripts/data/json/scripts/pages/corners/detailed/corners_league_json.php?clflc=abc&league=D1&season=2025/2026",
    "Serie A": "https://www.adamchoi.co.uk/scripts/data/json/scripts/pages/corners/detailed/corners_league_json.php?clflc=abc&league=I1&season=2025/2026",
    "Ligue 1": "https://www.adamchoi.co.uk/scripts/data/json/scripts/pages/corners/detailed/corners_league_json.php?clflc=abc&league=F1&season=2025/2026"
    # Brasileirão não está aqui, então o robô vai preservar os dados existentes dele.
}

MAPA_NOMES = {
    "Nott'm Forest": "Nottingham Forest",
    "Atlético Mineiro": "Atletico Mineiro",
    "Athletic Club": "Athletic Bilbao",
    "Manchester Utd": "Manchester United",
    "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves",
    "Brighton & Hove Albion": "Brighton",
    "Bayer Leverkusen": "Bayer Leverkusen",
    "Inter Milan": "Inter",
    "Milan": "AC Milan",
    "Paris S-G": "Paris Saint-Germain",
    "Betis": "Real Betis",
    "Sporting CP": "Sporting Lisbon",
    "Atletico Madrid": "Atl Madrid"
}

# ==============================================================================
# 2. MOTOR DE PROCESSAMENTO
# ==============================================================================
def limpar_nome(nome):
    return str(nome).strip()

def encontrar_time_no_csv(nome_site, lista_csv):
    nome_limpo = limpar_nome(nome_site)
    
    # 1. Mapa Manual
    if nome_limpo in MAPA_NOMES:
        alvo = MAPA_NOMES[nome_limpo]
        match = next((x for x in lista_csv if limpar_nome(x).lower() == alvo.lower()), None)
        if match: return match

    # 2. Busca Exata
    for nome_csv in lista_csv:
        if limpar_nome(nome_csv).lower() == nome_limpo.lower():
            return nome_csv

    # 3. Fuzzy Match
    matches = get_close_matches(nome_limpo, [limpar_nome(x) for x in lista_csv], n=1, cutoff=0.70)
    if matches:
        return next((x for x in lista_csv if limpar_nome(x) == matches[0]), None)
    return None

def processar_liga_api(nome_liga, url, df_csv, times_csv):
    print(f"📡 Conectando API: {nome_liga}...")
    updates = 0
    try:
        resp = requests.get(url, headers=HEADERS)
        if resp.status_code != 200: return 0
        
        dados_json = resp.json()
        stats_temp = {} # {Time: {acumuladores}}

        # Processa cada jogo do JSON
        for jogo in dados_json:
            home = limpar_nome(jogo.get('HomeTeam'))
            away = limpar_nome(jogo.get('AwayTeam'))
            
            # Garante chaves
            for t in [home, away]:
                if t not in stats_temp:
                    stats_temp[t] = {
                        'jogos_casa': 0, 'jogos_fora': 0,
                        'cantos_casa': 0, 'cantos_fora': 0,
                        'faltas_casa': 0, 'faltas_fora': 0,
                        'cartoes_casa': 0, 'cartoes_fora': 0,
                        'gols_pro_casa': 0, 'gols_sof_casa': 0,
                        'gols_pro_fora': 0, 'gols_sof_fora': 0
                    }

            # Acumula Mandante
            s = stats_temp[home]
            s['jogos_casa'] += 1
            s['cantos_casa'] += float(jogo.get('HomeCorners', 0))
            s['faltas_casa'] += float(jogo.get('HomeFouls', 0))
            s['cartoes_casa'] += float(jogo.get('HomeYellow', 0)) + float(jogo.get('HomeRed', 0))
            s['gols_pro_casa'] += float(jogo.get('HomeGoals', 0))
            s['gols_sof_casa'] += float(jogo.get('AwayGoals', 0))

            # Acumula Visitante
            s = stats_temp[away]
            s['jogos_fora'] += 1
            s['cantos_fora'] += float(jogo.get('AwayCorners', 0))
            s['faltas_fora'] += float(jogo.get('AwayFouls', 0))
            s['cartoes_fora'] += float(jogo.get('AwayYellow', 0)) + float(jogo.get('AwayRed', 0))
            s['gols_pro_fora'] += float(jogo.get('AwayGoals', 0))
            s['gols_sof_fora'] += float(jogo.get('HomeGoals', 0))

        # Calcula Médias e Atualiza CSV
        for time_site, dados in stats_temp.items():
            match_csv = encontrar_time_no_csv(time_site, times_csv)
            if match_csv:
                idx = df_csv.index[df_csv['Time'] == match_csv].tolist()[0]
                
                # CASA
                if dados['jogos_casa'] > 0:
                    df_csv.at[idx, 'Media_Escanteios_Casa'] = round(dados['cantos_casa'] / dados['jogos_casa'], 2)
                    df_csv.at[idx, 'Media_Faltas_Casa'] = round(dados['faltas_casa'] / dados['jogos_casa'], 2)
                    df_csv.at[idx, 'Media_Cartoes_Casa'] = round(dados['cartoes_casa'] / dados['jogos_casa'], 2)
                    # Se tiver colunas de gols no CSV:
                    # df_csv.at[idx, 'Media_Gols_Feitos_Casa'] = round(dados['gols_pro_casa'] / dados['jogos_casa'], 2)
                
                # FORA
                if dados['jogos_fora'] > 0:
                    df_csv.at[idx, 'Media_Escanteios_Fora'] = round(dados['cantos_fora'] / dados['jogos_fora'], 2)
                    df_csv.at[idx, 'Media_Faltas_Fora'] = round(dados['faltas_fora'] / dados['jogos_fora'], 2)
                    df_csv.at[idx, 'Media_Cartoes_Fora'] = round(dados['cartoes_fora'] / dados['jogos_fora'], 2)
                
                updates += 1
                
    except Exception as e:
        print(f"Erro na liga {nome_liga}: {e}")
        
    return updates

# ==============================================================================
# 3. MAIN
# ==============================================================================
def main():
    print("🚀 Iniciando Robô de Atualização (API JSON)...")
    
    try:
        df = pd.read_csv(NOME_ARQUIVO_CSV)
        # Limpa espaços nos nomes das colunas
        df.columns = [c.strip() for c in df.columns]
        times_csv = df['Time'].astype(str).tolist()
    except:
        print("❌ Erro: CSV não encontrado.")
        return

    total_updates = 0
    
    # Loop nas Ligas
    for liga, url in URLS_API.items():
        total_updates += processar_api_liga(liga, url, df, times_csv)
        time.sleep(1) # Respeito ao servidor

    if total_updates > 0:
        df.to_csv(NOME_ARQUIVO_CSV, index=False)
        print(f"\n✅ SUCESSO! {total_updates} times atualizados.")
        
        # Log
        hora = datetime.now(pytz.timezone('America/Sao_Paulo')).strftime("%d/%m/%Y %H:%M")
        try:
            with open("metadados.json", "w") as f:
                json.dump({"ultima_atualizacao": hora, "status": "API OK"}, f)
        except: pass
    else:
        print("\n🤷 Nenhuma atualização necessária.")

if __name__ == "__main__":
    main()
