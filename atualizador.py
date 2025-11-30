import pandas as pd
import requests
import json
import time
from datetime import datetime
import pytz
from difflib import get_close_matches
import io

# ==============================================================================
# 1. CONFIGURAÇÕES DA API (A MINA DE OURO 💎)
# ==============================================================================
NOME_ARQUIVO_CSV = "LISTA_COMPLETA_120_TIMES.csv"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Referer": "https://www.adamchoi.co.uk/"
}

# URLs das APIs JSON que você descobriu
URLS_API = {
    "Premier League": "https://www.adamchoi.co.uk/scripts/data/json/scripts/pages/corners/detailed/corners_league_json.php?clflc=abc&league=E0&season=2025/2026",
    "La Liga": "https://www.adamchoi.co.uk/scripts/data/json/scripts/pages/corners/detailed/corners_league_json.php?clflc=abc&league=SP1&season=2025/2026",
    "Bundesliga": "https://www.adamchoi.co.uk/scripts/data/json/scripts/pages/corners/detailed/corners_league_json.php?clflc=abc&league=D1&season=2025/2026",
    "Serie A": "https://www.adamchoi.co.uk/scripts/data/json/scripts/pages/corners/detailed/corners_league_json.php?clflc=abc&league=I1&season=2025/2026",
    "Ligue 1": "https://www.adamchoi.co.uk/scripts/data/json/scripts/pages/corners/detailed/corners_league_json.php?clflc=abc&league=F1&season=2025/2026"
}

# Mapa de Correção de Nomes (Site -> Teu CSV)
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
# 2. FUNÇÕES AUXILIARES
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

# ==============================================================================
# 3. MOTOR DE EXTRAÇÃO VIA API JSON
# ==============================================================================
def processar_api_liga(nome_liga, url, df_csv, times_csv):
    print(f"📡 Conectando API: {nome_liga}...")
    atualizacoes = 0
    
    try:
        resp = requests.get(url, headers=HEADERS)
        if resp.status_code != 200:
            print(f"❌ Erro {resp.status_code} na API.")
            return 0
            
        # O site retorna JSON direto
        dados_json = resp.json()
        
        # Estrutura para acumular estatísticas: {Time: {stats}}
        stats_temp = {}

        # Itera sobre todos os jogos no JSON
        for jogo in dados_json:
            try:
                # Nomes dos times
                home_team = limpar_nome(jogo.get('HomeTeam'))
                away_team = limpar_nome(jogo.get('AwayTeam'))
                
                # Inicializa se não existir
                if home_team not in stats_temp: stats_temp[home_team] = {'jogos_casa': 0, 'cantos_casa': 0, 'faltas_casa': 0, 'cartoes_casa': 0, 'gols_pro_casa': 0, 'gols_sof_casa': 0}
                if away_team not in stats_temp: stats_temp[away_team] = {'jogos_fora': 0, 'cantos_fora': 0, 'faltas_fora': 0, 'cartoes_fora': 0, 'gols_pro_fora': 0, 'gols_sof_fora': 0}

                # --- DADOS DO MANDANTE (CASA) ---
                stats_temp[home_team]['jogos_casa'] += 1
                stats_temp[home_team]['cantos_casa'] += int(jogo.get('HomeCorners', 0))
                stats_temp[home_team]['faltas_casa'] += int(jogo.get('HomeFouls', 0))
                stats_temp[home_team]['cartoes_casa'] += int(jogo.get('HomeYellow', 0)) + int(jogo.get('HomeRed', 0))
                stats_temp[home_team]['gols_pro_casa'] += int(jogo.get('HomeGoals', 0))
                stats_temp[home_team]['gols_sof_casa'] += int(jogo.get('AwayGoals', 0))

                # --- DADOS DO VISITANTE (FORA) ---
                stats_temp[away_team]['jogos_fora'] += 1
                stats_temp[away_team]['cantos_fora'] += int(jogo.get('AwayCorners', 0))
                stats_temp[away_team]['faltas_fora'] += int(jogo.get('AwayFouls', 0))
                stats_temp[away_team]['cartoes_fora'] += int(jogo.get('AwayYellow', 0)) + int(jogo.get('AwayRed', 0))
                stats_temp[away_team]['gols_pro_fora'] += int(jogo.get('AwayGoals', 0))
                stats_temp[away_team]['gols_sof_fora'] += int(jogo.get('HomeGoals', 0))

            except Exception as e:
                continue

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
                    # Se tivermos as colunas de gols no CSV, atualizamos também (opcional se seu CSV não tiver ainda)
                    # df_csv.at[idx, 'Media_Gols_Feitos_Casa'] = ... 
                
                # FORA
                if dados['jogos_fora'] > 0:
                    df_csv.at[idx, 'Media_Escanteios_Fora'] = round(dados['cantos_fora'] / dados['jogos_fora'], 2)
                    df_csv.at[idx, 'Media_Faltas_Fora'] = round(dados['faltas_fora'] / dados['jogos_fora'], 2)
                    df_csv.at[idx, 'Media_Cartoes_Fora'] = round(dados['cartoes_fora'] / dados['jogos_fora'], 2)

                atualizacoes += 1

    except Exception as e:
        print(f"Erro crítico na liga {nome_liga}: {e}")
    
    return atualizacoes

# ==============================================================================
# 4. MAIN
# ==============================================================================
def main():
    print(f"🚀 Iniciando Atualizador API v4.0...")
    
    # Carrega CSV Original
    try:
        df = pd.read_csv(NOME_ARQUIVO_CSV)
        # Garante colunas limpas
        df.columns = [c.strip() for c in df.columns]
        times_csv = df['Time'].astype(str).tolist()
    except:
        print("Erro: CSV não encontrado.")
        return

    total_updates = 0

    # Loop nas 5 Ligas Europeias (API)
    for liga, url in URLS_API.items():
        total_updates += processar_api_liga(liga, url, df, times_csv)

    # Salva Resultado (Preservando Brasileirão que não foi tocado)
    if total_updates > 0:
        df.to_csv(NOME_ARQUIVO_CSV, index=False)
        print(f"\n✅ SUCESSO! {total_updates} times atualizados via API.")
        
        # Log
        hora = datetime.now(pytz.timezone('America/Sao_Paulo')).strftime("%d/%m/%Y %H:%M")
        try:
            with open("metadados.json", "w") as f:
                json.dump({"ultima_atualizacao": hora, "metodo": "API JSON Direta"}, f)
        except: pass
    else:
        print("\n🤷 Nenhuma alteração necessária.")

if __name__ == "__main__":
    main()
