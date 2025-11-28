import pandas as pd
import requests
import io
import time
from difflib import get_close_matches

# ==============================================================================
# 1. CONFIGURAÇÕES E MAPAS
# ==============================================================================
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# Onde buscar os dados?
URL_ESCANTEIOS_ADAMCHOI = "https://www.adamchoi.co.uk/corners/detailed"

URLS_CARTOES_FBREF = [
    "https://fbref.com/en/comps/9/misc/Premier-League-Stats", # Premier League
    "https://fbref.com/en/comps/24/misc/Serie-A-Stats",       # Série A Brasil
    "https://fbref.com/en/comps/12/misc/La-Liga-Stats",       # La Liga
    "https://fbref.com/en/comps/11/misc/Serie-A-Stats",       # Serie A Italia
    "https://fbref.com/en/comps/13/misc/Ligue-1-Stats",       # Ligue 1
    "https://fbref.com/en/comps/20/misc/Bundesliga-Stats"     # Bundesliga
]

# Dicionário para corrigir nomes que os sites escrevem diferente do seu CSV
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
# 2. FUNÇÕES DE AJUDA (UTILITIES)
# ==============================================================================
def limpar_string(texto):
    """Remove espaços invisíveis e padroniza o texto."""
    return str(texto).strip()

def encontrar_nome_csv(nome_site, lista_nomes_csv):
    """
    Tenta encontrar o time do site dentro da sua lista do CSV.
    Usa: 1. Mapa Manual | 2. Nome Exato | 3. Semelhança (Fuzzy)
    """
    nome_site_clean = limpar_string(nome_site)
    
    # 1. Tenta pelo Mapa Manual
    if nome_site_clean in MAPA_MANUAL:
        target = MAPA_MANUAL[nome_site_clean]
        match = next((x for x in lista_nomes_csv if limpar_string(x).lower() == target.lower()), None)
        if match: return match

    # 2. Tenta busca exata (ignorando maiúsculas)
    for nome_csv in lista_nomes_csv:
        if limpar_string(nome_csv).lower() == nome_site_clean.lower():
            return nome_csv

    # 3. Tenta por semelhança (Fuzzy Match - útil para pequenos erros de digitação)
    lista_clean = [limpar_string(x) for x in lista_nomes_csv]
    matches = get_close_matches(nome_site_clean, lista_clean, n=1, cutoff=0.65)
    
    if matches:
        match_clean = matches[0]
        # Recupera o nome original
        original = next((x for x in lista_nomes_csv if limpar_string(x) == match_clean), None)
        return original
    
    return None

# ==============================================================================
# 3. MÓDULO DE ESCANTEIOS (ADAMCHOI)
# ==============================================================================
def get_adamchoi_corners():
    print("\n--- 🚩 Iniciando Coleta de Escanteios (Adamchoi) ---")
    medias_finais = {}
    
    try:
        resp = requests.get(URL_ESCANTEIOS_ADAMCHOI, headers=HEADERS)
        if resp.status_code != 200:
            print(f"⚠️ Erro ao conectar no Adamchoi: {resp.status_code}")
            return {}

        # O Pandas lê todas as tabelas da página
        dfs = pd.read_html(io.StringIO(resp.text))
        if not dfs: return {}
        
        df = dfs[0] # Pega a tabela principal
        print(f"📊 Analisando histórico de {len(df)} partidas...")
        
        dados_times = {} # Acumulador: {Time: [lista_de_escanteios]}

        for index, row in df.iterrows():
            try:
                # O Adamchoi costuma ter: Data | Mandante | Placar | Visitante
                # Ajuste os índices [1], [2], [3] se a tabela mudar
                time_casa = limpar_string(row[1])
                placar = str(row[2])
                time_fora = limpar_string(row[3])
                
                if "-" in placar:
                    # Separa "7-5" em 7 e 5
                    cantos_casa, cantos_fora = map(int, placar.split("-"))
                    
                    if time_casa not in dados_times: dados_times[time_casa] = []
                    dados_times[time_casa].append(cantos_casa)
                    
                    if time_fora not in dados_times: dados_times[time_fora] = []
                    dados_times[time_fora].append(cantos_fora)
            except:
                continue

        # Calcula a média simples para cada time
        for t, lista in dados_times.items():
            if lista:
                medias_finais[t] = round(sum(lista) / len(lista), 2)
        
        print(f"✅ Médias calculadas para {len(medias_finais)} times.")
        return medias_finais

    except Exception as e:
        print(f"⚠️ Erro crítico no módulo Adamchoi: {e}")
        return {}

# ==============================================================================
# 4. MÓDULO DE CARTÕES E FALTAS (FBREF)
# ==============================================================================
def processar_fbref(df_atual, times_csv):
    print("\n--- 🟨 Iniciando Coleta de Cartões/Faltas (FBref) ---")
    alteracoes = 0
    
    for url in URLS_CARTOES_FBREF:
        print(f"🌐 Lendo: {url}")
        try:
            resp = requests.get(url, headers=HEADERS)
            if resp.status_code != 200: continue
            
            dfs = pd.read_html(io.StringIO(resp.text))
            df_site = pd.DataFrame()
            
            # Procura a tabela certa ("Miscellaneous Stats")
            for t in dfs:
                # Limpa o cabeçalho duplo do FBref
                if isinstance(t.columns, pd.MultiIndex):
                    t.columns = [' '.join(col).strip() for col in t.columns.values]
                
                # Verifica se tem as colunas que precisamos
                if 'Squad' in t.columns and 'Performance CrdY' in t.columns:
                    df_site = t
                    break
            
            if df_site.empty: continue

            # Itera sobre os times da tabela
            for index, row in df_site.iterrows():
                nome_site = row['Squad']
                nome_match = encontrar_nome_csv(nome_site, times_csv)
                
                if nome_match:
                    # Extrai os dados
                    jogos = float(row.get('90s', 1)) # Usa 1 para evitar divisão por zero
                    if jogos > 0:
                        n_ama = round(float(row['Performance CrdY']) / jogos, 2)
                        n_ver = round(float(row['Performance CrdR']) / jogos, 2)
                        n_fal = round(float(row['Performance Fls']) / jogos, 2)
                        
                        # Atualiza no CSV
                        idx = df_atual.index[df_atual['Time'] == nome_match].tolist()[0]
                        media_atual_falta = float(df_atual.at[idx, 'Faltas'])
                        
                        # Só atualiza e avisa se houver mudança numérica
                        if abs(media_atual_falta - n_fal) > 0.01:
                            df_atual.at[idx, 'CartoesAmarelos'] = n_ama
                            df_atual.at[idx, 'CartoesVermelhos'] = n_ver
                            df_atual.at[idx, 'Faltas'] = n_fal
                            print(f"✅ {nome_match}: Faltas atualizadas ({media_atual_falta} -> {n_fal})")
                            alteracoes += 1
            
            time.sleep(2) # Pausa para não ser bloqueado pelo site
            
        except Exception as e:
            print(f"⚠️ Erro ao processar URL {url}: {e}")
            
    return alteracoes

# ==============================================================================
# 5. EXECUÇÃO PRINCIPAL (MAIN)
# ==============================================================================
def main():
    print("🚀 ROBÔ ATUALIZADOR INICIADO")
    
    # 1. Carregar Base de Dados Atual
    try:
        df_atual = pd.read_csv("data_times.csv")
        times_csv = df_atual['Time'].astype(str).tolist()
        print(f"📂 Base carregada com sucesso: {len(df_atual)} times.")
    except Exception as e:
        print(f"❌ Erro: Não foi possível ler o arquivo 'data_times.csv'. {e}")
        return

    total_mudancas = 0

    # 2. Executar atualização de Escanteios
    dict_cantos = get_adamchoi_corners()
    if dict_cantos:
        print("🔄 Sincronizando Escanteios...")
        for nome_site, media in dict_cantos.items():
            nome_match = encontrar_nome_csv(nome_site, times_csv)
            if nome_match:
                idx = df_atual.index[df_atual['Time'] == nome_match].tolist()[0]
                atual = float(df_atual.at[idx, 'Escanteios'])
                
                if abs(atual - media) > 0.1:
                    df_atual.at[idx, 'Escanteios'] = media
                    print(f"🚩 {nome_match}: Escanteios {atual} -> {media}")
                    total_mudancas += 1

    # 3. Executar atualização de Cartões/Faltas
    total_mudancas += processar_fbref(df_atual, times_csv)

    # 4. Salvar Resultado Final
    if total_mudancas > 0:
        df_atual.to_csv("data_times.csv", index=False)
        print(f"\n💾 SUCESSO! {total_mudancas} times foram atualizados na base de dados.")
    else:
        print("\n🤷 Nenhuma atualização necessária. Tudo está em dia.")

if __name__ == "__main__":
    main()
