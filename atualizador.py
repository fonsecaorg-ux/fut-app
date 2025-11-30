# ==============================================================================
# 3. MÓDULO DE ESCANTEIOS (AGORA USANDO JSON COM TRATAMENTO DE ERRO)
# ==============================================================================
def get_adamchoi_corners():
    print("\n--- 🚩 Adamchoi (Escanteios) ---")
    medias_finais = {}
    total_jogos_lidos = 0

    for url in URLS_ESCANTEIOS_JSON:
        try:
            print(f"Lendo URL: {url[-15:]}") # Mostra o final do link para debug
            resp = requests.get(url, headers=HEADERS)
            
            # --- VERIFICAÇÃO CRÍTICA AQUI ---
            if resp.status_code != 200:
                print(f"❌ Erro HTTP: {resp.status_code}. Site bloqueou o acesso.")
                time.sleep(5) # Pausa maior após erro
                continue
            
            # Tenta decodificar o JSON (ponto onde o erro anterior ocorria)
            data = resp.json() 
            df = pd.DataFrame(data['data'])

            # Iteramos sobre a estrutura de estatísticas
            for team, stats in data['team_stats'].items():
                if stats['Total_Corners'] > 0:
                    media = stats['Total_Corners'] / stats['Total_Matches']
                    medias_finais[team] = round(media, 2)
            
            total_jogos_lidos += len(df)
            print(f"✅ Sucesso. {len(df)} jogos lidos.")
            
            time.sleep(2) # Pausa ética entre requisições para evitar rate limiting

        except requests.exceptions.JSONDecodeError:
            print("❌ ERRO JSON: Resposta vazia ou não JSON. Site enviou página de erro.")
        except Exception as e:
            print(f"⚠️ Erro ao processar: {e}")
            
    print(f"✅ {len(medias_finais)} times processados.")
    return medias_finais, total_jogos_lidos
