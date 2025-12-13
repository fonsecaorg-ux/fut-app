import cloudscraper
import json
import time

scraper = cloudscraper.create_scraper()

ligas = {
    "serie_a_corners.json": {"league": "I1", "season": "2025/2026"},
    "serie_a_cards.json": {"league": "I1", "season": "2025/2026"},
    # você adiciona aqui as outras ligas
}

url = "https://www.adamchoi.co.uk/scripts/data/json/scripts/pages/corners/detailed/corners_league_json.php"

for arquivo, params in ligas.items():
    print(f"📡 Buscando {arquivo}...")
    time.sleep(3)
    response = scraper.get(url, params=params)

    if response.status_code == 200 and "application/json" in response.headers.get("Content-Type", ""):
        data = response.json()
        with open(arquivo, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"✅ {arquivo} atualizado com {len(data)} jogos")
    else:
        print(f"❌ Falha em {arquivo}: {response.status_code}")