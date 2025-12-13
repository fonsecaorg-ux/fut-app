import requests
import time

# Session com headers completos
session = requests.Session()

session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/javascript, */*; q=0.01',
    'Accept-Language': 'pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7',
    'Referer': 'https://www.adamchoi.co.uk/overs/detailed',
    'Origin': 'https://www.adamchoi.co.uk'
})

print("🌐 Visitando página principal primeiro...")
session.get('https://www.adamchoi.co.uk/overs/detailed')
time.sleep(3)

print("📡 Buscando dados da Serie A...")
url = "https://www.adamchoi.co.uk/scripts/data/json/scripts/pages/corners/detailed/corners_league_json.php"
params = {'clflc': 'abc', 'league': 'I1', 'season': '2025/2026'}

response = session.get(url, params=params)

print(f"Status: {response.status_code}")

if response.status_code == 200:
    try:
        data = response.json()
        print(f"✅ Sucesso! {len(data)} jogos encontrados")
        print(f"Primeiro jogo: {data[0]['homeTeam']} vs {data[0]['awayTeam']}")
    except:
        print(f"❌ Resposta: {response.text[:200]}")
else:
    print(f"❌ Bloqueado! Resposta: {response.text[:200]}")


---

## 💡 **RESUMO:**
```
PROBLEMA:
❌ Status 403
❌ Cloudflare bloqueou
❌ Resposta HTML em vez de JSON

SOLUÇÕES:
✅ Headers mais completos
✅ Session persistente
✅ Warmup (visitar site antes)
✅ Delays maiores (3-5 segundos)
✅ Verificar Content-Type
✅ Se nada funcionar: cloudscraper
