
import requests
import json

# URL da Serie A que você me passou
url = "https://www.adamchoi.co.uk/scripts/data/json/scripts/pages/corners/detailed/corners_league_json.php?clflc=abc&league=I1&season=2025/2026"

print("🔍 TESTANDO SERIE A (I1)\n")
print("="*70)

try:
    # Fazer requisição
    print("📡 Fazendo requisição...")
    response = requests.get(url, timeout=15)
    
    # Status
    print(f"✅ Status: {response.status_code}")
    
    # Parsear JSON
    data = response.json()
    
    # Verificar tipo
    print(f"\n📊 ESTRUTURA:")
    print(f"   Tipo: {type(data).__name__}")
    
    if isinstance(data, list):
        print(f"   ✅ É uma LISTA")
        print(f"   📊 Total de jogos: {len(data)}")
        
        if len(data) > 0:
            primeiro_jogo = data[0]
            
            print(f"\n🔍 PRIMEIRO JOGO:")
            print(f"   Chaves disponíveis: {list(primeiro_jogo.keys())}")
            print(f"\n   Exemplo completo:")
            print(json.dumps(primeiro_jogo, indent=2))
            
            print(f"\n⚽ DADOS DO JOGO:")
            print(f"   Data: {primeiro_jogo.get('date')}")
            print(f"   Casa: {primeiro_jogo.get('homeTeam')}")
            print(f"   Fora: {primeiro_jogo.get('awayTeam')}")
            print(f"   Resultado: {primeiro_jogo.get('result')}")
            print(f"   Escanteios casa: {primeiro_jogo.get('homeCorners')}")
            print(f"   Escanteios fora: {primeiro_jogo.get('awayCorners')}")
            
    elif isinstance(data, dict):
        print(f"   ⚠️ É um DICIONÁRIO")
        print(f"   Chaves: {list(data.keys())}")
        print(f"\n   Estrutura:")
        print(json.dumps(data, indent=2)[:1000])
    
    print("\n" + "="*70)
    
except requests.exceptions.JSONDecodeError as e:
    print(f"❌ ERRO JSON: {e}")
    print(f"\n   Resposta recebida:")
    print(response.text[:500])
except Exception as e:
    print(f"❌ ERRO: {e}")