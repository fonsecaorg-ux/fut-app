def get_available_markets_for_game(res: Dict, probs: Dict) -> List[Dict]:
    """Retorna lista de mercados disponíveis para seleção manual"""
    markets = []
    
    # Cantos Casa
    for l in [3.5, 4.5, 5.5]:
        p = probs['corners']['home'].get(f'Over {l}', 0)
        markets.append({'mercado': f"{res['home']} Over {l} Escanteios", 'prob': p, 'odd': get_fair_odd(p), 'type': 'Escanteios'})
        
    # Cantos Fora
    for l in [2.5, 3.5, 4.5]:
        p = probs['corners']['away'].get(f'Over {l}', 0)
        markets.append({'mercado': f"{res['away']} Over {l} Escanteios", 'prob': p, 'odd': get_fair_odd(p), 'type': 'Escanteios'})
        
    # Cartões
    for l in [1.5, 2.5]:
        p1 = probs['cards']['home'].get(f'Over {l}', 0)
        markets.append({'mercado': f"{res['home']} Over {l} Cartões", 'prob': p1, 'odd': get_fair_odd(p1), 'type': 'Cartões'})
        
        p2 = probs['cards']['away'].get(f'Over {l}', 0)
        markets.append({'mercado': f"{res['away']} Over {l} Cartões", 'prob': p2, 'odd': get_fair_odd(p2), 'type': 'Cartões'})
        
    return markets
