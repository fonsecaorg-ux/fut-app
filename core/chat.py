import re
import streamlit as st
from typing import Dict, List, Tuple, Optional
from .utils import normalize_name, get_prob_emoji, format_currency
from .engine import calcular_jogo_v31



def render_md(text: str) -> str:
    """Force Streamlit markdown to respect line breaks."""
    return (text or "").replace("\n", "  \n")


def parse_bilhete_texto(texto: str) -> List[Dict]:
    """Parser inteligente de bilhetes - Versão ULTRA"""
    linhas_originais = [l.strip() for l in texto.split('\n') if l.strip()]
    linhas = []
    i = 0
    
    # Juntar linhas quebradas
    while i < len(linhas_originais):
        linha = linhas_originais[i]
        if i + 1 < len(linhas_originais):
            proxima = linhas_originais[i + 1]
            tem_mercado = any(x in linha.lower() for x in ['canto', 'escanteio', 'cartão', 'card'])
            tem_num = bool(re.search(r'\d+\.5', linha))
            tem_num_prox = bool(re.search(r'\d+\.5', proxima))
            
            if tem_mercado and not tem_num and tem_num_prox:
                linhas.append(linha + ' ' + proxima)
                i += 2
                continue
        linhas.append(linha)
        i += 1
    
    jogos = []
    jogo_atual = None
    time_pendente = None
    mercados_pend = []
    
    for linha in linhas:
        if any(x in linha.lower() for x in ['criar aposta', 'stake', 'retorno']):
            continue
        
        # Detectar jogo
        if ' vs ' in linha or ' x ' in linha.lower():
            sep = ' vs ' if ' vs ' in linha else ' x '
            partes = linha.split(sep)
            if len(partes) == 2:
                jogo_atual = {'home': partes[0].strip(), 'away': partes[1].strip(), 'mercados': mercados_pend.copy()}
                jogos.append(jogo_atual)
                time_pendente = None
                mercados_pend = []
                continue
        
        # Detectar mercado
        if any(x in linha.lower() for x in ['total de', 'mais de', 'over']) and \
           any(y in linha.lower() for y in ['canto', 'escanteio', 'cartão', 'card']):
            tipo = 'corners' if any(x in linha.lower() for x in ['canto', 'escanteio']) else 'cards'
            nums = re.findall(r'\d+\.5', linha)
            if nums:
                line = float(nums[0])
                odds = re.findall(r'@?\d+\.\d+', linha)
                odd = float(odds[-1].replace('@', '')) if odds else 2.0
                mercado = {'tipo': tipo, 'location': 'total', 'line': line, 'odd': odd, 'desc': linha}
                if jogo_atual:
                    jogo_atual['mercados'].append(mercado)
                else:
                    mercados_pend.append(mercado)
                continue
        
        # Times sem vs
        if not any(x in linha.lower() for x in ['total', 'mais de', 'over']) and len(linha) > 2:
            if time_pendente is None:
                time_pendente = linha.strip()
            else:
                jogo_atual = {'home': time_pendente, 'away': linha.strip(), 'mercados': mercados_pend.copy()}
                jogos.append(jogo_atual)
                time_pendente = None
                mercados_pend = []
    
    return jogos

def validar_jogos_bilhete(jogos_parsed: List[Dict], stats_db: Dict) -> List[Dict]:
    """Valida e normaliza nomes dos times"""
    jogos_val = []
    times = list(stats_db.keys())
    
    for jogo in jogos_parsed:
        h_norm = normalize_name(jogo['home'], times)
        a_norm = normalize_name(jogo['away'], times)
        
        if h_norm and a_norm and h_norm in STATS_db and a_norm in STATS_db:
            jogos_val.append({
                'home': h_norm,
                'away': a_norm,
                'home_original': jogo['home'],
                'away_original': jogo['away'],
                'mercados': jogo['mercados'],
                'home_stats': stats_db[h_norm],
                'away_stats': stats_db[a_norm]
            })
    
    return jogos_val

def calcular_prob_bilhete(jogos_validados: List[Dict], n_sims: int = 3000) -> Dict:
    """Calcula probabilidade real do bilhete"""
    prob_total = 1.0
    detalhes = []
    
    for jogo in jogos_validados:
        sims = simulate_game_v31(jogo['home_stats'], jogo['away_stats'], {}, n_sims)
        
        for mercado in jogo['mercados']:
            data = sims['corners_total'] if mercado['tipo'] == 'corners' else sims['cards_total']
            prob = (data > mercado['line']).mean()
            prob_total *= prob
            
            detalhes.append({
                'jogo': f"{jogo['home']} vs {jogo['away']}",
                'mercado': mercado['desc'],
                'prob': prob * 100,
                'odd_casa': mercado['odd'],
                'fair_odd': 1.0 / prob if prob > 0 else 999,
                'value': prob * mercado['odd'] if prob > 0 else 0
            })
    
    return {'prob_total': prob_total * 100, 'detalhes': detalhes}

def processar_chat(mensagem, stats_db):
    """Processa mensagens do chat e retorna resposta apropriada"""
    if not mensagem or not stats_db:
        return "Por favor, digite uma pergunta válida."
    
    msg = mensagem.lower().strip()
    
    # 1. COMANDOS ESPECIAIS
    if msg in ['/ajuda', 'ajuda', 'help']:
        return """
🤖 **COMANDOS DISPONÍVEIS:**

📊 **Análise de Times:**
- Digite o nome de um time (ex: "Arsenal", "Real Madrid")
- "Como está o Liverpool"
- "Estatísticas do Bayern"

⚔️ **Comparação (vs ou x):**
- "Arsenal vs Chelsea"
- "Real Madrid x Barcelona"

📅 **Jogos de Hoje:**
- "jogos de hoje"
- "partidas hoje"

🏆 **Rankings:**
- "top 10 cantos"
- "top 10 cartões"
- "ranking gols"

💡 **Dica:** Basta digitar o nome do time!
        """
    
    if msg in ['oi', 'olá', 'ola', 'hello', 'hi']:
        return "👋 Olá! Sou o FutPrevisão AI Advisor. Digite o nome de um time ou 'ajuda' para ver os comandos."
    
    # 2. JOGOS DE HOJE
    if 'hoje' in msg or 'today' in msg:
        try:
            hoje = datetime.now().strftime('%d/%m/%Y')
            jogos_hoje = CAL[CAL['Data'] == hoje]
            
            if len(jogos_hoje) == 0:
                return f"📅 Não há jogos cadastrados para hoje ({hoje})"
            
            resp = f"📅 **JOGOS DE HOJE ({hoje}):**\n\n"
            for idx, jogo in jogos_hoje.head(8).iterrows():
                resp += f"🏟️ {jogo['Time_Casa']} x {jogo['Time_Visitante']}\n"
                resp += f"   ⏰ {jogo['Hora']} | 🏆 {jogo['Liga']}\n\n"
            
            return resp
        except:
            return "❌ Erro ao buscar jogos de hoje."
    
    # 3. RANKINGS
    if any(word in msg for word in ['top', 'ranking', 'melhor', 'melhores']):
        metrica = 'corners'
        if 'cartao' in msg or 'cartõe' in msg or 'card' in msg:
            metrica = 'cards'
        elif 'gol' in msg or 'goal' in msg:
            metrica = 'goals_f'
        
        try:
            ranking = sorted(stats_db.items(), 
                           key=lambda x: x[1].get(metrica, 0), 
                           reverse=True)[:10]
            
            resp = f"🏆 **TOP 10 - {metrica.upper()}:**\n\n"
            for i, (time, stats) in enumerate(ranking, 1):
                valor = stats.get(metrica, 0)
                resp += f"{i}. {time}: {valor:.1f}/jogo\n"
            
            return resp
        except:
            return "❌ Erro ao gerar ranking."
    
    # 4. ANÁLISE H2H (vs ou x)
    if ' vs ' in msg or ' x ' in msg:
        separator = ' vs ' if ' vs ' in msg else ' x '
        times = msg.split(separator)
        
        if len(times) == 2:
            time1 = times[0].strip()
            time2 = times[1].strip()
            
            # Normalizar nomes
            from difflib import get_close_matches
            known_teams = list(stats_db.keys())
            
            match1 = get_close_matches(time1, known_teams, n=1, cutoff=0.6)
            match2 = get_close_matches(time2, known_teams, n=1, cutoff=0.6)
            
            if match1 and match2:
                t1 = match1[0]
                t2 = match2[0]
                s1 = stats_db[t1]
                s2 = stats_db[t2]
                
                resp = f"⚔️ **{t1} vs {t2}**\n\n"
                resp += f"**{t1}:**\n"
                resp += f"⚽ Ataque: {s1.get('goals_f', 0):.1f} gols/jogo\n"
                resp += f"🛡️ Defesa: {s1.get('goals_a', 0):.1f} sofridos/jogo\n"
                resp += f"🚩 Escanteios: {s1.get('corners', 0):.1f}/jogo\n"
                resp += f"🟨 Cartões: {s1.get('cards', 0):.1f}/jogo\n\n"
                
                resp += f"**{t2}:**\n"
                resp += f"⚽ Ataque: {s2.get('goals_f', 0):.1f} gols/jogo\n"
                resp += f"🛡️ Defesa: {s2.get('goals_a', 0):.1f} sofridos/jogo\n"
                resp += f"🚩 Escanteios: {s2.get('corners', 0):.1f}/jogo\n"
                resp += f"🟨 Cartões: {s2.get('cards', 0):.1f}/jogo\n\n"
                
                resp += "💡 Digite o nome de um time para análise completa!"
                
                return resp
            else:
                return f"❌ Times não encontrados. Disponíveis: {', '.join(known_teams[:5])}..."
    
    # 5. ANÁLISE DE TIME ÚNICO
    # Tentar encontrar time mencionado
    from difflib import get_close_matches
    known_teams = list(stats_db.keys())
    
    # Limpar mensagem
    palavras_ignorar = ['como', 'está', 'esta', 'o', 'a', 'do', 'da', 'de', 'stats', 'estatistica']
    msg_limpa = ' '.join([word for word in msg.split() if word not in palavras_ignorar])
    
    match = get_close_matches(msg_limpa, known_teams, n=1, cutoff=0.5)
    
    if match:
        team = match[0]
        stats = stats_db[team]
        
        resp = f"📊 **{team.upper()}**\n\n"
        resp += f"🏆 Liga: {stats.get('league', 'N/A')}\n"
        resp += f"🎮 Jogos: {stats.get('games', 0)}\n\n"
        
        # Ataque
        gols_f = stats.get('goals_f', 0)
        emoji_atk = '🔥' if gols_f > 1.8 else '⚽' if gols_f > 1.2 else '⚪'
        resp += f"**⚔️ ATAQUE:** {emoji_atk}\n"
        resp += f"⚽ Gols feitos: {gols_f:.2f}/jogo\n\n"
        
        # Defesa
        gols_a = stats.get('goals_a', 0)
        emoji_def = '🛡️' if gols_a < 1.0 else '⚠️' if gols_a < 1.5 else '🔴'
        resp += f"**🛡️ DEFESA:** {emoji_def}\n"
        resp += f"🥅 Gols sofridos: {gols_a:.2f}/jogo\n\n"
        
        # Escanteios
        corners = stats.get('corners', 0)
        emoji_corner = '🔥' if corners > 6.0 else '🚩' if corners > 5.0 else '⚪'
        resp += f"**🚩 ESCANTEIOS:** {emoji_corner}\n"
        resp += f"📐 Média: {corners:.2f}/jogo\n\n"
        
        # Cartões
        cards = stats.get('cards', 0)
        emoji_card = '🔴' if cards > 3.0 else '🟡' if cards > 2.0 else '🟢'
        resp += f"**🟨 CARTÕES:** {emoji_card}\n"
        resp += f"📋 Média: {cards:.2f}/jogo\n\n"
        
        # Faltas
        fouls = stats.get('fouls', 0)
        resp += f"**⚠️ FALTAS:**\n"
        resp += f"🚫 Média: {fouls:.2f}/jogo\n\n"
        
        resp += "💡 **Dica:** Compare com outro time usando 'vs' (ex: Arsenal vs Chelsea)"
        
        return resp
    
    # 6. NÃO ENTENDEU
    return f"🤔 Não entendi. Digite:\n- Nome de um time\n- 'Time1 vs Time2'\n- 'jogos de hoje'\n- '/ajuda' para ver comandos"
