import math
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional


def calcular_jogo_v31(home_stats: Dict, away_stats: Dict, ref_data: Dict) -> Dict:
    """
    Motor de Cálculo V31 - Causality Engine
    
    Filosofia: CAUSA → EFEITO
    - Chutes no gol → Cantos
    - Faltas → Cartões
    - Árbitro → Rigidez
    """
    
    # ESCANTEIOS com boost de chutes
    base_corners_h = home_STATS.get('corners_home', home_STATS['corners'])
    base_corners_a = away_STATS.get('corners_away', away_STATS['corners'])
    
    # Boost baseado em chutes no gol
    shots_h = home_STATS.get('shots_home', 4.5)
    shots_a = home_STATS.get('shots_away', 4.0)
    
    if shots_h > 6.0:
        pressure_h = 1.20  # Alto
    elif shots_h > 4.5:
        pressure_h = 1.10  # Médio
    else:
        pressure_h = 1.0   # Baixo
    
    # Fator casa/fora
    corners_h = base_corners_h * 1.15 * pressure_h
    corners_a = base_corners_a * 0.90
    corners_total = corners_h + corners_a
    
    # CARTÕES
    fouls_h = home_STATS.get('fouls_home', home_STATS.get('fouls', 12.0))
    fouls_a = away_STATS.get('fouls_away', away_STATS.get('fouls', 12.0))
    
    # Fator de violência
    violence_h = 1.0 if fouls_h > 12.5 else 0.85
    violence_a = 1.0 if fouls_a > 12.5 else 0.85
    
    # Fator do árbitro
    ref_factor = ref_data.get('factor', 1.0) if ref_data else 1.0
    ref_red_rate = ref_data.get('red_rate', 0.08) if ref_data else 0.08
    
    # Rigidez do árbitro
    if ref_red_rate > 0.12:
        strictness = 1.15
    elif ref_red_rate > 0.08:
        strictness = 1.08
    else:
        strictness = 1.0
    
    cards_h_base = home_STATS.get('cards_home', home_STATS['cards'])
    cards_a_base = away_STATS.get('cards_away', away_STATS['cards'])
    
    cards_h = cards_h_base * violence_h * ref_factor * strictness
    cards_a = cards_a_base * violence_a * ref_factor * strictness
    cards_total = cards_h + cards_a
    
    # Probabilidade de cartão vermelho
    prob_red_card = ((0.05 + 0.05) / 2) * ref_red_rate * 100
    
    # xG (Expected Goals)
    xg_h = (home_STATS['goals_f'] * away_STATS['goals_a']) / 1.3
    xg_a = (away_STATS['goals_f'] * home_STATS['goals_a']) / 1.3
    
    return {
        'corners': {'h': corners_h, 'a': corners_a, 't': corners_total},
        'cards': {'h': cards_h, 'a': cards_a, 't': cards_total},
        'goals': {'h': xg_h, 'a': xg_a},
        'metadata': {
            'ref_factor': ref_factor,
            'violence_home': fouls_h > 12.5,
            'violence_away': fouls_a > 12.5,
            'pressure_home': pressure_h,
            'shots_home': shots_h,
            'shots_away': shots_a,
            'strictness': strictness,
            'prob_red_card': prob_red_card
        }
    }

def simulate_game_v31(home_stats: Dict, away_stats: Dict, ref_data: Dict, n_sims: int = 3000) -> Dict:
    """Simulador de Monte Carlo com distribuição de Poisson"""
    calc = calcular_jogo_v31(home_stats, away_stats, ref_data)
    
    return {
        'corners_h': np.random.poisson(calc['corners']['h'], n_sims),
        'corners_a': np.random.poisson(calc['corners']['a'], n_sims),
        'corners_total': np.random.poisson(calc['corners']['t'], n_sims),
        'cards_h': np.random.poisson(calc['cards']['h'], n_sims),
        'cards_a': np.random.poisson(calc['cards']['a'], n_sims),
        'cards_total': np.random.poisson(calc['cards']['t'], n_sims),
        'goals_h': np.random.poisson(calc['goals']['h'], n_sims),
        'goals_a': np.random.poisson(calc['goals']['a'], n_sims)
    }

def calculate_sharpe_ratio(returns: List[float]) -> float:
    """Calcula Sharpe Ratio (retorno ajustado ao risco)"""
    if not returns or len(returns) < 2:
        return 0.0
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    return (mean_return - 1.0) / std_return if std_return > 0 else 0.0

def calculate_max_drawdown(bankroll_history: List[float]) -> float:
    """Calcula Maximum Drawdown (maior queda)"""
    if len(bankroll_history) < 2:
        return 0.0
    peak = bankroll_history[0]
    max_dd = 0.0
    for value in bankroll_history:
        if value > peak:
            peak = value
        dd = (peak - value) / peak * 100
        if dd > max_dd:
            max_dd = dd
    return max_dd

def calculate_kelly_criterion(prob: float, odd: float, bankroll: float) -> Dict:
    """Calcula Kelly Criterion"""
    if prob <= 0 or prob >= 1 or odd <= 1:
        return {'fraction': 0, 'stake': 0, 'recommendation': 'Não apostar'}
    
    b = odd - 1
    p = prob
    q = 1 - prob
    
    kelly_fraction = (b * p - q) / b
    kelly_fraction = max(0, min(kelly_fraction, 0.10))  # Cap em 10%
    
    stake = bankroll * kelly_fraction
    
    if kelly_fraction >= 0.08:
        recommendation = 'Stake alto'
    elif kelly_fraction >= 0.05:
        recommendation = 'Stake médio'
    elif kelly_fraction > 0:
        recommendation = 'Stake baixo'
    else:
        recommendation = 'Não apostar'
    
    return {
        'fraction': kelly_fraction,
        'stake': stake,
        'percentage': kelly_fraction * 100,
        'recommendation': recommendation
    }

def calculate_roi(total_staked: float, total_profit: float) -> float:
    """Calcula ROI (Return on Investment)"""
    if total_staked == 0:
        return 0.0
    return (total_profit / total_staked) * 100

def generate_corner_distribution_chart(team_stats: Dict, team_name: str) -> go.Figure:
    """Gera gráfico de distribuição de cantos de um time"""
    corners_mean = team_STATS.get('corners', 5.5)
    corners_std = team_STATS.get('corners_std', 2.0)
    
    x = np.linspace(0, 15, 100)
    y = (1 / (corners_std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - corners_mean) / corners_std) ** 2)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, fill='tozeroy', name=team_name, line=dict(color='orange', width=2)))
    fig.update_layout(
        title=f'Distribuição de Cantos - {team_name}',
        xaxis_title='Número de Cantos',
        yaxis_title='Densidade',
        height=300
    )
    return fig

def generate_comparison_radar(home_stats: Dict, away_stats: Dict, home_name: str, away_name: str) -> go.Figure:
    """Gera radar chart comparativo entre dois times"""
    categories = ['Cantos', 'Cartões', 'Gols Marcados', 'Chutes', 'Faltas']
    
    home_values = [
        home_STATS.get('corners', 5.5) / 10 * 100,
        home_STATS.get('cards', 2.5) / 5 * 100,
        home_STATS.get('goals_f', 1.5) / 3 * 100,
        home_STATS.get('shots_on_target', 4.5) / 8 * 100,
        home_STATS.get('fouls', 12.0) / 15 * 100
    ]
    
    away_values = [
        away_STATS.get('corners', 5.5) / 10 * 100,
        away_STATS.get('cards', 2.5) / 5 * 100,
        away_STATS.get('goals_f', 1.5) / 3 * 100,
        away_STATS.get('shots_on_target', 4.5) / 8 * 100,
        away_STATS.get('fouls', 12.0) / 15 * 100
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=home_values,
        theta=categories,
        fill='toself',
        name=home_name,
        line=dict(color='blue')
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=away_values,
        theta=categories,
        fill='toself',
        name=away_name,
        line=dict(color='red')
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        height=400
    )
    
    return fig

def generate_heatmap_correlations(stats_db: Dict) -> go.Figure:
    """Gera heatmap de correlações entre métricas"""
    data_matrix = []
    
    for team, stats in STATS_db.items():
        data_matrix.append([
            STATS.get('corners', 5.5),
            STATS.get('cards', 2.5),
            STATS.get('goals_f', 1.5),
            STATS.get('fouls', 12.0),
            STATS.get('shots_on_target', 4.5)
        ])
    
    df = pd.DataFrame(data_matrix, columns=['Cantos', 'Cartões', 'Gols', 'Faltas', 'Chutes'])
    corr_matrix = df.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=['Cantos', 'Cartões', 'Gols', 'Faltas', 'Chutes'],
        y=['Cantos', 'Cartões', 'Gols', 'Faltas', 'Chutes'],
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 10}
    ))
    
    fig.update_layout(
        title='Matriz de Correlação entre Métricas',
        height=500
    )
    
    return fig

def calculate_poisson_probability(expected: float, actual: int) -> float:
    """Calcula probabilidade de Poisson para um valor específico"""
    return (expected ** actual) * np.exp(-expected) / math.factorial(actual)

def generate_poisson_distribution(expected: float, max_value: int = 20) -> go.Figure:
    """Gera gráfico de distribuição de Poisson"""
    x_values = list(range(max_value))
    y_values = [calculate_poisson_probability(expected, x) for x in x_values]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x_values,
        y=y_values,
        marker_color='lightblue',
        name='Probabilidade'
    ))
    
    fig.update_layout(
        title=f'Distribuição de Poisson (λ = {expected:.2f})',
        xaxis_title='Número de Eventos',
        yaxis_title='Probabilidade',
        height=350
    )
    
    return fig

def calculate_implied_probability(odds: List[float]) -> float:
    """Calcula probabilidade implícita total das odds"""
    total = sum(1/odd for odd in odds if odd > 0)
    margin = (total - 1) * 100
    return margin

def find_arbitrage_opportunities(odds_home: float, odds_draw: float, odds_away: float) -> Dict:
    """Detecta oportunidades de arbitragem"""
    implied_total = (1/odds_home) + (1/odds_draw) + (1/odds_away)
    
    if implied_total < 1.0:
        profit_pct = ((1 / implied_total) - 1) * 100
        
        stake_home = (1/odds_home) / implied_total * 100
        stake_draw = (1/odds_draw) / implied_total * 100
        stake_away = (1/odds_away) / implied_total * 100
        
        return {
            'exists': True,
            'profit_pct': profit_pct,
            'stake_home': stake_home,
            'stake_draw': stake_draw,
            'stake_away': stake_away
        }
    
    return {'exists': False}

def calculate_ev(probability: float, odds: float, stake: float) -> float:
    """Calcula Expected Value (valor esperado)"""
    win_amount = stake * (odds - 1)
    lose_amount = -stake
    
    ev = (probability * win_amount) + ((1 - probability) * lose_amount)
    return ev

def calculate_variance(returns: List[float]) -> float:
    """Calcula variância dos retornos"""
    if len(returns) < 2:
        return 0.0
    return np.var(returns)

def calculate_sortino_ratio(returns: List[float], target_return: float = 0.0) -> float:
    """Calcula Sortino Ratio (considera apenas downside risk)"""
    if len(returns) < 2:
        return 0.0
    
    mean_return = np.mean(returns)
    downside_returns = [r for r in returns if r < target_return]
    
    if not downside_returns:
        return 0.0
    
    downside_std = np.std(downside_returns)
    
    if downside_std == 0:
        return 0.0
    
    return (mean_return - target_return) / downside_std

def calculate_calmar_ratio(returns: List[float], max_drawdown: float) -> float:
    """Calcula Calmar Ratio (retorno / max drawdown)"""
    if max_drawdown == 0:
        return 0.0
    
    mean_return = np.mean(returns) if returns else 0.0
    return mean_return / (max_drawdown / 100)

def format_percentage(value: float, decimals: int = 1) -> str:
    """Formata percentual"""
    return f"{value:.{decimals}f}%"

def format_multiplier(value: float, decimals: int = 2) -> str:
    """Formata multiplicador"""
    return f"{value:.{decimals}f}x"

def get_league_emoji(league_name: str) -> str:
    """Retorna emoji da liga"""
    emojis = {
        'Premier League': '🏴󐁧󐁢󐁥󐁮󐁧󐁿',
        'La Liga': '🇪🇸',
        'Serie A': '🇮🇹',
        'Bundesliga': '🇩🇪',
        'Ligue 1': '🇫🇷',
        'Championship': '🏴󐁧󐁢󐁥󐁮󐁧󐁿',
        'Bundesliga 2': '🇩🇪',
        'Pro League': '🇧🇪',
        'Super Lig': '🇹🇷',
        'Premiership': '🏴󐁧󐁢󐁳󐁣󐁴󐁿'
    }
    return emojis.get(league_name, '⚽')

def calculate_bet_size_fixed_percentage(bankroll: float, percentage: float) -> float:
    """Calcula stake usando percentual fixo da banca"""
    return bankroll * (percentage / 100)

def calculate_bet_size_kelly_fractional(kelly_fraction: float, fraction: float = 0.25) -> float:
    """Calcula Kelly fracionário (mais conservador)"""
    return kelly_fraction * fraction

def calculate_break_even_wr(average_odds: float) -> float:
    """Calcula Win Rate necessário para break-even"""
    if average_odds <= 1.0:
        return 100.0
    return (1 / average_odds) * 100

def estimate_confidence_interval(win_rate: float, sample_size: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Calcula intervalo de confiança para win rate"""
    if sample_size == 0:
        return (0.0, 0.0)
    
    p = win_rate / 100
    z = 1.96 if confidence == 0.95 else 2.576  # 95% ou 99%
    
    se = np.sqrt((p * (1 - p)) / sample_size)
    margin = z * se
    
    lower = max(0, (p - margin) * 100)
    upper = min(100, (p + margin) * 100)
    
    return (lower, upper)

def generate_bankroll_projection(initial: float, roi_per_bet: float, n_bets: int) -> List[float]:
    """Projeta evolução da banca"""
    bankroll = [initial]
    
    for _ in range(n_bets):
        next_value = bankroll[-1] * (1 + roi_per_bet / 100)
        bankroll.append(next_value)
    
    return bankroll

def calculate_required_roi(initial: float, target: float, n_bets: int) -> float:
    """Calcula ROI necessário por aposta para atingir meta"""
    if n_bets == 0 or initial == 0:
        return 0.0
    
    multiplier = target / initial
    roi_per_bet = (multiplier ** (1 / n_bets) - 1) * 100
    
    return roi_per_bet

def calculate_risk_of_ruin(win_rate: float, avg_odds: float, bankroll_units: int = 100) -> float:
    """Calcula probabilidade de ruína"""
    if win_rate >= 100 or win_rate <= 0:
        return 0.0
    
    p = win_rate / 100
    q = 1 - p
    
    if avg_odds <= 1.0:
        return 100.0
    
    b = avg_odds - 1
    
    # Fórmula simplificada de Risk of Ruin
    if p * b > q:
        ror = ((q / (p * b)) ** bankroll_units) * 100
    else:
        ror = 100.0
    
    return min(100.0, ror)

def generate_monte_carlo_bankroll_simulation(initial: float, bets_per_day: int, days: int, 
                                             avg_stake_pct: float, win_rate: float, 
                                             avg_odds: float, n_simulations: int = 1000) -> Dict:
    """Simulação Monte Carlo de evolução da banca"""
    final_bankrolls = []
    
    for _ in range(n_simulations):
        bankroll = initial
        
        for _ in range(days * bets_per_day):
            stake = bankroll * (avg_stake_pct / 100)
            
            # Simular resultado
            if np.random.random() < (win_rate / 100):
                bankroll += stake * (avg_odds - 1)
            else:
                bankroll -= stake
            
            if bankroll <= 0:
                bankroll = 0
                break
        
        final_bankrolls.append(bankroll)
    
    final_bankrolls = np.array(final_bankrolls)
    
    return {
        'mean': np.mean(final_bankrolls),
        'median': np.median(final_bankrolls),
        'std': np.std(final_bankrolls),
        'min': np.min(final_bankrolls),
        'max': np.max(final_bankrolls),
        'p25': np.percentile(final_bankrolls, 25),
        'p75': np.percentile(final_bankrolls, 75),
        'prob_profit': (final_bankrolls > initial).mean() * 100,
        'prob_ruin': (final_bankrolls == 0).mean() * 100
    }

def analyze_betting_streak(results: List[bool]) -> Dict:
    """Analisa sequências de vitórias/derrotas"""
    if not results:
        return {'max_win_streak': 0, 'max_lose_streak': 0, 'current_streak': 0}
    
    max_win = 0
    max_lose = 0
    current = 0
    current_type = None
    
    for result in results:
        if result:  # Vitória
            if current_type == 'win':
                current += 1
            else:
                current = 1
                current_type = 'win'
            max_win = max(max_win, current)
        else:  # Derrota
            if current_type == 'lose':
                current += 1
            else:
                current = 1
                current_type = 'lose'
            max_lose = max(max_lose, current)
    
    return {
        'max_win_streak': max_win,
        'max_lose_streak': max_lose,
        'current_streak': current,
        'current_type': current_type
    }

def calculate_edge(true_prob: float, implied_prob: float) -> float:
    """Calcula edge (vantagem) da aposta"""
    return true_prob - implied_prob

def should_bet_based_on_kelly(kelly_fraction: float, min_kelly: float = 0.01) -> bool:
    """Determina se deve apostar baseado em Kelly"""
    return kelly_fraction >= min_kelly

def calculate_asian_handicap_probability(home_goals: float, away_goals: float, 
                                        handicap: float) -> float:
    """Calcula probabilidade de Asian Handicap"""
    # Simplificado - ajusta gols esperados
    adjusted_home = home_goals + handicap
    
    # Probabilidade de vitória ajustada
    if adjusted_home > away_goals:
        return 65.0 + (adjusted_home - away_goals) * 5
    elif adjusted_home < away_goals:
        return 35.0 - (away_goals - adjusted_home) * 5
    else:
        return 50.0

def format_asian_handicap(handicap: float) -> str:
    """Formata Asian Handicap"""
    if handicap > 0:
        return f"+{handicap:.2f}"
    return f"{handicap:.2f}"

def calculate_btts_probability(home_goals: float, away_goals: float) -> float:
    """Calcula probabilidade de Ambos Marcam (BTTS)"""
    prob_home_scores = 1 - calculate_poisson_probability(home_goals, 0)
    prob_away_scores = 1 - calculate_poisson_probability(away_goals, 0)
    
    return (prob_home_scores * prob_away_scores) * 100

def calculate_clean_sheet_probability(goals_conceded: float) -> float:
    """Calcula probabilidade de Clean Sheet"""
    return calculate_poisson_probability(goals_conceded, 0) * 100

def generate_league_comparison_table(stats_db: Dict) -> pd.DataFrame:
    """Gera tabela comparativa de ligas"""
    league_stats = defaultdict(lambda: {
        'cantos': [],
        'cartoes': [],
        'gols': [],
        'times': 0
    })
    
    for team, stats in STATS_db.items():
        league = STATS['league']
        league_STATS[league]['cantos'].append(STATS.get('corners', 5.5))
        league_STATS[league]['cartoes'].append(STATS.get('cards', 2.5))
        league_STATS[league]['gols'].append(STATS.get('goals_f', 1.5))
        league_STATS[league]['times'] += 1
    
    rows = []
    for league, data in league_stats.items():
        rows.append({
            'Liga': league,
            'Times': data['times'],
            'Cantos Médios': np.mean(data['cantos']),
            'Cartões Médios': np.mean(data['cartoes']),
            'Gols Médios': np.mean(data['gols'])
        })
    
    return pd.DataFrame(rows).sort_values('Cantos Médios', ascending=False)

def generate_betting_report(stats: Dict, bet_results: List[Dict]) -> str:
    """Gera relatório completo de apostas"""
    total = len(bet_results)
    
    if total == 0:
        return "Sem apostas para gerar relatório"
    
    won = sum(1 for b in bet_results if b.get('ganhou', False))
    wr = (won / total) * 100
    
    total_staked = sum(b.get('stake', 0) for b in bet_results)
    total_profit = sum(b.get('lucro', 0) for b in bet_results)
    roi = (total_profit / total_staked) * 100 if total_staked > 0 else 0
    
    report = f"""
📊 RELATÓRIO COMPLETO DE APOSTAS
{'=' * 50}

📈 ESTATÍSTICAS GERAIS:
• Total de Apostas: {total}
• Apostas Ganhas: {won}
• Apostas Perdidas: {total - won}
• Win Rate: {wr:.1f}%
• ROI: {roi:+.1f}%

💰 FINANCEIRO:
• Total Apostado: {format_currency(total_staked)}
• Lucro/Prejuízo: {format_currency(total_profit)}
• Stake Médio: {format_currency(total_staked / total)}

🎯 ANÁLISE:
{'✅ DESEMPENHO EXCELENTE!' if wr >= 65 and roi > 10 else '⚠️ Revisar estratégia'}

{'=' * 50}
"""
    
    return report

def export_data_to_csv(data: List[Dict], filename: str) -> str:
    """Exporta dados para CSV"""
    df = pd.DataFrame(data)
    filepath = f"/mnt/user-data/outputs/{filename}"
    df.to_csv(filepath, index=False)
    return filepath
