import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px

# ============================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================
st.set_page_config(
    page_title="FutPrevisão Pro",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CSS CUSTOMIZADO
# ============================================
st.markdown("""
<style>
    /* Fonte principal */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    /* Cards */
    .stat-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
        transition: transform 0.3s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.2rem;
    }
    
    .badge-success {
        background: #d4edda;
        color: #155724;
    }
    
    .badge-warning {
        background: #fff3cd;
        color: #856404;
    }
    
    .badge-danger {
        background: #f8d7da;
        color: #721c24;
    }
    
    .badge-info {
        background: #d1ecf1;
        color: #0c5460;
    }
    
    /* Validação visual */
    .validation-box {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    
    .validation-alta {
        border-left-color: #28a745;
        background: #d4edda;
    }
    
    .validation-media {
        border-left-color: #ffc107;
        background: #fff3cd;
    }
    
    .validation-baixa {
        border-left-color: #dc3545;
        background: #f8d7da;
    }
    
    /* Sequência de jogos */
    .sequence-container {
        display: flex;
        gap: 0.5rem;
        margin: 0.5rem 0;
        flex-wrap: wrap;
    }
    
    .sequence-item {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
        font-weight: bold;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    .sequence-win {
        background: #28a745;
        color: white;
    }
    
    .sequence-loss {
        background: #dc3545;
        color: white;
    }
    
    /* Alertas */
    .alert-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid;
    }
    
    .alert-danger {
        background: #f8d7da;
        border-left-color: #dc3545;
        color: #721c24;
    }
    
    .alert-warning {
        background: #fff3cd;
        border-left-color: #ffc107;
        color: #856404;
    }
    
    .alert-success {
        background: #d4edda;
        border-left-color: #28a745;
        color: #155724;
    }
    
    .alert-info {
        background: #d1ecf1;
        border-left-color: #17a2b8;
        color: #0c5460;
    }
    
    /* Time cards melhorados */
    .time-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        border-top: 4px solid #667eea;
    }
    
    .time-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #e9ecef;
    }
    
    /* Responsivo */
    @media (max-width: 768px) {
        .stat-card {
            padding: 1rem;
        }
        
        .sequence-item {
            width: 35px;
            height: 35px;
            font-size: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# CARREGADOR DE DADOS DO ADAM CHOI
# ============================================
class AdamChoiDataLoader:
    """
    Carrega e processa dados reais do Adam Choi
    """
    
    def __init__(self):
        self.dados = {}
        self.cache = {}
        self.ligas_disponiveis = {
            'Premier League': 'Escanteios_Preimier_League_-_codigo_fonte.txt',
            'La Liga': 'Escanteios_Espanha.txt',
            'Serie A': 'Escanteios_Italia.txt',
            'Bundesliga': 'Escanteios_Alemanha.txt',
            'Ligue 1': 'Escanteios_França.txt'
        }
        
    def carregar_todas_ligas(self) -> bool:
        """Carrega dados de todas as ligas disponíveis"""
        try:
            uploads_path = Path('/mnt/user-data/uploads')
            
            for liga, arquivo in self.ligas_disponiveis.items():
                caminho = uploads_path / arquivo
                
                if caminho.exists():
                    with open(caminho, 'r', encoding='utf-8') as f:
                        conteudo = f.read()
                        self.dados[liga] = json.loads(conteudo)
                else:
                    st.warning(f"⚠️ Arquivo não encontrado: {arquivo}")
                    
            return len(self.dados) > 0
            
        except Exception as e:
            st.error(f"❌ Erro ao carregar dados: {str(e)}")
            return False
    
    def buscar_time(self, liga: str, nome_time: str) -> Optional[Dict]:
        """Busca dados de um time específico"""
        if liga not in self.dados:
            return None
            
        # Cache key
        cache_key = f"{liga}_{nome_time.lower()}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Buscar time
        for time in self.dados[liga]['teams']:
            if time['teamName'].lower() == nome_time.lower():
                self.cache[cache_key] = time
                return time
                
        return None
    
    def listar_times(self, liga: str) -> List[str]:
        """Lista todos os times de uma liga"""
        if liga not in self.dados:
            return []
            
        return [time['teamName'] for time in self.dados[liga]['teams']]
    
    def extrair_estatisticas(self, time_data: Dict, local: str = 'overall') -> Dict:
        """
        Extrai estatísticas detalhadas de um time
        
        Args:
            time_data: Dados do time do Adam Choi
            local: 'overall', 'home' ou 'away'
            
        Returns:
            Dict com todas as estatísticas
        """
        prefix = 'overall' if local == 'overall' else ('home' if local == 'home' else 'away')
        
        stats = {
            'time': time_data['teamName'],
            'local': local,
            'jogos_totais': time_data.get(f'{prefix}TeamOver35', [0])[0],
            
            # Escanteios do time
            'time_35': self._extrair_linha(time_data, prefix, 'TeamOver35'),
            'time_45': self._extrair_linha(time_data, prefix, 'TeamOver45'),
            'time_55': self._extrair_linha(time_data, prefix, 'TeamOver55'),
            'time_65': self._extrair_linha(time_data, prefix, 'TeamOver65'),
            'time_75': self._extrair_linha(time_data, prefix, 'TeamOver75'),
            'time_85': self._extrair_linha(time_data, prefix, 'TeamOver85'),
            
            # Total do jogo
            'total_75': self._extrair_linha(time_data, prefix, 'Over75'),
            'total_85': self._extrair_linha(time_data, prefix, 'Over85'),
            'total_95': self._extrair_linha(time_data, prefix, 'Over95'),
            'total_105': self._extrair_linha(time_data, prefix, 'Over105'),
            'total_115': self._extrair_linha(time_data, prefix, 'Over115'),
            
            # Ambos fazem
            'ambos_25': self._extrair_linha(time_data, prefix, 'EachTeamOver25'),
            'ambos_35': self._extrair_linha(time_data, prefix, 'EachTeamOver35'),
            
            # Handicaps
            'handicap_-2': self._extrair_linha(time_data, prefix, 'Handicap_2'),
            'handicap_-1': self._extrair_linha(time_data, prefix, 'Handicap_1'),
            'handicap_0': self._extrair_linha(time_data, prefix, 'Handicap0'),
            'handicap_+1': self._extrair_linha(time_data, prefix, 'Handicap1'),
            'handicap_+2': self._extrair_linha(time_data, prefix, 'Handicap2'),
            
            # Últimos jogos
            'ultimos_jogos': self._extrair_ultimos_jogos(time_data, local, 5)
        }
        
        return stats
    
    def _extrair_linha(self, time_data: Dict, prefix: str, linha: str) -> Dict:
        """Extrai dados de uma linha específica"""
        key = f'{prefix}{linha}'
        if key in time_data:
            dados = time_data[key]
            return {
                'jogos': dados[0],
                'acertos': dados[1],
                'percentual': float(dados[2])
            }
        return {'jogos': 0, 'acertos': 0, 'percentual': 0.0}
    
    def _extrair_ultimos_jogos(self, time_data: Dict, local: str, n: int = 5) -> List[Dict]:
        """Extrai dados dos últimos N jogos"""
        all_matches = time_data.get('allMatches', [])
        
        if local == 'home':
            matches = [m for m in all_matches if m['homeTeam'] == time_data['teamName']]
        elif local == 'away':
            matches = [m for m in all_matches if m['awayTeam'] == time_data['teamName']]
        else:
            matches = all_matches
            
        # Últimos N jogos
        ultimos = matches[-n:] if len(matches) >= n else matches
        
        resultado = []
        for jogo in ultimos:
            if jogo['homeTeam'] == time_data['teamName']:
                escanteios_time = jogo.get('homeCorners', 0)
                escanteios_adversario = jogo.get('awayCorners', 0)
                adversario = jogo['awayTeam']
                mando = 'casa'
            else:
                escanteios_time = jogo.get('awayCorners', 0)
                escanteios_adversario = jogo.get('homeCorners', 0)
                adversario = jogo['homeTeam']
                mando = 'fora'
                
            resultado.append({
                'data': jogo['date'],
                'adversario': adversario,
                'mando': mando,
                'escanteios_time': escanteios_time,
                'escanteios_adversario': escanteios_adversario,
                'total': escanteios_time + escanteios_adversario
            })
            
        return resultado

# ============================================
# VALIDADOR HISTÓRICO
# ============================================
class ValidadorHistorico:
    """
    Valida previsões da IA com dados históricos reais
    """
    
    @staticmethod
    def validar_previsao(prob_ia: float, taxa_historica: float) -> Dict:
        """
        Valida previsão da IA contra dados históricos
        
        Returns:
            Dict com status, divergencia, recomendações
        """
        divergencia = prob_ia - taxa_historica
        divergencia_abs = abs(divergencia)
        
        # Determinar status
        if divergencia_abs <= 10:
            status = 'validado'
            confianca = 'ALTA'
            cor = 'success'
            icone = '✅'
            stake_rec = '5-7%'
        elif divergencia_abs <= 20:
            status = 'alerta'
            confianca = 'MÉDIA'
            cor = 'warning'
            icone = '⚠️'
            stake_rec = '2-4%'
        else:
            status = 'divergente'
            confianca = 'BAIXA'
            cor = 'danger'
            icone = '🚨'
            stake_rec = '1-2% ou EVITAR'
        
        # Mensagem
        if divergencia > 20:
            mensagem = f"IA muito OTIMISTA (+{divergencia:.1f}%)"
        elif divergencia < -20:
            mensagem = f"IA muito PESSIMISTA ({divergencia:.1f}%)"
        elif divergencia > 10:
            mensagem = f"IA levemente otimista (+{divergencia:.1f}%)"
        elif divergencia < -10:
            mensagem = f"IA levemente pessimista ({divergencia:.1f}%)"
        else:
            mensagem = "IA alinhada com histórico"
        
        return {
            'status': status,
            'confianca': confianca,
            'divergencia': divergencia,
            'divergencia_abs': divergencia_abs,
            'cor': cor,
            'icone': icone,
            'mensagem': mensagem,
            'stake_recomendado': stake_rec
        }
    
    @staticmethod
    def analisar_sequencia(ultimos_jogos: List[Dict], linha: float) -> Dict:
        """
        Analisa sequência dos últimos jogos
        
        Args:
            ultimos_jogos: Lista com últimos jogos
            linha: Linha de escanteios (ex: 4.5)
            
        Returns:
            Dict com análise da sequência
        """
        if not ultimos_jogos:
            return {
                'acertos': 0,
                'total': 0,
                'percentual': 0,
                'sequencia': [],
                'tendencia': 'Sem dados'
            }
        
        acertos = 0
        sequencia = []
        
        for jogo in ultimos_jogos:
            acertou = jogo['escanteios_time'] > linha
            if acertou:
                acertos += 1
                sequencia.append('✅')
            else:
                sequencia.append('❌')
        
        total = len(ultimos_jogos)
        percentual = (acertos / total * 100) if total > 0 else 0
        
        # Determinar tendência
        if acertos >= 4:
            tendencia = '🔥 SEQUÊNCIA QUENTE'
        elif acertos >= 3:
            tendencia = '✅ BOA FORMA'
        elif acertos == 2:
            tendencia = '⚠️ IRREGULAR'
        else:
            tendencia = '🥶 SEQUÊNCIA FRIA'
        
        return {
            'acertos': acertos,
            'total': total,
            'percentual': percentual,
            'sequencia': sequencia,
            'tendencia': tendencia
        }
    
    @staticmethod
    def gerar_alertas(stats: Dict, validacao: Dict, sequencia: Dict) -> List[str]:
        """Gera alertas automáticos baseados nos dados"""
        alertas = []
        
        # Alerta 1: Divergência muito grande
        if validacao['divergencia_abs'] > 30:
            alertas.append(f"🚨 ALERTA CRÍTICO: Divergência de {validacao['divergencia_abs']:.1f}% entre IA e histórico!")
        
        # Alerta 2: Sequência muito ruim
        if sequencia['acertos'] <= 1 and sequencia['total'] >= 4:
            alertas.append(f"⚠️ Time com sequência MUITO FRACA ({sequencia['acertos']}/{sequencia['total']} últimos jogos)")
        
        # Alerta 3: Sequência muito boa
        if sequencia['acertos'] >= 4 and sequencia['total'] >= 5:
            alertas.append(f"🔥 Time em FOGO! {sequencia['acertos']}/{sequencia['total']} últimos jogos")
        
        # Alerta 4: Poucos jogos
        if stats['jogos_totais'] < 5:
            alertas.append(f"⚠️ POUCOS DADOS: Apenas {stats['jogos_totais']} jogos na temporada")
        
        # Alerta 5: Taxa histórica muito baixa
        if validacao['divergencia'] > 20:  # IA otimista
            alertas.append("💡 DICA: Considere linha MENOR ou reduza stake")
        
        # Alerta 6: Taxa histórica muito alta mas IA pessimista
        if validacao['divergencia'] < -20:  # IA pessimista
            alertas.append("💡 OPORTUNIDADE: Histórico melhor que previsão da IA")
        
        return alertas

# ============================================
# SISTEMA DE AUTENTICAÇÃO
# ============================================
class AuthSystem:
    """Sistema simples de autenticação"""
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash simples de senha"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    @staticmethod
    def criar_usuario(username: str, password: str) -> bool:
        """Cria novo usuário"""
        if 'users' not in st.session_state:
            st.session_state.users = {}
            
        if username in st.session_state.users:
            return False
            
        st.session_state.users[username] = {
            'password': AuthSystem.hash_password(password),
            'created_at': datetime.now().isoformat(),
            'bilhetes': []
        }
        return True
    
    @staticmethod
    def login(username: str, password: str) -> bool:
        """Faz login do usuário"""
        if 'users' not in st.session_state:
            st.session_state.users = {}
            
        if username not in st.session_state.users:
            return False
            
        hashed = AuthSystem.hash_password(password)
        return st.session_state.users[username]['password'] == hashed
    
    @staticmethod
    def is_logged_in() -> bool:
        """Verifica se usuário está logado"""
        return 'logged_in' in st.session_state and st.session_state.logged_in

# ============================================
# GERADOR DE PREVISÕES (MANTIDO - NÃO MEXO)
# ============================================
class PrevisaoGenerator:
    """
    Gerador de previsões de escanteios
    NOTA: Mantido original - não mexo no algoritmo
    """
    
    @staticmethod
    def gerar_previsao_time(time: str, adversario: str, local: str = 'casa') -> Dict:
        """
        Gera previsão para um time específico
        ALGORITMO ORIGINAL MANTIDO
        """
        np.random.seed(hash(f"{time}{adversario}{local}") % 2**32)
        
        # Fatores base
        fator_casa = 1.2 if local == 'casa' else 0.9
        
        # Probabilidades base (mockadas - original)
        base_35 = np.random.uniform(0.55, 0.85) * fator_casa
        base_45 = base_35 * np.random.uniform(0.75, 0.90)
        base_55 = base_45 * np.random.uniform(0.70, 0.85)
        base_65 = base_55 * np.random.uniform(0.60, 0.80)
        
        return {
            'time': time,
            'adversario': adversario,
            'local': local,
            'linhas': {
                '3.5': min(max(base_35, 0.30), 0.95),
                '4.5': min(max(base_45, 0.25), 0.90),
                '5.5': min(max(base_55, 0.20), 0.85),
                '6.5': min(max(base_65, 0.15), 0.80)
            }
        }

# ============================================
# INICIALIZAÇÃO
# ============================================
def inicializar_app():
    """Inicializa estados e carrega dados"""
    
    # Session states
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    
    if 'username' not in st.session_state:
        st.session_state.username = None
        
    if 'bilhetes' not in st.session_state:
        st.session_state.bilhetes = []
        
    if 'data_loader' not in st.session_state:
        with st.spinner('📊 Carregando dados das 5 ligas europeias...'):
            loader = AdamChoiDataLoader()
            if loader.carregar_todas_ligas():
                st.session_state.data_loader = loader
                st.session_state.dados_carregados = True
            else:
                st.session_state.dados_carregados = False

# ============================================
# UI: LOGIN/REGISTRO
# ============================================
def pagina_login():
    """Página de login e registro"""
    
    st.markdown("""
    <div class="main-header">
        <h1>⚽ FutPrevisão Pro</h1>
        <p>Sistema Inteligente de Análise de Escanteios</p>
        <p style="font-size: 0.9em; opacity: 0.9;">📊 Dados Reais do Adam Choi | 5 Ligas Europeias</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["🔐 Login", "📝 Registrar"])
    
    with tab1:
        st.subheader("Entrar no Sistema")
        
        with st.form("login_form"):
            username = st.text_input("Usuário")
            password = st.text_input("Senha", type="password")
            submitted = st.form_submit_button("Entrar", use_container_width=True)
            
            if submitted:
                if AuthSystem.login(username, password):
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.success("✅ Login realizado!")
                    st.rerun()
                else:
                    st.error("❌ Usuário ou senha incorretos!")
    
    with tab2:
        st.subheader("Criar Nova Conta")
        
        with st.form("register_form"):
            new_username = st.text_input("Escolha um usuário")
            new_password = st.text_input("Escolha uma senha", type="password")
            confirm_password = st.text_input("Confirme a senha", type="password")
            submitted = st.form_submit_button("Registrar", use_container_width=True)
            
            if submitted:
                if len(new_username) < 3:
                    st.error("❌ Usuário deve ter pelo menos 3 caracteres!")
                elif len(new_password) < 4:
                    st.error("❌ Senha deve ter pelo menos 4 caracteres!")
                elif new_password != confirm_password:
                    st.error("❌ Senhas não coincidem!")
                elif AuthSystem.criar_usuario(new_username, new_password):
                    st.success("✅ Conta criada com sucesso! Faça login.")
                else:
                    st.error("❌ Usuário já existe!")

# ============================================
# UI: DASHBOARD PRINCIPAL
# ============================================
def pagina_dashboard():
    """Dashboard principal com dados reais"""
    
    st.markdown("""
    <div class="main-header">
        <h1>📊 Dashboard - FutPrevisão Pro</h1>
        <p>Bem-vindo, <strong>{}</strong>!</p>
    </div>
    """.format(st.session_state.username), unsafe_allow_html=True)
    
    # Verificar se dados foram carregados
    if not st.session_state.dados_carregados:
        st.error("❌ Erro ao carregar dados das ligas. Verifique os arquivos.")
        return
    
    loader = st.session_state.data_loader
    
    # Métricas gerais
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "🏆 Ligas Disponíveis",
            len(loader.dados),
            help="Ligas com dados carregados"
        )
    
    with col2:
        total_times = sum(len(loader.listar_times(liga)) for liga in loader.dados.keys())
        st.metric(
            "⚽ Times Cadastrados",
            total_times,
            help="Total de times em todas as ligas"
        )
    
    with col3:
        st.metric(
            "📋 Seus Bilhetes",
            len(st.session_state.bilhetes),
            help="Bilhetes criados por você"
        )
    
    with col4:
        st.metric(
            "✅ Validação Histórica",
            "ATIVA",
            help="Sistema comparando IA vs Dados Reais"
        )
    
    with col5:
        st.metric(
            "🔄 Última Atualização",
            "09/12/2025",
            help="Data dos dados do Adam Choi"
        )
    
    st.divider()
    
    # Ligas disponíveis
    st.subheader("🌍 Ligas Disponíveis")
    
    cols = st.columns(5)
    for idx, liga in enumerate(loader.dados.keys()):
        with cols[idx]:
            num_times = len(loader.listar_times(liga))
            st.markdown(f"""
            <div class="stat-card">
                <h3 style="margin:0; color:#667eea;">{liga}</h3>
                <p style="margin:0.5rem 0 0 0; color:#6c757d;">{num_times} times</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.divider()
    
    # Explorador rápido
    st.subheader("🔍 Explorador Rápido de Times")
    
    col1, col2 = st.columns(2)
    
    with col1:
        liga_selecionada = st.selectbox(
            "Selecione a Liga",
            list(loader.dados.keys())
        )
    
    with col2:
        times_disponiveis = loader.listar_times(liga_selecionada)
        time_selecionado = st.selectbox(
            "Selecione o Time",
            times_disponiveis
        )
    
    if st.button("📊 Ver Estatísticas Completas", use_container_width=True):
        mostrar_estatisticas_time(time_selecionado, liga_selecionada, loader)

# ============================================
# UI: MOSTRAR ESTATÍSTICAS DE TIME
# ============================================
def mostrar_estatisticas_time(time: str, liga: str, loader: AdamChoiDataLoader):
    """Mostra estatísticas completas de um time"""
    
    time_data = loader.buscar_time(liga, time)
    
    if not time_data:
        st.error(f"❌ Time {time} não encontrado na {liga}")
        return
    
    st.markdown(f"""
    <div class="time-card">
        <div class="time-header">
            <h2 style="margin:0;">⚽ {time}</h2>
            <span class="badge badge-info">{liga}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Tabs para casa/fora/geral
    tab1, tab2, tab3 = st.tabs(["🏠 Em Casa", "✈️ Fora", "📊 Geral"])
    
    with tab1:
        stats_casa = loader.extrair_estatisticas(time_data, 'home')
        mostrar_stats_detalhadas(stats_casa, "casa")
    
    with tab2:
        stats_fora = loader.extrair_estatisticas(time_data, 'away')
        mostrar_stats_detalhadas(stats_fora, "fora")
    
    with tab3:
        stats_geral = loader.extrair_estatisticas(time_data, 'overall')
        mostrar_stats_detalhadas(stats_geral, "geral")

def mostrar_stats_detalhadas(stats: Dict, tipo: str):
    """Mostra estatísticas detalhadas"""
    
    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total de Jogos",
            stats['jogos_totais']
        )
    
    with col2:
        st.metric(
            "Time +3.5",
            f"{stats['time_35']['percentual']:.1f}%",
            f"{stats['time_35']['acertos']}/{stats['time_35']['jogos']}"
        )
    
    with col3:
        st.metric(
            "Time +4.5",
            f"{stats['time_45']['percentual']:.1f}%",
            f"{stats['time_45']['acertos']}/{stats['time_45']['jogos']}"
        )
    
    with col4:
        st.metric(
            "Total +9.5",
            f"{stats['total_95']['percentual']:.1f}%",
            f"{stats['total_95']['acertos']}/{stats['total_95']['jogos']}"
        )
    
    st.divider()
    
    # Tabela com todas as linhas
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🚩 Escanteios do Time")
        
        linhas_time = []
        for linha in ['35', '45', '55', '65', '75', '85']:
            key = f'time_{linha}'
            if key in stats:
                linha_float = float(linha) / 10
                linhas_time.append({
                    'Linha': f'+{linha_float}',
                    'Acertos': f"{stats[key]['acertos']}/{stats[key]['jogos']}",
                    'Taxa': f"{stats[key]['percentual']:.1f}%",
                    'Status': get_status_emoji(stats[key]['percentual'])
                })
        
        df_time = pd.DataFrame(linhas_time)
        st.dataframe(df_time, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("### 🎯 Total do Jogo")
        
        linhas_total = []
        for linha in ['75', '85', '95', '105', '115']:
            key = f'total_{linha}'
            if key in stats:
                linha_float = float(linha) / 10
                linhas_total.append({
                    'Linha': f'+{linha_float}',
                    'Acertos': f"{stats[key]['acertos']}/{stats[key]['jogos']}",
                    'Taxa': f"{stats[key]['percentual']:.1f}%",
                    'Status': get_status_emoji(stats[key]['percentual'])
                })
        
        df_total = pd.DataFrame(linhas_total)
        st.dataframe(df_total, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # Últimos jogos
    st.markdown("### 📅 Últimos 5 Jogos")
    
    if stats['ultimos_jogos']:
        for jogo in reversed(stats['ultimos_jogos']):
            col1, col2, col3, col4 = st.columns([2, 3, 2, 2])
            
            with col1:
                st.write(f"📅 {jogo['data']}")
            
            with col2:
                mando_icon = "🏠" if jogo['mando'] == 'casa' else "✈️"
                st.write(f"{mando_icon} vs {jogo['adversario']}")
            
            with col3:
                st.write(f"🚩 {jogo['escanteios_time']} | 🆚 {jogo['escanteios_adversario']}")
            
            with col4:
                st.write(f"📊 Total: {jogo['total']}")
    else:
        st.info("Sem dados de jogos anteriores")

def get_status_emoji(percentual: float) -> str:
    """Retorna emoji baseado no percentual"""
    if percentual >= 70:
        return "🔥"
    elif percentual >= 55:
        return "✅"
    elif percentual >= 40:
        return "⚠️"
    else:
        return "❌"

# ============================================
# UI: FAZER PREVISÃO
# ============================================
def pagina_previsao():
    """Página de fazer previsão com validação histórica"""
    
    st.markdown("""
    <div class="main-header">
        <h1>🎯 Fazer Previsão</h1>
        <p>Sistema com Validação Histórica</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.dados_carregados:
        st.error("❌ Dados não carregados. Volte ao dashboard.")
        return
    
    loader = st.session_state.data_loader
    
    # Formulário de seleção
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏠 Time da Casa")
        
        liga_casa = st.selectbox(
            "Liga",
            list(loader.dados.keys()),
            key="liga_casa"
        )
        
        times_casa = loader.listar_times(liga_casa)
        time_casa = st.selectbox(
            "Time",
            times_casa,
            key="time_casa"
        )
    
    with col2:
        st.subheader("✈️ Time Visitante")
        
        liga_fora = st.selectbox(
            "Liga",
            list(loader.dados.keys()),
            key="liga_fora"
        )
        
        times_fora = loader.listar_times(liga_fora)
        time_fora = st.selectbox(
            "Time",
            times_fora,
            key="time_fora"
        )
    
    st.divider()
    
    # Linha de escanteios
    linha = st.selectbox(
        "📊 Linha de Escanteios",
        ['3.5', '4.5', '5.5', '6.5'],
        help="Selecione a linha para análise"
    )
    
    if st.button("🚀 Gerar Análise Completa", use_container_width=True, type="primary"):
        gerar_analise_completa(time_casa, liga_casa, time_fora, liga_fora, float(linha), loader)

# ============================================
# GERAR ANÁLISE COMPLETA
# ============================================
def gerar_analise_completa(time_casa: str, liga_casa: str, 
                          time_fora: str, liga_fora: str,
                          linha: float, loader: AdamChoiDataLoader):
    """Gera análise completa com validação histórica"""
    
    st.markdown("---")
    st.markdown(f"""
    <div class="jogo-titulo" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
         color: white; padding: 1.5rem; border-radius: 10px; text-align: center; margin: 1rem 0;">
        <h2 style="margin:0;">🏟️ {time_casa} vs {time_fora}</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Buscar dados reais
    data_casa = loader.buscar_time(liga_casa, time_casa)
    data_fora = loader.buscar_time(liga_fora, time_fora)
    
    if not data_casa or not data_fora:
        st.error("❌ Erro ao buscar dados dos times")
        return
    
    # Extrair estatísticas
    stats_casa = loader.extrair_estatisticas(data_casa, 'home')
    stats_fora = loader.extrair_estatisticas(data_fora, 'away')
    
    # Gerar previsões da IA (algoritmo original)
    prev_casa = PrevisaoGenerator.gerar_previsao_time(time_casa, time_fora, 'casa')
    prev_fora = PrevisaoGenerator.gerar_previsao_time(time_fora, time_casa, 'fora')
    
    # Pegar probabilidades da IA
    linha_str = str(linha)
    prob_ia_casa = prev_casa['linhas'][linha_str] * 100
    prob_ia_fora = prev_fora['linhas'][linha_str] * 100
    
    # Pegar taxas históricas
    linha_key = str(int(linha * 10))  # 4.5 -> '45'
    taxa_hist_casa = stats_casa[f'time_{linha_key}']['percentual']
    taxa_hist_fora = stats_fora[f'time_{linha_key}']['percentual']
    
    # Validar
    validacao_casa = ValidadorHistorico.validar_previsao(prob_ia_casa, taxa_hist_casa)
    validacao_fora = ValidadorHistorico.validar_previsao(prob_ia_fora, taxa_hist_fora)
    
    # Analisar sequências
    seq_casa = ValidadorHistorico.analisar_sequencia(stats_casa['ultimos_jogos'], linha)
    seq_fora = ValidadorHistorico.analisar_sequencia(stats_fora['ultimos_jogos'], linha)
    
    # Gerar alertas
    alertas_casa = ValidadorHistorico.gerar_alertas(stats_casa, validacao_casa, seq_casa)
    alertas_fora = ValidadorHistorico.gerar_alertas(stats_fora, validacao_fora, seq_fora)
    
    # Mostrar análise
    col1, col2 = st.columns(2)
    
    with col1:
        mostrar_card_analise(time_casa, "🏠 CASA", stats_casa, validacao_casa, 
                            seq_casa, alertas_casa, prob_ia_casa, taxa_hist_casa, linha)
    
    with col2:
        mostrar_card_analise(time_fora, "✈️ FORA", stats_fora, validacao_fora,
                            seq_fora, alertas_fora, prob_ia_fora, taxa_hist_fora, linha)
    
    st.divider()
    
    # Recomendação final
    st.markdown("### 💡 Recomendação Final")
    
    gerar_recomendacao_final(validacao_casa, validacao_fora, seq_casa, seq_fora,
                             time_casa, time_fora, linha)
    
    # Botão para adicionar ao bilhete
    st.divider()
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("➕ Adicionar ao Bilhete", use_container_width=True, type="primary"):
            adicionar_ao_bilhete(time_casa, time_fora, linha, validacao_casa, validacao_fora)

def mostrar_card_analise(time: str, tipo: str, stats: Dict, validacao: Dict,
                        sequencia: Dict, alertas: List[str], prob_ia: float,
                        taxa_hist: float, linha: float):
    """Mostra card de análise de um time"""
    
    st.markdown(f"""
    <div class="time-card">
        <div class="time-header">
            <h3 style="margin:0;">{tipo} {time}</h3>
            <span class="badge badge-{validacao['cor']}">{validacao['icone']} {validacao['confianca']}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Comparação IA vs Histórico
    st.markdown(f"""
    <div class="validation-box validation-{validacao['status']}">
        <h4 style="margin:0 0 0.5rem 0;">📊 Linha +{linha} Escanteios</h4>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin: 1rem 0;">
            <div>
                <p style="margin:0; color: #6c757d; font-size: 0.9em;">🤖 IA Prevê:</p>
                <p style="margin:0; font-size: 1.5em; font-weight: bold;">{prob_ia:.1f}%</p>
            </div>
            <div>
                <p style="margin:0; color: #6c757d; font-size: 0.9em;">📊 Histórico Real:</p>
                <p style="margin:0; font-size: 1.5em; font-weight: bold;">{taxa_hist:.1f}%</p>
                <p style="margin:0; font-size: 0.85em; color: #6c757d;">
                    {stats[f'time_{int(linha*10)}']['acertos']}/{stats[f'time_{int(linha*10)}']['jogos']} jogos
                </p>
            </div>
        </div>
        
        <p style="margin: 1rem 0 0 0; font-weight: 600; color: #495057;">
            {validacao['icone']} {validacao['mensagem']}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sequência visual
    st.markdown("### 📈 Últimos 5 Jogos")
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        st.markdown(f"""
        <div style="display: flex; gap: 0.5rem; margin: 0.5rem 0;">
            {"".join(f'<div class="sequence-item sequence-{"win" if s == "✅" else "loss"}">{s}</div>' 
                     for s in sequencia['sequencia'])}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.metric(
            "Taxa de Acerto",
            f"{sequencia['percentual']:.0f}%",
            f"{sequencia['acertos']}/{sequencia['total']}"
        )
    
    with col3:
        st.markdown(f"""
        <div style="text-align: center; padding: 0.5rem;">
            <div style="font-size: 0.85em; color: #6c757d;">Tendência</div>
            <div style="font-size: 1.1em; font-weight: bold; margin-top: 0.3rem;">
                {sequencia['tendencia']}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Stake recomendado
    st.markdown("### 💰 Gestão de Banca")
    
    st.markdown(f"""
    <div class="alert-box alert-{validacao['cor']}">
        <strong>Stake Recomendado:</strong> {validacao['stake_recomendado']} da banca
    </div>
    """, unsafe_allow_html=True)
    
    # Alertas
    if alertas:
        st.markdown("### ⚠️ Alertas")
        
        for alerta in alertas:
            tipo_alerta = 'danger' if '🚨' in alerta else ('warning' if '⚠️' in alerta else 'info')
            st.markdown(f"""
            <div class="alert-box alert-{tipo_alerta}">
                {alerta}
            </div>
            """, unsafe_allow_html=True)

def gerar_recomendacao_final(val_casa: Dict, val_fora: Dict, 
                             seq_casa: Dict, seq_fora: Dict,
                             time_casa: str, time_fora: str, linha: float):
    """Gera recomendação final do confronto"""
    
    # Análise combinada
    confianca_media = (
        (70 if val_casa['status'] == 'validado' else (50 if val_casa['status'] == 'alerta' else 30)) +
        (70 if val_fora['status'] == 'validado' else (50 if val_fora['status'] == 'alerta' else 30))
    ) / 2
    
    # Determinar melhor aposta
    if val_casa['status'] == 'validado' and seq_casa['acertos'] >= 3:
        melhor_aposta = f"✅ {time_casa} +{linha} (Casa validada + sequência boa)"
        cor = 'success'
    elif val_fora['status'] == 'validado' and seq_fora['acertos'] >= 3:
        melhor_aposta = f"✅ {time_fora} +{linha} (Fora validado + sequência boa)"
        cor = 'success'
    elif val_casa['status'] == 'alerta' or val_fora['status'] == 'alerta':
        melhor_aposta = f"⚠️ Apostar com CAUTELA - Divergências detectadas"
        cor = 'warning'
    else:
        melhor_aposta = f"🚫 EVITAR - Muitas divergências ou dados insuficientes"
        cor = 'danger'
    
    st.markdown(f"""
    <div class="alert-box alert-{cor}">
        <h4 style="margin: 0 0 1rem 0;">🎯 Melhor Opção:</h4>
        <p style="margin: 0; font-size: 1.1em; font-weight: 600;">
            {melhor_aposta}
        </p>
        <p style="margin: 1rem 0 0 0; font-size: 0.9em;">
            Confiança Geral: {confianca_media:.0f}%
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Insights adicionais
    st.markdown("### 🧠 Insights Adicionais")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <h4>{time_casa} (Casa)</h4>
            <ul style="margin: 0.5rem 0; padding-left: 1.5rem;">
                <li>Validação: {val_casa['icone']} {val_casa['confianca']}</li>
                <li>Sequência: {seq_casa['acertos']}/5 últimos</li>
                <li>Tendência: {seq_casa['tendencia']}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <h4>{time_fora} (Fora)</h4>
            <ul style="margin: 0.5rem 0; padding-left: 1.5rem;">
                <li>Validação: {val_fora['icone']} {val_fora['confianca']}</li>
                <li>Sequência: {seq_fora['acertos']}/5 últimos</li>
                <li>Tendência: {seq_fora['tendencia']}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

def adicionar_ao_bilhete(time_casa: str, time_fora: str, linha: float,
                        val_casa: Dict, val_fora: Dict):
    """Adiciona análise ao bilhete"""
    
    bilhete = {
        'time_casa': time_casa,
        'time_fora': time_fora,
        'linha': linha,
        'validacao_casa': val_casa,
        'validacao_fora': val_fora,
        'data': datetime.now().isoformat()
    }
    
    st.session_state.bilhetes.append(bilhete)
    st.success(f"✅ Jogo adicionado ao bilhete! Total: {len(st.session_state.bilhetes)} jogos")

# ============================================
# UI: MEUS BILHETES
# ============================================
def pagina_bilhetes():
    """Página de gerenciar bilhetes"""
    
    st.markdown("""
    <div class="main-header">
        <h1>📋 Meus Bilhetes</h1>
        <p>Gerencie suas análises salvas</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.bilhetes:
        st.info("📭 Você ainda não tem bilhetes salvos. Faça uma previsão primeiro!")
        return
    
    st.markdown(f"### Total: {len(st.session_state.bilhetes)} jogos no bilhete")
    
    # Mostrar cada jogo
    for idx, bilhete in enumerate(st.session_state.bilhetes):
        with st.expander(f"🎯 Jogo {idx + 1}: {bilhete['time_casa']} vs {bilhete['time_fora']} (+{bilhete['linha']})"):
            
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                st.markdown(f"""
                **{bilhete['time_casa']} (Casa)**
                - Validação: {bilhete['validacao_casa']['icone']} {bilhete['validacao_casa']['confianca']}
                - Stake: {bilhete['validacao_casa']['stake_recomendado']}
                """)
            
            with col2:
                st.markdown(f"""
                **{bilhete['time_fora']} (Fora)**
                - Validação: {bilhete['validacao_fora']['icone']} {bilhete['validacao_fora']['confianca']}
                - Stake: {bilhete['validacao_fora']['stake_recomendado']}
                """)
            
            with col3:
                if st.button("🗑️ Remover", key=f"rem_{idx}"):
                    st.session_state.bilhetes.pop(idx)
                    st.rerun()
    
    st.divider()
    
    # Ações do bilhete
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Limpar Todos", use_container_width=True):
            st.session_state.bilhetes = []
            st.success("✅ Bilhete limpo!")
            st.rerun()
    
    with col2:
        if st.button("💾 Salvar Bilhete", use_container_width=True):
            st.success("✅ Bilhete salvo! (funcionalidade em desenvolvimento)")

# ============================================
# MAIN
# ============================================
def main():
    """Função principal"""
    
    # Inicializar
    inicializar_app()
    
    # Verificar login
    if not AuthSystem.is_logged_in():
        pagina_login()
        return
    
    # Sidebar
    with st.sidebar:
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
             border-radius: 10px; color: white; margin-bottom: 1rem;">
            <h3 style="margin:0;">👤 {st.session_state.username}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📍 Navegação")
        
        pagina = st.radio(
            "Ir para:",
            ["📊 Dashboard", "🎯 Fazer Previsão", "📋 Meus Bilhetes"],
            label_visibility="collapsed"
        )
        
        st.divider()
        
        st.markdown("### 🌍 Ligas Disponíveis")
        
        if st.session_state.dados_carregados:
            loader = st.session_state.data_loader
            for liga in loader.dados.keys():
                st.markdown(f"✅ {liga}")
        else:
            st.error("❌ Dados não carregados")
        
        st.divider()
        
        if st.button("🚪 Sair", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.rerun()
    
    # Renderizar página
    if pagina == "📊 Dashboard":
        pagina_dashboard()
    elif pagina == "🎯 Fazer Previsão":
        pagina_previsao()
    elif pagina == "📋 Meus Bilhetes":
        pagina_bilhetes()

# ============================================
# EXECUTAR
# ============================================
if __name__ == "__main__":
    main()
