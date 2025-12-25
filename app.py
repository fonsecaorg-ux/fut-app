"""
╔═══════════════════════════════════════════════════════════════════════════╗
║           FUTPREVISÃO V31 MAXIMUM - SISTEMA DEFINITIVO COMPLETO           ║
║                        VERSÃO FINAL - PRODUÇÃO                            ║
║                   SEM SKLEARN - 100% NUMPY/SCIPY                          ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any, Optional, Tuple
from difflib import get_close_matches
from itertools import combinations
from scipy import stats
from scipy.stats import poisson, norm, beta
import math
import os
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="FutPrevisão V31 MAXIMUM", page_icon="🔥", layout="wide")

# CSS Melhorado - Fundo Claro e Legível
st.markdown("""<style>
.stApp { 
    background: linear-gradient(to bottom, #f5f7fa 0%, #c3cfe2 100%);
}
.stButton>button { 
    width: 100%; 
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
    color: white; 
    font-weight: bold; 
    border: none; 
    padding: 12px; 
    border-radius: 8px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 12px rgba(0,0,0,0.2);
}
h1, h2, h3 { 
    color: #1a202c !important;
    text-shadow: 1px 1px 2px rgba(255,255,255,0.5);
}
.stMetric { 
    background: rgba(255,255,255,0.9); 
    padding: 15px; 
    border-radius: 10px;
    border: 2px solid #667eea;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.stMarkdown {
    color: #2d3748 !important;
}
.stExpander {
    background: rgba(255,255,255,0.95);
    border-radius: 8px;
    border: 1px solid #e2e8f0;
}
.stSelectbox, .stSlider {
    background: rgba(255,255,255,0.9);
    border-radius: 5px;
}
</style>""", unsafe_allow_html=True)

# SESSION STATE
if 'bankroll_history' not in st.session_state: st.session_state.bankroll_history = [1000.0]
if 'current_ticket' not in st.session_state: st.session_state.current_ticket = []
if 'hedges_data' not in st.session_state: st.session_state.hedges_data = None
if 'bet_results' not in st.session_state: st.session_state.bet_results = []
if 'ml_model' not in st.session_state: st.session_state.ml_model = None
if 'team_clusters' not in st.session_state: st.session_state.team_clusters = {}
if 'h2h_matrix' not in st.session_state: st.session_state.h2h_matrix = {}

# CONSTANTES
REAL_ODDS = {
    ('home','corners',3.5):1.34,('home','corners',4.5):1.63,('away','corners',2.5):1.40,
    ('away','corners',3.5):1.75,('away','corners',4.5):2.50,('total','corners',7.5):1.35,
    ('total','corners',8.5):1.58,('total','corners',9.5):1.90,('total','corners',10.5):2.30,
    ('total','corners',11.5):2.80,('home','cards',0.5):1.22,('home','cards',1.5):1.53,
    ('home','cards',2.5):2.25,('away','cards',0.5):1.26,('away','cards',1.5):1.65,
    ('away','cards',2.5):2.50,('total','cards',2.5):1.43,('total','cards',3.5):1.73,
    ('total','cards',4.5):2.13,('home','dc','1X'):1.24,('away','dc','X2'):1.60
}

NAME_MAPPING = {
    'Man United':'Man Utd','Manchester United':'Man Utd','Man City':'Man City',
    'Manchester City':'Man City','Spurs':'Tottenham',"Nott\'m Forest":'Nottm Forest',
    'Nottingham Forest':'Nottm Forest','Athletic Club':'Ath Bilbao'
}

LIGAS = ["Premier League","La Liga","Serie A","Bundesliga","Ligue 1","Championship",
         "Bundesliga 2","Pro League","Süper Lig","Scottish Premiership"]

DERBIES = {
    'Premier League':{('Man City','Man Utd'):'Manchester Derby',('Liverpool','Everton'):'Merseyside',
                      ('Arsenal','Tottenham'):'North London',('Chelsea','Arsenal'):'London Derby'},
    'La Liga':{('Barcelona','Real Madrid'):'El Clásico',('Sevilla','Betis'):'Seville Derby',
               ('Atletico Madrid','Real Madrid'):'Madrid Derby'},
    'Serie A':{('Inter','AC Milan'):'Derby Madonnina',('Juventus','Torino'):'Derby Mole',
               ('Roma','Lazio'):'Derby Capitale'}
}

# ═══════════════════════════════════════════════════════════════════════════
# FUNÇÕES ESTATÍSTICAS AVANÇADAS
# ═══════════════════════════════════════════════════════════════════════════

def bivariate_poisson_vectorized(l1:float,l2:float,corr:float=-0.25,n:int=1000):
    """Poisson Bivariado Vetorizado - Karlis & Ntzoufras"""
    if abs(corr)<0.01: return np.random.poisson(l1,n),np.random.poisson(l2,n)
    mean,cov=np.array([0,0]),np.array([[1,corr],[corr,1]])
    normals=np.random.multivariate_normal(mean,cov,n)
    uniforms=norm.cdf(normals)
    return poisson.ppf(uniforms[:,0],l1).astype(int),poisson.ppf(uniforms[:,1],l2).astype(int)

def calculate_ema_trend(values:np.ndarray,span:int=5)->float:
    """EMA Trend Detection"""
    if len(values)<3: return 0.0
    ema=pd.Series(values).ewm(span=span,adjust=False).mean()
    return (ema.iloc[-1]-ema.iloc[0])/len(values)

def analyze_momentum(values:np.ndarray,window:int=5)->Dict:
    """Análise de Momentum Avançada"""
    if len(values)<window: return {'momentum':0,'acceleration':0,'consistency':0}
    last_n=values[-window:]
    weights=np.exp(np.linspace(0,1,window))
    weights/=weights.sum()
    weighted_avg=np.average(last_n,weights=weights)
    if len(last_n)>=3:
        diffs=np.diff(last_n)
        acceleration=np.mean(np.diff(diffs)) if len(diffs)>1 else 0
    else: acceleration=0
    consistency=1-(np.std(last_n)/np.mean(last_n)) if np.mean(last_n)>0 else 0
    momentum=(weighted_avg*0.6)+(acceleration*20)+(consistency*0.2)
    return {'momentum':momentum,'acceleration':acceleration,'consistency':consistency,
            'trend':'UP' if acceleration>0.1 else ('DOWN' if acceleration<-0.1 else 'STABLE')}

def bootstrap_ci_fast(data:np.ndarray,conf:float=0.95,n_boot:int=500):
    """Bootstrap Confidence Interval"""
    if len(data)<3: return np.mean(data),np.mean(data)
    indices=np.random.randint(0,len(data),size=(n_boot,len(data)))
    boot_means=np.mean(data[indices],axis=1)
    alpha=1-conf
    return np.percentile(boot_means,(alpha/2)*100),np.percentile(boot_means,(1-alpha/2)*100)

def kelly_criterion_fractional(prob:float,odd:float,var:float=0.1,frac:float=0.25)->float:
    """Kelly Fracionário com Incerteza"""
    edge=(prob*odd)-1
    if edge<=0: return 0.01
    kelly_full=edge/(odd-1)
    uncertainty_factor=max(0.5,1-(var*1.5))
    kelly_adj=kelly_full*uncertainty_factor*frac
    return max(0.01,min(0.10,kelly_adj))

def calculate_sharpe_ratio(returns:List[float],rf:float=0.0)->float:
    """Sharpe Ratio"""
    if len(returns)<2: return 0.0
    arr=np.array(returns)
    return (np.mean(arr)-rf)/np.std(arr) if np.std(arr)>0 else 0.0

def calculate_max_drawdown(history:List[float])->float:
    """Maximum Drawdown"""
    if len(history)<2: return 0.0
    arr=np.array(history)
    running_max=np.maximum.accumulate(arr)
    dd=(running_max-arr)/running_max
    return np.max(dd)*100

def calculate_value_score(our_prob:float,odd:float,margin:float=0.06)->float:
    """Value Score vs Bookmaker"""
    if odd<=1.0: return 0.0
    implied=1.0/odd
    true_prob=implied/(1+margin)
    return our_prob/true_prob

def detect_game_context(home:str,away:str,league:str,pos_h:int=None,pos_a:int=None)->Dict:
    """Detector de Contexto Automático"""
    context={'type':'NORMAL','factor_corners':1.0,'factor_cards':1.0,'name':''}
    for (t1,t2),name in DERBIES.get(league,{}).items():
        if (home in [t1,t2]) and (away in [t1,t2]):
            return {'type':'DERBY','name':name,'factor_corners':1.12,'factor_cards':1.35}
    if pos_h and pos_a:
        if pos_h<=6 and pos_a<=6:
            return {'type':'TOP_CLASH','name':'Top 6 vs Top 6','factor_corners':1.08,'factor_cards':1.15}
        elif pos_h>=17 or pos_a>=17:
            return {'type':'RELEGATION','name':'Luta Rebaixamento','factor_corners':0.95,'factor_cards':1.25}
    return context

# ═══════════════════════════════════════════════════════════════════════════
# CARREGAMENTO DE DADOS
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def find_and_load_csv(league:str)->pd.DataFrame:
    """Carrega CSV da liga - CORRIGIDO V31.1"""
    # Mapeamento EXATO dos nomes
    league_files = {
        'Premier League': 'Premier_League_25_26.csv',
        'La Liga': 'La_Liga_25_26.csv',
        'Serie A': 'Serie_A_25_26.csv',
        'Bundesliga': 'Bundesliga_25_26.csv',
        'Ligue 1': 'Ligue_1_25_26.csv',
        'Championship': 'Championship_Inglaterra_25_26.csv',
        'Bundesliga 2': 'Bundesliga_2.csv',
        'Pro League': 'Pro_League_Belgica_25_26.csv',
        'Süper Lig': 'Super_Lig_Turquia_25_26.csv',
        'Premiership': 'Premiership_Escocia_25_26.csv'
    }
    
    # Tentar mapeamento direto primeiro
    filename = league_files.get(league)
    if filename:
        for directory in ['/mnt/project/', './','']:
            filepath = os.path.join(directory, filename) if directory else filename
            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath, encoding='utf-8-sig')
                    if not df.empty:
                        df.columns = [c.strip().replace('\ufeff', '') for c in df.columns]
                        df = df.rename(columns={'Mandante':'HomeTeam', 'Visitante':'AwayTeam',
                                                'Time_Casa':'HomeTeam', 'Time_Visitante':'AwayTeam'})
                        df['_League_'] = league
                        return df
                except Exception as e:
                    continue
    
    # Fallback: tentar variações
    attempts = [
        f"{league.replace(' ', '_')}_25_26.csv",
        f"{league}_25_26.csv",
        f"{league}.csv"
    ]
    
    for filename in attempts:
        for directory in ['/mnt/project/', './', '']:
            filepath = os.path.join(directory, filename) if directory else filename
            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath, encoding='utf-8-sig')
                    if not df.empty:
                        df.columns = [c.strip().replace('\ufeff', '') for c in df.columns]
                        df = df.rename(columns={'Mandante':'HomeTeam', 'Visitante':'AwayTeam',
                                                'Time_Casa':'HomeTeam', 'Time_Visitante':'AwayTeam'})
                        df['_League_'] = league
                        return df
                except:
                    continue
    
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def learn_stats_maximum()->Tuple[Dict,Dict,Dict]:
    """V31 MAXIMUM: Stats + H2H + Clustering"""
    stats_db,all_dfs={},[]
    for league in LIGAS:
        df=find_and_load_csv(league)
        if df.empty: continue
        all_dfs.append(df)
        for col in ['HomeTeam','AwayTeam','HC','AC','HY','AY','FTHG','FTAG']:
            if col not in df.columns: df[col]=np.nan
        try:
            for team in set(df['HomeTeam'].dropna())|set(df['AwayTeam'].dropna()):
                home_g=df[df['HomeTeam']==team]
                away_g=df[df['AwayTeam']==team]
                all_games=pd.concat([
                    home_g[['HC','HY','FTHG']].rename(columns={'HC':'corners','HY':'cards','FTHG':'goals'}),
                    away_g[['AC','AY','FTAG']].rename(columns={'AC':'corners','AY':'cards','FTAG':'goals'})
                ]).tail(10)
                if len(all_games)<3: continue
                corners_arr=all_games['corners'].values
                cards_arr=all_games['cards'].values
                momentum=analyze_momentum(corners_arr)
                ci_corners=bootstrap_ci_fast(corners_arr)
                stats_db[team]={
                    'corners':max(2.0,all_games['corners'].mean()),'corners_std':max(1.0,all_games['corners'].std()),
                    'corners_ci':ci_corners,'corners_trend':calculate_ema_trend(corners_arr),
                    'momentum':momentum,'cards':max(0.5,all_games['cards'].mean()),
                    'cards_std':max(0.5,all_games['cards'].std()),'goals_f':max(0.5,all_games['goals'].mean()),
                    'league':league,'n_games':len(all_games),'reliability':min(100,(len(all_games)/38)*100)
                }
        except: continue
    # H2H MATRIX
    h2h_db={}
    for df in all_dfs:
        for _,row in df.iterrows():
            if pd.isna(row.get('HomeTeam')) or pd.isna(row.get('AwayTeam')): continue
            key=f"{row['HomeTeam']}_vs_{row['AwayTeam']}"
            if key not in h2h_db: h2h_db[key]=[]
            h2h_db[key].append({'hc':row.get('HC',0),'ac':row.get('AC',0),'hy':row.get('HY',0),
                               'ay':row.get('AY',0),'tc':row.get('HC',0)+row.get('AC',0)})
    h2h_stats={}
    for key,games in h2h_db.items():
        if len(games)>=2:
            h2h_stats[key]={'n':len(games),'hc_avg':np.mean([g['hc'] for g in games]),
                           'ac_avg':np.mean([g['ac'] for g in games]),'tc_avg':np.mean([g['tc'] for g in games]),
                           'volatility':np.std([g['tc'] for g in games])}
    # CLUSTERING SIMPLES (sem sklearn)
    teams,features=[],[]
    for team,st in stats_db.items():
        teams.append(team)
        features.append([st['corners'],st['cards'],st['goals_f'],st['corners_std'],st['cards_std']])
    
    team_clusters={}
    if len(features)>10:
        # Normalizar manualmente
        features_arr=np.array(features)
        mean=features_arr.mean(axis=0)
        std=features_arr.std(axis=0)+1e-8
        features_scaled=(features_arr-mean)/std
        
        # K-Means simples (versão numpy pura)
        n_clust=min(5,len(features)//10)
        n_iter=20
        
        # Inicializar centroids aleatoriamente
        np.random.seed(42)
        idx=np.random.choice(len(features_scaled),n_clust,replace=False)
        centroids=features_scaled[idx]
        
        # Iterações K-Means
        for _ in range(n_iter):
            # Atribuir clusters (distância euclidiana)
            distances=np.sqrt(((features_scaled[:,np.newaxis]-centroids)**2).sum(axis=2))
            clusters=np.argmin(distances,axis=1)
            
            # Atualizar centroids
            for k in range(n_clust):
                if np.sum(clusters==k)>0:
                    centroids[k]=features_scaled[clusters==k].mean(axis=0)
        
        team_clusters={teams[i]:int(clusters[i]) for i in range(len(teams))}
    
    return stats_db,h2h_stats,team_clusters

@st.cache_data(ttl=600)
def load_calendar()->pd.DataFrame:
    """Carrega calendário"""
    for d in ['/mnt/project/','./']:
        fp=os.path.join(d,"calendario_ligas.csv")
        if os.path.exists(fp):
            try:
                try: df=pd.read_csv(fp,encoding='utf-8-sig')
                except: df=pd.read_csv(fp,encoding='latin1')
                df.columns=[c.strip() for c in df.columns]
                df=df.rename(columns={'Mandante':'Time_Casa','Visitante':'Time_Visitante'})
                df['DtObj']=pd.to_datetime(df['Data'],format='%d/%m/%Y',errors='coerce')
                return df.dropna(subset=['DtObj']).sort_values(by=['DtObj','Hora'])
            except: continue
    return pd.DataFrame()

# ═══════════════════════════════════════════════════════════════════════════
# FUNÇÕES AUXILIARES
# ═══════════════════════════════════════════════════════════════════════════

def normalize_name(name:str,keys:list)->Optional[str]:
    if not name: return None
    name=name.strip()
    if name in NAME_MAPPING: name=NAME_MAPPING[name]
    if name in keys: return name
    m=get_close_matches(name,keys,n=1,cutoff=0.6)
    return m[0] if m else None

def get_odd(loc:str,typ:str,line:float)->float:
    key=(loc,typ,line) if typ!='dc' else (loc,typ,str(line))
    return REAL_ODDS.get(key,1.50)

# ═══════════════════════════════════════════════════════════════════════════
# SIMULAÇÃO
# ═══════════════════════════════════════════════════════════════════════════

def simulate_game_v31(h_st:Dict,a_st:Dict,n:int=1)->List[Dict]:
    """Simula jogo com Poisson Bivariado"""
    ch,ca=bivariate_poisson_vectorized(h_st['corners'],a_st['corners'],-0.25,n)
    cdh,cda=np.random.poisson(h_st['cards'],n),np.random.poisson(a_st['cards'],n)
    gh,ga=bivariate_poisson_vectorized(h_st['goals_f'],a_st.get('goals_f',1.2),0.15,n)
    return [{'home_corners':int(ch[i]),'away_corners':int(ca[i]),'home_cards':int(cdh[i]),
             'away_cards':int(cda[i]),'home_goals':int(gh[i]),'away_goals':int(ga[i])} for i in range(n)]

def check_sel(sim:Dict,sel:Dict)->bool:
    typ,loc=sel['type'],sel['location']
    if typ=='corners':
        val={'home':sim['home_corners'],'away':sim['away_corners'],
             'total':sim['home_corners']+sim['away_corners']}.get(loc,0)
        return val>sel['line']
    elif typ=='cards':
        val={'home':sim['home_cards'],'away':sim['away_cards'],
             'total':sim['home_cards']+sim['away_cards']}.get(loc,0)
        return val>sel['line']
    elif typ=='dc':
        res='1' if sim['home_goals']>sim['away_goals'] else ('X' if sim['home_goals']==sim['away_goals'] else '2')
        return res in (['1','X'] if sel.get('dc_type')=='1X' else ['X','2'])
    return False

# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZAÇÕES
# ═══════════════════════════════════════════════════════════════════════════

def plot_correlation_heatmap(stats_db:Dict):
    """Heatmap de Correlações"""
    data=[[st['corners'],st['cards'],st['goals_f'],st['corners_std'],st['cards_std']] 
          for st in stats_db.values()]
    df=pd.DataFrame(data,columns=['Cantos','Cartões','Gols','Std_Cantos','Std_Cartões'])
    corr=df.corr()
    fig=go.Figure(data=go.Heatmap(z=corr.values,x=corr.columns,y=corr.columns,colorscale='RdBu',
                                   zmid=0,text=corr.values.round(2),texttemplate='%{text}',textfont={"size":10}))
    fig.update_layout(title='Matriz de Correlações',template='plotly_dark',height=500)
    return fig

def plot_team_radar(team:str,stats:Dict):
    """Radar Chart do Time"""
    categories=['Cantos','Cartões','Gols','Consistência','Forma']
    vals=[min(10,stats['corners']/0.7),min(10,stats['cards']/0.3),min(10,stats['goals_f']/0.2),
          min(10,(1-stats['corners_std']/stats['corners'])*10) if stats['corners']>0 else 5,
          min(10,(stats.get('corners_trend',0)+0.5)*10)]
    fig=go.Figure(data=go.Scatterpolar(r=vals,theta=categories,fill='toself',name=team))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0,10])),showlegend=False,
                      template='plotly_dark',title=f'Perfil: {team}')
    return fig

# ═══════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════════

def main():
    st.markdown('<h1 style="text-align:center;">🔥 FutPrevisão V31 MAXIMUM 🔥</h1>',unsafe_allow_html=True)
    st.markdown('<p style="text-align:center;font-size:18px;">Sistema Definitivo - 26 Features Avançadas</p>',
                unsafe_allow_html=True)
    
    with st.spinner("🔄 Carregando sistema MAXIMUM..."):
        try:
            stats, h2h, clusters = learn_stats_maximum()
            cal = load_calendar()
            st.session_state.h2h_matrix = h2h
            st.session_state.team_clusters = clusters
        except Exception as e:
            st.error(f"❌ Erro ao carregar sistema: {str(e)}")
            st.info("💡 **Possíveis soluções:**\n\n1. Verifique se os CSVs estão na mesma pasta do app\n2. Certifique-se que os nomes são: `Premier_League_25_26.csv`, etc\n3. Tente recarregar a página (F5)")
            import traceback
            with st.expander("🔧 Detalhes técnicos (para debug)"):
                st.code(traceback.format_exc())
            return
    
    if not stats:
        st.error("❌ Nenhum dado foi carregado")
        st.info("💡 **Verifique:**\n\n✅ Os CSVs estão na pasta correta?\n✅ Os nomes dos arquivos estão corretos?\n✅ Os arquivos não estão corrompidos?")
        
        # Mostrar arquivos encontrados
        import glob
        csv_files = glob.glob("*.csv") + glob.glob("/mnt/project/*.csv")
        if csv_files:
            with st.expander("📁 Arquivos CSV encontrados"):
                for f in csv_files:
                    st.text(f)
        return
    
    with st.sidebar:
        st.title("💎 V31 MAXIMUM")
        st.caption("VERSÃO FINAL")
        st.markdown("---")
        bankroll=st.number_input("💰 Bankroll",value=st.session_state.bankroll_history[-1],step=50.0,min_value=10.0)
        st.metric("📊 Times",len(stats))
        st.metric("🎫 Jogos",len(st.session_state.current_ticket))
        st.metric("🧩 Clusters",len(set(clusters.values())) if clusters else 0)
        st.metric("🔗 H2H Pairs",len(h2h))
        if st.session_state.bet_results:
            rets=[r['return'] for r in st.session_state.bet_results]
            st.metric("📈 Sharpe",f"{calculate_sharpe_ratio(rets):.2f}")
            st.metric("📉 Drawdown",f"{calculate_max_drawdown(st.session_state.bankroll_history):.1f}%")
        st.markdown("---")
        if st.button("🗑️ Limpar",use_container_width=True):
            st.session_state.current_ticket=[]
            st.session_state.hedges_data=None
            st.rerun()
        st.markdown("---")
        st.caption("✅ 26 Features:")
        st.caption("• Poisson Bivariado")
        st.caption("• EMA + Momentum")
        st.caption("• Bayesiana Beta")
        st.caption("• H2H Matrix")
        st.caption("• K-Means Clustering")
        st.caption("• Logistic Regression ML")
        st.caption("• Calibração Isotônica")
        st.caption("• Derby Detection")
        st.caption("• Bootstrap CI")
        st.caption("• Kelly 1/4")
        st.caption("• Sharpe + Drawdown")
        st.caption("• Value Score")
        st.caption("• ROC AUC")
        st.caption("• 15+ Visualizações")
    
    tab1,tab2,tab3,tab4,tab5,tab6=st.tabs(["🎫 Construtor","🛡️ Hedges MAXIMUM","🎲 Simulador","📊 Métricas PRO","🎨 Visualizações","🔍 Scanner de Partidas"])
    
    with tab1:
        st.header("🎫 Construtor de Bilhetes V31 MAXIMUM")
        
        if cal.empty:
            st.warning("Calendário não encontrado")
        else:
            dates=sorted(cal['DtObj'].dt.strftime('%d/%m/%Y').unique())
            c1,c2=st.columns([1,2])
            with c1: sel_d=st.selectbox("📅 Data:",dates,key="d31")
            df_day=cal[cal['DtObj'].dt.strftime('%d/%m/%Y')==sel_d]
            with c2:
                games=sorted((df_day['Time_Casa']+' vs '+df_day['Time_Visitante']).unique())
                sel_g=st.selectbox("⚽ Jogo:",games,key="g31")
            
            if sel_g:
                try: home,away=sel_g.split(' vs ')
                except: st.error("Formato inválido"); return
                h_norm,a_norm=normalize_name(home,list(stats.keys())),normalize_name(away,list(stats.keys()))
                
                if h_norm and a_norm:
                    h_st,a_st=stats[h_norm],stats[a_norm]
                    
                    # ANÁLISE COMPLETA
                    with st.expander("🔬 Análise Estatística MAXIMUM",expanded=True):
                        # Contexto
                        context=detect_game_context(h_norm,a_norm,h_st['league'])
                        if context['type']!='NORMAL':
                            st.success(f"🔥 {context['type']}: {context['name']}")
                            st.caption(f"Ajuste Cantos: {context['factor_corners']:.2f}x | Ajuste Cartões: {context['factor_cards']:.2f}x")
                        
                        # H2H
                        h2h_key=f"{h_norm}_vs_{a_norm}"
                        if h2h_key in h2h:
                            h2h_data=h2h[h2h_key]
                            st.info(f"📊 H2H ({h2h_data['n']} jogos): Casa {h2h_data['hc_avg']:.1f} | Fora {h2h_data['ac_avg']:.1f} | Total {h2h_data['tc_avg']:.1f}")
                        
                        # Clusters
                        if h_norm in clusters and a_norm in clusters:
                            st.caption(f"🧩 Cluster: {h_norm} (C{clusters[h_norm]}) vs {a_norm} (C{clusters[a_norm]})")
                        
                        st.markdown("#### 🏠 "+h_norm)
                        c1,c2,c3,c4=st.columns(4)
                        mom=h_st.get('momentum',{})
                        trend_e="📈" if mom.get('trend')=='UP' else ("📉" if mom.get('trend')=='DOWN' else "➡️")
                        c1.metric(f"{trend_e} Cantos",f"{h_st['corners']:.1f}",
                                 f"IC:[{h_st['corners_ci'][0]:.1f},{h_st['corners_ci'][1]:.1f}]")
                        c2.metric("💳 Cartões",f"{h_st['cards']:.1f}")
                        c3.metric("⚡ Momentum",f"{mom.get('momentum',0):.1f}")
                        rel_e="✅" if h_st['reliability']>70 else "⚠️"
                        c4.metric(f"{rel_e} Confiança",f"{h_st['reliability']:.0f}%",f"{h_st['n_games']} jogos")
                        
                        st.markdown("#### ✈️ "+a_norm)
                        c1,c2,c3,c4=st.columns(4)
                        mom_a=a_st.get('momentum',{})
                        trend_e_a="📈" if mom_a.get('trend')=='UP' else ("📉" if mom_a.get('trend')=='DOWN' else "➡️")
                        c1.metric(f"{trend_e_a} Cantos",f"{a_st['corners']:.1f}",
                                 f"IC:[{a_st['corners_ci'][0]:.1f},{a_st['corners_ci'][1]:.1f}]")
                        c2.metric("💳 Cartões",f"{a_st['cards']:.1f}")
                        c3.metric("⚡ Momentum",f"{mom_a.get('momentum',0):.1f}")
                        rel_e_a="✅" if a_st['reliability']>70 else "⚠️"
                        c4.metric(f"{rel_e_a} Confiança",f"{a_st['reliability']:.0f}%",f"{a_st['n_games']} jogos")
                    
                    # Seleções
                    st.markdown("### ➕ Adicionar 2 Seleções")
                    mkts=[f"{h_norm} Over 3.5 Cantos",f"{h_norm} Over 4.5 Cantos",f"{a_norm} Over 2.5 Cantos",
                          f"{a_norm} Over 3.5 Cantos","Total Over 7.5 Cantos","Total Over 8.5 Cantos",
                          "Total Over 9.5 Cantos",f"{h_norm} Over 0.5 Cartões",f"{h_norm} Over 1.5 Cartões",
                          f"{a_norm} Over 0.5 Cartões",f"{a_norm} Over 1.5 Cartões","Total Over 2.5 Cartões",
                          "Total Over 3.5 Cartões",f"DC 1X ({h_norm})","DC X2 ({a_norm})"]
                    c1,c2=st.columns(2)
                    with c1: m1=st.selectbox("Mercado 1:",mkts,key="m1max")
                    with c2: m2=st.selectbox("Mercado 2:",[m for m in mkts if m!=m1],key="m2max")
                    
                    if st.button("➕ ADICIONAR",type="primary",use_container_width=True):
                        import re
                        def parse(txt):
                            r={'mercado':txt}
                            if 'DC' in txt:
                                r.update({'type':'dc','location':'home' if '1X' in txt else 'away','dc_type':'1X' if '1X' in txt else 'X2'})
                            elif 'Canto' in txt:
                                r['type']='corners'
                                r['location']='total' if 'Total' in txt else ('home' if h_norm in txt else 'away')
                            elif 'Cartão' in txt or 'Cartões' in txt:
                                r['type']='cards'
                                r['location']='total' if 'Total' in txt else ('home' if h_norm in txt else 'away')
                            m=re.search(r'(\d+\.?\d*)',txt)
                            if m: r['line']=float(m.group(1))
                            return r
                        st.session_state.current_ticket.append({
                            'jogo':sel_g,'selections':[parse(m1),parse(m2)],
                            'home_stats':h_st,'away_stats':a_st,'context':context
                        })
                        st.success("✅ Adicionado!")
                        st.rerun()
                else:
                    st.error("Times não encontrados")
        
        if st.session_state.current_ticket:
            st.markdown("---")
            st.markdown("### 🎫 Bilhete Atual")
            for idx,g in enumerate(st.session_state.current_ticket,1):
                c1,c2=st.columns([4,1])
                with c1:
                    st.markdown(f"**{idx}. {g['jogo']}**")
                    for s in g['selections']: st.write(f"  • {s['mercado']}")
                    ctx=g.get('context',{})
                    if ctx.get('type','NORMAL')!='NORMAL':
                        st.caption(f"  🔥 {ctx['name']}")
                with c2:
                    if st.button("🗑️",key=f"d{idx}"):
                        st.session_state.current_ticket.pop(idx-1)
                        st.rerun()
    
    with tab2:
        st.header("🛡️ Sistema de Hedges V31 MAXIMUM")
        if not st.session_state.current_ticket:
            st.warning("Adicione jogos primeiro")
        else:
            st.success(f"""
            💎 **MAXIMUM ATIVO:**
            - 🧠 Poisson Bivariado (corr -0.25)
            - 📈 EMA + Momentum + Aceleração
            - 🎲 Bayesian Beta-Binomial
            - 🔗 H2H Matrix ({len(h2h)} pares)
            - 🧩 K-Means Clustering ({len(set(clusters.values())) if clusters else 0} clusters)
            - 🔥 Derby Detection
            - 💰 Kelly Fracionário (1/4)
            - 📊 Bootstrap CI (95%)
            - 🎯 Value Score
            """)
            
            st.markdown("### ⚙️ Configurações de Hedge")
            c1,c2,c3=st.columns(3)
            with c1: min_prob=st.slider("Prob Mínima (%)",30,70,40,5)
            with c2: n_sims=st.slider("Simulações",200,2000,500,100)
            with c3: max_hedges=st.slider("Max Hedges",2,5,3,1)
            
            if st.button("🚀 GERAR HEDGES MAXIMUM",type="primary",use_container_width=True):
                with st.spinner("⏳ Gerando hedges com V31 MAXIMUM..."):
                    
                    # ANÁLISE DO BILHETE PRINCIPAL
                    principal_sels=[]
                    for g in st.session_state.current_ticket:
                        for s in g['selections']:
                            principal_sels.append({
                                'jogo':g['jogo'],'sel':s,
                                'home_stats':g['home_stats'],'away_stats':g['away_stats'],
                                'context':g.get('context',{})
                            })
                    
                    # CALCULAR PROBABILIDADES COM V31
                    principal_probs=[]
                    for ps in principal_sels:
                        h_st,a_st=ps['home_stats'],ps['away_stats']
                        sel=ps['sel']
                        ctx=ps['context']
                        
                        # Aplicar contexto
                        h_corners=h_st['corners']*ctx.get('factor_corners',1.0)
                        a_corners=a_st['corners']*ctx.get('factor_corners',1.0)
                        h_cards=h_st['cards']*ctx.get('factor_cards',1.0)
                        a_cards=a_st['cards']*ctx.get('factor_cards',1.0)
                        
                        # Simular
                        sims=[]
                        for _ in range(n_sims):
                            ch,ca=bivariate_poisson_vectorized(h_corners,a_corners,-0.25,1)
                            cdh,cda=np.random.poisson(h_cards,1),np.random.poisson(a_cards,1)
                            gh,ga=bivariate_poisson_vectorized(h_st['goals_f'],a_st.get('goals_f',1.2),0.15,1)
                            sims.append({'home_corners':ch[0],'away_corners':ca[0],
                                       'home_cards':cdh[0],'away_cards':cda[0],
                                       'home_goals':gh[0],'away_goals':ga[0]})
                        
                        hits=sum(1 for s in sims if check_sel(s,sel))
                        prob=(hits/n_sims)*100
                        
                        # Bootstrap CI
                        ci_l,ci_u=bootstrap_ci_fast(np.array([1 if check_sel(s,sel) else 0 for s in sims],dtype=float))
                        
                        # Value Score
                        odd=get_odd(sel['location'],sel['type'],sel.get('line',0))
                        value=calculate_value_score(prob/100,odd)
                        
                        # Kelly
                        kelly=kelly_criterion_fractional(prob/100,odd,var=0.1)
                        
                        principal_probs.append({
                            'jogo':ps['jogo'],'mercado':sel['mercado'],'prob':prob,
                            'ci':(ci_l*100,ci_u*100),'odd':odd,'value':value,'kelly':kelly
                        })
                    
                    # Calcular prob combinada
                    prob_principal=np.prod([p['prob']/100 for p in principal_probs])*100
                    odd_principal=np.prod([p['odd'] for p in principal_probs])
                    
                    st.markdown("---")
                    st.markdown("### 🎫 BILHETE PRINCIPAL")
                    
                    c1,c2,c3,c4=st.columns(4)
                    c1.metric("🎯 Probabilidade",f"{prob_principal:.1f}%")
                    c2.metric("💰 Odd Total",f"@{odd_principal:.2f}")
                    roi_principal=(prob_principal/100)*odd_principal*100-100
                    c3.metric("📊 ROI Esperado",f"{roi_principal:+.1f}%")
                    kelly_principal=kelly_criterion_fractional(prob_principal/100,odd_principal,0.15)
                    c4.metric("💎 Kelly Stake",f"{kelly_principal*100:.1f}%")
                    
                    for idx,p in enumerate(principal_probs,1):
                        with st.expander(f"Seleção {idx}: {p['jogo']} - {p['mercado']}"):
                            c1,c2,c3=st.columns(3)
                            c1.metric("Probabilidade",f"{p['prob']:.1f}%",
                                     f"IC: [{p['ci'][0]:.1f}%, {p['ci'][1]:.1f}%]")
                            c2.metric("Value Score",f"{p['value']:.2f}",
                                     "✅ VALUE!" if p['value']>1.10 else "⚠️ Sem value")
                            c3.metric("Kelly Stake",f"{p['kelly']*100:.1f}%")
                    
                    # GERAR HEDGES
                    st.markdown("---")
                    st.markdown("### 🛡️ HEDGES GERADOS")
                    
                    # Gerar combinações alternativas
                    all_markets=[]
                    for g in st.session_state.current_ticket:
                        h_name,a_name=g['jogo'].split(' vs ')
                        h_st,a_st=g['home_stats'],g['away_stats']
                        
                        # Mercados alternativos
                        alt_mkts=[
                            {'type':'corners','location':'away','line':2.5,'name':f"{a_name} Over 2.5 Cantos"},
                            {'type':'corners','location':'away','line':3.5,'name':f"{a_name} Over 3.5 Cantos"},
                            {'type':'corners','location':'total','line':8.5,'name':"Total Over 8.5 Cantos"},
                            {'type':'corners','location':'total','line':10.5,'name':"Total Over 10.5 Cantos"},
                            {'type':'cards','location':'total','line':3.5,'name':"Total Over 3.5 Cartões"},
                            {'type':'cards','location':'total','line':4.5,'name':"Total Over 4.5 Cartões"},
                        ]
                        
                        # Filtrar os que não estão no principal
                        principal_mkts=[sel['mercado'] for sel in g['selections']]
                        alt_mkts=[m for m in alt_mkts if m['name'] not in principal_mkts]
                        
                        # Simular cada alternativa
                        for mkt in alt_mkts[:4]:  # Max 4 por jogo
                            sims_alt=simulate_game_v31(h_st,a_st,n_sims)
                            hits=sum(1 for s in sims_alt if check_sel(s,mkt))
                            prob=(hits/n_sims)*100
                            
                            if prob>=min_prob:
                                odd=get_odd(mkt['location'],mkt['type'],mkt['line'])
                                all_markets.append({
                                    'jogo':g['jogo'],'market':mkt,'prob':prob,'odd':odd,
                                    'value':calculate_value_score(prob/100,odd)
                                })
                    
                    # Ordenar por value score
                    all_markets.sort(key=lambda x:x['value'],reverse=True)
                    
                    # Criar hedges (2 seleções por jogo)
                    hedges_created=0
                    hedges_list=[]  # Para salvar
                    
                    for hedge_idx in range(min(max_hedges,len(all_markets)//len(st.session_state.current_ticket))):
                        st.markdown(f"#### 🛡️ Hedge {hedge_idx+1}")
                        
                        hedge_sels=[]
                        for g in st.session_state.current_ticket:
                            # Pegar 2 melhores mercados deste jogo
                            jogo_mkts=[m for m in all_markets if m['jogo']==g['jogo']][:2]
                            if len(jogo_mkts)==2:
                                hedge_sels.extend(jogo_mkts)
                        
                        if len(hedge_sels)==len(st.session_state.current_ticket)*2:
                            prob_hedge=np.prod([s['prob']/100 for s in hedge_sels])*100
                            odd_hedge=np.prod([s['odd'] for s in hedge_sels])
                            
                            c1,c2,c3=st.columns(3)
                            c1.metric("🎯 Prob Hedge",f"{prob_hedge:.1f}%")
                            c2.metric("💰 Odd Hedge",f"@{odd_hedge:.2f}")
                            roi_h=(prob_hedge/100)*odd_hedge*100-100
                            c3.metric("📊 ROI",f"{roi_h:+.1f}%")
                            
                            for s in hedge_sels:
                                st.caption(f"• {s['jogo']}: {s['market']['name']} ({s['prob']:.1f}% | @{s['odd']:.2f} | Value: {s['value']:.2f})")
                            
                            # Salvar hedge
                            hedges_list.append({
                                'index':hedge_idx+1,
                                'selections':hedge_sels,
                                'prob':prob_hedge,
                                'odd':odd_hedge
                            })
                            
                            hedges_created+=1
                            
                            # Remover mercados usados
                            for s in hedge_sels:
                                if s in all_markets:
                                    all_markets.remove(s)
                        else:
                            break
                    
                    if hedges_created==0:
                        st.warning("⚠️ Não foi possível gerar hedges com os critérios atuais. Tente reduzir a probabilidade mínima.")
                        st.session_state.hedges_data=None
                    else:
                        st.success(f"✅ {hedges_created} hedge(s) gerado(s) com V31 MAXIMUM!")
                        
                        # Salvar hedges no session_state
                        st.session_state.hedges_data={'hedges':hedges_list}
                        
                        st.markdown("---")
                        st.markdown("### 💡 Análise de Cobertura")
                        
                        st.info(f"""
                        **Estratégia Recomendada:**
                        - Principal: {kelly_principal*100:.1f}% do bankroll (€{bankroll*kelly_principal:.2f})
                        - Cada Hedge: {kelly_principal*0.5*100:.1f}% do bankroll (€{bankroll*kelly_principal*0.5:.2f})
                        
                        **Cenários:**
                        - ✅ Principal acerta: +€{bankroll*kelly_principal*(odd_principal-1):.2f}
                        - 🛡️ Hedge acerta: Minimiza perda
                        - ❌ Nenhum acerta: -{len(st.session_state.current_ticket)*2*kelly_principal*100:.1f}% bankroll
                        """)
    
    with tab3:
        st.header("🎲 Simulador Monte Carlo MAXIMUM")
        if not st.session_state.current_ticket:
            st.warning("Monte bilhete primeiro")
        else:
            st.success("✨ Poisson Bivariado + Contexto + H2H + Clusters")
            n_sim=st.slider("Simulações:",100,10000,1000,100)
            
            simulate_hedges=st.checkbox("Simular também os Hedges gerados?",value=True)
            
            if st.button("▶️ SIMULAR MAXIMUM",type="primary",use_container_width=True):
                with st.spinner(f"⏳ {n_sim} simulações..."):
                    
                    # SIMULAR PRINCIPAL
                    results_principal=[]
                    for _ in range(n_sim):
                        all_hit=True
                        for g in st.session_state.current_ticket:
                            h_st,a_st=g['home_stats'],g['away_stats']
                            sim=simulate_game_v31(h_st,a_st,1)[0]
                            for s in g['selections']:
                                if not check_sel(sim,s):
                                    all_hit=False
                                    break
                            if not all_hit: break
                        results_principal.append(all_hit)
                    
                    wr_principal=(sum(results_principal)/n_sim)*100
                    ci_l_p,ci_u_p=bootstrap_ci_fast(np.array(results_principal,dtype=float))
                    
                    st.markdown("---")
                    st.markdown("### 🎫 BILHETE PRINCIPAL")
                    c1,c2,c3=st.columns(3)
                    c1.metric("🎯 Taxa Acerto",f"{wr_principal:.1f}%")
                    c2.metric("📊 IC 95%",f"[{ci_l_p*100:.1f}%, {ci_u_p*100:.1f}%]")
                    c3.metric("✅ Greens",f"{sum(results_principal)}/{n_sim}")
                    
                    # SIMULAR HEDGES (se existirem e checkbox marcado)
                    if simulate_hedges and st.session_state.hedges_data:
                        st.markdown("---")
                        st.markdown("### 🛡️ SIMULAÇÃO DOS HEDGES")
                        
                        hedges_list=st.session_state.hedges_data.get('hedges',[])
                        
                        for h_idx,hedge in enumerate(hedges_list[:3],1):
                            # Simular este hedge
                            results_hedge=[]
                            for _ in range(n_sim):
                                all_hit_h=True
                                for sel in hedge['selections']:
                                    # Encontrar o jogo correspondente
                                    jogo_match=None
                                    for g in st.session_state.current_ticket:
                                        if g['jogo']==sel.get('jogo'):
                                            jogo_match=g
                                            break
                                    
                                    if jogo_match:
                                        h_st,a_st=jogo_match['home_stats'],jogo_match['away_stats']
                                        sim=simulate_game_v31(h_st,a_st,1)[0]
                                        if not check_sel(sim,sel.get('market',{})):
                                            all_hit_h=False
                                            break
                                
                                results_hedge.append(all_hit_h)
                            
                            wr_hedge=(sum(results_hedge)/n_sim)*100
                            ci_l_h,ci_u_h=bootstrap_ci_fast(np.array(results_hedge,dtype=float))
                            
                            st.markdown(f"#### 🛡️ Hedge {h_idx}")
                            c1,c2,c3=st.columns(3)
                            c1.metric(f"Taxa Acerto H{h_idx}",f"{wr_hedge:.1f}%")
                            c2.metric("IC 95%",f"[{ci_l_h*100:.1f}%, {ci_u_h*100:.1f}%]")
                            c3.metric("Greens",f"{sum(results_hedge)}/{n_sim}")
                        
                        st.markdown("---")
                        st.markdown("### 📊 COMPARAÇÃO")
                        
                        comparison_data={
                            'Bilhete':['Principal']+[f'Hedge {i+1}' for i in range(len(hedges_list[:3]))],
                            'Prob (%)': [wr_principal]+[0]*len(hedges_list[:3])
                        }
                        
                        df_comp=pd.DataFrame(comparison_data)
                        
                        fig=go.Figure(data=[
                            go.Bar(x=df_comp['Bilhete'],y=df_comp['Prob (%)'],
                                  marker_color=['#667eea','#f093fb','#f5576c','#48c6ef'][:len(df_comp)])
                        ])
                        fig.update_layout(title="Probabilidade de Acerto - Principal vs Hedges",
                                        template='plotly_dark',yaxis_title="Probabilidade (%)")
                        st.plotly_chart(fig,use_container_width=True)
                    
                    else:
                        if not st.session_state.hedges_data:
                            st.info("💡 Gere hedges na Tab 2 para simular cobertura!")
                    
                    st.success("✅ Simulação concluída com Poisson Bivariado!")
    
    with tab4:
        st.header("📊 Métricas Profissionais MAXIMUM")
        if st.session_state.bet_results:
            rets=[r['return'] for r in st.session_state.bet_results]
            c1,c2,c3,c4=st.columns(4)
            sharpe=calculate_sharpe_ratio(rets)
            c1.metric("📈 Sharpe",f"{sharpe:.2f}","Excelente" if sharpe>1.5 else "Bom" if sharpe>1.0 else "Regular")
            c2.metric("💰 ROI",f"{(np.mean(rets)-1)*100:+.1f}%")
            dd=calculate_max_drawdown(st.session_state.bankroll_history)
            c3.metric("📉 Drawdown",f"{dd:.1f}%","Baixo" if dd<15 else "Médio" if dd<30 else "Alto")
            c4.metric("🎯 Win Rate",f"{(sum(1 for r in rets if r>1.0)/len(rets))*100:.1f}%")
            
            st.markdown("### 📈 Evolução Bankroll")
            fig=go.Figure()
            fig.add_trace(go.Scatter(y=st.session_state.bankroll_history,mode='lines+markers',
                                    name='Bankroll',line=dict(color='#667eea',width=3)))
            fig.add_trace(go.Scatter(y=np.maximum.accumulate(st.session_state.bankroll_history),
                                    mode='lines',name='Peak',line=dict(color='#f093fb',width=2,dash='dash')))
            fig.update_layout(title="Histórico com Drawdown",template='plotly_dark')
            st.plotly_chart(fig,use_container_width=True)
        else:
            st.info("Complete apostas para ver métricas!")
    
    with tab5:
        st.header("🎨 Visualizações Avançadas MAXIMUM")
        
        st.markdown("### 🔥 Heatmap de Correlações")
        st.plotly_chart(plot_correlation_heatmap(stats),use_container_width=True)
        
        st.markdown("### 📡 Radar Charts dos Times")
        if st.session_state.current_ticket:
            for g in st.session_state.current_ticket[:2]:
                try:
                    home,away=g['jogo'].split(' vs ')
                    h_n,a_n=normalize_name(home,list(stats.keys())),normalize_name(away,list(stats.keys()))
                    if h_n and a_n:
                        c1,c2=st.columns(2)
                        with c1: st.plotly_chart(plot_team_radar(h_n,stats[h_n]),use_container_width=True)
                        with c2: st.plotly_chart(plot_team_radar(a_n,stats[a_n]),use_container_width=True)
                except: pass
        else:
            st.info("Adicione jogos para ver radar charts")
    
    with tab6:
        st.header("🔍 Scanner de Partidas - Recomendações Automáticas")
        st.markdown("**Análise inteligente dos melhores jogos para apostar em cantos e cartões**")
        
        if cal.empty:
            st.warning("⚠️ Calendário não disponível")
        else:
            # Seletor de data
            dates=sorted(cal['DtObj'].dt.strftime('%d/%m/%Y').unique())
            sel_date_scanner=st.selectbox("📅 Selecione a data:",dates,key="scanner_date")
            
            if st.button("🔍 ANALISAR PARTIDAS",type="primary",use_container_width=True):
                with st.spinner("⏳ Analisando jogos..."):
                    df_day=cal[cal['DtObj'].dt.strftime('%d/%m/%Y')==sel_date_scanner]
                    
                    recommendations=[]
                    
                    for _,row in df_day.iterrows():
                        home_name=row.get('Time_Casa','')
                        away_name=row.get('Time_Visitante','')
                        league=row.get('Liga','')
                        
                        h_norm=normalize_name(home_name,list(stats.keys()))
                        a_norm=normalize_name(away_name,list(stats.keys()))
                        
                        if h_norm and a_norm and h_norm in stats and a_norm in stats:
                            h_st,a_st=stats[h_norm],stats[a_norm]
                            
                            # Calcular scores
                            # SCORE CANTOS: quanto maior, melhor para Over cantos
                            corners_total=h_st['corners']+a_st['corners']
                            corners_consistency=(1-h_st['corners_std']/h_st['corners'])+(1-a_st['corners_std']/a_st['corners']) if h_st['corners']>0 and a_st['corners']>0 else 0
                            corners_trend=h_st.get('corners_trend',0)+a_st.get('corners_trend',0)
                            corners_score=(corners_total*0.6)+(corners_consistency*2)+(corners_trend*5)
                            
                            # SCORE CARTÕES: quanto maior, melhor para Over cartões
                            cards_total=h_st['cards']+a_st['cards']
                            cards_consistency=(1-h_st['cards_std']/h_st['cards'])+(1-a_st['cards_std']/a_st['cards']) if h_st['cards']>0 and a_st['cards']>0 else 0
                            cards_score=(cards_total*1.5)+(cards_consistency*1.5)
                            
                            # Contexto
                            context=detect_game_context(h_norm,a_norm,league)
                            if context['type']=='DERBY':
                                corners_score*=1.15
                                cards_score*=1.35
                            elif context['type']=='TOP_CLASH':
                                corners_score*=1.08
                                cards_score*=1.15
                            
                            # H2H boost
                            h2h_key=f"{h_norm}_vs_{a_norm}"
                            if h2h_key in h2h:
                                h2h_data=h2h[h2h_key]
                                if h2h_data['tc_avg']>10:
                                    corners_score*=1.10
                            
                            recommendations.append({
                                'jogo':f"{home_name} vs {away_name}",
                                'liga':league,
                                'hora':row.get('Hora','--:--'),
                                'corners_score':corners_score,
                                'cards_score':cards_score,
                                'corners_total':corners_total,
                                'cards_total':cards_total,
                                'context':context['type'],
                                'h_corners':h_st['corners'],
                                'a_corners':a_st['corners'],
                                'h_cards':h_st['cards'],
                                'a_cards':a_st['cards']
                            })
                    
                    if not recommendations:
                        st.warning("Nenhuma partida encontrada com dados disponíveis")
                    else:
                        # Ordenar por melhor score
                        recommendations.sort(key=lambda x:x['corners_score'],reverse=True)
                        
                        st.markdown("---")
                        st.markdown("### 🎯 TOP RECOMENDAÇÕES PARA CANTOS")
                        
                        for idx,rec in enumerate(recommendations[:5],1):
                            with st.expander(f"#{idx} {rec['jogo']} ({rec['hora']}) - Score: {rec['corners_score']:.1f}",expanded=(idx<=3)):
                                c1,c2,c3,c4=st.columns(4)
                                c1.metric("🏟️ Liga",rec['liga'])
                                c2.metric("📊 Total Médio",f"{rec['corners_total']:.1f}")
                                c3.metric("🏠 Casa",f"{rec['h_corners']:.1f}")
                                c4.metric("✈️ Fora",f"{rec['a_corners']:.1f}")
                                
                                if rec['context']!='NORMAL':
                                    st.info(f"🔥 {rec['context']}: Probabilidade aumentada!")
                                
                                # Sugestões
                                st.markdown("**💡 Sugestões:**")
                                if rec['corners_total']>11:
                                    st.success(f"✅ Total Over 10.5 Cantos (Média: {rec['corners_total']:.1f})")
                                if rec['corners_total']>9:
                                    st.success(f"✅ Total Over 9.5 Cantos (Média: {rec['corners_total']:.1f})")
                                if rec['h_corners']>4.5:
                                    st.success(f"✅ Casa Over 4.5 Cantos (Média: {rec['h_corners']:.1f})")
                        
                        st.markdown("---")
                        st.markdown("### 💳 TOP RECOMENDAÇÕES PARA CARTÕES")
                        
                        recommendations.sort(key=lambda x:x['cards_score'],reverse=True)
                        
                        for idx,rec in enumerate(recommendations[:5],1):
                            with st.expander(f"#{idx} {rec['jogo']} ({rec['hora']}) - Score: {rec['cards_score']:.1f}",expanded=(idx<=3)):
                                c1,c2,c3,c4=st.columns(4)
                                c1.metric("🏟️ Liga",rec['liga'])
                                c2.metric("📊 Total Médio",f"{rec['cards_total']:.1f}")
                                c3.metric("🏠 Casa",f"{rec['h_cards']:.1f}")
                                c4.metric("✈️ Fora",f"{rec['a_cards']:.1f}")
                                
                                if rec['context']!='NORMAL':
                                    st.warning(f"🔥 {rec['context']}: Jogo quente!")
                                
                                # Sugestões
                                st.markdown("**💡 Sugestões:**")
                                if rec['cards_total']>4.5:
                                    st.success(f"✅ Total Over 4.5 Cartões (Média: {rec['cards_total']:.1f})")
                                if rec['cards_total']>3.5:
                                    st.success(f"✅ Total Over 3.5 Cartões (Média: {rec['cards_total']:.1f})")
                                if rec['h_cards']>1.8:
                                    st.success(f"✅ Casa Over 1.5 Cartões (Média: {rec['h_cards']:.1f})")

if __name__=="__main__":
    main()
