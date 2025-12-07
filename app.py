# app.py
"""
Aplicação única (Flask) que:
- carrega dados raw da pasta `raw_json/` ou `output/consolidated_football_stats_complete.csv`
- normaliza e gera agregações para o Dashboard
- serve API /analytics e página HTML em /
- fornece endpoint POST /run-etl para forçar reprocessamento

Como usar:
1) pip install flask pandas python-dateutil
2) Coloque seus .json raw em ./raw_json/ ou o CSV em ./output/consolidated_football_stats_complete.csv
3) python app.py
4) Abrir http://127.0.0.1:5000/ no navegador
"""

from flask import Flask, jsonify, send_file, request, make_response
from pathlib import Path
import pandas as pd
import json
from dateutil import parser as dparser
import datetime
import traceback

app = Flask(__name__, static_folder=None)

# Config
RAW_DIR = Path("raw_json")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)
DASH_JSON_PATH = OUTPUT_DIR / "dashboard_input.json"

# Utility: safe parse date
def safe_parse_date(v):
    if pd.isna(v) or v is None:
        return None
    try:
        return pd.to_datetime(v)
    except Exception:
        try:
            return dparser.parse(str(v))
        except Exception:
            return None

def load_matches():
    """
    Carrega partidas:
    - Prioridade 1: output/consolidated_football_stats_complete.csv (se existir)
    - Senão: concatena todos os JSONs de RAW_DIR
    """
    csv_path = OUTPUT_DIR / "consolidated_football_stats_complete.csv"
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path, low_memory=False)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
            return df
        except Exception as e:
            print("Erro ao ler CSV consolidado:", e)
    # fallback: ler todos jsons na pasta raw_json
    frames = []
    if not RAW_DIR.exists():
        print("Pasta raw_json/ não encontrada. Crie e coloque seus JSONs lá.")
        return pd.DataFrame()
    for p in RAW_DIR.glob("*.json"):
        try:
            j = json.loads(p.read_text(encoding="utf-8"))
            # se o JSON for um objeto com key 'teams' ou 'matches', tentamos extrair os matches
            if isinstance(j, dict):
                # tentativas comuns de estrutura
                if 'teams' in j and isinstance(j['teams'], list):
                    # alguns arquivos tem times com 'allMatches' -> explodir
                    for t in j['teams']:
                        if isinstance(t, dict) and 'allMatches' in t and isinstance(t['allMatches'], list):
                            for m in t['allMatches']:
                                # attach teamName maybe
                                frames.append(pd.json_normalize(m))
                        elif isinstance(t, dict) and 'matches' in t:
                            for m in t['matches']:
                                frames.append(pd.json_normalize(m))
                elif 'matches' in j and isinstance(j['matches'], list):
                    for m in j['matches']:
                        frames.append(pd.json_normalize(m))
                else:
                    # se for lista
                    # tenta tratar j como lista de partidas
                    for k,v in j.items():
                        pass
                    if isinstance(j, list):
                        for item in j:
                            frames.append(pd.json_normalize(item))
                    else:
                        # caso: JSON é um objeto com keys não padrão -> tentar extrair arrays
                        for key, val in j.items():
                            if isinstance(val, list):
                                for item in val:
                                    if isinstance(item, dict):
                                        frames.append(pd.json_normalize(item))
            elif isinstance(j, list):
                for item in j:
                    frames.append(pd.json_normalize(item))
        except Exception as e:
            print(f"Falha ao ler {p.name}: {e}")
    if not frames:
        return pd.DataFrame()
    try:
        df = pd.concat(frames, ignore_index=True, sort=False)
        # normalize common date fields
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        # try to derive timestamp if present
        if 'timestamp' in df.columns:
            # some timestamps are milliseconds as string
            def try_ts(x):
                try:
                    if pd.isna(x):
                        return None
                    xi = int(x)
                    # check length -> if > 10 likely ms
                    if xi > 1_000_000_0000:
                        return pd.to_datetime(xi, unit='ms')
                    else:
                        return pd.to_datetime(xi, unit='s')
                except Exception:
                    return None
            df['ts_date'] = df['timestamp'].apply(try_ts)
            if df['date'].isna().any():
                df['date'] = df['date'].fillna(df['ts_date'])
        return df
    except Exception as e:
        print("Erro concat JSONs:", e)
        return pd.DataFrame()

def normalize_matches(df):
    """
    Garante que as colunas existam e sejam coerentes.
    Retorna DataFrame com colunas básicas normalizadas.
    """
    if df is None or df.empty:
        return df
    # copy
    df = df.copy()
    # normalizar data
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    else:
        df['date'] = pd.NaT
    # try to fill missing from 'matchDate' or 'fixtureDate'
    for alt in ['matchDate','fixtureDate','kickoff','kickoff_time']:
        if df['date'].isna().any() and alt in df.columns:
            df['date'] = df['date'].fillna(pd.to_datetime(df[alt], errors='coerce'))
    # teams
    if 'homeTeam' in df.columns:
        df['home_team'] = df['homeTeam']
    elif 'home' in df.columns:
        df['home_team'] = df['home']
    else:
        df['home_team'] = df.get('home_team', df.get('home_team_name', None))
    if 'awayTeam' in df.columns:
        df['away_team'] = df['awayTeam']
    elif 'away' in df.columns:
        df['away_team'] = df['away']
    else:
        df['away_team'] = df.get('away_team', df.get('away_team_name', None))
    # goals / corners / cards / shots / possession / fouls
    numeric_map = {
        'homeGoals': 'home_goals', 'awayGoals': 'away_goals',
        'homeCorners':'home_corners','awayCorners':'away_corners',
        'homeCards':'home_cards','awayCards':'away_cards',
        'homeShots':'home_shots','awayShots':'away_shots',
        'homePossession':'home_possession','awayPossession':'away_possession',
        'homeFouls':'home_fouls','awayFouls':'away_fouls',
        'home_offsides':'home_offsides','away_offsides':'away_offsides'
    }
    for src, dst in numeric_map.items():
        if src in df.columns:
            df[dst] = pd.to_numeric(df[src], errors='coerce')
        elif dst in df.columns:
            df[dst] = pd.to_numeric(df[dst], errors='coerce')
        else:
            df[dst] = 0
    # competition
    if 'league' in df.columns:
        df['competition'] = df['league']
    elif 'competition' in df.columns:
        df['competition'] = df['competition']
    else:
        # try slug/name
        df['competition'] = df.get('leagueName', df.get('competition_name', None))
    # match id
    if 'id' in df.columns:
        df['match_id'] = df['id']
    elif 'matchId' in df.columns:
        df['match_id'] = df['matchId']
    else:
        df['match_id'] = df.index.astype(str)
    # timestamp ms
    if 'timestamp' in df.columns:
        df['timestamp_ms'] = pd.to_numeric(df['timestamp'], errors='coerce')
    else:
        df['timestamp_ms'] = None
    # ensure date exists (fallback to timestamp)
    if df['date'].isna().any() and df['timestamp_ms'].notna().any():
        def ts_to_date(x):
            try:
                if pd.isna(x): return pd.NaT
                xi = int(x)
                if xi > 1_000_000_0000:
                    return pd.to_datetime(xi, unit='ms')
                else:
                    return pd.to_datetime(xi, unit='s')
            except:
                return pd.NaT
        df['date'] = df['date'].fillna(df['timestamp_ms'].apply(ts_to_date))
    # trim team strings
    df['home_team'] = df['home_team'].apply(lambda x: str(x).strip() if pd.notna(x) else None)
    df['away_team'] = df['away_team'].apply(lambda x: str(x).strip() if pd.notna(x) else None)
    return df

def build_aggregates(df_matches, df_tickets=None):
    """
    Gera dicionário com agregações úteis para o dashboard.
    """
    out = {}
    if df_matches is None or df_matches.empty:
        out['ok'] = False
        out['message'] = "no matches"
        return out
    df = df_matches.copy()
    # ensure date column
    if 'date' not in df.columns:
        df['date'] = pd.NaT
    # daily counts by date
    df['match_date'] = df['date'].dt.date
    daily_counts = df.groupby('match_date').agg(matches=('match_id','count')).reset_index()
    daily_counts['match_date'] = daily_counts['match_date'].astype(str)
    out['daily_matches'] = daily_counts.to_dict(orient='records')

    # by market - best effort (many raws may not have market)
    if 'market' in df.columns:
        by_market = df.groupby('market').agg(matches=('match_id','count')).reset_index()
        out['by_market'] = by_market.to_dict(orient='records')
    else:
        out['by_market'] = []

    # by competition
    if 'competition' in df.columns:
        by_comp = df.groupby('competition').agg(matches=('match_id','count')).reset_index()
        out['by_competition'] = by_comp.sort_values('matches', ascending=False).to_dict(orient='records')
    else:
        out['by_competition'] = []

    # heatmap day/hour
    if df['date'].notna().any():
        df['dow'] = df['date'].dt.day_name()
        df['hour'] = df['date'].dt.hour
        heat = df.groupby(['dow','hour']).size().reset_index(name='count')
        out['heatmap'] = heat.to_dict(orient='records')
    else:
        out['heatmap'] = []

    # basic team-level aggregates (corners and goals averages)
    team_stats = []
    if 'home_team' in df.columns:
        homes = df.groupby('home_team').agg(
            home_matches=('match_id','count'),
            avg_home_corners=('home_corners','mean'),
            avg_home_goals=('home_goals','mean')
        ).reset_index().rename(columns={'home_team':'team'})
        aways = df.groupby('away_team').agg(
            away_matches=('match_id','count'),
            avg_away_corners=('away_corners','mean'),
            avg_away_goals=('away_goals','mean')
        ).reset_index().rename(columns={'away_team':'team'})
        merged = pd.merge(homes, aways, on='team', how='outer').fillna(0)
        merged['total_matches'] = merged['home_matches'] + merged['away_matches']
        team_stats = merged.sort_values('total_matches', ascending=False).head(50).to_dict(orient='records')
    out['team_stats'] = team_stats

    # If tickets provided -> daily pnl and summary
    if df_tickets is not None and not df_tickets.empty:
        dt = df_tickets.copy()
        dt['date'] = pd.to_datetime(dt['date'], errors='coerce')
        daily = dt.groupby(dt['date'].dt.date).agg(
            profit=('profit','sum'),
            stake=('stake','sum'),
            count=('ticket_id','count')
        ).reset_index().sort_values('date')
        daily['cumulative_profit'] = daily['profit'].cumsum()
        daily['date'] = daily['date'].astype(str)
        out['daily'] = daily.to_dict(orient='records')
        total_profit = float(dt['profit'].sum())
        total_stake = float(dt['stake'].sum()) if 'stake' in dt.columns else 0.0
        win_rate = float((dt['result']=='Green').mean()) if 'result' in dt.columns else 0.0
        out['summary'] = {
            'total_profit': total_profit,
            'total_stake': total_stake,
            'roi': (total_profit/total_stake) if total_stake else 0.0,
            'win_rate': win_rate
        }
    else:
        out['daily'] = []
        out['summary'] = {'total_profit': 0.0, 'total_stake': 0.0, 'roi': 0.0, 'win_rate': 0.0}

    out['ok'] = True
    out['generated_at'] = datetime.datetime.utcnow().isoformat() + "Z"
    return out

def run_etl_and_save():
    """
    Executa pipeline completo e grava dashboard_input.json
    Retorna o dicionário gerado.
    """
    try:
        df_matches = load_matches()
        if df_matches is None or df_matches.empty:
            result = {'ok': False, 'message': 'no match files found'}
            # salvar para inspeção
            DASH_JSON_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
            return result
        df_matches = normalize_matches(df_matches)
        # load tickets if exists
        tickets_path = OUTPUT_DIR / "tickets.csv"
        df_tickets = None
        if tickets_path.exists():
            try:
                df_tickets = pd.read_csv(tickets_path)
            except Exception as e:
                print("Erro lendo tickets.csv:", e)
                df_tickets = None
        analytics = build_aggregates(df_matches, df_tickets)
        # save file
        DASH_JSON_PATH.write_text(json.dumps(analytics, ensure_ascii=False, indent=2), encoding='utf-8')
        print("Saved dashboard JSON to", DASH_JSON_PATH)
        return analytics
    except Exception as e:
        tb = traceback.format_exc()
        print("ETL failure:", e, tb)
        err = {'ok': False, 'message': 'etl_error', 'error': str(e)}
        DASH_JSON_PATH.write_text(json.dumps(err, ensure_ascii=False, indent=2), encoding='utf-8')
        return err

# run ETL at startup
print("Starting ETL at startup...")
startup_result = run_etl_and_save()

@app.route("/analytics", methods=["GET"])
def analytics():
    if DASH_JSON_PATH.exists():
        txt = DASH_JSON_PATH.read_text(encoding='utf-8')
        try:
            return make_response(txt, 200, {"Content-Type":"application/json; charset=utf-8"})
        except Exception:
            return jsonify({'ok':False,'message':'read_error'})
    else:
        return jsonify({'ok':False,'message':'no_dashboard_file'})

@app.route("/run-etl", methods=["POST"])
def run_etl():
    # proteção básica: aceita POST sem autenticação (para testes). Em produção adicione auth.
    res = run_etl_and_save()
    return jsonify(res)

@app.route("/", methods=["GET"])
def index():
    # Serve página HTML com dashboard simples que consome /analytics
    html = """
<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Dashboard - Gestão de Banca</title>
<link rel="preconnect" href="https://cdnjs.cloudflare.com">
<style>
  :root{
    --bankGreen:#0B6E4F;
    --posGreen:#16A34A;
    --negRed:#E11D48;
    --bgCard:#F7FAFC;
    --neutral:#374151;
  }
  body{font-family:Inter, system-ui, -apple-system, "Helvetica Neue", Arial; margin:0; padding:16px; background:#ffffff; color:var(--neutral);}
  .container{max-width:980px;margin:0 auto;}
  .card{background:#fff;border-radius:10px;padding:16px;box-shadow:0 6px 18px rgba(15,23,42,0.06);margin-bottom:16px;}
  .kpi{display:flex;gap:12px;}
  .kpi .item{flex:1;padding:12px;background:var(--bgCard);border-radius:8px;text-align:center;}
  h1{margin:0 0 8px 0;font-size:28px;color:var(--bankGreen)}
  .small{font-size:13px;color:#6b7280}
  .bank-value{font-size:36px;font-weight:700;color:var(--bankGreen)}
  .muted{color:#6b7280}
  .btn{display:inline-block;padding:8px 12px;border-radius:8px;background:var(--bankGreen);color:white;text-decoration:none;}
  footer{margin-top:20px;color:#9CA3AF;font-size:13px}
  @media (max-width:600px){
    .kpi{flex-direction:column}
  }
</style>
</head>
<body>
  <div class="container">
    <h1>Gestão Profissional de Banca — Dashboard</h1>
    <p class="small">Dados derivados de arquivos locais (pasta <code>raw_json/</code> ou arquivo CSV consolidado).</p>

    <div id="main-area">
      <div class="card" id="card-banca">
        <div style="display:flex;justify-content:space-between;align-items:center">
          <div>
            <div class="small">Banca Atual</div>
            <div class="bank-value" id="bank-value">R$ 0.00</div>
            <div class="small" id="bank-trend">↑ 0.00</div>
          </div>
          <div style="width:420px;height:160px;">
            <canvas id="chart-line"></canvas>
          </div>
        </div>
      </div>

      <div class="card kpi">
        <div class="item">
          <div class="small">ROI</div>
          <div id="kpi-roi" style="font-weight:700">0.00%</div>
        </div>
        <div class="item">
          <div class="small">Win rate</div>
          <div id="kpi-win" style="font-weight:700">0.00%</div>
        </div>
        <div class="item">
          <div class="small">Total Profit</div>
          <div id="kpi-profit" style="font-weight:700">R$ 0.00</div>
        </div>
      </div>

      <div class="card">
        <h3 style="margin-top:0">Desempenho por Mercado</h3>
        <div style="width:100%;height:320px;"><canvas id="chart-bars"></canvas></div>
      </div>

      <div class="card" style="display:flex;gap:12px;flex-wrap:wrap;">
        <div style="flex:1;min-width:260px">
          <h3 style="margin-top:0">Por Competição</h3>
          <div style="width:100%;height:280px;"><canvas id="chart-pie"></canvas></div>
        </div>
        <div style="flex:1;min-width:260px">
          <h3 style="margin-top:0">Heatmap (dia/hora)</h3>
          <div id="heat" style="max-height:280px;overflow:auto"></div>
        </div>
      </div>

      <div class="card">
        <button id="btn-refresh" class="btn">Atualizar (rodar ETL)</button>
        <span style="margin-left:12px;color:#6b7280">Clique para forçar reprocessamento (POST /run-etl).</span>
      </div>

    </div>

    <footer>Gerado por app.py • copie este arquivo e adapte conforme necessário.</footer>
  </div>

  <!-- Chart.js CDN -->
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <script>
    async function fetchAnalytics(){
      try{
        const r = await fetch('/analytics');
        if(!r.ok) throw new Error('failed');
        const data = await r.json();
        return data;
      }catch(e){
        console.error(e);
        return null;
      }
    }

    function formatMoney(v){
      return 'R$ ' + Number(v || 0).toLocaleString('pt-BR', {minimumFractionDigits:2, maximumFractionDigits:2});
    }

    function drawLine(canvas, daily){
      const ctx = canvas.getContext('2d');
      const labels = daily.map(d=>d.date || d[0] || '');
      const data = daily.map(d=> (d.cumulative_profit!==undefined? d.cumulative_profit : d.profit) || 0 );
      if(window.lineChart) window.lineChart.destroy();
      window.lineChart = new Chart(ctx, {
        type:'line',
        data:{ labels, datasets:[{ label:'Lucro acumulado', data, fill:true, borderColor:'#16A34A', backgroundColor:'rgba(22,163,74,0.12)', tension:0.2 }]},
        options:{ responsive:true, maintainAspectRatio:false }
      });
    }

    function drawBars(canvas, items){
      const ctx = canvas.getContext('2d');
      const labels = items.map(i=>i.market || i.name || '--');
      const data = items.map(i=> i.profit!==undefined? i.profit : (i.matches || 0));
      if(window.barChart) window.barChart.destroy();
      window.barChart = new Chart(ctx, {
        type:'bar',
        data:{ labels, datasets:[{ label:'Profit/Matches', data, backgroundColor: labels.map((l,i)=> data[i] >= 0 ? '#16A34A' : '#E11D48') }]},
        options:{ indexAxis:'x', responsive:true, maintainAspectRatio:false }
      });
    }

    function drawPie(canvas, items){
      const ctx = canvas.getContext('2d');
      const labels = items.map(i=> i.competition || i[0] || i.name || '');
      const data = items.map(i=> i.matches || 0);
      if(window.pieChart) window.pieChart.destroy();
      window.pieChart = new Chart(ctx, {
        type:'pie',
        data:{ labels, datasets:[{ data, backgroundColor: labels.map((l,i)=> ['#0B6E4F','#16A34A','#F59E0B','#3B82F6','#A855F7'][i%5]) }]},
        options:{ responsive:true, maintainAspectRatio:false }
      });
    }

    function buildHeatmap(container, heat){
      // heat: [{dow, hour, count}, ...]
      // build matrix day x hour
      const days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'];
      let map = {};
      heat.forEach(r=>{
        const day = r.dow || r[0] || '';
        const hour = String(r.hour || r[1] || 0);
        const count = r.count || r[2] || 0;
        map[day+'_'+hour] = count;
      });
      let html = '<table style="width:100%;border-collapse:collapse"><thead><tr><th style="text-align:left">Dia\\Hora</th>';
      for(let h=0; h<24; h++) html += '<th style="padding:4px;font-size:11px">'+h+'</th>';
      html += '</tr></thead><tbody>';
      for(const d of days){
        html += '<tr><td style="padding:6px 8px;font-weight:600">'+d+'</td>';
        for(let h=0; h<24; h++){
          const v = map[d+'_'+String(h)] || 0;
          const bg = v>0 ? `background: rgba(11,110,79,${Math.min(0.9, 0.08+v/10)})` : '';
          html += `<td style="padding:4px;text-align:center;${bg}">` + (v||'') + '</td>';
        }
        html += '</tr>';
      }
      html += '</tbody></table>';
      container.innerHTML = html;
    }

    async function refreshAll(){
      const data = await fetchAnalytics();
      if(!data || !data.ok){
        console.warn("No analytics available", data);
        return;
      }
      const dash = data;
      // BANK value: we use summary.total_profit as proxy
      const bankVal = (dash.summary && dash.summary.total_profit) || 0;
      document.getElementById('bank-value').innerText = formatMoney(bankVal);
      document.getElementById('kpi-profit').innerText = formatMoney(dash.summary && dash.summary.total_profit || 0);
      document.getElementById('kpi-roi').innerText = ((dash.summary && dash.summary.roi || 0)*100).toFixed(2) + '%';
      document.getElementById('kpi-win').innerText = ((dash.summary && dash.summary.win_rate || 0)*100).toFixed(2) + '%';

      // draw line - expects dash.daily array: [{date, profit, cumulative_profit}, ...]
      drawLine(document.getElementById('chart-line'), dash.daily || []);

      // bars - by_market or fallback empty
      drawBars(document.getElementById('chart-bars'), dash.by_market || []);

      // pie - by_competition
      drawPie(document.getElementById('chart-pie'), dash.by_competition || []);

      // heatmap
      buildHeatmap(document.getElementById('heat'), dash.heatmap || []);
    }

    document.getElementById('btn-refresh').addEventListener('click', async ()=>{
      // call run-etl
      try{
        const r = await fetch('/run-etl', {method:'POST'});
        const j = await r.json();
        console.log('ETL result', j);
        await refreshAll();
        alert('ETL executado. Ver console para detalhes.');
      }catch(e){
        console.error(e);
        alert('Falha ao executar ETL. Veja console.');
      }
    });

    // initial load
    refreshAll();
  </script>
</body>
</html>
    """
    return html

if __name__ == "__main__":
    # run dev server
    app.run(host="0.0.0.0", port=5000, debug=True)