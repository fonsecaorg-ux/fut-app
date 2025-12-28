"""
🤖 ATUALIZADOR AUTOMÁTICO - FutPrevisão V32.1
Atualiza CSVs direto do Football-Data.co.uk
✅ Backup automático antes de atualizar
✅ Tratamento de erros robusto
✅ Relatório detalhado
✅ Verificação de integridade
"""

import requests
import pandas as pd
import os
import shutil
from datetime import datetime

print("╔═══════════════════════════════════════════════════╗")
print("║     ATUALIZADOR AUTOMÁTICO - FUTPREVISÃO V32.1    ║")
print("╚═══════════════════════════════════════════════════╝")
print()

# Mapeamento das ligas
LEAGUES = {
    'E0': 'Premier_League_25_26.csv',
    'SP1': 'La_Liga_25_26.csv',
    'I1': 'Serie_A_25_26.csv',
    'D1': 'Bundesliga_25_26.csv',
    'F1': 'Ligue_1_25_26.csv',
    'E1': 'Championship_Inglaterra_25_26.csv',
    'D2': 'Bundesliga_2.csv',
    'B1': 'Pro_League_Belgica_25_26.csv',
    'T1': 'Super_Lig_Turquia_25_26.csv',
    'SC0': 'Premiership_Escocia_25_26.csv'
}

# Criar backup antes de atualizar
backup_folder = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
backup_count = 0

print("💾 Criando backup de segurança...\n")

for filename in LEAGUES.values():
    if os.path.exists(filename):
        try:
            if not os.path.exists(backup_folder):
                os.makedirs(backup_folder)
            shutil.copy2(filename, os.path.join(backup_folder, filename))
            backup_count += 1
        except Exception as e:
            print(f"⚠️  Erro ao fazer backup de {filename}: {e}")

if backup_count > 0:
    print(f"✅ {backup_count} arquivos salvos em: {backup_folder}\n")

print("─────────────────────────────────────────────────────")
print("🔄 Iniciando atualização...\n")

success_count = 0
error_count = 0
total = len(LEAGUES)
errors = []

for code, filename in LEAGUES.items():
    try:
        url = f"https://www.football-data.co.uk/mmz4281/2526/{code}.csv"
        
        print(f"📥 {filename:45s}", end=" ")
        
        response = requests.get(url, timeout=15)
        
        if response.status_code == 200:
            # Verificar se CSV é válido
            try:
                # Testar leitura
                test_df = pd.read_csv(pd.io.common.BytesIO(response.content))
                
                if len(test_df) > 0:
                    # Salvar
                    with open(filename, 'wb') as f:
                        f.write(response.content)
                    
                    # Recarregar para contar jogos
                    df = pd.read_csv(filename)
                    num_games = len(df)
                    
                    print(f"✅ {num_games:3d} jogos")
                    success_count += 1
                else:
                    print(f"⚠️  Vazio")
                    error_count += 1
                    errors.append(f"{filename}: CSV vazio")
                    
            except Exception as e:
                print(f"❌ CSV inválido")
                error_count += 1
                errors.append(f"{filename}: CSV inválido - {str(e)[:50]}")
        else:
            print(f"❌ HTTP {response.status_code}")
            error_count += 1
            errors.append(f"{filename}: HTTP {response.status_code}")
    
    except requests.exceptions.Timeout:
        print(f"❌ Timeout")
        error_count += 1
        errors.append(f"{filename}: Timeout após 15s")
    
    except Exception as e:
        print(f"❌ Erro")
        error_count += 1
        errors.append(f"{filename}: {str(e)[:50]}")

print()
print("─────────────────────────────────────────────────────")
print()

# Resultado final
if success_count == total:
    print(f"🎉 SUCESSO TOTAL! {success_count}/{total} ligas atualizadas")
elif success_count > 0:
    print(f"⚠️  PARCIAL: {success_count}/{total} ligas atualizadas")
    print(f"❌ {error_count} erros encontrados")
else:
    print(f"❌ FALHA: Nenhuma liga atualizada")

print()

# Gerar relatório detalhado
report = f"""
╔═══════════════════════════════════════════════════╗
║           RELATÓRIO DE ATUALIZAÇÃO                ║
║           {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}                        ║
╚═══════════════════════════════════════════════════╝

📊 RESULTADO:
   ✅ Atualizadas: {success_count}/{total}
   ❌ Erros: {error_count}/{total}
   💾 Backup: {backup_folder}

📅 Data: {datetime.now().strftime('%d/%m/%Y')}
⏰ Hora: {datetime.now().strftime('%H:%M:%S')}

───────────────────────────────────────────────────────

DETALHES POR LIGA:
"""

for code, filename in LEAGUES.items():
    if os.path.exists(filename):
        try:
            df = pd.read_csv(filename)
            size = len(df)
            modified = datetime.fromtimestamp(os.path.getmtime(filename))
            
            # Verificar última data
            if 'Date' in df.columns and len(df) > 0:
                last_date = df['Date'].iloc[0]
                report += f"\n✅ {filename:45s} {size:3d} jogos (último: {last_date})"
            else:
                report += f"\n✅ {filename:45s} {size:3d} jogos"
                
        except Exception as e:
            report += f"\n⚠️  {filename:45s} Erro ao ler: {str(e)[:30]}"
    else:
        report += f"\n❌ {filename:45s} NÃO ENCONTRADO"

# Adicionar erros se houver
if errors:
    report += "\n\n───────────────────────────────────────────────────────"
    report += "\n\n❌ ERROS ENCONTRADOS:\n"
    for error in errors:
        report += f"\n   • {error}"

report += "\n\n───────────────────────────────────────────────────────"
report += "\n\n💡 PRÓXIMOS PASSOS:"
report += "\n   1. Verificar CSVs atualizados (opcional)"
report += "\n   2. Execute: streamlit run futprevisao_v32_1_MAXIMUM.py"
report += "\n   3. Sistema pronto com dados frescos! 🚀"
report += "\n"

# Salvar relatório
with open('relatorio_atualizacao.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print(report)
print("📄 Relatório salvo: relatorio_atualizacao.txt")
print()

# Aviso final
if success_count == total:
    print("✅ ATUALIZAÇÃO CONCLUÍDA COM SUCESSO!")
elif success_count > 0:
    print("⚠️  ATUALIZAÇÃO PARCIAL - Verifique os erros acima")
    print(f"   Backup disponível em: {backup_folder}")
else:
    print("❌ ATUALIZAÇÃO FALHOU - Verifique sua conexão")
    print(f"   Arquivos originais preservados em: {backup_folder}")

print()
input("Pressione ENTER para sair...")
