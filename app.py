# 🚀 COMO EXECUTAR O FUTPREVISÃO V14.0

## 📋 PRÉ-REQUISITOS

```bash
# Instalar dependências (se necessário)
pip install streamlit pandas numpy
```

## ⚙️ ESTRUTURA DE ARQUIVOS

Certifique-se de que sua estrutura está assim:

```
projeto/
├── futprevisao_v14_0.py          # ← Código principal
├── Premier_League_25_26.csv      # ← CSVs das ligas
├── La_Liga_25_26.csv
├── Serie_A_25_26.csv
├── Bundesliga_25_26.csv
├── Ligue_1_25_26.csv
├── Championship_Inglaterra_25_26.csv
├── Bundesliga_2.csv
├── Pro_League_Belgica_25_26.csv
├── Super_Lig_Turquia_25_26.csv
├── Premiership_Escocia_25_26.csv
├── arbitros.csv                   # ← Árbitros
├── arbitros_5_ligas_2025_2026.csv
└── calendario_ligas.csv           # ← Jogos agendados
```

## 🎯 OPÇÃO 1: EXECUTAR DO DIRETÓRIO DO PROJETO

```bash
# Navegar para o diretório onde estão os CSVs
cd /caminho/para/seu/projeto

# Copiar o arquivo Python para lá
cp futprevisao_v14_0.py .

# Executar
streamlit run futprevisao_v14_0.py
```

## 🎯 OPÇÃO 2: AJUSTAR O CAMINHO NO CÓDIGO

Se seus CSVs estão em outro diretório, edite a linha 43 do código:

```python
# Linha 43 de futprevisao_v14_0.py
BASE_PATH = "/caminho/completo/para/seus/csvs/"  # ← Mudar aqui
```

**Exemplos:**
```python
# Windows
BASE_PATH = "C:/Users/Diego/Documents/FutPrevisao/"

# Mac/Linux
BASE_PATH = "/home/diego/futprevisao/"

# Path relativo (se executar do mesmo diretório)
BASE_PATH = "./"
```

Depois execute:
```bash
streamlit run futprevisao_v14_0.py
```

## ✅ VERIFICAR SE ESTÁ FUNCIONANDO

1. **Abrir o navegador**: http://localhost:8501
2. **Verificar a mensagem**: "✅ X times carregados | Y árbitros cadastrados"
3. **Se aparecer "0 times"**: O caminho dos CSVs está errado

## 🔧 SOLUÇÃO DE PROBLEMAS

### Erro: "FileNotFoundError"
**Causa**: Os CSVs não estão no caminho correto

**Solução**:
1. Verifique onde estão seus CSVs:
   ```bash
   ls -la *.csv
   ```
2. Ajuste a variável `BASE_PATH` no código (linha 43)

### Erro: "No module named 'streamlit'"
**Causa**: Dependências não instaladas

**Solução**:
```bash
pip install streamlit pandas numpy
```

### App carrega mas não mostra dados
**Causa**: Problema no formato dos CSVs

**Solução**:
1. Verifique se os CSVs têm as colunas corretas
2. Execute o teste de validação:
   ```bash
   python TESTE_V14.py
   ```

## 📊 EXECUTAR TESTES

Para validar a implementação:

```bash
python TESTE_V14.py
```

Você deve ver:
```
🎯 RESULTADO FINAL: 6/6 testes passaram (100.0%)
🎉 SUCESSO! Todas as implementações V14.0 estão funcionando corretamente!
```

## 🎮 USANDO O APLICATIVO

### Modo 1: Análise Única
1. Selecionar "🎯 Análise Única"
2. Digitar: Time Casa, Time Visitante, Árbitro (opcional)
3. Clicar em "🔍 Analisar Jogo"

### Modo 2: Jogos Agendados
1. Selecionar "📅 Jogos Agendados"
2. Filtrar por liga e data
3. Clicar em "Analisar" no jogo desejado

### Modo 3: Teste PSG x Flamengo
1. Selecionar "🧪 Teste PSG x Flamengo"
2. Clicar em "🚀 Executar Teste"
3. Ver validação das implementações V14.0

## 📱 RESULTADO ESPERADO

Ao carregar o app, você deve ver:

```
╔══════════════════════════════════════╗
║     ⚽ FutPrevisão V14.0             ║
║  🧠 Causality Engine                 ║
║  🆕 Chutes + Vermelhos               ║
║  📊 85% Precisão                     ║
╚══════════════════════════════════════╝

✅ 117 times carregados | 62 árbitros cadastrados
```

## 🆘 PRECISA DE AJUDA?

1. **Verifique os logs**: Streamlit mostra erros no terminal
2. **Teste o carregamento**: Execute o script de teste
3. **Valide os CSVs**: Abra um CSV para ver se está correto
4. **Revise a documentação**: Leia EXPLICACAO_TECNICA_V14.md

## 🎯 DICAS

✅ **Execute do mesmo diretório dos CSVs** (mais fácil)  
✅ **Use caminhos absolutos** se executar de outro lugar  
✅ **Verifique encoding dos CSVs** (deve ser UTF-8)  
✅ **Mantenha os nomes dos arquivos** (case-sensitive no Linux)

---

**Desenvolvido por:** Diego  
**Versão:** V14.0 Causality Engine  
**Data:** Dezembro 2025

🚀 **Boa sorte com as apostas!**
