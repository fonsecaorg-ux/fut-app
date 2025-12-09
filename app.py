# ⚽ FutPrevisão Pro - PARA GITHUB

Sistema Inteligente de Análise de Escanteios com Validação Histórica

---

## 🚀 COMO USAR

### **1. PREPARAR ARQUIVOS**

Coloque os 5 arquivos .txt do Adam Choi **NA MESMA PASTA** do `app.py`:

```
sua_pasta/
├── app.py
├── Escanteios_Preimier_League_-_codigo_fonte.txt
├── Escanteios_Espanha.txt
├── Escanteios_Italia.txt
├── Escanteios_Alemanha.txt
└── Escanteios_França.txt
```

### **2. INSTALAR DEPENDÊNCIAS**

```bash
pip install streamlit pandas numpy plotly
```

### **3. EXECUTAR**

```bash
streamlit run app.py
```

---

## ✅ O QUE FAZ

### **Sistema Completo:**
- ✅ Login/Registro de usuários
- ✅ Dashboard com 5 ligas europeias
- ✅ Sistema de previsão com IA
- ✅ **VALIDAÇÃO HISTÓRICA** (IA vs Dados Reais)
- ✅ Análise de sequências dos últimos jogos
- ✅ Alertas automáticos inteligentes
- ✅ Recomendações de stake
- ✅ Sistema de bilhetes
- ✅ Interface profissional

### **Exemplo de Análise:**
```
🏟️ Atalanta vs Chelsea

🏠 Atalanta (Casa) +4.5 escanteios:
🤖 IA: 72% | 📊 Real: 71.4% (5/7 jogos)
✅ VALIDADO - Divergência: +0.6%
📈 Últimos 5: ✅✅✅❌✅ (80%)
💰 Stake: 5-7%

✈️ Chelsea (Fora) +4.5 escanteios:
🤖 IA: 65% | 📊 Real: 62.5% (5/8 jogos)
✅ VALIDADO - Divergência: +2.5%
📈 Últimos 5: ✅❌✅✅✅ (80%)
💰 Stake: 5-7%

🎯 Recomendação: AMBOS VALIDADOS! 🔥
```

---

## 📊 DADOS INCLUSOS

- ✅ **Premier League** - 20 times
- ✅ **La Liga** - 20 times
- ✅ **Serie A** - 20 times
- ✅ **Bundesliga** - 18 times
- ✅ **Ligue 1** - 18 times

**Total: 96 times cadastrados!**

---

## 🎯 FUNCIONALIDADES PRINCIPAIS

### **1. Validação Histórica**
Compara previsões da IA com dados reais do Adam Choi:
- ✅ Divergência < 10% = VALIDADO (confiança ALTA)
- ⚠️ Divergência 10-20% = ALERTA (confiança MÉDIA)
- 🚨 Divergência > 20% = DIVERGENTE (confiança BAIXA)

### **2. Análise de Sequências**
Mostra últimos 5 jogos visualmente:
- ✅✅✅❌✅ = 80% de acerto
- 🔥 SEQUÊNCIA QUENTE (4-5 acertos)
- ✅ BOA FORMA (3 acertos)
- ⚠️ IRREGULAR (2 acertos)
- 🥶 SEQUÊNCIA FRIA (0-1 acertos)

### **3. Alertas Automáticos**
Sistema detecta automaticamente:
- 🚨 IA muito otimista/pessimista
- ⚠️ Time em sequência fraca
- 🔥 Time em fogo
- 💡 Dicas de gestão de banca

### **4. Recomendações de Stake**
Baseado na confiança da análise:
- ✅ Alta confiança: 5-7% da banca
- ⚠️ Média confiança: 2-4% da banca
- 🚨 Baixa confiança: 1-2% ou EVITAR

---

## 📱 COMO USAR O APP

### **Passo 1: Login**
1. Crie uma conta (Registrar)
2. Faça login

### **Passo 2: Dashboard**
1. Veja métricas gerais
2. Explore times por liga
3. Veja estatísticas completas

### **Passo 3: Fazer Previsão**
1. Selecione time casa + liga
2. Selecione time fora + liga
3. Escolha linha de escanteios (3.5, 4.5, 5.5, 6.5)
4. Clique em "Gerar Análise Completa"

### **Passo 4: Ver Análise**
1. Compare IA vs Histórico
2. Veja sequência dos últimos jogos
3. Leia os alertas
4. Veja recomendação de stake
5. Tome sua decisão

### **Passo 5: Adicionar ao Bilhete**
1. Clique em "Adicionar ao Bilhete"
2. Vá para "Meus Bilhetes"
3. Gerencie suas apostas

---

## ⚠️ IMPORTANTE

### **Arquivos .txt na MESMA PASTA do app.py!**
```
✅ CORRETO:
futebol/
├── app.py
├── Escanteios_Preimier_League_-_codigo_fonte.txt
└── ...

❌ ERRADO:
futebol/
├── app.py
dados/
├── Escanteios_Preimier_League_-_codigo_fonte.txt
```

### **Nomes EXATOS dos arquivos:**
- `Escanteios_Preimier_League_-_codigo_fonte.txt` (sim, com erro "Preimier")
- `Escanteios_Espanha.txt`
- `Escanteios_Italia.txt`
- `Escanteios_Alemanha.txt`
- `Escanteios_França.txt`

---

## 🔧 TROUBLESHOOTING

### **Problema: "Arquivo não encontrado"**
**Solução:** Coloque os 5 .txt na mesma pasta do app.py

### **Problema: "Dados não carregados"**
**Solução:** Verifique os nomes EXATOS dos arquivos

### **Problema: "ModuleNotFoundError"**
**Solução:** Execute `pip install streamlit pandas numpy plotly`

---

## 📈 EXEMPLO REAL

### **Arsenal (Casa) vs Liverpool (Fora) - Linha +4.5**

**Arsenal:**
```
🤖 IA: 68%
📊 Real: 72% (5/7 jogos em casa)
✅ VALIDADO - Divergência: -4%
📈 Últimos 5: ✅✅✅❌✅
🔥 SEQUÊNCIA QUENTE (4/5)
💰 Stake: 5-7%
```

**Liverpool:**
```
🤖 IA: 45%
📊 Real: 37.5% (3/8 jogos fora)
⚠️ ALERTA - Divergência: +7.5%
📈 Últimos 5: ❌✅✅❌❌
⚠️ IRREGULAR (2/5)
💰 Stake: 2-4%
```

**Recomendação Final:**
```
🎯 APOSTAR: Arsenal +4.5 (VALIDADO)
⚠️ EVITAR: Liverpool +4.5 (fraco fora)
💡 Melhor opção: Arsenal sozinho ou linha menor para Liverpool
```

---

## 🎉 PRONTO!

**Agora é só:**
1. Copiar `app.py` para seu GitHub
2. Colocar os 5 .txt na mesma pasta
3. Executar `streamlit run app.py`
4. Fazer apostas INTELIGENTES! 🚀

---

## 💡 DICAS

### **Quando APOSTAR:**
- ✅ Validação ALTA (≤10% divergência)
- ✅ Sequência BOA (3+ acertos em 5)
- ✅ Sem alertas críticos
- ✅ Stake 5-7%

### **Quando TER CAUTELA:**
- ⚠️ Validação MÉDIA (10-20% divergência)
- ⚠️ Sequência IRREGULAR (2/5)
- ⚠️ Alguns alertas
- ⚠️ Stake 2-4%

### **Quando EVITAR:**
- 🚨 Validação BAIXA (>20% divergência)
- 🚨 Sequência FRIA (0-1/5)
- 🚨 Muitos alertas
- 🚨 Stake 1-2% ou NÃO APOSTAR

---

## 📞 SUPORTE

**Dúvidas?** Veja os exemplos acima ou teste com alguns times conhecidos!

---

**Versão: 2.0**  
**Data: 09/12/2025**  
**Desenvolvido por: Claude + Diego** 🔥
