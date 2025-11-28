# ⚽ FutPrevisão Pro (v2.6)

![Status](https://img.shields.io/badge/STATUS-PRODUCTION-brightgreen?style=for-the-badge)
![Python](https://img.shields.io/badge/PYTHON-3.9%2B-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/FRAMEWORK-STREAMLIT-red?style=for-the-badge&logo=streamlit)

> **Sistema de Inteligência Esportiva para Análise Probabilística de Futebol.**
> *Desenvolvido como parte do portfólio acadêmico (Análise e Desenvolvimento de Sistemas - Unisanta).*

---

## 🎯 Visão Geral
O **FutPrevisão Pro** é uma ferramenta analítica que utiliza **Modelagem Estatística (Distribuição de Poisson)** e **Algoritmos de Tensão de Jogo** para prever cenários em partidas de futebol. Diferente de sites de apostas comuns, este sistema foca na "matemática do jogo", cruzando dados históricos reais de 2025 para gerar probabilidades de:
* **Escanteios (Cantos)**
* **Cartões (Disciplinar)**
* **Gols (Poder Ofensivo/Defensivo)**

## 🏗️ Arquitetura do Sistema

O projeto segue uma arquitetura modular focada em resiliência e tratamento de dados:

```mermaid
graph TD
    A[Usuário] -->|Login Seguro| B(Interface Streamlit)
    B --> C{Processador Lógico}
    C -->|Leitura| D[dados_times.csv]
    C -->|Fator Humano| E[arbitros.csv]
    C -->|Fallback| F[Dicionário de Backup]
    D -->|Dados Brutos| G[Logs de Jogos 2025]
