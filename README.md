# 📈 Stock Analyzer

Projeto pessoal em Python para **análise técnica, backtesting, gestão de risco e apoio à decisão** em ações e outros ativos (via Yahoo Finance).

O objetivo não é "prever o mercado", mas construir uma base **validada estatisticamente** onde o estado técnico, a evidência histórica, os fundamentais e o risco são combinados numa decisão explicável.

> ⚠️ **Aviso:** projeto educativo e experimental. Nada neste repositório constitui aconselhamento financeiro. Resultados de backtests não garantem desempenho futuro.

---

## ✨ Funcionalidades

### Indicadores técnicos
- SMA, EMA, RSI, MACD e ATR implementados de raiz (`indicators.py`)

### Technical State e Predictive Score
- **Technical State** categórico: Trend (SMA/EMA), Momentum (MACD) e condição RSI
- **Predictive Score** normalizado entre `-1` e `+1`, com nível de confiança (Low / Medium / High)
- **Predictive Evidence**: retorno médio, mediano, *positive rate* e nº de observações a 5, 20 e 60 dias, comparados com uma baseline (*Mean Edge*, *Median Edge*, *Positive Rate Edge*)

### Backtesting
- Execução no **dia seguinte, ao preço de abertura** (evita *look-ahead bias*)
- Comissões e slippage configuráveis
- Gestão de posição aberta no fim do período
- Comparação com **Buy & Hold**
- Métricas: retorno, win rate, melhor/pior trade, média e mediana dos trades, duração média, **Max Drawdown**, **Sharpe**, **Sortino** e **Calmar**
- Estratégias comparáveis: SMA, EMA, MACD, RSI e combinações

### Gestão de risco
- *Position sizing* baseado em ATR e risco por trade
- Limite máximo de alocação por posição
- **Trailing stop** por ATR, com tratamento de *gap down*

### Validação
- Separação **In-Sample / Out-of-Sample** (70% / 30%)
- Cálculo de *edge decay* e estabilidade entre IS e OOS
- Validação *cross-asset* com taxa de estabilidade agregada

### Análise fundamental
- Métricas via `yfinance`: Forward P/E, PEG, crescimento de receita/lucros, ROE, margens, dívida/capital, P/FCF, EV/EBITDA, EPS, ROA
- Score relativo com decisão `PASS` / `NEUTRAL` / `FAIL`
- Diagnóstico de crescimento vs. valuation

### Painel de decisão
Combina sinal diário (1D), confirmação intradiária (4H), fundamentais, dimensionamento de posição, stop inicial e métricas históricas de qualidade, e imprime uma recomendação (`BUY`, `WAIT`, `WATCH`, `HOLD`, `SELL`) com a respetiva justificação.

### Gráficos
- Dashboard técnico (preço, médias, RSI, MACD)
- Preço com entradas, saídas e trailing stop
- Curva de capital vs. Buy & Hold

---

## 🗂️ Estrutura do projeto

| Ficheiro | Responsabilidade |
|---|---|
| `main.py` | Pontos de entrada: painel de decisão e validação cross-asset |
| `data.py` | Download de dados (diários e 4H) via `yfinance` |
| `indicators.py` | SMA, EMA, RSI, MACD, ATR |
| `technical_score.py` | Technical State, Predictive Score e Predictive Evidence |
| `backtest.py` | Sinais, motor de backtest, métricas e benchmark |
| `risk.py` | Position sizing, trailing stop e deteção de stop |
| `validation.py` | Validação In-Sample / Out-of-Sample |
| `fundamentals.py` | Obtenção e avaliação de métricas fundamentais |
| `domain.py` | Dataclasses (`Trade`, `RiskConfig`, `FundamentalMetrics`, …) e thresholds |
| `charts.py` | Gráficos com `matplotlib` |
| `analyzer.py` | Versão inicial do Technical Score (legado) |

---

## 🚀 Instalação

```bash
git clone https://github.com/<o-teu-utilizador>/<o-teu-repositorio>.git
cd <o-teu-repositorio>
pip install yfinance pandas numpy matplotlib
```

## ▶️ Utilização

**Painel de decisão para um ativo:**

```python
from main import get_daily_decision

get_daily_decision("NVDA", capital=10000.0)
```

**Validação de risco em vários ativos (5 anos):**

```python
from main import run_cross_asset_validation

run_cross_asset_validation()
```

Ou diretamente:

```bash
python main.py
```

### Configuração de risco

```python
from domain import RiskConfig

risk_config = RiskConfig(
    risk_per_trade=0.05,   # 5% do capital arriscado por trade
    atr_multiplier=3.5,    # stop a 3.5x ATR
    max_allocation=1.0,    # máx. 100% do capital numa posição
    atr_period=14,
)
```

---

## 🧭 Roadmap

| Fase | Tema | Estado |
|---|---|---|
| 1–5 | Base do sistema e backtesting | ✅ Concluído |
| 6 | Validação do Technical Score | ✅ Concluído |
| 7 | Predictive Evidence (7.1–7.3) | ✅ Concluído |
| 7.4 | Medir robustez | 🔄 Em progresso |
| 7.5–7.7 | Treino/teste, Predictive Score e confiança | ⏳ Em curso / por validar |
| 8 | Refactor do código | ⏳ Em curso |
| 9 | Charts do backtest | ⏳ Em curso |
| 10 | Gestão de risco | ⏳ Em curso |
| 11 | Análise fundamental | ⏳ Em curso |
| 12 | Decision Support | ⏳ Em curso |
| 13 | Validação do sistema (walk-forward, ciclos de mercado) | 📋 Planeado |
| 14 | Portfolio, base de dados, notícias, macro, ML | 📋 Planeado |

---

## ⚠️ Limitações conhecidas

- Os dados vêm do Yahoo Finance e dependem da sua disponibilidade e qualidade.
- Os dados intradiários (4H) estão limitados aos últimos ~60 dias.
- O sistema está em desenvolvimento ativo: os thresholds e as regras ainda estão a ser validados.

## 📄 Licença

Por definir.
