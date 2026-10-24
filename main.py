from data import get_stock_data, get_stock_data_4h
from domain import DEFAULT_THRESHOLDS, RiskConfig, FundamentalMetrics
from fundamentals import (
    evaluate_fundamentals,
    get_fundamental_data,
    get_stock_fundamental_metrics,
)
from indicators import calculate_atr
from risk import calculate_position_size
from technical_score import (
    calculate_predictive_score,
    calculate_technical_score_4h,
    get_contradictory_factors,
)
from validation import run_validation_pipeline
from backtest import (
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    compare_strategies,
    macd_crossover_signals,
    evaluate_strategy,
    predictive_score_signals,
    run_backtest_with_risk,
    calculate_trade_metrics,
    calculate_max_drawdown,
)
from charts import plot_price_with_trades, plot_equity_curve

ALL_SYMBOLS = [
    "NVDA",
    "TSLA",
    "AAPL",
    "MSFT",
    "AMD",
    "META",
    "JPM",
    "XOM",
    "KO",
    "GC=F",
    "EURUSD=X",
    "ETH-USD",
]


def run_cross_asset_validation():
    risk_config = RiskConfig(
        risk_per_trade=0.05, atr_multiplier=3.5, max_allocation=1.0, atr_period=14
    )

    results = []

    for symbol in ALL_SYMBOLS:
        print(f"Processing symbol: {symbol}")
        stock_data = get_stock_data(symbol, period="5y")

        score_signals = predictive_score_signals(stock_data, min_score=0.75)
        score_no_risk = evaluate_strategy(stock_data, score_signals)

        trade, trade_summary, final_capital, equity_curve, stop_history = (
            run_backtest_with_risk(stock_data, score_signals, risk_config)
        )

        portfolio_return = ((final_capital - 10000) / 10000) * 100
        total_trades, win_rate, _, _, _, _, _, _, _ = calculate_trade_metrics(
            trade_summary
        )
        max_dd_risk, _, _, _, _ = calculate_max_drawdown(equity_curve, stock_data)

        calmar_ratio = portfolio_return / abs(max_dd_risk) if max_dd_risk != 0 else 0.0

        sharpe_ratio = calculate_sharpe_ratio(equity_curve, risk_free_rate=0.04)

        sortino_ratio = calculate_sortino_ratio(equity_curve, risk_free_rate=0.04)

        results.append(
            {
                "symbol": symbol,
                "ret_no_risk": score_no_risk["strategy_return"],
                "dd_no_risk": score_no_risk["max_drawdown"],
                "ret_with_risk": portfolio_return,
                "dd_with_risk": max_dd_risk,
                "total_trades": total_trades,
                "win_rate": win_rate,
                "calmar_risk": calmar_ratio,
                "sharpe_ratio": sharpe_ratio,
                "sortino_ratio": sortino_ratio,
            }
        )

    print("\n=== CROSS-ASSET RISK VALIDATION (5Y) ===")
    print(
        f"{'Symbol':<10} {'Ret No Risk':<15} {'DD No Risk':<15} {'Ret With Risk':<15} {'DD With Risk':<15} {'Total Trades':<15} {'Win Rate':<15} {'Calmar Ratio':<15} {'Sharpe Ratio':<15} {'Sortino Ratio':<15}"
    )
    print("-" * 120)

    for result in results:
        print(
            f"{result['symbol']:<10} {result['ret_no_risk']:<15.2f} {result['dd_no_risk']:<15.2f} {result['ret_with_risk']:<15.2f} {result['dd_with_risk']:<15.2f} {result['total_trades']:<15} {result['win_rate']:<15.2f} {result['calmar_risk']:<15.2f} {result['sharpe_ratio']:<15.2f} {result['sortino_ratio']:<15.2f}"
        )

    stock_data = get_stock_data("NVDA", period="5y")

    score_signals = predictive_score_signals(stock_data, min_score=0.75)

    trades, trade_summary, final_capital, equity_curve, stop_history = (
        run_backtest_with_risk(stock_data, score_signals, risk_config)
    )

    plot_price_with_trades(stock_data, trades, stop_history, "NVDA")
    plot_equity_curve(equity_curve, stock_data, "NVDA")


def validation_summary(symbols, period="5y"):
    results, summary = run_validation_pipeline(symbols, period)

    print("\n=== VALIDATION SUMMARY ===")
    print(
        f"{'Symbol':<10}  {'In-Sample Baseline Count':<20} {'Out-of-Sample Baseline Count':<20} {'In-Sample Bullish Count':<20} {'Out-of-Sample Bullish Count':<20} {'Stable':<10}"
    )
    for result in results:
        print(
            f"{result.symbol:<10} {result.baseline_in_sample.count:<20} {result.baseline_out_of_sample.count:<20} {result.bullish_in_sample.count:<20} {result.bullish_out_of_sample.count:<20} {'Yes' if result.is_stable else 'No':<10}"
        )


def get_daily_decision(symbol: str, capital: float = 10000.0):
    # 1. Carregar dados recentes do ativo
    stock_data = get_stock_data(symbol, period="5y")
    if "ATR" not in stock_data.columns:
        stock_data["ATR"] = calculate_atr(stock_data, window=14)

    # Configuração de Risco Padrão (5% Risco / 3.5x ATR)
    risk_cfg = RiskConfig(
        risk_per_trade=0.05, atr_multiplier=3.5, max_allocation=1.0, atr_period=14
    )

    technical_score_4h = calculate_technical_score_4h(symbol)
    # 2. Obter Sinais e Métricas Históricas
    score_sigs = predictive_score_signals(stock_data, min_score=0.75)
    trades, trade_summary, final_capital, equity_curve, _ = run_backtest_with_risk(
        stock_data, score_sigs, risk_cfg
    )

    portfolio_return = ((final_capital - capital) / capital) * 100.0
    max_dd, _, _, _, _ = calculate_max_drawdown(equity_curve, stock_data)
    sortino = calculate_sortino_ratio(equity_curve, risk_free_rate=0.04)
    calmar = (portfolio_return / abs(max_dd)) if max_dd != 0 else 0.0

    # 3. Ler dados da última barra disponível (Hoje / Último Fecho)
    last_row = stock_data.iloc[-1]
    last_date = last_row["Date"] if "Date" in last_row else stock_data.index[-1]
    last_close = last_row["Close"]
    last_atr = last_row["ATR"]
    latest_signal = score_sigs.iloc[-1]
    rsi = last_row["RSI"] if "RSI" in last_row else None

    # 4. Calcular Dimensionamento de Posição e Stop
    shares_to_buy, suggested_stop = calculate_position_size(
        portfolio_capital=capital,
        entry_price=last_close,
        atr=last_atr,
        risk_config=risk_cfg,
    )

    position_cost = shares_to_buy * last_close
    portfolio_allocation_pct = (position_cost / capital) * 100.0

    fundamental_metrics = get_stock_fundamental_metrics(symbol)

    rel_score_fmt = (
        f"{fundamental_metrics.relative_score:+.2f}"
        if fundamental_metrics.relative_score is not None
        else "N/A"
    )
    rev_growth_fmt = (
        f"{fundamental_metrics.revenueGrowth:.1%}"
        if fundamental_metrics.revenueGrowth is not None
        else "N/A"
    )
    earn_growth_fmt = (
        f"{fundamental_metrics.earningsGrowth:.1%}"
        if fundamental_metrics.earningsGrowth is not None
        else "N/A"
    )
    avg_growth_fmt = (
        f"{fundamental_metrics.growth:.1%}"
        if fundamental_metrics.growth is not None
        else "N/A"
    )
    eps_fmt = (
        f"${fundamental_metrics.eps:.2f}"
        if fundamental_metrics.eps is not None
        else "N/A"
    )
    feps_fmt = (
        f"${fundamental_metrics.feps:.2f}"
        if fundamental_metrics.feps is not None
        else "N/A"
    )
    roa_fmt = (
        f"{fundamental_metrics.roa:.1%}"
        if fundamental_metrics.roa is not None
        else "N/A"
    )
    roe_fmt = (
        f"{fundamental_metrics.returnOnEquity:.1%}"
        if fundamental_metrics.returnOnEquity is not None
        else "N/A"
    )
    state_4h_str = technical_score_4h if technical_score_4h is not None else "N/A"

    contradictions = get_contradictory_factors(
        stock_data, latest_signal, technical_score_4h, fundamental_metrics
    )

    # 5. Imprimir Painel de Suporte à Decisão no Terminal
    print("=" * 65)
    print(f"      PAINEL DE DECISÃO OPERACIONAL: {symbol}")
    print(f"      Data de Análise: {last_date}")
    print("=" * 65)
    print(f"Preço de Fecho Atual : ${last_close:.2f}")
    print(f"ATR (14 dias)        : ${last_atr:.2f}")
    print(f"Score Técnico 4H     : {technical_score_4h}")
    print(f"Estado Intradiário 4H: {state_4h_str}")
    print("-" * 65)
    print("FATORES CONTRADITÓRIOS E ALERTAS DE DIVERGÊNCIA:")
    if contradictions:
        for factor in contradictions:
            print(f"  ⚠️  [ALERTA] {factor}")
    else:
        print("  ✅ [OK] Nenhum conflito detetado entre indicadores e timeframes.")

    # Decisão de Ação (Árvore de Decisão Reestruturada)
    if latest_signal == "BUY":
        if fundamental_metrics.relative_score is None:
            print(
                f"SINAL DO SISTEMA     : [ WAIT / FUNDAMENTAL INSUFFICIENT ] (Confiança >= 0.75)"
            )
            print(
                f"Ação Recomendada     : AGUARDAR - MÉTRICAS FUNDAMENTAIS INSUFICIENTES"
            )
            print(
                f"Motivo da Cautela    : Métricas Fundamentais insuficientes para avaliação. Aguardar mais dados ou informações."
            )
        elif (
            fundamental_metrics.relative_score is not None
            and fundamental_metrics.relative_score > -0.20
            and fundamental_metrics.relative_score < 0
        ):
            print(f"SINAL DO SISTEMA     : [ WAIT / CAUTIOUS ] (Confiança >= 0.75)")
            print(
                f"Ação Recomendada     : ESPERAR - AVALIAR FUNDAMENTOS ANTES DE ABRIR POSIÇÃO"
            )
            print(
                f"Motivo da Cautela    : Métricas Fundamentais medianas. Avaliar risco antes de entrar."
            )
            print(
                f"Score Relativo       : {rel_score_fmt} | Decisão: {fundamental_metrics.decision}"
            )
        elif (
            fundamental_metrics.relative_score is not None
            and fundamental_metrics.relative_score <= -0.20
            and fundamental_metrics.relative_score >= -0.50
        ):
            print(f"SINAL DO SISTEMA     : [ WATCH / BLOQUEADO ] (Confiança >= 0.75)")
            print(f"Ação Recomendada     : NÃO ABRIR POSIÇÃO - AVALIAR FUNDAMENTOS")
            print(
                f"Motivo da Cautela    : Métricas Fundamentais fracas. Vale a pena aguardar uma oportunidade melhor."
            )
            print(
                f"Score Relativo       : {rel_score_fmt} | Decisão: {fundamental_metrics.decision}"
            )
        elif (
            fundamental_metrics.relative_score is not None
            and fundamental_metrics.relative_score < -0.50
            or fundamental_metrics.growth_level == "OVERVALUED WITHOUT SUPPORT"
        ):
            print(f"SINAL DO SISTEMA     : [ AVOID ] (Confiança >= 0.75)")
            print(f"Ação Recomendada     : NÃO ABRIR POSIÇÃO - AVALIAR FUNDAMENTOS")
            print(
                f"Motivo da Cautela    : Métricas Fundamentais extremamente fracas. É melhor ignorar este ativo por enquanto."
            )
            print(
                f"Score Relativo       : {rel_score_fmt} | Decisão: {fundamental_metrics.decision}"
            )
        else:
            if technical_score_4h == "Bullish":
                if rsi is not None and rsi > 75:
                    print(
                        f"SINAL DO SISTEMA     : [ REDUCE / COMPRA COM RESALVA OU TOMAR LUCROS ] (Confiança >= 0.75)"
                    )
                    print(
                        f"Ação Recomendada     : ABRIR POSIÇÃO COM CAUTELA (1D e 4H em Alta, mas RSI > 75)"
                    )
                    print(
                        f"Motivo da Cautela    : RSI elevado ({rsi:.1f}), indicando possível sobrecompra. Avaliar risco antes de entrar."
                    )
                    print(
                        f"Stop-Loss Inicial    : ${suggested_stop:.2f} (-{risk_cfg.atr_multiplier}x ATR)"
                    )
                    print(
                        f"Tamanho da Posição   : {shares_to_buy/2} ações (~${position_cost/2:.2f} | {portfolio_allocation_pct/2:.1f}% do capital)"
                    )
                    print(
                        f"Se tiver posição aberta, considere reduzir 50% da posição para tomar lucros devido ao RSI elevado."
                    )
                else:
                    print(
                        f"SINAL DO SISTEMA     : [ BUY / COMPRA ] (Confiança >= 0.75)"
                    )
                    print(
                        f"Ação Recomendada     : ABRIR NOVA POSIÇÃO (COMPRA CONFIRMADA em 1D + 4H)"
                    )

                    print(
                        f"Stop-Loss Inicial    : ${suggested_stop:.2f} (-{risk_cfg.atr_multiplier}x ATR)"
                    )
                    print(
                        f"Tamanho da Posição   : {shares_to_buy} ações (~${position_cost:.2f} | {portfolio_allocation_pct:.1f}% do capital)"
                    )
            elif technical_score_4h == "Bearish":
                if rsi is not None and rsi > 75:
                    print(
                        f"SINAL DO SISTEMA     : [ WAIT / PULLBACK NECESSÁRIO ] (Confiança >= 0.75)"
                    )
                    print(
                        f"Ação Recomendada     : AGUARDAR PULLBACK (1D em Alta, mas 4H em Recuo e RSI > 75)"
                    )
                    print(
                        f"Motivo da Cautela    : RSI elevado ({rsi:.1f}), indicando possível sobrecompra. Aguardar exaustão do recuo antes de entrar."
                    )
                else:
                    print(
                        f"SINAL DO SISTEMA     : [ WAIT / PULLBACK ] (Confiança >= 0.75)"
                    )
                    print(
                        f"Ação Recomendada     : AGUARDAR PULLBACK (1D em Alta, mas 4H em Recuo)"
                    )
                    print(
                        f"Motivo da Cautela    : 4H em Tendência de Baixa (Aguardar Exaustão do Recuo)"
                    )
            elif technical_score_4h == "Neutral":
                print(f"SINAL DO SISTEMA     : [ WAIT / AGUARDAR ] (Confiança >= 0.75)")
                print(
                    f"Ação Recomendada     : AGUARDAR CONFIRMAÇÃO (1D em Alta, mas 4H Neutro)"
                )
                print(f"Motivo da Cautela    : 4H sem Direção Definida")
            else:
                print(f"SINAL DO SISTEMA     : [ BUY / ATENÇÃO ] (Confiança >= 0.75)")
                print(
                    f"Ação Recomendada     : ABRIR POSIÇÃO COM CAUTELA (Dados de 4H Indisponíveis)"
                )
                print(
                    f"Stop-Loss Inicial    : ${suggested_stop:.2f} (-{risk_cfg.atr_multiplier}x ATR)"
                )

    elif latest_signal == "SELL":
        print(f"SINAL DO SISTEMA     : [ SELL / VENDA ] (Confiança >= 0.75)")
        print(f"Ação Recomendada     : FECHAR POSIÇÃO EXISTENTE")
    else:
        print(f"SINAL DO SISTEMA     : [ HOLD / NEUTRO ]")
        print(f"Ação Recomendada     : MANTER EM CAIXA / AGUARDAR OPORTUNIDADE")

    print("-" * 65)
    print("ANÁLISE FUNDAMENTALISTA DO ATIVO:")
    print(f"  Decisão          : {fundamental_metrics.decision}")
    print(f"  Score Relativo   : {rel_score_fmt}")
    print(
        f"  Critérios Válidos: {fundamental_metrics.total_criteria_evaluated} / {len(DEFAULT_THRESHOLDS)}"
    )

    print("-" * 65)
    print("CRESCIMENTO DO ATIVO:")
    print(f"  Crescimento de Receita: {rev_growth_fmt}")
    print(f"  Crescimento de Lucro  : {earn_growth_fmt}")
    print(f"  Crescimento Médio     : {avg_growth_fmt}")
    print(f"  DIAGNÓSTICO           : {fundamental_metrics.growth_level}")

    print("-" * 65)
    print("MÉTRICAS DE LUCRO E EFICIÊNCIA DO ATIVO:")
    print(f"  EPS (Trailing)  : {eps_fmt}")
    print(f"  EPS (Forward)   : {feps_fmt}")
    print(f"  ROA (Ativos)    : {roa_fmt}")
    print(f"  ROE (Patrimônio): {roe_fmt}")

    print("-" * 65)
    print("MÉTRICAS HISTÓRICAS DE QUALIDADE DO ATIVO (5 Anos):")
    print(
        f"  Sortino Ratio      : {sortino:.2f} ({'Excelente' if sortino > 1.0 else 'Atenção ao Risco'})"
    )
    print(f"  Calmar Ratio       : {calmar:.2f}")
    print(f"  Max Drawdown Risco : {max_dd:.2f}%")
    print("=" * 65)


if __name__ == "__main__":
    validation_summary(ALL_SYMBOLS, period="5y")
