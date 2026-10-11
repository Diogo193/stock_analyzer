from typing import List, Tuple

from domain import Trade, RiskConfig
from indicators import (
    calculate_atr,
    calculate_ema,
    calculate_macd,
    calculate_sma,
    calculate_rsi,
)
import pandas as pd
from risk import calculate_position_size, update_trailing_stop, check_stop_trigger
from technical_score import calculate_predictive_score
import numpy as np


# region Signal Generation
def detect_crossover(fast, slow):
    bullish = (fast > slow) & (fast.shift(1) <= slow.shift(1))
    bearish = (fast < slow) & (fast.shift(1) >= slow.shift(1))
    signals = pd.Series(index=slow.index, dtype="object")
    signals[bullish] = "BUY"
    signals[bearish] = "SELL"
    signals[~(bullish | bearish)] = "HOLD"

    return signals


def sma_crossover_signals(stock_data):
    sma_20 = calculate_sma(stock_data["Close"], window=20)
    sma_50 = calculate_sma(stock_data["Close"], window=50)

    signals = detect_crossover(sma_20, sma_50)

    return signals


def ema_crossover_signals(stock_data):
    ema_20 = calculate_ema(stock_data["Close"], window=20)
    ema_50 = calculate_ema(stock_data["Close"], window=50)

    signals = detect_crossover(ema_20, ema_50)

    return signals


def macd_crossover_signals(stock_data):
    macd_data = calculate_macd(stock_data["Close"], 12, 26, 9)
    macd_line = macd_data["MACD"]
    signal_line = macd_data["Signal"]

    signals = detect_crossover(macd_line, signal_line)

    return signals


def rsi_signals(stock_data):
    rsi = calculate_rsi(stock_data["Close"], window=14)

    signals = pd.Series(index=rsi.index, dtype="object")
    signals[rsi < 30] = "BUY"
    signals[rsi > 70] = "SELL"
    signals[(rsi >= 30) & (rsi <= 70)] = "HOLD"

    return signals


def ema_macd_crossover_signals(stock_data):
    ema_20 = calculate_ema(stock_data["Close"], window=20)
    ema_50 = calculate_ema(stock_data["Close"], window=50)
    macd_data = calculate_macd(stock_data["Close"], 12, 26, 9)
    macd_line = macd_data["MACD"]
    signal_line = macd_data["Signal"]

    ema_bullish = ema_20 > ema_50
    macd_signals = detect_crossover(macd_line, signal_line)
    signals = pd.Series("HOLD", index=stock_data.index, dtype="object")

    buy_condition = (ema_bullish) & (macd_signals == "BUY")
    sell_condition = macd_signals == "SELL"

    signals[buy_condition] = "BUY"
    signals[sell_condition] = "SELL"

    return signals


def ema_rsi_signals(stock_data):
    ema_20 = calculate_ema(stock_data["Close"], window=20)
    ema_50 = calculate_ema(stock_data["Close"], window=50)
    rsi = calculate_rsi(stock_data["Close"], window=14)

    ema_bullish = ema_20 > ema_50
    buy_condition = (ema_bullish) & (rsi < 30)
    sell_condition = rsi > 70

    signals = pd.Series("HOLD", index=stock_data.index, dtype="object")

    signals[buy_condition] = "BUY"
    signals[sell_condition] = "SELL"

    return signals


# endregion


# region Backtest Engine
def run_backtest(stock_data, signals, commission_rate=0.0, slippage_rate=0.0):
    trades = []
    position = False

    for i in range(len(signals) - 1):

        signal = signals.iloc[i]
        if signal == "BUY" and not position:
            entry_price = (
                stock_data.loc[stock_data.index[i + 1], "Open"]
                * (1 + slippage_rate)
                * (1 + commission_rate)
            )
            new_trade = Trade(
                entry_date=stock_data["Date"].iloc[i + 1],
                exit_date=None,
                entry_type="Buy",
                exit_type=None,
                entry_price=entry_price,
                exit_price=None,
                absolute_profit_loss=None,
                return_percentage=None,
            )
            position = True
            trades.append(new_trade)

        elif signal == "SELL" and position:
            exit_price = (
                stock_data.loc[stock_data.index[i + 1], "Open"]
                * (1 - slippage_rate)
                * (1 - commission_rate)
            )
            trades[-1].close_trade(
                exit_date=stock_data["Date"].iloc[i + 1],
                exit_type="Sell",
                exit_price=exit_price,
            )

            position = False

    if position:
        exit_price = (
            stock_data["Close"].iloc[-1] * (1 - slippage_rate) * (1 - commission_rate)
        )
        trades[-1].close_trade(
            exit_date=stock_data["Date"].iloc[-1],
            exit_type="END",
            exit_price=exit_price,
        )

    trade_summary = []
    for trade in trades:
        if trade.exit_type in ("Sell", "END"):
            trade_summary.append(trade)
    return trades, trade_summary


def run_backtest_with_risk(
    stock_data: pd.DataFrame,
    signals: pd.Series,
    risk_config: RiskConfig,
    initial_capital: float = 10000.0,
    commission_rate: float = 0.001,
    slippage_rate: float = 0.001,
) -> Tuple[List[Trade], List[Trade], float]:
    # 1. Garantir o cálculo do ATR na DataFrame de dados
    df = stock_data.copy()
    if "ATR" not in df.columns:
        df["ATR"] = calculate_atr(df, window=risk_config.atr_period)

    trades: List[Trade] = []
    portfolio_capital = initial_capital

    position = False
    current_shares = 0
    current_stop_price = 0.0
    highest_price_seen = 0.0
    active_trade: Trade = None

    equity_curve: List[float] = [initial_capital]

    stop_history: List[float] = [np.nan]

    for i in range(1, len(df)):
        current_bar = df.iloc[i]
        prev_signal = signals.iloc[i - 1]

        current_date = current_bar["Date"]
        open_p = current_bar["Open"]
        high_p = current_bar["High"]
        low_p = current_bar["Low"]
        close_p = current_bar["Close"]
        atr_v = current_bar["ATR"]

        if position:
            # --- PASSO A: Testar Disparo de Stop-Loss ---
            stop_hit, exit_price = check_stop_trigger(
                current_open=open_p,
                current_low=low_p,
                current_stop_price=current_stop_price,
                slippage_rate=slippage_rate,
            )

            if stop_hit:
                # Ajustar por comissão de venda
                final_exit_price = exit_price * (1 - commission_rate)
                active_trade.close_trade(
                    exit_date=current_date,
                    exit_type="Stop-Loss",
                    exit_price=final_exit_price,
                )

                # Atualizar capital com base na P/L absoluta em ações
                portfolio_capital += active_trade.absolute_profit_loss
                stop_history.append(current_stop_price)
                position = False
                continue  # Passa à próxima barra

            # --- PASSO B: Testar Sinal de Saída do Predictive Score ---
            elif prev_signal == "SELL":
                exit_price = open_p * (1 - slippage_rate) * (1 - commission_rate)
                active_trade.close_trade(
                    exit_date=current_date,
                    exit_type="Signal Sell",
                    exit_price=exit_price,
                )
                portfolio_capital += active_trade.absolute_profit_loss
                stop_history.append(current_stop_price)
                position = False
                continue

            # --- PASSO C: Atualizar Trailing Stop (Se mantido) ---
            else:
                highest_price_seen, current_stop_price = update_trailing_stop(
                    current_high=high_p,
                    current_atr=atr_v,
                    previous_highest_price=highest_price_seen,
                    previous_stop_price=current_stop_price,
                    atr_multiplier=risk_config.atr_multiplier,
                )
                stop_history.append(current_stop_price)
        else:
            # --- PASSO D: Entrada numa Nova Posição ---
            if prev_signal == "BUY" and not pd.isna(atr_v):
                entry_price = open_p * (1 + slippage_rate) * (1 + commission_rate)

                shares, initial_stop = calculate_position_size(
                    portfolio_capital=portfolio_capital,
                    entry_price=entry_price,
                    atr=atr_v,
                    risk_config=risk_config,
                )

                if shares > 0:
                    position = True
                    current_shares = shares
                    current_stop_price = initial_stop
                    highest_price_seen = (
                        high_p  # O pico inicial é o High da própria barra de entrada
                    )

                    active_trade = Trade(
                        entry_date=current_date,
                        entry_type="Buy",
                        entry_price=entry_price,
                        initial_stop_price=initial_stop,
                        highest_price_seen=highest_price_seen,
                        position_shares=shares,
                    )
                    trades.append(active_trade)
                    stop_history.append(current_stop_price)
                else:
                    stop_history.append(np.nan)
            else:
                stop_history.append(np.nan)

        if position:
            unrealized_pl = active_trade.position_shares * (
                close_p - active_trade.entry_price
            )
            daily_equity = portfolio_capital + unrealized_pl
        else:
            daily_equity = portfolio_capital
        equity_curve.append(daily_equity)
    # Tratamento de posição aberta no fim do backtest
    if position:
        final_close = df["Close"].iloc[-1] * (1 - slippage_rate) * (1 - commission_rate)
        active_trade.close_trade(
            exit_date=df["Date"].iloc[-1], exit_type="END", exit_price=final_close
        )
        portfolio_capital += active_trade.absolute_profit_loss

    trade_summary = [t for t in trades if t.exit_type is not None]
    return trades, trade_summary, portfolio_capital, equity_curve, stop_history


def predictive_score_signals(stock_data, min_score=0.75):
    # 1. Calcular o Score Preditivo
    score_df = calculate_predictive_score(stock_data)
    scores = score_df["Score"]

    # 2. Criar sinais com base no Score
    signals = pd.Series("HOLD", index=stock_data.index, dtype="object")

    signals[scores >= min_score] = "BUY"

    signals[scores < 0.25] = "SELL"

    return signals


# endregion


# region Trade Metrics


def calculate_trade_metrics(trade_summary):

    total_trades = len(trade_summary)

    returns = [
        trade.return_percentage
        for trade in trade_summary
        if trade.return_percentage is not None
    ]
    strategy_growth = 1.0

    for return_pct in returns:
        strategy_growth *= 1 + return_pct / 100

    strategy_return = (strategy_growth - 1) * 100

    mean_return = sum(returns) / len(returns) if returns else 0
    highest_return = max(returns) if returns else 0
    lowest_return = min(returns) if returns else 0

    winning_trades = [return_pct for return_pct in returns if return_pct > 0]
    wins = len(winning_trades)
    win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0

    losing_trades = [return_pct for return_pct in returns if return_pct < 0]

    average_winning_trade = (
        (sum(winning_trades) / len(winning_trades)) if winning_trades else 0
    )
    average_losing_trade = (
        (sum(losing_trades) / len(losing_trades)) if losing_trades else 0
    )

    sorted_returns = sorted(returns)
    n = len(sorted_returns)
    if n == 0:
        median_return = 0
    elif n % 2 == 1:
        median_return = sorted_returns[n // 2]
    else:
        median_return = (sorted_returns[n // 2 - 1] + sorted_returns[n // 2]) / 2

    return (
        total_trades,
        win_rate,
        strategy_return,
        mean_return,
        highest_return,
        lowest_return,
        average_winning_trade,
        average_losing_trade,
        median_return,
    )


def calculate_average_trade_duration(trade_summary):
    total_duration = sum(
        (trade.exit_date - trade.entry_date).days
        for trade in trade_summary
        if trade.exit_date and trade.entry_date
    )
    return total_duration / len(trade_summary) if trade_summary else 0


# endregion


# region Equity and Risk


def calculate_equity_curve(stock_data, trade_summary):
    capital = 1.0
    position = False
    entry_price = None
    equity_curve = []

    for i in range(len(stock_data)):
        current_date = stock_data["Date"].iloc[i]

        for trade in trade_summary:
            if current_date == trade.entry_date and trade.entry_type == "Buy":
                position = True
                entry_price = trade.entry_price

            elif current_date == trade.exit_date and trade.exit_type in ("Sell", "END"):
                position = False
                exit_price = trade.exit_price
                capital *= 1 + (exit_price - entry_price) / entry_price
                entry_price = None

        if position:
            current_equity = capital * (stock_data["Close"].iloc[i] / entry_price)
        else:
            current_equity = capital

        equity_curve.append(current_equity)

    assert len(equity_curve) == len(stock_data)

    return equity_curve


def calculate_sharpe_ratio(equity_curve, risk_free_rate=0.04):
    if len(equity_curve) < 2:
        return 0.0

    returns = [
        (equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1]
        for i in range(1, len(equity_curve))
    ]

    daily_rf = risk_free_rate / 252

    excess_returns = [r - daily_rf for r in returns]

    mean_excess_return = sum(excess_returns) / len(excess_returns)
    std_dev = (
        sum((r - mean_excess_return) ** 2 for r in excess_returns) / len(excess_returns)
    ) ** 0.5

    if std_dev == 0:
        return 0.0

    sharpe_ratio = mean_excess_return / std_dev * (252**0.5)

    return sharpe_ratio


def calculate_sortino_ratio(equity_curve, risk_free_rate=0.04):
    if len(equity_curve) < 2:
        return 0.0

    returns = [
        (equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1]
        for i in range(1, len(equity_curve))
    ]

    daily_rf = risk_free_rate / 252
    excess_returns = [r - daily_rf for r in returns]

    mean_excess_return = sum(excess_returns) / len(excess_returns)

    negative_returns = [r for r in excess_returns if r < 0]
    downside_deviation = (
        (sum(r**2 for r in negative_returns) / len(returns)) ** 0.5 if returns else 0.0
    )

    if downside_deviation == 0:
        return 0.0

    sortino_ratio = mean_excess_return / downside_deviation * (252**0.5)

    return sortino_ratio


def calculate_max_drawdown(equity_curve, stock_data):
    if not equity_curve:
        return 0, None, None, None, None
    max_drawdown = 0
    peak = equity_curve[0]
    peak_index = 0
    max_drawdown_peak_index = 0
    max_drawdown_trough_index = 0

    for i, value in enumerate(equity_curve):
        if value > peak:
            peak = value
            peak_index = i
        drawdown = (value - peak) / peak
        if drawdown < max_drawdown:
            max_drawdown = drawdown
            max_drawdown_peak_index = peak_index
            max_drawdown_trough_index = i

    peak_date = stock_data["Date"].iloc[max_drawdown_peak_index]
    trough_date = stock_data["Date"].iloc[max_drawdown_trough_index]

    peak_equity = equity_curve[max_drawdown_peak_index]
    trough_equity = equity_curve[max_drawdown_trough_index]

    return max_drawdown * 100, peak_date, trough_date, peak_equity, trough_equity


# endregion


# region Benchmark


def calculate_buy_hold_return(stock_data, commission_rate=0.0, slippage_rate=0.0):
    first_close = (
        stock_data["Close"].iloc[0] * (1 + slippage_rate) * (1 + commission_rate)
    )
    last_close = (
        stock_data["Close"].iloc[-1] * (1 - slippage_rate) * (1 - commission_rate)
    )
    buy_hold_return = ((last_close - first_close) / first_close) * 100
    return buy_hold_return


def calculate_buy_hold_equity_curve(stock_data, commission_rate=0.0, slippage_rate=0.0):
    first_close = (
        stock_data["Close"].iloc[0] * (1 + slippage_rate) * (1 + commission_rate)
    )
    equity_curve_bah = (stock_data["Close"] / first_close).tolist()
    return equity_curve_bah


def compare_strategy_to_benchmark(
    strategy_return, buy_hold_return, strategy_drawdown, buy_hold_drawdown
):
    excess_return = strategy_return - buy_hold_return
    drawdown_improvement = abs(buy_hold_drawdown) - abs(strategy_drawdown)

    return excess_return, drawdown_improvement


# endregion


# region Strategy Evaluation


def evaluate_strategy(stock_data, signals, commission_rate=0.0, slippage_rate=0.0):
    trades, trade_summary = run_backtest(
        stock_data, signals, commission_rate, slippage_rate
    )
    (
        total_trades,
        win_rate,
        strategy_return,
        mean_return,
        highest_return,
        lowest_return,
        average_winning_trade,
        average_losing_trade,
        median_return,
    ) = calculate_trade_metrics(trade_summary)

    average_trade_duration = calculate_average_trade_duration(trade_summary)

    equity_curve = calculate_equity_curve(stock_data, trade_summary)

    max_drawdown, peak_date, trough_date, peak_equity, trough_equity = (
        calculate_max_drawdown(equity_curve, stock_data)
    )

    return {
        "trades": trades,
        "trade_summary": trade_summary,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "strategy_return": strategy_return,
        "mean_return": mean_return,
        "median_return": median_return,
        "highest_return": highest_return,
        "lowest_return": lowest_return,
        "average_winning_trade": average_winning_trade,
        "average_losing_trade": average_losing_trade,
        "average_trade_duration": average_trade_duration,
        "max_drawdown": max_drawdown,
        "equity_curve": equity_curve,
    }


def compare_strategies(stock_data):

    sma_signals = sma_crossover_signals(stock_data)
    ema_signals = ema_crossover_signals(stock_data)
    macd_signals = macd_crossover_signals(stock_data)
    rsi_signals_data = rsi_signals(stock_data)
    combination_ema_macd_signals = ema_macd_crossover_signals(stock_data)
    combination_ema_rsi_signals = ema_rsi_signals(stock_data)

    sma_results = evaluate_strategy(stock_data, sma_signals, 0.001, 0.001)
    ema_results = evaluate_strategy(stock_data, ema_signals, 0.001, 0.001)
    macd_results = evaluate_strategy(stock_data, macd_signals, 0.001, 0.001)
    rsi_results = evaluate_strategy(stock_data, rsi_signals_data, 0.001, 0.001)
    combination_ema_macd_results = evaluate_strategy(
        stock_data, combination_ema_macd_signals, 0.001, 0.001
    )
    combination_ema_rsi_results = evaluate_strategy(
        stock_data, ema_rsi_signals(stock_data), 0.001, 0.001
    )

    buy_hold_return = calculate_buy_hold_return(stock_data, 0.001, 0.001)
    equity_curve_bah = calculate_buy_hold_equity_curve(stock_data, 0.001, 0.001)
    max_drawdown_bah, _, _, _, _ = calculate_max_drawdown(equity_curve_bah, stock_data)

    return {
        "sma_results": sma_results,
        "ema_results": ema_results,
        "macd_results": macd_results,
        "rsi_results": rsi_results,
        "combination_ema_macd_results": combination_ema_macd_results,
        "combination_ema_rsi_results": combination_ema_rsi_results,
        "buy_hold_return": buy_hold_return,
        "sma_drawdown": sma_results["max_drawdown"],
        "ema_drawdown": ema_results["max_drawdown"],
        "macd_drawdown": macd_results["max_drawdown"],
        "rsi_drawdown": rsi_results["max_drawdown"],
        "combination_ema_macd_drawdown": combination_ema_macd_results["max_drawdown"],
        "combination_ema_rsi_drawdown": combination_ema_rsi_results["max_drawdown"],
        "max_drawdown_bah": max_drawdown_bah,
        "sma_trades": sma_results["total_trades"],
        "ema_trades": ema_results["total_trades"],
        "macd_trades": macd_results["total_trades"],
        "rsi_trades": rsi_results["total_trades"],
        "combination_ema_macd_trades": combination_ema_macd_results["total_trades"],
        "combination_ema_rsi_trades": combination_ema_rsi_results["total_trades"],
    }


# endregion


# region Reporting


def print_trade_report(trade_summary, strategy_name):
    print(f"\n{strategy_name} Trade Report:")
    print(
        f"{'Number':<8} {'Buy Date':<12} {'Sell Date':<12} {'Buy':<10} {'Exit Reason':<50} {'Buy Price':<12} {'Sell Price':<12} {'P/L':10} {'Return %':10}"
    )

    for number, trade in enumerate(trade_summary, start=1):
        (
            buy_date,
            sell_date,
            buy_type,
            exit_type,
            effective_buy_price,
            effective_sell_price,
            profit_loss,
            return_pct,
        ) = trade

        exit_reason = (
            f"Bearish {strategy_name} crossover"
            if exit_type == "Sell"
            else "End of backtest"
        )
        print(
            f"{number:<8} {buy_date.strftime('%Y-%m-%d'):<12} {sell_date.strftime('%Y-%m-%d'):<12} {buy_type:<10} {exit_reason:<50} {effective_buy_price:<12.2f} {effective_sell_price:<12.2f} {profit_loss:+10.2f} {return_pct:+10.2f}%"
        )


def print_strategy_summary(
    symbol,
    total_trades,
    win_rate,
    strategy_return,
    mean_return,
    highest_return,
    lowest_return,
    average_winning_trade,
    average_losing_trade,
    median_return,
    average_trade_duration,
    buy_hold_return,
    max_drawdown,
    max_drawdown_bah,
    excess_return,
    drawdown_improvement,
):
    print(f"\nStrategy Summary: {symbol}")
    print(f"Performance Metrics:")
    print(f"Strategy Return: {strategy_return:+.2f}%")
    print(f"Buy and Hold Return: {buy_hold_return:+.2f}%")
    print(f"Excess Return: {excess_return:+.2f}%")

    print(f"\nRisk Metrics:")
    print(f"Maximum Drawdown: {max_drawdown:.2f}%")
    print(f"Buy and Hold Maximum Drawdown: {max_drawdown_bah:.2f}%")
    print(f"Drawdown Improvement: {drawdown_improvement:+.2f}%")

    print(f"\nTrades Summary:")
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Average Trade: {mean_return:+.2f}%")
    print(f"Median Trade: {median_return:+.2f}%")
    print(f"Average Winner: {average_winning_trade:+.2f}%")
    print(f"Average Loser: {average_losing_trade:+.2f}%")
    print(f"Best Trade: {highest_return:+.2f}%")
    print(f"Worst Trade: {lowest_return:+.2f}%")
    print(f"Average Trade Duration: {average_trade_duration:.2f} days")


def print_comparison_table(results):
    print("\nComparison Table:")
    print(
        f"{'Ticker':<10} {'SMA Return':<12} {'EMA Return':<12} {'MACD Return':<12} {'RSI Return':<12} {'Combination EMA MACD Return':<12} {'Combination EMA RSI Return':<12} {'Buy & Hold':<12} {'SMA Drawdown':<15} {'EMA Drawdown':<15} {'MACD Drawdown':<15} {'RSI Drawdown':<15} {'Combination EMA MACD Drawdown':<20} {'Combination EMA RSI Drawdown':<20} {'SMA Trades':<12} {'EMA Trades':<12} {'MACD Trades':<12} {'RSI Trades':<12} {'Combination EMA MACD Trades':<20} {'Combination EMA RSI Trades':<20}"
    )

    for result in results:
        ticker = result["ticker"]
        sma_return = result["sma_results"]["strategy_return"]
        ema_return = result["ema_results"]["strategy_return"]
        macd_return = result["macd_results"]["strategy_return"]
        rsi_return = result["rsi_results"]["strategy_return"]
        combination_ema_macd_return = result["combination_ema_macd_results"][
            "strategy_return"
        ]
        combination_ema_rsi_return = result["combination_ema_rsi_results"][
            "strategy_return"
        ]
        buy_hold_return = result["buy_hold_return"]
        sma_drawdown = result["sma_results"]["max_drawdown"]
        ema_drawdown = result["ema_results"]["max_drawdown"]
        macd_drawdown = result["macd_results"]["max_drawdown"]
        rsi_drawdown = result["rsi_results"]["max_drawdown"]
        combination_ema_macd_drawdown = result["combination_ema_macd_results"][
            "max_drawdown"
        ]
        combination_ema_rsi_drawdown = result["combination_ema_rsi_results"][
            "max_drawdown"
        ]
        max_drawdown_bah = result["max_drawdown_bah"]
        sma_trades = result["sma_results"]["total_trades"]
        ema_trades = result["ema_results"]["total_trades"]
        macd_trades = result["macd_results"]["total_trades"]
        rsi_trades = result["rsi_results"]["total_trades"]
        combination_ema_macd_trades = result["combination_ema_macd_results"][
            "total_trades"
        ]
        combination_ema_rsi_trades = result["combination_ema_rsi_results"][
            "total_trades"
        ]

        print(
            f"{ticker:<10} "
            f"{sma_return:+.2f}%{'':<6} "
            f"{ema_return:+.2f}%{'':<6} "
            f"{macd_return:+.2f}%{'':<6} "
            f"{rsi_return:+.2f}%{'':<6} "
            f"{combination_ema_macd_return:+.2f}%{'':<6} "
            f"{combination_ema_rsi_return:+.2f}%{'':<6} "
            f"{buy_hold_return:+.2f}%{'':<6} "
            f"{sma_drawdown:.2f}%{'':<6} "
            f"{ema_drawdown:.2f}%{'':<6} "
            f"{macd_drawdown:.2f}%{'':<6} "
            f"{rsi_drawdown:.2f}%{'':<6} "
            f"{combination_ema_macd_drawdown:.2f}%{'':<6} "
            f"{combination_ema_rsi_drawdown:.2f}%{'':<6} "
            f"{max_drawdown_bah:.2f}%{'':<6} "
            f"{sma_trades:<12} "
            f"{ema_trades:<12} "
            f"{macd_trades:<12} "
            f"{rsi_trades:<12} "
            f"{combination_ema_macd_trades:<20} "
            f"{combination_ema_rsi_trades:<20}"
        )


# endregion
