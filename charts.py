import matplotlib.pyplot as plt
import pandas as pd
from domain import Trade
from indicators import calculate_macd, calculate_sma, calculate_ema, calculate_rsi


def plot_stock_price(stock_data, symbol):
    x = stock_data.index
    y = stock_data["Close"]

    plt.plot(x, y, label="Close Price", color="blue")
    sma_20 = calculate_sma(stock_data["Close"], window=20)
    plt.plot(x, sma_20, label="SMA 20", color="orange")
    ema_20 = calculate_ema(stock_data["Close"], window=20)
    plt.plot(x, ema_20, label="EMA 20", color="green")

    sma_50 = calculate_sma(stock_data["Close"], window=50)
    plt.plot(x, sma_50, label="SMA 50", color="red")
    ema_50 = calculate_ema(stock_data["Close"], window=50)
    plt.plot(x, ema_50, label="EMA 50", color="purple")

    plt.title(f"{symbol} Stock Price")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_rsi(stock_data, symbol):
    rsi_14 = calculate_rsi(stock_data["Close"], window=14)
    x = stock_data.index
    y = rsi_14

    plt.plot(x, y, label="RSI 14", color="blue")
    plt.axhline(70, color="red", linestyle="--", label="Overbought (70)")
    plt.axhline(30, color="green", linestyle="--", label="Oversold (30)")
    plt.ylim(0, 100)

    plt.title(f"{symbol} Relative Strength Index (RSI)")
    plt.xlabel("Date")
    plt.ylabel("RSI Value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_macd(stock_data, symbol):
    macd_data = calculate_macd(stock_data["Close"], 12, 26, 9)
    x = stock_data.index

    macd_line = macd_data["MACD"]
    signal_line = macd_data["Signal"]
    histogram = macd_data["Histogram"]

    plt.plot(x, macd_line, label="MACD Line", color="blue")
    plt.plot(x, signal_line, label="Signal Line", color="orange")
    plt.bar(x, histogram, label="Histogram", color="gray", alpha=0.5)
    plt.axhline(0, color="black", linestyle="--")

    plt.title(f"{symbol} Moving Average Convergence Divergence (MACD)")
    plt.xlabel("Date")
    plt.ylabel("MACD Value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_technical_dashboard(stock_data, symbol):
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    sma_20 = calculate_sma(stock_data["Close"], window=20)
    ema_20 = calculate_ema(stock_data["Close"], window=20)
    sma_50 = calculate_sma(stock_data["Close"], window=50)
    ema_50 = calculate_ema(stock_data["Close"], window=50)

    bullish_cross = (sma_20 > sma_50) & (sma_20.shift(1) <= sma_50.shift(1))
    bearish_cross = (sma_20 < sma_50) & (sma_20.shift(1) >= sma_50.shift(1))

    axs[0].plot(
        stock_data.index, stock_data["Close"], label="Close Price", color="blue"
    )
    axs[0].plot(stock_data.index, sma_20, label="SMA 20", color="orange")
    axs[0].plot(stock_data.index, ema_20, label="EMA 20", color="green")
    axs[0].plot(stock_data.index, sma_50, label="SMA 50", color="red")
    axs[0].plot(stock_data.index, ema_50, label="EMA 50", color="purple")
    axs[0].scatter(
        stock_data.index[bullish_cross],
        stock_data.loc[bullish_cross, "Close"],
        marker="^",
        color="green",
        label="Bullish Cross",
        s=100,
    )
    axs[0].scatter(
        stock_data.index[bearish_cross],
        stock_data.loc[bearish_cross, "Close"],
        marker="v",
        color="red",
        label="Bearish Cross",
        s=100,
    )

    axs[0].set_title(f"{symbol} Stock Price with SMA and EMA")
    axs[0].set_ylabel("Price")
    axs[0].grid(True)
    axs[0].legend()

    rsi_14 = calculate_rsi(stock_data["Close"], window=14)
    axs[1].plot(stock_data.index, rsi_14, label="RSI 14", color="blue")
    axs[1].axhline(70, color="red", linestyle="--", label="Overbought (70)")
    axs[1].axhline(30, color="green", linestyle="--", label="Oversold (30)")
    axs[1].set_title(f"{symbol} Relative Strength Index (RSI)")
    axs[1].set_ylabel("RSI Value")
    axs[1].set_ylim(0, 100)
    axs[1].grid(True)
    axs[1].legend()

    macd_data = calculate_macd(stock_data["Close"], 12, 26, 9)
    macd_line = macd_data["MACD"]
    signal_line = macd_data["Signal"]
    histogram = macd_data["Histogram"]

    macd_bullish_cross = (macd_line > signal_line) & (
        macd_line.shift(1) <= signal_line.shift(1)
    )
    macd_bearish_cross = (macd_line < signal_line) & (
        macd_line.shift(1) >= signal_line.shift(1)
    )

    axs[2].plot(stock_data.index, macd_line, label="MACD Line", color="blue")
    axs[2].plot(stock_data.index, signal_line, label="Signal Line", color="orange")
    axs[2].bar(stock_data.index, histogram, label="Histogram", color="gray", alpha=0.5)
    axs[2].scatter(
        stock_data.index[macd_bullish_cross],
        macd_line[macd_bullish_cross],
        marker="^",
        color="green",
        label="MACD Bullish Cross",
        s=100,
    )
    axs[2].scatter(
        stock_data.index[macd_bearish_cross],
        macd_line[macd_bearish_cross],
        marker="v",
        color="red",
        label="MACD Bearish Cross",
        s=100,
    )
    axs[2].axhline(0, color="black", linestyle="--")
    axs[2].set_title(f"{symbol} Moving Average Convergence Divergence (MACD)")
    axs[2].set_ylabel("MACD Value")
    axs[2].set_xlabel("Date")
    axs[2].grid(True)
    axs[2].legend()

    plt.tight_layout()
    plt.show()


def plot_price_with_trades(stock_data, trades, stop_history, symbol):
    # Linha de fecho do ativo ($Close$).
    # Triângulos verdes ($\Delta$) nos dias de entrada (BUY).
    # Triângulos vermelhos ($\nabla$) nos dias de saída (SELL / STOP).
    # Linha tracejada do Trailing Stop no decorrer das posições abertas.

    plt.figure(figsize=(14, 7))
    dates = stock_data["Date"] if "Date" in stock_data.columns else stock_data.index
    plt.plot(
        dates,
        stock_data["Close"],
        label="Close Price",
        color="royalblue",
        alpha=0.75,
        linewidth=1.5,
    )

    if len(stop_history) == len(dates):
        plt.plot(
            dates,
            stop_history,
            label="Trailing Stop (3.5x ATR)",
            color="orange",
            linestyle="--",
            linewidth=1.5,
        )

    entry_dates = [t.entry_date for t in trades]
    entry_prices = [t.entry_price for t in trades]
    exit_dates = [t.exit_date for t in trades]
    exit_prices = [t.exit_price for t in trades]

    plt.scatter(
        entry_dates,
        entry_prices,
        marker="^",
        color="green",
        s=120,
        label="Buy Entry",
        zorder=5,
    )
    plt.scatter(
        exit_dates,
        exit_prices,
        marker="v",
        color="red",
        label="Sell / Stop Exit",
        s=120,
        zorder=5,
    )

    plt.title(f"{symbol} - Execuções de Ordem e Estratégia de Risco", fontsize=14)
    plt.xlabel("Data", fontsize=11)
    plt.ylabel("Price", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()


def plot_equity_curve(equity_curve, stock_data, symbol, initial_capital=10000):
    # Curva de evolução dos $\$10,000$ da estratégia ao longo do tempo.
    # Curva de evolução dos $\$10,000$ na estratégia Buy & Hold para comparação direta.

    plt.figure(figsize=(14, 7))
    dates = stock_data["Date"] if "Date" in stock_data.columns else stock_data.index

    curve_dates = (
        dates[-len(equity_curve) :] if len(equity_curve) != len(dates) else dates
    )

    plt.plot(
        curve_dates,
        equity_curve,
        label="Estratégia Predictive Score (Com Risco)",
        color="forestgreen",
        linewidth=2.0,
    )

    buy_and_hold = (stock_data["Close"] / stock_data["Close"].iloc[0]) * initial_capital
    plt.plot(
        dates,
        buy_and_hold,
        label="Buy & Hold (100% Alocado)",
        color="grey",
        linestyle="--",
        alpha=0.7,
    )

    plt.title(
        f"{symbol} - Crescimento do Capital({initial_capital}) vs Buy & Hold",
        fontsize=14,
    )
    plt.xlabel("Data", fontsize=11)
    plt.ylabel("Capital ($)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()
