import pandas as pd
from indicators import calculate_macd, calculate_sma, calculate_ema, calculate_rsi


def compare_price_to_sma(last_close, last_sma):
    if last_close > last_sma:
        return "The last closing price is above the SMA."
    elif last_close < last_sma:
        return "The last closing price is below the SMA."
    else:
        return "The last closing price is equal to the SMA."


def compare_smas(sma_20, sma_50):
    if sma_20 > sma_50:
        return "The 20-day SMA is above the 50-day SMA."
    elif sma_20 < sma_50:
        return "The 20-day SMA is below the 50-day SMA."
    else:
        return "The 20-day SMA is equal to the 50-day SMA."


def interpret_rsi(rsi_value):
    if rsi_value > 70:
        return "The stock is overbought (RSI > 70)."
    elif rsi_value < 30:
        return "The stock is oversold (RSI < 30)."
    else:
        return "The stock is in a neutral state (30 <= RSI <= 70)."


def compare_price_to_ema(last_close, last_ema):
    if last_close > last_ema:
        return "The last closing price is above the EMA."
    elif last_close < last_ema:
        return "The last closing price is below the EMA."
    else:
        return "The last closing price is equal to the EMA."


def compare_emas(ema_20, ema_50):
    if ema_20 > ema_50:
        return "The 20-day EMA is above the 50-day EMA."
    elif ema_20 < ema_50:
        return "The 20-day EMA is below the 50-day EMA."
    else:
        return "The 20-day EMA is equal to the 50-day EMA."


def compare_macd_to_signal(macd, signal_line):
    if macd > signal_line:
        return "Bullish momentum."
    elif macd < signal_line:
        return "Bearish momentum."
    else:
        return "The MACD is equal to the Signal Line."


def calculate_technical_score(sma_20, sma_50, rsi, ema_20, ema_50, macd, signal_line):
    score = 0
    classification = ""
    available_indicators = 0

    # SMA comparison
    if not pd.isna(sma_20) and not pd.isna(sma_50):
        available_indicators += 1
        if sma_20 > sma_50:
            score += 1
        elif sma_20 < sma_50:
            score -= 1

    # RSI interpretation
    if not pd.isna(rsi):
        available_indicators += 1
        if rsi > 70:
            score -= 1
        elif rsi < 30:
            score += 1

    # EMA comparison
    if not pd.isna(ema_20) and not pd.isna(ema_50):
        available_indicators += 1
        if ema_20 > ema_50:
            score += 1
        elif ema_20 < ema_50:
            score -= 1

    # MACD comparison
    if not pd.isna(macd) and not pd.isna(signal_line):
        available_indicators += 1
        if macd > signal_line:
            score += 1
        elif macd < signal_line:
            score -= 1

    normalized_score = score / available_indicators if available_indicators > 0 else 0
    # Classification based on score

    if normalized_score >= +0.75:
        classification = "Strong Bullish"
    elif normalized_score >= 0.5 and normalized_score < 0.75:
        classification = "Bullish"
    elif normalized_score >= 0.25 and normalized_score < 0.5:
        classification = "Weak Bullish"
    elif normalized_score >= -0.25 and normalized_score < 0.25:
        classification = "Neutral"
    elif normalized_score >= -0.5 and normalized_score < -0.25:
        classification = "Weak Bearish"
    elif normalized_score >= -0.75 and normalized_score < -0.5:
        classification = "Bearish"
    elif normalized_score <= -0.75:
        classification = "Strong Bearish"

    return normalized_score, classification, available_indicators
