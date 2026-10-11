from data import get_stock_data_4h
from indicators import (
    calculate_sma,
    calculate_ema,
    calculate_macd,
    calculate_rsi,
)

from typing import Optional

import pandas as pd

# region Score Generation


def calculate_historical_indicator_scores(stock_data):
    sma_20 = calculate_sma(stock_data["Close"], window=20)
    sma_50 = calculate_sma(stock_data["Close"], window=50)
    ema_20 = calculate_ema(stock_data["Close"], window=20)
    ema_50 = calculate_ema(stock_data["Close"], window=50)
    macd_data = calculate_macd(stock_data["Close"], 12, 26, 9)
    macd_line = macd_data["MACD"]
    signal_line = macd_data["Signal"]
    rsi = calculate_rsi(stock_data["Close"], window=14)

    valid_indicators = (
        sma_20.notna()
        & sma_50.notna()
        & ema_20.notna()
        & ema_50.notna()
        & macd_line.notna()
        & signal_line.notna()
        & rsi.notna()
    )

    sma_score = pd.Series(0.0, index=stock_data.index)
    ema_score = pd.Series(0.0, index=stock_data.index)
    macd_score = pd.Series(0.0, index=stock_data.index)
    rsi_score = pd.Series(0.0, index=stock_data.index)

    sma_score[~valid_indicators] = float("nan")
    ema_score[~valid_indicators] = float("nan")
    macd_score[~valid_indicators] = float("nan")
    rsi_score[~valid_indicators] = float("nan")

    sma_score[sma_20 > sma_50] = 1
    sma_score[sma_20 < sma_50] = -1

    ema_score[ema_20 > ema_50] = 1
    ema_score[ema_20 < ema_50] = -1

    macd_score[macd_line > signal_line] = 1
    macd_score[macd_line < signal_line] = -1

    rsi_score[rsi < 30] = 1
    rsi_score[rsi > 70] = -1

    indicator_scores = pd.DataFrame(
        {
            "SMA_Score": sma_score,
            "EMA_Score": ema_score,
            "MACD_Score": macd_score,
            "RSI_Score": rsi_score,
        }
    )

    return indicator_scores


def calculate_directional_edge(indicator_summary):
    directional_edge = {}

    bullish_mean = indicator_summary[indicator_summary.index > 0]["Future_Return_60d"][
        "mean"
    ].mean()
    bearish_mean = indicator_summary[indicator_summary.index < 0]["Future_Return_60d"][
        "mean"
    ].mean()

    mean_edge = bullish_mean - bearish_mean
    positive_rate_edge = (
        indicator_summary[indicator_summary.index > 0]["Positive_Rate_60d"].mean()
        - indicator_summary[indicator_summary.index < 0]["Positive_Rate_60d"].mean()
    )

    directional_edge["Mean_Edge"] = mean_edge
    directional_edge["Positive_Rate_Edge"] = positive_rate_edge

    return directional_edge


def calculate_technical_state(stock_data):
    sma_20 = calculate_sma(stock_data["Close"], window=20)
    sma_50 = calculate_sma(stock_data["Close"], window=50)
    ema_20 = calculate_ema(stock_data["Close"], window=20)
    ema_50 = calculate_ema(stock_data["Close"], window=50)

    last_sma_20 = sma_20.iloc[-1]
    last_sma_50 = sma_50.iloc[-1]
    last_ema_20 = ema_20.iloc[-1]
    last_ema_50 = ema_50.iloc[-1]

    if (
        pd.isna(last_sma_20)
        or pd.isna(last_sma_50)
        or pd.isna(last_ema_20)
        or pd.isna(last_ema_50)
    ):
        trend_state = "Unavailable"
        sma_trend = "Unavailable"
        ema_trend = "Unavailable"

    else:

        sma_trend = (
            "Bullish"
            if last_sma_20 > last_sma_50
            else "Bearish" if last_sma_20 < last_sma_50 else "Neutral"
        )
        ema_trend = (
            "Bullish"
            if last_ema_20 > last_ema_50
            else "Bearish" if last_ema_20 < last_ema_50 else "Neutral"
        )

        if sma_trend == "Bullish" and ema_trend == "Bullish":
            trend_state = "Bullish"
        elif sma_trend == "Bearish" and ema_trend == "Bearish":
            trend_state = "Bearish"
        elif sma_trend == "Neutral" and ema_trend == "Neutral":
            trend_state = "Neutral"
        else:
            trend_state = "Mixed"

    macd_data = calculate_macd(stock_data["Close"], 12, 26, 9)
    macd_line = macd_data["MACD"].iloc[-1]
    signal_line = macd_data["Signal"].iloc[-1]

    if pd.isna(macd_line) or pd.isna(signal_line):
        macd_trend = "Unavailable"
    else:
        if macd_line > signal_line:
            macd_trend = "Bullish"
        elif macd_line < signal_line:
            macd_trend = "Bearish"
        else:
            macd_trend = "Neutral"

    rsi = calculate_rsi(stock_data["Close"], window=14).iloc[-1]

    if pd.isna(rsi):
        rsi_condition = "Unavailable"
    else:
        if rsi > 70:
            rsi_condition = "Overbought"
        elif rsi < 30:
            rsi_condition = "Oversold"
        else:
            rsi_condition = "Neutral"

    results = {
        "Trend": trend_state,
        "SMA Trend": sma_trend,
        "EMA Trend": ema_trend,
        "Momentum": macd_trend,
        "RSI Condition": rsi_condition,
    }

    return results


def calculate_technical_score_4h(symbol):
    stock_data_4h = get_stock_data_4h(symbol)

    score_4h = calculate_predictive_score(stock_data_4h)

    score_4h = score_4h["Score"].iloc[-1] if not score_4h.empty else None

    if score_4h is None or pd.isna(score_4h):
        return None
    elif score_4h >= 0.25:
        return "Bullish"
    elif score_4h <= -0.25:
        return "Bearish"
    else:
        return "Neutral"


# endregion


# region Score Validation


def calculate_future_return(stock_data, days):

    current_close = stock_data["Close"]
    future_close = stock_data["Close"].shift(-days)

    future_return = (future_close - current_close) / current_close * 100

    return future_return


def evaluate_technical_scores(stock_data, score_function):
    scores = score_function(stock_data)

    future_return_5d = calculate_future_return(stock_data, 5)
    future_return_20d = calculate_future_return(stock_data, 20)
    future_return_60d = calculate_future_return(stock_data, 60)

    validation_data = pd.DataFrame(
        {
            "Score": scores,
            "Future_Return_5d": future_return_5d,
            "Future_Return_20d": future_return_20d,
            "Future_Return_60d": future_return_60d,
        }
    )

    positive_5d = (
        validation_data.dropna(subset=["Future_Return_5d"])
        .groupby("Score")["Future_Return_5d"]
        .apply(lambda x: (x > 0).mean())
        * 100
    )
    positive_20d = (
        validation_data.dropna(subset=["Future_Return_20d"])
        .groupby("Score")["Future_Return_20d"]
        .apply(lambda x: (x > 0).mean())
        * 100
    )
    positive_60d = (
        validation_data.dropna(subset=["Future_Return_60d"])
        .groupby("Score")["Future_Return_60d"]
        .apply(lambda x: (x > 0).mean())
        * 100
    )

    validation_data["Positive_5d"] = validation_data["Future_Return_5d"] > 0
    validation_data["Positive_20d"] = validation_data["Future_Return_20d"] > 0
    validation_data["Positive_60d"] = validation_data["Future_Return_60d"] > 0

    score_summary = validation_data.groupby("Score").agg(
        {
            "Future_Return_5d": ["mean", "count", "median"],
            "Future_Return_20d": ["mean", "count", "median"],
            "Future_Return_60d": ["mean", "count", "median"],
        }
    )

    score_summary["Positive_Rate_5d"] = positive_5d
    score_summary["Positive_Rate_20d"] = positive_20d
    score_summary["Positive_Rate_60d"] = positive_60d

    return score_summary


# endregion

# region Indicator Analysis


def calculate_agreement(indicator_scores, indicator_1, indicator_2):
    agreement = indicator_scores[indicator_1] == indicator_scores[indicator_2]

    return agreement.mean() * 100


def evaluate_indicator_predictiveness(stock_data):
    indicator_scores = calculate_historical_indicator_scores(stock_data)

    future_return_5d = calculate_future_return(stock_data, 5)
    future_return_20d = calculate_future_return(stock_data, 20)
    future_return_60d = calculate_future_return(stock_data, 60)

    validation_data = pd.DataFrame(
        {
            "SMA_Score": indicator_scores["SMA_Score"],
            "EMA_Score": indicator_scores["EMA_Score"],
            "MACD_Score": indicator_scores["MACD_Score"],
            "RSI_Score": indicator_scores["RSI_Score"],
            "Future_Return_5d": future_return_5d,
            "Future_Return_20d": future_return_20d,
            "Future_Return_60d": future_return_60d,
        }
    )

    indicators = ["SMA_Score", "EMA_Score", "MACD_Score", "RSI_Score"]

    results = {}

    for indicator in indicators:
        positive_5d = (
            validation_data.dropna(subset=["Future_Return_5d"])
            .groupby(indicator)["Future_Return_5d"]
            .apply(lambda x: (x > 0).mean())
            * 100
        )
        positive_20d = (
            validation_data.dropna(subset=["Future_Return_20d"])
            .groupby(indicator)["Future_Return_20d"]
            .apply(lambda x: (x > 0).mean())
            * 100
        )
        positive_60d = (
            validation_data.dropna(subset=["Future_Return_60d"])
            .groupby(indicator)["Future_Return_60d"]
            .apply(lambda x: (x > 0).mean())
            * 100
        )

        indicator_summary = validation_data.groupby(indicator).agg(
            {
                "Future_Return_5d": ["mean", "count", "median"],
                "Future_Return_20d": ["mean", "count", "median"],
                "Future_Return_60d": ["mean", "count", "median"],
            }
        )

        indicator_summary["Positive_Rate_5d"] = positive_5d
        indicator_summary["Positive_Rate_20d"] = positive_20d
        indicator_summary["Positive_Rate_60d"] = positive_60d

        results[indicator] = indicator_summary

    return results


def get_indicator_state_evidence(indicator_results, current_state, indicator_name):

    state_mappings = {
        "SMA_Score": {
            "Bullish": 1,
            "Bearish": -1,
            "Neutral": 0,
        },
        "EMA_Score": {
            "Bullish": 1,
            "Bearish": -1,
            "Neutral": 0,
        },
        "MACD_Score": {
            "Bullish": 1,
            "Bearish": -1,
            "Neutral": 0,
        },
        "RSI_Score": {
            "Oversold": 1,
            "Neutral": 0,
            "Overbought": -1,
        },
    }

    state_keys = {
        "SMA_Score": "SMA Trend",
        "EMA_Score": "EMA Trend",
        "MACD_Score": "Momentum",
        "RSI_Score": "RSI Condition",
    }

    state_key = state_keys[indicator_name]

    state_label = current_state[state_key]

    if state_label == "Unavailable":
        return None

    indicator_summary = indicator_results[indicator_name]

    numeric_state = state_mappings[indicator_name][state_label]

    if numeric_state not in indicator_summary.index:
        return None

    return indicator_summary.loc[numeric_state]


def structure_indicator_evidence(evidence):
    structured_evidence = {}

    for indicator, data in evidence.items():
        if data is not None:
            structured_evidence[indicator] = {
                "5d": {
                    "Mean": data[("Future_Return_5d", "mean")],
                    "Median": data[("Future_Return_5d", "median")],
                    "Positive_Rate": data["Positive_Rate_5d", ""],
                    "Count": data[("Future_Return_5d", "count")],
                },
                "20d": {
                    "Mean": data[("Future_Return_20d", "mean")],
                    "Median": data[("Future_Return_20d", "median")],
                    "Positive_Rate": data["Positive_Rate_20d", ""],
                    "Count": data[("Future_Return_20d", "count")],
                },
                "60d": {
                    "Mean": data[("Future_Return_60d", "mean")],
                    "Median": data[("Future_Return_60d", "median")],
                    "Positive_Rate": data["Positive_Rate_60d", ""],
                    "Count": data[("Future_Return_60d", "count")],
                },
            }
        else:
            structured_evidence[indicator] = None

    return structured_evidence


def calculate_predictive_baseline(stock_data):
    future_return_5d = calculate_future_return(stock_data, 5)
    future_return_20d = calculate_future_return(stock_data, 20)
    future_return_60d = calculate_future_return(stock_data, 60)

    future_return_5d = future_return_5d.dropna()
    future_return_20d = future_return_20d.dropna()
    future_return_60d = future_return_60d.dropna()

    baseline = {
        "5d": {
            "Mean": future_return_5d.mean(),
            "Median": future_return_5d.median(),
            "Positive_Rate": (future_return_5d > 0).mean() * 100,
            "Count": future_return_5d.count(),
        },
        "20d": {
            "Mean": future_return_20d.mean(),
            "Median": future_return_20d.median(),
            "Positive_Rate": (future_return_20d > 0).mean() * 100,
            "Count": future_return_20d.count(),
        },
        "60d": {
            "Mean": future_return_60d.mean(),
            "Median": future_return_60d.median(),
            "Positive_Rate": (future_return_60d > 0).mean() * 100,
            "Count": future_return_60d.count(),
        },
    }

    return baseline


def calculate_evidence_edge(structured_evidence, baseline):
    edge = {}
    for indicator, data in structured_evidence.items():
        if data is not None:
            edge[indicator] = {}
            for period, metrics in data.items():
                if period in baseline:
                    baseline_metrics = baseline[period]
                    edge[indicator][period] = {
                        "Mean_Edge": metrics["Mean"] - baseline_metrics["Mean"],
                        "Median_Edge": metrics["Median"] - baseline_metrics["Median"],
                        "Positive_Rate_Edge": metrics["Positive_Rate"]
                        - baseline_metrics["Positive_Rate"],
                        "Count": metrics["Count"],
                    }
        else:
            edge[indicator] = None
    return edge


def calculate_predictive_score(stock_data):
    # 1. Obter Indicadores
    sma_20 = calculate_sma(stock_data["Close"], 20)
    sma_50 = calculate_sma(stock_data["Close"], 50)
    ema_20 = calculate_ema(stock_data["Close"], 20)
    ema_50 = calculate_ema(stock_data["Close"], 50)
    macd_data = calculate_macd(stock_data["Close"], 12, 26, 9)
    rsi = calculate_rsi(stock_data["Close"], 14)

    # 2. Avaliar presença de dados válidos
    valid_mask = (
        sma_20.notna()
        & sma_50.notna()
        & ema_20.notna()
        & ema_50.notna()
        & macd_data["MACD"].notna()
        & macd_data["Signal"].notna()
        & rsi.notna()
    )

    scores = pd.Series(0.0, index=stock_data.index)
    scores[~valid_mask] = float("nan")

    # 3. Pontuação dos componentes (-1, 0, +1)
    scores[sma_20 > sma_50] += 1.0
    scores[sma_20 < sma_50] -= 1.0

    scores[ema_20 > ema_50] += 1.0
    scores[ema_20 < ema_50] -= 1.0

    scores[macd_data["MACD"] > macd_data["Signal"]] += 1.0
    scores[macd_data["MACD"] < macd_data["Signal"]] -= 1.0

    scores[rsi < 30] += 1.0  # Sobrevenda (Bullish)
    scores[rsi > 70] -= 1.0  # Sobrecompra (Bearish)

    # 4. Normalização (-1.0 a +1.0)
    normalized_scores = scores / 4.0

    # 5. Atribuição de Níveis de Confiança (Fase 7.7)
    confidence_levels = pd.Series("Low", index=stock_data.index)
    confidence_levels[normalized_scores.abs() >= 0.75] = "High"
    confidence_levels[
        (normalized_scores.abs() >= 0.50) & (normalized_scores.abs() < 0.75)
    ] = "Medium"
    confidence_levels[~valid_mask] = "Unavailable"

    return pd.DataFrame({"Score": normalized_scores, "Confidence": confidence_levels})


def get_contradictory_factors(
    stock_data: pd.DataFrame,
    signal_1d: str,
    state_4h: Optional[str],
    fundamental_metrics
) -> list[str]:
    factors = []
    
    last_row = stock_data.iloc[-1]
    rsi = last_row.get("RSI")
    close = last_row.get("Close")
    sma_50 = last_row.get("SMA_50")  # Se calculada
    atr = last_row.get("ATR")
    
    # 1. Divergência Temporal (1D vs 4H)
    if signal_1d == "BUY" and state_4h == "Bearish":
        factors.append("Tempo: 1D em Alta (BUY), mas 4H em Recuo (Bearish).")
    elif signal_1d == "BUY" and state_4h == "Neutral":
        factors.append("Tempo: 1D em Alta (BUY), mas 4H sem Direção Definida (Neutral).")
        
    # 2. RSI Extremo
    if rsi is not None:
        if signal_1d == "BUY" and rsi > 70:
            factors.append(f"Momento: Sinal BUY, mas RSI em Sobrecompra Extrema ({rsi:.1f} > 70).")
        elif signal_1d in ["SELL", "NEUTRAL"] and rsi < 30:
            factors.append(f"Momento: Tendência Fraca, mas RSI em Sobrevenda Extrema ({rsi:.1f} < 30).")
            
    # 3. Valuation Fricção
    if signal_1d == "BUY" and fundamental_metrics is not None:
        if fundamental_metrics.growth_level == "OVERVALUED GROWTH":
            factors.append("Valuation: Sinal Técnico BUY, mas Empresa Sobreavaliada (Overvalued Growth).")
        elif fundamental_metrics.decision == "FAIL":
            factors.append("Fundamentais: Sinal Técnico BUY em Conflito com Fundamentais Reprovados (FAIL).")

    # 4. Esticamento de Preço vs Média (Se SMA_50 e ATR existirem)
    if close is not None and sma_50 is not None and atr is not None and atr > 0:
        dist_sma = (close - sma_50) / atr
        if dist_sma > 2.5:
            factors.append(f"Volatilidade: Preço esticado {dist_sma:.1f}x ATR acima da Média de 50 dias.")

    return factors

# endregion

# region Reporting


def scores_table(score_summary, symbol):

    print("\nTechnical Score Summary Table:")
    print(
        f'{"Ticker":<10} {"Score":<15} {"Count":<15} {"Mean 60d":<15} {"Median 60d":<15} {"Positive Rate":<20}'
    )
    for score in score_summary.index:
        mean_60d = score_summary.loc[score, ("Future_Return_60d", "mean")]
        median_60d = score_summary.loc[score, ("Future_Return_60d", "median")]
        positive_rate_60d = score_summary.loc[score, "Positive_Rate_60d"]

        print(
            f'{symbol:<10} {score:<15.2f} {score_summary.loc[score, ("Future_Return_60d", "count")]:<15.2f} {mean_60d:<15.2f}% {median_60d:<15.2f}% {positive_rate_60d:<20.2f}%'
        )


def agreement_table(indicator_scores, symbol):
    indicators = ["SMA_Score", "EMA_Score", "MACD_Score", "RSI_Score"]
    print("\nIndicator Agreement Table:")
    print(
        f'{"Ticker":<10} {"Indicator 1":<15} {"Indicator 2":<15} {"Agreement (%)":<20}'
    )
    for i in range(len(indicators)):
        for j in range(i + 1, len(indicators)):
            indicator_1 = indicators[i]
            indicator_2 = indicators[j]
            agreement = calculate_agreement(indicator_scores, indicator_1, indicator_2)
            print(
                f"{symbol:<10} {indicator_1:<15} {indicator_2:<15} {agreement:<20.2f}%"
            )


def print_technical_state(technical_state, symbol):
    print("\nTechnical State:")
    print(
        f'{"Ticker":<10} {"Trend":<15} {"SMA Trend":<15} {"EMA Trend":<15} {"Momentum":<15} {"RSI Condition":<20}'
    )
    print(
        f"{symbol:<10} {technical_state['Trend']:<15} {technical_state['SMA Trend']:<15} {technical_state['EMA Trend']:<15} {technical_state['Momentum']:<15} {technical_state['RSI Condition']:<20}"
    )


def print_predictive_evidence(structured_evidence, technical_state, symbol):
    print("\n" + "=" * 70)
    print(symbol)
    print("=" * 70)
    print("Technical State:")
    print(f"Trend: {technical_state['Trend']}")
    print(f"SMA Trend: {technical_state['SMA Trend']}")
    print(f"EMA Trend: {technical_state['EMA Trend']}")
    print(f"Momentum: {technical_state['Momentum']}")
    print(f"RSI Condition: {technical_state['RSI Condition']}")

    print("\nPredictive Evidence:")
    for indicator, data in structured_evidence.items():
        if data is not None:
            print(f"\n{indicator}:")
            for period, metrics in data.items():
                print(f"  {period}:")
                for metric_name, value in metrics.items():
                    if metric_name == "Count":
                        print(f"    {metric_name}: {int(value)}")
                    else:
                        print(f"    {metric_name}: {value:.2f}%")
        else:
            print(f"\n{indicator}: No evidence available.")


def print_predictive_edge(structured_evidence, evidence_edge, baseline, symbol):

    print("\n" + "=" * 70)
    print(symbol)
    print("=" * 70)

    for indicator, data in structured_evidence.items():
        if data is not None and evidence_edge[indicator] is not None:
            print(f"\n{indicator}:")
            for period in ["5d", "20d", "60d"]:
                if period in data and period in evidence_edge[indicator]:
                    observed_metrics = data[period]
                    baseline_metrics = baseline[period]
                    edge_metrics = evidence_edge[indicator][period]

                    print(f"\n  {period}:")
                    print(
                        f"    Observed Mean:          {observed_metrics['Mean']:.2f}%"
                    )
                    print(
                        f"    Baseline Mean:          {baseline_metrics['Mean']:.2f}%"
                    )
                    print(
                        f"    Mean Edge:              {edge_metrics['Mean_Edge']:.2f} pp"
                    )
                    print()

                    print(
                        f"    Observed Median:        {observed_metrics['Median']:.2f}%"
                    )
                    print(
                        f"    Baseline Median:        {baseline_metrics['Median']:.2f}%"
                    )
                    print(
                        f"    Median Edge:            {edge_metrics['Median_Edge']:.2f} pp"
                    )
                    print()

                    print(
                        f"    Observed Positive Rate: {observed_metrics['Positive_Rate']:.2f}%"
                    )
                    print(
                        f"    Baseline Positive Rate: {baseline_metrics['Positive_Rate']:.2f}%"
                    )
                    print(
                        f"    Positive Rate Edge:     {edge_metrics['Positive_Rate_Edge']:.2f} pp"
                    )
                    print()

                    print(
                        f"    Observations:           {int(observed_metrics['Count'])}"
                    )
        else:
            print(f"\n{indicator}: No evidence available.")


# endregion
