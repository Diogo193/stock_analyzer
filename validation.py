from data import get_stock_data
from indicators import calculate_ema, calculate_macd, calculate_rsi, calculate_sma
from technical_score import calculate_future_return
from domain import ValidationResult, PerformanceMetrics
import pandas as pd


def prepare_oos_data(stock_data, horizon=20, in_sample_ratio=0.7):
    # 1. Ordenar os dados por data
    df = stock_data.sort_values("Date").reset_index(drop=True)

    # 2. Calcular o MACD e definir o Macd_Score
    sma_200 = calculate_sma(df["Close"], 200)
    sma_50 = calculate_sma(df["Close"], 50)
    ema_20 = calculate_ema(df["Close"], 20)

    macro_bullish = df["Close"] > sma_200
    mid_bullish = ema_20 > sma_50
    short_bullish = df["Close"] > ema_20

    df["Composite_Score"] = 0

    df.loc[(macro_bullish) & (mid_bullish) & (short_bullish), "Composite_Score"] = 1
    df.loc[(~macro_bullish) & (~mid_bullish) & (~short_bullish), "Composite_Score"] = -1

    # 3. Calcular o retorno futuro
    df["Future_Return_20d"] = calculate_future_return(df, horizon)

    # 4. Fatiamento In-Sample (70%) e Out-of-Sample (30%)
    split_index = int(len(df) * in_sample_ratio)
    df_in_sample = df.iloc[:split_index]
    df_out_of_sample = df.iloc[split_index:]

    return df_in_sample, df_out_of_sample


def evaluate_slice(df_slice, indicator_col, target_state):
    # 1. Limpeza de nulos no subconjunto
    clean_df = df_slice.dropna(subset=[indicator_col, "Future_Return_20d"])

    # 2. Métricas do Baseline (Amostra Geral)
    baseline_mean = clean_df["Future_Return_20d"].mean()
    baseline_median = clean_df["Future_Return_20d"].median()
    baseline_count = clean_df.shape[0]
    baseline_pos_rate = (
        (clean_df["Future_Return_20d"] > 0).mean() * 100 if baseline_count > 0 else 0.0
    )

    # 3. Métricas do Estado Alvo (Bullish)
    filtered_df = clean_df[clean_df[indicator_col] == target_state]
    obs_mean = filtered_df["Future_Return_20d"].mean()
    obs_median = filtered_df["Future_Return_20d"].median()
    obs_count = filtered_df.shape[0]
    obs_pos_rate = (
        (filtered_df["Future_Return_20d"] > 0).mean() * 100 if obs_count > 0 else 0.0
    )

    # 4. Cálculo do Mean Edge
    mean_edge = obs_mean - baseline_mean if baseline_mean is not None else 0.0

    return {
        "baseline_mean": baseline_mean,
        "baseline_median": baseline_median,
        "baseline_count": baseline_count,
        "baseline_positive_rate": baseline_pos_rate,
        "obs_mean": obs_mean,
        "obs_median": obs_median,
        "obs_count": obs_count,
        "obs_positive_rate": obs_pos_rate,
        "mean_edge": mean_edge,
    }


def validate_symbol(symbol, period):
    stock_data = get_stock_data(symbol, period)

    df_is_data, df_oos_data = prepare_oos_data(stock_data)

    is_metrics = evaluate_slice(df_is_data, "Composite_Score", 1)

    oos_metrics = evaluate_slice(df_oos_data, "Composite_Score", 1)

    edge_decay = oos_metrics["mean_edge"] - is_metrics["mean_edge"]
    is_stable = (is_metrics["mean_edge"] > 0) and (oos_metrics["mean_edge"] > 0)

    new_results = ValidationResult(
        symbol=symbol,
        in_sample_performance=is_metrics["mean_edge"],
        out_of_sample_performance=oos_metrics["mean_edge"],
        baseline_in_sample=PerformanceMetrics(
            mean=is_metrics["baseline_mean"],
            median=is_metrics["baseline_median"],
            count=is_metrics["baseline_count"],
            positive_rate=is_metrics["baseline_positive_rate"],
        ),
        baseline_out_of_sample=PerformanceMetrics(
            mean=oos_metrics["baseline_mean"],
            median=oos_metrics["baseline_median"],
            count=oos_metrics["baseline_count"],
            positive_rate=oos_metrics["baseline_positive_rate"],
        ),
        bullish_in_sample=PerformanceMetrics(
            mean=is_metrics["obs_mean"],
            median=is_metrics["obs_median"],
            count=is_metrics["obs_count"],
            positive_rate=is_metrics["obs_positive_rate"],
        ),
        bullish_out_of_sample=PerformanceMetrics(
            mean=oos_metrics["obs_mean"],
            median=oos_metrics["obs_median"],
            count=oos_metrics["obs_count"],
            positive_rate=oos_metrics["obs_positive_rate"],
        ),
        is_stable=is_stable,
    )

    return new_results


def run_validation_pipeline(symbols, period="5y"):
    results = []

    for symbol in symbols:
        result = validate_symbol(symbol, period)
        results.append(result)

    # Agregação Cross-Asset
    stable_count = sum(1 for r in results if r.is_stable)
    stability_rate = (stable_count / len(results)) * 100 if results else 0
    passed_validation = stability_rate >= 60.0

    summary = {
        "total_assets": len(results),
        "stable_assets": stable_count,
        "stability_rate": stability_rate,
        "passed_validation": passed_validation,
    }

    return results, summary
