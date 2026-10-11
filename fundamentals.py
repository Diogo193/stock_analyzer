import yfinance as yf
from domain import DEFAULT_THRESHOLDS, FundamentalMetrics


def get_fundamental_data(symbol):
    stock = yf.Ticker(symbol).info

    market_cap = stock.get("marketCap")
    free_cash_flow = stock.get("freeCashflow")

    if free_cash_flow is not None and market_cap is not None and free_cash_flow > 0:
        pFreeCashFlow = market_cap / free_cash_flow
    else:
        pFreeCashFlow = None

    revenue_growth = stock.get("revenueGrowth")
    earnings_growth = stock.get("earningsGrowth")
    growth = None

    if revenue_growth is not None and earnings_growth is not None:
        growth = (earnings_growth + revenue_growth) / 2

    fundamental_data = {
        "symbol": symbol,
        "forwardPE": stock.get("forwardPE"),
        "pegRatio": stock.get("pegRatio"),
        "revenueGrowth": revenue_growth,
        "earningsGrowth": earnings_growth,
        "returnOnEquity": stock.get("returnOnEquity"),
        "profitMargins": stock.get("profitMargins"),
        "debtToEquity": stock.get("debtToEquity"),
        "pFreeCashFlow": pFreeCashFlow,
        "evEbitda": stock.get("enterpriseToEbitda"),
        "marketCap": market_cap,
        "freeCashFlow": free_cash_flow,
        "trailingEps": stock.get("trailingEps"),
        "forwardEps": stock.get("forwardEps"),
        "returnOnAssets": stock.get("returnOnAssets"),
        "growth": growth,
    }

    return fundamental_data


def evaluate_fundamentals(fundamental_data, thresholds=DEFAULT_THRESHOLDS):
    score = 0
    total_criteria = 0
    for key, threshold in thresholds.items():
        value = fundamental_data.get(key)
        if value is not None:
            total_criteria += 1
            if key in [
                "forwardPE",
                "pegRatio",
                "debtToEquity",
                "evEbitda",
                "pFreeCashFlow",
            ]:
                if value > 0 and value < threshold:
                    score += 1
                else:
                    score -= 1
            else:
                if value > threshold:
                    score += 1
                else:
                    score -= 1

    if total_criteria == 0:
        return (
            0.0,
            "NOT APPLICABLE TO THIS STOCK",
            0,
        )  # Return a special value if no criteria were applicable

    relative_score = (
        score / total_criteria
    )  # Calculate the relative score based on the number of criteria met
    decision = (
        "PASS"
        if relative_score > 0.20
        else "FAIL" if relative_score < -0.20 else "NEUTRAL"
    )
    return relative_score, decision, total_criteria


def get_stock_fundamental_metrics(symbol):
    fundamental_data = get_fundamental_data(symbol)
    relative_score, decision, total_criteria = evaluate_fundamentals(
        fundamental_data, thresholds=DEFAULT_THRESHOLDS
    )

    return FundamentalMetrics(
        symbol=symbol,
        forwardPE=fundamental_data.get("forwardPE"),
        pegRatio=fundamental_data.get("pegRatio"),
        revenueGrowth=fundamental_data.get("revenueGrowth"),
        earningsGrowth=fundamental_data.get("earningsGrowth"),
        returnOnEquity=fundamental_data.get("returnOnEquity"),
        profitMargins=fundamental_data.get("profitMargins"),
        debtToEquity=fundamental_data.get("debtToEquity"),
        pFreeCashFlow=fundamental_data.get("pFreeCashFlow"),
        evEbitda=fundamental_data.get("evEbitda"),
        marketCap=fundamental_data.get("marketCap"),
        eps=fundamental_data.get("trailingEps"),
        roa=fundamental_data.get("returnOnAssets"),
        feps=fundamental_data.get("forwardEps"),
        freeCashFlow=fundamental_data.get("freeCashFlow"),
        growth=fundamental_data.get("growth"),
        relative_score=relative_score,
        decision=decision,
        total_criteria_evaluated=total_criteria,
    )
