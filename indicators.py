import pandas as pd


def calculate_sma(data, window):
    """
    Calculate the Simple Moving Average (SMA) for the given data and window size.

    Parameters:
    data (pd.Series): The input data series (e.g., closing prices).
    window (int): The window size for calculating the SMA.

    Returns:
    pd.Series: The SMA values.
    """
    if not isinstance(data, pd.Series):
        raise ValueError("Input data must be a pandas Series.")
    if window <= 0:
        raise ValueError("Window size must be a positive integer.")

    sma = data.rolling(window=window).mean()
    return sma


def calculate_rsi(data, window):
    """
    Calculate the Relative Strength Index (RSI) for the given data and window size.

    Parameters:
    data (pd.Series): The input data series (e.g., closing prices).
    window (int): The window size for calculating the RSI.

    Returns:
    pd.Series: The RSI values.
    """
    if not isinstance(data, pd.Series):
        raise ValueError("Input data must be a pandas Series.")
    if window <= 0:
        raise ValueError("Window size must be a positive integer.")

    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_ema(data, window):
    """
    Calculate the Exponential Moving Average (EMA) for the given data and window size.

    Parameters:
    data (pd.Series): The input data series (e.g., closing prices).
    window (int): The window size for calculating the EMA.

    Returns:
    pd.Series: The EMA values.
    """
    if not isinstance(data, pd.Series):
        raise ValueError("Input data must be a pandas Series.")
    if window <= 0:
        raise ValueError("Window size must be a positive integer.")

    ema = data.ewm(span=window, adjust=False, min_periods=window).mean()
    return ema


def calculate_macd(data, short_window, long_window, signal_window):
    """
    Calculate the Moving Average Convergence Divergence (MACD) for the given data.

    Parameters:
    data (pd.Series): The input data series (e.g., closing prices).
    short_window (int): The window size for the short-term EMA.
    long_window (int): The window size for the long-term EMA.
    signal_window (int): The window size for the signal line EMA.

    Returns:
    pd.DataFrame: A DataFrame containing the MACD line, the signal line and the histogram.
    """
    if not isinstance(data, pd.Series):
        raise ValueError("Input data must be a pandas Series.")
    if short_window <= 0 or long_window <= 0 or signal_window <= 0:
        raise ValueError("Window sizes must be positive integers.")
    if short_window >= long_window:
        raise ValueError("Short window must be less than long window.")

    ema_short = calculate_ema(data, short_window)
    ema_long = calculate_ema(data, long_window)

    macd_line = ema_short - ema_long
    signal_line = calculate_ema(macd_line, signal_window)
    histogram = macd_line - signal_line

    macd_df = pd.DataFrame(
        {"MACD": macd_line, "Signal": signal_line, "Histogram": histogram}
    )

    return macd_df


def calculate_atr(data, window):
    """
    Calculate the Average True Range (ATR) for the given data and window size.

    Parameters:
    data (pd.DataFrame): The input data DataFrame (e.g., OHLC data).
    window (int): The window size for calculating the ATR.

    Returns:
    pd.Series: The ATR values.
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame.")
    if window <= 0:
        raise ValueError("Window size must be a positive integer.")

    high = data["High"]
    low = data["Low"]
    close = data["Close"]

    tr0 = abs(high - low)
    tr1 = abs(high - close.shift(1))
    tr2 = abs(low - close.shift(1))
    tr = pd.DataFrame({"tr0": tr0, "tr1": tr1, "tr2": tr2}).max(axis=1)

    atr = tr.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()

    return atr
