import yfinance as yf


# Fetch stock data for a given symbol and period
def get_stock_data(symbol, period):
    stock = yf.Ticker(symbol)
    data = stock.history(period=period)
    data = data.reset_index()  # Reset index to have a clean DataFrame

    data.columns = [col.capitalize() for col in data.columns]  # Capitalize column names

    data = data[
        ["Date", "Open", "High", "Low", "Close", "Volume"]
    ]  # Keep only relevant columns
    data = data.dropna(subset=["Open", "Close"])

    data = data.sort_values("Date").reset_index(
        drop=True
    )  # Sort by date and reset index

    return data


def get_stock_data_4h(symbol, period="60d"):
    stock = yf.Ticker(symbol)
    data = stock.history(period=period, interval="1h")

    aggregation_rules = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }

    data_4h = data.resample("4h").agg(aggregation_rules)

    data_4h = data_4h.dropna(
        subset=["Open", "Close"]
    )  # Drop rows with NaN values in Open or Close

    data_4h = data_4h.reset_index()  # Reset index to have a clean DataFrame

    data_4h = data_4h.rename(
        columns={"Datetime": "Date", "index": "Date"}
    )  # Rename 'Datetime' column to 'Date'
    data_4h.columns = [
        col.capitalize() for col in data_4h.columns
    ]  # Capitalize column names

    data_4h = data_4h[
        ["Date", "Open", "High", "Low", "Close", "Volume"]
    ]  # Keep only relevant columns

    data_4h = data_4h.sort_values("Date").reset_index(
        drop=True
    )  # Sort by date and reset index

    return data_4h
