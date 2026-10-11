import pandas as pd
from typing import Optional, Tuple
from domain import RiskConfig


def calculate_position_size(
    portfolio_capital: float, entry_price: float, atr: float, risk_config: RiskConfig
) -> Tuple[int, float]:

    if atr <= 0:
        return 0, 0.0  # Evita divisão por zero ou valores negativos de ATR

    dstop = atr * risk_config.atr_multiplier
    initial_stop_price = entry_price - dstop
    risk_amount = portfolio_capital * risk_config.risk_per_trade

    raw_shares = risk_amount / dstop

    max_shares = (portfolio_capital * risk_config.max_allocation) / entry_price

    final_shares = min(raw_shares, max_shares)

    return int(final_shares), initial_stop_price


def update_trailing_stop(
    current_high: float,
    current_atr: float,
    previous_highest_price: float,
    previous_stop_price: float,
    atr_multiplier: float,
) -> Tuple[float, float]:
    new_highest_price = max(previous_highest_price, current_high)
    candidate_stop = new_highest_price - (current_atr * atr_multiplier)
    new_stop = max(previous_stop_price, candidate_stop)

    return new_highest_price, new_stop


def check_stop_trigger(
    current_open: float,
    current_low: float,
    current_stop_price: float,
    slippage_rate: float,
) -> Tuple[bool, Optional[float]]:
    if current_low <= current_stop_price:
        # Seleciona o menor valor entre o stop e o preço de abertura (para tratar gap down)
        raw_exit = min(current_stop_price, current_open)
        effective_exit = raw_exit * (1 - slippage_rate)
        return True, effective_exit

    return False, None
