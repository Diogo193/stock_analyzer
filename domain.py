import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class PerformanceMetrics:
    mean: float
    median: float
    count: int
    positive_rate: float


@dataclass
class RiskConfig:
    risk_per_trade: (
        float  # Percentual do capital a arriscar por trade (ex: 0.01 para 1%)
    )
    atr_multiplier: (
        float  # Multiplicador do ATR para definir o stop loss (ex: 2.0 para 2x ATR)
    )
    max_allocation: float  # Percentual máximo do capital a ser alocado em uma única posição (ex: 1.0 para 100%)
    atr_period: int  # Período para calcular o ATR (ex: 14 dias)


@dataclass
class Trade:
    # 1. Campos obrigatórios (sem valor por defeito)
    entry_date: pd.Timestamp
    entry_type: str
    entry_price: float

    # 2. Campos opcionais (com valor por defeito = None)
    exit_date: Optional[pd.Timestamp] = None
    exit_type: Optional[str] = None
    exit_price: Optional[float] = None
    absolute_profit_loss: Optional[float] = None
    return_percentage: Optional[float] = None
    initial_stop_price: Optional[float] = None
    highest_price_seen: Optional[float] = None
    position_shares: Optional[float] = None

    def close_trade(self, exit_date: pd.Timestamp, exit_type: str, exit_price: float):
        self.exit_date = exit_date
        self.exit_type = exit_type
        self.exit_price = exit_price
        shares = self.position_shares if self.position_shares is not None else 1
        self.absolute_profit_loss = (self.exit_price - self.entry_price) * shares
        self.return_percentage = (
            (self.exit_price - self.entry_price) / self.entry_price
        ) * 100


DEFAULT_THRESHOLDS = {
    "forwardPE": 25.0,
    "pegRatio": 1.5,
    "revenueGrowth": 0.08,
    "earningsGrowth": 0.08,
    "returnOnEquity": 0.15,
    "profitMargins": 0.10,
    "debtToEquity": 150.0,
    "pFreeCashFlow": 25.0,
    "evEbitda": 18.0,
}


@dataclass
class FundamentalMetrics:
    symbol: str
    forwardPE: Optional[float] = None
    pegRatio: Optional[float] = None
    revenueGrowth: Optional[float] = None
    earningsGrowth: Optional[float] = None
    returnOnEquity: Optional[float] = None
    profitMargins: Optional[float] = None
    debtToEquity: Optional[float] = None
    pFreeCashFlow: Optional[float] = None
    evEbitda: Optional[float] = None
    marketCap: Optional[float] = None
    eps: Optional[float] = None
    roa: Optional[float] = None
    freeCashFlow: Optional[float] = None
    feps: Optional[float] = None
    growth: Optional[float] = None
    relative_score: Optional[float] = None
    decision: Optional[str] = None
    total_criteria_evaluated: Optional[int] = None

    @property
    def is_approved(self) -> bool:
        if self.decision is None or "FAIL" in self.decision:
            return False
        else:
            return True

    @property
    def growth_level(self) -> str:
        # Se faltarem dados essenciais para o diagnóstico
        if self.growth is None or self.forwardPE is None:
            return "UNDETERMINED"

        # 1. Empresas de Alto Crescimento (Growth >= 8%)
        if self.growth >= 0.08:
            if self.pegRatio is not None and self.pegRatio <= 1.5:
                return "HEALTHY GROWTH"
            else:
                return "OVERVALUED GROWTH"  # Cresce bem, mas o valuation está esticado

        # 2. Empresas de Baixo Crescimento (Growth < 8%)
        else:
            if self.forwardPE > 25.0:
                return "OVERVALUED WITHOUT SUPPORT"  # Sem crescimento e cara
            elif self.forwardPE <= 20.0:
                return "DEFENSIVE VALUATION"  # Empresa madura a preço razoável
            else:
                return "UNDETERMINED"


@dataclass
class ValidationResult:
    symbol: str
    in_sample_performance: float
    out_of_sample_performance: float
    baseline_in_sample: PerformanceMetrics
    baseline_out_of_sample: PerformanceMetrics
    bullish_in_sample: PerformanceMetrics
    bullish_out_of_sample: PerformanceMetrics
    is_stable: bool

    @property
    def edge_decay(self) -> float:
        return self.out_of_sample_performance - self.in_sample_performance
