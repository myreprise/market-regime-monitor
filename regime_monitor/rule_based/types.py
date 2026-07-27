from dataclasses import dataclass
import pandas as pd
from typing import Optional

@dataclass
class MarketSnapshot:
    symbol: str
    timeframe: str  # 'D', '4H', etc.
    ohlcv: pd.DataFrame  # index: datetime, columns: ['open', 'high', 'low', 'close', 'volume']
    vix_ohlc: Optional[pd.DataFrame] = None  # optional, for index / regime use
