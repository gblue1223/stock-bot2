"""
Program Trading Algorithm Package (`algo_trader/program`)

실시간 프로그램 매매 수급을 모니터링하여 10분할 분할 매수/매도를 집행하는 알고리즘 트레이딩 패키지입니다.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# dotenv 우선 로드
_project_root = Path(__file__).resolve().parent.parent.parent
_env_path = _project_root / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

from .config import ProgramTradingConfig
from .strategy import ProgramTradingStrategy, ProgramSignal
from .trader import RealtimeProgramTrader
from .recorder import ProgramTradeRecorder

__all__ = [
    "ProgramTradingConfig",
    "ProgramTradingStrategy",
    "ProgramSignal",
    "RealtimeProgramTrader",
    "ProgramTradeRecorder",
]
