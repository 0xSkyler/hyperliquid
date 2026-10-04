from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum

MAINNET_API = "https://api.hyperliquid.xyz"
TESTNET_API = "https://api.hyperliquid-testnet.xyz"
LIVE_CONFIRM_PHRASE = "I_UNDERSTAND_REAL_MONEY"


class Mode(StrEnum):
    RESEARCH = "research"
    BACKTEST = "backtest"
    SHADOW = "shadow"
    PAPER = "paper"
    TESTNET = "testnet"
    LIVE = "live"


@dataclass(frozen=True)
class Settings:
    mode: Mode = Mode.PAPER
    coin: str = "BTC"
    paper_equity: float = 1000.0

    # Forecasting
    horizon_s: float = 60.0  # forward-return horizon the model predicts
    decision_interval_s: float = 1.0
    forgetting: float = 0.9999  # RLS forgetting factor (slow structural memory)
    cal_forgetting: float = 0.9998  # calibration memory (faster)
    cal_z: float = 2.0  # lower-confidence-bound z on the out-of-sample slope
    min_indep_samples: float = 30.0  # independent resolved forecasts before any trust

    # Model arena: the first available name starts as champion. "tree" needs lightgbm;
    # "ridge_disc" needs a discovered-features file (python -m app.research.discovery).
    models: tuple[str, ...] = ("ridge", "mlp", "tree", "ridge_disc")
    promote_z: float = 2.5  # paired-test t-statistic a challenger must exceed to be promoted
    tree_min_train: int = 3000
    tree_refit_every: int = 1500
    discovered_path: str = "data/discovered_features.json"
    chart_model_dir: str = "models"  # chart_<timeframe>.txt, from python -m app.research.chart_train
    state_path: str = ""  # learned state, saved every few minutes; default <data_dir>/state/engine-<coin>.pkl

    # LLM news analysis: off unless HL_LLM_NEWS=1 (each new headline cluster is one paid API call)
    llm_news: bool = False
    llm_model: str = "claude-opus-5-5"
    llm_max_per_poll: int = 5

    # Preferences (these are the operator's risk preferences, not trading rules)
    risk_aversion: float = 4.0  # CRRA gamma; 1 = full Kelly, 4 ~ quarter Kelly
    jump_prob: float = 1e-4  # per-horizon probability of a gap move, each direction
    jump_size: float = 0.015  # size of that gap as a fraction of price
    ruin_floor: float = 0.02  # wealth fraction assumed left after liquidation

    # Costs (overridden by the venue's real fee schedule when an address is known)
    taker_fee: float = 0.00045
    maker_fee: float = 0.00015
    latency_ms: float = 150.0  # simulated order latency for paper/backtest

    # Safety kernel: instrumentation checks only
    stale_book_s: float = 3.0  # no market-data message of any kind for this long
    stale_depth_s: float = 20.0  # no book update (BBO only changes when the touch changes)
    stale_account_s: float = 15.0
    unacked_order_s: float = 5.0

    # Infrastructure
    data_dir: str = "data"
    record_raw: bool = False
    database_url: str = ""
    dashboard_host: str = "127.0.0.1"
    dashboard_port: int = 8787
    account_address: str = ""
    news_feeds: tuple[str, ...] = (
        "https://www.coindesk.com/arcade/outboundfeeds/rss/",
        "https://cointelegraph.com/rss",
    )

    @property
    def api_url(self) -> str:
        return TESTNET_API if self.mode is Mode.TESTNET else MAINNET_API

    @property
    def state_file(self) -> str:
        return self.state_path or f"{self.data_dir}/state/engine-{self.coin}.pkl"

    @property
    def horizon_ticks(self) -> int:
        return max(1, round(self.horizon_s / self.decision_interval_s))

    @classmethod
    def from_env(cls) -> Settings:
        env = os.environ
        mode = Mode(env.get("HL_MODE", "paper").lower())
        if mode is Mode.LIVE and env.get("HL_LIVE_CONFIRM") != LIVE_CONFIRM_PHRASE:
            raise SystemExit(
                "HL_MODE=live requires HL_LIVE_CONFIRM=" + LIVE_CONFIRM_PHRASE + " (see docs/LIVE.md)"
            )
        d = cls()
        feeds = env.get("HL_NEWS_FEEDS")
        return cls(
            mode=mode,
            coin=env.get("HL_COIN", d.coin),
            paper_equity=float(env.get("HL_PAPER_EQUITY", d.paper_equity)),
            horizon_s=float(env.get("HL_HORIZON_S", d.horizon_s)),
            risk_aversion=float(env.get("HL_RISK_AVERSION", d.risk_aversion)),
            jump_prob=float(env.get("HL_JUMP_PROB", d.jump_prob)),
            jump_size=float(env.get("HL_JUMP_SIZE", d.jump_size)),
            latency_ms=float(env.get("HL_LATENCY_MS", d.latency_ms)),
            models=tuple(m for m in env.get("HL_MODELS", ",".join(d.models)).split(",") if m),
            discovered_path=env.get("HL_DISCOVERED_PATH", d.discovered_path),
            chart_model_dir=env.get("HL_CHART_MODEL_DIR", d.chart_model_dir),
            state_path=env.get("HL_STATE_PATH", ""),
            llm_news=env.get("HL_LLM_NEWS", "0") == "1",
            llm_model=env.get("HL_LLM_MODEL", d.llm_model),
            llm_max_per_poll=int(env.get("HL_LLM_MAX_PER_POLL", d.llm_max_per_poll)),
            data_dir=env.get("HL_DATA_DIR", d.data_dir),
            record_raw=env.get("HL_RECORD_RAW", "0") == "1",
            database_url=env.get("DATABASE_URL", ""),
            dashboard_host=env.get("HL_DASHBOARD_HOST", d.dashboard_host),
            dashboard_port=int(env.get("HL_DASHBOARD_PORT", d.dashboard_port)),
            account_address=env.get("HL_ACCOUNT_ADDRESS", ""),
            news_feeds=tuple(f for f in feeds.split(",") if f) if feeds is not None else d.news_feeds,
        )
