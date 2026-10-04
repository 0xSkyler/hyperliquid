"""Expected-utility decision engine.

Every tick the engine scores a grid of target exposures f = signed notional / equity,
from -max_leverage to +max_leverage (0 = flat, current f = hold), and picks the one with
the highest expected CRRA utility of next-horizon wealth. Size, leverage, direction,
"do nothing", pyramiding, reducing, reversing and maker-vs-taker all fall out of that one
comparison; there are no entry rules, stop percentages or leverage caps beyond the venue's.

Accounting convention: opening exposure is charged the full round trip (entry + eventual
exit), so reducing exposure is free at the margin. That makes "hold a position whose edge
has gone" strictly worse than "close it" (Jensen), and gives a natural no-trade band.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from app.config.settings import Settings
from app.exchange.base import AccountState, AssetMeta, Book, OrderIntent, impact_frac

_NODES, _WEIGHTS = np.polynomial.hermite_e.hermegauss(15)
_WEIGHTS = _WEIGHTS / _WEIGHTS.sum()
# Fat tails: 95% "normal" regime + 5% with 4x the scale, total variance normalised to 1.
_S1 = 1.0 / math.sqrt(0.95 + 0.05 * 16.0)
_Z = np.concatenate([_NODES * _S1, _NODES * _S1 * 4.0])
_ZW = np.concatenate([_WEIGHTS * 0.95, _WEIGHTS * 0.05])


@dataclass(slots=True)
class Forecast:
    mu_bps: float  # calibrated expected return over the horizon
    mu_raw_bps: float
    sigma_bps: float  # return std over the horizon
    param_sigma_bps: float  # uncertainty of the forecast mean
    beta: float  # out-of-sample credibility in [0, 1]
    half_life_s: float
    horizon_s: float
    sell_rate: float = 0.0
    buy_rate: float = 0.0
    maker_adverse_bps: float = 1.0


@dataclass(slots=True)
class Decision:
    ts: float
    action: str  # HOLD | BUY | SELL | HALT
    reason: str
    f_current: float
    f_target: float
    equity: float
    order: OrderIntent | None = None
    exec_style: str = ""
    exec_reason: str = ""
    eu_gain: float = 0.0  # utility of acting minus utility of holding
    expected_edge_bps: float = 0.0
    p_adverse: float = 0.5  # P(return over horizon goes against the target)
    loss_1sigma_pct: float = 0.0  # equity lost on a 1-sigma adverse move at target exposure
    liq_distance_pct: float = 0.0
    cost_bps: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["order"] = asdict(self.order) if self.order else None
        return d


def _utility(w: np.ndarray, gamma: float) -> np.ndarray:
    if abs(gamma - 1.0) < 1e-9:
        return np.log(w)
    return (w ** (1.0 - gamma) - 1.0) / (1.0 - gamma)


def expected_utility(
    f: np.ndarray, cost: np.ndarray, mu: float, sigma: float, maint: float, s: Settings
) -> np.ndarray:
    """E[u(wealth)] for each exposure f with up-front cost `cost` (fractions of equity)."""
    # Gap scenarios are not shifted by the forecast: a gap swamps whatever the model expected.
    r = np.concatenate([mu + sigma * _Z, [-s.jump_size, s.jump_size]])
    pw = np.concatenate([_ZW * (1 - 2 * s.jump_prob), [s.jump_prob, s.jump_prob]])
    w = 1.0 + f[:, None] * r[None, :] - cost[:, None]
    liquidated = w <= maint * np.abs(f)[:, None]
    w = np.where(liquidated, s.ruin_floor, np.maximum(w, s.ruin_floor))
    return _utility(w, s.risk_aversion) @ pw


def _opened(f: np.ndarray, f_cur: float) -> np.ndarray:
    """Exposure newly opened by moving from f_cur to f (reductions open nothing)."""
    same = (np.sign(f) == np.sign(f_cur)) | (f_cur == 0)
    return np.where(same, np.maximum(np.abs(f) - abs(f_cur), 0.0), np.abs(f))


def decide(
    now: float, fc: Forecast, acct: AccountState, book: Book, meta: AssetMeta, s: Settings
) -> Decision:
    eq, mid = acct.equity, book.mid
    if eq <= 0:
        return Decision(now, "HALT", "no equity", 0.0, 0.0, eq)
    f_cur = acct.position * mid / eq
    lmax = meta.max_leverage
    grid = np.unique(np.concatenate([np.linspace(-lmax, lmax, 161), [0.0, f_cur]]))
    mu = fc.mu_bps * 1e-4
    sigma = math.hypot(fc.sigma_bps, fc.param_sigma_bps) * 1e-4
    maint = meta.maintenance_margin
    opened = _opened(grid, f_cur)
    i_hold = int(np.argmin(np.abs(grid - f_cur)))

    # Taker: pay fee + half spread + book impact now, and assume the same on the way out.
    slip = np.array([
        impact_frac(book.asks if f > f_cur else book.bids, mid, abs(f - f_cur) * eq) if o > 0 else 0.0
        for f, o in zip(grid, opened, strict=True)
    ])  # fmt: skip
    cost_t = opened * 2.0 * (s.taker_fee + slip)
    eu_t = expected_utility(grid, cost_t, mu, sigma, maint, s)
    i_t = int(np.argmax(eu_t))
    eu_hold = float(eu_t[i_hold])

    # Maker: cheaper entry, but it may not fill and fills are adversely selected.
    cost_m = opened * (s.maker_fee + fc.maker_adverse_bps * 1e-4 + s.taker_fee + slip)
    eu_m = expected_utility(grid, cost_m, mu, sigma, maint, s)
    i_m = int(np.argmax(eu_m))
    ttl = float(min(max(fc.half_life_s, 1.0), 15.0))
    buying_m = grid[i_m] > f_cur
    rate = fc.sell_rate if buying_m else fc.buy_rate
    queue = float(book.bids[0, 1] if buying_m else book.asks[0, 1])
    own = abs(grid[i_m] - f_cur) * eq / mid
    p_fill = 1.0 - math.exp(-rate * ttl / (queue + own)) if queue + own > 0 else 0.0
    eu_maker = p_fill * float(eu_m[i_m]) + (1 - p_fill) * eu_hold

    use_maker = opened[i_m] > 0 and eu_maker > float(eu_t[i_t])
    i_best, eu_best = (i_m, eu_maker) if use_maker else (i_t, float(eu_t[i_t]))
    f_tgt = float(grid[i_best])
    gain = eu_best - eu_hold

    d = Decision(now, "HOLD", "", f_cur, f_cur, eq, eu_gain=gain, expected_edge_bps=fc.mu_bps)
    if gain <= 1e-9 or i_best == i_hold:
        d.reason = (
            "no validated edge (calibration beta = 0)" if fc.beta == 0
            else "current exposure is already the utility-maximising one"
        )  # fmt: skip
        return d

    delta = (f_tgt - f_cur) * eq / mid
    is_buy = delta > 0
    closing = f_tgt == 0.0
    sz = abs(acct.position) if closing else meta.round_sz(abs(delta))
    if sz <= 0 or (sz * mid < meta.min_notional and not closing):
        d.reason = f"optimal change ${abs(delta) * mid:.2f} is below the venue minimum order"
        return d

    reduce_only = opened[i_best] == 0
    if use_maker:
        px = book.best_bid if is_buy else book.best_ask
        intent = OrderIntent(meta.coin, is_buy, sz, px, "Alo", reduce_only, ttl)
        d.exec_style = "maker"
        d.exec_reason = (
            f"half-life {fc.half_life_s:.1f}s vs fill probability {p_fill:.2f} within {ttl:.1f}s: "
            "posting has higher expected utility than crossing"
        )
    else:
        # Marketable limit: cross, but refuse to chase more than the edge is worth (min 5 bps).
        tol = max(abs(fc.mu_bps), 5.0) * 1e-4
        px = meta.round_px(book.best_ask * (1 + tol) if is_buy else book.best_bid * (1 - tol))
        intent = OrderIntent(meta.coin, is_buy, sz, px, "Ioc", reduce_only)
        d.exec_style = "taker"
        d.exec_reason = (
            "reducing exposure: certainty of execution preferred" if reduce_only
            else f"half-life {fc.half_life_s:.1f}s: crossing has higher expected utility than queueing"
        )  # fmt: skip

    side = 1.0 if f_tgt > 0 else -1.0
    z = side * mu / sigma if sigma > 0 else 0.0
    d.action = "BUY" if is_buy else "SELL"
    d.f_target = f_tgt
    d.order = intent
    d.p_adverse = 0.5 * math.erfc(z / math.sqrt(2.0)) if f_tgt else 0.5
    d.loss_1sigma_pct = abs(f_tgt) * sigma * 100
    d.liq_distance_pct = (1.0 / abs(f_tgt) - maint) * 100 if f_tgt else float("inf")
    d.cost_bps = float((cost_m if use_maker else cost_t)[i_best] / max(opened[i_best], 1e-12)) * 1e4
    d.reason = (
        "edge gone or reversed: reducing risk" if reduce_only
        else "expected utility of new exposure exceeds holding, net of round-trip cost"
    )  # fmt: skip
    return d
