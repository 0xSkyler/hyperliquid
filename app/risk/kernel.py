"""Safety kernel: the only absolute limits in the system.

It makes no market judgments. It answers one question: can we trust our instruments?
If not, the engine places no orders until the fault clears.
"""

from __future__ import annotations

from app.config.settings import Settings
from app.exchange.base import AccountState, Book


class SafetyKernel:
    def __init__(self, s: Settings) -> None:
        self.s = s
        self.expected_pos: float | None = None
        self._mismatch_since: float | None = None

    def on_fill(self, signed_sz: float) -> None:
        if self.expected_pos is not None:
            self.expected_pos += signed_sz

    def check(
        self, now: float, book: Book | None, acct: AccountState, lot: float, feed_ts: float | None = None
    ) -> list[str]:
        faults: list[str] = []
        if book is None:
            faults.append("no_market_data")
        elif not book.valid():
            faults.append("crossed_or_empty_book")
        elif now - (book.ts if feed_ts is None else feed_ts) > self.s.stale_book_s:
            faults.append("stale_market_data")  # the connection itself has gone quiet
        elif now - book.ts > self.s.stale_depth_s:
            faults.append("stale_order_book")

        if not acct.known:
            faults.append("account_state_unknown")
            return faults
        if now - acct.ts > self.s.stale_account_s:
            faults.append("stale_account_state")
        if acct.inflight and now - acct.oldest_inflight_ts > self.s.unacked_order_s:
            faults.append("unacknowledged_order")

        if self.expected_pos is None:
            self.expected_pos = acct.position  # startup: adopt the exchange's truth, never assume flat
        if abs(self.expected_pos - acct.position) > lot:
            if self._mismatch_since is None:
                self._mismatch_since = now
            # Fills and account snapshots can arrive a moment apart; persistent disagreement is a fault.
            if now - self._mismatch_since > self.s.stale_account_s:
                faults.append("position_mismatch")
        else:
            self._mismatch_since = None
        return faults
