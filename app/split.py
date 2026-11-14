"""FPO split mode: is it better to share a big lot between mandis than to send it all to one?

Why it can pay off: one mandi only absorbs so much per day. Pour a lot on top of its normal
arrivals and the price drops for the WHOLE lot. Sharing the lot softens that drop, and the saving
can beat the extra transport.

Important honesty rule: this module does NOT chase price differences between mandis (we found those
too noisy to trust). Every outlet is assumed to pay the same base price, so any gain from splitting
comes only from the volume effect, net of transport.

Model (all assumptions are explicit and adjustable):
    price paid for the whole lot = base price * (1 + flexibility * lot / usual daily arrivals)
    flexibility is negative, e.g. -0.25: arrivals 10% above usual push the price down about 2.5%.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import ceil

from app.engine import Assumptions, transport_cost


@dataclass(frozen=True)
class Outlet:
    name: str
    price_rs_qtl: float        # base price for our lot here, before our own effect on the price
    daily_arrivals_qtl: float  # usual quintals arriving per day
    road_km: float             # 0 for the home mandi


@dataclass(frozen=True)
class SplitAssumptions:
    flexibility: float = -0.25     # % price change per 1% extra arrivals (must be <= 0)
    max_price_drop: float = 0.50   # never assume the price falls by more than this
    min_load_qtl: float = 30.0     # smallest load worth sending to one mandi (a small truck)
    max_outlets: int = 3           # sharing across more than 3 mandis is not practical


@dataclass(frozen=True)
class Plan:
    allocation: dict   # mandi name -> quintals
    revenue_rs: float
    transport_rs: float
    net_rs: float


@dataclass(frozen=True)
class SplitResult:
    best_single: Plan
    best_split: Plan
    gain_rs: float
    gain_pct: float


def sale_revenue(o: Outlet, qtl: float, s: SplitAssumptions) -> float:
    if qtl <= 0:
        return 0.0
    change = s.flexibility * qtl / o.daily_arrivals_qtl      # negative number
    factor = max(1.0 + change, 1.0 - s.max_price_drop)
    return qtl * o.price_rs_qtl * factor


def haul_cost(o: Outlet, qtl: float, a: Assumptions) -> float:
    return 0.0 if o.road_km <= 0 else transport_cost(o.road_km, qtl, a)


def _plan(parts: list, s: SplitAssumptions, a: Assumptions) -> Plan:
    revenue = sum(sale_revenue(o, q, s) for o, q in parts)
    transport = sum(haul_cost(o, q, a) for o, q in parts)
    return Plan({o.name: round(q, 1) for o, q in parts}, revenue, transport, revenue - transport)


def _compositions(n: int, k: int, m: int):
    """All ways to write n as k whole numbers, each at least m."""
    if k == 1:
        if n >= m:
            yield (n,)
        return
    for first in range(m, n - m * (k - 1) + 1):
        for rest in _compositions(n - first, k - 1, m):
            yield (first,) + rest


def optimise(outlets: list, total_qtl: float, s: SplitAssumptions | None = None,
             a: Assumptions | None = None) -> SplitResult:
    s = s or SplitAssumptions()
    a = a or Assumptions()
    if not outlets:
        raise ValueError("Need at least one outlet.")
    if total_qtl <= 0:
        raise ValueError("The lot must be positive.")
    if s.flexibility > 0:
        raise ValueError("Flexibility must be zero or negative.")
    if any(o.daily_arrivals_qtl <= 0 for o in outlets):
        raise ValueError("Every outlet needs positive usual arrivals.")

    best_single = max((_plan([(o, total_qtl)], s, a) for o in outlets), key=lambda p: p.net_rs)
    best = best_single

    n = max(1, round(total_qtl / 10))        # work in chunks of about 10 quintals (1 tonne)
    chunk = total_qtl / n
    min_chunks = max(1, ceil(s.min_load_qtl / chunk))

    for k in range(2, min(s.max_outlets, len(outlets)) + 1):
        if n < k * min_chunks:
            break
        for combo in combinations(outlets, k):
            for comp in _compositions(n, k, min_chunks):
                plan = _plan([(o, c * chunk) for o, c in zip(combo, comp)], s, a)
                if plan.net_rs > best.net_rs:
                    best = plan

    gain = best.net_rs - best_single.net_rs
    return SplitResult(best_single, best, gain, gain / best_single.net_rs)
