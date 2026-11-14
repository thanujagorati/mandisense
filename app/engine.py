"""Decision engine: turns a price forecast + current prices into SELL NOW / WAIT / MOVE.

Pure functions, no database, so it is easy to test and easy to explain.
Every number that is an ASSUMPTION lives in `Assumptions`, and the app shows them to the user.
Money is in rupees (Rs), quantity in quintals (1 quintal = 100 kg, 1 tonne = 10 quintals).

Key design point: mandis keep persistent price gaps (for example Bowenpally usually quotes about
20% above Gudimalkapur, probably because of grade mix and reporting). Those gaps are not
opportunities. So a destination's price is divided by its USUAL ratio to the home mandi, and only
the unusual part of the gap counts.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import asin, cos, radians, sin, sqrt

# Typical 7-day move when the outcome is down / flat / up. The demo script replaces these with
# values measured from our own data; these are only fallbacks.
DEFAULT_MEAN_RETURNS = {"down": -0.06, "flat": 0.0, "up": 0.06}


@dataclass(frozen=True)
class Assumptions:
    """Placeholders to be checked against real local numbers. All adjustable."""
    freight_rs_per_qtl_km: float = 0.30       # truck freight: Rs per quintal per km (assumed)
    road_factor: float = 1.30                 # road distance is longer than straight-line distance
    handling_rs_per_qtl: float = 15.0         # loading / unloading when moving produce to another mandi
    holding_cost_pct_per_day: float = 0.003   # shrinkage + cost of money while waiting (0.3% per day)
    wait_days: int = 7                        # matches the forecast horizon
    min_advantage_pct: float = 0.02           # WAIT must beat selling now by at least 2%
    move_min_advantage_pct: float = 0.06      # MOVE must beat selling now by at least 6% (price gaps are noisy)
    min_distance_km: float = 10.0             # coordinates are approximate, so never treat a trip as free


@dataclass(frozen=True)
class Forecast:
    p_down: float
    p_flat: float
    p_up: float

    def __post_init__(self):
        probs = (self.p_down, self.p_flat, self.p_up)
        if any(p < 0 for p in probs) or abs(sum(probs) - 1.0) > 0.01:
            raise ValueError("Probabilities must be non-negative and add up to 1.")

    @property
    def top_probability(self) -> float:
        return max(self.p_down, self.p_flat, self.p_up)


@dataclass(frozen=True)
class MandiQuote:
    name: str
    price_rs_qtl: float
    lat: float | None = None
    lon: float | None = None
    # How this mandi usually prices compared with the HOME mandi (1.20 = usually 20% higher).
    usual_ratio: float = 1.0


@dataclass(frozen=True)
class Option:
    action: str      # SELL_NOW, WAIT or MOVE
    mandi: str
    gross_rs: float  # expected sale value before costs
    cost_rs: float   # transport / holding costs
    net_rs: float    # what the farmer is expected to keep
    note: str


@dataclass(frozen=True)
class Recommendation:
    action: str
    mandi: str
    options: list
    advantage_rs: float
    advantage_pct: float
    confidence: str
    reasons: list


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Straight-line distance between two points on Earth, in km."""
    p1, p2 = radians(lat1), radians(lat2)
    dphi = p2 - p1
    dlmb = radians(lon2 - lon1)
    h = sin(dphi / 2) ** 2 + cos(p1) * cos(p2) * sin(dlmb / 2) ** 2
    return 2 * 6371.0 * asin(sqrt(h))


def road_distance_km(a: MandiQuote, b: MandiQuote, assumptions: Assumptions) -> float | None:
    if None in (a.lat, a.lon, b.lat, b.lon):
        return None
    d = haversine_km(a.lat, a.lon, b.lat, b.lon) * assumptions.road_factor
    return max(d, assumptions.min_distance_km)


def transport_cost(distance_km: float, quintals: float, assumptions: Assumptions) -> float:
    per_quintal = assumptions.freight_rs_per_qtl_km * distance_km + assumptions.handling_rs_per_qtl
    return per_quintal * quintals


def expected_return(forecast: Forecast, mean_returns: dict) -> float:
    """Probability-weighted typical price change over the next 7 days."""
    return (forecast.p_down * mean_returns["down"]
            + forecast.p_flat * mean_returns["flat"]
            + forecast.p_up * mean_returns["up"])


def confidence_label(forecast: Forecast) -> str:
    """In our backtest, forecasts with a top probability of 0.60 or more were right about 66% of the time."""
    p = forecast.top_probability
    if p >= 0.60:
        return "high"
    if p >= 0.45:
        return "moderate"
    return "low"


def recommend(home: MandiQuote, others: list, forecast: Forecast, quintals: float,
              mean_returns: dict | None = None,
              assumptions: Assumptions | None = None) -> Recommendation:
    if quintals <= 0:
        raise ValueError("Quantity must be positive.")
    a = assumptions or Assumptions()
    mr = mean_returns or DEFAULT_MEAN_RETURNS

    sell_value = home.price_rs_qtl * quintals
    sell = Option("SELL_NOW", home.name, sell_value, 0.0, sell_value, "Sell at the home mandi today")

    exp_ret = expected_return(forecast, mr)
    wait_gross = sell_value * (1 + exp_ret)
    hold_cost = a.holding_cost_pct_per_day * a.wait_days * sell_value
    wait = Option("WAIT", home.name, wait_gross, hold_cost, wait_gross - hold_cost,
                  f"Hold {a.wait_days} days and sell at the expected price")

    options = [sell, wait]
    for m in others:
        d = road_distance_km(home, m, a)
        if d is None:
            continue  # no coordinates, so we cannot honestly estimate transport
        # Home-equivalent price: remove the usual gap, keep only the unusual part.
        adjusted = m.price_rs_qtl / m.usual_ratio
        gross = adjusted * quintals
        cost = transport_cost(d, quintals, a)
        options.append(Option(
            "MOVE", m.name, gross, cost, gross - cost,
            f"{d:.0f} km by road; quoted Rs {m.price_rs_qtl:,.0f}, usually {m.usual_ratio:.2f}x home, "
            f"so worth Rs {adjusted:,.0f} at home quality"))

    # An alternative only wins if it clears its own margin over selling now.
    qualifying = []
    if wait.net_rs / sell.net_rs - 1 >= a.min_advantage_pct:
        qualifying.append(wait)
    for o in options:
        if o.action == "MOVE" and o.net_rs / sell.net_rs - 1 >= a.move_min_advantage_pct:
            qualifying.append(o)
    chosen = max(qualifying, key=lambda o: o.net_rs) if qualifying else sell

    advantage = chosen.net_rs - sell.net_rs
    reasons = [
        f"Price today at {home.name}: Rs {home.price_rs_qtl:,.0f} per quintal.",
        (f"Model outlook for the next 7 days: down {forecast.p_down:.0%}, flat {forecast.p_flat:.0%}, "
         f"up {forecast.p_up:.0%} (expected change {exp_ret:+.1%} before holding costs)."),
    ]
    if chosen.action == "SELL_NOW":
        bar = f"Waiting must beat selling now by {a.min_advantage_pct:.0%}"
        if any(o.action == "MOVE" for o in options):
            bar += f" and moving by {a.move_min_advantage_pct:.0%}"
        reasons.append(bar + " after costs; nothing clears that bar, so selling now is the safe choice.")
        confidence = confidence_label(forecast)
    elif chosen.action == "WAIT":
        reasons.append(
            f"Waiting is expected to keep Rs {advantage:,.0f} more ({advantage / sell.net_rs:+.1%}) "
            f"after about Rs {hold_cost:,.0f} of holding cost, but it depends on the forecast being right.")
        confidence = confidence_label(forecast)
    else:
        reasons.append(
            f"{chosen.mandi} is paying unusually well compared with its normal gap to {home.name} "
            f"({chosen.note}). After transport of about Rs {chosen.cost_rs:,.0f}, that is "
            f"Rs {advantage:,.0f} more ({advantage / sell.net_rs:+.1%}).")
        reasons.append("Based on reported prices, which vary from day to day, and it assumes your "
                       "onions fetch the usual price for their quality there.")
        confidence = "moderate (based on reported prices, which vary day to day)"

    return Recommendation(chosen.action, chosen.mandi, options, advantage, advantage / sell.net_rs,
                          confidence, reasons)
