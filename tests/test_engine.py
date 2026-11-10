"""Tests for the decision engine. Run from the mandisense folder:  python -m pytest -q"""
import pytest

from app.engine import (Forecast, MandiQuote, confidence_label, haversine_km, recommend)

HOME = MandiQuote("Home", 1500.0, 17.0, 78.0)
FLAT = Forecast(p_down=0.2, p_flat=0.6, p_up=0.2)


def test_haversine_hyderabad_pune_is_reasonable():
    d = haversine_km(17.385, 78.487, 18.520, 73.857)
    assert 450 < d < 560


def test_strong_up_forecast_recommends_wait():
    f = Forecast(p_down=0.1, p_flat=0.1, p_up=0.8)
    rec = recommend(MandiQuote("Home", 2000.0, 17.0, 78.0), [], f, 80,
                    mean_returns={"down": -0.06, "flat": 0.0, "up": 0.10})
    assert rec.action == "WAIT"


def test_much_better_price_nearby_recommends_move():
    f = Forecast(p_down=0.7, p_flat=0.2, p_up=0.1)
    other = MandiQuote("Near", 2200.0, 17.9, 78.0)   # about 100 km north
    rec = recommend(HOME, [other], f, 80)
    assert rec.action == "MOVE" and rec.mandi == "Near"
    assert rec.confidence.startswith("moderate")


def test_transport_cost_can_cancel_a_better_price():
    far = MandiQuote("Far", 1550.0, 21.5, 78.0)       # about 500 km north
    assert recommend(HOME, [far], FLAT, 80).action == "SELL_NOW"


def test_small_edge_below_margin_stays_with_sell_now():
    twin = MandiQuote("Twin", 1530.0, 17.0, 78.0)     # same place, +2% price, minus handling
    assert recommend(HOME, [twin], FLAT, 80).action == "SELL_NOW"


def test_usual_gap_is_not_an_opportunity():
    # Destination always pays about 25% more, so 1875 is just its normal level (worth 1500 at home).
    usual = MandiQuote("Usually higher", 1875.0, 17.9, 78.0, usual_ratio=1.25)
    assert recommend(HOME, [usual], FLAT, 80).action == "SELL_NOW"


def test_unusual_gap_counts_after_adjustment():
    # Normally 1.20x home, today 1.40x home: the extra part is a real signal.
    unusual = MandiQuote("Unusually high", 2100.0, 17.9, 78.0, usual_ratio=1.20)   # worth 1750 at home
    rec = recommend(HOME, [unusual], FLAT, 80)
    assert rec.action == "MOVE" and rec.mandi == "Unusually high"


def test_move_needs_more_than_a_four_percent_edge():
    # About +4% after transport: enough for WAIT's bar, not enough for MOVE's 6% bar.
    modest = MandiQuote("Modest", 1620.0, 17.9, 78.0)
    assert recommend(HOME, [modest], FLAT, 80).action == "SELL_NOW"


def test_probabilities_must_add_up():
    with pytest.raises(ValueError):
        Forecast(p_down=0.5, p_flat=0.5, p_up=0.5)


def test_quantity_must_be_positive():
    with pytest.raises(ValueError):
        recommend(HOME, [], Forecast(0.3, 0.4, 0.3), 0)


def test_confidence_labels():
    assert confidence_label(Forecast(0.1, 0.2, 0.7)) == "high"
    assert confidence_label(Forecast(0.2, 0.3, 0.5)) == "moderate"
    assert confidence_label(Forecast(0.34, 0.33, 0.33)) == "low"
