"""Tests for the decision engine. Run from the mandisense folder:  python -m pytest -q"""
import pytest

from app.engine import (Forecast, MandiQuote, confidence_label, haversine_km, recommend)

HOME = MandiQuote("Home", 1500.0, 17.0, 78.0)


def test_haversine_hyderabad_pune_is_reasonable():
    d = haversine_km(17.385, 78.487, 18.520, 73.857)
    assert 450 < d < 560


def test_strong_up_forecast_recommends_wait():
    f = Forecast(p_down=0.1, p_flat=0.1, p_up=0.8)
    rec = recommend(MandiQuote("Home", 2000.0, 17.0, 78.0), [], f, 80,
                    mean_returns={"down": -0.06, "flat": 0.0, "up": 0.10})
    assert rec.action == "WAIT"


def test_better_price_nearby_recommends_move():
    f = Forecast(p_down=0.7, p_flat=0.2, p_up=0.1)
    other = MandiQuote("Near", 2200.0, 17.9, 78.0)   # about 100 km north
    rec = recommend(HOME, [other], f, 80)
    assert rec.action == "MOVE" and rec.mandi == "Near"


def test_transport_cost_can_cancel_a_better_price():
    f = Forecast(p_down=0.2, p_flat=0.6, p_up=0.2)
    far = MandiQuote("Far", 1550.0, 21.5, 78.0)       # about 500 km north
    rec = recommend(HOME, [far], f, 80)
    assert rec.action == "SELL_NOW"


def test_small_edge_below_margin_stays_with_sell_now():
    f = Forecast(p_down=0.2, p_flat=0.6, p_up=0.2)
    twin = MandiQuote("Twin", 1530.0, 17.0, 78.0)     # same place, +2% price, minus handling
    rec = recommend(HOME, [twin], f, 80)
    assert rec.action == "SELL_NOW"


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
