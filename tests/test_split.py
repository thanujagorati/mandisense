"""Tests for FPO split mode. Run from the mandisense folder:  python -m pytest -q"""
import pytest

from app.split import Outlet, SplitAssumptions, optimise

HOME = Outlet("Small", price_rs_qtl=1000.0, daily_arrivals_qtl=200.0, road_km=0.0)
BIG = Outlet("Big", price_rs_qtl=1000.0, daily_arrivals_qtl=2000.0, road_km=100.0)


def test_no_price_effect_means_no_reason_to_split():
    res = optimise([HOME, BIG], 80, SplitAssumptions(flexibility=0.0))
    assert res.gain_rs == pytest.approx(0.0)
    assert set(res.best_split.allocation) == {"Small"}   # stays home, no transport


def test_strong_price_effect_makes_splitting_worth_it():
    res = optimise([HOME, BIG], 80, SplitAssumptions(flexibility=-0.3, min_load_qtl=10))
    assert res.gain_rs > 0
    assert set(res.best_split.allocation) == {"Small", "Big"}
    assert sum(res.best_split.allocation.values()) == pytest.approx(80)
    assert res.best_split.net_rs > res.best_single.net_rs


def test_minimum_load_blocks_tiny_splits():
    res = optimise([HOME, BIG], 40, SplitAssumptions(flexibility=-0.3, min_load_qtl=30))
    assert len(res.best_split.allocation) == 1


def test_price_drop_is_capped():
    tiny = Outlet("Tiny", 1000.0, 1.0, 0.0)
    res = optimise([tiny], 100, SplitAssumptions(flexibility=-0.3, max_price_drop=0.5))
    assert res.best_single.revenue_rs == pytest.approx(100 * 1000.0 * 0.5)


def test_bad_inputs_are_rejected():
    with pytest.raises(ValueError):
        optimise([], 80)
    with pytest.raises(ValueError):
        optimise([HOME], 0)
    with pytest.raises(ValueError):
        optimise([HOME], 80, SplitAssumptions(flexibility=0.2))
