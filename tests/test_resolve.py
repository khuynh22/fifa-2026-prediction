import numpy as np
from fifa2026.knockout.resolve import resolve_tie, shootout_prob

def test_shootout_is_near_even_by_default():
    assert abs(shootout_prob(0.5, 0.5, 0.0, 0.0) - 0.5) < 1e-9

def test_better_penalty_team_favored():
    assert shootout_prob(0.8, 0.2, 0.0, 0.0) > 0.5

def test_resolve_tie_accounts_for_draw_split():
    # 50% reg win for a, 20% draw, 30% reg win for b; even shootout
    p = np.array([0.5, 0.2, 0.3])
    adv = resolve_tie(p, pen_a=0.5, pen_b=0.5)
    assert abs(adv - (0.5 + 0.2 * 0.5)) < 1e-9   # 0.6

def test_resolve_tie_full_certainty():
    p = np.array([1.0, 0.0, 0.0])
    assert resolve_tie(p) == 1.0
