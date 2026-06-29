from fifa2026.pipeline import PredictionResult
from fifa2026 import viz


def _result():
    champ = {"France": 0.25, "Brazil": 0.2, "Spain": 0.18, "Argentina": 0.15,
             "England": 0.12, "Germany": 0.1}
    rounds = {t: {"reach_R16": 1.0, "reach_QF": 0.6, "reach_SF": 0.4,
                  "reach_final": 0.3, "win": p} for t, p in champ.items()}
    ties = [{"home": "France", "away": "Brazil", "p_home": 0.55}]
    return PredictionResult(champion_probs=champ, round_probs=rounds, tie_probs=ties, as_of="2026-06-28")


def test_champion_bar_has_data():
    fig = viz.champion_bar(_result(), top_n=3)
    assert len(fig.data) >= 1
    # sorted descending, top team first
    assert list(fig.data[0].x)[0] == "France" or list(fig.data[0].y)[0] == "France"


def test_market_compare_two_series():
    fig = viz.market_compare({"France": 0.25, "Brazil": 0.2},
                             {"France": 0.2, "Brazil": 0.25}, top_n=2)
    assert len(fig.data) == 2  # model + market


def test_survival_heatmap_builds():
    fig = viz.survival_heatmap(_result(), top_n=5)
    assert fig is not None and len(fig.data) >= 1
