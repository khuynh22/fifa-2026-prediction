import importlib


def test_build_dashboard_returns_figures():
    app = importlib.import_module("app")
    prediction = {
        "champion_probs": {"France": 0.3, "Brazil": 0.25, "Spain": 0.2, "England": 0.25},
        "round_probs": {t: {"reach_R16": 1.0, "reach_QF": 0.5, "reach_SF": 0.4,
                            "reach_final": 0.3, "win": p}
                        for t, p in {"France":0.3,"Brazil":0.25,"Spain":0.2,"England":0.25}.items()},
        "tie_probs": [{"home": "France", "away": "Brazil", "p_home": 0.55}],
        "as_of": "2026-06-28", "meta": {},
    }
    figs = app.build_dashboard(prediction, {"metrics": {"log_loss": 1.0}, "calibration": []}, None)
    assert "champion" in figs and "bracket" in figs and "survival" in figs
