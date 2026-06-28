import numpy as np
import pandas as pd
from fifa2026.features.squad_features import team_aggregates, impute_tier_b

def _players():
    return pd.DataFrame({
        "team": ["Brazil", "Brazil", "France"],
        "player": ["A", "B", "C"], "position": ["Attacker", "Defender", "Attacker"],
        "age": [26, 31, 24], "market_value_eu": [90e6, 25e6, 80e6],
        "season_minutes": [2700, 2500, 2600], "season_xg": [18.2, 1.0, 15.0],
        "season_xa": [7.1, 1.4, 6.0], "injured": [False, True, False],
    })

def test_team_aggregates():
    agg = team_aggregates(_players())
    assert agg.loc["Brazil", "squad_value"] == 115e6
    assert agg.loc["Brazil", "top_xg"] == 18.2
    assert agg.loc["Brazil", "n_injured"] == 1

def test_impute_adds_indicator_and_fills():
    df = pd.DataFrame({"squad_value": [100.0, np.nan]})
    out = impute_tier_b(df)
    assert out["squad_value_isna"].tolist() == [0, 1]
    assert out["squad_value"].tolist() == [100.0, 100.0]  # median fill
