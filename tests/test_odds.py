from fifa2026.ingest.odds import parse_winner_odds, implied_champion_probs

def test_parse_and_implied():
    rows = [{"team": "Brazil", "decimal_odds": 5.0},
            {"team": "France", "decimal_odds": 6.0},
            {"team": "Spain", "decimal_odds": 7.0}]
    odds = parse_winner_odds(rows)
    assert odds["Brazil"] == 5.0
    probs = implied_champion_probs(odds)
    assert abs(sum(probs.values()) - 1.0) < 1e-9
    assert probs["Brazil"] > probs["Spain"]  # shorter odds -> higher prob
