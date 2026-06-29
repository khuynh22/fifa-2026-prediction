from pathlib import Path
from fifa2026.ingest.reference import load_confederations, load_venues

REF = Path(__file__).resolve().parents[1] / "data" / "reference"

def test_confederations_cover_2026_hosts():
    conf = load_confederations(REF / "confederations.csv")
    for host in ("United States", "Mexico", "Canada"):
        assert conf[host] == "CONCACAF"
    assert conf["Brazil"] == "CONMEBOL"
    assert conf["Morocco"] == "CAF"
    assert len(conf) >= 32

def test_venues_have_mexico_city_altitude():
    venues = load_venues(REF / "venues_2026.csv")
    mc = venues[venues["city"] == "Mexico City"].iloc[0]
    assert mc["altitude_m"] == 2240
