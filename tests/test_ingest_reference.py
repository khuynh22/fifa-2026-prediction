from pathlib import Path
from fifa2026.ingest.reference import (
    load_confederations, load_venues, team_country_coords, is_host,
)

FIX = Path(__file__).parent / "fixtures"

def test_confederations():
    conf = load_confederations(FIX / "confederations.csv")
    assert conf["Brazil"] == "CONMEBOL"
    assert conf["Germany"] == "UEFA"

def test_venues_and_coords():
    venues = load_venues(FIX / "venues.csv")
    assert {"city", "country", "lat", "lon", "altitude_m"} <= set(venues.columns)
    coords = team_country_coords(venues)
    assert coords["Mexico"] == (19.43, -99.13)

def test_is_host():
    assert is_host("Mexico", ["United States", "Mexico", "Canada"])
    assert not is_host("Brazil", ["United States", "Mexico", "Canada"])
