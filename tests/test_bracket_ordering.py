from pathlib import Path
import yaml

CFG = Path(__file__).resolve().parents[1] / "config" / "bracket_2026.yaml"

def _r16_pairs(teams):
    # Each consecutive block of 4 forms one R16 match between the two ties' slots.
    pairs = []
    for i in range(0, len(teams), 4):
        block = teams[i:i+4]
        pairs.append(((block[0], block[1]), (block[2], block[3])))
    return pairs

def test_slot_order_reproduces_real_r16_pairings():
    teams = yaml.safe_load(CFG.read_text(encoding="utf-8"))["teams"]
    assert len(teams) == 32
    pairs = _r16_pairs(teams)
    # M89 = (M74 Germany/Paraguay) vs (M77 France/Sweden)
    assert pairs[0] == (("Germany", "Paraguay"), ("France", "Sweden"))
    # M90 = (M73 Canada/SA) vs (M75 Netherlands/Morocco)
    assert pairs[1] == (("Canada", "South Africa"), ("Netherlands", "Morocco"))
    # M91 = (M76 Brazil/Japan) vs (M78 Ivory Coast/Norway)
    assert pairs[4] == (("Brazil", "Japan"), ("Ivory Coast", "Norway"))
    # M95 = (M86 Argentina/Cape Verde) vs (M88 Australia/Egypt)
    assert pairs[6] == (("Argentina", "Cape Verde"), ("Australia", "Egypt"))
