from fifa2026.squad_strength import load_injuries, availability_adjustment

def test_availability_adjustment_counts_and_caps():
    inj = {"France": ["A", "B"], "Spain": ["C"], "Brazil": [], "Italy": ["D","E","F","G","H"]}
    adj = availability_adjustment(inj, penalty_per_player=10.0, cap=40.0)
    assert adj["France"] == -20.0      # 2 * 10
    assert adj["Spain"] == -10.0       # 1 * 10
    assert "Brazil" not in adj         # empty list -> no adjustment
    assert adj["Italy"] == -40.0       # 5 * 10 capped at 40

def test_load_injuries_parses_and_missing_file(tmp_path):
    p = tmp_path / "inj.yaml"
    p.write_text("injuries:\n  France: [A, B]\n  Spain: [C]\n", encoding="utf-8")
    got = load_injuries(p)
    assert got == {"France": ["A", "B"], "Spain": ["C"]}
    assert load_injuries(tmp_path / "nope.yaml") == {}   # missing -> empty
