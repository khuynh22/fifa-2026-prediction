from fifa2026.config import load_config, RANDOM_SEED

def test_load_config_defaults():
    cfg = load_config()
    assert cfg.train_start == "2010-01-01"
    assert cfg.random_seed == RANDOM_SEED == 42
    assert cfg.raw["hosts_2026"] == ["United States", "Mexico", "Canada"]
    assert str(cfg.raw_dir).replace("\\", "/").endswith("data/raw")
