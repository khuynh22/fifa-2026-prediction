from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml

RANDOM_SEED = 42
_DEFAULT = Path(__file__).resolve().parents[2] / "config" / "default.yaml"

@dataclass(frozen=True)
class Config:
    data_dir: Path
    raw_dir: Path
    processed_dir: Path
    models_dir: Path
    reports_dir: Path
    train_start: str
    random_seed: int
    raw: dict

def load_config(path: str | None = None) -> Config:
    cfg_path = Path(path) if path else _DEFAULT
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    p = raw["paths"]
    return Config(
        data_dir=Path(p["data_dir"]),
        raw_dir=Path(p["raw_dir"]),
        processed_dir=Path(p["processed_dir"]),
        models_dir=Path(p["models_dir"]),
        reports_dir=Path(p["reports_dir"]),
        train_start=raw["train_start"],
        random_seed=raw["random_seed"],
        raw=raw,
    )
