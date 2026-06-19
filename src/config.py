"""Carregamento da configuração a partir de um arquivo YAML."""

from pathlib import Path
import yaml


def load_config(path):
    """Lê um arquivo YAML e retorna um dicionário."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_paths(cfg, base_dir):
    """Resolve caminhos relativos de mapas em relação a ``base_dir``."""
    base = Path(base_dir)
    cfg = dict(cfg)
    data = dict(cfg.get("data", {}))
    if "map_files" in data:
        data["map_files"] = [str((base / p).resolve()) for p in data["map_files"]]
    cfg["data"] = data
    return cfg
