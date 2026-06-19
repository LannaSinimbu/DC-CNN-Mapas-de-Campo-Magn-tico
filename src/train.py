"""
Treinamento da U-Net 2D.

Uso:
    python -m src.train --config configs/config.yaml

Lê os mapas, monta o dataset aumentado, gera os pares entrada/alvo via
downward continuation + deconvolução, treina a rede e salva os pesos.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .config import load_config, resolve_paths
from .data import (
    load_maps,
    build_augmented_dataset,
    build_pairs,
    extract_patches,
    BzPatchDataset,
)
from .model import UNet2D


def main():
    parser = argparse.ArgumentParser(description="Treinar a U-Net 2D para mapas Bz.")
    parser.add_argument("--config", default="configs/config.yaml",
                        help="Caminho do arquivo de configuração YAML.")
    args = parser.parse_args()

    project_root = Path(args.config).resolve().parent.parent
    cfg = resolve_paths(load_config(args.config), project_root)

    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    phys_cfg = cfg["physics"]

    torch.manual_seed(train_cfg.get("seed", 0))
    np.random.seed(train_cfg.get("seed", 0))

    # --- Dados ---
    dx, dy = phys_cfg["dx"], phys_cfg["dy"]
    z_list = [z * 1e-6 for z in data_cfg["z_um"]]
    raw_maps = load_maps(data_cfg["map_files"])
    target_shape = raw_maps[0].shape
    print("Forma de referência:", target_shape)

    bz_maps, z_array = build_augmented_dataset(
        raw_maps, z_list, target_shape, seed=train_cfg.get("seed", 0)
    )
    print("Dataset aumentado:", bz_maps.shape)

    inputs, targets, idx_ref, z_ref = build_pairs(
        bz_maps, z_array, dx, dy,
        alpha_dc=phys_cfg["alpha_dc"],
        sigma_psf=phys_cfg["sigma_psf"],
        beta=phys_cfg["beta"],
    )
    print(f"Plano de referência: idx={idx_ref}, z_ref={z_ref*1e6:.1f} um")
    print("Pares:", inputs.shape[0])

    in_patches = extract_patches(inputs, train_cfg["patch_size"], train_cfg["stride"])
    tg_patches = extract_patches(targets, train_cfg["patch_size"], train_cfg["stride"])
    print("Patches de entrada:", in_patches.shape)

    loader = DataLoader(
        BzPatchDataset(in_patches, tg_patches),
        batch_size=train_cfg["batch_size"],
        shuffle=True,
    )

    # --- Modelo ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Dispositivo:", device)
    model = UNet2D(in_ch=1, out_ch=1).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=train_cfg["lr"])

    # --- Loop de treino ---
    n_epochs = train_cfg["n_epochs"]
    for epoch in range(n_epochs):
        model.train()
        running = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)
        epoch_loss = running / len(loader.dataset)
        if (epoch + 1) % train_cfg.get("log_every", 10) == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{n_epochs} - Loss: {epoch_loss:.4e}")

    # --- Salvar ---
    out_path = project_root / train_cfg["model_path"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print("Modelo salvo em:", out_path)


if __name__ == "__main__":
    main()
