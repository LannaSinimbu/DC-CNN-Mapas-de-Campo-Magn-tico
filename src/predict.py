"""
Predição: gera um mapa Bz em uma nova altura usando o modelo treinado.

Uso:
    python -m src.predict --config configs/config.yaml \
        --input data/MAPx260620241658.txt \
        --z-from 200 --z-to 140 \
        --output outputs/Bz_140um_CNN.txt

Pipeline: downward continuation -> deconvolução -> recorte para múltiplo de 8
-> normalização -> CNN -> desnormalização -> salvar (e opcionalmente plotar).
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from .config import load_config, resolve_paths
from .data import load_map
from .model import UNet2D, crop_to_mult8
from .physics import get_k_grid, downward_continuation, deconvolve_footprint


def predict_map(bz, z_from, z_to, model, device, dx, dy,
                alpha_dc, sigma_psf, beta):
    ny, nx = bz.shape
    _, _, k = get_k_grid(nx, ny, dx, dy)

    bz_dc = downward_continuation(bz, z_from, z_to, k, alpha_dc)
    bz_dc_deconv = deconvolve_footprint(bz_dc, k, sigma_psf, beta)

    bz_crop = crop_to_mult8(bz_dc_deconv)
    scale = np.max(np.abs(bz_crop)) or 1.0
    bz_norm = bz_crop / scale

    with torch.no_grad():
        x_t = torch.from_numpy(bz_norm[None, None, ...]).float().to(device)
        y_pred = model(x_t).cpu().numpy()[0, 0]

    return y_pred * scale, bz_dc_deconv, bz_crop


def main():
    parser = argparse.ArgumentParser(description="Gerar mapa Bz em nova altura.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--input", required=True, help="Mapa .txt medido.")
    parser.add_argument("--z-from", type=float, required=True, help="Altura medida (um).")
    parser.add_argument("--z-to", type=float, required=True, help="Altura desejada (um).")
    parser.add_argument("--output", default=None, help="Arquivo .txt de saída.")
    parser.add_argument("--plot", action="store_true", help="Mostrar comparação.")
    args = parser.parse_args()

    project_root = Path(args.config).resolve().parent.parent
    cfg = resolve_paths(load_config(args.config), project_root)
    phys_cfg = cfg["physics"]
    dx, dy = phys_cfg["dx"], phys_cfg["dy"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet2D(in_ch=1, out_ch=1).to(device)
    model_path = project_root / cfg["train"]["model_path"]
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Modelo carregado de:", model_path)

    bz = load_map(args.input)
    bz_pred, bz_dc_deconv, _ = predict_map(
        bz, args.z_from * 1e-6, args.z_to * 1e-6, model, device,
        dx, dy, phys_cfg["alpha_dc"], phys_cfg["sigma_psf"], phys_cfg["beta"],
    )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(out_path, bz_pred)
        print("Mapa salvo em:", out_path)

    if args.plot:
        import matplotlib.pyplot as plt
        ny_c, nx_c = bz_dc_deconv.shape
        extent = [0, nx_c * dx * 1000, 0, ny_c * dy * 1000]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, img, title in zip(
            axes,
            [bz, bz_dc_deconv, bz_pred],
            [f"Medido em {args.z_from:.0f} um",
             f"DC + Deconv ({args.z_from:.0f} -> {args.z_to:.0f} um)",
             f"CNN em {args.z_to:.0f} um"],
        ):
            im = ax.imshow(img, cmap="jet", origin="lower")
            ax.set_title(title)
            ax.set_xlabel("x (px)")
            ax.set_ylabel("y (px)")
            fig.colorbar(im, ax=ax, label="T")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
