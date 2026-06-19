"""
Carregamento de dados, aumento (data augmentation) e construção do dataset.

Fluxo:
1. carregar os mapas .txt;
2. recortar todos para uma forma comum;
3. aplicar aumentos geométricos e de fundo/ruído fisicamente seguros;
4. construir pares (entrada continuada -> alvo no plano de referência);
5. extrair patches 2D;
6. expor um ``torch.utils.data.Dataset``.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from .physics import downward_continuation, deconvolve_footprint, get_k_grid


# ---------------------------------------------------------------------------
# Carregamento
# ---------------------------------------------------------------------------
def load_map(path):
    """Carrega um mapa Bz de um arquivo de texto (np.loadtxt)."""
    return np.loadtxt(path)


def load_maps(paths):
    """Carrega uma lista de mapas."""
    return [load_map(p) for p in paths]


# ---------------------------------------------------------------------------
# Forma comum
# ---------------------------------------------------------------------------
def force_shape(bz, target_shape):
    """Recorta o centro de ``bz`` para ``target_shape`` (sem corrigir rotação)."""
    ny_t, nx_t = target_shape
    ny, nx = bz.shape
    if ny < ny_t or nx < nx_t:
        raise ValueError(f"Imagem menor que a referência: {bz.shape}")
    y0 = (ny - ny_t) // 2
    x0 = (nx - nx_t) // 2
    return bz[y0:y0 + ny_t, x0:x0 + nx_t]


# ---------------------------------------------------------------------------
# Aumento de dados
# ---------------------------------------------------------------------------
def augment_geometric(bz):
    """Aumentos geométricos seguros (sem rot90/rot270, que mudariam a física)."""
    return [
        bz,                          # original
        np.flipud(bz),               # flip vertical
        np.fliplr(bz),               # flip horizontal
        np.flipud(np.fliplr(bz)),    # flip duplo
        np.rot90(bz, 2),             # rotação de 180 graus
    ]


def add_background_jitter(bz, amp_ratio=0.03, rng=None):
    """Adiciona um fundo planar aleatório (offset + gradiente linear)."""
    rng = rng or np.random.default_rng()
    ny, nx = bz.shape
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    xx, yy = np.meshgrid(x, y)
    amp = amp_ratio * np.max(np.abs(bz))
    a, b, c = rng.uniform(-amp, amp, size=3)
    return bz + a + b * xx + c * yy


def add_noise(bz, noise_ratio=0.01, rng=None):
    """Adiciona ruído gaussiano proporcional ao desvio padrão do mapa."""
    rng = rng or np.random.default_rng()
    sigma = noise_ratio * np.std(bz)
    return bz + sigma * rng.standard_normal(bz.shape)


def build_augmented_dataset(raw_maps, z_list, target_shape, seed=0):
    """Constrói o conjunto aumentado de mapas e suas alturas correspondentes.

    Retorna (bz_maps, z_array) com formas (N, Ny, Nx) e (N,).
    """
    rng = np.random.default_rng(seed)
    bz_aug, z_aug = [], []

    def append_safe(bz, z):
        bz_aug.append(force_shape(bz, target_shape))
        z_aug.append(z)

    for bz, z in zip(raw_maps, z_list):
        for bz_g in augment_geometric(bz):
            append_safe(bz_g, z)
            if rng.random() < 0.5:
                append_safe(add_background_jitter(bz_g, rng=rng), z)
            if rng.random() < 0.3:
                append_safe(add_noise(bz_g, rng=rng), z)

    return np.stack(bz_aug), np.array(z_aug)


# ---------------------------------------------------------------------------
# Construção dos pares entrada/alvo
# ---------------------------------------------------------------------------
def build_pairs(bz_maps, z_list, dx, dy, alpha_dc=1e-4,
                sigma_psf=80e-6, beta=1e-2):
    """Continua cada mapa até o plano de referência (menor z) e gera pares.

    Para cada mapa i != ref:
        1. downward continuation de z_i -> z_ref;
        2. deconvolução do footprint;
        3. normalização por max|.|.
    O alvo é sempre o mapa de referência normalizado.
    """
    nz, ny, nx = bz_maps.shape
    _, _, k = get_k_grid(nx, ny, dx, dy)

    idx_ref = int(np.argmin(z_list))
    z_ref = z_list[idx_ref]
    bz_ref_target = bz_maps[idx_ref]

    scale_t = np.max(np.abs(bz_ref_target)) or 1.0
    bz_t_norm = bz_ref_target / scale_t

    inputs, targets = [], []
    for i in range(nz):
        if i == idx_ref:
            continue
        bz_dc = downward_continuation(bz_maps[i], z_list[i], z_ref, k, alpha_dc)
        bz_dc_deconv = deconvolve_footprint(bz_dc, k, sigma_psf, beta)
        scale = np.max(np.abs(bz_dc_deconv)) or 1.0
        inputs.append(bz_dc_deconv / scale)
        targets.append(bz_t_norm)

    return np.array(inputs), np.array(targets), idx_ref, z_ref


# ---------------------------------------------------------------------------
# Patches
# ---------------------------------------------------------------------------
def extract_patches(bz_array, patch_size=48, stride=24):
    """Extrai patches 2D sobrepostos. Entrada (N, Ny, Nx) -> (N_patches, p, p)."""
    patches = []
    n, ny, nx = bz_array.shape
    for idx in range(n):
        img = bz_array[idx]
        for y0 in range(0, ny - patch_size + 1, stride):
            for x0 in range(0, nx - patch_size + 1, stride):
                patches.append(img[y0:y0 + patch_size, x0:x0 + patch_size])
    return np.array(patches)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class BzPatchDataset(Dataset):
    """Dataset de patches (entrada, alvo) com um canal."""

    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(self.inputs[idx][None, ...]).float()
        y = torch.from_numpy(self.targets[idx][None, ...]).float()
        return x, y
