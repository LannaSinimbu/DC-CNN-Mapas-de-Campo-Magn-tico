"""
Operações de campo potencial no domínio de Fourier.

Reúne as rotinas físicas usadas no pipeline:
- montagem da grade de números de onda (k);
- estimativa da distância sensor-amostra (z0);
- continuação para baixo (downward continuation);
- deconvolução do footprint gaussiano do sensor.

As convenções de FFT seguem o notebook original. Todos os comprimentos
estão em metros e os campos em tesla.
"""

import numpy as np


def get_k_grid(nx, ny, dx, dy):
    """Retorna as grades kx, ky e o número de onda radial k (em rad/m).

    Parameters
    ----------
    nx, ny : int
        Número de pontos em x e y.
    dx, dy : float
        Passo espacial em x e y (metros).
    """
    fx = np.fft.fftfreq(nx, d=dx)
    fy = np.fft.fftfreq(ny, d=dy)
    kx = 2 * np.pi * fx
    ky = 2 * np.pi * fy
    kx, ky = np.meshgrid(kx, ky)
    k = np.sqrt(kx**2 + ky**2)
    return kx, ky, k


def estimate_z0(bz, dx, dy, n_bins=250):
    """Estima um parâmetro de distância sensor-amostra a partir do
    decaimento radial do espectro de potência.

    Ajusta uma reta a log(|FFT(Bz)|) em função de k; a inclinação é usada
    como estimativa de z0. Observação: este é o procedimento heurístico do
    notebook original — a inclinação tem unidade de comprimento, mas não é
    uma medida física rigorosa da altura de voo.
    """
    ny, nx = bz.shape
    _, _, k = get_k_grid(nx, ny, dx, dy)

    bz_fft = np.fft.fftshift(np.fft.fftn(bz))

    bins = np.linspace(0, k.max(), n_bins)
    radial_k, radial_b = [], []
    for i in range(len(bins) - 1):
        mask = (k >= bins[i]) & (k < bins[i + 1])
        if np.any(mask):
            radial_k.append((bins[i] + bins[i + 1]) / 2)
            radial_b.append(np.mean(np.abs(bz_fft[mask])))

    radial_k = np.array(radial_k)
    radial_b = np.array(radial_b)

    valid = (radial_k > 0) & (radial_b > 0)
    coef = np.polyfit(radial_k[valid], np.log(radial_b[valid]), 1)
    inclination = coef[0]
    return inclination


def downward_continuation(bz, z_from, z_to, k, alpha_dc=1e-4):
    """Continua o mapa de z_from para z_to (com z_to < z_from).

    Usa o fator exp(k * dz) com regularização de Tikhonov para evitar a
    amplificação explosiva das altas frequências.

    Parameters
    ----------
    bz : np.ndarray
        Mapa 2D do campo.
    z_from, z_to : float
        Alturas inicial e final (metros). dz = z_from - z_to > 0.
    k : np.ndarray
        Grade do número de onda radial (de :func:`get_k_grid`).
    alpha_dc : float
        Parâmetro de regularização.
    """
    dz = z_from - z_to
    bz_fft = np.fft.fftshift(np.fft.fftn(bz))
    dc_factor = np.exp(k * dz)
    dc_reg = dc_factor / (1 + alpha_dc * dc_factor**2)
    bz_dc_fft = bz_fft * dc_reg
    bz_dc = np.fft.ifftn(np.fft.ifftshift(bz_dc_fft)).real
    return bz_dc


def deconvolve_footprint(bz, k, sigma_psf=80e-6, beta=1e-2):
    """Deconvolução de um footprint gaussiano do sensor.

    Parameters
    ----------
    bz : np.ndarray
        Mapa 2D do campo.
    k : np.ndarray
        Grade do número de onda radial.
    sigma_psf : float
        Largura efetiva da PSF do sensor (metros).
    beta : float
        Regularização para evitar divisão por zero nas altas frequências.
    """
    bz_fft = np.fft.fftshift(np.fft.fftn(bz))
    psf_k = np.exp(-0.5 * (sigma_psf**2) * (k**2))
    bz_deconv_fft = bz_fft / (psf_k + beta)
    bz_deconv = np.fft.ifftn(np.fft.ifftshift(bz_deconv_fft)).real
    return bz_deconv
