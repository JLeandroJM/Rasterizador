"""
Metricas comunes que cada experimento reporta sobre cada clip.

Funciones:
  - psnr(a, b)
  - calcular_ssim(a, b)         (SSIM propio para no depender de pytorch-msssim)
  - calcular_lpips(a, b)        (intenta usar 'lpips' si esta instalado, sino None)
  - calcular_loss(render, obj)  (L1 + lambda * DSSIM)
  - tamano_bytes_state_dict(sd) (cuenta exacta de bytes)
  - guardar_curva, guardar_csv
  - guardar_secuencia_frames, guardar_gif, guardar_comparativa
"""
import csv
import json
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image



# ===== LPIPS opcional ======================================================
_LPIPS_FN = None
_LPIPS_INTENTADO = False

def _intentar_cargar_lpips(device='cpu'):
    """Carga lpips si esta instalado. Si no, devuelve None y deja LPIPS=None."""
    global _LPIPS_FN, _LPIPS_INTENTADO
    if _LPIPS_INTENTADO:
        return _LPIPS_FN
    _LPIPS_INTENTADO = True
    try:
        import lpips
        _LPIPS_FN = lpips.LPIPS(net='alex', verbose=False).to(device).eval()
    except Exception as e:
        print(f"[metricas] lpips no disponible ({e}), LPIPS se reportara como None")
        _LPIPS_FN = None
    return _LPIPS_FN



# ===== PSNR ================================================================
def psnr(a, b):
    """a, b: tensores (..., 3) en [0, 1]. Devuelve PSNR en dB."""
    mse = torch.mean((a - b) ** 2)
    if mse <= 0:
        return float('inf')
    return float(-10.0 * torch.log10(mse))



# ===== SSIM propio =========================================================
def _kernel_gauss_2d(tamano=11, sigma=1.5, device='cpu', dtype=torch.float32):
    coords = torch.arange(tamano, dtype=dtype, device=device) - (tamano - 1) / 2.0
    g1 = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g1 = g1 / g1.sum()
    return g1[:, None] * g1[None, :]


def calcular_ssim(x, y):
    """
    x, y: (H, W, 3) en [0, 1].
    Devuelve SSIM promedio sobre canales y posiciones.
    """
    # a (B, C, H, W)
    x_b = x.permute(2, 0, 1).unsqueeze(0).clamp(0, 1)
    y_b = y.permute(2, 0, 1).unsqueeze(0).clamp(0, 1)
    C = x_b.shape[1]
    device = x_b.device

    kernel = _kernel_gauss_2d(11, 1.5, device, x_b.dtype)
    kernel = kernel.expand(C, 1, 11, 11)

    mu_x = F.conv2d(x_b, kernel, padding=5, groups=C)
    mu_y = F.conv2d(y_b, kernel, padding=5, groups=C)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sx2 = F.conv2d(x_b * x_b, kernel, padding=5, groups=C) - mu_x2
    sy2 = F.conv2d(y_b * y_b, kernel, padding=5, groups=C) - mu_y2
    sxy = F.conv2d(x_b * y_b, kernel, padding=5, groups=C) - mu_xy

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    num = (2 * mu_xy + C1) * (2 * sxy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sx2 + sy2 + C2)
    return float((num / den).mean())



# ===== LPIPS ===============================================================
def calcular_lpips(x, y):
    """
    x, y: (H, W, 3) en [0, 1].
    Devuelve LPIPS o None si la libreria no esta instalada.
    """
    fn = _intentar_cargar_lpips(device=x.device)
    if fn is None:
        return None
    # lpips espera (B, 3, H, W) en [-1, 1]
    x_b = (x.permute(2, 0, 1).unsqueeze(0).clamp(0, 1) * 2.0 - 1.0).to(next(fn.parameters()).device)
    y_b = (y.permute(2, 0, 1).unsqueeze(0).clamp(0, 1) * 2.0 - 1.0).to(next(fn.parameters()).device)
    with torch.no_grad():
        val = fn(x_b, y_b)
    return float(val.item())



# ===== loss L1 + DSSIM =====================================================
def calcular_loss(render, objetivo, lambda_dssim=0.2):
    l1 = torch.mean(torch.abs(render - objetivo))
    ssim = calcular_ssim(render, objetivo)
    dssim = (1.0 - ssim) / 2.0
    # ssim no es un tensor con grad: usamos la version diferenciable
    # via convoluciones. Reconstruyamos el camino diferenciable:
    return (1.0 - lambda_dssim) * l1 + lambda_dssim * _dssim_diferenciable(render, objetivo)



def _dssim_diferenciable(x, y):
    """DSSIM que SI mantiene grad (a diferencia de calcular_ssim que castea a float)."""
    x_b = x.permute(2, 0, 1).unsqueeze(0).clamp(0, 1)
    y_b = y.permute(2, 0, 1).unsqueeze(0).clamp(0, 1)
    C = x_b.shape[1]
    device = x_b.device
    kernel = _kernel_gauss_2d(11, 1.5, device, x_b.dtype).expand(C, 1, 11, 11)

    mu_x = F.conv2d(x_b, kernel, padding=5, groups=C)
    mu_y = F.conv2d(y_b, kernel, padding=5, groups=C)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y
    sx2 = F.conv2d(x_b * x_b, kernel, padding=5, groups=C) - mu_x2
    sy2 = F.conv2d(y_b * y_b, kernel, padding=5, groups=C) - mu_y2
    sxy = F.conv2d(x_b * y_b, kernel, padding=5, groups=C) - mu_xy
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = ((2 * mu_xy + C1) * (2 * sxy + C2)) / ((mu_x2 + mu_y2 + C1) * (sx2 + sy2 + C2))
    return (1.0 - ssim_map.mean()) / 2.0



# ===== tamano de modelo ====================================================
BYTES_POR_DTYPE = {
    torch.float32: 4, torch.float64: 8, torch.float16: 2, torch.bfloat16: 2,
    torch.int64: 8, torch.int32: 4, torch.int16: 2, torch.int8: 1, torch.uint8: 1,
}


def tamano_bytes_state_dict(sd):
    """Cuenta los bytes de todos los tensores en un dict."""
    total = 0
    for v in sd.values():
        if torch.is_tensor(v):
            total += v.numel() * BYTES_POR_DTYPE.get(v.dtype, 4)
        elif isinstance(v, dict):
            total += tamano_bytes_state_dict(v)
        elif isinstance(v, (list, tuple)):
            for item in v:
                if torch.is_tensor(item):
                    total += item.numel() * BYTES_POR_DTYPE.get(item.dtype, 4)
                elif isinstance(item, dict):
                    total += tamano_bytes_state_dict(item)
    return total



def ratio_compresion(num_frames, alto, ancho, bytes_modelo):
    raw = num_frames * alto * ancho * 3
    return raw / bytes_modelo if bytes_modelo > 0 else float('inf')



# ===== guardar curvas ======================================================
def guardar_curva(valores, titulo, ylabel, ruta, log_y=False, xlabel='iteracion'):
    fig, ax = plt.subplots(figsize=(7, 4), dpi=100)
    ax.plot(valores)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(titulo)
    if log_y:
        ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)


def guardar_csv(filas, cabecera, ruta):
    with open(ruta, "w", newline='') as f:
        w = csv.writer(f)
        w.writerow(cabecera)
        for fila in filas:
            w.writerow(fila)


def guardar_metricas_por_frame_png(psnrs, ssims, lpipss, ruta):
    fig, axes = plt.subplots(3, 1, figsize=(8, 7), dpi=100, sharex=True)
    axes[0].plot(psnrs); axes[0].set_ylabel("PSNR (dB)"); axes[0].grid(True, alpha=0.3)
    axes[1].plot(ssims); axes[1].set_ylabel("SSIM"); axes[1].grid(True, alpha=0.3)
    if any(l is not None for l in lpipss):
        axes[2].plot([l if l is not None else 0 for l in lpipss])
        axes[2].set_ylabel("LPIPS")
    else:
        axes[2].text(0.5, 0.5, "LPIPS no disponible", ha='center', va='center', transform=axes[2].transAxes)
    axes[2].set_xlabel("frame_idx")
    axes[2].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)



# ===== guardar secuencias / videos =========================================
def asegurar_carpeta(ruta):
    os.makedirs(ruta, exist_ok=True)


def guardar_frame(tensor, ruta):
    arr = tensor.detach().clamp(0, 1).cpu().numpy()
    Image.fromarray((arr * 255).astype(np.uint8)).save(ruta)


def guardar_secuencia(frames, carpeta):
    asegurar_carpeta(carpeta)
    for i, f in enumerate(frames):
        guardar_frame(f, os.path.join(carpeta, f"frame_{i:04d}.png"))


def guardar_gif(frames, ruta, paso=1, duracion=0.066):
    """Guarda un GIF con cada `paso` frames."""
    import imageio.v2 as imageio
    arrs = []
    for i, f in enumerate(frames):
        if i % paso != 0:
            continue
        if torch.is_tensor(f):
            f = f.detach().clamp(0, 1).cpu().numpy()
        arrs.append((f * 255).astype(np.uint8))
    imageio.mimsave(ruta, arrs, duration=duracion)


def guardar_gif_comparativa(originales, reconstruidos, ruta,
                             paso=3, duracion=0.1, factor_diff=5.0):
    """Crea GIF lado-a-lado: original | reconstruido | diff_amplificada."""
    import imageio.v2 as imageio
    arrs = []
    for i in range(len(originales)):
        if i % paso != 0:
            continue
        o = originales[i]
        r = reconstruidos[i]
        if torch.is_tensor(o):
            o = o.detach().clamp(0, 1).cpu().numpy()
        if torch.is_tensor(r):
            r = r.detach().clamp(0, 1).cpu().numpy()
        diff = np.clip(np.abs(o - r) * factor_diff, 0, 1)
        concat = np.concatenate([o, r, diff], axis=1)
        arrs.append((concat * 255).astype(np.uint8))
    imageio.mimsave(ruta, arrs, duration=duracion)



# ===== util para escribir metricas.json ====================================
def escribir_metricas_json(ruta, **kwargs):
    with open(ruta, "w") as f:
        json.dump(kwargs, f, indent=2, default=str)
