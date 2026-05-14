"""
Metricas finales del experimento sobre todo el video.

PSNR, SSIM, LPIPS por frame -> promedios.
Tamano del modelo (bytes) y ratio de compresion vs raw.
"""
import torch
import torch.nn.functional as F



def psnr(a, b):
    mse = torch.mean((a - b) ** 2)
    if mse <= 0:
        return float('inf')
    return float(-10.0 * torch.log10(mse))



def _kernel_gauss_2d(tamano=11, sigma=1.5, device='cpu', dtype=torch.float32):
    coords = torch.arange(tamano, dtype=dtype, device=device) - (tamano - 1) / 2.0
    g1 = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g1 = g1 / g1.sum()
    return g1[:, None] * g1[None, :]


def calcular_ssim(x, y):
    """SSIM propio (mismo kernel 11x11 que exp00-02)."""
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
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    num = (2 * mu_xy + C1) * (2 * sxy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sx2 + sy2 + C2)
    return float((num / den).mean())



# LPIPS opcional ------------------------------------------------------------
_LPIPS_FN = None
_LPIPS_INTENTADO = False

def calcular_lpips(x, y, device='cpu'):
    global _LPIPS_FN, _LPIPS_INTENTADO
    if not _LPIPS_INTENTADO:
        _LPIPS_INTENTADO = True
        try:
            import lpips
            _LPIPS_FN = lpips.LPIPS(net='alex', verbose=False).to(device).eval()
        except Exception as e:
            print(f"[metricas] lpips no disponible ({e}), se reportara None")
            _LPIPS_FN = None
    if _LPIPS_FN is None:
        return None
    x_b = (x.permute(2, 0, 1).unsqueeze(0).clamp(0, 1) * 2.0 - 1.0).to(next(_LPIPS_FN.parameters()).device)
    y_b = (y.permute(2, 0, 1).unsqueeze(0).clamp(0, 1) * 2.0 - 1.0).to(next(_LPIPS_FN.parameters()).device)
    with torch.no_grad():
        return float(_LPIPS_FN(x_b, y_b).item())



# tamano del modelo ----------------------------------------------------------
BYTES_POR_DTYPE = {
    torch.float32: 4, torch.float64: 8, torch.float16: 2, torch.bfloat16: 2,
}

def tamano_bytes_coefs(sd_coefs):
    """Cuenta bytes de los tensores en el dict devuelto por modelo.state_dict_coefs()."""
    total = 0
    for k, v in sd_coefs.items():
        if torch.is_tensor(v):
            total += v.numel() * BYTES_POR_DTYPE.get(v.dtype, 4)
    return total


def ratio_compresion(n_frames, H, W, bytes_modelo):
    raw = n_frames * H * W * 3
    return raw / bytes_modelo if bytes_modelo > 0 else float('inf')
