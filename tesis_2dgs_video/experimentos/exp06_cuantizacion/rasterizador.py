"""
exp00 — rasterizador 2DGS diferenciable. Toma los parametros evaluados
del modelo y produce la imagen (alto, ancho, 3).
"""
import torch



def inverza_matriz_2(M):
    """Inversa de matriz 2x2 cerrada. Evitamos torch.linalg.inv (MPS)."""
    a = M[..., 0, 0]
    b = M[..., 0, 1]
    c = M[..., 1, 0]
    d = M[..., 1, 1]
    determi = a * d - b * c
    inv = torch.stack([
        torch.stack([ d, -b], dim=-1),
        torch.stack([-c,  a], dim=-1)
    ], dim=-2)
    return inv / determi.unsqueeze(-1).unsqueeze(-1)



def construir_covarianzas(escala, theta):
    """Sigma = R S S^T R^T, con S = diag(exp(escala))."""
    escalas = torch.exp(escala)

    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)

    R = torch.stack([
        torch.stack([cos_t, -sin_t], dim=-1),
        torch.stack([sin_t,  cos_t], dim=-1)
    ], dim=-2)

    S = torch.diag_embed(escalas)
    return R @ S @ S.transpose(-1, -2) @ R.transpose(-1, -2)



def rasterizar_diferenciable(centro, escala, theta, opacidad, color,
                              profundidad, alto, ancho):
    """
    centro      : (N, 2)
    escala      : (N, 2)   escala real = exp(escala)
    theta       : (N,)     rotacion en radianes
    opacidad    : (N,)     opacidad real = sigmoid(opacidad)
    color       : (N, 3)   color real = sigmoid(color)
    profundidad : (N,)     para ordenar (no requiere grad)
    """
    device = centro.device
    dtype = centro.dtype

    # ordenamos por profundidad (cercanas primero, front-to-back)
    indices = torch.argsort(profundidad)

    cov = construir_covarianzas(escala, theta)
    cov_inv = inverza_matriz_2(cov)

    opacidades = torch.sigmoid(opacidad)
    colores = torch.sigmoid(color)

    centro_o   = centro[indices]
    cov_inv_o  = cov_inv[indices]
    op_o       = opacidades[indices]
    color_o    = colores[indices]

    # malla de pixeles
    filas = torch.arange(alto, device=device, dtype=dtype) + 0.5
    columnas = torch.arange(ancho, device=device, dtype=dtype) + 0.5
    rr, cc = torch.meshgrid(filas, columnas, indexing='ij')
    pixeles = torch.stack([rr, cc], dim=-1).reshape(-1, 2)   # (P, 2)

    diff = pixeles.unsqueeze(1) - centro_o.unsqueeze(0)        # (P, N, 2)

    tmp = torch.einsum('pni,nij->pnj', diff, cov_inv_o)        # (P, N, 2)
    exponente = -0.5 * (tmp * diff).sum(dim=-1)                # (P, N)

    G = torch.exp(exponente)
    alpha = op_o.unsqueeze(0) * G
    alpha = torch.clamp(alpha, max=0.99)

    # transmitancia en log-space (evita NaN en cumprod con N grande)
    log_uno_menos_alpha = torch.log(1.0 - alpha + 1e-10)
    log_T_inclusivo = torch.cumsum(log_uno_menos_alpha, dim=1)
    log_T = torch.cat([
        torch.zeros_like(log_T_inclusivo[:, :1]),
        log_T_inclusivo[:, :-1]
    ], dim=1)
    T = torch.exp(log_T)

    pesos = (alpha * T).unsqueeze(-1)
    color_pixel = (pesos * color_o.unsqueeze(0)).sum(dim=1)    # (P, 3)
    return color_pixel.reshape(alto, ancho, 3)



@torch.no_grad()
def clampear_escala(modelo, alto, ancho):
    """Clamp post-step para evitar gaussianas mas grandes que la imagen."""
    import numpy as np
    log_max = float(np.log(max(alto, ancho)))
    log_min = float(np.log(0.5))
    modelo.escala.data.clamp_(min=log_min, max=log_max)
