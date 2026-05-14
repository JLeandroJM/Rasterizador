"""
Rasterizador 2D diferenciable vectorizado.

Recibe los parametros YA evaluados (output de modelo.evaluar_en_frame) y
produce la imagen (H, W, 3).

Mismo nucleo matematico que el rasterizador de exp02 (copiado y adaptado a
la firma "dict de parametros"):
  Σ = R(theta) · diag(scale^2) · R(theta)^T
  G(x) = exp(-0.5 (x-mu)^T Σ^-1 (x-mu))
  α = opacity · G,   clamp α ≤ 0.99
  T_i = ∏_{j<i} (1 - α_j)              (en log-space para estabilidad)
  C(x) = Σ_i color_i · α_i · T_i
"""
import torch



def _inversa_2x2(M):
    """Inversa cerrada 2x2 (evita torch.linalg.inv que no siempre va en MPS)."""
    a = M[..., 0, 0]
    b = M[..., 0, 1]
    c = M[..., 1, 0]
    d = M[..., 1, 1]
    det = a * d - b * c
    inv = torch.stack([
        torch.stack([ d, -b], dim=-1),
        torch.stack([-c,  a], dim=-1)
    ], dim=-2)
    return inv / det.unsqueeze(-1).unsqueeze(-1)



def _construir_covarianza(escala, theta):
    """escala: (N, 2)  -- escala REAL ya con exp; theta: (N,) en radianes."""
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    R = torch.stack([
        torch.stack([cos_t, -sin_t], dim=-1),
        torch.stack([sin_t,  cos_t], dim=-1)
    ], dim=-2)                                                        # (N, 2, 2)
    S = torch.diag_embed(escala)                                      # (N, 2, 2)
    return R @ S @ S.transpose(-1, -2) @ R.transpose(-1, -2)



def rasterizar_diferenciable(params, H, W):
    """
    params: dict con
        mu:      (N, 2)
        scale:   (N, 2)   ya con exp
        theta:   (N,)
        opacity: (N,)     ya con sigmoid
        color:   (N, 3)   ya con sigmoid
        depth:   (N,)     sin grad
    Returns: imagen (H, W, 3).

    Sin if/break durante training (vectorizado completo).
    """
    mu       = params['mu']
    escala   = params['scale']
    theta    = params['theta']
    opacidad = params['opacity']
    color    = params['color']
    depth    = params['depth']

    device = mu.device
    dtype  = mu.dtype

    # ordenamos por depth (cercanas primero, front-to-back).
    # argsort no es diferenciable, pero la permutacion no necesita grad.
    indices = torch.argsort(depth)

    sigma = _construir_covarianza(escala, theta)
    sigma_inv = _inversa_2x2(sigma)

    mu_o       = mu[indices]
    sigma_inv_o = sigma_inv[indices]
    op_o       = opacidad[indices]
    color_o    = color[indices]

    # malla de pixeles centrada en cada pixel
    filas = torch.arange(H, device=device, dtype=dtype) + 0.5
    columnas = torch.arange(W, device=device, dtype=dtype) + 0.5
    rr, cc = torch.meshgrid(filas, columnas, indexing='ij')
    pixeles = torch.stack([rr, cc], dim=-1).reshape(-1, 2)             # (P, 2)

    # diferencia pixel - centro para todas las combinaciones
    diff = pixeles.unsqueeze(1) - mu_o.unsqueeze(0)                    # (P, N, 2)

    # exponente -0.5 * diff^T Σ^-1 diff
    tmp = torch.einsum('pni,nij->pnj', diff, sigma_inv_o)
    exponente = -0.5 * (tmp * diff).sum(dim=-1)                        # (P, N)

    G = torch.exp(exponente)
    alpha = op_o.unsqueeze(0) * G
    alpha = torch.clamp(alpha, max=0.99)

    # transmitancia en log-space para evitar underflow/NaN con N grande
    log_uno_menos_alpha = torch.log(1.0 - alpha + 1e-10)
    log_T_inclusivo = torch.cumsum(log_uno_menos_alpha, dim=1)
    log_T = torch.cat([
        torch.zeros_like(log_T_inclusivo[:, :1]),
        log_T_inclusivo[:, :-1]
    ], dim=1)
    T = torch.exp(log_T)

    pesos = (alpha * T).unsqueeze(-1)
    color_pixel = (pesos * color_o.unsqueeze(0)).sum(dim=1)            # (P, 3)
    return color_pixel.reshape(H, W, 3)
