"""
Rasterizador 2D diferenciable.

Estimacion de memoria (importante para RTX 4050 con 6 GB de VRAM):
------------------------------------------------------------------
El tensor intermedio dominante es `diff` de shape (P, N, 2) en
`rasterizar_un_frame`, donde P = H*W y N = numero de gaussianas.

  256x256 con N=1000:  P*N*2 = 131M floats = 524 MB por frame
  256x256 con N=2000:  P*N*2 = 262M floats = 1.05 GB por frame

Hay ~10 tensores intermedios del mismo orden -> pico ~5 GB para 1 frame
con N=1000 y H=W=256. Entra en una 4050 con 6 GB pero CERCA del limite.

Para batch full sobre n_frames=90 sin sub-batches, el grafo de autograd
necesitaria 90x esa memoria -> imposible.

DECISION (manejo de memoria):
-----------------------------
Implementamos render PER-FRAME (`rasterizar_un_frame`) y la version "batch"
(`rasterizar_batch`) hace internamente un loop. El trainer usa GRADIENT
ACCUMULATION (loss.backward() despues de cada frame, sin guardar el grafo
completo). Eso da memoria pico = 1 frame, equivalente matematicamente a
batch full porque el gradiente es lineal.

Si la GPU es muy grande y se quiere construir el grafo entero, se puede
pasar `sub_batch_frames=None` a rasterizar_batch y devolvera el batch
completo como un solo tensor. Por defecto procesa de a uno (modo seguro).
"""
import torch



def _inversa_2x2(M):
    """Inversa cerrada 2x2 (mas estable que torch.linalg.inv en MPS/CUDA viejo)."""
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
    """Sigma = R(theta) S S^T R(theta)^T, con S = diag(escala) ya activada."""
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    R = torch.stack([
        torch.stack([cos_t, -sin_t], dim=-1),
        torch.stack([sin_t,  cos_t], dim=-1)
    ], dim=-2)
    S = torch.diag_embed(escala)
    return R @ S @ S.transpose(-1, -2) @ R.transpose(-1, -2)



def rasterizar_un_frame(params_frame, H, W):
    """
    Rasteriza UN frame.

    params_frame: dict con
        mu      : (N, 2)   filas, columnas
        scale   : (N, 2)   ya con exp
        theta   : (N,)
        opacity : (N,)     ya con sigmoid
        color   : (N, 3)   ya con sigmoid
        depth   : (N,)     sin grad necesario

    Returns: imagen (H, W, 3).
    """
    mu       = params_frame['mu']
    escala   = params_frame['scale']
    theta    = params_frame['theta']
    opacidad = params_frame['opacity']
    color    = params_frame['color']
    depth    = params_frame['depth']

    device = mu.device
    dtype  = mu.dtype

    # orden front-to-back por depth (permutacion sin grad)
    indices = torch.argsort(depth)

    sigma = _construir_covarianza(escala, theta)
    sigma_inv = _inversa_2x2(sigma)

    mu_o        = mu[indices]
    sigma_inv_o = sigma_inv[indices]
    op_o        = opacidad[indices]
    color_o     = color[indices]

    # malla de pixeles
    filas    = torch.arange(H, device=device, dtype=dtype) + 0.5
    columnas = torch.arange(W, device=device, dtype=dtype) + 0.5
    rr, cc = torch.meshgrid(filas, columnas, indexing='ij')
    pixeles = torch.stack([rr, cc], dim=-1).reshape(-1, 2)         # (P, 2)

    diff = pixeles.unsqueeze(1) - mu_o.unsqueeze(0)                 # (P, N, 2)

    tmp = torch.einsum('pni,nij->pnj', diff, sigma_inv_o)
    exponente = -0.5 * (tmp * diff).sum(dim=-1)                     # (P, N)

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
    color_pixel = (pesos * color_o.unsqueeze(0)).sum(dim=1)         # (P, 3)
    return color_pixel.reshape(H, W, 3)



def rasterizar_batch(params_batch, H, W, sub_batch_frames=None):
    """
    Rasteriza varios frames juntos.

    params_batch: dict con tensores (n_frames, N, ...)  (output de
                  modelo.evaluar_batch_completo).
    sub_batch_frames: cuantos frames procesar a la vez. None -> uno por uno.

    Returns: tensor (n_frames, H, W, 3).

    NOTA: si esto se usa en training, ten en cuenta que TODOS los frames
    procesados quedan en el grafo de autograd hasta el siguiente backward.
    Para training memory-safe usa rasterizar_un_frame en loop con
    backward() por frame (gradient accumulation), no esta funcion.
    """
    n_frames = params_batch['mu'].shape[0]
    if sub_batch_frames is None:
        sub_batch_frames = 1

    outs = []
    for inicio in range(0, n_frames, sub_batch_frames):
        fin = min(inicio + sub_batch_frames, n_frames)
        for j in range(inicio, fin):
            params_j = {k: v[j] for k, v in params_batch.items()}
            outs.append(rasterizar_un_frame(params_j, H, W))
    return torch.stack(outs, dim=0)                                 # (n_frames, H, W, 3)
