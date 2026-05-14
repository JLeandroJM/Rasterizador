"""
Pruning post-training basado en la maxima contribucion de opacity de cada
gaussiana a lo largo del clip.

Idea (lottery ticket):
  Empezamos con N generoso (ej. 1000). Durante training, las que terminan
  siendo "utiles" desarrollan opacity > umbral en algun frame. Las que no,
  se quedan en opacity baja en todo t.
  Al final, evaluamos max_t [sigma(opacity_i(t))] muestreando densamente, y
  eliminamos las que tengan ese max < umbral.
"""
import torch

from chebyshev import construir_matriz_chebyshev



@torch.no_grad()
def calcular_contribucion_maxima(modelo, n_samples=200):
    """
    Para cada gaussiana, devuelve max_t [sigma(opacity_raw_i(t))] muestreado
    en n_samples puntos uniformes de t en [0, n_frames-1].

    Returns: tensor (N,) con la contribucion maxima.
    """
    n_frames_orig = modelo.n_frames
    grado_op = modelo.grados['opacity']

    # construimos una matriz de Chebyshev con n_samples frames "virtuales"
    # mapeados al rango [-1, 1]
    B = construir_matriz_chebyshev(n_samples, grado_op,
                                    device=modelo.opacity_a0.device,
                                    dtype=modelo.opacity_a0.dtype)

    # coefs completos de opacity: (N, 1, grado+1)
    coefs = torch.cat([modelo.opacity_a0, modelo.opacity_high], dim=-1)
    # evaluacion en todos los samples a la vez: (N, 1, grado+1) @ (grado+1, n_samples)
    # -> (N, 1, n_samples)
    raw_t = coefs @ B.T                                              # (N, 1, n_samples)
    raw_t = raw_t.squeeze(1)                                          # (N, n_samples)

    op_t = torch.sigmoid(raw_t)
    return op_t.max(dim=-1).values                                    # (N,)



@torch.no_grad()
def prunear_post_training(modelo, umbral=0.05, n_samples=200):
    """
    Elimina gaussianas con max_t sigma(opacity(t)) < umbral.
    Modifica `modelo` IN-PLACE reemplazando sus nn.Parameter por versiones
    mas chicas. Devuelve (n_original, n_final, indices_eliminados).

    Como cambian los Parameters, hay que recrear el optimizer despues de
    llamar esta funcion (en correr.py).
    """
    contribs = calcular_contribucion_maxima(modelo, n_samples=n_samples)
    mantener = contribs >= umbral

    n_original = int(mantener.shape[0])
    n_final = int(mantener.sum().item())
    indices_eliminados = (~mantener).nonzero(as_tuple=False).squeeze(-1).tolist()

    if n_final == n_original:
        return n_original, n_final, indices_eliminados

    # filtrar cada nn.Parameter por la mascara `mantener` (axis 0 = gaussianas)
    from torch import nn
    for nombre in ['mu', 'opacity', 'color', 'scale', 'theta', 'depth']:
        for sufijo in ['_a0', '_high']:
            attr = nombre + sufijo
            t = getattr(modelo, attr).data
            nuevo = nn.Parameter(t[mantener].clone())
            setattr(modelo, attr, nuevo)

    # actualizar contador interno
    modelo.N = n_final

    return n_original, n_final, indices_eliminados
