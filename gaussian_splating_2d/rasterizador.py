import torch



# inversa de matriz 2x2 cerrada (evitamos torch.linalg.inv: en MPS no
# siempre esta soportado, y total para 2x2 es trivial)
def inverza_matriz_2(M):
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



# Σ = R S Sᵀ Rᵀ (escala + rotacion)
def construir_covarianzas(escala, theta):

    escalas = torch.exp(escala)

    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)

    # matriz de rotacion (N, 2, 2)
    R = torch.stack([
        torch.stack([cos_t, -sin_t], dim=-1),
        torch.stack([sin_t,  cos_t], dim=-1)
    ], dim=-2)

    S = torch.diag_embed(escalas)

    return R @ S @ S.transpose(-1, -2) @ R.transpose(-1, -2)



# rasterizador diferenciable. Vectorizado, sin if/break, para no romper
# el grafo computacional. Las reglas del paper (skip alpha < 1/255,
# early termination) se aplicarian solo en inferencia, no en training.
def rasterizar_diferenciable(modelo, alto, ancho):

    centro = modelo.centro
    escala = modelo.escala
    theta = modelo.theta
    opacidad = modelo.opacidad
    color = modelo.color
    profundidad = modelo.profundidad

    device = centro.device
    dtype = centro.dtype

    # NUEVO: ordenamos por profundidad (cercanas primero, front-to-back).
    # No es diferenciable pero la permutacion no necesita gradiente.
    indices = torch.argsort(profundidad)

    matriz_covarianza = construir_covarianzas(escala, theta)
    matriz_covarianza_inv = inverza_matriz_2(matriz_covarianza)

    # activaciones: sigmoid en (0, 1)
    opacidades = torch.sigmoid(opacidad)
    colores = torch.sigmoid(color)

    # reordenamos todo segun profundidad
    centro_o = centro[indices]
    matriz_covarianza_inv_o = matriz_covarianza_inv[indices]
    opacidades_o = opacidades[indices]
    colores_o = colores[indices]

    # malla de pixeles en (fila+0.5, columna+0.5)
    filas = torch.arange(alto, device=device, dtype=dtype) + 0.5
    columnas = torch.arange(ancho, device=device, dtype=dtype) + 0.5
    rr, cc = torch.meshgrid(filas, columnas, indexing='ij')
    pixeles = torch.stack([rr, cc], dim=-1).reshape(-1, 2)

    # diferencia pixel - centro para todas las combinaciones
    diff = pixeles.unsqueeze(1) - centro_o.unsqueeze(0)

    # exponente -0.5 * diff^T Σ^-1 diff
    tmp = torch.einsum('pni,nij->pnj', diff, matriz_covarianza_inv_o)
    exponente = -0.5 * (tmp * diff).sum(dim=-1)

    G = torch.exp(exponente)

    alpha = opacidades_o.unsqueeze(0) * G

    # clamp a 0.99 (no rompe el gradiente para los valores debajo)
    alpha = torch.clamp(alpha, max=0.99)

    # NUEVO: transmitancia en log-space. El cumprod directo subdesborda
    # con N grande (1-α=0.01)^N=0 en float32) y su gradiente da 0/0=NaN.
    # exp(cumsum(log(...))) es estable: cuando T es muy chico el gradiente
    # tambien lo es, sin NaN.
    log_transmiracia_alpha = torch.log(1.0 - alpha + 1e-10)
    log_T_inclusivo = torch.cumsum(log_transmiracia_alpha, dim=1)
    log_T = torch.cat([
        torch.zeros_like(log_T_inclusivo[:, :1]),
        log_T_inclusivo[:, :-1]
    ], dim=1)
    T = torch.exp(log_T)

    # contribucion de cada gaussiana: alpha * T * color
    pesos = (alpha * T).unsqueeze(-1)
    color_pixel = (pesos * colores_o.unsqueeze(0)).sum(dim=1)

    imagen = color_pixel.reshape(alto, ancho, 3)
    return imagen
