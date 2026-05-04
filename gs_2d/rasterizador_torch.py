import torch


# NUEVO: inversa de matrices 2x2 cerrada. Evitamos torch.linalg.inv porque
# en MPS no siempre esta soportado y total para 2x2 es trivial.
def inversa_2x2(M):
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


# NUEVO: construir Σ = R S Sᵀ Rᵀ. R(theta) y S = diag(exp(scale_raw)).
def construir_covarianzas(scale_raw, theta_raw):

    escalas = torch.exp(scale_raw)  # (N, 2) siempre positivas

    cos_t = torch.cos(theta_raw)
    sin_t = torch.sin(theta_raw)

    # rotacion (N, 2, 2)
    R = torch.stack([
        torch.stack([cos_t, -sin_t], dim=-1),
        torch.stack([sin_t,  cos_t], dim=-1)
    ], dim=-2)

    S = torch.diag_embed(escalas)  # (N, 2, 2)

    sigma = R @ S @ S.transpose(-1, -2) @ R.transpose(-1, -2)
    return sigma


# NUEVO: rasterizador diferenciable para training. Vectorizado, SIN if/break,
# para no romper el grafo computacional. Las reglas de skip / early
# termination del paper se aplican solo en inferencia (otra funcion).
def rasterizar_diferenciable(modelo, alto, ancho):

    mu = modelo.mu
    scale_raw = modelo.scale_raw
    theta_raw = modelo.theta_raw
    opacity_raw = modelo.opacity_raw
    color_raw = modelo.color_raw
    depth = modelo.depth

    device = mu.device
    dtype = mu.dtype

    # ordenamos por profundidad (ascendente, las mas cercanas primero).
    # detach: la permutacion no necesita gradiente
    indices = torch.argsort(depth)

    # construimos covarianza y la invertimos
    sigma = construir_covarianzas(scale_raw, theta_raw)
    sigma_inv = inversa_2x2(sigma)

    # aplicamos activaciones (sigmoid)
    opacidades = torch.sigmoid(opacity_raw)
    colores = torch.sigmoid(color_raw)

    # reordenamos todo segun depth
    mu_o = mu[indices]
    sigma_inv_o = sigma_inv[indices]
    opacidades_o = opacidades[indices]
    colores_o = colores[indices]

    # malla de pixeles en (fila+0.5, columna+0.5)
    filas = torch.arange(alto, device=device, dtype=dtype) + 0.5
    columnas = torch.arange(ancho, device=device, dtype=dtype) + 0.5
    rr, cc = torch.meshgrid(filas, columnas, indexing='ij')
    pixeles = torch.stack([rr, cc], dim=-1).reshape(-1, 2)  # (P, 2)

    # diferencia pixel - centro para todas las combinaciones
    # diff: (P, N, 2) = pixeles[:, None, :] - mu_o[None, :, :]
    diff = pixeles.unsqueeze(1) - mu_o.unsqueeze(0)

    # exponente -0.5 * diff^T Σ^-1 diff
    tmp = torch.einsum('pni,nij->pnj', diff, sigma_inv_o)
    exponente = -0.5 * (tmp * diff).sum(dim=-1)  # (P, N)

    G = torch.exp(exponente)

    alpha = opacidades_o.unsqueeze(0) * G  # (P, N)

    # clamp a 0.99 (los valores debajo de 0.99 conservan gradiente)
    alpha = torch.clamp(alpha, max=0.99)

    # transmitancia exclusiva: T_i = Π_{j<i} (1 - α_j)
    # la calculamos con cumprod desplazado
    one_minus_alpha = 1.0 - alpha
    T = torch.cat([
        torch.ones_like(one_minus_alpha[:, :1]),
        torch.cumprod(one_minus_alpha[:, :-1], dim=1)
    ], dim=1)

    # contribucion de cada gaussiana: α * T * color
    pesos = (alpha * T).unsqueeze(-1)  # (P, N, 1)
    color_pixel = (pesos * colores_o.unsqueeze(0)).sum(dim=1)  # (P, 3)

    imagen = color_pixel.reshape(alto, ancho, 3)
    return imagen


# NUEVO: rasterizador para inferencia (mismas reglas del paper, con if/break).
# Util para verificar que el render final coincide y para ahorrar tiempo en
# imagenes grandes despues del entrenamiento.
@torch.no_grad()
def rasterizar_inferencia(modelo, alto, ancho, alfa_min=1.0/255.0, alfa_max=0.99, t_min=1e-4):

    mu = modelo.mu
    device = mu.device
    dtype = mu.dtype

    indices = torch.argsort(modelo.depth)

    sigma = construir_covarianzas(modelo.scale_raw, modelo.theta_raw)
    sigma_inv = inversa_2x2(sigma)
    opacidades = torch.sigmoid(modelo.opacity_raw)
    colores = torch.sigmoid(modelo.color_raw)

    mu_o = mu[indices]
    sigma_inv_o = sigma_inv[indices]
    opacidades_o = opacidades[indices]
    colores_o = colores[indices]

    imagen = torch.zeros(alto, ancho, 3, device=device, dtype=dtype)

    for i in range(alto):
        for j in range(ancho):
            transmitancia = 1.0
            color_pixel = torch.zeros(3, device=device, dtype=dtype)
            pixel = torch.tensor([i + 0.5, j + 0.5], device=device, dtype=dtype)

            for k in range(mu_o.shape[0]):
                d = pixel - mu_o[k]
                expo = -0.5 * d @ sigma_inv_o[k] @ d
                if expo > 0:
                    valor = 1.0
                else:
                    valor = float(torch.exp(expo))
                alfa = float(opacidades_o[k]) * valor
                if alfa < alfa_min:
                    continue
                if alfa > alfa_max:
                    alfa = alfa_max
                color_pixel = color_pixel + colores_o[k] * (alfa * transmitancia)
                transmitancia = transmitancia * (1.0 - alfa)
                if transmitancia < t_min:
                    break
            imagen[i, j] = color_pixel

    return imagen
