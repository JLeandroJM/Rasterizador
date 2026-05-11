import torch


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

#usar con gradiente

def construir_covarianzas(escala, theta):

    escalas = torch.exp(escala) 

    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)

  #mateiz de rotacion

  #R.S.ST.RT
    R = torch.stack([
        torch.stack([cos_t, -sin_t], dim=-1),
        torch.stack([sin_t,  cos_t], dim=-1)
    ], dim=-2)

    S = torch.diag_embed(escalas) #cubo

    return R @ S @ S.transpose(-1, -2) @ R.transpose(-1, -2)



def rasterizar_diferenciable(modelo, alto, ancho):

    centro = modelo.centro
    escala = modelo.escala
    theta = modelo.theta
    opacidad = modelo.opacidad
    color = modelo.color
   
    profundidad = modelo.profundidad

    device = centro.device
    dtype = centro.dtype


    indices = torch.arange(profundidad.shape[0], device=profundidad.device)


  
    matriz_covarianza = construir_covarianzas(escala, theta)
    matriz_covarianza_inv = inverza_matriz_2(matriz_covarianza)


    opacidades = torch.sigmoid(opacidad)
    colores = torch.sigmoid(color)

 
    centro_o = centro[indices]
    matriz_covarianza_inv_o = matriz_covarianza_inv[indices]
    opacidades_o = opacidades[indices]

    colores_o = colores[indices]

 # imagen de salida
    filas = torch.arange(alto, device=device, dtype=dtype) + 0.5
    columnas = torch.arange(ancho, device=device, dtype=dtype) + 0.5
    rr, cc = torch.meshgrid(filas, columnas, indexing='ij')
    pixeles = torch.stack([rr, cc], dim=-1).reshape(-1, 2)  # (P, 2)


    diff = pixeles.unsqueeze(1) - centro_o.unsqueeze(0)

    # exponente -0.5 * diff^T Σ^-1 diff
    tmp = torch.einsum('pni,nij->pnj', diff, matriz_covarianza_inv_o)
    exponente = -0.5 * (tmp * diff).sum(dim=-1)  # (P, N)

    G = torch.exp(exponente)
#aca e,pieza alpha blending
    alpha = opacidades_o.unsqueeze(0) * G  # (P, N)


    alpha = torch.clamp(alpha, max=0.99)

 # T *= (1 - alpha) 
    log_transmiracia_alpha = torch.log(1.0 - alpha) 

    log_T_inclusivo = torch.cumsum(log_transmiracia_alpha, dim=1)

    log_T = torch.cat([
        torch.zeros_like(log_T_inclusivo[:, :1]),
        log_T_inclusivo[:, :-1]
    ], dim=1)

    T = torch.exp(log_T)

   
    pesos = (alpha * T).unsqueeze(-1)  
    color_pixel = (pesos * colores_o.unsqueeze(0)).sum(dim=1) 

    imagen = color_pixel.reshape(alto, ancho, 3)
    return imagen


