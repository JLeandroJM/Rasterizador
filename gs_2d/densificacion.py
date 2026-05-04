import numpy as np
import torch

from utilidades import inverse_sigmoid


# NUEVO: contenedor para acumular el gradiente de mu durante un intervalo
# de densificacion (ver paper 3DGS, ‖∇μ‖ promedio).
class AcumuladorGradiente:

    def __init__(self, n, device):
        self.suma = torch.zeros(n, device=device)
        self.contador = torch.zeros(n, device=device)
        self.device = device

    def acumular(self, mu_grad):
        if mu_grad is None:
            return
        self.suma = self.suma + mu_grad.norm(dim=1).detach()
        self.contador = self.contador + 1.0

    def promedio(self):
        return self.suma / self.contador.clamp(min=1.0)

    def reset_total(self, n):
        self.suma = torch.zeros(n, device=self.device)
        self.contador = torch.zeros(n, device=self.device)


# NUEVO: aplicar mascara de "mantener" (sirve para prune y para split)
@torch.no_grad()
def aplicar_mascara(modelo, mascara):
    return (
        modelo.mu[mascara],
        modelo.scale_raw[mascara],
        modelo.theta_raw[mascara],
        modelo.opacity_raw[mascara],
        modelo.color_raw[mascara],
        modelo.depth[mascara],
    )


# NUEVO: concatenar gaussianas existentes con las nuevas
@torch.no_grad()
def concatenar(base, nuevos):
    mu_b, sr_b, th_b, op_b, co_b, dp_b = base
    mu_n, sr_n, th_n, op_n, co_n, dp_n = nuevos
    return (
        torch.cat([mu_b, mu_n], dim=0),
        torch.cat([sr_b, sr_n], dim=0),
        torch.cat([th_b, th_n], dim=0),
        torch.cat([op_b, op_n], dim=0),
        torch.cat([co_b, co_n], dim=0),
        torch.cat([dp_b, dp_n], dim=0),
    )


# NUEVO: clone -> duplicar gaussianas chicas con gradiente alto.
# Devolvemos los tensores nuevos a anadir.
@torch.no_grad()
def clonar(modelo, indices):
    return (
        modelo.mu[indices].clone(),
        modelo.scale_raw[indices].clone(),
        modelo.theta_raw[indices].clone(),
        modelo.opacity_raw[indices].clone(),
        modelo.color_raw[indices].clone(),
        modelo.depth[indices].clone(),
    )


# NUEVO: split -> reemplazar gaussianas grandes por 2 hijos sampleados de
# la covarianza de la madre, con escala / factor.
@torch.no_grad()
def dividir(modelo, indices, factor_escala=1.6):

    mu = modelo.mu[indices]
    scale_raw = modelo.scale_raw[indices]
    theta_raw = modelo.theta_raw[indices]
    opacity_raw = modelo.opacity_raw[indices]
    color_raw = modelo.color_raw[indices]
    depth = modelo.depth[indices]

    n = mu.shape[0]
    if n == 0:
        return None

    # construimos cov original y su cholesky 2x2 manual para samplear
    escalas = torch.exp(scale_raw)
    cos_t = torch.cos(theta_raw)
    sin_t = torch.sin(theta_raw)
    R = torch.stack([
        torch.stack([cos_t, -sin_t], dim=-1),
        torch.stack([sin_t,  cos_t], dim=-1)
    ], dim=-2)
    S = torch.diag_embed(escalas)
    sigma = R @ S @ S.transpose(-1, -2) @ R.transpose(-1, -2)

    a = sigma[..., 0, 0]
    b = sigma[..., 0, 1]
    c = sigma[..., 1, 1]
    L00 = torch.sqrt(torch.clamp(a, min=1e-8))
    L10 = b / L00
    L11 = torch.sqrt(torch.clamp(c - L10 ** 2, min=1e-8))

    # 2 hijos por madre: sampleamos offset = L * z, z ~ N(0, I)
    nuevas_mu = []
    for _ in range(2):
        z = torch.randn(n, 2, device=mu.device, dtype=mu.dtype)
        off_y = L00 * z[:, 0]
        off_x = L10 * z[:, 0] + L11 * z[:, 1]
        offset = torch.stack([off_y, off_x], dim=-1)
        nuevas_mu.append(mu + offset)
    mu_hijos = torch.cat(nuevas_mu, dim=0)

    # escala / factor en log -> log(s/k) = log(s) - log(k)
    nueva_sr = scale_raw - float(np.log(factor_escala))
    sr_hijos = torch.cat([nueva_sr, nueva_sr], dim=0)

    th_hijos = torch.cat([theta_raw, theta_raw], dim=0)
    op_hijos = torch.cat([opacity_raw, opacity_raw], dim=0)
    co_hijos = torch.cat([color_raw, color_raw], dim=0)
    dp_hijos = torch.cat([depth, depth], dim=0)

    return (mu_hijos, sr_hijos, th_hijos, op_hijos, co_hijos, dp_hijos)


# NUEVO: operacion de densificacion completa (clone + split). Devuelve los
# nuevos tensores listos para reemplazar en el modelo, mas un mensaje.
@torch.no_grad()
def densificar(modelo, grad_promedio, grad_threshold, size_threshold,
               max_gaussians, hacer_split=True):

    n = modelo.numero_gausianas()

    # quien tiene gradiente alto
    mascara_grad = grad_promedio > grad_threshold

    if not mascara_grad.any():
        return None, "sin candidatos por gradiente"

    # max(escala) por gaussiana
    escalas = modelo.escalas_actuales()
    max_esc = escalas.max(dim=1)[0]

    # clone: chicas con gradiente alto
    mascara_clone = mascara_grad & (max_esc <= size_threshold)
    # split: grandes con gradiente alto
    mascara_split = mascara_grad & (max_esc > size_threshold) if hacer_split else torch.zeros_like(mascara_grad)

    indices_clone = torch.nonzero(mascara_clone, as_tuple=False).squeeze(1)
    indices_split = torch.nonzero(mascara_split, as_tuple=False).squeeze(1)

    # respetar max_gaussians: si nos pasamos, recortamos los mas prioritarios
    nuevos_total = len(indices_clone) + 2 * len(indices_split) - len(indices_split)
    # nuevos_total cuenta cuantas gaussianas se agregan netas
    # clone agrega 1, split agrega 1 (2 hijos - 1 madre)
    espacio = max_gaussians - n
    if nuevos_total > espacio and espacio >= 0:
        # priorizar clone (mas barato y suele ser mas efectivo en gaussianas chicas)
        if len(indices_clone) > espacio:
            indices_clone = indices_clone[:espacio]
            indices_split = indices_split[:0]
        else:
            restante = espacio - len(indices_clone)
            indices_split = indices_split[:restante]

    # base = todas las gaussianas que se mantienen (todas excepto madres a dividir)
    mascara_mantener = torch.ones(n, dtype=torch.bool, device=modelo.mu.device)
    mascara_mantener[indices_split] = False

    base = aplicar_mascara(modelo, mascara_mantener)

    # acumulamos los hijos
    nuevos_clone = clonar(modelo, indices_clone) if len(indices_clone) > 0 else None
    nuevos_split = dividir(modelo, indices_split) if len(indices_split) > 0 else None

    resultado = base
    if nuevos_clone is not None:
        resultado = concatenar(resultado, nuevos_clone)
    if nuevos_split is not None:
        resultado = concatenar(resultado, nuevos_split)

    msg = f"clone={len(indices_clone)} split={len(indices_split)}"
    return resultado, msg


# NUEVO: prune -> eliminar gaussianas con opacidad muy baja o escala enorme
@torch.no_grad()
def podar(modelo, opacity_threshold=0.005, escala_max=None):
    opacidades = modelo.opacidades_actuales()
    mascara = opacidades > opacity_threshold

    if escala_max is not None:
        max_esc = modelo.escalas_actuales().max(dim=1)[0]
        mascara = mascara & (max_esc < escala_max)

    eliminadas = int((~mascara).sum().item())
    if eliminadas == 0:
        return None, 0

    return aplicar_mascara(modelo, mascara), eliminadas


# NUEVO: reset opacity -> poner opacity_raw a inverse_sigmoid(valor)
# Solo modifica el contenido in-place para no romper el optimizer.
@torch.no_grad()
def reset_opacidad(modelo, valor=0.01):
    nuevo = float(inverse_sigmoid(valor))
    modelo.opacity_raw.data.fill_(nuevo)
