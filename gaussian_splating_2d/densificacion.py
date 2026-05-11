import numpy as np
import torch



# acumulador de gradiente para decidir candidatos de densificacion.
# Acumula ‖∇centro‖ entre intervalos de densificacion y promedia.
class AcumuladorGradiente:

    def __init__(self, n, device):
        self.suma = torch.zeros(n, device=device)
        self.contador = torch.zeros(n, device=device)
        self.device = device

    def acumular(self, grad):
        if grad is None:
            return
        self.suma = self.suma + grad.norm(dim=1).detach()
        self.contador = self.contador + 1.0

    def promedio(self):
        return self.suma / self.contador.clamp(min=1.0)

    def reset_total(self, n):
        self.suma = torch.zeros(n, device=self.device)
        self.contador = torch.zeros(n, device=self.device)



# aplicar mascara booleana: mantiene solo las gaussianas en True
@torch.no_grad()
def aplicar_mascara(modelo, mascara):
    return (
        modelo.centro[mascara],
        modelo.escala[mascara],
        modelo.theta[mascara],
        modelo.opacidad[mascara],
        modelo.color[mascara],
        modelo.profundidad[mascara],
    )



# concatenar dos conjuntos de tensores apilados (base + nuevos)
@torch.no_grad()
def concatenar(base, nuevos):
    c_b, e_b, t_b, o_b, k_b, p_b = base
    c_n, e_n, t_n, o_n, k_n, p_n = nuevos
    return (
        torch.cat([c_b, c_n], dim=0),
        torch.cat([e_b, e_n], dim=0),
        torch.cat([t_b, t_n], dim=0),
        torch.cat([o_b, o_n], dim=0),
        torch.cat([k_b, k_n], dim=0),
        torch.cat([p_b, p_n], dim=0),
    )



# clone: duplicar las gaussianas chicas con gradiente alto.
# NUEVO: las dos copias salen con un offset aleatorio chico para romper
# simetria. Si quedan en la misma posicion exacta el render no cambia al
# moverlas y el gradiente cae a ~0 (jamas se separan).
@torch.no_grad()
def clonar(modelo, indices, offset_px=0.5):
    centro_copia = modelo.centro[indices].clone()
    perturbacion = (torch.rand_like(centro_copia) - 0.5) * 2.0 * offset_px
    centro_copia = centro_copia + perturbacion
    return (
        centro_copia,
        modelo.escala[indices].clone(),
        modelo.theta[indices].clone(),
        modelo.opacidad[indices].clone(),
        modelo.color[indices].clone(),
        modelo.profundidad[indices].clone(),
    )



# split: reemplazar las gaussianas grandes con gradiente alto por 2
# hijos sampleados de la covarianza original, con escala / factor.
@torch.no_grad()
def dividir(modelo, indices, factor_escala=1.6):

    centro = modelo.centro[indices]
    escala = modelo.escala[indices]
    theta = modelo.theta[indices]
    opacidad = modelo.opacidad[indices]
    color = modelo.color[indices]
    profundidad = modelo.profundidad[indices]

    n = centro.shape[0]
    if n == 0:
        return None

    # construimos cov original para samplear hijos
    escalas = torch.exp(escala)
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    R = torch.stack([
        torch.stack([cos_t, -sin_t], dim=-1),
        torch.stack([sin_t,  cos_t], dim=-1)
    ], dim=-2)
    S = torch.diag_embed(escalas)
    sigma = R @ S @ S.transpose(-1, -2) @ R.transpose(-1, -2)

    # cholesky 2x2 manual: L = [[sqrt(a), 0], [b/sqrt(a), sqrt(c - b^2/a)]]
    a = sigma[..., 0, 0]
    b = sigma[..., 0, 1]
    c = sigma[..., 1, 1]
    L00 = torch.sqrt(torch.clamp(a, min=1e-8))
    L10 = b / L00
    L11 = torch.sqrt(torch.clamp(c - L10 ** 2, min=1e-8))

    # 2 hijos por madre, sampleados como mu + L * z, z ~ N(0, I)
    nuevos_centros = []
    for _ in range(2):
        z = torch.randn(n, 2, device=centro.device, dtype=centro.dtype)
        off_y = L00 * z[:, 0]
        off_x = L10 * z[:, 0] + L11 * z[:, 1]
        offset = torch.stack([off_y, off_x], dim=-1)
        nuevos_centros.append(centro + offset)
    centro_hijos = torch.cat(nuevos_centros, dim=0)

    # escala / factor en log: log(s/k) = log(s) - log(k)
    nueva_escala = escala - float(np.log(factor_escala))
    escala_hijos = torch.cat([nueva_escala, nueva_escala], dim=0)

    theta_hijos = torch.cat([theta, theta], dim=0)
    opacidad_hijos = torch.cat([opacidad, opacidad], dim=0)
    color_hijos = torch.cat([color, color], dim=0)
    profundidad_hijos = torch.cat([profundidad, profundidad], dim=0)

    return (centro_hijos, escala_hijos, theta_hijos,
            opacidad_hijos, color_hijos, profundidad_hijos)



# densificacion completa: clone (chicas) + split (grandes) por gradiente
# alto. size_threshold separa "chica" de "grande".
@torch.no_grad()
def densificar(modelo, grad_promedio, grad_threshold, size_threshold,
               max_gaussians, hacer_split=True):

    n = modelo.numero_gausianas()

    # candidatos: gradiente promedio por encima del umbral
    mascara_grad = grad_promedio > grad_threshold
    if not mascara_grad.any():
        return None, "sin candidatos por gradiente"

    # max(escala) por gaussiana (en pixeles)
    escalas = modelo.escalas_actuales()
    max_esc = escalas.max(dim=1)[0]

    # clone: chicas con grad alto
    mascara_clone = mascara_grad & (max_esc <= size_threshold)
    # split: grandes con grad alto
    if hacer_split:
        mascara_split = mascara_grad & (max_esc > size_threshold)
    else:
        mascara_split = torch.zeros_like(mascara_grad)

    indices_clone = torch.nonzero(mascara_clone, as_tuple=False).squeeze(1)
    indices_split = torch.nonzero(mascara_split, as_tuple=False).squeeze(1)

    # respetar max_gaussians (clone agrega +1 c/u, split tambien +1 neto c/u)
    espacio = max_gaussians - n
    nuevos_total = len(indices_clone) + len(indices_split)
    if nuevos_total > espacio and espacio >= 0:
        # priorizamos clone (mas barato y suele ayudar mas en gaussianas chicas)
        if len(indices_clone) > espacio:
            indices_clone = indices_clone[:espacio]
            indices_split = indices_split[:0]
        else:
            restante = espacio - len(indices_clone)
            indices_split = indices_split[:restante]

    # base = todas las gaussianas que se mantienen (todas excepto las madres
    # que se van a dividir, esas se reemplazan por sus hijos)
    mascara_mantener = torch.ones(n, dtype=torch.bool, device=modelo.centro.device)
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



# prune: eliminar gaussianas con opacidad muy baja o escala enorme
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



# reset opacity: forzar opacidad muy baja en todas las gaussianas.
# Modifica .data in-place para no romper la referencia que guarda el
# optimizer (asi NO hay que reconstruirlo despues del reset).
@torch.no_grad()
def reset_opacidad(modelo, valor=0.01):
    nuevo = float(np.log(valor / (1.0 - valor)))   # logit(valor)
    modelo.opacidad.data.fill_(nuevo)
