"""
exp04 — los TRES modelos temporales duplicados aqui para que el experimento
sea autocontenido. La eleccion de cual usar se hace via config["base"].

- Modelo2DGSTemporalPos    (centro polinomico)        -> base "exp01_..."
- Modelo2DGSTemporalPosOp  (centro + opacidad)        -> base "exp02_..."
- Modelo2DGSTemporalFull   (todo polinomico)          -> base "exp03_..."

Estos son copias EXACTAS de los modelos de los respectivos experimentos.
Si modificas alguno en los exp originales, sincroniza aqui manualmente.
"""
import numpy as np
import torch
from torch import nn



# === Modelo de exp01 =======================================================
class Modelo2DGSTemporalPos:

    def __init__(self, N, q, alto, ancho, device='cpu',
                 escala_inicial=4.0, semilla=0):
        self.q = q
        self.alto = alto
        self.ancho = ancho

        g = torch.Generator(device='cpu').manual_seed(semilla)
        centro_coefs = torch.zeros(N, 2, q + 1)
        centro_coefs[:, 0, 0] = torch.rand(N, generator=g) * alto
        centro_coefs[:, 1, 0] = torch.rand(N, generator=g) * ancho

        escala = torch.full((N, 2), float(np.log(escala_inicial)))
        theta = (torch.rand(N, generator=g) - 0.5) * 2 * np.pi
        opacidad = torch.zeros(N)
        color = (torch.rand(N, 3, generator=g) - 0.5) * 2.0
        profundidad = torch.rand(N, generator=g)

        self.centro_coefs = nn.Parameter(centro_coefs.to(device))
        self.escala       = nn.Parameter(escala.to(device))
        self.theta        = nn.Parameter(theta.to(device))
        self.opacidad     = nn.Parameter(opacidad.to(device))
        self.color        = nn.Parameter(color.to(device))
        self.profundidad  = profundidad.to(device)

    def numero_gausianas(self):
        return self.centro_coefs.shape[0]

    def evaluar_en_t(self, t_norm):
        device = self.centro_coefs.device
        dtype = self.centro_coefs.dtype
        t_powers = torch.tensor(
            [t_norm ** k for k in range(self.q + 1)],
            device=device, dtype=dtype,
        )
        centro_t = (self.centro_coefs * t_powers).sum(dim=-1)
        return (centro_t, self.escala, self.theta,
                self.opacidad, self.color, self.profundidad)

    def state_dict(self):
        return {
            "centro_coefs": self.centro_coefs.detach().cpu(),
            "escala":       self.escala.detach().cpu(),
            "theta":        self.theta.detach().cpu(),
            "opacidad":     self.opacidad.detach().cpu(),
            "color":        self.color.detach().cpu(),
            "profundidad":  self.profundidad.detach().cpu(),
            "q":            self.q,
        }

    def grupos_optimizer(self, lrs):
        return [
            {'params': [self.centro_coefs], 'lr': lrs['centro']},
            {'params': [self.escala],       'lr': lrs['escala']},
            {'params': [self.theta],        'lr': lrs['theta']},
            {'params': [self.opacidad],     'lr': lrs['opacidad']},
            {'params': [self.color],        'lr': lrs['color']},
        ]

    def post_step(self):
        # clamp escala estatica
        log_max = float(np.log(max(self.alto, self.ancho)))
        log_min = float(np.log(0.5))
        with torch.no_grad():
            self.escala.data.clamp_(min=log_min, max=log_max)



# === Modelo de exp02 =======================================================
class Modelo2DGSTemporalPosOp:

    def __init__(self, N, q, alto, ancho, device='cpu',
                 escala_inicial=4.0, semilla=0):
        self.q = q
        self.alto = alto
        self.ancho = ancho

        g = torch.Generator(device='cpu').manual_seed(semilla)
        centro_coefs = torch.zeros(N, 2, q + 1)
        centro_coefs[:, 0, 0] = torch.rand(N, generator=g) * alto
        centro_coefs[:, 1, 0] = torch.rand(N, generator=g) * ancho

        opacidad_coefs = torch.zeros(N, q + 1)

        escala = torch.full((N, 2), float(np.log(escala_inicial)))
        theta = (torch.rand(N, generator=g) - 0.5) * 2 * np.pi
        color = (torch.rand(N, 3, generator=g) - 0.5) * 2.0
        profundidad = torch.rand(N, generator=g)

        self.centro_coefs   = nn.Parameter(centro_coefs.to(device))
        self.opacidad_coefs = nn.Parameter(opacidad_coefs.to(device))
        self.escala         = nn.Parameter(escala.to(device))
        self.theta          = nn.Parameter(theta.to(device))
        self.color          = nn.Parameter(color.to(device))
        self.profundidad    = profundidad.to(device)

    def numero_gausianas(self):
        return self.centro_coefs.shape[0]

    def evaluar_en_t(self, t_norm):
        device = self.centro_coefs.device
        dtype = self.centro_coefs.dtype
        t_powers = torch.tensor(
            [t_norm ** k for k in range(self.q + 1)],
            device=device, dtype=dtype,
        )
        centro_t = (self.centro_coefs * t_powers).sum(dim=-1)
        opacidad_raw_t = (self.opacidad_coefs * t_powers).sum(dim=-1)
        return (centro_t, self.escala, self.theta,
                opacidad_raw_t, self.color, self.profundidad)

    def state_dict(self):
        return {
            "centro_coefs":   self.centro_coefs.detach().cpu(),
            "opacidad_coefs": self.opacidad_coefs.detach().cpu(),
            "escala":         self.escala.detach().cpu(),
            "theta":          self.theta.detach().cpu(),
            "color":          self.color.detach().cpu(),
            "profundidad":    self.profundidad.detach().cpu(),
            "q":              self.q,
        }

    def grupos_optimizer(self, lrs):
        return [
            {'params': [self.centro_coefs],   'lr': lrs['centro']},
            {'params': [self.opacidad_coefs], 'lr': lrs['opacidad']},
            {'params': [self.escala],         'lr': lrs['escala']},
            {'params': [self.theta],          'lr': lrs['theta']},
            {'params': [self.color],          'lr': lrs['color']},
        ]

    def post_step(self):
        log_max = float(np.log(max(self.alto, self.ancho)))
        log_min = float(np.log(0.5))
        with torch.no_grad():
            self.escala.data.clamp_(min=log_min, max=log_max)



# === Modelo de exp03 =======================================================
class Modelo2DGSTemporalFull:

    def __init__(self, N, q, alto, ancho, device='cpu',
                 escala_inicial=4.0, semilla=0):
        self.q = q
        self.alto = alto
        self.ancho = ancho

        g = torch.Generator(device='cpu').manual_seed(semilla)
        centro_coefs = torch.zeros(N, 2, q + 1)
        centro_coefs[:, 0, 0] = torch.rand(N, generator=g) * alto
        centro_coefs[:, 1, 0] = torch.rand(N, generator=g) * ancho

        escala_coefs = torch.zeros(N, 2, q + 1)
        escala_coefs[:, :, 0] = float(np.log(escala_inicial))

        theta_coefs = torch.zeros(N, q + 1)
        theta_coefs[:, 0] = (torch.rand(N, generator=g) - 0.5) * 2 * np.pi

        opacidad_coefs = torch.zeros(N, q + 1)

        color_coefs = torch.zeros(N, 3, q + 1)
        color_coefs[:, :, 0] = (torch.rand(N, 3, generator=g) - 0.5) * 2.0

        profundidad = torch.rand(N, generator=g)

        self.centro_coefs   = nn.Parameter(centro_coefs.to(device))
        self.escala_coefs   = nn.Parameter(escala_coefs.to(device))
        self.theta_coefs    = nn.Parameter(theta_coefs.to(device))
        self.opacidad_coefs = nn.Parameter(opacidad_coefs.to(device))
        self.color_coefs    = nn.Parameter(color_coefs.to(device))
        self.profundidad    = profundidad.to(device)

    def numero_gausianas(self):
        return self.centro_coefs.shape[0]

    def evaluar_en_t(self, t_norm):
        device = self.centro_coefs.device
        dtype = self.centro_coefs.dtype
        t_powers = torch.tensor(
            [t_norm ** k for k in range(self.q + 1)],
            device=device, dtype=dtype,
        )
        centro_t   = (self.centro_coefs   * t_powers).sum(dim=-1)
        escala_t   = (self.escala_coefs   * t_powers).sum(dim=-1)
        theta_t    = (self.theta_coefs    * t_powers).sum(dim=-1)
        opacidad_t = (self.opacidad_coefs * t_powers).sum(dim=-1)
        color_t    = (self.color_coefs    * t_powers).sum(dim=-1)

        # clamp escala_t porque es polinomico
        log_max = float(np.log(max(self.alto, self.ancho)))
        log_min = float(np.log(0.5))
        escala_t = escala_t.clamp(min=log_min, max=log_max)

        return (centro_t, escala_t, theta_t, opacidad_t, color_t, self.profundidad)

    def state_dict(self):
        return {
            "centro_coefs":   self.centro_coefs.detach().cpu(),
            "escala_coefs":   self.escala_coefs.detach().cpu(),
            "theta_coefs":    self.theta_coefs.detach().cpu(),
            "opacidad_coefs": self.opacidad_coefs.detach().cpu(),
            "color_coefs":    self.color_coefs.detach().cpu(),
            "profundidad":    self.profundidad.detach().cpu(),
            "q":              self.q,
        }

    def grupos_optimizer(self, lrs):
        return [
            {'params': [self.centro_coefs],   'lr': lrs['centro']},
            {'params': [self.escala_coefs],   'lr': lrs['escala']},
            {'params': [self.theta_coefs],    'lr': lrs['theta']},
            {'params': [self.opacidad_coefs], 'lr': lrs['opacidad']},
            {'params': [self.color_coefs],    'lr': lrs['color']},
        ]

    def post_step(self):
        # escala se clampea dentro de evaluar_en_t
        pass



# === selector dinamico =====================================================
def construir_modelo_por_base(base_nombre, N, q, alto, ancho, device,
                                escala_inicial, semilla):
    if base_nombre.startswith("exp01"):
        return Modelo2DGSTemporalPos(N, q, alto, ancho, device, escala_inicial, semilla)
    if base_nombre.startswith("exp02"):
        return Modelo2DGSTemporalPosOp(N, q, alto, ancho, device, escala_inicial, semilla)
    if base_nombre.startswith("exp03"):
        return Modelo2DGSTemporalFull(N, q, alto, ancho, device, escala_inicial, semilla)
    raise ValueError(f"base desconocida: {base_nombre}")
