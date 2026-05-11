"""
exp03 — modelo 2DGS full temporal. TODOS los parametros son polinomicos.

centro_coefs   : (N, 2, q+1)
escala_coefs   : (N, 2, q+1)
theta_coefs    : (N, q+1)
opacidad_coefs : (N, q+1)
color_coefs    : (N, 3, q+1)
profundidad    : (N,) sin grad
"""
import numpy as np
import torch
from torch import nn



class Modelo2DGSTemporalFull:

    def __init__(self, N, q, alto, ancho, device='cpu',
                 escala_inicial=4.0, semilla=0):

        self.q = q
        self.alto = alto
        self.ancho = ancho

        g = torch.Generator(device='cpu').manual_seed(semilla)

        # --- centro ---
        centro_coefs = torch.zeros(N, 2, q + 1)
        centro_coefs[:, 0, 0] = torch.rand(N, generator=g) * alto
        centro_coefs[:, 1, 0] = torch.rand(N, generator=g) * ancho

        # --- escala ---
        escala_coefs = torch.zeros(N, 2, q + 1)
        escala_coefs[:, :, 0] = float(np.log(escala_inicial))

        # --- theta ---
        theta_coefs = torch.zeros(N, q + 1)
        theta_coefs[:, 0] = (torch.rand(N, generator=g) - 0.5) * 2 * np.pi

        # --- opacidad ---
        opacidad_coefs = torch.zeros(N, q + 1)

        # --- color ---
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


    def parametros(self):
        return [self.centro_coefs, self.escala_coefs, self.theta_coefs,
                self.opacidad_coefs, self.color_coefs]


    def evaluar_en_t(self, t_norm):
        """Evalua todos los polinomios en t_norm. Devuelve parametros para el rasterizador."""
        device = self.centro_coefs.device
        dtype = self.centro_coefs.dtype

        t_powers = torch.tensor(
            [t_norm ** k for k in range(self.q + 1)],
            device=device, dtype=dtype,
        )   # (q+1,)

        centro_t   = (self.centro_coefs   * t_powers).sum(dim=-1)        # (N, 2)
        escala_t   = (self.escala_coefs   * t_powers).sum(dim=-1)        # (N, 2)
        theta_t    = (self.theta_coefs    * t_powers).sum(dim=-1)        # (N,)
        opacidad_t = (self.opacidad_coefs * t_powers).sum(dim=-1)        # (N,)
        color_t    = (self.color_coefs    * t_powers).sum(dim=-1)        # (N, 3)

        # NUEVO: en exp03 escala_t es polinomico, no hay clampear_escala post-step
        # como en exp01/02. Clampeamos AQUI para evitar exp(escala) gigante.
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
