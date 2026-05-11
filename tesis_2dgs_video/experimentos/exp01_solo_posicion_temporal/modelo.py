"""
exp01 — modelo 2DGS temporal con centro polinomico, resto estatico.

centro_coefs : (N, 2, q+1)   coeficientes del polinomio por gaussiana / dim
escala       : (N, 2)        estatico
theta        : (N,)          estatico
opacidad     : (N,)          estatico
color        : (N, 3)        estatico
profundidad  : (N,)          estatico, sin grad
"""
import numpy as np
import torch
from torch import nn



class Modelo2DGSTemporalPos:

    def __init__(self, N, q, alto, ancho, device='cpu',
                 escala_inicial=4.0, semilla=0):

        self.q = q
        self.alto = alto
        self.ancho = ancho

        g = torch.Generator(device='cpu').manual_seed(semilla)

        # centro_coefs: el coef a_0 es la posicion inicial (uniforme dentro de
        # la imagen). Los coefs a_1..a_q arrancan en 0 para que el polinomio
        # empiece como constante = a_0 y aprenda movimiento desde ahi.
        centro_coefs = torch.zeros(N, 2, q + 1)
        centro_coefs[:, 0, 0] = torch.rand(N, generator=g) * alto    # filas
        centro_coefs[:, 1, 0] = torch.rand(N, generator=g) * ancho   # columnas

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


    def parametros(self):
        return [self.centro_coefs, self.escala, self.theta, self.opacidad, self.color]


    def evaluar_en_t(self, t_norm):
        """
        Evalua el polinomio en t_norm ∈ [0, 1] y devuelve los parametros que
        consume el rasterizador para ese frame.

        t_norm: escalar (Python float o tensor escalar). Las potencias se
        calculan SIN grad — t_norm es una constante para cada frame.
        """
        device = self.centro_coefs.device
        dtype = self.centro_coefs.dtype

        # potencias [t^0, t^1, ..., t^q] como tensor sin grad
        # DECISION: base monomial, no Chebyshev
        t_powers = torch.tensor(
            [t_norm ** k for k in range(self.q + 1)],
            device=device, dtype=dtype,
        )   # (q+1,)

        # broadcast: (N, 2, q+1) * (q+1,) -> sum sobre el ultimo eje -> (N, 2)
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
