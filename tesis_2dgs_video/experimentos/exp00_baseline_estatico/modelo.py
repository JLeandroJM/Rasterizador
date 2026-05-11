"""
exp00 — modelo 2DGS estatico (sin temporal). Un modelo por frame.
"""
import numpy as np
import torch
from torch import nn



class Modelo2DGSEstatico:
    """Tensores apilados (N, K), todos los parametros son estaticos para un frame."""

    def __init__(self, N, alto, ancho, device='cpu',
                 escala_inicial=4.0, semilla=0):

        g = torch.Generator(device='cpu').manual_seed(semilla)

        centro = torch.empty(N, 2)
        centro[:, 0] = torch.rand(N, generator=g) * alto
        centro[:, 1] = torch.rand(N, generator=g) * ancho

        escala = torch.full((N, 2), float(np.log(escala_inicial)))
        theta = (torch.rand(N, generator=g) - 0.5) * 2 * np.pi
        opacidad = torch.zeros(N)
        color = (torch.rand(N, 3, generator=g) - 0.5) * 2.0
        profundidad = torch.rand(N, generator=g)

        self.centro   = nn.Parameter(centro.to(device))
        self.escala   = nn.Parameter(escala.to(device))
        self.theta    = nn.Parameter(theta.to(device))
        self.opacidad = nn.Parameter(opacidad.to(device))
        self.color    = nn.Parameter(color.to(device))
        self.profundidad = profundidad.to(device)


    def numero_gausianas(self):
        return self.centro.shape[0]


    def parametros(self):
        return [self.centro, self.escala, self.theta, self.opacidad, self.color]


    def evaluar(self):
        """En el modelo estatico, "evaluar" solo devuelve los parametros tal cual."""
        return (self.centro, self.escala, self.theta,
                self.opacidad, self.color, self.profundidad)


    def state_dict(self):
        return {
            "centro": self.centro.detach().cpu(),
            "escala": self.escala.detach().cpu(),
            "theta": self.theta.detach().cpu(),
            "opacidad": self.opacidad.detach().cpu(),
            "color": self.color.detach().cpu(),
            "profundidad": self.profundidad.detach().cpu(),
        }
