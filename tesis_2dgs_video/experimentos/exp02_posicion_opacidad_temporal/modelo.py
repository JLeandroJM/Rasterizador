"""
exp02 — modelo 2DGS temporal con centro y opacidad polinomicos.

centro_coefs   : (N, 2, q+1)
opacidad_coefs : (N, q+1)
escala         : (N, 2)          estatico
theta          : (N,)            estatico
color          : (N, 3)          estatico
profundidad    : (N,)            estatico, sin grad
"""
import numpy as np
import torch
from torch import nn



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

        # opacidad_coefs: a_0 = 0 -> sigmoid(0) = 0.5 inicial, resto en 0
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


    def parametros(self):
        return [self.centro_coefs, self.opacidad_coefs,
                self.escala, self.theta, self.color]


    def evaluar_en_t(self, t_norm):
        """Evalua los polinomios en t_norm. Devuelve los parametros para el rasterizador."""
        device = self.centro_coefs.device
        dtype = self.centro_coefs.dtype

        t_powers = torch.tensor(
            [t_norm ** k for k in range(self.q + 1)],
            device=device, dtype=dtype,
        )   # (q+1,)

        # centro: (N, 2, q+1) * (q+1,) -> (N, 2)
        centro_t = (self.centro_coefs * t_powers).sum(dim=-1)

        # opacidad_raw: (N, q+1) * (q+1,) -> (N,)
        # IMPORTANTE: NO aplicamos sigmoid aqui. El rasterizador la aplica.
        opacidad_raw_t = (self.opacidad_coefs * t_powers).sum(dim=-1)

        return (centro_t, self.escala, self.theta,
                opacidad_raw_t, self.color, self.profundidad)


    @torch.no_grad()
    def heatmap_vida(self, T, umbral=0.1):
        """
        Devuelve matriz (N, T) booleana donde celda (i, t) = 1 si
        sigmoid(opacidad_t(i)) > umbral. Util para visualizar cuales
        gaussianas estan "vivas" en cada frame.
        """
        device = self.centro_coefs.device
        dtype = self.centro_coefs.dtype
        ts = torch.linspace(0, 1, T, device=device, dtype=dtype)        # (T,)
        # potencias: (T, q+1)
        t_powers = torch.stack([ts ** k for k in range(self.q + 1)], dim=-1)
        # opacidad_coefs (N, q+1), t_powers (T, q+1) -> (N, T)
        op_raw = self.opacidad_coefs @ t_powers.T                       # (N, T)
        op = torch.sigmoid(op_raw)
        return (op > umbral).float().cpu().numpy(), op.cpu().numpy()


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
