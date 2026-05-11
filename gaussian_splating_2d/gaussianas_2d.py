import numpy as np
import torch
from torch import nn



# NUEVO: tensores apilados (N, K) en lugar de una lista de objetos.
# Asi al densificar (clone, split, prune) solo concatenas filas a cada
# tensor o filtras filas con una mascara booleana. Es la forma como estan
# organizadas las implementaciones reales de 3DGS.
#
# Si manana quiero agregar un parametro experimental nuevo (beta, skew,
# gradiente de color, ...), lo agrego como otro nn.Parameter de shape
# (N, K_nuevo) y la densificacion sigue funcionando sin tocar nada.
class Modelo2DGS:

    def __init__(self, N, alto, ancho, device='cpu',
                 escala_inicial=4.0, gaussianas_iniciales=0):

        # NUEVO: gaussianas_iniciales se usa como semilla del generador,
        # asi cada corrida con N distinto puede arrancar diferente.
        g = torch.Generator(device='cpu').manual_seed(gaussianas_iniciales)

        # centro en pixeles (fila, columna), uniforme dentro de la imagen
        centro = torch.empty(N, 2)
        centro[:, 0] = torch.rand(N, generator=g) * alto
        centro[:, 1] = torch.rand(N, generator=g) * ancho

        # escala_raw: la escala real es exp(escala) -> siempre positiva
        escala = torch.full((N, 2), float(np.log(escala_inicial)))

        # rotacion en radianes, uniforme en (-pi, pi)
        theta = (torch.rand(N, generator=g) - 0.5) * 2 * np.pi

        # opacidad_raw: sigmoid(opacidad) en (0, 1). Inicial 0 -> 0.5
        opacidad = torch.zeros(N)

        # color_raw: sigmoid(color) en (0, 1) por canal
        color = (torch.rand(N, 3, generator=g) - 0.5) * 2.0

        # profundidad: NO se optimiza, solo se usa para ordenar
        profundidad = torch.rand(N, generator=g)

        # parametros optimizables (entran al optimizer)
        self.centro   = nn.Parameter(centro.to(device))
        self.escala   = nn.Parameter(escala.to(device))
        self.theta    = nn.Parameter(theta.to(device))
        self.opacidad = nn.Parameter(opacidad.to(device))
        self.color    = nn.Parameter(color.to(device))

        # tensor regular sin requires_grad
        self.profundidad = profundidad.to(device)


    def numero_gausianas(self):
        return self.centro.shape[0]


    # parametros que se le pasan al optimizer
    def parametros(self):
        return [self.centro, self.escala, self.theta, self.opacidad, self.color]


    # NUEVO: activaciones aplicadas (los valores "reales" que ve el rasterizador)
    @torch.no_grad()
    def escalas_actuales(self):
        return torch.exp(self.escala)


    @torch.no_grad()
    def opacidades_actuales(self):
        return torch.sigmoid(self.opacidad)


    @torch.no_grad()
    def colores_actuales(self):
        return torch.sigmoid(self.color)


    # NUEVO: reemplazar TODOS los tensores apilados por nuevos (cambia N).
    # Despues hay que reconstruir el optimizer porque los nn.Parameter son
    # objetos nuevos y el optimizer guarda referencias a los viejos.
    @torch.no_grad()
    def reemplazar(self, centro, escala, theta, opacidad, color, profundidad):
        device = self.centro.device
        self.centro   = nn.Parameter(centro.detach().to(device))
        self.escala   = nn.Parameter(escala.detach().to(device))
        self.theta    = nn.Parameter(theta.detach().to(device))
        self.opacidad = nn.Parameter(opacidad.detach().to(device))
        self.color    = nn.Parameter(color.detach().to(device))
        self.profundidad = profundidad.detach().to(device)
