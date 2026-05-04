import torch


# NUEVO: en pytorch necesitamos parametrizar las gaussianas con tensores
# que tengan requires_grad=True para poder hacer backprop. Las activaciones
# (exp / sigmoid) las aplicamos al usar los parametros, no al guardarlos.
class Gaussianas_2d_Torch:

    def __init__(self, mu, scale_raw, theta_raw, opacity_raw, color_raw, depth):
        # mu: (N, 2) en coordenadas (fila, columna), igual al rasterizador 2d numpy
        self.mu = mu.detach().clone().requires_grad_(True)

        # scale_raw: (N, 2). La escala real es exp(scale_raw) -> siempre positiva
        self.scale_raw = scale_raw.detach().clone().requires_grad_(True)

        # theta_raw: (N,) en radianes, directo
        self.theta_raw = theta_raw.detach().clone().requires_grad_(True)

        # opacity_raw: (N,). La opacidad real es sigmoid(opacity_raw) en (0, 1)
        self.opacity_raw = opacity_raw.detach().clone().requires_grad_(True)

        # color_raw: (N, 3). El color real es sigmoid(color_raw) en (0, 1)
        self.color_raw = color_raw.detach().clone().requires_grad_(True)

        # depth: (N,) escalar fijo, NO se optimiza, solo sirve para ordenar
        self.depth = depth.detach().clone()

    def numero_gausianas(self):
        return self.mu.shape[0]

    def parametros(self):
        return [self.mu, self.scale_raw, self.theta_raw, self.opacity_raw, self.color_raw]

    # NUEVO: para densificar / podar reemplazamos los tensores enteros
    # (cambia el numero de gaussianas) y luego se recrea el optimizador
    def reemplazar(self, mu, scale_raw, theta_raw, opacity_raw, color_raw, depth):
        self.mu = mu.detach().clone().requires_grad_(True)
        self.scale_raw = scale_raw.detach().clone().requires_grad_(True)
        self.theta_raw = theta_raw.detach().clone().requires_grad_(True)
        self.opacity_raw = opacity_raw.detach().clone().requires_grad_(True)
        self.color_raw = color_raw.detach().clone().requires_grad_(True)
        self.depth = depth.detach().clone()

    @torch.no_grad()
    def escalas_actuales(self):
        return torch.exp(self.scale_raw)

    @torch.no_grad()
    def opacidades_actuales(self):
        return torch.sigmoid(self.opacity_raw)

    @torch.no_grad()
    def colores_actuales(self):
        return torch.sigmoid(self.color_raw)
