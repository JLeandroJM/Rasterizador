import torch



class Gaussianas_2d_Torch:

    def __init__(self, centro, escala, theta, opacidad, color, profundidad):
        
        self.centro = centro.detach().clone().requires_grad_(True)
        self.escala = escala.detach().clone().requires_grad_(True)
        self.theta = theta.detach().clone().requires_grad_(True)
        self.opacidad = opacidad.detach().clone().requires_grad_(True)
        self.color = color.detach().clone().requires_grad_(True)
        self.profundidad = profundidad.detach().clone()

    def numero_gausianas(self):
        return self.centro.shape[0]
    


    def parametros(self):
        return [self.centro, 
                self.escala, self.theta, 
                self.opacidad, self.color]

    
    # ( ..cambia el numero de gaussianas - para el dividir y clonar)
    def reemplazar(self, centro, escala, theta, opacidad, color, profundidad):
        
        self.centro = centro.detach().clone().requires_grad_(True)
        self.escala = escala.detach().clone().requires_grad_(True)
        self.theta = theta.detach().clone().requires_grad_(True)

        self.opacidad = opacidad.detach().clone().requires_grad_(True)
        self.color = color.detach().clone().requires_grad_(True)
        self.profundidad = profundidad.detach().clone()


    def escalas_actuales(self):
        return torch.exp(self.escala)


    def opacidades_actuales(self):
        return torch.sigmoid(self.opacidad)


    def colores_actuales(self):
        return torch.sigmoid(self.color)
