import numpy as np

class Gaussiana:

    def __init__(self,centro, escala, rotacion, color, opacidad , profundidad):
        self.centro = centro
        self.escala = escala
        self.rotacion_grados = rotacion
        self.color = color
        self.opacidad = opacidad
        self.profundidad = profundidad

    
    def hallar_covarianza(self):

        teta = np.deg2rad(self.rotacion_grados)

        rotacion = np.array([[np.cos(teta), -np.sin(teta)], [np.sin(teta), np.cos(teta)]])

        covarianza = np.diag(self.escala)

        resultado = rotacion @ covarianza @ covarianza.T @ rotacion.T

        return resultado
    
    