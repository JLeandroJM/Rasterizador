import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def covarianza_3d()



def crear_escena():
    for gausiana in GAUSSIANAS:
        g = Gaussiana(
            centro=gausiana["centro"],
            escala=gausiana["escala"],
            rotacion=gausiana["rotacion_grados"],
            color=gausiana["color"],
            opacidad=gausiana["opacidad"],
            profundidad=gausiana["profundidad"])
        
        lista_gaussianas.append(g)

    return lista_gaussianas