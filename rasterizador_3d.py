import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from crear_gausiana_3d import Gaussiana_3d , cuaternion_a_matriz_rotacion

from crear_camara import Camara , crear_orbita_camara


IMAGE_SIZE = (400, 400)

GAUSSIANAS = [
   
    {
        "centro": [-1.0, -1.0, -1.0],   
        "escala": [0.18, 0.18, 0.18], 
        "cuaternion": [1.0, 0.0, 0.0, 0.0], 
        "color": [1.0, 0.2, 0.2],      
        "opacidad": 0.9
    },
    
    {
        "centro": [-1.0, -1.0, 1.0],
        "escala": [0.18, 0.18, 0.18],
        "cuaternion": [1.0, 0.0, 0.0, 0.0],
        "color": [0.2, 1.0, 0.2],  
        "opacidad": 0.9
    },
   
    {
        "centro": [-1.0, 1.0, -1.0],
        "escala": [0.18, 0.18, 0.18],
        "cuaternion": [1.0, 0.0, 0.0, 0.0],
        "color": [0.2, 0.2, 1.0],     
        "opacidad": 0.9
    }
]


CAMARAS = [
    {
        "angulo_camara": 0.0,  # Ángulo de la cámara alrededor del centro
        "radio": 4.0,     # Qué tan lejos está la cámara del centro
        "altura": 1.5,    
        "campo_vision": 45.0    # Campo de visión
    }
]


def covarianza_3d(escala , cuaternion):
    
    matriz_rotacion = cuaternion_a_matriz_rotacion(cuaternion)
    escala_matriz = np.diag(escala)
    comvarianza_3d = matriz_rotacion @ escala_matriz @ escala_matriz.T @ matriz_rotacion.T

    return comvarianza_3d



def jacobiano(camara_puntos , funcion_x , funcion_y):

    tx , ty , tz = camara_puntos

    jacobiano = np.array([
        [funcion_x / tz, 0.0, -funcion_x * tx / (tz**2)],
        [0.0, funcion_y / tz, -funcion_y * ty / (tz**2)]
        
    ], dtype=np.float64)

    return jacobiano


def proyecctar_covarianza_3da2d(covarianza_3d ,camara_puntos, matriz_vista , funcion_x , funcion_y):

    jacobiano = jacobiano(camara_puntos, funcion_x, funcion_y)

    covarianza_2d = jacobiano @ matriz_vista @ covarianza_3d @ matriz_vista.T @ jacobiano.T

    return covarianza_2d




def crear_escena_3d():
    lista_gaussianas = []
    for gausiana in GAUSSIANAS:
        g = Gaussiana_3d(
            centro=gausiana["centro"],
            escala=gausiana["escala"],
            rotacion=gausiana["rotacion_grados"],
            color=gausiana["color"],
            opacidad=gausiana["opacidad"],
            profundidad=gausiana["profundidad"])
        
        lista_gaussianas.append(g)
    camaras = []
    for camara in CAMARAS:
        camaras.append(camara)

    return lista_gaussianas , camaras

def hacer_proyeccion(gaussiana , camara):
    



def main():

    gaussianas, camaras = crear_escena_3d()
    cantidad_vistas = len(camaras)
    vistas = []
    pop_indices = []

    for i, camara_c in enumerate(camaras):
        camara_vista = crear_orbita_camara(np.array([0.0, 0.0, 0.0]), camara_c["radio"], camara_c["angulo_camara"], camara_c["altura"], camara_c["campo_vision"])




