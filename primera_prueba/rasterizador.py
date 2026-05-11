import numpy as np
import matplotlib.pyplot as plt

from crear_gausiana import Gaussiana

ALFA_MIN = 1.0 / 255.0
ALFA_MAX = 0.99


# IMAGE_SIZE = (512, 512)

# GAUSSIANAS = [
#     {
#         "centro": [200, 256],
#         "escala": [160, 130],
#         "rotacion_grados": 25,
#         "color": [1.0, 0.25, 0.25],
#         "opacidad": 0.85,
#         "profundidad": 2.0,
#     },
#     {
#         "centro": [320, 256],
#         "escala": [150, 150],
#         "rotacion_grados": 0,
#         "color": [0.2, 0.5, 1.0],
#         "opacidad": 0.85,
#         "profundidad": 1.0,
#     },
# ]

IMAGE_SIZE = (512, 512)

GAUSSIANAS = [
   
    {
        "centro": [256, 256],
        "escala": [180, 180],
        "rotacion_grados": 0,
        "color": [0.1, 0.3, 1.0], # Azul
        "opacidad": 0.9,
        "profundidad": 5.0,       # La más lejana
    },
   
    {
        "centro": [200, 200],
        "escala": [140, 40],
        "rotacion_grados": 45,
        "color": [1.0, 0.1, 0.1], # Rojo
        "opacidad": 0.75,
        "profundidad": 4.0,
    },
   
    {
        "centro": [312, 200],
        "escala": [140, 40],
        "rotacion_grados": -45,
        "color": [0.1, 1.0, 0.1], # Verde
        "opacidad": 0.75,
        "profundidad": 3.0,
    },

    {
        "centro": [256, 320],
        "escala": [90, 90],
        "rotacion_grados": 0,
        "color": [1.0, 1.0, 0.1], # Amarillo
        "opacidad": 0.6,
        "profundidad": 2.0,
    },
   
    {
        "centro": [256, 256],
        "escala": [20, 200],
        "rotacion_grados": 90,
        "color": [1.0, 0.0, 1.0], 
        "opacidad": 0.4,
        "profundidad": 0.5,      
    },
]

lista_gaussianas = []


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
        

def rasterizar(gausianas, alto, ancho):

    fondo_imagen = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    imagen = np.zeros((alto, ancho, 3), dtype=np.float64)

   # transparencia_mapeo = np.ones((alto, ancho), dtype=np.float64)

    

# me devuelve los indices de las gaussianas ordenadas por profundidad

    gausianas_ordenadas = sorted(gausianas, key=lambda g: g.profundidad)
   # gausianas_ordenadas = sorted(gausianas, key=lambda g: g.profundidad, reverse=True)

# apra guardar paramtros
    covarianzas =[]
    centros = []
    colores = []
    opacidades = []

    for gauss in gausianas_ordenadas:
        #gauss = gausianas[i]
        covarianza = gauss.hallar_covarianza()
        #con la inversa es
        covarianzas.append(np.linalg.inv(covarianza))

        centros.append(np.array(gauss.centro, dtype=np.float64))
        colores.append(np.array(gauss.color, dtype=np.float64))
        
        opacidades.append(float(gauss.opacidad))


    # para graficarlas

    for i in range(alto):
        for j in range(ancho):
            transmitancia = 1.0
            Color = np.zeros(3, dtype=np.float64)
#ahi no se porque mas 0.5``
            pixel = np.array([i+0.5, j+0.5], dtype=np.float64)

            limite = len(gausianas_ordenadas)

            for k in range(len(gausianas_ordenadas)):

                #distancia entre el pizel y la gaussiana 
                distancia_gausiana = pixel - centros[k]

                #calculamos la parte exponencial de la gaussiana

                exponencial = -0.5 * distancia_gausiana @ covarianzas[k] @ distancia_gausiana

# validacion que no sea positivo
                if exponencial > 0:
                    valor_gaussiana = 1.0
                else:   
                    valor_gaussiana = np.exp(exponencial)

                alfa = opacidades[k] * valor_gaussiana

                if alfa < ALFA_MIN:
                    continue
                elif alfa > ALFA_MAX:
                    alfa = ALFA_MAX

                Color += colores[k] * alfa * transmitancia

                transmitancia *= (1.0 - alfa)

                if transmitancia < ALFA_MIN:
                    limite = k+1
                    break

            #componer el fondo con el color calculado

            Color += np.asarray(fondo_imagen, dtype=np.float64) * transmitancia

            imagen[i, j] = Color
            #transparencia_mapeo[i, j] = transmitancia
    return imagen

                   
def renderizar_una_gaussiana(gaussianas, ancho , alto):
    salida= []
    for g in gaussianas:
        covarianza_invertida = np.linalg.inv(g.hallar_covarianza())
        rgba = np.zeros((alto, ancho, 4), dtype=np.float64)

        for i in range(alto):
            for j in range(ancho):

                distancia_gausiana = np.array([i + g.centro[0], j + g.centro[1]]) 

                exponencial = -0.5 * distancia_gausiana @ covarianza_invertida @ distancia_gausiana

                if exponencial > 0:
                    valor_gaussiana = 1.0
                else:   
                    valor_gaussiana = np.exp(exponencial)
                
                alfa = g.opacidad * valor_gaussiana

                if alfa < ALFA_MIN:
                    continue
                elif alfa > ALFA_MAX:
                    alfa = ALFA_MAX
                
                rgba[i,j,:3] = g.color
                rgba[i,j,3] = alfa
        
        salida.append(rgba)

    return salida


             