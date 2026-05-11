import numpy as np

class Gaussiana_3d:
    def __init__(self , centro , escala , cuaternion , color , opacidad):
        self.centro = centro
        self.escala = escala
        self.cuaternion = cuaternion
        self.color = color
        self.opacidad = opacidad

    
    

def cuaternion_a_matriz_rotacion(cuaternion):

    cuaternion = np.asarray(cuaternion, dtype=np.float64)

    normalizado = np.linalg.norm(cuaternion)

    if normalizado == 0:
        return np.eye(3) # mariz indentidad
    

    w, x, y, z = cuaternion/normalizado

    matriz_rotacion = np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],

        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ], dtype=np.float64)
    
    return matriz_rotacion