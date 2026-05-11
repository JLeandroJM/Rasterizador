
import numpy as np


# mirada debe serun vector
class Camara:
    def __init__(self, posicion , punto_mira , arriba , focal , size ):
        self.posicion = posicion
        self.punto_mira = punto_mira
        self.arriba = arriba
        self.focal = focal
        self.size = size

    def calcular_matriz_vista(self):
        
        z = self.punto_mira - self.posicion
        z = z/np.linalg.norm(z)

        x = np.cross(z, self.arriba)
        nx = np.linalg.norm(x)

        if nx < 1e-6:
            print("no funcionaa aca xd")

            x = np.cross(z, np.array( [1.0, 0.0, 0.0]))
            nx = np.linalg.norm(x)
        
        x = x/nx

        y = np.cross(z, x)
# apilar para contruir la matriz 3x3
        matriz_rotacion = np.stack([x, y, z], axis=0)
        traslacion = -matriz_rotacion @ self.posicion

# esto crea una amtriz donde concatenamos la rotacion y la traslacion 
        matriz_vista = np.eye(4)
        matriz_vista[:3, :3] = matriz_rotacion
        matriz_vista[:3, 3] = traslacion

        return matriz_vista
    


def crear_orbita_camara(punto_inicio , radio , angulo , altura , focal):

    angulo_radian = np.deg2rad(angulo)
    posicion = punto_inicio + np.array([radio * np.sin(angulo_radian), altura, radio * np.cos(angulo_radian)])

    camara_final = Camara(posicion = posicion,punto_mira = punto_inicio,
                          arriba = np.array([0.0, 1.0, 0.0]), focal = focal , size = (512, 512))
    return camara_final