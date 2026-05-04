import numpy as np
import matplotlib.pyplot as plt

from crear_gaussiana_3d import Gaussiana_3d , cuaternion_a_matriz_rotacion

from crear_camara import Camara , crear_orbita_camara


ALFA_MIN = 1.0 / 255.0
ALFA_MAX = 0.99

IMAGE_SIZE = (400, 400)

GAUSSIANAS = [

    # gaussiana grande roja en el centro (se cruza con varias)
    {
        "centro": [0.0, 0.0, 0.0],
        "escala": [0.6, 0.25, 0.25],
        "cuaternion": [1.0, 0.0, 0.0, 0.0],
        "color": [1.0, 0.2, 0.2],
        "opacidad": 0.85
    },

    # gaussiana verde alargada que cruza la roja en X
    {
        "centro": [0.0, 0.0, 0.0],
        "escala": [0.25, 0.6, 0.25],
        "cuaternion": [1.0, 0.0, 0.0, 0.0],
        "color": [0.2, 1.0, 0.2],
        "opacidad": 0.85
    },

    # gaussiana azul alargada en Z (forma cruz 3d con las dos anteriores)
    {
        "centro": [0.0, 0.0, 0.0],
        "escala": [0.25, 0.25, 0.6],
        "cuaternion": [1.0, 0.0, 0.0, 0.0],
        "color": [0.2, 0.4, 1.0],
        "opacidad": 0.85
    },

    # amarilla rotada 45 grados en Z (cruza la roja y la verde)
    {
        "centro": [0.3, 0.3, 0.0],
        "escala": [0.5, 0.15, 0.2],
        "cuaternion": [0.9239, 0.0, 0.0, 0.3827],
        "color": [1.0, 1.0, 0.2],
        "opacidad": 0.8
    },

    # magenta rotada 45 grados en Y (cruza la azul)
    {
        "centro": [0.0, -0.2, 0.3],
        "escala": [0.5, 0.15, 0.15],
        "cuaternion": [0.9239, 0.0, 0.3827, 0.0],
        "color": [1.0, 0.2, 1.0],
        "opacidad": 0.8
    },

    # cyan rotada en X
    {
        "centro": [-0.4, 0.2, 0.2],
        "escala": [0.15, 0.45, 0.15],
        "cuaternion": [0.9239, 0.3827, 0.0, 0.0],
        "color": [0.2, 1.0, 1.0],
        "opacidad": 0.8
    },

    # naranja arriba a la derecha
    {
        "centro": [0.7, 0.5, -0.2],
        "escala": [0.25, 0.25, 0.3],
        "cuaternion": [1.0, 0.0, 0.0, 0.0],
        "color": [1.0, 0.55, 0.0],
        "opacidad": 0.9
    },

    # violeta abajo a la izquierda (intersecta con cyan)
    {
        "centro": [-0.5, -0.4, 0.0],
        "escala": [0.3, 0.2, 0.4],
        "cuaternion": [0.9239, 0.0, 0.0, 0.3827],
        "color": [0.6, 0.2, 0.9],
        "opacidad": 0.85
    },

    # blanca pequena en el frente (cruza la azul)
    {
        "centro": [0.1, 0.1, 0.5],
        "escala": [0.18, 0.18, 0.18],
        "cuaternion": [1.0, 0.0, 0.0, 0.0],
        "color": [1.0, 1.0, 1.0],
        "opacidad": 0.7
    },

    # gris atras (la mas lejana, queda detras de las demas)
    {
        "centro": [0.2, -0.1, -0.6],
        "escala": [0.35, 0.3, 0.2],
        "cuaternion": [0.9239, 0.0, 0.0, 0.3827],
        "color": [0.5, 0.5, 0.5],
        "opacidad": 0.85
    }
]


CAMARAS = [
    {
        "angulo_camara": 0.0,
        "radio": 4.0,
        "altura": 1.0,
        "campo_vision": 45.0
    },
    {
        "angulo_camara": 45.0,
        "radio": 4.0,
        "altura": 1.5,
        "campo_vision": 45.0
    },
    {
        "angulo_camara": 90.0,
        "radio": 4.0,
        "altura": 0.5,
        "campo_vision": 45.0
    },
    {
        "angulo_camara": 180.0,
        "radio": 4.0,
        "altura": 1.0,
        "campo_vision": 45.0
    },
    {
        "angulo_camara": 270.0,
        "radio": 4.0,
        "altura": 2.0,
        "campo_vision": 45.0
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


def proyectar_covarianza_3da2d(comvarianza_3d , camara_puntos, W , funcion_x , funcion_y):

    J = jacobiano(camara_puntos, funcion_x, funcion_y)

    covarianza_2d = J @ W @ comvarianza_3d @ W.T @ J.T

    # regularizador para evitar aliasing (queda como bloque 2x2)
    covarianza_2d += 0.3 * np.eye(2)

    return covarianza_2d




def crear_escena_3d():
    lista_gaussianas = []
    for gausiana in GAUSSIANAS:
        g = Gaussiana_3d(
            centro=gausiana["centro"],
            escala=gausiana["escala"],
            cuaternion=gausiana["cuaternion"],
            color=gausiana["color"],
            opacidad=gausiana["opacidad"])

        lista_gaussianas.append(g)
    camaras = []
    for camara in CAMARAS:
        camaras.append(camara)

    return lista_gaussianas , camaras


def hacer_proyeccion(gaussiana , camara):

    matriz_vista = camara.calcular_matriz_vista()

    # bloque rotacional 3x3 y traslacion
    W = matriz_vista[:3, :3]
    t = matriz_vista[:3, 3]

    centro_mundo = np.asarray(gaussiana.centro, dtype=np.float64)
    centro_camara = W @ centro_mundo + t

    tx , ty , tz = centro_camara

    # focales en pixeles (asumimos misma fx y fy)
    funcion_x = camara.focal
    funcion_y = camara.focal

    # centro de la imagen
    cx = camara.size[0] / 2.0
    cy = camara.size[1] / 2.0

    # proyeccion perspectiva del centro al plano imagen (px = x, py = y)
    px = funcion_x * tx / tz + cx
    py = funcion_y * ty / tz + cy
    centro_2d = np.array([px, py], dtype=np.float64)

    # covarianza 3d en mundo y proyeccion EWA a 2d
    sigma_3d = covarianza_3d(gaussiana.escala, gaussiana.cuaternion)
    sigma_2d = proyectar_covarianza_3da2d(sigma_3d, centro_camara, W, funcion_x, funcion_y)

    return centro_2d , sigma_2d , tz



def rasterizar_3d(gaussianas, camara, alto, ancho):

    fondo_imagen = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    imagen = np.zeros((alto, ancho, 3), dtype=np.float64)

# proyectamos todas las gaussianas y guardamos solo las que estan delante
    centros = []
    covarianzas = []
    colores = []
    opacidades = []
    z_camaras = []

    for g in gaussianas:
        centro_2d , sigma_2d , z_cam = hacer_proyeccion(g, camara)

        # las que quedan detras de la camara no se renderizan
        if z_cam <= 0:
            continue

        centros.append(centro_2d)
        covarianzas.append(np.linalg.inv(sigma_2d))
        colores.append(np.array(g.color, dtype=np.float64))
        opacidades.append(float(g.opacidad))
        z_camaras.append(z_cam)

# ordenamos por profundidad en espacio camara (las mas cercanas primero)
    indices_ordenados = sorted(range(len(z_camaras)), key=lambda k: z_camaras[k])

    centros = [centros[k] for k in indices_ordenados]
    covarianzas = [covarianzas[k] for k in indices_ordenados]
    colores = [colores[k] for k in indices_ordenados]
    opacidades = [opacidades[k] for k in indices_ordenados]


    for i in range(alto):
        for j in range(ancho):
            transmitancia = 1.0
            Color = np.zeros(3, dtype=np.float64)

            # pixel en coordenadas (x, y) = (columna, fila)
            pixel = np.array([j + 0.5, i + 0.5], dtype=np.float64)

            for k in range(len(centros)):

                distancia_gausiana = pixel - centros[k]

                exponencial = -0.5 * distancia_gausiana @ covarianzas[k] @ distancia_gausiana

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
                    break

            Color += fondo_imagen * transmitancia

            imagen[i, j] = Color

    return imagen



def main():

    gaussianas, camaras = crear_escena_3d()
    ancho , alto = IMAGE_SIZE

    for idx , camara_c in enumerate(camaras):

        # convertimos el campo de vision (grados) a focal en pixeles
        fov_rad = np.deg2rad(camara_c["campo_vision"])
        focal = (ancho / 2.0) / np.tan(fov_rad / 2.0)

        camara = crear_orbita_camara(
            np.array([0.0, 0.0, 0.0]),
            camara_c["radio"],
            camara_c["angulo_camara"],
            camara_c["altura"],
            focal)

        # forzamos el tamano de la camara al de la imagen
        camara.size = (ancho, alto)

        print(f"Rasterizando vista {idx} ...")
        imagen = rasterizar_3d(gaussianas, camara, alto, ancho)

        fig, ax = plt.subplots(figsize=(5, 5), dpi=120)
        ax.imshow(np.clip(imagen, 0, 1))
        fig.savefig(f"resultado_3d_{idx}.png")
        plt.show()


if __name__ == "__main__":
    main()
