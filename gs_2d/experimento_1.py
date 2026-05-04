"""
Experimento 1: conteo fijo de gaussianas (sin densificacion).

Probamos N = 4, 10, 50, 200 y comparamos visualmente. La idea es ver desde
que numero de gaussianas la imagen empieza a parecerse decentemente al target.
"""
import os
import sys
import time

import torch

# permitimos importar los modulos del mismo directorio
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gaussiana_2d_torch import Gaussianas_2d_Torch
from rasterizador_torch import rasterizar_diferenciable
from utilidades import (
    obtener_dispositivo,
    cargar_imagen_target,
    inicializar_gaussianas_aleatorias,
    calcular_loss,
    crear_optimizador,
    asegurar_carpeta,
    guardar_imagen,
    guardar_curva,
    guardar_comparacion,
    LRS_DEFAULT,
)


# NUEVO: una sola corrida de entrenamiento con N fijo
def entrenar_conteo_fijo(target, N, iteraciones, device, semilla=0, lrs=None):

    alto, ancho, _ = target.shape

    mu, sr, th, op, co, dp = inicializar_gaussianas_aleatorias(
        N, alto, ancho, device, escala_inicial=max(alto, ancho) / 20.0, semilla=semilla
    )
    modelo = Gaussianas_2d_Torch(mu, sr, th, op, co, dp)

    if lrs is None:
        lrs = LRS_DEFAULT
    optimizador = crear_optimizador(modelo, lrs)

    losses = []
    t0 = time.time()
    for it in range(iteraciones):
        optimizador.zero_grad()
        render = rasterizar_diferenciable(modelo, alto, ancho)
        loss = calcular_loss(render, target)
        loss.backward()
        optimizador.step()
        losses.append(float(loss.item()))

        if (it + 1) % 200 == 0:
            print(f"  N={N:4d} it={it+1:4d} loss={loss.item():.4f}  ({time.time()-t0:.1f}s)")

    # render final sin gradiente
    with torch.no_grad():
        render_final = rasterizar_diferenciable(modelo, alto, ancho)
    return modelo, render_final, losses


def main():
    device = obtener_dispositivo()
    print(f"dispositivo: {device}")

    salida = os.path.join(os.path.dirname(os.path.abspath(__file__)), "salidas_exp1")
    asegurar_carpeta(salida)

    # NUEVO: usamos el output del rasterizador 2d numpy como target
    ruta_target = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "resultado_orden.png")
    alto, ancho = 128, 128
    target = cargar_imagen_target(ruta_target, alto, ancho, device)
    guardar_imagen(target, os.path.join(salida, "target.png"))

    iteraciones = 2000
    Ns = [4, 10, 50, 200]

    renders = []
    titulos = []
    losses_finales = []

    for N in Ns:
        print(f"--- entrenando con N={N} ---")
        modelo, render, losses = entrenar_conteo_fijo(target, N, iteraciones, device, semilla=N)

        guardar_imagen(render, os.path.join(salida, f"render_N{N}.png"))
        guardar_curva(losses, f"loss N={N}", "loss", os.path.join(salida, f"loss_N{N}.png"))

        renders.append(render)
        titulos.append(f"N={N}")
        losses_finales.append(losses[-1])

    # comparacion side-by-side
    guardar_comparacion(target, renders, titulos, losses_finales,
                        os.path.join(salida, "comparacion.png"))
    print(f"listo. resultados en: {salida}")


if __name__ == "__main__":
    main()
