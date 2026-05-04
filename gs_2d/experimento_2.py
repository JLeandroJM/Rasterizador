"""
Experimento 2: solo crecimiento (clone, sin split ni prune).

Empezamos con N=1 gaussiana en el centro y cada `densify_interval` pasos
clonamos las gaussianas con gradiente alto. Sin split, sin prune.
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gaussiana_2d_torch import Gaussianas_2d_Torch
from rasterizador_torch import rasterizar_diferenciable
from densificacion import (
    AcumuladorGradiente,
    densificar,
)
from utilidades import (
    obtener_dispositivo,
    cargar_imagen_target,
    calcular_loss,
    crear_optimizador,
    asegurar_carpeta,
    guardar_imagen,
    guardar_curva,
    crear_gif,
    LRS_DEFAULT,
)


# NUEVO: una gaussiana inicial en el centro de la imagen
def gaussiana_inicial_central(alto, ancho, device, escala_inicial):
    import numpy as np
    mu = torch.tensor([[alto / 2.0, ancho / 2.0]], device=device)
    sr = torch.tensor([[float(np.log(escala_inicial)), float(np.log(escala_inicial))]], device=device)
    th = torch.zeros(1, device=device)
    op = torch.zeros(1, device=device)              # sigmoid(0) = 0.5
    co = torch.zeros(1, 3, device=device)           # sigmoid(0) = 0.5 -> gris medio
    dp = torch.tensor([0.5], device=device)
    return mu, sr, th, op, co, dp


def main():
    device = obtener_dispositivo()
    print(f"dispositivo: {device}")

    salida = os.path.join(os.path.dirname(os.path.abspath(__file__)), "salidas_exp2")
    asegurar_carpeta(salida)

    ruta_target = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "resultado_orden.png")
    alto, ancho = 128, 128
    target = cargar_imagen_target(ruta_target, alto, ancho, device)
    guardar_imagen(target, os.path.join(salida, "target.png"))

    iteraciones = 3000
    densify_interval = 100
    densify_inicio = 200
    grad_threshold = 0.0002
    max_gaussians = 200

    # NUEVO: arrancamos con UNA sola gaussiana
    mu, sr, th, op, co, dp = gaussiana_inicial_central(alto, ancho, device, escala_inicial=20.0)
    modelo = Gaussianas_2d_Torch(mu, sr, th, op, co, dp)
    optimizador = crear_optimizador(modelo, LRS_DEFAULT)

    acumulador = AcumuladorGradiente(modelo.numero_gausianas(), device)

    losses = []
    n_gauss_historial = []
    frames_gif = []

    t0 = time.time()
    for it in range(iteraciones):
        optimizador.zero_grad()
        render = rasterizar_diferenciable(modelo, alto, ancho)
        loss = calcular_loss(render, target)
        loss.backward()

        # NUEVO: acumulamos ‖∇μ‖ antes del step
        acumulador.acumular(modelo.mu.grad)
        optimizador.step()

        losses.append(float(loss.item()))
        n_gauss_historial.append(modelo.numero_gausianas())

        # NUEVO: cada 50 iteraciones guardamos un frame para el gif
        if it % 50 == 0:
            with torch.no_grad():
                frames_gif.append(rasterizar_diferenciable(modelo, alto, ancho).cpu())

        # NUEVO: densificacion cada densify_interval pasos (solo clone aqui)
        if it >= densify_inicio and (it + 1) % densify_interval == 0:
            grad_prom = acumulador.promedio()
            # forzamos hacer_split=False y size_threshold gigante para que TODAS
            # las que entren caigan en clone
            resultado, msg = densificar(
                modelo,
                grad_promedio=grad_prom,
                grad_threshold=grad_threshold,
                size_threshold=1e9,
                max_gaussians=max_gaussians,
                hacer_split=False,
            )
            if resultado is not None:
                modelo.reemplazar(*resultado)
                # rebuild optimizer porque cambiaron los tensores
                optimizador = crear_optimizador(modelo, LRS_DEFAULT)
                print(f"  it={it+1:4d}  N={modelo.numero_gausianas():3d}  {msg}  loss={loss.item():.4f}")
            acumulador.reset_total(modelo.numero_gausianas())

        if (it + 1) % 500 == 0:
            print(f"it={it+1:4d}  N={modelo.numero_gausianas():3d}  loss={loss.item():.4f}  ({time.time()-t0:.1f}s)")

    # render final
    with torch.no_grad():
        render_final = rasterizar_diferenciable(modelo, alto, ancho)
    guardar_imagen(render_final, os.path.join(salida, "render_final.png"))

    # graficas y gif
    guardar_curva(losses, "loss vs iteracion", "loss", os.path.join(salida, "loss.png"))
    guardar_curva(n_gauss_historial, "# gaussianas vs iteracion", "N", os.path.join(salida, "n_gaussianas.png"))
    crear_gif(frames_gif, os.path.join(salida, "entrenamiento.gif"), duracion=0.08)
    print(f"listo. resultados en: {salida}")


if __name__ == "__main__":
    main()
