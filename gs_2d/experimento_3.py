"""
Experimento 3: crecimiento con clone + split (sin prune).

Distinguimos clone (gaussianas chicas con gradiente alto -> duplicar) vs
split (gaussianas grandes con gradiente alto -> 2 hijos sampleados de la
covarianza, escala / 1.6).
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
    guardar_overlay_tamanos,
    crear_gif,
    LRS_DEFAULT,
)
from experimento_2 import gaussiana_inicial_central


def main():
    device = obtener_dispositivo()
    print(f"dispositivo: {device}")

    salida = os.path.join(os.path.dirname(os.path.abspath(__file__)), "salidas_exp3")
    asegurar_carpeta(salida)

    ruta_target = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "resultado_orden.png")
    alto, ancho = 128, 128
    target = cargar_imagen_target(ruta_target, alto, ancho, device)
    guardar_imagen(target, os.path.join(salida, "target.png"))

    iteraciones = 3000
    densify_interval = 100
    densify_inicio = 200
    grad_threshold = 0.0002
    max_gaussians = 500

    # NUEVO: el size_threshold separa "chica" (clone) de "grande" (split).
    # En 2D usamos un porcentaje del lado de la imagen.
    size_threshold = max(alto, ancho) * 0.05

    mu, sr, th, op, co, dp = gaussiana_inicial_central(alto, ancho, device, escala_inicial=20.0)
    modelo = Gaussianas_2d_Torch(mu, sr, th, op, co, dp)
    optimizador = crear_optimizador(modelo, LRS_DEFAULT)

    acumulador = AcumuladorGradiente(modelo.numero_gausianas(), device)

    losses = []
    n_gauss_historial = []
    frames_gif = []

    # NUEVO: snapshots para overlay de tamanos en momentos clave
    snapshots_iters = [500, 1500, 2999]

    t0 = time.time()
    for it in range(iteraciones):
        optimizador.zero_grad()
        render = rasterizar_diferenciable(modelo, alto, ancho)
        loss = calcular_loss(render, target)
        loss.backward()

        acumulador.acumular(modelo.mu.grad)
        optimizador.step()

        losses.append(float(loss.item()))
        n_gauss_historial.append(modelo.numero_gausianas())

        if it % 50 == 0:
            with torch.no_grad():
                frames_gif.append(rasterizar_diferenciable(modelo, alto, ancho).cpu())

        # NUEVO: aqui ya activamos hacer_split=True
        if it >= densify_inicio and (it + 1) % densify_interval == 0:
            grad_prom = acumulador.promedio()
            resultado, msg = densificar(
                modelo,
                grad_promedio=grad_prom,
                grad_threshold=grad_threshold,
                size_threshold=size_threshold,
                max_gaussians=max_gaussians,
                hacer_split=True,
            )
            if resultado is not None:
                modelo.reemplazar(*resultado)
                optimizador = crear_optimizador(modelo, LRS_DEFAULT)
                print(f"  it={it+1:4d}  N={modelo.numero_gausianas():3d}  {msg}  loss={loss.item():.4f}")
            acumulador.reset_total(modelo.numero_gausianas())

        # snapshots de tamano de gaussianas
        if it in snapshots_iters:
            with torch.no_grad():
                render_snap = rasterizar_diferenciable(modelo, alto, ancho)
            guardar_overlay_tamanos(
                render_snap,
                modelo.mu,
                modelo.escalas_actuales(),
                os.path.join(salida, f"overlay_tamanos_it{it}.png"),
                titulo=f"iter {it}, N={modelo.numero_gausianas()}",
            )

        if (it + 1) % 500 == 0:
            print(f"it={it+1:4d}  N={modelo.numero_gausianas():3d}  loss={loss.item():.4f}  ({time.time()-t0:.1f}s)")

    with torch.no_grad():
        render_final = rasterizar_diferenciable(modelo, alto, ancho)
    guardar_imagen(render_final, os.path.join(salida, "render_final.png"))

    guardar_curva(losses, "loss vs iteracion", "loss", os.path.join(salida, "loss.png"))
    guardar_curva(n_gauss_historial, "# gaussianas vs iteracion", "N", os.path.join(salida, "n_gaussianas.png"))
    crear_gif(frames_gif, os.path.join(salida, "entrenamiento.gif"), duracion=0.08)
    print(f"listo. resultados en: {salida}")


if __name__ == "__main__":
    main()
