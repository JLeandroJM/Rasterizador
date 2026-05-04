"""
Experimento 5: pipeline completo + reset de opacity.

Cada `reset_interval` pasos forzamos opacity_raw = inverse_sigmoid(0.01)
para todas las gaussianas. Esto las hace casi transparentes y obliga al
optimizer a "redescubrir" cuales realmente vale la pena mantener.
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
    podar,
    reset_opacidad,
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
from experimento_2 import gaussiana_inicial_central


def main():
    device = obtener_dispositivo()
    print(f"dispositivo: {device}")

    salida = os.path.join(os.path.dirname(os.path.abspath(__file__)), "salidas_exp5")
    asegurar_carpeta(salida)

    ruta_target = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "resultado_orden.png")
    alto, ancho = 128, 128
    target = cargar_imagen_target(ruta_target, alto, ancho, device)
    guardar_imagen(target, os.path.join(salida, "target.png"))

    iteraciones = 4000
    densify_interval = 100
    prune_interval = 100
    densify_inicio = 200
    grad_threshold = 0.0002
    max_gaussians = 500
    size_threshold = max(alto, ancho) * 0.05
    opacity_threshold = 0.005
    escala_max_prune = max(alto, ancho) * 0.5

    # NUEVO: cada reset_interval pasos hacemos reset de opacity
    reset_interval = 1000
    reset_valor = 0.01

    mu, sr, th, op, co, dp = gaussiana_inicial_central(alto, ancho, device, escala_inicial=20.0)
    modelo = Gaussianas_2d_Torch(mu, sr, th, op, co, dp)
    optimizador = crear_optimizador(modelo, LRS_DEFAULT)
    acumulador = AcumuladorGradiente(modelo.numero_gausianas(), device)

    losses = []
    n_gauss_historial = []
    opacidad_media_historial = []
    frames_gif = []

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
        opacidad_media_historial.append(float(modelo.opacidades_actuales().mean().item()))

        if it % 50 == 0:
            with torch.no_grad():
                frames_gif.append(rasterizar_diferenciable(modelo, alto, ancho).cpu())

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
                print(f"  it={it+1:4d} densif  N={modelo.numero_gausianas():3d}  {msg}")
            acumulador.reset_total(modelo.numero_gausianas())

        if it >= densify_inicio and (it + 1) % prune_interval == 0:
            resultado, eliminadas = podar(
                modelo,
                opacity_threshold=opacity_threshold,
                escala_max=escala_max_prune,
            )
            if resultado is not None:
                modelo.reemplazar(*resultado)
                optimizador = crear_optimizador(modelo, LRS_DEFAULT)
                acumulador.reset_total(modelo.numero_gausianas())
                print(f"  it={it+1:4d} prune   N={modelo.numero_gausianas():3d}  eliminadas={eliminadas}")

        # NUEVO: reset de opacity
        if (it + 1) % reset_interval == 0:
            reset_opacidad(modelo, valor=reset_valor)
            print(f"  it={it+1:4d} RESET opacity -> {reset_valor}")

        if (it + 1) % 500 == 0:
            print(f"it={it+1:4d}  N={modelo.numero_gausianas():3d}  loss={loss.item():.4f}  ({time.time()-t0:.1f}s)")

    with torch.no_grad():
        render_final = rasterizar_diferenciable(modelo, alto, ancho)
    guardar_imagen(render_final, os.path.join(salida, "render_final.png"))

    guardar_curva(losses, "loss vs iteracion (con resets)", "loss", os.path.join(salida, "loss.png"))
    guardar_curva(n_gauss_historial, "# gaussianas vs iteracion", "N", os.path.join(salida, "n_gaussianas.png"))
    guardar_curva(opacidad_media_historial, "opacidad media vs iteracion", "opacidad", os.path.join(salida, "opacidad_media.png"))
    crear_gif(frames_gif, os.path.join(salida, "entrenamiento.gif"), duracion=0.08)
    print(f"listo. resultados en: {salida}")


if __name__ == "__main__":
    main()
