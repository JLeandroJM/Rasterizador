
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
from utilidades import *
import numpy as np

grad_threshold = 1e-6

def gaussiana_inicial_central(alto, ancho, device, escala_inicial):
    
    centro = torch.tensor([[alto / 2.0, ancho / 2.0]], device=device)
    escala = torch.tensor([[float(np.log(escala_inicial)), float(np.log(escala_inicial))]], device=device)
    theta = torch.zeros(1, device=device)
    opacidad = torch.zeros(1, device=device)    

    color = torch.zeros(1, 3, device=device)           
    profundidad = torch.tensor([0.5], device=device)

    return centro, escala, theta, opacidad, color, profundidad


def main():
    device = torch.device('mps')
    densify_interval = 100
    salida = "salidas_exp2"

    # NUEVO: aseguramos que la carpeta de salida exista antes de escribir
    asegurar_carpeta(salida)

    ruta_imagen_final = "./imagenes_prueba/gatof.jpg"
    alto = 512
    ancho = 512
    target = cargar_imagen_objetivo(ruta_imagen_final, alto, ancho, device)
    guardar_imagen(target, os.path.join(salida, "target.png"))

    iteraciones = 1000
    
    densify_inicio = 200
    
    max_gaussians = 1000

    N_inicial = 30


    centro, escala, theta, opacidad, color, profundidad = inicializar_gaussianas_aleatorias(
        N_inicial, alto, ancho, device, escala_inicial=15.0
    )

    modelo = Gaussianas_2d_Torch(centro, escala, theta, opacidad, color, profundidad)
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

        
        acumulador.acumular(modelo.centro.grad)
        optimizador.step()

        clampear_escala(modelo, alto, ancho)

        losses.append(float(loss.item()))
        n_gauss_historial.append(modelo.numero_gausianas())

        # NUEVO: cada 50 iteraciones guardamos un frame para el gif
        if it % 50 == 0:
            with torch.no_grad():
                frames_gif.append(rasterizar_diferenciable(modelo, alto, ancho).cpu())


        if it >= densify_inicio and (it + 1) % densify_interval == 0:
            grad_prom = acumulador.promedio()

         
            # size_threshold ni hacer_split.
            resultado, msg = densificar(
                modelo,
                grad_promedio=grad_prom,
                grad_threshold=grad_threshold,
                max_gaussians=max_gaussians,
            )
            if resultado is not None:
                modelo.reemplazar(*resultado)
              
                optimizador = crear_optimizador(modelo, LRS_DEFAULT)
                print(f"  iteracion={it+1:4d}  Numero gau={modelo.numero_gausianas():3d}  {msg}  loss={loss.item():.4f}")
            acumulador.reset_total(modelo.numero_gausianas())

        if (it + 1) % 500 == 0:
            print(f"it={it+1:4d}  N={modelo.numero_gausianas():3d}  loss={loss.item():.4f}  ({time.time()-t0:.1f}s)")

    
    with torch.no_grad():
        render_final = rasterizar_diferenciable(modelo, alto, ancho)
    guardar_imagen(render_final, os.path.join(salida, "render_final.png"))


    guardar_curva(losses, "loss vs iteracion", "loss", os.path.join(salida, "loss.png"))
    guardar_curva(n_gauss_historial, "# gaussianas vs iteracion", "N", os.path.join(salida, "n_gaussianas.png"))


    guardar_frames(frames_gif, os.path.join(salida, "frames"))



if __name__ == "__main__":
    main()
