
import os
import sys
import time

import torch



from gaussiana_2d_torch import Gaussianas_2d_Torch
from rasterizador_torch import rasterizar_diferenciable
from utilidades import *

PARAMETROS_DEFECTO = {
    'centro':      1e-3 * 100,  
    'escala':   5e-3,
    'theta':   1e-3,
    'opacidad': 5e-2,
    'color':   1e-2,
}

def entrenamiento(imagen_modelo, N, iteraciones, device, gaussianas_iniciales=0):

    alto, ancho, _ = imagen_modelo.shape

    centro, escala, tetha, opacidad, color, profundidad = inicializar_gaussianas_aleatorias(
        N, alto, ancho, device, escala_inicial=max(alto, ancho) / 20.0, gaussianas_iniciales=gaussianas_iniciales
    )

    modelo = Gaussianas_2d_Torch(centro, escala, tetha, opacidad, color, profundidad)

    
    lrs = PARAMETROS_DEFECTO
    optimizador = crear_optimizador(modelo, lrs)

    losses = []

    t0 = time.time()
    for it in range(iteraciones):
        optimizador.zero_grad()

        render = rasterizar_diferenciable(modelo, alto, ancho)
        loss = calcular_loss(render, imagen_modelo)
        loss.backward()
        optimizador.step()
        losses.append(float(loss.item()))

        if (it + 1) % 200 == 0:
            print(f"  N={N:4d} it={it+1:4d} loss={loss.item():.4f}  ({time.time()-t0:.1f}s)")

   
    with torch.no_grad():
        render_final = rasterizar_diferenciable(modelo, alto, ancho)
    return modelo, render_final, losses


def main():

    device = torch.device('mps')

    salida = "salidas_exp1"

    
    asegurar_carpeta(salida)

    ruta_imagen_modelo = "./imagenes_prueba/gatof.jpg"
    alto = 128
    ancho = 128


    img = Image.open(ruta_imagen_modelo).convert('RGB').resize((ancho, alto))
    arr = np.array(img, dtype=np.float32) / 255.0
    imagen_modelo = torch.from_numpy(arr).to(device)


    arr = imagen_modelo.detach().clamp(0, 1).cpu().numpy()
    arr_uint8 = (arr * 255).astype(np.uint8)
    Image.fromarray(arr_uint8).save(os.path.join(salida, "esperado.png"))

    iteraciones = 2000
    Ns = [300]

    renders = []
    titulos = []
    losses_finales = []

    for N in Ns:
        print(f"entrenando con N={N} ---------------")
        modelo, render, losses = entrenamiento(imagen_modelo, N, iteraciones, device, gaussianas_iniciales=N)

        guardar_imagen(render, os.path.join(salida, f"render_N{N}.png"))
        guardar_curva(losses, f"loss n_gaussianas={N}", "loss", os.path.join(salida, f"loss_N{N}.png"))

        renders.append(render)
        titulos.append(f"N={N}")
        losses_finales.append(losses[-1])

    # comparacion side-by-side
    guardar_comparacion(imagen_modelo, renders, titulos, losses_finales,
                        os.path.join(salida, "comparacion.png"))
   


if __name__ == "__main__":
    main()
