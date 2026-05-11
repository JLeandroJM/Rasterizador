"""
Pipeline completo de Gaussian Splatting 2D:
  - rasterizador diferenciable
  - clone + split + prune + reset opacity
  - tensores apilados (modelo Modelo2DGS)

Corre desde dentro de la carpeta:
  cd gaussian_splating_2d
  python main.py
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gaussianas_2d import Modelo2DGS
from rasterizador import rasterizar_diferenciable
from densificacion import (
    AcumuladorGradiente,
    densificar,
    podar,
    reset_opacidad,
)
from utilidades import (
    asegurar_carpeta,
    cargar_imagen_objetivo,
    calcular_loss,
    crear_optimizador,
    clampear_escala,
    guardar_imagen,
    guardar_curva,
    guardar_frames,
    LRS_DEFAULT,
)



def main():

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"dispositivo: {device}")

    # NUEVO: la imagen objetivo se lee desde ./images/<nombre>. Cambia
    # nombre_imagen a "imagen _bolas_reflejos.jpg" para correr la otra.
    nombre_imagen = "gatof.jpg"
    ruta_objetivo = os.path.join("images", nombre_imagen)

    alto, ancho = 256, 256


    # hiperparametros del entrenamiento
    iteraciones = 1200

    # NUEVO: la carpeta de salida se nombra segun la imagen de entrada y
    # las iteraciones, asi cada corrida queda en su propia carpeta sin
    # pisar resultados anteriores.
    nombre_base = os.path.splitext(nombre_imagen)[0].strip()
    salida = f"salidas_{nombre_base}_{iteraciones}it"
    asegurar_carpeta(salida)

    objetivo = cargar_imagen_objetivo(ruta_objetivo, alto, ancho, device)
    guardar_imagen(objetivo, os.path.join(salida, "objetivo.png"))


    densify_inicio = 200
    densify_interval = 100
    prune_interval = 100
    reset_interval = 1000

    # NUEVO: hasta que iteracion seguir densificando / podando / reseteando.
    # Despues de este punto el entrenamiento es solo refinamiento (Adam
    # ajustando los parametros existentes sin agregar / quitar gaussianas).
    # Asi el render_final llega a un estado bien estabilizado y la curva
    # de loss termina suave, en vez de mostrar picos hasta el ultimo paso.
    # En el paper 3DGS hacen densif hasta ~iter 15000 y refinan hasta 30000.
    densificar_hasta = int(iteraciones * 0.7)

    # threshold ajustado a coordenadas en pixeles (no normalizadas).
    # En el paper 3DGS centro vive en (-1, 1) y usan 2e-4. Aca centro vive
    # en pixeles (0-128), entonces ‖∇centro‖ es mas chico y necesitamos
    # un threshold mucho mas bajo.
    grad_threshold = 1e-7

    max_gaussians = 2000
    size_threshold = max(alto, ancho) * 0.02      # chica vs grande para clone/split
    opacity_threshold = 0.005                     # opacidad minima antes de podar
    escala_max_prune = max(alto, ancho) * 0.5     # escala maxima antes de podar

    # arrancar con N inicial chico aleatorio. Con N=1 el sistema cae en un
    # plateau muy plano (gradiente ~0) y nunca densifica.
    N_inicial = 15
    modelo = Modelo2DGS(N_inicial, alto, ancho, device,
                        escala_inicial=15.0, gaussianas_iniciales=0)

    optimizador = crear_optimizador(modelo, LRS_DEFAULT)
    acumulador = AcumuladorGradiente(modelo.numero_gausianas(), device)

    losses = []
    n_gauss_historial = []
    opacidad_media_historial = []
    frames = []

    t0 = time.time()
    for it in range(iteraciones):

        optimizador.zero_grad()
        render = rasterizar_diferenciable(modelo, alto, ancho)
        loss = calcular_loss(render, objetivo)
        loss.backward()

        # acumulamos ‖∇centro‖ antes del step
        acumulador.acumular(modelo.centro.grad)
        optimizador.step()

        # clamp escala para evitar gaussianas gigantes -> NaN
        clampear_escala(modelo, alto, ancho)

        losses.append(float(loss.item()))
        n_gauss_historial.append(modelo.numero_gausianas())
        opacidad_media_historial.append(float(modelo.opacidades_actuales().mean().item()))

        # snapshot para los frames cada 50 iteraciones
        if it % 50 == 0:
            with torch.no_grad():
                frames.append(rasterizar_diferenciable(modelo, alto, ancho).cpu())


        # NUEVO: solo densificamos / podamos / reseteamos hasta densificar_hasta.
        # Despues entramos en fase de refinamiento puro: Adam sigue ajustando
        # los parametros pero el numero de gaussianas no cambia mas. Asi la
        # curva de loss termina suave y el render_final esta bien estabilizado.
        en_fase_densif = (it < densificar_hasta)


        # densificacion (clone + split)
        if en_fase_densif and it >= densify_inicio and (it + 1) % densify_interval == 0:

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
                # reconstruir optimizer porque cambiaron los nn.Parameter
                optimizador = crear_optimizador(modelo, LRS_DEFAULT)
                print(f"  it={it+1:4d} densif  N={modelo.numero_gausianas():3d}  {msg}")
            acumulador.reset_total(modelo.numero_gausianas())


        # pruning
        if en_fase_densif and it >= densify_inicio and (it + 1) % prune_interval == 0:

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


        # reset opacity (modifica .data in-place, NO hay que reconstruir optimizer)
        if en_fase_densif and (it + 1) % reset_interval == 0:
            reset_opacidad(modelo, valor=0.01)
            print(f"  it={it+1:4d} RESET opacity")


        if (it + 1) % 500 == 0:
            print(f"it={it+1:4d}  N={modelo.numero_gausianas():3d}  loss={loss.item():.4f}  ({time.time()-t0:.1f}s)")


    # render final
    with torch.no_grad():
        render_final = rasterizar_diferenciable(modelo, alto, ancho)
    guardar_imagen(render_final, os.path.join(salida, "render_final.png"))

    # graficas y frames
    guardar_curva(losses, "loss vs iteracion", "loss", os.path.join(salida, "loss.png"))
    guardar_curva(n_gauss_historial, "# gaussianas vs iteracion", "N", os.path.join(salida, "n_gaussianas.png"))
    guardar_curva(opacidad_media_historial, "opacidad media vs iteracion", "opacidad", os.path.join(salida, "opacidad_media.png"))
    guardar_frames(frames, os.path.join(salida, "frames"))

    print(f"listo. resultados en: {salida}")



if __name__ == "__main__":
    main()
