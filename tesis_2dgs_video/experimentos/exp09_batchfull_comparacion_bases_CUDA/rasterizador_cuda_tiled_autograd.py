import os
import sys
import math
import torch

AQUI = os.path.dirname(os.path.abspath(__file__))

RAIZ_RASTERIZADOR = os.path.abspath(
    os.path.join(AQUI, "..", "..", "..")
)

RUTA_RASTER_CUDA = os.path.join(RAIZ_RASTERIZADOR, "raster_cuda")

print(f"[rasterizador_cuda_tiled_autograd] buscando raster_cuda en: {RUTA_RASTER_CUDA}", flush=True)

if not os.path.isdir(RUTA_RASTER_CUDA):
    raise FileNotFoundError(f"No existe la carpeta raster_cuda: {RUTA_RASTER_CUDA}")

if RUTA_RASTER_CUDA not in sys.path:
    sys.path.insert(0, RUTA_RASTER_CUDA)

try:
    import raster_cuda
except Exception as e:
    print("[rasterizador_cuda_tiled_autograd] sys.path usado:", flush=True)
    for p in sys.path[:8]:
        print("  ", p, flush=True)
    raise e