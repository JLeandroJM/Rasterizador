import os
import sys
import time
import argparse

import torch
import raster_cuda


# Ruta al rasterizador original de exp09
RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

EXP09 = os.path.join(
    RAIZ,
    "tesis_2dgs_video",
    "experimentos",
    "exp09_batchfull_comparacion_bases"
)

sys.path.insert(0, EXP09)

from rasterizador import rasterizar_un_frame


def medir_tiempo_cuda(fn, n_warmup=10, n_iters=100):
    """
    Mide tiempo promedio de una funcion que corre en CUDA.
    Necesitamos synchronize porque CUDA es asincrono.
    """
    for _ in range(n_warmup):
        _ = fn()

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for _ in range(n_iters):
        _ = fn()

    torch.cuda.synchronize()
    t1 = time.perf_counter()

    return (t1 - t0) / n_iters


def crear_gaussianas_aleatorias(N, H, W, device, seed=0):
    """
    Crea X=N gaussianas aleatorias compatibles con tu rasterizador.

    Formato:
        mu      : (N, 2)  -> fila, columna
        scale   : (N, 2)
        theta   : (N,)
        opacity : (N,)
        color   : (N, 3)
        depth   : (N,)
    """
    torch.manual_seed(seed)

    mu = torch.empty(N, 2, device=device, dtype=torch.float32)
    mu[:, 0] = torch.rand(N, device=device) * H
    mu[:, 1] = torch.rand(N, device=device) * W

    scale = torch.empty(N, 2, device=device, dtype=torch.float32)
    scale[:, 0] = 2.0 + torch.rand(N, device=device) * 18.0
    scale[:, 1] = 2.0 + torch.rand(N, device=device) * 18.0

    theta = (torch.rand(N, device=device, dtype=torch.float32) - 0.5) * 6.28318530718

    # Opacidad moderada para evitar saturacion total
    opacity = torch.rand(N, device=device, dtype=torch.float32) * 0.8

    color = torch.rand(N, 3, device=device, dtype=torch.float32)

    depth = torch.rand(N, device=device, dtype=torch.float32)

    # Ordenamos por depth una sola vez para que PyTorch y CUDA usen el mismo orden
    idx = torch.argsort(depth)

    mu = mu[idx].contiguous()
    scale = scale[idx].contiguous()
    theta = theta[idx].contiguous()
    opacity = opacity[idx].contiguous()
    color = color[idx].contiguous()
    depth = depth[idx].contiguous()

    return {
        "mu": mu,
        "scale": scale,
        "theta": theta,
        "opacity": opacity,
        "color": color,
        "depth": depth,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=200, help="numero de gaussianas")
    parser.add_argument("--H", type=int, default=128, help="alto de imagen")
    parser.add_argument("--W", type=int, default=128, help="ancho de imagen")
    parser.add_argument("--iters", type=int, default=100, help="iteraciones para benchmark")
    parser.add_argument("--warmup", type=int, default=10, help="iteraciones de calentamiento")
    parser.add_argument("--seed", type=int, default=0, help="semilla aleatoria")
    args = parser.parse_args()

    device = "cuda"

    print("=== Config ===")
    print(f"N gaussianas : {args.N}")
    print(f"resolucion   : {args.H} x {args.W}")
    print(f"iters        : {args.iters}")
    print(f"warmup       : {args.warmup}")
    print(f"seed         : {args.seed}")
    print(f"device       : {torch.cuda.get_device_name(0)}")

    params = crear_gaussianas_aleatorias(
        N=args.N,
        H=args.H,
        W=args.W,
        device=device,
        seed=args.seed
    )

    def render_torch():
        return rasterizar_un_frame(params, args.H, args.W)

    def render_cuda_custom():
        return raster_cuda.forward(
            params["mu"],
            params["scale"],
            params["theta"],
            params["opacity"],
            params["color"],
            args.H,
            args.W
        )

    # Comparacion numerica
    with torch.no_grad():
        out_torch = render_torch().clamp(0, 1)
        out_cuda = render_cuda_custom().clamp(0, 1)

    max_diff = (out_torch - out_cuda).abs().max().item()
    mean_diff = (out_torch - out_cuda).abs().mean().item()

    print("\n=== Comparacion numerica ===")
    print(f"max_diff  : {max_diff:.10f}")
    print(f"mean_diff : {mean_diff:.10f}")

    # Benchmark
    with torch.no_grad():
        t_torch = medir_tiempo_cuda(
            render_torch,
            n_warmup=args.warmup,
            n_iters=args.iters
        )

        t_cuda = medir_tiempo_cuda(
            render_cuda_custom,
            n_warmup=args.warmup,
            n_iters=args.iters
        )

    print("\n=== Tiempo promedio por frame ===")
    print(f"PyTorch rasterizador actual : {t_torch * 1000:.4f} ms")
    print(f"CUDA custom forward         : {t_cuda * 1000:.4f} ms")
    print(f"Speedup                     : {t_torch / t_cuda:.2f}x")


if __name__ == "__main__":
    main()