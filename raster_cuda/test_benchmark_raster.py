import os
import sys
import time

import torch

import raster_cuda

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
    # Warmup: las primeras llamadas suelen ser mas lentas
    for _ in range(n_warmup):
        _ = fn()

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for _ in range(n_iters):
        _ = fn()

    torch.cuda.synchronize()
    t1 = time.perf_counter()

    return (t1 - t0) / n_iters


def main():
    device = "cuda"
    torch.manual_seed(0)

    H = 128
    W = 128
    N = 200

    mu = torch.empty(N, 2, device=device, dtype=torch.float32)
    mu[:, 0] = torch.rand(N, device=device) * H
    mu[:, 1] = torch.rand(N, device=device) * W

    scale = torch.empty(N, 2, device=device, dtype=torch.float32)
    scale[:, 0] = 3.0 + torch.rand(N, device=device) * 12.0
    scale[:, 1] = 3.0 + torch.rand(N, device=device) * 12.0

    theta = (torch.rand(N, device=device, dtype=torch.float32) - 0.5) * 6.28
    opacity = torch.rand(N, device=device, dtype=torch.float32) * 0.8
    color = torch.rand(N, 3, device=device, dtype=torch.float32)

    # Orden por depth para que ambos usen el mismo orden
    depth = torch.rand(N, device=device, dtype=torch.float32)
    indices = torch.argsort(depth)

    mu_o = mu[indices].contiguous()
    scale_o = scale[indices].contiguous()
    theta_o = theta[indices].contiguous()
    opacity_o = opacity[indices].contiguous()
    color_o = color[indices].contiguous()
    depth_o = depth[indices].contiguous()

    params = {
        "mu": mu_o,
        "scale": scale_o,
        "theta": theta_o,
        "opacity": opacity_o,
        "color": color_o,
        "depth": depth_o,
    }

    def render_torch():
        return rasterizar_un_frame(params, H, W)

    def render_custom_cuda():
        return raster_cuda.forward(
            mu_o,
            scale_o,
            theta_o,
            opacity_o,
            color_o,
            H,
            W,
        )

    # Calcular una vez para comparar
    with torch.no_grad():
        out_torch = render_torch().clamp(0, 1)
        out_cuda = render_custom_cuda().clamp(0, 1)

    max_diff = (out_torch - out_cuda).abs().max().item()
    mean_diff = (out_torch - out_cuda).abs().mean().item()

    print("=== Comparacion numerica ===")
    print("max_diff :", max_diff)
    print("mean_diff:", mean_diff)

    # Benchmark
    with torch.no_grad():
        t_torch = medir_tiempo_cuda(render_torch, n_warmup=10, n_iters=50)
        t_cuda = medir_tiempo_cuda(render_custom_cuda, n_warmup=10, n_iters=50)

    print("\n=== Tiempo promedio por frame ===")
    print(f"PyTorch rasterizador actual : {t_torch * 1000:.3f} ms")
    print(f"CUDA custom forward         : {t_cuda * 1000:.3f} ms")
    print(f"Speedup                     : {t_torch / t_cuda:.2f}x")


if __name__ == "__main__":
    main()