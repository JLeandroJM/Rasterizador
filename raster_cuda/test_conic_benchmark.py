import os
import sys
import time
import argparse

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
    torch.manual_seed(seed)

    mu = torch.empty(N, 2, device=device, dtype=torch.float32)
    mu[:, 0] = torch.rand(N, device=device) * H
    mu[:, 1] = torch.rand(N, device=device) * W

    scale = torch.empty(N, 2, device=device, dtype=torch.float32)
    scale[:, 0] = 2.0 + torch.rand(N, device=device) * 18.0
    scale[:, 1] = 2.0 + torch.rand(N, device=device) * 18.0

    theta = (torch.rand(N, device=device, dtype=torch.float32) - 0.5) * 6.28318530718
    opacity = torch.rand(N, device=device, dtype=torch.float32) * 0.8
    color = torch.rand(N, 3, device=device, dtype=torch.float32)
    depth = torch.rand(N, device=device, dtype=torch.float32)

    idx = torch.argsort(depth)

    return {
        "mu": mu[idx].contiguous(),
        "scale": scale[idx].contiguous(),
        "theta": theta[idx].contiguous(),
        "opacity": opacity[idx].contiguous(),
        "color": color[idx].contiguous(),
        "depth": depth[idx].contiguous(),
    }


def construir_conic(scale, theta):
    sx = scale[:, 0]
    sy = scale[:, 1]

    c = torch.cos(theta)
    s = torch.sin(theta)

    inv_sx2 = 1.0 / (sx * sx + 1e-8)
    inv_sy2 = 1.0 / (sy * sy + 1e-8)

    m00 = c * c * inv_sx2 + s * s * inv_sy2
    m01 = c * s * (inv_sx2 - inv_sy2)
    m11 = s * s * inv_sx2 + c * c * inv_sy2

    return torch.stack([m00, m01, m11], dim=1).contiguous()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=1000)
    parser.add_argument("--H", type=int, default=256)
    parser.add_argument("--W", type=int, default=256)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = "cuda"

    print("=== Config ===")
    print(f"N gaussianas : {args.N}")
    print(f"resolucion   : {args.H} x {args.W}")
    print(f"iters        : {args.iters}")
    print(f"device       : {torch.cuda.get_device_name(0)}")

    params = crear_gaussianas_aleatorias(args.N, args.H, args.W, device, args.seed)
    conic = construir_conic(params["scale"], params["theta"])

    def render_torch():
        return rasterizar_un_frame(params, args.H, args.W)

    def render_cuda_original():
        return raster_cuda.forward(
            params["mu"],
            params["scale"],
            params["theta"],
            params["opacity"],
            params["color"],
            args.H,
            args.W
        )

    def render_cuda_conic():
        return raster_cuda.forward_conic(
            params["mu"],
            conic,
            params["opacity"],
            params["color"],
            args.H,
            args.W
        )

    with torch.no_grad():
        out_torch = render_torch().clamp(0, 1)
        out_cuda_original = render_cuda_original().clamp(0, 1)
        out_cuda_conic = render_cuda_conic().clamp(0, 1)

    print("\n=== Comparacion numerica ===")
    print("Torch vs CUDA original")
    print("max_diff :", (out_torch - out_cuda_original).abs().max().item())
    print("mean_diff:", (out_torch - out_cuda_original).abs().mean().item())

    print("\nTorch vs CUDA conic")
    print("max_diff :", (out_torch - out_cuda_conic).abs().max().item())
    print("mean_diff:", (out_torch - out_cuda_conic).abs().mean().item())

    with torch.no_grad():
        t_torch = medir_tiempo_cuda(render_torch, args.warmup, args.iters)
        t_cuda_original = medir_tiempo_cuda(render_cuda_original, args.warmup, args.iters)
        t_cuda_conic = medir_tiempo_cuda(render_cuda_conic, args.warmup, args.iters)

    print("\n=== Tiempo promedio por frame ===")
    print(f"PyTorch rasterizador actual : {t_torch * 1000:.4f} ms")
    print(f"CUDA custom original        : {t_cuda_original * 1000:.4f} ms")
    print(f"CUDA custom conic           : {t_cuda_conic * 1000:.4f} ms")
    print(f"Speedup original vs PyTorch : {t_torch / t_cuda_original:.2f}x")
    print(f"Speedup conic vs PyTorch    : {t_torch / t_cuda_conic:.2f}x")
    print(f"Mejora conic vs original    : {t_cuda_original / t_cuda_conic:.2f}x")


if __name__ == "__main__":
    main()