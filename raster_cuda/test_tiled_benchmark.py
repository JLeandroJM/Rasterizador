import os
import sys
import time
import math
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

    return mu, scale, theta, opacity, color, depth


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


def calcular_tiles_tocados(mu, scale, theta, depth, H, W, tile_size=16, k_sigma=3.5):
    device = mu.device
    N = mu.shape[0]

    tiles_x = math.ceil(W / tile_size)
    tiles_y = math.ceil(H / tile_size)
    total_tiles = tiles_x * tiles_y

    pares_tile = []
    pares_depth = []
    pares_gauss = []

    mu_cpu = mu.detach().cpu()
    scale_cpu = scale.detach().cpu()
    theta_cpu = theta.detach().cpu()
    depth_cpu = depth.detach().cpu()

    for i in range(N):
        fila = float(mu_cpu[i, 0])
        col = float(mu_cpu[i, 1])

        sx = float(scale_cpu[i, 0])
        sy = float(scale_cpu[i, 1])
        th = float(theta_cpu[i])

        c = math.cos(th)
        s = math.sin(th)

        extent_fila = k_sigma * math.sqrt((c * sx) ** 2 + (s * sy) ** 2)
        extent_col = k_sigma * math.sqrt((s * sx) ** 2 + (c * sy) ** 2)

        min_f = max(0, int(math.floor(fila - extent_fila)))
        max_f = min(H - 1, int(math.ceil(fila + extent_fila)))
        min_c = max(0, int(math.floor(col - extent_col)))
        max_c = min(W - 1, int(math.ceil(col + extent_col)))

        tile_y_min = min_f // tile_size
        tile_y_max = max_f // tile_size
        tile_x_min = min_c // tile_size
        tile_x_max = max_c // tile_size

        for ty in range(tile_y_min, tile_y_max + 1):
            for tx in range(tile_x_min, tile_x_max + 1):
                tile_id = ty * tiles_x + tx
                pares_tile.append(tile_id)
                pares_depth.append(float(depth_cpu[i]))
                pares_gauss.append(i)

    tile_ids = torch.tensor(pares_tile, device=device, dtype=torch.int64)
    depths = torch.tensor(pares_depth, device=device, dtype=torch.float32)
    gaussian_ids = torch.tensor(pares_gauss, device=device, dtype=torch.int64)

    depth_norm = (depths * 1_000_000).to(torch.int64)
    keys = tile_ids * 1_000_000 + depth_norm
    order = torch.argsort(keys)

    tile_ids = tile_ids[order].contiguous()
    gaussian_ids = gaussian_ids[order].contiguous()

    ranges = torch.full((total_tiles, 2), -1, device=device, dtype=torch.int64)

    tile_cpu = tile_ids.detach().cpu()
    start = 0
    while start < tile_cpu.numel():
        tid = int(tile_cpu[start])
        end = start + 1
        while end < tile_cpu.numel() and int(tile_cpu[end]) == tid:
            end += 1

        ranges[tid, 0] = start
        ranges[tid, 1] = end
        start = end

    return tile_ids, gaussian_ids, ranges, tiles_x, tiles_y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=1000)
    parser.add_argument("--H", type=int, default=256)
    parser.add_argument("--W", type=int, default=256)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--tile", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = "cuda"

    print("=== Config ===")
    print(f"N gaussianas : {args.N}")
    print(f"resolucion   : {args.H} x {args.W}")
    print(f"tile         : {args.tile}")
    print(f"iters        : {args.iters}")
    print(f"device       : {torch.cuda.get_device_name(0)}")

    mu, scale, theta, opacity, color, depth = crear_gaussianas_aleatorias(
        args.N, args.H, args.W, device, args.seed
    )

    # Para PyTorch y brute-force usamos orden global por depth
    idx = torch.argsort(depth)
    mu_o = mu[idx].contiguous()
    scale_o = scale[idx].contiguous()
    theta_o = theta[idx].contiguous()
    opacity_o = opacity[idx].contiguous()
    color_o = color[idx].contiguous()
    depth_o = depth[idx].contiguous()

    conic_o = construir_conic(scale_o, theta_o)

    params = {
        "mu": mu_o,
        "scale": scale_o,
        "theta": theta_o,
        "opacity": opacity_o,
        "color": color_o,
        "depth": depth_o,
    }

    tile_ids, gaussian_ids, ranges, tiles_x, tiles_y = calcular_tiles_tocados(
        mu_o, scale_o, theta_o, depth_o, args.H, args.W, tile_size=args.tile
    )

    longitudes = ranges[:, 1] - ranges[:, 0]
    longitudes = longitudes[longitudes > 0]

    print("\n=== Info tiles ===")
    print(f"tiles_x, tiles_y          : {tiles_x}, {tiles_y}")
    print(f"instancias tile-gaussiana : {gaussian_ids.numel()}")
    print(f"gaussianas promedio/tile  : {longitudes.float().mean().item():.2f}")
    print(f"gaussianas max/tile       : {longitudes.max().item()}")

    def render_torch():
        return rasterizar_un_frame(params, args.H, args.W)

    def render_cuda_conic():
        return raster_cuda.forward_conic(
            mu_o, conic_o, opacity_o, color_o, args.H, args.W
        )

    def render_cuda_tiled():
        return raster_cuda.forward_tiled(
            mu_o,
            conic_o,
            opacity_o,
            color_o,
            gaussian_ids,
            ranges,
            args.H,
            args.W,
            args.tile
        )

    with torch.no_grad():
        out_torch = render_torch().clamp(0, 1)
        out_conic = render_cuda_conic().clamp(0, 1)
        out_tiled = render_cuda_tiled().clamp(0, 1)

    print("\n=== Comparacion numerica ===")
    print("Torch vs CUDA conic")
    print("max_diff :", (out_torch - out_conic).abs().max().item())
    print("mean_diff:", (out_torch - out_conic).abs().mean().item())

    print("\nTorch vs CUDA tiled")
    print("max_diff :", (out_torch - out_tiled).abs().max().item())
    print("mean_diff:", (out_torch - out_tiled).abs().mean().item())

    with torch.no_grad():
        t_torch = medir_tiempo_cuda(render_torch, args.warmup, args.iters)
        t_conic = medir_tiempo_cuda(render_cuda_conic, args.warmup, args.iters)
        t_tiled = medir_tiempo_cuda(render_cuda_tiled, args.warmup, args.iters)

    print("\n=== Tiempo promedio por frame ===")
    print(f"PyTorch rasterizador actual : {t_torch * 1000:.4f} ms")
    print(f"CUDA conic brute-force      : {t_conic * 1000:.4f} ms")
    print(f"CUDA tiled                  : {t_tiled * 1000:.4f} ms")
    print(f"Speedup conic vs PyTorch    : {t_torch / t_conic:.2f}x")
    print(f"Speedup tiled vs PyTorch    : {t_torch / t_tiled:.2f}x")
    print(f"Mejora tiled vs conic       : {t_conic / t_tiled:.2f}x")


if __name__ == "__main__":
    main()