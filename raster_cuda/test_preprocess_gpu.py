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


def calcular_tiles_tocados_gpu(mu, scale, theta, depth, H, W, tile_size=16, k_sigma=3.5):
    """
    Version GPU del preprocess:

    Para cada gaussiana:
        1. Calcula AABB en pantalla
        2. Calcula tiles min/max
        3. Duplica la gaussiana por cada tile tocado
        4. Crea key = tile_id + depth
        5. Ordena
        6. Construye ranges[tile_id] = [inicio, fin)

    Todo usa tensores CUDA.
    """
    device = mu.device
    N = mu.shape[0]

    tiles_x = math.ceil(W / tile_size)
    tiles_y = math.ceil(H / tile_size)
    total_tiles = tiles_x * tiles_y

    fila = mu[:, 0]
    col = mu[:, 1]

    sx = scale[:, 0]
    sy = scale[:, 1]

    c = torch.cos(theta)
    s = torch.sin(theta)

    # AABB de elipse rotada
    extent_fila = k_sigma * torch.sqrt((c * sx) ** 2 + (s * sy) ** 2)
    extent_col = k_sigma * torch.sqrt((s * sx) ** 2 + (c * sy) ** 2)

    min_f = torch.floor(fila - extent_fila).clamp(0, H - 1).to(torch.int64)
    max_f = torch.ceil(fila + extent_fila).clamp(0, H - 1).to(torch.int64)
    min_c = torch.floor(col - extent_col).clamp(0, W - 1).to(torch.int64)
    max_c = torch.ceil(col + extent_col).clamp(0, W - 1).to(torch.int64)

    tile_y_min = torch.div(min_f, tile_size, rounding_mode="floor")
    tile_y_max = torch.div(max_f, tile_size, rounding_mode="floor")
    tile_x_min = torch.div(min_c, tile_size, rounding_mode="floor")
    tile_x_max = torch.div(max_c, tile_size, rounding_mode="floor")

    num_y = tile_y_max - tile_y_min + 1
    num_x = tile_x_max - tile_x_min + 1
    counts = num_y * num_x

    total_instancias = int(counts.sum().item())

    # gaussian_ids repetidos segun cuantos tiles toca cada gaussiana
    gaussian_ids = torch.repeat_interleave(
        torch.arange(N, device=device, dtype=torch.int64),
        counts
    )

    # Para cada instancia, calcular su offset local dentro de los tiles de su gaussiana
    starts = torch.cumsum(counts, dim=0) - counts
    local_offsets = torch.arange(total_instancias, device=device, dtype=torch.int64) - starts[gaussian_ids]

    nx = num_x[gaussian_ids]

    local_y = torch.div(local_offsets, nx, rounding_mode="floor")
    local_x = local_offsets - local_y * nx

    tx = tile_x_min[gaussian_ids] + local_x
    ty = tile_y_min[gaussian_ids] + local_y

    tile_ids = ty * tiles_x + tx

    # Key simple:
    # tile en la parte alta, depth cuantizado en la parte baja.
    # En el paper esto se hace con key 64-bit y radix sort.
    depth_vals = depth[gaussian_ids]
    depth_norm = (depth_vals * 1_000_000).to(torch.int64)
    keys = tile_ids * 1_000_000 + depth_norm

    order = torch.argsort(keys)

    tile_ids = tile_ids[order].contiguous()
    gaussian_ids = gaussian_ids[order].contiguous()

    ranges = torch.full((total_tiles, 2), -1, device=device, dtype=torch.int64)

    if tile_ids.numel() > 0:
        # Detectar inicios de grupos por tile en GPU
        is_start = torch.empty_like(tile_ids, dtype=torch.bool)
        is_start[0] = True
        is_start[1:] = tile_ids[1:] != tile_ids[:-1]

        starts_idx = torch.nonzero(is_start, as_tuple=False).flatten()
        unique_tiles = tile_ids[starts_idx]

        ends_idx = torch.empty_like(starts_idx)
        if starts_idx.numel() > 1:
            ends_idx[:-1] = starts_idx[1:]
        ends_idx[-1] = tile_ids.numel()

        ranges[unique_tiles, 0] = starts_idx
        ranges[unique_tiles, 1] = ends_idx

    return gaussian_ids.contiguous(), ranges.contiguous(), tiles_x, tiles_y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=1000)
    parser.add_argument("--H", type=int, default=256)
    parser.add_argument("--W", type=int, default=256)
    parser.add_argument("--tile", type=int, default=16)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k_sigma", type=float, default=3.5)
    args = parser.parse_args()

    device = "cuda"

    print("=== Config ===")
    print(f"N gaussianas : {args.N}")
    print(f"resolucion   : {args.H} x {args.W}")
    print(f"tile         : {args.tile}")
    print(f"k_sigma      : {args.k_sigma}")
    print(f"iters        : {args.iters}")
    print(f"device       : {torch.cuda.get_device_name(0)}")

    params = crear_gaussianas_aleatorias(args.N, args.H, args.W, device, args.seed)
    conic = construir_conic(params["scale"], params["theta"])

    def render_torch():
        return rasterizar_un_frame(params, args.H, args.W)

    def render_conic():
        return raster_cuda.forward_conic(
            params["mu"],
            conic,
            params["opacity"],
            params["color"],
            args.H,
            args.W
        )

    def preprocess_gpu():
        return calcular_tiles_tocados_gpu(
            params["mu"],
            params["scale"],
            params["theta"],
            params["depth"],
            args.H,
            args.W,
            tile_size=args.tile,
            k_sigma=args.k_sigma
        )

    # Preprocess una vez para revisar info
    gaussian_ids, ranges, tiles_x, tiles_y = preprocess_gpu()

    longitudes = ranges[:, 1] - ranges[:, 0]
    longitudes = longitudes[longitudes > 0]

    print("\n=== Info tiles GPU ===")
    print(f"tiles_x, tiles_y          : {tiles_x}, {tiles_y}")
    print(f"instancias tile-gaussiana : {gaussian_ids.numel()}")
    print(f"gaussianas promedio/tile  : {longitudes.float().mean().item():.2f}")
    print(f"gaussianas max/tile       : {longitudes.max().item()}")

    def render_tiled_precalculado():
        return raster_cuda.forward_tiled(
            params["mu"],
            conic,
            params["opacity"],
            params["color"],
            gaussian_ids,
            ranges,
            args.H,
            args.W,
            args.tile
        )

    def preprocess_gpu_y_render():
        gids, rng, _, _ = preprocess_gpu()
        return raster_cuda.forward_tiled(
            params["mu"],
            conic,
            params["opacity"],
            params["color"],
            gids,
            rng,
            args.H,
            args.W,
            args.tile
        )

    with torch.no_grad():
        try:
            out_torch = render_torch().clamp(0, 1)
            torch_ok = True
        except torch.OutOfMemoryError as e:
            print("\n[WARN] PyTorch rasterizador original hizo OOM.")
            print("       Se saltara comparacion contra PyTorch.")
            torch.cuda.empty_cache()
            out_torch = None
            torch_ok = False

        out_conic = render_conic().clamp(0, 1)
        out_tiled = render_tiled_precalculado().clamp(0, 1)

    print("\n=== Comparacion numerica ===")

    if torch_ok:
        print("Torch vs CUDA conic")
        print("max_diff :", (out_torch - out_conic).abs().max().item())
        print("mean_diff:", (out_torch - out_conic).abs().mean().item())

        print("\nTorch vs CUDA tiled GPU-preprocess")
        print("max_diff :", (out_torch - out_tiled).abs().max().item())
        print("mean_diff:", (out_torch - out_tiled).abs().mean().item())
    else:
        print("CUDA conic vs CUDA tiled GPU-preprocess")
        print("max_diff :", (out_conic - out_tiled).abs().max().item())
        print("mean_diff:", (out_conic - out_tiled).abs().mean().item())

    with torch.no_grad():
        if torch_ok:
            t_torch = medir_tiempo_cuda(render_torch, args.warmup, args.iters)
        else:
            t_torch = None        
        t_conic = medir_tiempo_cuda(render_conic, args.warmup, args.iters)
        t_preprocess = medir_tiempo_cuda(preprocess_gpu, args.warmup, args.iters)
        t_tiled_only = medir_tiempo_cuda(render_tiled_precalculado, args.warmup, args.iters)
        t_total = medir_tiempo_cuda(preprocess_gpu_y_render, args.warmup, args.iters)

    print("\n=== Tiempo promedio ===")
    if t_torch is not None:
        print(f"PyTorch rasterizador actual : {t_torch * 1000:.4f} ms")
    else:
        print("PyTorch rasterizador actual : OOM")
    print(f"CUDA conic brute-force      : {t_conic * 1000:.4f} ms")
    print(f"Preprocess GPU             : {t_preprocess * 1000:.4f} ms")
    print(f"CUDA tiled only             : {t_tiled_only * 1000:.4f} ms")
    print(f"GPU preprocess + tiled      : {t_total * 1000:.4f} ms")

    print("\n=== Speedups ===")
    if t_torch is not None:
        print(f"Speedup conic vs PyTorch    : {t_torch / t_conic:.2f}x")
        print(f"Speedup tiled only vs Torch : {t_torch / t_tiled_only:.2f}x")
        print(f"Speedup total vs PyTorch    : {t_torch / t_total:.2f}x")
    else:
        print("Speedup vs PyTorch          : no disponible porque PyTorch hizo OOM")

    print(f"Mejora total vs conic       : {t_conic / t_total:.2f}x")
    print(f"Mejora total vs conic       : {t_conic / t_total:.2f}x")


if __name__ == "__main__":
    main()