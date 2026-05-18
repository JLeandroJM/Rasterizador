import torch
import math


def crear_gaussianas_aleatorias(N, H, W, device, seed=0):
    torch.manual_seed(seed)

    mu = torch.empty(N, 2, device=device, dtype=torch.float32)
    mu[:, 0] = torch.rand(N, device=device) * H
    mu[:, 1] = torch.rand(N, device=device) * W

    scale = torch.empty(N, 2, device=device, dtype=torch.float32)
    scale[:, 0] = 2.0 + torch.rand(N, device=device) * 18.0
    scale[:, 1] = 2.0 + torch.rand(N, device=device) * 18.0

    theta = (torch.rand(N, device=device) - 0.5) * 6.28318530718
    opacity = torch.rand(N, device=device) * 0.8
    color = torch.rand(N, 3, device=device)
    depth = torch.rand(N, device=device)

    return mu, scale, theta, opacity, color, depth


def calcular_tiles_tocados(mu, scale, theta, depth, H, W, tile_size=16, k_sigma=3.0):
    """
    Adaptacion 2D del paso:
    DuplicateWithKeys(M', T)

    Para cada gaussiana:
        - calcula bbox aproximado de 99%
        - calcula tiles tocados
        - crea pares (tile_id, depth, gaussian_id)
    """
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

        # AABB de una elipse rotada.
        # k_sigma=3 aprox intervalo de confianza usado para decidir soporte.
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

    # Orden fiel al paper en idea:
    # primero tile, luego profundidad.
    # Para hacerlo simple en PyTorch:
    # key = tile_id * constante + depth_normalizado
    # En CUDA final se haria con radix sort y key 64-bit.
    depth_norm = (depths * 1_000_000).to(torch.int64)
    keys = tile_ids * 1_000_000 + depth_norm

    order = torch.argsort(keys)

    tile_ids = tile_ids[order].contiguous()
    gaussian_ids = gaussian_ids[order].contiguous()
    depths = depths[order].contiguous()

    # Rangos por tile: [inicio, fin)
    ranges = torch.full((total_tiles, 2), -1, device=device, dtype=torch.int64)

    if tile_ids.numel() > 0:
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

    return tile_ids, gaussian_ids, depths, ranges, tiles_x, tiles_y


def main():
    device = "cuda"
    N = 1000
    H = 256
    W = 256
    tile_size = 16

    mu, scale, theta, opacity, color, depth = crear_gaussianas_aleatorias(
        N=N,
        H=H,
        W=W,
        device=device,
        seed=0
    )

    tile_ids, gaussian_ids, depths, ranges, tiles_x, tiles_y = calcular_tiles_tocados(
        mu, scale, theta, depth, H, W, tile_size=tile_size
    )

    total_tiles = tiles_x * tiles_y
    total_instancias = gaussian_ids.numel()
    promedio_por_gauss = total_instancias / N

    tiles_usados = (ranges[:, 0] >= 0).sum().item()

    longitudes = ranges[:, 1] - ranges[:, 0]
    longitudes = longitudes[longitudes > 0]

    print("=== Preprocess tiles ===")
    print(f"N gaussianas              : {N}")
    print(f"resolucion                : {H} x {W}")
    print(f"tile_size                 : {tile_size}")
    print(f"tiles_x, tiles_y          : {tiles_x}, {tiles_y}")
    print(f"total tiles               : {total_tiles}")
    print(f"tiles usados              : {tiles_usados}")
    print(f"instancias tile-gaussiana : {total_instancias}")
    print(f"promedio tiles/gaussiana  : {promedio_por_gauss:.2f}")
    print(f"gaussianas promedio/tile  : {longitudes.float().mean().item():.2f}")
    print(f"gaussianas max/tile       : {longitudes.max().item()}")

    print("\nPrimeros 10 pares ordenados:")
    for i in range(min(10, total_instancias)):
        print(
            "tile=",
            tile_ids[i].item(),
            "gauss=",
            gaussian_ids[i].item(),
            "depth=",
            depths[i].item()
        )


if __name__ == "__main__":
    main()