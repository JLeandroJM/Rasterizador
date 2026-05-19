import os
import sys
import torch

AQUI = os.path.dirname(os.path.abspath(__file__))
if AQUI not in sys.path:
    sys.path.insert(0, AQUI)

import raster_cuda


def construir_conic_torch(scale, theta):
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


def grad_conic_torch(scale, theta, grad_conic):
    sx = scale[:, 0]
    sy = scale[:, 1]
    c = torch.cos(theta)
    s = torch.sin(theta)

    den_x = sx * sx + 1e-8
    den_y = sy * sy + 1e-8
    A = 1.0 / den_x
    B = 1.0 / den_y

    gm00 = grad_conic[:, 0]
    gm01 = grad_conic[:, 1]
    gm11 = grad_conic[:, 2]

    grad_A = gm00 * c * c + gm01 * c * s + gm11 * s * s
    grad_B = gm00 * s * s - gm01 * c * s + gm11 * c * c

    dA_dsx = -2.0 * sx / (den_x * den_x)
    dB_dsy = -2.0 * sy / (den_y * den_y)

    grad_sx = grad_A * dA_dsx
    grad_sy = grad_B * dB_dsy
    grad_scale = torch.stack([grad_sx, grad_sy], dim=1)

    dm00_dtheta = 2.0 * c * s * (B - A)
    dm01_dtheta = (c * c - s * s) * (A - B)
    dm11_dtheta = 2.0 * c * s * (A - B)
    grad_theta = gm00 * dm00_dtheta + gm01 * dm01_dtheta + gm11 * dm11_dtheta
    return grad_scale, grad_theta


def bench(fn, iters=200):
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters


def main():
    device = "cuda"
    N = 15000
    iters = 300
    torch.manual_seed(0)

    scale = (torch.rand(N, 2, device=device) * 5.0 + 0.5).contiguous()
    theta = (torch.rand(N, device=device) * 6.2831853).contiguous()
    grad_conic = torch.randn(N, 3, device=device).contiguous()

    conic_ref = construir_conic_torch(scale, theta)
    conic_cuda = raster_cuda.build_conic(scale, theta)
    torch.cuda.synchronize()

    print("=== build_conic ===")
    print("max_diff :", float((conic_ref - conic_cuda).abs().max().item()))
    print("mean_diff:", float((conic_ref - conic_cuda).abs().mean().item()))
    print("torch ms :", bench(lambda: construir_conic_torch(scale, theta), iters))
    print("cuda ms  :", bench(lambda: raster_cuda.build_conic(scale, theta), iters))

    gs_ref, gt_ref = grad_conic_torch(scale, theta, grad_conic)
    gs_cuda, gt_cuda = raster_cuda.grad_conic_to_scale_theta(scale, theta, grad_conic)
    torch.cuda.synchronize()

    print("\n=== grad_conic_to_scale_theta ===")
    print("grad_scale max_diff :", float((gs_ref - gs_cuda).abs().max().item()))
    print("grad_scale mean_diff:", float((gs_ref - gs_cuda).abs().mean().item()))
    print("grad_theta max_diff :", float((gt_ref - gt_cuda).abs().max().item()))
    print("grad_theta mean_diff:", float((gt_ref - gt_cuda).abs().mean().item()))
    print("torch ms :", bench(lambda: grad_conic_torch(scale, theta, grad_conic), iters))
    print("cuda ms  :", bench(lambda: raster_cuda.grad_conic_to_scale_theta(scale, theta, grad_conic), iters))


if __name__ == "__main__":
    main()
