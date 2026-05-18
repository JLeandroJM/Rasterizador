import torch
import raster_cuda


class RasterConicCUDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mu, conic, opacity, color, H, W):
        mu_c = mu.contiguous()
        conic_c = conic.contiguous()
        opacity_c = opacity.contiguous()
        color_c = color.contiguous()

        out = raster_cuda.forward_conic(
            mu_c,
            conic_c,
            opacity_c,
            color_c,
            H,
            W
        )

        ctx.save_for_backward(mu_c, conic_c, opacity_c, color_c)
        ctx.H = H
        ctx.W = W

        return out

    @staticmethod
    def backward(ctx, grad_out):
        mu, conic, opacity, color = ctx.saved_tensors

        grad_out_c = grad_out.contiguous()

        grad_mu, grad_conic, grad_opacity, grad_color = raster_cuda.backward_conic(
            mu,
            conic,
            opacity,
            color,
            grad_out_c,
            ctx.H,
            ctx.W
        )

        return grad_mu, grad_conic, grad_opacity, grad_color, None, None


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


def render_conic_torch(mu, conic, opacity, color, H, W):
    device = mu.device
    dtype = mu.dtype
    N = mu.shape[0]

    filas = torch.arange(H, device=device, dtype=dtype) + 0.5
    columnas = torch.arange(W, device=device, dtype=dtype) + 0.5
    rr, cc = torch.meshgrid(filas, columnas, indexing="ij")
    pixeles = torch.stack([rr, cc], dim=-1).reshape(-1, 2)

    diff = pixeles.unsqueeze(1) - mu.unsqueeze(0)

    dx = diff[:, :, 0]
    dy = diff[:, :, 1]

    m00 = conic[:, 0].unsqueeze(0)
    m01 = conic[:, 1].unsqueeze(0)
    m11 = conic[:, 2].unsqueeze(0)

    quad = m00 * dx * dx + 2.0 * m01 * dx * dy + m11 * dy * dy
    G = torch.exp(-0.5 * quad)

    alpha = opacity.unsqueeze(0) * G
    alpha = torch.clamp(alpha, max=0.99)

    log_uno_menos_alpha = torch.log(1.0 - alpha + 1e-10)
    log_T_inclusivo = torch.cumsum(log_uno_menos_alpha, dim=1)

    log_T = torch.cat(
        [
            torch.zeros_like(log_T_inclusivo[:, :1]),
            log_T_inclusivo[:, :-1],
        ],
        dim=1
    )

    T = torch.exp(log_T)
    pesos = (alpha * T).unsqueeze(-1)

    out = (pesos * color.unsqueeze(0)).sum(dim=1)
    return out.reshape(H, W, 3)


def comparar(nombre, a, b):
    diff = (a - b).abs()
    print(nombre)
    print("  max_diff :", diff.max().item())
    print("  mean_diff:", diff.mean().item())


def main():
    torch.manual_seed(123)

    device = "cuda"
    H = 16
    W = 16
    N = 6

    # Datos pequenos para evitar saturacion y evitar el break por T bajo.
    mu = torch.empty(N, 2, device=device)
    mu[:, 0] = torch.rand(N, device=device) * H
    mu[:, 1] = torch.rand(N, device=device) * W
    mu.requires_grad_(True)

    scale = torch.empty(N, 2, device=device)
    scale[:, 0] = 2.0 + torch.rand(N, device=device) * 3.0
    scale[:, 1] = 2.0 + torch.rand(N, device=device) * 3.0

    theta = (torch.rand(N, device=device) - 0.5) * 6.28318530718
    conic_base = construir_conic(scale, theta)

    conic = conic_base.detach().clone().requires_grad_(True)

    opacity = (torch.rand(N, device=device) * 0.25).requires_grad_(True)
    color = torch.rand(N, 3, device=device).requires_grad_(True)

    # Clones para referencia PyTorch
    mu_t = mu.detach().clone().requires_grad_(True)
    conic_t = conic.detach().clone().requires_grad_(True)
    opacity_t = opacity.detach().clone().requires_grad_(True)
    color_t = color.detach().clone().requires_grad_(True)

    out_cuda = RasterConicCUDA.apply(mu, conic, opacity, color, H, W)
    out_torch = render_conic_torch(mu_t, conic_t, opacity_t, color_t, H, W)

    comparar("forward CUDA vs torch", out_cuda, out_torch)

    target_grad = torch.randn_like(out_cuda)

    loss_cuda = (out_cuda * target_grad).sum()
    loss_torch = (out_torch * target_grad).sum()

    loss_cuda.backward()
    loss_torch.backward()

    print("\n=== Gradientes ===")
    comparar("grad_mu", mu.grad, mu_t.grad)
    comparar("grad_conic", conic.grad, conic_t.grad)
    comparar("grad_opacity", opacity.grad, opacity_t.grad)
    comparar("grad_color", color.grad, color_t.grad)


if __name__ == "__main__":
    main()