import torch
import raster_cuda


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


def grad_conic_a_scale_theta(scale, theta, grad_conic):
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

    # Gradientes respecto a A = 1/sx^2 y B = 1/sy^2
    grad_A = gm00 * c * c + gm01 * c * s + gm11 * s * s
    grad_B = gm00 * s * s - gm01 * c * s + gm11 * c * c

    dA_dsx = -2.0 * sx / (den_x * den_x)
    dB_dsy = -2.0 * sy / (den_y * den_y)

    grad_sx = grad_A * dA_dsx
    grad_sy = grad_B * dB_dsy

    grad_scale = torch.stack([grad_sx, grad_sy], dim=1)

    # Derivadas respecto a theta
    dm00_dtheta = 2.0 * c * s * (B - A)
    dm01_dtheta = (c * c - s * s) * (A - B)
    dm11_dtheta = 2.0 * c * s * (A - B)

    grad_theta = (
        gm00 * dm00_dtheta +
        gm01 * dm01_dtheta +
        gm11 * dm11_dtheta
    )

    return grad_scale, grad_theta


class RasterScaleThetaCUDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mu, scale, theta, opacity, color, H, W):
        mu_c = mu.contiguous()
        scale_c = scale.contiguous()
        theta_c = theta.contiguous()
        opacity_c = opacity.contiguous()
        color_c = color.contiguous()

        conic = construir_conic(scale_c, theta_c)

        out = raster_cuda.forward_conic(
            mu_c,
            conic,
            opacity_c,
            color_c,
            H,
            W
        )

        ctx.save_for_backward(mu_c, scale_c, theta_c, conic, opacity_c, color_c)
        ctx.H = H
        ctx.W = W

        return out

    @staticmethod
    def backward(ctx, grad_out):
        mu, scale, theta, conic, opacity, color = ctx.saved_tensors

        grad_mu, grad_conic, grad_opacity, grad_color = raster_cuda.backward_conic(
            mu,
            conic,
            opacity,
            color,
            grad_out.contiguous(),
            ctx.H,
            ctx.W
        )

        grad_scale, grad_theta = grad_conic_a_scale_theta(
            scale,
            theta,
            grad_conic
        )

        return grad_mu, grad_scale, grad_theta, grad_opacity, grad_color, None, None


def render_torch(mu, scale, theta, opacity, color, H, W):
    conic = construir_conic(scale, theta)

    device = mu.device
    dtype = mu.dtype

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

    mu = torch.empty(N, 2, device=device)
    mu[:, 0] = torch.rand(N, device=device) * H
    mu[:, 1] = torch.rand(N, device=device) * W
    mu.requires_grad_(True)

    scale = torch.empty(N, 2, device=device)
    scale[:, 0] = 2.0 + torch.rand(N, device=device) * 3.0
    scale[:, 1] = 2.0 + torch.rand(N, device=device) * 3.0
    scale.requires_grad_(True)

    theta = ((torch.rand(N, device=device) - 0.5) * 6.28318530718).requires_grad_(True)

    opacity = (torch.rand(N, device=device) * 0.25).requires_grad_(True)
    color = torch.rand(N, 3, device=device).requires_grad_(True)

    # Clones para PyTorch
    mu_t = mu.detach().clone().requires_grad_(True)
    scale_t = scale.detach().clone().requires_grad_(True)
    theta_t = theta.detach().clone().requires_grad_(True)
    opacity_t = opacity.detach().clone().requires_grad_(True)
    color_t = color.detach().clone().requires_grad_(True)

    out_cuda = RasterScaleThetaCUDA.apply(
        mu,
        scale,
        theta,
        opacity,
        color,
        H,
        W
    )

    out_torch = render_torch(
        mu_t,
        scale_t,
        theta_t,
        opacity_t,
        color_t,
        H,
        W
    )

    comparar("forward CUDA vs torch", out_cuda, out_torch)

    target_grad = torch.randn_like(out_cuda)

    loss_cuda = (out_cuda * target_grad).sum()
    loss_torch = (out_torch * target_grad).sum()

    loss_cuda.backward()
    loss_torch.backward()

    print("\n=== Gradientes ===")
    comparar("grad_mu", mu.grad, mu_t.grad)
    comparar("grad_scale", scale.grad, scale_t.grad)
    comparar("grad_theta", theta.grad, theta_t.grad)
    comparar("grad_opacity", opacity.grad, opacity_t.grad)
    comparar("grad_color", color.grad, color_t.grad)


if __name__ == "__main__":
    main()