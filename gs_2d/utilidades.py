import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image


# NUEVO: Mac M2 -> usar MPS si esta, sino CPU
def obtener_dispositivo():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# NUEVO: inversa de sigmoid (logit). La uso para "reset opacity": dado un
# valor de opacidad p deseado, calculo el opacity_raw que la produce.
def inverse_sigmoid(p):
    if isinstance(p, float):
        return float(np.log(p / (1.0 - p)))
    return torch.log(p / (1.0 - p))


# NUEVO: cargar imagen target (la usamos como ground truth a reconstruir)
def cargar_imagen_target(ruta, alto, ancho, device):
    img = Image.open(ruta).convert('RGB').resize((ancho, alto))
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).to(device)


# NUEVO: inicializacion aleatoria de N gaussianas. Posiciones uniformes
# dentro de la imagen, escala chica, color random, opacidad ~ 0.5.
def inicializar_gaussianas_aleatorias(N, alto, ancho, device, escala_inicial=4.0, semilla=0):
    g = torch.Generator(device='cpu').manual_seed(semilla)

    mu = torch.empty(N, 2)
    mu[:, 0] = torch.rand(N, generator=g) * alto    # filas
    mu[:, 1] = torch.rand(N, generator=g) * ancho   # columnas

    # exp(scale_raw) ~ escala_inicial, igual en x e y
    scale_raw = torch.full((N, 2), float(np.log(escala_inicial)))

    theta_raw = (torch.rand(N, generator=g) - 0.5) * 2 * np.pi

    # opacity = sigmoid(0) = 0.5
    opacity_raw = torch.zeros(N)

    # color_raw uniforme: sigmoid(uniforme(-1, 1)) entrega colores variados
    color_raw = (torch.rand(N, 3, generator=g) - 0.5) * 2.0

    # depth fijo aleatorio
    depth = torch.rand(N, generator=g)

    return (
        mu.to(device),
        scale_raw.to(device),
        theta_raw.to(device),
        opacity_raw.to(device),
        color_raw.to(device),
        depth.to(device),
    )


# NUEVO: SSIM minimo (kernel gaussiano fijo, una escala). No uso
# pytorch-msssim para no agregar dependencia.
def _kernel_gaussiano(tamano=11, sigma=1.5, device='cpu'):
    coords = torch.arange(tamano, dtype=torch.float32, device=device) - (tamano - 1) / 2.0
    g1 = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g1 = g1 / g1.sum()
    kernel = g1[:, None] * g1[None, :]  # (T, T)
    return kernel


def _ssim_simple(x, y):
    # x, y: (B, C, H, W) en [0, 1]
    C, _, H, W = x.shape[1], x.shape[0], x.shape[2], x.shape[3]
    device = x.device
    kernel = _kernel_gaussiano(11, 1.5, device).to(x.dtype)
    kernel = kernel.expand(C, 1, 11, 11)

    mu_x = F.conv2d(x, kernel, padding=5, groups=C)
    mu_y = F.conv2d(y, kernel, padding=5, groups=C)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, kernel, padding=5, groups=C) - mu_x2
    sigma_y2 = F.conv2d(y * y, kernel, padding=5, groups=C) - mu_y2
    sigma_xy = F.conv2d(x * y, kernel, padding=5, groups=C) - mu_xy

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    return (num / den).mean()


# NUEVO: loss del paper L = (1-λ) L1 + λ DSSIM
def calcular_loss(render, target, lambda_dssim=0.2):
    l1 = torch.mean(torch.abs(render - target))

    # SSIM espera (B, C, H, W)
    r = render.permute(2, 0, 1).unsqueeze(0).clamp(0, 1)
    t = target.permute(2, 0, 1).unsqueeze(0).clamp(0, 1)
    ssim_val = _ssim_simple(r, t)
    dssim = (1.0 - ssim_val) / 2.0

    return (1.0 - lambda_dssim) * l1 + lambda_dssim * dssim


# NUEVO: helpers para guardar figuras
def asegurar_carpeta(ruta):
    os.makedirs(ruta, exist_ok=True)


def guardar_imagen(imagen_tensor, ruta):
    arr = imagen_tensor.detach().clamp(0, 1).cpu().numpy()
    arr_uint8 = (arr * 255).astype(np.uint8)
    Image.fromarray(arr_uint8).save(ruta)


def guardar_curva(valores, titulo, ylabel, ruta, log_y=False):
    fig, ax = plt.subplots(figsize=(7, 4), dpi=100)
    ax.plot(valores)
    ax.set_xlabel('iteracion')
    ax.set_ylabel(ylabel)
    ax.set_title(titulo)
    if log_y:
        ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)


def guardar_comparacion(target, renders, titulos, losses, ruta):
    n = 1 + len(renders)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), dpi=100)
    if n == 1:
        axes = [axes]
    axes[0].imshow(target.detach().clamp(0, 1).cpu().numpy())
    axes[0].set_title('target')
    axes[0].axis('off')
    for i, (r, t, l) in enumerate(zip(renders, titulos, losses)):
        axes[i + 1].imshow(r.detach().clamp(0, 1).cpu().numpy())
        axes[i + 1].set_title(f'{t}\nloss={l:.4f}')
        axes[i + 1].axis('off')
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)


def guardar_overlay_tamanos(imagen, mu, escalas, ruta, titulo=''):
    # NUEVO: para exp3, visualizar donde estan las gaussianas chicas vs grandes
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    ax.imshow(imagen.detach().clamp(0, 1).cpu().numpy())
    mu_np = mu.detach().cpu().numpy()
    esc_max = escalas.detach().max(dim=1)[0].cpu().numpy()
    # mu = (fila, columna) -> scatter (x=col, y=fila)
    sc = ax.scatter(mu_np[:, 1], mu_np[:, 0], c=esc_max, s=8, cmap='plasma',
                    edgecolors='black', linewidths=0.3)
    plt.colorbar(sc, ax=ax, label='max(escala)')
    ax.set_title(titulo)
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)


def crear_gif(frames, ruta, duracion=0.05):
    # NUEVO: GIF del entrenamiento, usa imageio que ya esta en requirements
    try:
        import imageio.v2 as imageio
    except ImportError:
        import imageio
    arrs = []
    for f in frames:
        if torch.is_tensor(f):
            f = f.detach().clamp(0, 1).cpu().numpy()
        arrs.append((f * 255).astype(np.uint8))
    imageio.mimsave(ruta, arrs, duration=duracion)


# NUEVO: helper para crear el optimizer Adam con lrs distintos por param group
def crear_optimizador(modelo, lrs):
    return torch.optim.Adam([
        {'params': [modelo.mu],          'lr': lrs['mu']},
        {'params': [modelo.scale_raw],   'lr': lrs['scale']},
        {'params': [modelo.theta_raw],   'lr': lrs['theta']},
        {'params': [modelo.opacity_raw], 'lr': lrs['opacity']},
        {'params': [modelo.color_raw],   'lr': lrs['color']},
    ])


# NUEVO: lrs por defecto, escalados a imagen pequena (~ pixeles)
LRS_DEFAULT = {
    'mu':      1e-3 * 100,   # mu vive en pixeles, lo escalamos
    'scale':   5e-3,
    'theta':   1e-3,
    'opacity': 5e-2,
    'color':   1e-2,
}


# NUEVO: clamp de scale_raw despues del optimizer.step() para que ninguna
# gaussiana se vuelva mas grande que la imagen. Sin esto, el optimizer
# puede crecer la escala sin limite -> Σ huge -> exp/inv inestable -> NaN.
@torch.no_grad()
def clampear_escala(modelo, alto, ancho):
    log_max = float(np.log(max(alto, ancho)))   # max ~ tamano de la imagen
    log_min = float(np.log(0.5))                # min ~ medio pixel
    modelo.scale_raw.data.clamp_(min=log_min, max=log_max)
