import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image



def inicializar_gaussianas_aleatorias(N, alto, ancho, device, escala_inicial=4.0, gaussianas_iniciales=0):
    
    g = torch.Generator(device='cpu').manual_seed(gaussianas_iniciales)

    centro = torch.empty(N, 2)
    centro[:, 0] = torch.rand(N, generator=g) * alto
    centro[:, 1] = torch.rand(N, generator=g) * ancho

    escala = torch.full((N, 2), float(np.log(escala_inicial)))

    theta = (torch.rand(N, generator=g) - 0.5) * 2 * np.pi

    opacidad = torch.zeros(N)

    color = (torch.rand(N, 3, generator=g) - 0.5) * 2.0
    profundidad = torch.rand(N, generator=g)

    return (
        centro.to(device),
        escala.to(device),
        theta.to(device),
        opacidad.to(device),
        color.to(device),
        profundidad.to(device),
    )



def actualizar_opacidad(p):
    if isinstance(p, float):
        return float(np.log(p / (1.0 - p)))
    return torch.log(p / (1.0 - p))



def cargar_imagen_objetivo(ruta, alto, ancho, device):
    img = Image.open(ruta).convert('RGB').resize((ancho, alto))
    arr = np.array(img, dtype=np.float32) / 255.0

    return torch.from_numpy(arr).to(device)



# NUEVO: helper para crear carpetas de salida
def asegurar_carpeta(ruta):
    os.makedirs(ruta, exist_ok=True)



def kernel_para_ssim(tamano=11, sigma=1.5, device='cpu'):
    # NUEVO: agregue parametro device para que el kernel se construya en el
    # mismo device del tensor de entrada (sino MPS choca con CPU).
    coords = torch.arange(tamano, dtype=torch.float32, device=device) - (tamano - 1) / 2.0
    g1 = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g1 = g1 / g1.sum()
    kernel = g1[:, None] * g1[None, :]
    return kernel


#IMPORTANBTEEE
def calcular_ssim(x, y):

    C = x.shape[1]
    device = x.device
    kernel = kernel_para_ssim(11, 1.5, device).to(x.dtype)
    kernel = kernel.expand(C, 1, 11, 11)

    centro_x = F.conv2d(x, kernel, padding=5, groups=C)
    centro_y = F.conv2d(y, kernel, padding=5, groups=C)

    centro_x2 = centro_x * centro_x
    centro_y2 = centro_y * centro_y
    centro_xy = centro_x * centro_y

    sigma_x2 = F.conv2d(x * x, kernel, padding=5, groups=C) - centro_x2
    sigma_y2 = F.conv2d(y * y, kernel, padding=5, groups=C) - centro_y2
    sigma_xy = F.conv2d(x * y, kernel, padding=5, groups=C) - centro_xy

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    numerador = (2 * centro_xy + C1) * (2 * sigma_xy + C2)
    den = (centro_x2 + centro_y2 + C1) * (sigma_x2 + sigma_y2 + C2)

    return (numerador / den).mean()


# L = (1-loss) L1 + loss DSSIM
def calcular_loss(render, objetivo, lambda_dssim=0.2):
    l1 = torch.mean(torch.abs(render - objetivo))

    # NUEVO: era 'percentrote' (typo), corregido a 'permute'
    r = render.permute(2, 0, 1).unsqueeze(0).clamp(0, 1)
    t = objetivo.permute(2, 0, 1).unsqueeze(0).clamp(0, 1)
    ssim_val = calcular_ssim(r, t)

    dssim = (1.0 - ssim_val) / 2.0

    return (1.0 - lambda_dssim) * l1 + lambda_dssim * dssim




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


def guardar_comparacion(objetivo, renders, titulos, losses, ruta):
    n = 1 + len(renders)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), dpi=100)
    if n == 1:
        axes = [axes]
    axes[0].imshow(objetivo.detach().clamp(0, 1).cpu().numpy())
    axes[0].set_title('objetivo')
    axes[0].axis('off')
    for i, (r, t, l) in enumerate(zip(renders, titulos, losses)):
        axes[i + 1].imshow(r.detach().clamp(0, 1).cpu().numpy())
        axes[i + 1].set_title(f'{t}\nloss={l:.4f}')
        axes[i + 1].axis('off')
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)


# NUEVO: en vez de un GIF, guardamos cada frame como PNG en una carpeta
# para poder revisarlos imagen por imagen y ver como se construye la escena.
def guardar_frames(frames, carpeta):
    asegurar_carpeta(carpeta)
    for i, f in enumerate(frames):
        ruta = os.path.join(carpeta, f"frame_{i:04d}.png")
        guardar_imagen(f, ruta)




# modelo (centro, escala, theta, opacidad, color).
LRS_DEFAULT = {
    'centro':   1e-3 * 100,
    'escala':   5e-3,
    'theta':    1e-3,
    'opacidad': 5e-2,
    'color':    1e-2,
}



def crear_optimizador(modelo, lrs):
    return torch.optim.Adam([
        {'params': [modelo.centro],   'lr': lrs['centro']},
        {'params': [modelo.escala],   'lr': lrs['escala']},
        {'params': [modelo.theta],    'lr': lrs['theta']},
        {'params': [modelo.opacidad], 'lr': lrs['opacidad']},
        {'params': [modelo.color],    'lr': lrs['color']},
    ])







@torch.no_grad()
def clampear_escala(modelo, alto, ancho):
    log_max = float(np.log(max(alto, ancho)))  
    log_min = float(np.log(0.5))           
    modelo.escala.data.clamp_(min=log_min, max=log_max)
