"""
Visualizaciones clave para la tesis.

1. trayectorias.png          : frame 0 + lineas mu_i(t) superpuestas.
2. heatmap_opacity_temporal  : (N, T) con sigma(opacity_i(t_j)).
3. evolucion_parametros.png  : 5 gaussianas mas "representativas" en detalle.
4. reconstruccion_vs_original.gif : original | reconstruido | diff x5.
"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import imageio.v2 as imageio

from chebyshev import evaluar_polinomio_cheb
from rasterizador import rasterizar_diferenciable



@torch.no_grad()
def _evaluar_parametro_todos_frames(p_a0, p_high, matriz_cheb):
    """Devuelve tensor (..., n_frames) con el polinomio evaluado en cada frame."""
    coefs = torch.cat([p_a0, p_high], dim=-1)        # (..., grado+1)
    # matriz_cheb: (n_frames, grado+1)
    # broadcast: (..., grado+1) @ (grado+1, n_frames) -> (..., n_frames)
    return coefs @ matriz_cheb.T



@torch.no_grad()
def generar_trayectorias_png(modelo, frame_objetivo, matrices_cheb, ruta,
                              umbral_opacity_visualizar=0.1, max_lineas=200):
    """
    Dibuja el frame objetivo (H, W, 3) + lineas mu_i(t) superpuestas. Solo
    para las gaussianas con max_t opacity > umbral. Color de la linea = color
    promedio temporal. Grosor proporcional a opacity media.
    """
    grado_mu = modelo.grados['mu']
    grado_op = modelo.grados['opacity']
    grado_co = modelo.grados['color']
    B_mu = matrices_cheb[grado_mu]
    B_op = matrices_cheb[grado_op]
    B_co = matrices_cheb[grado_co]

    # mu(t) shape (N, 2, T)
    mu_t = _evaluar_parametro_todos_frames(modelo.mu_a0, modelo.mu_high, B_mu)
    op_t = torch.sigmoid(
        _evaluar_parametro_todos_frames(modelo.opacity_a0, modelo.opacity_high, B_op).squeeze(1)
    )                                                     # (N, T)
    color_t = torch.sigmoid(
        _evaluar_parametro_todos_frames(modelo.color_a0, modelo.color_high, B_co)
    )                                                     # (N, 3, T)

    op_max = op_t.max(dim=-1).values                      # (N,)
    op_mean = op_t.mean(dim=-1)
    color_mean = color_t.mean(dim=-1)                     # (N, 3)

    # ordenar por opacity max desc, quedarnos con max_lineas
    mascara = op_max > umbral_opacity_visualizar
    indices = mascara.nonzero(as_tuple=False).squeeze(-1)
    if len(indices) > max_lineas:
        # priorizar las mas opacas en promedio
        orden = torch.argsort(op_mean[indices], descending=True)
        indices = indices[orden[:max_lineas]]

    fig, ax = plt.subplots(figsize=(7, 7), dpi=110)
    img = frame_objetivo.detach().clamp(0, 1).cpu().numpy()
    ax.imshow(img)

    mu_np = mu_t.cpu().numpy()                            # (N, 2, T)  -- (fila, col, t)
    op_mean_np = op_mean.cpu().numpy()
    color_mean_np = color_mean.cpu().numpy()

    for i in indices.cpu().numpy():
        # imshow tiene eje x = columna, y = fila
        xs = mu_np[i, 1, :]
        ys = mu_np[i, 0, :]
        ax.plot(xs, ys,
                color=tuple(color_mean_np[i].clip(0, 1)),
                linewidth=0.3 + 2.5 * op_mean_np[i],
                alpha=0.8)

    ax.set_xlim(0, frame_objetivo.shape[1])
    ax.set_ylim(frame_objetivo.shape[0], 0)
    ax.axis('off')
    ax.set_title(f"Trayectorias mu_i(t)  (N visibles = {len(indices)})")
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)



@torch.no_grad()
def generar_heatmap_opacity(modelo, matrices_cheb, ruta):
    """Heatmap (N, T) con sigma(opacity_i(t)). Filas ordenadas por "tiempo medio de activacion"."""
    grado_op = modelo.grados['opacity']
    B_op = matrices_cheb[grado_op]
    op_t = torch.sigmoid(
        _evaluar_parametro_todos_frames(modelo.opacity_a0, modelo.opacity_high, B_op).squeeze(1)
    )                                                     # (N, T)

    op_np = op_t.cpu().numpy()
    T = op_np.shape[1]
    # tiempo medio de activacion ponderado por opacity
    eje_t = np.arange(T)
    pesos = op_np / (op_np.sum(axis=1, keepdims=True) + 1e-8)
    tiempo_medio = (pesos * eje_t).sum(axis=1)
    orden = np.argsort(tiempo_medio)
    op_ordenado = op_np[orden]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=110)
    im = ax.imshow(op_ordenado, aspect='auto', cmap='inferno', interpolation='nearest')
    ax.set_xlabel("frame_idx")
    ax.set_ylabel("gaussiana (ordenada por t medio de activacion)")
    ax.set_title("sigma(opacity_i(t))")
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)



@torch.no_grad()
def generar_evolucion_parametros(modelo, matrices_cheb, ruta, n_seleccionadas=5):
    """
    Para las n_seleccionadas gaussianas con mayor opacity promedio,
    grafica todos sus parametros vs t.
    """
    grado_op = modelo.grados['opacity']
    B_op = matrices_cheb[grado_op]
    op_t = torch.sigmoid(
        _evaluar_parametro_todos_frames(modelo.opacity_a0, modelo.opacity_high, B_op).squeeze(1)
    )                                                     # (N, T)
    op_mean = op_t.mean(dim=-1)
    top_idx = torch.argsort(op_mean, descending=True)[:n_seleccionadas].cpu().numpy()

    # eval todos los parametros en todos los frames
    todos = {}
    for nombre, (a0, hi, grado, dim_p) in modelo.parametros_temporales().items():
        v = _evaluar_parametro_todos_frames(a0, hi, matrices_cheb[grado])  # (N, dim_p, T)
        if nombre == 'opacity':
            v = torch.sigmoid(v)
        elif nombre == 'color':
            v = torch.sigmoid(v)
        elif nombre == 'scale':
            v = torch.exp(v.clamp(min=np.log(0.5), max=np.log(max(modelo.H, modelo.W))))
        todos[nombre] = v.cpu().numpy()

    n_params = len(todos)
    fig, axes = plt.subplots(n_params, 1, figsize=(10, 2.0 * n_params), dpi=100, sharex=True)
    if n_params == 1:
        axes = [axes]
    for ax, (nombre, v) in zip(axes, todos.items()):
        for i in top_idx:
            arr = v[i]                                    # (dim_p, T)
            for d in range(arr.shape[0]):
                ax.plot(arr[d], label=f"g{i}-d{d}" if v.shape[1] > 1 else f"g{i}",
                        alpha=0.85)
        ax.set_ylabel(nombre)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("frame_idx")
    axes[0].legend(loc='upper right', fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)



@torch.no_grad()
def generar_reconstruccion_gif(modelo, frames, matrices_cheb, ruta_gif,
                                paso=2, factor_diff=5.0, duracion=0.066):
    """
    Renderiza todos los frames con el modelo y arma un gif lado-a-lado:
        original | reconstruido | |orig - recon| * factor_diff
    """
    N_frames, H, W, _ = frames.shape

    cuadros = []
    for j in range(0, N_frames, paso):
        params_j = modelo.evaluar_en_frame(j, matrices_cheb)
        render_j = rasterizar_diferenciable(params_j, H, W).clamp(0, 1).cpu().numpy()
        target = frames[j].cpu().numpy()
        diff = np.clip(np.abs(target - render_j) * factor_diff, 0, 1)
        concat = np.concatenate([target, render_j, diff], axis=1)
        cuadros.append((concat * 255).astype(np.uint8))

    imageio.mimsave(ruta_gif, cuadros, duration=duracion)
