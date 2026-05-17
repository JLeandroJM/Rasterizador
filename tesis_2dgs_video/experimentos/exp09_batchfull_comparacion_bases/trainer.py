"""
Loop de entrenamiento epoch-based con "batch full" via GRADIENT ACCUMULATION.

DECISION (manejo de memoria - importante):
------------------------------------------
"Batch full" en la teoria significa: loss promediado sobre TODOS los frames
en una sola pasada -> 1 step de optimizer por epoch. En la practica con un
clip de 90 frames y N=1000 gaussianas a 256x256, construir el grafo completo
de autograd para todos los frames simultaneamente requiere ~50 GB de VRAM,
imposible en una 4050 con 6 GB.

Implementacion equivalente con memoria de un solo frame:
    optimizer.zero_grad()
    para j en 0..n_frames-1:
        params_j = modelo.evaluar_en_frame(j)
        render_j = rasterizar(params_j)
        loss_j   = loss_render(render_j, frames[j]) / n_frames
        loss_j.backward()        <-- acumula grad en cada Parameter
    loss_s = beta * loss_smoothness(modelo); loss_s.backward()
    optimizer.step()

Esto es matematicamente identico a:
    loss = mean_j L_j + beta * L_smooth
    loss.backward()
    optimizer.step()
porque el gradiente es lineal. Memoria pico = 1 frame.

Si la GPU es grande (12+ GB), se puede usar sub_batch_frames > 1 para mejor
throughput (autograd hace mejor uso de la GPU con batches mas grandes).
Default: sub_batch_frames=1.
"""
import os
import sys
import time

import torch

from rasterizador import rasterizar_un_frame
from perdidas import loss_render_frame, loss_smoothness



def _psnr(a, b):
    mse = torch.mean((a - b) ** 2)
    if mse <= 0:
        return float('inf')
    return float(-10.0 * torch.log10(mse))



@torch.no_grad()
def _evaluar_psnr_promedio(modelo, frames, matrices_base):
    """PSNR promedio sobre todos los frames (no diferenciable)."""
    n_frames, H, W, _ = frames.shape
    suma = 0.0
    for j in range(n_frames):
        params_j = modelo.evaluar_en_frame(j, matrices_base)
        r = rasterizar_un_frame(params_j, H, W).clamp(0, 1)
        suma += _psnr(r, frames[j])
    return suma / n_frames



def entrenar_batch_full(modelo, frames, matrices_base, optimizer, config,
                         carpeta_salida=None):
    """
    Args:
        modelo         : GaussianasPolinomial2D
        frames         : (n_frames, H, W, 3) en [0, 1] ya en device
        matrices_base  : dict {grado: B}
        optimizer      : Adam con param_groups
        config         : dict
        carpeta_salida : str opcional, donde guardar checkpoints y verificaciones

    Returns dict con:
        losses_por_epoch_render, losses_por_epoch_smooth, losses_por_epoch_total
        psnrs_promedio_por_epoch (cada checkpoint_cada_n_epochs)
        tiempo_por_epoch
        tiempo_total
    """
    n_frames, H, W, _ = frames.shape
    n_epochs       = config["n_epochs"]
    beta           = config["beta_smoothness"]
    pesos_smooth   = config.get("pesos_smoothness", None)
    chk_each       = config.get("checkpoint_cada_n_epochs", 50)
    lambda_dssim   = config.get("lambda_dssim", 0.2)
    sub_batch_fr   = config.get("sub_batch_frames", 1)   # 1 = un frame por backward
    early_plateau  = config.get("early_stop_plateau", None)  # opcional

    losses_render = []
    losses_smooth = []
    losses_total  = []
    psnrs_chk     = []
    tiempos       = []

    plateau_count = 0
    mejor_loss = float('inf')

    t_inicio = time.time()
    for epoch in range(n_epochs):
        t_epoch = time.time()

        # === FORWARD/BACKWARD con gradient accumulation =====================
        optimizer.zero_grad()

        loss_render_epoch_total = 0.0
        # acumulamos en sub-batches de frames
        for inicio in range(0, n_frames, sub_batch_fr):
            fin = min(inicio + sub_batch_fr, n_frames)

            loss_sub = None
            for j in range(inicio, fin):
                params_j = modelo.evaluar_en_frame(j, matrices_base)
                render_j = rasterizar_un_frame(params_j, H, W)
                l_j = loss_render_frame(render_j, frames[j], lambda_dssim=lambda_dssim)
                # dividimos por n_frames para que el gradiente acumulado sea
                # exactamente el de la media (equivalente a batch full).
                l_j = l_j / n_frames
                loss_sub = l_j if loss_sub is None else (loss_sub + l_j)

            loss_sub.backward()
            loss_render_epoch_total += float(loss_sub.detach().item()) * n_frames / (fin - inicio)
            # ^ desnormalizamos solo para reportar la loss "como si fuera mean"

        # loss render promediado sobre el epoch
        loss_render_avg = loss_render_epoch_total / (n_frames / max(1, sub_batch_fr))

        # === smoothness ====================================================
        loss_smooth = loss_smoothness(modelo, pesos_por_param=pesos_smooth)
        loss_smooth_escalado = beta * loss_smooth
        loss_smooth_escalado.backward()

        # === step ==========================================================
        optimizer.step()

        # === logging =======================================================
        loss_render_reportado = float(loss_render_avg)
        loss_smooth_reportado = float(loss_smooth.detach().item())
        loss_total_reportado  = loss_render_reportado + beta * loss_smooth_reportado

        losses_render.append(loss_render_reportado)
        losses_smooth.append(loss_smooth_reportado)
        losses_total.append(loss_total_reportado)
        tiempos.append(time.time() - t_epoch)

        # checkpoint y log detallado
        es_checkpoint = (epoch + 1) % chk_each == 0 or epoch == 0 or epoch == n_epochs - 1
        if es_checkpoint:
            psnr_avg = _evaluar_psnr_promedio(modelo, frames, matrices_base)
            psnrs_chk.append((epoch, psnr_avg))

            t_corrido = time.time() - t_inicio
            t_promedio = t_corrido / (epoch + 1)
            eta = t_promedio * (n_epochs - epoch - 1)
            print(f"  epoch {epoch+1:4d}/{n_epochs}  "
                  f"loss_r={loss_render_reportado:.5f}  "
                  f"loss_s={loss_smooth_reportado:.3e}  "
                  f"PSNR_avg={psnr_avg:.2f}  "
                  f"t_epoch={tiempos[-1]:.1f}s  "
                  f"eta={eta/60:.1f}min",
                  flush=True)

            if carpeta_salida is not None:
                # checkpoint pesado
                torch.save({
                    'state_dict_coefs': modelo.state_dict_coefs(),
                    'optimizer_state': optimizer.state_dict(),
                    'epoch_completado': epoch + 1,
                    'config': config,
                }, os.path.join(carpeta_salida, f"checkpoint_epoch{epoch+1:04d}.pt"))

                # verificacion visual: renderizar y guardar primer y ultimo frame
                _guardar_verificacion_visual(modelo, frames, matrices_base,
                                              carpeta_salida, epoch + 1)
        else:
            # log minimo para no spamear
            if (epoch + 1) % max(1, chk_each // 5) == 0:
                t_corrido = time.time() - t_inicio
                t_promedio = t_corrido / (epoch + 1)
                eta = t_promedio * (n_epochs - epoch - 1)
                print(f"  epoch {epoch+1:4d}/{n_epochs}  "
                      f"loss_r={loss_render_reportado:.5f}  "
                      f"loss_s={loss_smooth_reportado:.3e}  "
                      f"t_epoch={tiempos[-1]:.1f}s  "
                      f"eta={eta/60:.1f}min",
                      flush=True)

        # early stop si la loss esta en plateau
        if early_plateau is not None:
            if loss_total_reportado < mejor_loss - 1e-6:
                mejor_loss = loss_total_reportado
                plateau_count = 0
            else:
                plateau_count += 1
            if plateau_count >= early_plateau:
                print(f"  early stop en epoch {epoch+1} (plateau de {early_plateau} epochs)", flush=True)
                break

    return {
        'losses_render': losses_render,
        'losses_smooth': losses_smooth,
        'losses_total':  losses_total,
        'psnrs_chk':     psnrs_chk,
        'tiempos_por_epoch': tiempos,
        'tiempo_total':  time.time() - t_inicio,
    }



@torch.no_grad()
def _guardar_verificacion_visual(modelo, frames, matrices_base, carpeta, epoch):
    """Renderiza primer y ultimo frame y los guarda como PNG."""
    from PIL import Image
    import numpy as np
    n_frames, H, W, _ = frames.shape
    sub = os.path.join(carpeta, "verificacion")
    os.makedirs(sub, exist_ok=True)

    for nombre, j in [("primer", 0), ("ultimo", n_frames - 1)]:
        params_j = modelo.evaluar_en_frame(j, matrices_base)
        r = rasterizar_un_frame(params_j, H, W).clamp(0, 1).cpu().numpy()
        Image.fromarray((r * 255).astype(np.uint8)).save(
            os.path.join(sub, f"epoch{epoch:04d}_{nombre}_frame.png"))
