"""
Loop de entrenamiento incremental con replay.

Para cada nuevo frame k que "abrimos", optimizamos M_iter pasos. En cada
paso se replayean varios frames (modo "completo" -> 0..k, modo "parcial" ->
muestreo aleatorio + el k):
    loss_iter = sum_j loss_render(render_j, frame_j) + beta * loss_smoothness

Test de no-olvido:
------------------
Cada `intervalo_log` frames imprimimos PSNR(frame=0). Si decae fuerte
conforme avanzamos en k, hay olvido catastrofico y el replay no esta
haciendo lo que debe.
"""
import os
import time
import torch

from rasterizador import rasterizar_diferenciable
from perdidas import loss_render, loss_smoothness



def _psnr(a, b):
    mse = torch.mean((a - b) ** 2)
    if mse <= 0:
        return float('inf')
    return float(-10.0 * torch.log10(mse))



def _seleccionar_frames_a_replayear(k, modo, K_replay, generador):
    """
    Devuelve lista de indices de frames a renderizar en esta iteracion.

    modo 'completo': [0, 1, ..., k]
    modo 'parcial' : [k] + muestreo aleatorio sin reemplazo de K_replay frames
                     de {0..k-1}. Si k < K_replay tomamos todos.
    """
    if modo == 'completo':
        return list(range(k + 1))
    if modo == 'parcial':
        if k == 0:
            return [0]
        # muestreamos K_replay de {0..k-1}, sin incluir k
        K_eff = min(K_replay, k)
        # torch.randperm reproducible si pasamos generator (pero solo en CPU)
        idx = torch.randperm(k, generator=generador)[:K_eff].tolist()
        return idx + [k]
    raise ValueError(f"modo replay desconocido: {modo}")



def entrenar_incremental(modelo, frames, matrices_cheb, optimizer, config,
                         carpeta_salida=None):
    """
    Args:
        modelo         : GaussianasChebyshev2D
        frames         : tensor (n_frames, H, W, 3) en [0, 1] ya en device
        matrices_cheb  : dict {grado: B}
        optimizer      : Adam con param groups
        config         : dict con M_iter_por_frame, modo_replay, K_replay,
                         beta_smoothness, pesos_smoothness,
                         checkpoint_interval, intervalo_log

    Returns:
        dict con 'losses_por_frame' (list de listas), 'psnr_frame0_por_k',
                  'tiempo_por_frame', 'tiempo_total'.
    """
    N_frames, H, W, _ = frames.shape
    M_iter            = config["M_iter_por_frame"]
    modo_replay       = config["modo_replay"]
    K_replay          = config.get("K_replay", 10)
    beta              = config["beta_smoothness"]
    pesos_smooth      = config.get("pesos_smoothness", None)
    intervalo_log     = config.get("intervalo_log", max(1, N_frames // 10))
    chk_interval      = config.get("checkpoint_interval", 10)

    gen_replay = torch.Generator()
    gen_replay.manual_seed(0)

    losses_por_frame = []
    psnr_frame0_por_k = []
    tiempo_por_frame = []

    t_inicio = time.time()
    for k in range(N_frames):
        t_k = time.time()

        losses_iter = []
        for it in range(M_iter):
            optimizer.zero_grad()
            indices_replay = _seleccionar_frames_a_replayear(k, modo_replay, K_replay, gen_replay)

            loss_acumulada = None
            for j in indices_replay:
                params_j = modelo.evaluar_en_frame(j, matrices_cheb)
                render_j = rasterizar_diferenciable(params_j, H, W)
                l_j = loss_render(render_j, frames[j])
                loss_acumulada = l_j if loss_acumulada is None else (loss_acumulada + l_j)

            # promediamos por #frames replayados para que beta tenga sentido
            # constante a lo largo del entrenamiento
            loss_render_avg = loss_acumulada / len(indices_replay)
            loss_smooth = loss_smoothness(modelo, pesos_por_param=pesos_smooth)
            loss_total = loss_render_avg + beta * loss_smooth

            loss_total.backward()
            optimizer.step()
            losses_iter.append(float(loss_total.item()))

        tiempo_por_frame.append(time.time() - t_k)
        losses_por_frame.append(losses_iter)

        # === logging y test de no olvido ====================================
        with torch.no_grad():
            # PSNR en frame 0 (test de no-olvido) y en el frame recien abierto
            p0 = modelo.evaluar_en_frame(0, matrices_cheb)
            r0 = rasterizar_diferenciable(p0, H, W).clamp(0, 1)
            psnr_0 = _psnr(r0, frames[0])
            psnr_frame0_por_k.append(psnr_0)

            pk = modelo.evaluar_en_frame(k, matrices_cheb)
            rk = rasterizar_diferenciable(pk, H, W).clamp(0, 1)
            psnr_k = _psnr(rk, frames[k])

        if (k + 1) % intervalo_log == 0 or k == 0 or k == N_frames - 1:
            t_total = time.time() - t_inicio
            t_promedio_frame = t_total / (k + 1)
            t_restante_estimado = t_promedio_frame * (N_frames - k - 1)
            print(f"  frame {k+1:3d}/{N_frames}  "
                  f"loss_ult={losses_iter[-1]:.4f}  "
                  f"PSNR(f=0)={psnr_0:.2f}  PSNR(f=k)={psnr_k:.2f}  "
                  f"t_frame={tiempo_por_frame[-1]:.1f}s  "
                  f"eta={t_restante_estimado/60:.1f}min")

        # === checkpoint ====================================================
        if carpeta_salida and chk_interval > 0 and (k + 1) % chk_interval == 0:
            ruta_chk = os.path.join(carpeta_salida, f"checkpoint_k{k+1:04d}.pt")
            torch.save({
                'state_dict_coefs': modelo.state_dict_coefs(),
                'optimizer_state': optimizer.state_dict(),
                'k_completado': k + 1,
                'config': config,
            }, ruta_chk)

    t_total = time.time() - t_inicio
    return {
        'losses_por_frame': losses_por_frame,
        'psnr_frame0_por_k': psnr_frame0_por_k,
        'tiempo_por_frame': tiempo_por_frame,
        'tiempo_total': t_total,
    }
