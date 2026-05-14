"""
Base de polinomios de Chebyshev de primer tipo T_k(t), t en [-1, 1].

Por que Chebyshev y no monomios estandar:
-----------------------------------------
Los polinomios monomiales p(t) = sum a_k * t^k son numericamente inestables
para grado alto. El "fenomeno de Runge" hace que el ajuste oscile salvajemente
cerca de los bordes del intervalo. Con grado 40 es practicamente intratable.

Los polinomios de Chebyshev son una BASE ORTOGONAL sobre [-1, 1] (con peso
1/sqrt(1-t^2)). Esto significa que los coeficientes obtenidos por ajuste por
minimos cuadrados son aproximadamente independientes entre si, lo que en la
practica significa:
  - Mucho mejor condicionamiento numerico del problema de optimizacion.
  - Los coeficientes de orden alto tienden a ser pequenos automaticamente.
  - El error de interpolacion se distribuye uniformemente en el intervalo
    (en lugar de explotar en los bordes).

Definicion (primera especie):
    T_0(t) = 1
    T_1(t) = t
    T_k(t) = 2 t T_{k-1}(t) - T_{k-2}(t)        recurrencia de 3 terminos

Propiedades clave:
  - |T_k(t)| <= 1 para todo t en [-1, 1]. Es una base acotada.
  - Sus raices son t_j = cos((2j-1)pi / (2n)) -- "nodos de Chebyshev"
    (no los usamos, pero es la razon por la cual son optimos).

Mapeo de tiempo:
    Los frames van j = 0, 1, ..., N-1. Lo mapeamos a t_cheb en [-1, 1]:
        t_cheb(j) = 2 * j / (N - 1) - 1
    Asi t_cheb(0) = -1 y t_cheb(N-1) = 1.

Implementacion practica:
    Precomputamos la matriz de bases B[j, k] = T_k(t_cheb_j) UNA sola vez al
    inicio del entrenamiento (no se entrena, no necesita grad). Para evaluar
    el polinomio de la gaussiana i en el frame j:
        p_i(t_j) = sum_k coefs_i[k] * B[j, k]    <-- producto matriz-vector
    Vectorizado sobre todas las gaussianas y todas las dimensiones:
        params(t_j) = coefs @ B[j, :grado+1]
"""
import torch



# DECISION: precomputamos UNA matriz de bases por cada grado distinto que
# aparezca en config["grados"]. Como solo hay tipicamente 2 grados distintos
# (40 para mu/opacity, 20 para color/scale/theta/depth), el costo de memoria
# es despreciable (~ N_frames * 41 floats = 3.6KB para 90 frames y grado 40)
# y simplifica la indexacion. La alternativa de "una matriz grande con el
# grado maximo y slicing" funciona pero es mas facil equivocarse de indices.

def construir_matriz_chebyshev(n_frames, grado_max, device='cpu', dtype=torch.float32):
    """
    Devuelve un tensor (n_frames, grado_max + 1) donde:
        B[j, k] = T_k(t_cheb_j),  con t_cheb_j = 2*j/(n_frames-1) - 1

    Calculado por recurrencia, en doble precision para minimizar error
    acumulado, despues cast al dtype pedido.
    """
    # DECISION: construimos en CPU con float64 (mejor precision en la
    # recurrencia) y despues movemos al device pedido en el dtype objetivo.
    # MPS no soporta float64 nativo, asi que NO podemos crear directamente
    # en device si el dtype intermedio es float64.
    if n_frames < 2:
        t = torch.zeros(n_frames, dtype=torch.float64)
    else:
        idx = torch.arange(n_frames, dtype=torch.float64)
        t = 2.0 * idx / (n_frames - 1) - 1.0

    B = torch.empty(n_frames, grado_max + 1, dtype=torch.float64)
    B[:, 0] = 1.0
    if grado_max >= 1:
        B[:, 1] = t
    for k in range(2, grado_max + 1):
        B[:, k] = 2.0 * t * B[:, k - 1] - B[:, k - 2]

    return B.to(dtype=dtype, device=device)



def evaluar_polinomio_cheb(coeficientes, base_matrix, frame_idx):
    """
    Evalua un polinomio de Chebyshev en un frame especifico.

    Args:
        coeficientes : tensor (..., grado+1) con los coeficientes a_k
        base_matrix  : tensor (n_frames, grado_max+1) precomputado, con grado_max >= grado
        frame_idx    : int, el frame en el que evaluar (0 <= frame_idx < n_frames)

    Returns:
        tensor (...) con el polinomio evaluado.

    Si grado < grado_max, usamos solo las primeras grado+1 columnas de la base.
    """
    grado_mas_1 = coeficientes.shape[-1]
    # fila correspondiente al frame, recortada al grado del polinomio: shape (grado+1,)
    fila = base_matrix[frame_idx, :grado_mas_1]
    # producto: (..., grado+1) * (grado+1,) -> suma sobre el ultimo eje -> (...)
    return (coeficientes * fila).sum(dim=-1)



# === pequenos tests de sanidad ============================================
def _tests_basicos():
    """Verifica T_0=1, T_1=t, T_2=2t^2-1, etc. para 5 frames y grado 4."""
    B = construir_matriz_chebyshev(n_frames=5, grado_max=4)
    # t_j para j=0..4 con N=5:  -1, -0.5, 0, 0.5, 1
    t = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])

    # T_0 = 1 para todo j
    assert torch.allclose(B[:, 0], torch.ones(5)), "T_0 != 1"
    # T_1 = t
    assert torch.allclose(B[:, 1], t), "T_1 != t"
    # T_2 = 2 t^2 - 1
    assert torch.allclose(B[:, 2], 2 * t ** 2 - 1, atol=1e-6), "T_2 != 2t^2-1"
    # T_3 = 4 t^3 - 3 t
    assert torch.allclose(B[:, 3], 4 * t ** 3 - 3 * t, atol=1e-6), "T_3 != 4t^3-3t"
    # T_4 = 8 t^4 - 8 t^2 + 1
    assert torch.allclose(B[:, 4], 8 * t ** 4 - 8 * t ** 2 + 1, atol=1e-6), "T_4 != 8t^4-8t^2+1"

    # |T_k(t)| <= 1 en [-1, 1] -- propiedad de cota
    assert (B.abs() <= 1.0 + 1e-6).all(), "Chebyshev sale del rango [-1, 1]"

    # evaluacion con coefs sencillos
    # p(t) = 3 * T_0 + 2 * T_1  =>  p(0) = 3, p(1) = 5, p(-1) = 1
    coefs = torch.tensor([3.0, 2.0])
    assert torch.isclose(evaluar_polinomio_cheb(coefs, B, frame_idx=2), torch.tensor(3.0))
    assert torch.isclose(evaluar_polinomio_cheb(coefs, B, frame_idx=4), torch.tensor(5.0))
    assert torch.isclose(evaluar_polinomio_cheb(coefs, B, frame_idx=0), torch.tensor(1.0))

    print("[chebyshev] tests OK")


if __name__ == "__main__":
    _tests_basicos()
