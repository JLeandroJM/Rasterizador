import torch



class AcumuladorGradiente:

    def __init__(self, n, device):
        self.suma = torch.zeros(n, device=device)
        self.contador = torch.zeros(n, device=device)
        self.device = device

    def acumular(self, mu_grad):
        if mu_grad is None:
            return
        self.suma = self.suma + mu_grad.norm(dim=1).detach()
        self.contador = self.contador + 1.0

    def promedio(self):
        return self.suma / self.contador.clamp(min=1.0)

    def reset_total(self, n):
        self.suma = torch.zeros(n, device=self.device)
        self.contador = torch.zeros(n, device=self.device)




@torch.no_grad()
def aplicar_mascara(modelo, mascara):
    return (
        modelo.centro[mascara],
        modelo.escala[mascara],
        modelo.theta[mascara],
        modelo.opacidad[mascara],
        modelo.color[mascara],
        modelo.profundidad[mascara],
    )



@torch.no_grad()
def concatenar(base, nuevos):
    c_b, e_b, t_b, o_b, k_b, p_b = base
    c_n, e_n, t_n, o_n, k_n, p_n = nuevos
    return (
        torch.cat([c_b, c_n], dim=0),
        torch.cat([e_b, e_n], dim=0),
        torch.cat([t_b, t_n], dim=0),
        torch.cat([o_b, o_n], dim=0),
        torch.cat([k_b, k_n], dim=0),
        torch.cat([p_b, p_n], dim=0),
    )



# NUEVO: si las dos copias quedan en la MISMA posicion, el render no cambia
# al moverlas y el gradiente cae a ~0 (jamas se separan). Le agregamos un
# pequeno offset aleatorio en pixeles para romper la simetria.
@torch.no_grad()
def clonar(modelo, indices, offset_px=0.5):
    centro_copia = modelo.centro[indices].clone()
    perturbacion = (torch.rand_like(centro_copia) - 0.5) * 2.0 * offset_px
    centro_copia = centro_copia + perturbacion
    return (
        centro_copia,
        modelo.escala[indices].clone(),
        modelo.theta[indices].clone(),
        modelo.opacidad[indices].clone(),
        modelo.color[indices].clone(),
        modelo.profundidad[indices].clone(),
    )



# NUEVO: solo clone (sin split). Como exp2 es "solo crecimiento" y no
# usabamos split en ningun caso, simplifique la funcion para que solo
# duplique las gaussianas con gradiente alto.
@torch.no_grad()
def densificar(modelo, grad_promedio, grad_threshold, max_gaussians):

    n = modelo.numero_gausianas()

    # quien tiene gradiente promedio por encima del umbral
    mascara_grad = grad_promedio > grad_threshold

    if not mascara_grad.any():
        return None, "sin candidatos por gradiente"

    indices_clone = torch.nonzero(mascara_grad, as_tuple=False).squeeze(1)

    # respetar max_gaussians: si nos pasamos, recortamos
    espacio = max_gaussians - n
    if espacio <= 0:
        return None, "max_gaussians alcanzado"
    if len(indices_clone) > espacio:
        indices_clone = indices_clone[:espacio]

    # base = todas las gaussianas existentes (no se elimina ninguna en clone)
    mascara_mantener = torch.ones(n, dtype=torch.bool, device=modelo.centro.device)
    base = aplicar_mascara(modelo, mascara_mantener)

    nuevos_clone = clonar(modelo, indices_clone)
    resultado = concatenar(base, nuevos_clone)

    msg = f"clone={len(indices_clone)}"
    return resultado, msg
