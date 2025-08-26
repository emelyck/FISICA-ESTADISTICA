# PUNTO 7
import numpy as np
import matplotlib.pyplot as plt
def camina_1d(N, a=1.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    pasos = rng.choice([-a, a], size=N)   # siguiendo las fórmulas del libro: saltos simétricos
    return float(pasos.sum())
def corre_varias(N, veces, a=1.0, semilla=None):
    rng = np.random.default_rng(semilla)
    # lo hago vectorizado para que no tarde mil años
    pasos = rng.choice([-a, a], size=(veces, N))
    return pasos.sum(axis=1).astype(float)
def histo_y_tcl(N, veces, a=1.0, semilla=123):
    xs = corre_varias(N, veces, a=a, semilla=semilla)
    mu_sim = xs.mean()
    sig_sim = xs.std(ddof=0)

    mu_th = 0.0
    sig_th = a * np.sqrt(N)  # sí, esto sale del TCL: var = a^2 N
    bordes = np.arange(xs.min() - a, xs.max() + 2*a, 2*a)

    plt.figure(figsize=(9,5))
    plt.hist(xs, bins=bordes, density=True, alpha=0.6, edgecolor='k',
             label=f"simulación ({veces} corridas)")
    grid = np.linspace(xs.min()-2*a, xs.max()+2*a, 600)
    gauss = (1/(np.sqrt(2*np.pi)*sig_th))*np.exp(-(grid-mu_th)**2/(2*sig_th**2))
    plt.plot(grid, gauss, 'r-', lw=2, label='gauss TCL (μ=0, σ²=a²N)')

    plt.title(f"Random walk 1D — N={N}, a={a}")
    plt.xlabel("posición final x")
    plt.ylabel("densidad")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()
  
    print(f"<x> sim = {mu_sim:.4f}   σ_sim = {sig_sim:.4f}   σ_teo = {sig_th:.4f}")

if __name__ == "__main__":
    histo_y_tcl(N=200, veces=50000, a=1.0, semilla=42)


# punto8
import numpy as np
import matplotlib.pyplot as plt

def varias_caminatas(N, reps, paso=1.0, semilla=None):
    """Mismo simulador: N pasos ±paso, repetido 'reps' veces."""
    rng = np.random.default_rng(semilla)
    saltos = rng.choice([-paso, paso], size=(reps, N))
    return saltos.sum(axis=1).astype(float)
def varianza_vs_N(lista_N, reps, paso=1.0, tau=1.0, semilla=7):
    """Hace el barrido en N y ajusta <x^2> ≈ s·N. Luego D_hat = s/(2τ)."""
    rng = np.random.default_rng(semilla)
    medias = []
    var_prom = []   
    for N in lista_N:
        seed_i = int(rng.integers(0, 2**31-1))
        xfin = varias_caminatas(N, reps, paso=paso, semilla=seed_i)
        medias.append(float(xfin.mean()))
        var_prom.append(float((xfin**2).mean()))

    N_arr = np.asarray(lista_N, float)
    x2_arr = np.asarray(var_prom, float)
    s_hat = float((N_arr @ x2_arr) / (N_arr @ N_arr))
    D_hat = s_hat / (2.0 * tau)  # siguiendo el apunte: a^2 ~ s_hat
    morado = "#6F4E7C"
    naranja = "#F28E2B"
    verde = "#59A14F"
    plt.figure(figsize=(8.2, 5.2))
    plt.scatter(N_arr, x2_arr, s=36, color=morado, label=r"simulación  $\langle x^2\rangle$")
    gN = np.linspace(0, 1.05*N_arr.max(), 250)
    plt.plot(gN, (paso**2)*gN, color=verde, lw=2.3, label=rf"teórica  $a^2 N$  (a={paso})")
    plt.plot(gN, s_hat*gN, color=naranja, lw=2.3, ls="--", label=rf"ajuste  $sN$  (s={s_hat:.3g})")
    plt.title("Escalamiento de ⟨x²⟩ con N (difusión)", fontsize=13)
    plt.xlabel("N (número de pasos)")
    plt.ylabel("⟨x²⟩")
    plt.grid(alpha=0.25, linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.show()
    print(f"pendiente s ≈ a^2  = {s_hat:.6g}")
    print(f"D estimado = s/(2τ) = {D_hat:.6g}")
    print(f" D = a^2/(2τ)      = {(paso*paso)/(2.0*tau):.6g}")

if __name__ == "__main__":
    Ns = [25, 50, 100, 200, 400, 800]
    varianza_vs_N(Ns, reps=20000, paso=1.0, tau=1.0, semilla=111)

  
