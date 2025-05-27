import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import streamlit as st

st.set_page_config(layout="wide")
st.title("Simulación del Consumo de Sustancias y Tolerancia")
st.markdown("Este modelo simula el efecto de diferentes patrones de consumo sobre el cuerpo humano, incluyendo la tolerancia desarrollada con el tiempo.")

# Parámetros configurables desde la interfaz
ke = st.sidebar.slider("Tasa de eliminación (ke)", 0.1, 1.0, 0.5, step=0.05)
alpha = st.sidebar.slider("Aumento de tolerancia (alpha)", 0.0, 1.0, 0.3, step=0.05)
beta = st.sidebar.slider("Reducción de tolerancia (beta)", 0.0, 1.0, 0.1, step=0.05)

# Tiempo
t_max = st.sidebar.slider("Tiempo máximo de simulación", 10, 100, 50)
t = np.linspace(0, t_max, 500)

# Funciones de consumo
def u_singular(t): return 0
def u_constante(t, R0=1.0): return R0
def u_lineal(t, a=0.2): return a * t
def u_periodica(t, D=5, T=5):
    return D * sum(np.isclose(t, n * T, atol=0.1) for n in range(int(t // T) + 1))

# EDO
def modelo(y, t, ke, alpha, beta, u_func):
    C, T = y
    u_t = u_func(t)
    dCdt = -ke * C + u_t
    dTdt = alpha * u_t - beta * T
    return [dCdt, dTdt]

# Condiciones iniciales
y0_singular = [10, 2]
y0_general = [0, 0]

# Selección de modelo
st.sidebar.markdown("### Tipo de consumo")
tipo = st.sidebar.radio("", ["Dosis única", "Consumo continuo", "Consumo lineal", "Consumo periódico"])

# Configuración adicional
if tipo == "Consumo periódico":
    D = st.sidebar.slider("Dosis periódica (D)", 1, 10, 5)
    T_per = st.sidebar.slider("Período (T)", 1, 20, 5)
    u_func = lambda t: u_periodica(t, D, T_per)
    y0 = y0_general
elif tipo == "Consumo continuo":
    R0 = st.sidebar.slider("Tasa constante de consumo (R0)", 0.1, 5.0, 1.0)
    u_func = lambda t: u_constante(t, R0)
    y0 = y0_general
elif tipo == "Consumo lineal":
    a = st.sidebar.slider("Pendiente de consumo (a)", 0.01, 1.0, 0.2)
    u_func = lambda t: u_lineal(t, a)
    y0 = y0_general
else:
    u_func = u_singular
    y0 = y0_singular

# Solución
sol = odeint(modelo, y0, t, args=(ke, alpha, beta, u_func))

# Gráfico
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(t, sol[:, 0], label='C(t): Sustancia')
ax.plot(t, sol[:, 1], label='T(t): Tolerancia')
ax.set_xlabel("Tiempo")
ax.set_ylabel("Cantidad")
ax.set_title(f"Simulación: {tipo}")
ax.legend()
ax.grid(True)

# Mostrar gráfico
st.pyplot(fig)

# Mostrar valores finales
st.markdown(f"**C(t_final):** {sol[-1,0]:.2f} | **T(t_final):** {sol[-1,1]:.2f}")
