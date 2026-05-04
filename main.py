"""
Домашнее задание №1
Расчет элементов траектории ЛА на пассивном (баллистическом) участке
Вариант 1: V01 = 230 м/с, V02 = 930 м/с

Использует:
- atmosphere.py - модуль атмосферы ГОСТ 4401-81
- ODE_solvers.py - солверы ОДУ (Euler, RungeKutta4)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
from math import sin, cos, sqrt, radians, degrees

# ============================================================
# ПОДКЛЮЧЕНИЕ МОДУЛЕЙ
# ============================================================
from atmosphere import Atmosphere_GOST_4401_81
from ODE_solvers import RungeKutta4

# ============================================================
# ИСХОДНЫЕ ДАННЫЕ (Вариант 1)
# ============================================================
V01 = 230.0      # м/с - начальная скорость 1
V02 = 930.0      # м/с - начальная скорость 2
g0 = 9.80665     # м/с² - ускорение свободного падения
m0 = 800.0       # кг - начальная масса ЛА
Jz = 120.0       # кг·м² - момент инерции
Sm = 0.2         # м² - характерная площадь
dt = 0.1         # с - шаг интегрирования

# Начальные углы бросания (градусы)
theta0_degrees = [20.0, 30.0, 40.0, 50.0]

# ============================================================
# ТАБЛИЦЫ АЭРОДИНАМИЧЕСКИХ КОЭФФИЦИЕНТОВ
# ============================================================
M_table = np.array([0.01, 0.55, 0.8, 0.9, 1.0, 1.06, 1.1, 1.2, 1.3, 1.4, 2.0, 2.6, 3.4, 6.0, 10.0])
Cxa_table = np.array([0.30, 0.30, 0.55, 0.70, 0.84, 0.86, 0.87, 0.83, 0.80, 0.79, 0.65, 0.55, 0.50, 0.45, 0.40])
Cya_table = np.array([0.25, 0.25, 0.25, 0.20, 0.30, 0.31, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25])

Cxa_interp = interp1d(M_table, Cxa_table, kind='linear', fill_value='extrapolate')
Cya_interp = interp1d(M_table, Cya_table, kind='linear', fill_value='extrapolate')

# ============================================================
# ИНИЦИАЛИЗАЦИЯ АТМОСФЕРЫ
# ============================================================
atm = Atmosphere_GOST_4401_81()

# ============================================================
# ФУНКЦИИ ДЛЯ РАСЧЕТА АЭРОДИНАМИЧЕСКИХ СИЛ
# ============================================================
def get_aeroforces(V, y, alpha=0.0):
    T = atm.T(y)
    a_sound = atm.a(y)
    rho = atm.rho(y)
    M = V / a_sound if a_sound > 0 else 0
    Cxa = float(Cxa_interp(M))
    Cya = float(Cya_interp(M))
    q = rho * V**2 / 2.0
    Xa = Cxa * Sm * q
    Ya = Cya * Sm * q * alpha
    return Xa, Ya, Cxa, Cya, M, a_sound, rho, T

# ============================================================
# СИСТЕМА ДИФФУРОВ И ФУНКЦИИ ДЛЯ ODE_solvers
# ============================================================
def ode_system(t, state):
    V, theta_c, x, y = state
    Xa, Ya, Cxa, Cya, M, a_sound, rho, T = get_aeroforces(V, y, alpha=0.0)

    dV_dt = -Xa / m0 - g0 * sin(theta_c)
    dtheta_c_dt = -g0 * cos(theta_c) / V
    dx_dt = V * cos(theta_c)
    dy_dt = V * sin(theta_c)

    return np.array([dV_dt, dtheta_c_dt, dx_dt, dy_dt])

def stop_conditions(t, state):
    return state[3] + 1e-6

def report(t, state):
    V, theta_c, x, y = state
    Xa, Ya, Cxa, Cya, M, a_sound, rho, T = get_aeroforces(V, y, alpha=0.0)

    dV_dt = -Xa / m0 - g0 * sin(theta_c)
    dtheta_dt = -g0 * cos(theta_c) / V
    dx_dt = V * cos(theta_c)
    dy_dt = V * sin(theta_c)

    alpha_deg = 0.0
    theta_deg = degrees(theta_c)
    omega_z = 0.0
    Mz = 0.0
    p = rho * atm.R * T

    return np.array([
        Xa, Ya, Cxa, Cya, M, a_sound, rho, T,
        dV_dt, dtheta_dt, dx_dt, dy_dt,
        alpha_deg, theta_deg, omega_z, Mz, p
    ])

# ============================================================
# ИНТЕГРИРОВАНИЕ ТРАЕКТОРИИ
# ============================================================
def integrate_trajectory(V0, theta0_deg, dt=0.1):
    theta0 = radians(theta0_deg)
    init_conditions = np.array([V0, theta0, 0.0, 0.0])

    res = RungeKutta4(
        ode_system=ode_system,
        init_conditions=init_conditions,
        stop_conditions=stop_conditions,
        report=report,
        dx=dt,
        x_0=0.0,
        max_steps=100_000
    )

    results = {
        't': res[:, 0],
        'V': res[:, 1],
        'theta_c': res[:, 2],
        'theta_c_deg': np.degrees(res[:, 2]),
        'x': res[:, 3],
        'y': res[:, 4],
        'Xa': res[:, 5],
        'Ya': res[:, 6],
        'Cxa': res[:, 7],
        'Cya': res[:, 8],
        'M': res[:, 9],
        'a': res[:, 10],
        'rho': res[:, 11],
        'T': res[:, 12],
        'dV_dt': res[:, 13],
        'dtheta_dt': res[:, 14],
        'dx_dt': res[:, 15],
        'dy_dt': res[:, 16],
        'alpha_deg': res[:, 17],
        'theta_deg': res[:, 18],
        'omega_z': res[:, 19],
        'Mz': res[:, 20],
        'p': res[:, 21],
    }

    return results

# ============================================================
# ПОИСК ОПТИМАЛЬНОГО УГЛА
# ============================================================
def find_optimal_angle(V0, dt=0.1):
    def negative_range(theta_deg):
        if theta_deg <= 0 or theta_deg >= 90:
            return 0
        traj = integrate_trajectory(V0, theta_deg, dt)
        return -traj['x'][-1]

    result = minimize_scalar(negative_range, bounds=(10, 80), method='bounded')
    return result.x, -result.fun

# ============================================================
# ВЫВОД ТАБЛИЦЫ (шаг 1 секунда)
# ============================================================
def print_table(results, V0, theta0_deg, step_sec=1.0):
    print(f"\n{'='*120}")
    print(f"ТАБЛИЦА 3. Результаты расчета для V0 = {V0} м/с, theta0 = {theta0_deg}°")
    print(f"{'='*120}")

    header = (
        "N | t,с | m,кг | V,м/с | a,м/с | M | Cxa | Xa,Н | alpha,град | "
        "theta_c,град | dV/dt,м/с² | Ya,Н || rho,кг/м³ | p,Па | dy/dt,м/с | "
        "x,м | dx/dt,м/с | theta,град | omega_z,с⁻¹ | domega/dt,с⁻² | Mz,Н·м | y,м"
    )
    print(header)
    print("-" * 160)

    t_array = results['t']
    idx_step = int(step_sec / dt)

    row_num = 1
    for i in range(0, len(t_array), idx_step):
        domega_dt = 0.0

        row = (
            f"{row_num:2d} | {t_array[i]:5.1f} | {m0:6.1f} | {results['V'][i]:7.2f} | "
            f"{results['a'][i]:7.2f} | {results['M'][i]:5.3f} | {results['Cxa'][i]:5.3f} | "
            f"{results['Xa'][i]:9.2f} | {results['alpha_deg'][i]:7.4f} | "
            f"{results['theta_c_deg'][i]:9.4f} | {results['dV_dt'][i]:10.4f} | "
            f"{results['Ya'][i]:8.2f} || {results['rho'][i]:10.6f} | {results['p'][i]:10.2f} | "
            f"{results['dy_dt'][i]:8.2f} | {results['x'][i]:10.2f} | "
            f"{results['dx_dt'][i]:8.2f} | {results['theta_deg'][i]:8.4f} | "
            f"{results['omega_z'][i]:6.4f} | {domega_dt:8.4f} | {results['Mz'][i]:8.2f} | "
            f"{results['y'][i]:8.2f}"
        )
        print(row)
        row_num += 1

    # Последняя точка (касание)
    last_idx = len(t_array) - 1
    if last_idx % idx_step != 0 and last_idx > 0:
        domega_dt = 0.0
        row = (
            f"{row_num:2d} | {t_array[last_idx]:5.1f} | {m0:6.1f} | {results['V'][last_idx]:7.2f} | "
            f"{results['a'][last_idx]:7.2f} | {results['M'][last_idx]:5.3f} | {results['Cxa'][last_idx]:5.3f} | "
            f"{results['Xa'][last_idx]:9.2f} | {results['alpha_deg'][last_idx]:7.4f} | "
            f"{results['theta_c_deg'][last_idx]:9.4f} | {results['dV_dt'][last_idx]:10.4f} | "
            f"{results['Ya'][last_idx]:8.2f} || {results['rho'][last_idx]:10.6f} | {results['p'][last_idx]:10.2f} | "
            f"{results['dy_dt'][last_idx]:8.2f} | {results['x'][last_idx]:10.2f} | "
            f"{results['dx_dt'][last_idx]:8.2f} | {results['theta_deg'][last_idx]:8.4f} | "
            f"{results['omega_z'][last_idx]:6.4f} | {domega_dt:8.4f} | {results['Mz'][last_idx]:8.2f} | "
            f"{results['y'][last_idx]:8.2f}"
        )
        print(row)

    print(f"\nМаксимальная дальность: {results['x'][-1]:.2f} м")
    print(f"Время полета: {results['t'][-1]:.2f} с")
    print(f"Скорость в момент касания: {results['V'][-1]:.2f} м/с")

# ============================================================
# ГРАФИКИ — ПУНКТ 1 (4 угла на одном графике)
# ============================================================
def plot_trajectories(all_results, V0, title_suffix=""):
    # 1. y(x)
    plt.figure(figsize=(10, 6))
    for theta0, results in all_results.items():
        plt.plot(results['x'], results['y'], label=f'θ₀ = {theta0}°', linewidth=1.5)
    plt.xlabel('x, м')
    plt.ylabel('y, м')
    plt.title(f'Траектории ЛА y(x) {title_suffix}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'trajectories_yx_V{V0:.0f}.png', dpi=150)
    plt.show()

    # 2. V(t)
    plt.figure(figsize=(10, 6))
    for theta0, results in all_results.items():
        plt.plot(results['t'], results['V'], label=f'θ₀ = {theta0}°', linewidth=1.5)
    plt.xlabel('t, с')
    plt.ylabel('V, м/с')
    plt.title(f'Скорость V(t) {title_suffix}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'velocity_Vt_V{V0:.0f}.png', dpi=150)
    plt.show()

    # 3. theta_c(t)
    plt.figure(figsize=(10, 6))
    for theta0, results in all_results.items():
        plt.plot(results['t'], results['theta_c_deg'], label=f'θ₀ = {theta0}°', linewidth=1.5)
    plt.xlabel('t, с')
    plt.ylabel('θ_c, °')
    plt.title(f'Угол наклона траектории θ_c(t) {title_suffix}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'theta_ct_V{V0:.0f}.png', dpi=150)
    plt.show()

    # 4. y(t)
    plt.figure(figsize=(10, 6))
    for theta0, results in all_results.items():
        plt.plot(results['t'], results['y'], label=f'θ₀ = {theta0}°', linewidth=1.5)
    plt.xlabel('t, с')
    plt.ylabel('y, м')
    plt.title(f'Высота y(t) {title_suffix}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'height_yt_V{V0:.0f}.png', dpi=150)
    plt.show()

    # 5. x(t)
    plt.figure(figsize=(10, 6))
    for theta0, results in all_results.items():
        plt.plot(results['t'], results['x'], label=f'θ₀ = {theta0}°', linewidth=1.5)
    plt.xlabel('t, с')
    plt.ylabel('x, м')
    plt.title(f'Дальность x(t) {title_suffix}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'range_xt_V{V0:.0f}.png', dpi=150)
    plt.show()

    # 6. V(x)
    plt.figure(figsize=(10, 6))
    for theta0, results in all_results.items():
        plt.plot(results['x'], results['V'], label=f'θ₀ = {theta0}°', linewidth=1.5)
    plt.xlabel('x, м')
    plt.ylabel('V, м/с')
    plt.title(f'Скорость V(x) {title_suffix}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'velocity_Vx_V{V0:.0f}.png', dpi=150)
    plt.show()

    # 7. theta_c(x)
    plt.figure(figsize=(10, 6))
    for theta0, results in all_results.items():
        plt.plot(results['x'], results['theta_c_deg'], label=f'θ₀ = {theta0}°', linewidth=1.5)
    plt.xlabel('x, м')
    plt.ylabel('θ_c, °')
    plt.title(f'Угол наклона θ_c(x) {title_suffix}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'theta_cx_V{V0:.0f}.png', dpi=150)
    plt.show()

    # 8. theta(t) — УГОЛ ТАНГАЖА (при alpha=0 совпадает с theta_c)
    plt.figure(figsize=(10, 6))
    for theta0, results in all_results.items():
        plt.plot(results['t'], results['theta_deg'], label=f'θ₀ = {theta0}°', linewidth=1.5)
    plt.xlabel('t, с')
    plt.ylabel('ϑ, °')
    plt.title(f'Угол тангажа ϑ(t) {title_suffix}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'theta_t_V{V0:.0f}.png', dpi=150)
    plt.show()

    # 9. alpha(t) — УГОЛ АТАКИ (при баллистическом полете = 0)
    plt.figure(figsize=(10, 6))
    for theta0, results in all_results.items():
        plt.plot(results['t'], results['alpha_deg'], label=f'θ₀ = {theta0}°', linewidth=1.5)
    plt.xlabel('t, с')
    plt.ylabel('α, °')
    plt.title(f'Угол атаки α(t) {title_suffix}\n(α = 0 при баллистическом полете без управления)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'alpha_t_V{V0:.0f}.png', dpi=150)
    plt.show()

    # 10. omega_z(t)
    plt.figure(figsize=(10, 6))
    for theta0, results in all_results.items():
        plt.plot(results['t'], results['omega_z'], label=f'θ₀ = {theta0}°', linewidth=1.5)
    plt.xlabel('t, с')
    plt.ylabel('ω_z, с⁻¹')
    plt.title(f'Угловая скорость ω_z(t) {title_suffix}\n(ω_z = 0 при α = 0)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'omega_zt_V{V0:.0f}.png', dpi=150)
    plt.show()

# ============================================================
# ГРАФИКИ — ПУНКТ 2 (оптимальные углы)
# ============================================================
def plot_optimal_comparison(results1, results2, V1, V2, theta1, theta2):
    # 1. V(t)
    plt.figure(figsize=(10, 6))
    plt.plot(results1['t'], results1['V'],
            label=f'V₀ = {V1} м/с, θ₀ = {theta1:.2f}°', linewidth=1.5)
    plt.plot(results2['t'], results2['V'],
            label=f'V₀ = {V2} м/с, θ₀ = {theta2:.2f}°', linewidth=1.5)
    plt.xlabel('t, с')
    plt.ylabel('V, м/с')
    plt.title('Сравнение оптимальных траекторий: V(t)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('optimal_Vt.png', dpi=150)
    plt.show()

    # 2. theta_c(t)
    plt.figure(figsize=(10, 6))
    plt.plot(results1['t'], results1['theta_c_deg'],
            label=f'V₀ = {V1} м/с, θ₀ = {theta1:.2f}°', linewidth=1.5)
    plt.plot(results2['t'], results2['theta_c_deg'],
            label=f'V₀ = {V2} м/с, θ₀ = {theta2:.2f}°', linewidth=1.5)
    plt.xlabel('t, с')
    plt.ylabel('θ_c, °')
    plt.title('Сравнение оптимальных траекторий: θ_c(t)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('optimal_theta_ct.png', dpi=150)
    plt.show()

    # 3. y(t)
    plt.figure(figsize=(10, 6))
    plt.plot(results1['t'], results1['y'],
            label=f'V₀ = {V1} м/с, θ₀ = {theta1:.2f}°', linewidth=1.5)
    plt.plot(results2['t'], results2['y'],
            label=f'V₀ = {V2} м/с, θ₀ = {theta2:.2f}°', linewidth=1.5)
    plt.xlabel('t, с')
    plt.ylabel('y, м')
    plt.title('Сравнение оптимальных траекторий: y(t)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('optimal_yt.png', dpi=150)
    plt.show()

    # 4. x(t)
    plt.figure(figsize=(10, 6))
    plt.plot(results1['t'], results1['x'],
            label=f'V₀ = {V1} м/с, θ₀ = {theta1:.2f}°', linewidth=1.5)
    plt.plot(results2['t'], results2['x'],
            label=f'V₀ = {V2} м/с, θ₀ = {theta2:.2f}°', linewidth=1.5)
    plt.xlabel('t, с')
    plt.ylabel('x, м')
    plt.title('Сравнение оптимальных траекторий: x(t)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('optimal_xt.png', dpi=150)
    plt.show()

    # 5. theta(t) — УГОЛ ТАНГАЖА
    plt.figure(figsize=(10, 6))
    plt.plot(results1['t'], results1['theta_deg'],
            label=f'V₀ = {V1} м/с, θ₀ = {theta1:.2f}°', linewidth=1.5)
    plt.plot(results2['t'], results2['theta_deg'],
            label=f'V₀ = {V2} м/с, θ₀ = {theta2:.2f}°', linewidth=1.5)
    plt.xlabel('t, с')
    plt.ylabel('ϑ, °')
    plt.title('Сравнение оптимальных траекторий: ϑ(t)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('optimal_theta_t.png', dpi=150)
    plt.show()

    # 6. alpha(t) — УГОЛ АТАКИ
    plt.figure(figsize=(10, 6))
    plt.plot(results1['t'], results1['alpha_deg'],
            label=f'V₀ = {V1} м/с, θ₀ = {theta1:.2f}°', linewidth=1.5)
    plt.plot(results2['t'], results2['alpha_deg'],
            label=f'V₀ = {V2} м/с, θ₀ = {theta2:.2f}°', linewidth=1.5)
    plt.xlabel('t, с')
    plt.ylabel('α, °')
    plt.title('Сравнение оптимальных траекторий: α(t)\n(α = 0 при баллистическом полете)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('optimal_alpha_t.png', dpi=150)
    plt.show()

    # 7. theta_c(x)
    plt.figure(figsize=(10, 6))
    plt.plot(results1['x'], results1['theta_c_deg'],
            label=f'V₀ = {V1} м/с, θ₀ = {theta1:.2f}°', linewidth=1.5)
    plt.plot(results2['x'], results2['theta_c_deg'],
            label=f'V₀ = {V2} м/с, θ₀ = {theta2:.2f}°', linewidth=1.5)
    plt.xlabel('x, м')
    plt.ylabel('θ_c, °')
    plt.title('Сравнение оптимальных траекторий: θ_c(x)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('optimal_theta_cx.png', dpi=150)
    plt.show()

    # 8. y(x)
    plt.figure(figsize=(10, 6))
    plt.plot(results1['x'], results1['y'],
            label=f'V₀ = {V1} м/с, θ₀ = {theta1:.2f}°', linewidth=1.5)
    plt.plot(results2['x'], results2['y'],
            label=f'V₀ = {V2} м/с, θ₀ = {theta2:.2f}°', linewidth=1.5)
    plt.xlabel('x, м')
    plt.ylabel('y, м')
    plt.title('Сравнение оптимальных траекторий: y(x)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('optimal_yx.png', dpi=150)
    plt.show()

    # 9. omega_z(t)
    plt.figure(figsize=(10, 6))
    plt.plot(results1['t'], results1['omega_z'],
            label=f'V₀ = {V1} м/с, θ₀ = {theta1:.2f}°', linewidth=1.5)
    plt.plot(results2['t'], results2['omega_z'],
            label=f'V₀ = {V2} м/с, θ₀ = {theta2:.2f}°', linewidth=1.5)
    plt.xlabel('t, с')
    plt.ylabel('ω_z, с⁻¹')
    plt.title('Сравнение оптимальных траекторий: ω_z(t)\n(ω_z = 0 при α = 0)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('optimal_omega_zt.png', dpi=150)
    plt.show()

# ============================================================
# ГЛАВНАЯ ПРОГРАММА
# ============================================================
if __name__ == "__main__":

    print("=" * 80)
    print("ДОМАШНЕЕ ЗАДАНИЕ №1")
    print("Расчет элементов траектории ЛА на пассивном участке")
    print("Вариант 1: V01 = 230 м/с, V02 = 930 м/с")
    print("=" * 80)

    # ============================================================
    # ПУНКТ 1: Интегрирование для V01 = 230 м/с и 4 углов
    # ============================================================
    print("\n" + "=" * 80)
    print("ПУНКТ 1: Интегрирование для V0 = 230 м/с")
    print("=" * 80)

    results_V1 = {}
    for theta0_deg in theta0_degrees:
        print(f"\n>>> Интегрирование для θ₀ = {theta0_deg}°...")
        results = integrate_trajectory(V01, theta0_deg, dt)
        results_V1[theta0_deg] = results
        print_table(results, V01, theta0_deg)

    print("\n>>> Построение графиков для V0 = 230 м/с...")
    plot_trajectories(results_V1, V01, "(V₀ = 230 м/с)")

    # ============================================================
    # ПУНКТ 2: Поиск оптимальных углов
    # ============================================================
    print("\n" + "=" * 80)
    print("ПУНКТ 2: Поиск оптимальных углов бросания")
    print("=" * 80)

    print(f"\n>>> Поиск оптимального угла для V0 = {V01} м/с...")
    theta_opt1, max_range1 = find_optimal_angle(V01, dt)
    print(f"Оптимальный угол для V0 = {V01} м/с: θ₀_opt = {theta_opt1:.4f}°")
    print(f"Максимальная дальность: {max_range1:.2f} м")

    print(f"\n>>> Поиск оптимального угла для V0 = {V02} м/с...")
    theta_opt2, max_range2 = find_optimal_angle(V02, dt)
    print(f"Оптимальный угол для V0 = {V02} м/с: θ₀_opt = {theta_opt2:.4f}°")
    print(f"Максимальная дальность: {max_range2:.2f} м")

    # Интегрирование для оптимальных углов
    print(f"\n>>> Интегрирование для оптимального угла θ₀ = {theta_opt1:.4f}° (V0 = {V01} м/с)...")
    results_opt1 = integrate_trajectory(V01, theta_opt1, dt)
    print_table(results_opt1, V01, theta_opt1)

    print(f"\n>>> Интегрирование для оптимального угла θ₀ = {theta_opt2:.4f}° (V0 = {V02} м/с)...")
    results_opt2 = integrate_trajectory(V02, theta_opt2, dt)
    print_table(results_opt2, V02, theta_opt2)

    print("\n>>> Построение сравнительных графиков оптимальных траекторий...")
    plot_optimal_comparison(results_opt1, results_opt2, V01, V02, theta_opt1, theta_opt2)

    # ============================================================
    # СВОДНАЯ ТАБЛИЦА
    # ============================================================
    print("\n" + "=" * 80)
    print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print("=" * 80)
    print(f"\nV0 = {V01} м/с:")
    for theta0_deg in theta0_degrees:
        r = results_V1[theta0_deg]['x'][-1]
        t_f = results_V1[theta0_deg]['t'][-1]
        print(f"  θ₀ = {theta0_deg:4.1f}° -> Дальность = {r:10.2f} м, Время = {t_f:6.2f} с")
    print(f"  θ₀_opt = {theta_opt1:7.4f}° -> Дальность = {max_range1:10.2f} м")

    print(f"\nV0 = {V02} м/с:")
    print(f"  θ₀_opt = {theta_opt2:7.4f}° -> Дальность = {max_range2:10.2f} м")

    print("\n" + "=" * 80)
    print("РАСЧЕТ ЗАВЕРШЕН")
    print("=" * 80)