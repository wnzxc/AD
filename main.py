import pandas as pd
import numpy as np
from atmosphere import Atmosphere_GOST_4401_81
from ODE_solvers import RungeKutta4
from math import sin, cos, pi
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import time

# ============================================================
# ЦВЕТОВАЯ ПАЛИТРА (из 1-го кода)
# ============================================================
COLOR_PALETTE = [
    '#4C72B0',
    '#55A868',
    '#C44E52',
    '#8172B2',
    '#CCB974',
    '#64B5CD'
]

# ============================================================
# ИСХОДНЫЕ ДАННЫЕ (Вариант 1 — из 2-го кода)
# ============================================================
V_initial_1 = 268.0  # м/с
V_initial_2 = 948.0  # м/с
time_start = 0.0
coord_x_start = 0.0
coord_y_start = 0.0
angular_velocity_start = 0.0

theta_c_start_1 = np.radians(20.0)
theta_c_start_2 = np.radians(30.0)
theta_c_start_3 = np.radians(40.0)
theta_c_start_4 = np.radians(50.0)

mass_vehicle = 800.0  # кг
inertia_moment = 120.0  # кг·м²
distance_cp_cm = 0.4  # м — плечо аэродин. момента (из 1-го кода)
reference_area = 0.2  # м²
time_step = 0.1  # с

# ============================================================
# ИНИЦИАЛИЗАЦИЯ АТМОСФЕРЫ
# ============================================================
atmosphere_model = Atmosphere_GOST_4401_81()
gravity_accel = atmosphere_model.g

# ============================================================
# ПАРАМЕТРЫ ОПТИМИЗАЦИИ
# ============================================================
initial_theta_guess = np.radians(45.0)
theta_bounds = [(10.0 * pi / 180, 80.0 * pi / 180)]  # границы из 2-го кода
decimal_rounding = 5

# ============================================================
# ТАБЛИЦЫ АЭРОДИНАМИЧЕСКИХ КОЭФФИЦИЕНТОВ
# ============================================================
mach_numbers = [0.01, 0.55, 0.8, 0.9, 1, 1.06, 1.1, 1.2, 1.3, 1.4, 2, 2.6, 3.4, 6, 10]
drag_coeff_data = [0.3, 0.3, 0.55, 0.7, 0.84, 0.86, 0.87, 0.83, 0.8, 0.79, 0.65, 0.55, 0.5, 0.45, 0.4]
lift_coeff_data = [0.25, 0.25, 0.25, 0.2, 0.3, 0.31, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]

drag_interpolator = interp1d(mach_numbers, drag_coeff_data, kind="linear", fill_value="extrapolate")
lift_interpolator = interp1d(mach_numbers, lift_coeff_data, kind="linear", fill_value="extrapolate")


def get_drag_coefficient(mach):
    return float(drag_interpolator(mach))


def get_lift_coefficient(mach):
    return float(lift_interpolator(mach))


# ============================================================
# СИСТЕМА ДИФФУРОВ (6 DOF — логика из 1-го кода)
# ============================================================
def equations_of_motion(time, state_vector):
    derivatives = np.zeros(6)

    velocity = state_vector[0]
    flight_path_angle = state_vector[1]
    coord_x = state_vector[2]
    coord_y = state_vector[3]
    angular_rate = state_vector[4]
    pitch_angle = state_vector[5]

    mach_current = velocity / atmosphere_model.a(coord_y)
    angle_of_attack = pitch_angle - flight_path_angle

    air_density = atmosphere_model.rho(coord_y)
    drag_force = 0.5 * get_drag_coefficient(mach_current) * reference_area * air_density * velocity ** 2
    lift_force = 0.5 * get_lift_coefficient(
        mach_current) * reference_area * air_density * angle_of_attack * velocity ** 2
    pitch_moment = -0.5 * (get_drag_coefficient(mach_current) + get_lift_coefficient(
        mach_current)) * reference_area * distance_cp_cm * velocity ** 2

    derivatives[0] = -drag_force / mass_vehicle - gravity_accel * sin(flight_path_angle)
    derivatives[1] = lift_force / (mass_vehicle * velocity) - gravity_accel * cos(flight_path_angle) / velocity
    derivatives[2] = velocity * cos(flight_path_angle)
    derivatives[3] = velocity * sin(flight_path_angle)
    derivatives[4] = pitch_moment * angle_of_attack / inertia_moment
    derivatives[5] = angular_rate

    return derivatives


def ground_impact_condition(time, state_vector):
    return state_vector[3] + 1e-10


def compute_report_parameters(time, state_vector):
    report_data = np.zeros(16)

    velocity = state_vector[0]
    flight_path_angle = state_vector[1]
    coord_y = state_vector[3]
    pitch_angle = state_vector[5]

    mach_current = velocity / atmosphere_model.a(coord_y)
    angle_of_attack = pitch_angle - flight_path_angle
    air_density = atmosphere_model.rho(coord_y)

    drag_force = 0.5 * get_drag_coefficient(mach_current) * reference_area * air_density * velocity ** 2
    lift_force = 0.5 * get_lift_coefficient(
        mach_current) * reference_area * air_density * angle_of_attack * velocity ** 2
    pitch_moment = -0.5 * (get_drag_coefficient(mach_current) + get_lift_coefficient(
        mach_current)) * reference_area * distance_cp_cm * velocity ** 2

    report_data[0] = angle_of_attack
    report_data[1] = mass_vehicle
    report_data[2] = atmosphere_model.a(coord_y)
    report_data[3] = mach_current
    report_data[4] = get_drag_coefficient(mach_current)
    report_data[5] = drag_force
    report_data[6] = -drag_force / mass_vehicle - gravity_accel * sin(flight_path_angle)
    report_data[7] = get_lift_coefficient(mach_current)
    report_data[8] = lift_force
    report_data[9] = lift_force / (mass_vehicle * velocity) - gravity_accel * cos(flight_path_angle) / velocity
    report_data[10] = velocity * sin(flight_path_angle)
    report_data[11] = velocity * cos(flight_path_angle)
    report_data[12] = pitch_moment
    report_data[13] = pitch_moment * angle_of_attack / inertia_moment
    report_data[14] = air_density
    report_data[15] = atmosphere_model.p(coord_y)

    return report_data


# ============================================================
# КОРРЕКЦИЯ ТОЧКИ КАСАНИЯ (из 1-го кода)
# ============================================================
def correct_impact_point(trajectory):
    y_previous = trajectory[-2, 4]
    y_current = trajectory[-1, 4]

    if y_current > 0:
        return trajectory

    interpolation_fraction = y_previous / (y_previous - y_current)
    corrected_trajectory = trajectory.copy()

    for col_idx in range(corrected_trajectory.shape[1]):
        corrected_trajectory[-1, col_idx] = trajectory[-2, col_idx] + interpolation_fraction * (
                    trajectory[-1, col_idx] - trajectory[-2, col_idx])

    return corrected_trajectory


def solve_trajectory(initial_conditions):
    V0, theta_c0 = initial_conditions
    pitch_angle_initial = theta_c0
    initial_state = [V0, theta_c0, coord_x_start, coord_y_start, angular_velocity_start, pitch_angle_initial]

    trajectory = RungeKutta4(
        equations_of_motion,
        initial_state,
        ground_impact_condition,
        compute_report_parameters,
        time_step,
        x_0=time_start,
        max_steps=100_000
    )
    return correct_impact_point(trajectory)


# ============================================================
# ОПТИМИЗАЦИЯ УГЛА (логика из 1-го кода, границы из 2-го)
# ============================================================
def optimize_launch_angle(initial_velocity):
    def objective_function(angle_vars):
        launch_angle = angle_vars[0]
        pitch_angle_initial = launch_angle
        initial_state = [initial_velocity, launch_angle, coord_x_start, coord_y_start, angular_velocity_start,
                         pitch_angle_initial]
        trajectory = correct_impact_point(
            RungeKutta4(
                equations_of_motion,
                initial_state,
                ground_impact_condition,
                compute_report_parameters,
                time_step,
                x_0=time_start,
                max_steps=100_000
            )
        )
        return -trajectory[-1, 3]  # максимизация дальности x

    result = minimize(objective_function, bounds=theta_bounds, x0=initial_theta_guess, method='SLSQP')
    return result


# ============================================================
# ВЫВОД ТАБЛИЦЫ — КАК В 1-М КОДЕ (pandas + Excel)
# ============================================================
def create_results_dataframe(trajectory_data, filename):
    dataframe = pd.DataFrame({
        't': np.round(trajectory_data[:, 0], decimal_rounding),
        'm': np.round(trajectory_data[:, 8], decimal_rounding),
        'V': np.round(trajectory_data[:, 1], decimal_rounding),
        'a': np.round(trajectory_data[:, 9], decimal_rounding),
        'M': np.round(trajectory_data[:, 10], decimal_rounding),
        'Cxa': np.round(trajectory_data[:, 11], decimal_rounding),
        'Xa': np.round(trajectory_data[:, 12], decimal_rounding),
        'α': np.round(np.degrees(trajectory_data[:, 7]), decimal_rounding),
        'θ_c': np.round(np.degrees(trajectory_data[:, 2]), decimal_rounding),
        'dV/dt': np.round(trajectory_data[:, 13], decimal_rounding),
        'Cya': np.round(trajectory_data[:, 14], decimal_rounding),
        'Ya': np.round(trajectory_data[:, 15], decimal_rounding),
        'dθ_c/dt': np.round(trajectory_data[:, 16], decimal_rounding),
        'ϑ': np.round(np.degrees(trajectory_data[:, 6]), decimal_rounding),
        'y': np.round(trajectory_data[:, 4], decimal_rounding),
        'dy/dt': np.round(trajectory_data[:, 17], decimal_rounding),
        'x': np.round(trajectory_data[:, 3], decimal_rounding),
        'dx/dt': np.round(trajectory_data[:, 18], decimal_rounding),
        'Mza': np.round(trajectory_data[:, 19], decimal_rounding),
        'ω_z': np.round(trajectory_data[:, 5], decimal_rounding),
        'dω_z/dt': np.round(trajectory_data[:, 20], decimal_rounding),
        'ρ': np.round(trajectory_data[:, 21], decimal_rounding),
        'p': np.round(trajectory_data[:, 22], decimal_rounding)
    })

    # Фильтрация с шагом 1 с (как в 1-м коде: t % 1 == 0)
    dataframe_formatted = dataframe[dataframe['t'] % 1 == 0].reset_index(drop=True)
    dataframe_formatted.to_excel(filename, index=False)
    print(f"[OK] Таблица сохранена: {filename}  (строк: {len(dataframe_formatted)})")


# ============================================================
# ГРАФИКИ (стиль из 2-го кода, данные из 1-го)
# ============================================================
def plot_trajectory_data(trajectories_list, x_col, y_col, legend_labels, convert_to_degrees, x_label, y_label,
                         output_filename):
    fig, ax = plt.subplots(figsize=(10, 6))

    for idx, trajectory in enumerate(trajectories_list):
        color_idx = idx % len(COLOR_PALETTE)

        if convert_to_degrees:
            ax.plot(trajectory[:, x_col], np.degrees(trajectory[:, y_col]),
                    label=legend_labels[idx], color=COLOR_PALETTE[color_idx], linewidth=1.5)
        else:
            ax.plot(trajectory[:, x_col], trajectory[:, y_col],
                    label=legend_labels[idx], color=COLOR_PALETTE[color_idx], linewidth=1.5)

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('white')

    plt.tight_layout()
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()


# ============================================================
# ГЛАВНАЯ ПРОГРАММА
# ============================================================
if __name__ == '__main__':
    start_time = time.time()

    print("=" * 80)
    print("ДОМАШНЕЕ ЗАДАНИЕ №1")
    print("Расчет элементов траектории ЛА на пассивном участке")
    print("Вариант 1: V01 = 230 м/с, V02 = 930 м/с  (6 DOF, полная аэродинамика)")
    print("=" * 80)

    # ----------------------------------------------------------
    # ПУНКТ 1: 4 угла для V0 = 230 м/с
    # ----------------------------------------------------------
    print("\n" + "=" * 80)
    print("ПУНКТ 1: Интегрирование для V0 = 230 м/с")
    print("=" * 80)

    with ProcessPoolExecutor(max_workers=4) as executor:
        trajectory_futures = [
            executor.submit(solve_trajectory, (V_initial_1, theta_c_start_1)),
            executor.submit(solve_trajectory, (V_initial_1, theta_c_start_2)),
            executor.submit(solve_trajectory, (V_initial_1, theta_c_start_3)),
            executor.submit(solve_trajectory, (V_initial_1, theta_c_start_4))
        ]
        trajectory_20deg, trajectory_30deg, trajectory_40deg, trajectory_50deg = [f.result() for f in
                                                                                  trajectory_futures]

    # Сохранение таблиц (формат 1-го кода)
    create_results_dataframe(trajectory_20deg, f'results_theta_{round(np.degrees(theta_c_start_1), 3)}_deg.xlsx')
    create_results_dataframe(trajectory_30deg, f'results_theta_{round(np.degrees(theta_c_start_2), 3)}_deg.xlsx')
    create_results_dataframe(trajectory_40deg, f'results_theta_{round(np.degrees(theta_c_start_3), 3)}_deg.xlsx')
    create_results_dataframe(trajectory_50deg, f'results_theta_{round(np.degrees(theta_c_start_4), 3)}_deg.xlsx')

    # Графики Пункта 1
    trajectories_list = [trajectory_20deg, trajectory_30deg, trajectory_40deg, trajectory_50deg]
    legend_labels = [
        f'θ₀ = {round(np.degrees(theta_c_start_1), 3)}°',
        f'θ₀ = {round(np.degrees(theta_c_start_2), 3)}°',
        f'θ₀ = {round(np.degrees(theta_c_start_3), 3)}°',
        f'θ₀ = {round(np.degrees(theta_c_start_4), 3)}°'
    ]

    print("\n>>> Построение графиков для V0 = 230 м/с...")
    plot_trajectory_data(trajectories_list, 3, 4, legend_labels, False, 'x, м', 'y, м',
                         f'trajectories_yx_V{V_initial_1:.0f}.png')
    plot_trajectory_data(trajectories_list, 0, 1, legend_labels, False, 't, с', 'V, м/с',
                         f'velocity_Vt_V{V_initial_1:.0f}.png')
    plot_trajectory_data(trajectories_list, 0, 2, legend_labels, True, 't, с', 'θ_c, °',
                         f'theta_ct_V{V_initial_1:.0f}.png')
    plot_trajectory_data(trajectories_list, 0, 4, legend_labels, False, 't, с', 'y, м',
                         f'height_yt_V{V_initial_1:.0f}.png')
    plot_trajectory_data(trajectories_list, 0, 3, legend_labels, False, 't, с', 'x, м',
                         f'range_xt_V{V_initial_1:.0f}.png')
    plot_trajectory_data(trajectories_list, 3, 1, legend_labels, False, 'x, м', 'V, м/с',
                         f'velocity_Vx_V{V_initial_1:.0f}.png')
    plot_trajectory_data(trajectories_list, 3, 2, legend_labels, True, 'x, м', 'θ_c, °',
                         f'theta_cx_V{V_initial_1:.0f}.png')
    plot_trajectory_data(trajectories_list, 0, 6, legend_labels, True, 't, с', 'ϑ, °',
                         f'theta_t_V{V_initial_1:.0f}.png')
    plot_trajectory_data(trajectories_list, 0, 7, legend_labels, True, 't, с', 'α, °',
                         f'alpha_t_V{V_initial_1:.0f}.png')
    plot_trajectory_data(trajectories_list, 0, 5, legend_labels, False, 't, с', 'ω_z, с⁻¹',
                         f'omega_zt_V{V_initial_1:.0f}.png')

    # ----------------------------------------------------------
    # ПУНКТ 2: Оптимальные углы
    # ----------------------------------------------------------
    print("\n" + "=" * 80)
    print("ПУНКТ 2: Поиск оптимальных углов бросания")
    print("=" * 80)

    optimization_start = time.time()

    with ProcessPoolExecutor(max_workers=2) as executor:
        optimization_futures = [
            executor.submit(optimize_launch_angle, V_initial_1),
            executor.submit(optimize_launch_angle, V_initial_2)
        ]
        opt_result_1, opt_result_2 = [f.result() for f in optimization_futures]

        optimal_angle_1 = opt_result_1.x[0]
        optimal_angle_2 = opt_result_2.x[0]

        print(f"\nОптимальный угол для V0 = {V_initial_1} м/с: {round(np.degrees(optimal_angle_1), decimal_rounding)}°")
        print(f"Максимальная дальность: {-opt_result_1.fun:.2f} м")
        print(f"\nОптимальный угол для V0 = {V_initial_2} м/с: {round(np.degrees(optimal_angle_2), decimal_rounding)}°")
        print(f"Максимальная дальность: {-opt_result_2.fun:.2f} м")

        # Интегрирование для оптимальных углов
        trajectory_futures = [
            executor.submit(solve_trajectory, (V_initial_1, optimal_angle_1)),
            executor.submit(solve_trajectory, (V_initial_2, optimal_angle_2))
        ]
        optimal_traj_1, optimal_traj_2 = [f.result() for f in trajectory_futures]

    # Сохранение таблиц оптимальных траекторий
    create_results_dataframe(optimal_traj_1,
                             f'optimal_V0_{round(V_initial_1, 3)}_theta_{round(np.degrees(optimal_angle_1), 3)}_deg.xlsx')
    create_results_dataframe(optimal_traj_2,
                             f'optimal_V0_{round(V_initial_2, 3)}_theta_{round(np.degrees(optimal_angle_2), 3)}_deg.xlsx')

    # Графики сравнения оптимальных траекторий
    optimal_trajectories = [optimal_traj_1, optimal_traj_2]
    optimal_labels = [
        f'V₀ = {V_initial_1} м/с, θ₀ = {round(np.degrees(optimal_angle_1), 3)}°',
        f'V₀ = {V_initial_2} м/с, θ₀ = {round(np.degrees(optimal_angle_2), 3)}°'
    ]

    print("\n>>> Построение сравнительных графиков оптимальных траекторий...")
    plot_trajectory_data(optimal_trajectories, 0, 1, optimal_labels, False, 't, с', 'V, м/с', 'optimal_Vt.png')
    plot_trajectory_data(optimal_trajectories, 0, 2, optimal_labels, True, 't, с', 'θ_c, °', 'optimal_theta_ct.png')
    plot_trajectory_data(optimal_trajectories, 0, 4, optimal_labels, False, 't, с', 'y, м', 'optimal_yt.png')
    plot_trajectory_data(optimal_trajectories, 0, 3, optimal_labels, False, 't, с', 'x, м', 'optimal_xt.png')
    plot_trajectory_data(optimal_trajectories, 0, 6, optimal_labels, True, 't, с', 'ϑ, °', 'optimal_theta_t.png')
    plot_trajectory_data(optimal_trajectories, 0, 7, optimal_labels, True, 't, с', 'α, °', 'optimal_alpha_t.png')
    plot_trajectory_data(optimal_trajectories, 3, 2, optimal_labels, True, 'x, м', 'θ_c, °', 'optimal_theta_cx.png')
    plot_trajectory_data(optimal_trajectories, 3, 4, optimal_labels, False, 'x, м', 'y, м', 'optimal_yx.png')
    plot_trajectory_data(optimal_trajectories, 0, 5, optimal_labels, False, 't, с', 'ω_z, с⁻¹', 'optimal_omega_zt.png')

    # ----------------------------------------------------------
    # СВОДНАЯ ТАБЛИЦА
    # ----------------------------------------------------------
    print("\n" + "=" * 80)
    print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print("=" * 80)

    theta_list = [
        (np.degrees(theta_c_start_1), trajectory_20deg),
        (np.degrees(theta_c_start_2), trajectory_30deg),
        (np.degrees(theta_c_start_3), trajectory_40deg),
        (np.degrees(theta_c_start_4), trajectory_50deg),
    ]

    print(f"\nV0 = {V_initial_1} м/с:")
    for theta_deg, traj in theta_list:
        print(f"  θ₀ = {theta_deg:6.2f}° -> Дальность = {traj[-1, 3]:10.2f} м, Время = {traj[-1, 0]:6.2f} с")
    print(f"  θ₀_opt = {np.degrees(optimal_angle_1):7.4f}° -> Дальность = {optimal_traj_1[-1, 3]:10.2f} м")

    print(f"\nV0 = {V_initial_2} м/с:")
    print(f"  θ₀_opt = {np.degrees(optimal_angle_2):7.4f}° -> Дальность = {optimal_traj_2[-1, 3]:10.2f} м")

    total_time = time.time() - start_time
    print(f"\nОбщее время выполнения: {total_time:.2f} с")
    print("=" * 80)
    print("РАСЧЕТ ЗАВЕРШЕН")
    print("=" * 80)