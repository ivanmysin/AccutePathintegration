"""
Burak & Fiete (2009) CANN grid cell model driven by maze trajectory.

Reads position from simulate_maze.py output, simulates grid cell
population dynamics on a CANN torus, encodes initial position from
the maze trajectory, decodes estimated position from population activity.
"""
import numpy as np
import h5py, argparse, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


from connectivity import compute_weight_matrix, assign_coordinates_and_directions, compute_feedforward_input

def update_network(s, W, B, tau=10, dt=0.5):
    """
    Один шаг динамики сети по уравнению (1).

    Параметры:
    ----------
    s : np.ndarray, shape (N,)
        Текущая активность (скорость firing) нейронов.
    W : np.ndarray, shape (N, N)
        Матрица рекуррентных весов.
    B : np.ndarray, shape (N,)
        Внешний вход (feedforward) в данный момент.
    tau : float
        Постоянная времени нейрона (10 мс = 0.01 с).
    dt : float
        Шаг интегрирования (0.5 мс = 0.0005 с).

    Возвращает:
    ----------
    s_new : np.ndarray
        Обновлённая активность.
    """
    total_input = np.dot(W, s) + B
    # Нелинейность f(x) = max(0, x)
    f_in = np.maximum(total_input, 0.0)
    s_new = s + (dt / tau) * (-s + f_in)
    return s_new

def create_periodic_pattern(n, lambda_net=13.0, amplitude=1.0):
    """
    Создаёт идеальную треугольную решётку активности для периодической сети.

    Параметры:
    ----------
    n : int
        Размер квадратной сетки нейронов (число нейронов по стороне).
    lambda_net : float
        Период решётки в нейронах (по умолчанию 13, как в статье).
    amplitude : float
        Желаемая максимальная активность (пик решётки), по умолчанию 1.

    Возвращает:
    ----------
    s : np.ndarray, shape (n*n,)
        Вектор активности нейронов (неотрицательный).
    """
    # Координаты нейронов с центром в середине сетки
    x = np.arange(n) - (n - 1) / 2.0
    y = np.arange(n) - (n - 1) / 2.0
    xx, yy = np.meshgrid(x, y)
    r = np.stack([xx.ravel(), yy.ravel()], axis=1)  # (N, 2)

    # Волновой вектор для основного направления (горизонталь)
    k = 2.0 * np.pi / lambda_net
    k1 = np.array([k, 0.0])
    k2 = np.array([-k/2.0, k * np.sqrt(3)/2.0])
    k3 = np.array([-k/2.0, -k * np.sqrt(3)/2.0])

    # Косинусные компоненты
    cos1 = np.cos(np.dot(r, k1))
    cos2 = np.cos(np.dot(r, k2))
    cos3 = np.cos(np.dot(r, k3))

    # Треугольная решётка (сумма трёх косинусов)
    pattern = (cos1 + cos2 + cos3) / 3.0

    # Сдвиг, чтобы минимум был 0 (активность не может быть отрицательной)
    pattern = pattern - pattern.min()
    # Масштабирование к заданной амплитуде
    pattern = amplitude * pattern / pattern.max()

    return pattern

def main():
    # ===== parse =====
    p = argparse.ArgumentParser()
    p.add_argument('--maze', type=str, default='simulated_pos.hdf5', help='Trajectory file')
    p.add_argument('--N', type=int, default=32, help='Torus lattice size')
    p.add_argument('--torus_scale', type=float, default=1.0, help='Torus physical scale (m)')
    p.add_argument('--output', type=str, default='activity.hdf5')
    args = p.parse_args()
    N = int(args.N)

    dt = 0.05
    tau_m = 10.0

    # ===== load trajectory =====
    with h5py.File(args.maze, 'r') as f:
        true_pos = np.array(f['position'])       # (T, 2)
        velocity = np.array(f['speed'])   * 0.01    # (T, 2)

    n_steps = len(true_pos)

    # ===== precompute kernel =====

    neuron_positions, neuron_directions = assign_coordinates_and_directions(N)
    W = compute_weight_matrix(neuron_positions, neuron_directions, periodic=True)

    rates = np.zeros((n_steps, N**2), dtype=float)
    # ===== simulate =====
    for step in range(n_steps):
        if step == 0:
            rates0 = create_periodic_pattern(N)
            for i in range(10):
                B =  compute_feedforward_input(neuron_positions, neuron_directions, np.asarray([0, 0]), periodic=True)
                rates0 = update_network(rates0, W, B, tau=tau_m, dt=dt)

        else:
            rates0 = rates[step - 1]

        B =  compute_feedforward_input(neuron_positions, neuron_directions, velocity[step, :], periodic=True)
        rates[step] = update_network(rates0, W, B, tau=tau_m, dt=dt)


    # ===== save =====
    with h5py.File(args.output, 'w') as f:
        f.create_dataset('activity', data=rates)
        f.create_dataset('pos', data=true_pos)


if __name__ == '__main__':
    main()
