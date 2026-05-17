"""
Burak & Fiete (2009) CANN grid cell model driven by maze trajectory.

Reads position from simulate_maze.py output, simulates grid cell
population dynamics on a CANN torus, encodes initial position from
the maze trajectory, decodes estimated position from population activity.
"""
import numpy as np
import h5py, argparse
import matplotlib
matplotlib.use('qt5Agg')
import matplotlib.pyplot as plt


from connectivity import compute_weight_matrix, assign_coordinates_and_directions, compute_feedforward_input

def update_network(s, W, B, tau=10.0, dt=0.5):
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

    # f_in = np.sqrt(f_in)

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


def initial_state_from_position(n, x0, y0, lambda_space=0.48, lambda_net=13.0, amplitude=1.0):
    """
    Генерирует начальную активность сети, соответствующую реальной позиции (x0, y0).

    Параметры:
    ----------
    n : int
        Размер квадратной сетки нейронов (n x n).
    x0, y0 : float
        Координаты старта в метрах.
    lambda_space : float
        Пространственный период решётки grid-клеток в метрах (например, 0.48).
    lambda_net : float
        Период решётки в нейронах (например, 13).
    amplitude : float
        Желаемая максимальная активность (пик решётки).

    Возвращает:
    ----------
    s0 : np.ndarray, shape (n*n,)
        Начальная активность нейронов (неотрицательная).
    """
    # 1. Координаты нейронов в нейронном пространстве (с центром в 0)
    x_neurons = np.arange(n) - (n - 1) / 2.0
    y_neurons = np.arange(n) - (n - 1) / 2.0
    xx, yy = np.meshgrid(x_neurons, y_neurons)
    r = np.stack([xx.ravel(), yy.ravel()], axis=1)   # (N, 2)

    # 2. Волновые векторы для треугольной решётки
    k = 2.0 * np.pi / lambda_net
    k1 = np.array([k, 0.0])
    k2 = np.array([-k/2.0, k * np.sqrt(3)/2.0])
    k3 = np.array([-k/2.0, -k * np.sqrt(3)/2.0])

    # 3. Базовый паттерн (без сдвига)
    cos1 = np.cos(np.dot(r, k1))
    cos2 = np.cos(np.dot(r, k2))
    cos3 = np.cos(np.dot(r, k3))
    pattern = (cos1 + cos2 + cos3) / 3.0
    pattern = pattern - pattern.min()                 # сделать неотрицательным
    if pattern.max() > 0:
        pattern = amplitude * pattern / pattern.max()
    else:
        pattern = np.zeros_like(pattern)

    # 4. Преобразование реальных координат в сдвиг в нейронах
    scale = lambda_net / lambda_space
    shift_x = x0 * scale
    shift_y = y0 * scale

    # 5. Циклический сдвиг (с учётом периодических границ)
    pattern_2d = pattern.reshape(n, n)
    # Округляем до ближайшего целого (допустимо для дискретной сетки)
    dx = int(round(shift_x))
    dy = int(round(shift_y))
    shifted = np.roll(pattern_2d, shift=dx, axis=1)   # сдвиг по x
    shifted = np.roll(shifted, shift=dy, axis=0)      # сдвиг по y

    return shifted.ravel()


def main():
    # ===== parse =====
    p = argparse.ArgumentParser()
    p.add_argument('--maze', type=str, default='true_simulated_pos.hdf5', help='Trajectory file')
    p.add_argument('--N', type=int, default=32, help='Torus lattice size')
    p.add_argument('--torus_scale', type=float, default=1.0, help='Torus physical scale (m)')
    p.add_argument('--output', type=str, default='activity.hdf5')
    args = p.parse_args()
    N = int(args.N)

    dt = 0.5
    tau_m = 10.0
    alpha = 1.0 * 0.10315
    a = 1.0

    Trelax = 50

    # ===== load trajectory =====
    with h5py.File(args.maze, 'r') as f:
        true_pos = np.array(f['position'])       # (T, 2)
        velocity = np.array(f['speed'])          # (T, 2)

    n_steps = true_pos.shape[0]

    # ===== precompute kernel =====

    neuron_positions, neuron_directions = assign_coordinates_and_directions(N)
    W = 4.0 * compute_weight_matrix(neuron_positions, neuron_directions, a=a, periodic=True)

    # with h5py.File('W.h5', 'w') as f:
    #     f.create_dataset('W', data=W)


    # ===== simulate =====
    chunk_size = 5000
    if n_steps < chunk_size:
        chunk_size = n_steps

    saving_file = h5py.File(args.output, 'w')
    saving_file.create_dataset('activity', (n_steps, N**2), dtype='float32', chunks=(chunk_size, N**2))
    saving_file.create_dataset('pos', data=true_pos)

    rates_chunk = np.zeros((chunk_size, N**2), dtype=np.float32)
    idx = 0

    rates0 = initial_state_from_position(N, true_pos[0, 0], true_pos[0, 1])
    nsteps_pred = int(Trelax/dt)

    print('Simulating...')
    for step in range(n_steps + nsteps_pred):
        step_real = step - nsteps_pred
        if step < nsteps_pred:
            vel = np.asarray([0.0, 0.0])
        else:
            vel = velocity[step_real, :]

        B = compute_feedforward_input(neuron_positions, neuron_directions, vel, alpha=alpha, periodic=True)
        rates = update_network(rates0, W, B, tau=tau_m, dt=dt)
        rates0 = rates

        if step_real < 0:
            continue



        rates_chunk[idx] = rates
        idx += 1

        if idx == chunk_size or step_real == n_steps - 1:
            start = step_real - idx + 1
            end = step_real + 1
            saving_file['activity'][start:end] = rates_chunk[:idx]
            idx = 0

        if step % 5000 == 0:
            print(f' step {step_real}/{n_steps}')

    # ===== save =====

    if  len(saving_file['activity']) < 50000:
        fig, axes = plt.subplots()

        axes.imshow( saving_file['activity'][:].T, aspect='auto', cmap='viridis')

        plt.show()

    saving_file.close()


if __name__ == '__main__':
    main()
