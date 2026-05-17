"""
Plot activity maps from CANN grid cell simulation (Burak & Fiete 2009).
Updated to match current simulate.py HDF5 schema.
"""
import numpy as np
import h5py
import argparse
import os
import matplotlib.pyplot as plt
from scipy.signal import correlate2d
from typing import Tuple, Optional

def autocorr_2d(signal, centered=True, normalize=False):
    """
    Вычисление двумерной автокорреляционной функции через БПФ.

    Параметры
    ----------
    signal : np.ndarray
        Двумерный массив (вещественный сигнал).
    centered : bool, optional (default=True)
        Если True, нулевая задержка помещается в центр выходного массива
        (используется fftshift). Иначе нулевая задержка находится в индексе (0,0).
    normalize : bool, optional (default=False)
        Если True, из сигнала вычитается среднее, а результат делится на
        значение автокорреляции в нулевой задержке.
        Так получается автокорреляция, нормированная на 1 в нуле.

    Возвращает
    -------
    corr : np.ndarray
        Массив автокорреляции формы (2*H - 1, 2*W - 1), где H, W — размеры signal.
    """
    H, W = signal.shape

    # При необходимости вычитаем среднее
    work_signal = signal.astype(np.float64, copy=False)
    if normalize:
        work_signal = work_signal - np.mean(work_signal)

    # Размер для линейной корреляции (без циклических наложений)
    out_h, out_w = 2 * H - 1, 2 * W - 1

    # Дополнение нулями
    padded = np.zeros((out_h, out_w), dtype=work_signal.dtype)
    padded[:H, :W] = work_signal

    # Вычисляем БПФ
    F = np.fft.fft2(padded)

    # Автокорреляция: обратное БПФ от |F|^2
    corr = np.fft.ifft2(F * np.conj(F)).real

    # Центрируем (нулевая задержка в центр)
    if centered:
        corr = np.fft.fftshift(corr)

    # Нормировка (после центрирования, чтобы правильно найти нулевую задержку)
    if normalize:
        # Позиция нулевой задержки: при centered – центр массива, иначе (0,0)
        if centered:
            zero_lag = (corr.shape[0] // 2, corr.shape[1] // 2)
        else:
            zero_lag = (0, 0)
        corr = corr / corr[zero_lag]

    return corr


def load_neurons_and_traj(path, n_neurons):
    with h5py.File(path, 'r') as f:
        traj = f['pos'][:]
    return traj, n_neurons

def gaussian_kernel_2d(size, sigma):
    """Создает нормированное 2D гауссово ядро размера size x size."""
    kernel = np.zeros((size, size))
    center = size // 2
    total = 0.0
    for i in range(size):
        for j in range(size):
            x, y = i - center, j - center
            value = np.exp(-(x**2 + y**2) / (2 * sigma**2))
            kernel[i, j] = value
            total += value
    return kernel / total  # Нормировка: сумма = 1

def convolve_2d(image, kernel):
    """Простая 2D свёртка с нулевым дополнением (zero padding)."""
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    result = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            result[i, j] = np.sum(padded[i:i+kh, j:j+kw] * kernel)
    return result

def build_spatial_maps(
        activity: np.ndarray,
        positions: np.ndarray,
        bins: int = 50,
        range_xy: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Строит карту посещений, карту активности и нормированную карту пространственной
    активности нейрона (например, клетки места).

    Параметры
    ----------
    activity : np.ndarray, shape (T,)
        Сглаженная непрерывная активность нейрона в каждый момент времени.
    positions : np.ndarray, shape (T, 2)
        Координаты животного в те же моменты времени (x, y).
    bins : int or (int, int)
        Число бинов по осям. Если одно число, то одинаковое по x и y.
    range_xy : ((xmin, xmax), (ymin, ymax)) или None
        Диапазон координат для гистограммы. Если None, вычисляется по данным.
    plot : bool
        Если True, выводит нормированную карту на экран.
    title : str
        Заголовок для графика (используется только при plot=True).

    Возвращает
    ----------
    occupancy_map : np.ndarray, shape (bins_y, bins_x)
        Карта посещений (количество временных шагов в каждом бине).
    activity_map : np.ndarray, shape (bins_y, bins_x)
        Суммарная активность нейрона в каждом бине.
    rate_map : np.ndarray, shape (bins_y, bins_x)
        Нормированная карта: activity_map / occupancy_map.
    info : dict
        Поля: 'x_edges', 'y_edges' – границы бинов.
    """
    # Проверка формы входных данных
    if activity.ndim != 1 or positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError(
            "activity должен быть одномерным массивом, positions – двумерным с двумя столбцами."
        )
    if len(activity) != positions.shape[0]:
        raise ValueError("Длины activity и positions должны совпадать.")

    # Определяем диапазон, если не задан
    if range_xy is None:
        xmin, xmax = positions[:, 0].min(), positions[:, 0].max()
        ymin, ymax = positions[:, 1].min(), positions[:, 1].max()
        range_xy = ((xmin, xmax), (ymin, ymax))

    # Карта посещений (вес каждого шага = 1)
    occupancy_map, x_edges, y_edges = np.histogram2d(
        positions[:, 0], positions[:, 1],
        bins=bins,
        range=range_xy
    )

    # Карта активности (вес = активность нейрона)
    activity_map, _, _ = np.histogram2d(
        positions[:, 0], positions[:, 1],
        bins=bins,
        range=range_xy,
        weights=activity
    )

    # Параметры фильтра
    kernel_size = 15
    sigma = 1.0

    # Создание нормированного гауссового ядра
    kernel = gaussian_kernel_2d(kernel_size, sigma)

    activity_map = convolve_2d(activity_map, kernel)
    occupancy_map = convolve_2d(occupancy_map, kernel)

    # Нормировка: делим активность на посещения, избегая деления на 0
    with np.errstate(divide='ignore', invalid='ignore'):
        rate_map = np.divide(activity_map, occupancy_map)
        rate_map[occupancy_map == 0] = np.nan   # не посещённые бины — NaN

    info = {'x_edges': x_edges, 'y_edges': y_edges}
    return rate_map, occupancy_map, activity_map, info


def main(inpath, prefix):
    print('Getting trajectory and neuron count ...')

    f =  h5py.File(inpath, 'r')
    traj = f['pos'][:]
    n_neurons = f['activity'].shape[1]
    act_ds = f['activity']

    range_xy = ((0, 1), (0, 1))

    for i in range(n_neurons):
        print('Plotting map %d/%d ...' % (i+1, n_neurons))
        activity_i = act_ds[:, i]

        rate_map, occupancy_map, activity_map, info = build_spatial_maps(activity_i, traj, bins=50, range_xy=range_xy)

        #autocorr = autocorr_2d(rate_map)
        autocorr = correlate2d(rate_map, rate_map, mode='same')

        fig, axes = plt.subplots(1, 2, figsize=(9, 4))

        axes[0].imshow(rate_map, cmap='rainbow', origin='lower')
        axes[0].set_title('Rate map')

        axes[1].imshow(autocorr, cmap='rainbow', origin='lower')
        axes[1].set_title('Autocorrelogram')

        os.makedirs('results', exist_ok=True)
        fig.savefig('results/' + prefix + '_' + str(i) + '.png')
        plt.close(fig)

    f.close()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='activity.hdf5', help='HDF5 from simulate.py')
    ap.add_argument('--save-prefix', default='grid_maps')
    args = ap.parse_args()
    print('Loading data ...')
    main(args.input, args.save_prefix)
