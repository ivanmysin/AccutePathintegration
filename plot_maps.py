"""
Plot activity maps from CANN grid cell simulation (Burak & Fiete 2009).
Updated to match current simulate.py HDF5 schema.
"""
import numpy as np
import h5py
import argparse
import os

matplotlib = __import__('matplotlib')
matplotlib.use('Agg')
import matplotlib.pyplot as plt


from typing import Tuple, Optional


def load_data(path):
    with h5py.File(path, 'r') as f:
        activity  = f['activity'][:]
        trajectory = f['pos'][:]
    return activity, trajectory



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

    # Нормировка: делим активность на посещения, избегая деления на 0
    with np.errstate(divide='ignore', invalid='ignore'):
        rate_map = np.divide(activity_map, occupancy_map)
        rate_map[occupancy_map == 0] = np.nan   # не посещённые бины — NaN

    info = {'x_edges': x_edges, 'y_edges': y_edges}
    return rate_map, occupancy_map, activity_map, info


def main(inpath, prefix):
    activity, pos = load_data(inpath)

    print(activity.shape)

    range_xy = [[0, 1], [0, 1]]

    for i in range(activity.shape[1]):
        print('Plotting map %d/%d ...' % (i+1, activity.shape[1]))

        rate_map, occupancy_map, activity_map, info = build_spatial_maps(activity[:, i], pos, bins = 50, range_xy=range_xy)

        fig, axes = plt.subplots(figsize=(5, 5))

        axes.imshow(rate_map, cmap='rainbow', origin='lower')

        fig.savefig( 'results/' + prefix + '_' + str(i) + '.png')

        plt.close(fig)

        # if i > 10:
        #     break






if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='activity.hdf5', help='HDF5 from simulate.py')
    ap.add_argument('--save-prefix', default='grid_maps')
    args = ap.parse_args()
    print('Loading data ...')
    main(args.input, args.save_prefix)
