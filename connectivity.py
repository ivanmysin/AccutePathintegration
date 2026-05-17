import numpy as np

def assign_coordinates_and_directions(n):
    """
    Назначает каждому нейрону координаты и вектор предпочтительного направления.

    Параметры:
    ----------
    n : int
        Размер квадратной сетки нейронов (число нейронов по каждой стороне).

    Возвращает:
    ----------
    positions : np.ndarray, shape (N, 2)
        Массив координат нейронов (x, y), где x, y - целые от 0 до n-1.
    directions : np.ndarray, shape (N, 2)
        Массив единичных векторов предпочтительных направлений.
        Возможные векторы: (0,1) - север, (1,0) - восток, (0,-1) - юг, (-1,0) - запад.
    """
    # N = n * n
    # positions = np.zeros((N, 2), dtype=int)
    # directions = np.zeros((N, 2), dtype=int)
    #
    # # Словарь для отображения остатков координат (x%2, y%2) в направление
    # # В каждом 2x2 блоке порядок (по строкам, столбцам):
    # # (0,0): N (0,1)
    # # (1,0): E (1,0)
    # # (0,1): W (-1,0)
    # # (1,1): S (0,-1)
    # dir_map = {
    #     (0, 0): (0, 1),   # N
    #     (1, 0): (1, 0),   # E
    #     (0, 1): (-1, 0),  # W
    #     (1, 1): (0, -1)   # S
    # }
    #
    # idx = 0
    # for y in range(n):
    #     for x in range(n):
    #         positions[idx] = (x, y)
    #         # Определяем остатки от деления на 2
    #         dx, dy = dir_map[(x % 2, y % 2)]
    #         directions[idx] = (dx, dy)
    #         idx += 1
    #
    # positions = positions - n / 2


    e = np.asarray( [ [0.0, 90.0],
                      [180.0, 270.0], ] )

    e = np.deg2rad(e)

    direcs = np.tile(e, reps=[n//2, n//2])

    direcs_x = np.cos(direcs)
    direcs_y = np.sin(direcs)


    g = np.arange(-n//2, n//2)
    pos_x, pos_y = np.meshgrid(g, g)

    # print(pos_x)
    # print('=======' * 10)
    # print(pos_y)


    directions = np.stack([direcs_x.ravel(), direcs_y.ravel()], axis=1)


    positions = np.stack([pos_x.ravel(), pos_y.ravel()], axis=1)


    return positions, directions

def compute_weight_matrix(positions, directions, a=1.0, beta=None, gamma=None, l=2.0, periodic=True, normalize=False):
    """
    Вычисляет матрицу рекуррентных весов W на основе заданных координат и направлений нейронов.

    Параметры:
    ----------
    positions : np.ndarray, shape (N, 2)
        Координаты нейронов (x, y), целые числа.
    directions : np.ndarray, shape (N, 2)
        Единичные векторы предпочтительных направлений для каждого нейрона.
    a : float, optional
        Амплитуда возбуждающей компоненты (по умолчанию 1.0).
    beta : float, optional
        Параметр ширины тормозной компоненты. Если None, вычисляется из lambda_net.
    gamma : float, optional
        Параметр ширины возбуждающей компоненты. Если None, gamma = 1.05 * beta.
    l : float, optional
        Длина вектора сдвига в единицах расстояния между нейронами (по умолчанию 2).
    periodic : bool, optional
        Если True, используются периодические граничные условия (тор).
        Иначе – апериодические (с обрывом на границах).

    Возвращает:
    ----------
    W : np.ndarray, shape (N, N)
        Матрица рекуррентных весов.
    """
    N = positions.shape[0]
    # Определяем размер сетки n (предполагаем квадратную сетку)
    n = int(np.sqrt(N))
    if n * n != N:
        raise ValueError("Количество нейронов должно быть точным квадратом целого числа.")

    # Параметры решётки (из статьи)
    lambda_net = 13.0          # период формирующейся решётки в нейронах
    if beta is None:
        beta = 3.0 / (lambda_net ** 2)
    if gamma is None:
        gamma = 1.05 * beta

    # Вспомогательная функция для W0(r^2)
    def w0(r2):
        return a * np.exp(-gamma * r2) - np.exp(-beta * r2)

    # Предвычисление векторов сдвига для каждого постсинаптического нейрона
    shift_vectors = l * directions   # (N, 2)

    # Инициализация матрицы весов
    W = np.zeros((N, N))

    # Заполнение матрицы
    for i in range(N):
        xi, yi = positions[i]
        for j in range(N):
            sx, sy = shift_vectors[j]
            xj, yj = positions[j]

            dx = xi - xj - sx
            dy = yi - yj - sy

            if periodic:
                # Периодические границы: минимальное расстояние на торе
                dx = dx - n * np.round(dx / n)
                dy = dy - n * np.round(dy / n)

            r2 = dx*dx + dy*dy
            W[i, j] = w0(r2)

    if normalize:
        W = W / N

    return W

def compute_feedforward_input(positions, directions, velocity, alpha=0.10315, periodic=True, envelope_params=None):
    """
    Вычисляет вектор входного тока B_i для всех нейронов.

    Формула: B_i = A(x_i) * (1 + alpha * (e_i · v))

    Параметры:
    ----------
    positions : np.ndarray, shape (N, 2)
        Координаты нейронов (x, y), целые числа.
    directions : np.ndarray, shape (N, 2)
        Единичные векторы предпочтительных направлений для каждого нейрона.
    velocity : np.ndarray, shape (2,)
        Вектор скорости крысы (vx, vy) в м/с.
    alpha : float, optional
        Константа чувствительности к скорости (по умолчанию 0.10315).
    periodic : bool, optional
        Если True, используются периодические границы (A(x)=1 для всех).
        Если False, применяется огибающая A(x) из уравнения (5).
    envelope_params : dict, optional
        Параметры огибающей для апериодической сети. Должен содержать ключи:
        'R' : float — радиус сети (половина размера, т.е. n/2),
        'delta_r' : float — ширина области спада,
        'a0' : float — крутизна спада (по умолчанию 4.0).
        Если None и periodic=False, используются значения по умолчанию,
        вычисляемые из размера сети.

    Возвращает:
    ----------
    B : np.ndarray, shape (N,)
        Входной ток для каждого нейрона.
    """
    N = positions.shape[0]
    # Вычисляем скалярное произведение e_i · v для каждого нейрона
    dot_product = np.dot(directions, velocity)  # (N,)

    # Базовый член (1 + alpha * dot)
    B_base = 1.0 + alpha * dot_product

    if periodic:
        # Для периодической сети огибающая равна 1
        A = np.ones(N)
    else:
        # Для апериодической сети вычисляем огибающую A(x) по формуле (5)
        # Определяем размер сети n (предполагаем квадратную)
        n = int(np.sqrt(N))
        if n * n != N:
            raise ValueError("Количество нейронов должно быть точным квадратом целого числа.")
        # Вычисляем расстояние от центра (центр в ( (n-1)/2, (n-1)/2 ))
        center = (n - 1) / 2.0
        # Координаты относительно центра
        xc = positions[:, 0] - center
        yc = positions[:, 1] - center
        r = np.sqrt(xc**2 + yc**2)
        # Параметры огибающей
        if envelope_params is None:
            R = n / 2.0  # радиус сети
            delta_r = R   # по умолчанию, как в большинстве симуляций статьи
            a0 = 4.0
        else:
            R = envelope_params.get('R', n / 2.0)
            delta_r = envelope_params.get('delta_r', R)
            a0 = envelope_params.get('a0', 4.0)

        A = np.ones(N)
        # Индексы, где r > R - delta_r (область спада)
        mask = r > (R - delta_r)
        # Для них вычисляем A по формуле (5): exp( -a0 * ((r - (R - delta_r))/delta_r)^2 )
        if np.any(mask):
            t = (r[mask] - (R - delta_r)) / delta_r
            A[mask] = np.exp(-a0 * t**2)
        # Для r >= R (за пределами сети) — формально обнуляем, но в статье сетка заканчивается на R,
        # поэтому за пределами нейронов нет. Оставляем как есть.
    # Итоговый вход
    B = A * B_base
    return B

# Пример использования (для маленькой сети, чтобы не перегружать память)
if __name__ == "__main__":
    n_small = 8   # 8x8 = 64 нейрона
    positions, directions = assign_coordinates_and_directions(n_small)


    print("Координаты и направления:")
    print(positions)
    print(directions)