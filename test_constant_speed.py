import numpy as np
import h5py

if __name__ == '__main__':
    output = 'simulated_pos.hdf5'
    T = 5000
    dt = 0.5

    Vconst = np.array([0.0, 4.0])

    x0 = 0.5
    y0 = 0.5

    t = np.arange(0, T, dt) * 0.001
    x = x0 + Vconst[0] * t
    y = y0 + Vconst[1] * t

    pos = np.stack([x, y], axis=1)
    vel = np.zeros_like(pos) + Vconst
    with h5py.File(output, 'w') as f:
        f.create_dataset('position', data=pos)

        f['position'].attrs['desc'] = 'X,Y [m]'
        f.create_dataset('speed', data=vel)
        f['speed'].attrs['desc'] = 'Speed [m/s]'

        f.attrs['dt'] = dt
        f.attrs['duration'] = T

        f.attrs['scale'] = 1.0