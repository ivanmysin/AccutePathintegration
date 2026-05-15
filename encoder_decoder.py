"""
Grid cell position encoder/decoder for CANN torus.

Encoder:   physical position -> population activity on torus (Gaussian wave packet)
Decoder:   population activity on torus -> estimated physical position

Burak & Fiete (2009): CANN torus [0,1]x[0,1] has grid spacing "grid_scale".
The agent's position is mapped onto phases on this torus via the grid pattern.
"""
import numpy as np


def _pref_phases(N):
    """Unit-square preferred phases on the torus."""
    ph = np.linspace(0, 1, N, endpoint=False)
    X, Y = np.meshgrid(ph, ph, indexing='ij')
    return np.stack([X, Y], axis=-1)  # (N, N, 2)


def encode(position, N, torus_scale=1.0, grid_scale=0.3, sigma=0.05):
    """
    Encode agent's physical position into CANN torus activity.

    Parameters
    ----------
    position : (2,) float
        Agent position (m).
    N : int
        Torus lattice size.
    torus_scale : float
        Physical dimension of torus (m).

    Returns
    -------
    activity : (N, N) float
        Population activity on the torus.
    """
    ph = _pref_phases(N)
    tx = (position[0] / torus_scale) % 1.0  # agent phase (unit torus)
    ty = (position[1] / torus_scale) % 1.0

    grid_ph = np.arange(0, 1, grid_scale)

    act = np.zeros((N, N))
    for gp in grid_ph:
        dx = np.abs(ph[:, :, 0] - tx - gp)
        dx = np.minimum(dx, 1 - dx)
        dy = np.abs(ph[:, :, 1] - ty - gp)
        dy = np.minimum(dy, 1 - dy)
        r = np.sqrt(dx ** 2 + dy ** 2)
        act += np.exp(-r ** 2 / (2 * sigma ** 2))

    return np.clip(act, 0, 1)


def decode(activity_on_torus, N, torus_scale=1.0, refine=64):
    """
    Decode physical position from CANN torus activity.

    Cross-correlate activity with itself at all shifts; the shift giving
    maximum self-correlation equals the activity location on the torus.

    Parameters
    ----------
    activity_on_torus : (N, N) float
        Population activity on the torus.
    N : int
        Torus lattice size.
    torus_scale : float
        Physical dimension of torus (m).

    Returns
    --
    position : (2,) float
        Estimated position in metres (on the torus).
    """
    # coarse: FFT-based cross-correlation
    corr = np.abs(np.fft.ifft2(np.fft.fft2(activity_on_torus) ** 2))
    shift = np.unravel_index(np.argmax(corr), (N, N))

    # refine around coarse estimate
    best_score = corr[shift[0], shift[1]]
    best_tx = shift[0] / N
    best_ty = shift[1] / N

    for dt in np.linspace(-0.5, 0.5, refine) / N:
        for dy in np.linspace(-0.5, 0.5, refine) / N:
            sx = (shift[0] + round(dt * N)) % N
            sy = (shift[1] + round(dy * N)) % N
            s = corr[sx, sy]
            if s > best_score:
                best_score = s
                best_tx = sx / N
                best_ty = sy / N

    return np.array([best_tx * torus_scale, best_ty * torus_scale])
