"""
Maze simulation with ratinabox (coordinates + speed only).

Usage:
    .venv/bin/python simulate_maze.py --duration 300 --scale 20 --output maze.hdf5
"""
import numpy as np
import h5py
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ratinabox
import os


def build_maze(scale=1.0):
    env = ratinabox.Environment(params={
        'scale': scale,
        'aspect': 1.0,
        'dimensionality': '2D',
        'boundary_conditions': 'solid'
    })

    return env


def simulate_maze(scale=1.0, duration=300.0, dt=0.05, seed=42, output='maze.hdf5'):
    np.random.seed(seed)
    print(f'[MAZE] {duration:.1f}s in {scale}m maze, dt={dt}')

    env = build_maze(scale)
    agent = ratinabox.Agent(env)
    agent.params['speed_mean'] = 0.08
    agent.params['speed_std'] = 0.04
    agent.params['dt'] = dt

    n_steps = int(duration / dt)
    for i in range(n_steps):
        agent.update()
        if i % 5000 == 0:
            print(f'  step {i}/{n_steps}  pos=({agent.pos[0]:.2f},{agent.pos[1]:.2f})')

    pos = np.array(agent.history['pos'])
    vel = np.array(agent.history['vel'])
    speed = np.sqrt((vel**2).sum(axis=1))

    # HDF5
    with h5py.File(output, 'w') as f:
        f.create_dataset('position', data=pos)
        f['position'].attrs['desc'] = 'X,Y [m]'
        f.create_dataset('speed', data=vel)
        f['speed'].attrs['desc'] = 'Speed [m/s]'

        f.attrs['dt'] = dt
        f.attrs['duration'] = duration

        f.attrs['scale'] = scale


    print(f'[MAZE] saved {output} ({os.path.getsize(output)/1e3:.0f} KB)')
    print(f'[MAZE] pos=({pos[-1,0]:.2f},{pos[-1,1]:.2f})  max_speed={speed.max():.3f}')

    # Plot arena + trajectory
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    W = scale
    B = np.array([[0, 0], [W, 0], [W, W], [0, W], [0, 0]])
    ax.plot(*B.T, 'k-', lw=2.5)
    for wall in env.walls:
        ax.plot(*wall.T, 'k-', lw=2.5)
    ax.plot(pos[:, 0], pos[:, 1], 'b-', lw=0.8)
    ax.plot(pos[0, 0], pos[0, 1], 'go', ms=8, label='Start')
    ax.plot(pos[-1, 0], pos[-1, 1], 'rx', ms=10, label='End')
    ax.invert_yaxis()
    ax.set_title(f'Maze Trajectory  ({duration}s)')
    ax.legend()
    ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]')
    plt.tight_layout()
    name = os.path.splitext(output)[0]
    plt.savefig(f'{name}_trajectory.png', dpi=100)
    plt.close()
    print(f'[MAZE] plots -> {name}_trajectory.png')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scale', type=float, default=1.0)
    p.add_argument('--duration', type=float, default=5000.0)
    p.add_argument('--dt', type=float, default=0.05)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output', type=str, default='simulated_pos.hdf5')
    args = p.parse_args()
    simulate_maze(args.scale, args.duration, args.dt, args.seed, args.output)


if __name__ == '__main__':
    main()
