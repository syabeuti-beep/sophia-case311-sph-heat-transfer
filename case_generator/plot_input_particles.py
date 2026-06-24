#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

LABEL_TO_NAME = {
    1: 'x', 2: 'y', 3: 'z', 4: 'ux', 5: 'uy', 6: 'uz', 7: 'm', 8: 'p_type',
    9: 'h', 10: 'temp', 11: 'pres', 12: 'rho', 27: 'rad', 28: 'ri', 29: 'dem_idx',
    30: 'wx', 31: 'wy', 32: 'wz'
}

def read_xyz_ptype(path: Path):
    with path.open('r', encoding='utf-8') as f:
        labels = [int(x) for x in f.readline().split()]
        idx = {LABEL_TO_NAME[label]: i for i, label in enumerate(labels) if label in LABEL_TO_NAME}
        required = ['x', 'y', 'z', 'p_type', 'temp']
        missing = [r for r in required if r not in idx]
        if missing:
            raise SystemExit(f'missing labels in input.txt: {missing}')
        vals = []
        ncols = len(labels)
        for line_no, line in enumerate(f, start=2):
            s = line.strip()
            if not s:
                continue
            row = [float(x) for x in s.split()]
            if len(row) != ncols:
                raise SystemExit(f'line {line_no}: expected {ncols} tab-separated values, got {len(row)}')
            vals.append((row[idx['x']], row[idx['y']], row[idx['z']], row[idx['p_type']], row[idx['temp']]))
    arr = np.asarray(vals, dtype=np.float64)
    return arr

def decimate(arr, max_points: int, seed: int = 311):
    if len(arr) <= max_points:
        return arr
    rng = np.random.default_rng(seed)
    take = rng.choice(len(arr), size=max_points, replace=False)
    return arr[np.sort(take)]

def set_equalish(ax, x, y):
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linewidth=0.3, alpha=0.35)

def main():
    ap = argparse.ArgumentParser(description='Plot SOPHIA input.txt particle positions for visual verification')
    ap.add_argument('--case-root', default='.')
    ap.add_argument('--input', default='source_modified/input/input.txt')
    ap.add_argument('--outdir', default='validation/input_plots')
    ap.add_argument('--max-scatter', type=int, default=120000, help='maximum points per particle class for scatter plots')
    args = ap.parse_args()

    case = Path(args.case_root).resolve()
    input_path = (case / args.input).resolve() if not Path(args.input).is_absolute() else Path(args.input)
    outdir = case / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    arr = read_xyz_ptype(input_path)
    dem = arr[arr[:, 3] > 1000]
    sph = arr[arr[:, 3] <= 1000]
    dem_s = decimate(dem, args.max_scatter)
    sph_s = decimate(sph, args.max_scatter)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    # 2D projection panel: this is the main visual check and uses physical coordinates in mm.
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=180)
    fig.suptitle('SOPHIA Case 3-1-1 input particle positions\nDEM solid vs gas SPH carrier', fontsize=14)
    specs = [
        (axes[0, 0], 0, 1, 'x [mm]', 'y [mm]', 'XY / top projection'),
        (axes[0, 1], 0, 2, 'x [mm]', 'z [mm]', 'XZ / side projection'),
        (axes[1, 0], 1, 2, 'y [mm]', 'z [mm]', 'YZ / cross-section projection'),
    ]
    for ax, a, b, xl, yl, title in specs:
        if len(sph_s):
            ax.scatter(sph_s[:, a]*1000, sph_s[:, b]*1000, s=0.25, c='#1f77b4', alpha=0.12, label='gas SPH carrier')
        if len(dem_s):
            ax.scatter(dem_s[:, a]*1000, dem_s[:, b]*1000, s=0.25, c='#d62728', alpha=0.10, label='DEM solid')
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(title); set_equalish(ax, a, b)
        ax.legend(markerscale=8, loc='best')

    ax = axes[1, 1]
    bins = 80
    ax.hist(dem[:, 0]*1000, bins=bins, alpha=0.65, label=f'DEM x count={len(dem):,}', color='#d62728')
    ax.hist(sph[:, 0]*1000, bins=bins, alpha=0.45, label=f'gas SPH x count={len(sph):,}', color='#1f77b4')
    ax.set_xlabel('x [mm]'); ax.set_ylabel('particle count'); ax.set_title('x-direction count distribution')
    ax.grid(True, linewidth=0.3, alpha=0.35); ax.legend()
    fig.tight_layout()
    overview = outdir / 'input_particles_overview.png'
    fig.savefig(overview)
    plt.close(fig)

    # 3D sampled view.
    fig = plt.figure(figsize=(12, 8), dpi=180)
    ax = fig.add_subplot(111, projection='3d')
    dem3 = decimate(dem, min(args.max_scatter, 60000), seed=312)
    sph3 = decimate(sph, min(args.max_scatter, 60000), seed=313)
    if len(sph3):
        ax.scatter(sph3[:, 0]*1000, sph3[:, 1]*1000, sph3[:, 2]*1000, s=0.3, c='#1f77b4', alpha=0.08, label='gas SPH carrier')
    if len(dem3):
        ax.scatter(dem3[:, 0]*1000, dem3[:, 1]*1000, dem3[:, 2]*1000, s=0.3, c='#d62728', alpha=0.10, label='DEM solid')
    ax.set_xlabel('x [mm]'); ax.set_ylabel('y [mm]'); ax.set_zlabel('z [mm]')
    ax.set_title('3D sampled particle-position check')
    ax.view_init(elev=22, azim=-62)
    ax.legend(markerscale=8)
    sampled3d = outdir / 'input_particles_3d_sample.png'
    fig.tight_layout()
    fig.savefig(sampled3d)
    plt.close(fig)

    summary = outdir / 'input_plot_summary.md'
    xyz_min = arr[:, :3].min(axis=0) * 1000
    xyz_max = arr[:, :3].max(axis=0) * 1000
    summary.write_text(
        '# Input particle plot summary\n\n'
        f'- input: `{input_path}`\n'
        f'- total particles: {len(arr):,}\n'
        f'- DEM particles: {len(dem):,}\n'
        f'- gas SPH carrier particles: {len(sph):,}\n'
        f'- x range [mm]: {xyz_min[0]:.6g} to {xyz_max[0]:.6g}\n'
        f'- y range [mm]: {xyz_min[1]:.6g} to {xyz_max[1]:.6g}\n'
        f'- z range [mm]: {xyz_min[2]:.6g} to {xyz_max[2]:.6g}\n'
        f'- overview image: `{overview.relative_to(case)}`\n'
        f'- 3D sample image: `{sampled3d.relative_to(case)}`\n',
        encoding='utf-8'
    )
    print(f'WROTE {overview}')
    print(f'WROTE {sampled3d}')
    print(f'WROTE {summary}')

if __name__ == '__main__':
    main()
