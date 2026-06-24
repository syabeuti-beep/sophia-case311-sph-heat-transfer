from __future__ import annotations

def linspace_points(lo: float, hi: float, spacing: float):
    n = int((hi - lo) / spacing + 0.5) + 1
    return [lo + i * spacing for i in range(n)]

def regular_block(domain: dict, spacing: float):
    xs = linspace_points(*domain['x'], spacing)
    ys = linspace_points(*domain['y'], spacing)
    zs = linspace_points(*domain['z'], spacing)
    for x in xs:
        for y in ys:
            for z in zs:
                yield x, y, z
